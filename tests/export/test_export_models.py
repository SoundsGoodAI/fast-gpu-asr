#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Integration tests for exported Zipformer and Parakeet encoder modules."""

from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from fast_gpu_asr.constants import (
    ONNX_OPSET_VERSION,
    PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
    PARAKEET_FEATURE_PLUGIN_NAME,
    PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
    ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_FEATURE_PLUGIN_NAME,
    ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
    ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.parakeet.attention import RelPositionMultiHeadAttention
from fast_gpu_asr.export.model.parakeet.parakeet import (
    ConformerConvolution,
    ConvSubsampling,
    FastConformer,
    ParakeetTDTEncoder,
)
from fast_gpu_asr.export.model.zipformer.activation import SwooshL, SwooshR
from fast_gpu_asr.export.model.zipformer.subsampling import BiasNorm
from fast_gpu_asr.export.model.zipformer.zipformer import (
    BypassModule,
    ConvolutionModule,
    FeedforwardModule,
    SimpleDownsample,
    Zipformer2,
)

FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FLOAT_DTYPE_IDS = ("fp32", "fp16", "bf16")
DTYPE_TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 3e-2),
}
ONNX_DTYPES = {
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}
ZIPFORMER_ENCODER_DIMS = (8, 12, 16, 20, 16, 12)


@pytest.fixture(autouse=True)
def isolate_torch_rng() -> Iterator[None]:
    """Seed CPU model initialization and restore the caller's RNG afterward.

    Yields
    ------
    None
        Test scope with an isolated, deterministic CPU generator.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        yield


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a deterministic CPU tensor without changing global RNG state.

    Parameters
    ----------
    shape : tuple[int, ...]
        Tensor dimensions.
    seed : int
        Seed for the private CPU generator.
    dtype : torch.dtype
        Floating-point type used to generate the samples.

    Returns
    -------
    torch.Tensor
        Standard-normal samples with the requested shape and dtype.
    """

    return torch.randn(
        shape,
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    )


def assert_balanced_partition_sizes(
    partition_sizes: list[int], batch_size: int, partition_count: int
) -> None:
    """Require nonempty, balanced partitions that cover the complete batch.

    Parameters
    ----------
    partition_sizes : list[int]
        Batch sizes observed by the convolution hooks.
    batch_size : int
        Expected total number of utterances.
    partition_count : int
        Expected number of convolution calls.
    """

    assert len(partition_sizes) == partition_count
    assert sum(partition_sizes) == batch_size
    assert min(partition_sizes) > 0
    assert max(partition_sizes) - min(partition_sizes) <= 1


def get_onnx_element_type(model: onnx.ModelProto, name: str) -> int:
    """Return a graph value or initializer element type by name.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported graph to inspect.
    name : str
        Tensor name in the graph.

    Returns
    -------
    int
        ONNX tensor element-type enum value.

    Raises
    ------
    AssertionError
        If the tensor has no recorded type information.
    """

    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name:
            return value.type.tensor_type.elem_type
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer.data_type
    raise AssertionError(f"ONNX value {name} has no type information.")


def get_onnx_shape(model: onnx.ModelProto, name: str) -> tuple[int | str, ...]:
    """Return fixed ONNX dimensions as integers and symbolic ones as strings.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported graph to inspect.
    name : str
        Tensor name in the graph's value information.

    Returns
    -------
    tuple[int | str, ...]
        Dimensions in their original axis order.

    Raises
    ------
    AssertionError
        If the tensor has no recorded shape information.
    """

    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name:
            return tuple(
                dimension.dim_param or dimension.dim_value
                for dimension in value.type.tensor_type.shape.dim
            )
    raise AssertionError(f"ONNX value {name} has no shape information.")


def get_onnx_producer(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    """Return the unique ONNX node that produces ``name``.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported graph to inspect.
    name : str
        Output tensor whose producer is required.

    Returns
    -------
    onnx.NodeProto
        Sole producer of the named value.

    Raises
    ------
    AssertionError
        If zero or multiple nodes produce the value.
    """

    producers = [node for node in model.graph.node if name in node.output]
    assert len(producers) == 1, (
        f"Expected one producer for {name!r}, got {len(producers)}."
    )
    return producers[0]


def check_onnx_model_with_custom_plugins(
    model: onnx.ModelProto, plugin_counts: dict[str, int]
) -> None:
    """Check live plugin nodes, then validate a copy with custom ONNX domains.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported graph, left unchanged by this check.
    plugin_counts : dict[str, int]
        Expected node count for each TensorRT plugin operator.

    Notes
    -----
    Assigning custom domains only in the copy lets the ONNX checker validate
    graph wiring without requiring schemas for TensorRT-specific operators.
    """

    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(model)
    custom_nodes = [
        node for node in checker_model.graph.node if node.op_type in plugin_counts
    ]
    assert Counter(node.op_type for node in custom_nodes) == plugin_counts
    live_values = {value.name for value in checker_model.graph.output}
    for node in reversed(checker_model.graph.node):
        if live_values.intersection(node.output):
            live_values.update(node.input)
    for node in custom_nodes:
        assert node.domain == ""
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes["plugin_namespace"] == TENSORRT_PLUGIN_NAMESPACE.encode()
        assert set(node.output) <= live_values, f"{node.op_type} has unused outputs."
        node.domain = TENSORRT_PLUGIN_NAMESPACE
    checker_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(checker_model)


def assert_encoder_onnx_interface(
    model: onnx.ModelProto, output_dim: int, output_dtype: int
) -> None:
    """Check the fixed two-item batch and dynamic time at the encoder boundary.

    Parameters
    ----------
    model : onnx.ModelProto
        Encoder graph with audio inputs and encoded-feature outputs.
    output_dim : int
        Expected encoder output channel count.
    output_dtype : int
        Expected ONNX element-type enum for encoder features.
    """

    assert [value.name for value in model.graph.input] == ["audio", "audio_lengths"]
    assert [value.name for value in model.graph.output] == [
        "encoder_output",
        "encoder_output_lengths",
    ]
    batch, samples = get_onnx_shape(model, "audio")
    output_batch, frames, channels = get_onnx_shape(model, "encoder_output")
    assert batch == output_batch == 2
    assert channels == output_dim
    assert isinstance(samples, str) and isinstance(frames, str)
    for name in ("audio_lengths", "encoder_output_lengths"):
        assert get_onnx_shape(model, name) == (2,)
    for name, dtype in (
        ("audio", onnx.TensorProto.FLOAT),
        ("audio_lengths", onnx.TensorProto.INT64),
        ("encoder_output", output_dtype),
        ("encoder_output_lengths", onnx.TensorProto.INT32),
    ):
        assert get_onnx_element_type(model, name) == dtype, name


def add_zipformer_right_context(
    audio: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Append 200 reflected samples to test waveforms longer than that context.

    Parameters
    ----------
    audio : torch.Tensor
        CPU waveforms of shape ``(batch, samples)``.
    lengths : torch.Tensor
        Valid sample counts of shape ``(batch,)``, each at least 200.

    Returns
    -------
    torch.Tensor
        Same-dtype audio of shape ``(batch, samples + 200)`` with reflected
        utterance tails and zero batch padding.
    """

    padded = audio.new_zeros(audio.size(0), audio.size(1) + 200)
    for index, length in enumerate(lengths.tolist()):
        padded[index, :length] = audio[index, :length]
        context = audio[index, length - 200 : length].flip(0)
        padded[index, length : length + 200] = context
    return padded


def make_fast_conformer(subsampling_batch_partitions: int = 1) -> FastConformer:
    """Build a compact Fast Conformer using the test's isolated CPU generator.

    Parameters
    ----------
    subsampling_batch_partitions : int
        Number of subsampling batch partitions.

    Returns
    -------
    FastConformer
        Two-layer, 16-channel CPU encoder in evaluation mode.
    """

    encoder = FastConformer(
        input_dim=16,
        n_layers=2,
        model_dim=16,
        subsampling_conv_channels=4,
        feed_forward_expansion_factor=2,
        n_heads=4,
        pos_emb_max_len=64,
        conv_kernel_size=3,
        subsampling_batch_partitions=subsampling_batch_partitions,
    )
    return encoder.eval()


def reference_parakeet_subsampling(
    module: ConvSubsampling,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use functional convolutions and recompute valid lengths at each stage.

    Parameters
    ----------
    module : ConvSubsampling
        Source of convolution and projection parameters.
    features : torch.Tensor
        CPU features of shape ``(batch, frames, mel_bins)``.
    feature_lengths : torch.Tensor
        Valid input frame counts of shape ``(batch,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Projected time-major features and their subsampled valid lengths.
    """

    output = features.unsqueeze(1)
    output_lengths = feature_lengths
    for convolution, pointwise in (
        (module.conv1, None),
        (module.conv2, module.pointwise_conv1),
        (module.conv3, module.pointwise_conv2),
    ):
        output = torch.nn.functional.conv2d(
            output,
            convolution.weight,
            convolution.bias,
            stride=convolution.stride,
            padding=convolution.padding,
            groups=convolution.groups,
        )
        if pointwise is not None:
            output = torch.nn.functional.conv2d(
                output, pointwise.weight, pointwise.bias
            )
        output_lengths = (output_lengths + 1) // 2
        padding_mask = torch.arange(output.size(2))[None] >= output_lengths[:, None]
        output = torch.relu(output.masked_fill(padding_mask[:, None, :, None], 0.0))

    batch_size, channels, num_frames, frequency_dim = output.shape
    output = output.permute(0, 2, 1, 3).reshape(
        batch_size, num_frames, channels * frequency_dim
    )
    return (
        torch.nn.functional.linear(output, module.out.weight, module.out.bias),
        output_lengths,
    )


def make_parakeet_encoder(dtype: torch.dtype = torch.float16) -> ParakeetTDTEncoder:
    """Build a compact Parakeet encoder with production frontend settings.

    Parameters
    ----------
    dtype : torch.dtype
        Precision passed to the encoder constructor.

    Returns
    -------
    ParakeetTDTEncoder
        One-layer CPU model in evaluation mode, initialized using the test's
        isolated generator.
    """

    encoder = ParakeetTDTEncoder(
        samp_freq=16000,
        frame_shift_ms=10,
        frame_length_ms=25,
        feature_dim=16,
        preemph=0.97,
        low_freq=0,
        high_freq=8000,
        n_layers=1,
        model_dim=16,
        subsampling_conv_channels=4,
        feed_forward_expansion_factor=2,
        n_heads=4,
        pos_emb_max_len=64,
        conv_kernel_size=3,
        subsampling_batch_partitions=1,
        dtype=dtype,
    )
    return encoder.eval()


def make_zipformer(
    subsampling_batch_partitions: int = 1,
    dtype: torch.dtype = torch.float16,
    use_ctc: bool = False,
) -> Zipformer2:
    """Build a compact six-stack Zipformer using the test's isolated generator.

    Parameters
    ----------
    subsampling_batch_partitions : int
        Number of subsampling batch partitions.
    dtype : torch.dtype
        Precision passed to the encoder constructor.
    use_ctc : bool
        Whether the final projection produces CTC log probabilities.

    Returns
    -------
    Zipformer2
        CPU model in evaluation mode with initialized normalization and
        downsampling buffers, without loading a checkpoint.
    """

    encoder = Zipformer2(
        samp_freq=16000,
        frame_shift_ms=10,
        frame_length_ms=25,
        feature_dim=16,
        preemph=0.97,
        low_freq=20,
        high_freq=7600,
        min_frames=9,
        subsample_output_dim=8,
        subsample_layer1_channels=2,
        subsample_layer2_channels=4,
        subsample_layer3_channels=8,
        subsampling_batch_partitions=subsampling_batch_partitions,
        encoder_dims=list(ZIPFORMER_ENCODER_DIMS),
        num_encoder_layers=[1] * 6,
        downsampling_factors=[1, 2, 4, 8, 4, 2],
        bypass_scales=[torch.ones(dimension) for dimension in ZIPFORMER_ENCODER_DIMS],
        num_heads=[1] * 6,
        feedforward_dims=[16, 20, 24, 28, 24, 20],
        cnn_module_kernels=[3] * 6,
        query_head_dim=4,
        pos_head_dim=4,
        value_head_dim=4,
        pos_dim=4,
        pos_max_len=64,
        output_dim=10,
        use_ctc=use_ctc,
        dtype=dtype,
    ).eval()
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, BiasNorm):
                module.scale.fill_(1.0)
            elif isinstance(module, SimpleDownsample):
                module.weights.fill_(1.0 / module.weights.size(0))
    return encoder


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_precision_policy(dtype: torch.dtype) -> None:
    encoder = make_zipformer(dtype=dtype)
    first_convolution_dtype = torch.float16 if dtype == torch.bfloat16 else dtype

    floating_tensors = [
        (name, tensor)
        for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers())
        if tensor.is_floating_point()
    ]
    assert floating_tensors
    for name, tensor in floating_tensors:
        if name.startswith(("feature_extractor.", "projection_output.")):
            expected_dtype = torch.float32
        elif name.startswith("subsampling.conv1."):
            expected_dtype = first_convolution_dtype
        else:
            expected_dtype = dtype
        assert tensor.dtype == expected_dtype, name


@pytest.mark.parametrize("use_ctc", (False, True), ids=("transducer", "ctc"))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_output_projection(dtype: torch.dtype, use_ctc: bool) -> None:
    encoder = make_zipformer(dtype=dtype, use_ctc=use_ctc)
    logits = torch.arange(10, dtype=torch.float32)
    with torch.no_grad():
        encoder.projection_output.weight.zero_()
        encoder.projection_output.bias.copy_(logits)
    lengths = torch.tensor([3200, 5171] if use_ctc else [3200], dtype=torch.int64)
    audio = add_zipformer_right_context(
        make_random_tensor((len(lengths), int(lengths.max())), 1), lengths
    )
    projected_outputs: list[torch.Tensor] = []
    with (
        encoder.projection_output.register_forward_hook(
            lambda _module, _inputs, output: projected_outputs.append(output.clone())
        ),
        torch.inference_mode(),
    ):
        output, output_lengths = encoder(audio, lengths)

    assert output.shape == ((2, 6, 10) if use_ctc else (1, 3, 10))
    assert output_lengths.dtype == torch.int32
    assert output_lengths.tolist() == ([3, 6] if use_ctc else [3])
    (projection,) = projected_outputs
    torch.testing.assert_close(projection, logits.expand_as(output), atol=0, rtol=0)
    expected = logits.log_softmax(dim=0) if use_ctc else logits
    torch.testing.assert_close(output, expected.expand_as(output), atol=0, rtol=0)


def test_zipformer_eager_output_assembly_matches_surviving_bands() -> None:
    encoder = make_zipformer()
    lengths = torch.tensor([3200], dtype=torch.int64)
    audio = add_zipformer_right_context(make_random_tensor((1, 3200), 4), lengths)
    stack_outputs: list[torch.Tensor] = []
    assembly_inputs: list[torch.Tensor] = []
    with ExitStack() as hooks, torch.inference_mode():
        for module in (encoder.encoder_4, encoder.encoder_5, encoder.encoder_6):
            hooks.enter_context(
                module.register_forward_hook(
                    lambda _module, _inputs, output: stack_outputs.append(output)
                )
            )
        hooks.enter_context(
            encoder.downsample_output.register_forward_pre_hook(
                lambda _module, inputs: assembly_inputs.append(inputs[0])
            )
        )
        encoder(audio, lengths)

    stack4, stack5, stack6 = stack_outputs
    (assembly,) = assembly_inputs
    expected = torch.cat((stack6, stack5[:, :, 12:], stack4[:, :, 16:]), dim=2)
    torch.testing.assert_close(assembly, expected)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_convolution_lengths_match_masked_fill(dtype: torch.dtype) -> None:
    torch.default_generator.manual_seed(10)
    module = ConvolutionModule(12, 5).to(dtype)
    x = make_random_tensor((3, 17, 12), 10, dtype)
    padding_mask = torch.zeros(3, 17, dtype=torch.bool)
    padding_mask[1, 13:] = True
    padding_mask[2, 9:] = True
    valid_lengths = torch.tensor((17, 13, 9), dtype=torch.int32)

    projection = module.in_proj(x)
    value, gate = projection.chunk(2, dim=2)
    expected = value * torch.sigmoid(gate)
    expected = module.depthwise_conv(
        expected.masked_fill(padding_mask.unsqueeze(2), 0.0).permute(0, 2, 1)
    ).permute(0, 2, 1)
    shifted = expected - 1.0
    expected = module.out_proj(
        torch.maximum(shifted, torch.zeros_like(shifted))
        + torch.log1p(torch.exp(-torch.abs(shifted)))
        - 0.08 * expected
        - 0.313261687
    )

    actual = module(x, valid_lengths)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert torch.isfinite(actual).all()


def test_zipformer_modules_use_checkpoint_swoosh_variants() -> None:
    subsampling = make_zipformer().subsampling

    assert isinstance(subsampling.conv_activation, SwooshR)
    assert isinstance(subsampling.convnext_activation, SwooshL)
    assert isinstance(ConvolutionModule(8, 3).activation, SwooshR)
    assert isinstance(FeedforwardModule(8, 12).activation, SwooshL)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_bypass_uses_scale_as_later_weight(dtype: torch.dtype) -> None:
    module = BypassModule(4).to(dtype)
    with torch.no_grad():
        module.bypass_scale.copy_(torch.tensor((0.0, 0.25, 0.75, 1.0)))
    early = make_random_tensor((2, 7, 4), 28, dtype)
    later = make_random_tensor((2, 7, 4), 29, dtype)
    early[:, :, 3] = 0

    actual = module(early, later)
    scale = module.bypass_scale.float()
    expected = ((1.0 - scale) * early.float() + scale * later.float()).to(dtype)

    torch.testing.assert_close(actual[:, :, 0], early[:, :, 0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual[:, :, 3], later[:, :, 3], atol=0.0, rtol=0.0)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_zipformer_feedforward_matches_independent_swoosh_l_reference() -> None:
    torch.default_generator.manual_seed(29)
    module = FeedforwardModule(8, 12)
    inputs = make_random_tensor((2, 7, 8), 29)
    projected = module.in_proj(inputs)
    shifted = projected - 4.0
    activation = (
        torch.maximum(shifted, torch.zeros_like(shifted))
        + torch.log1p(torch.exp(-torch.abs(shifted)))
        - 0.08 * projected
        - 0.035
    )
    expected = module.out_proj(activation)

    actual = module(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "convolution_type,plugin_name,length_name",
    (
        (ConvolutionModule, ZIPFORMER_CONVOLUTION_PLUGIN_NAME, "valid_lengths"),
        (
            ConformerConvolution,
            PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
            "output_lengths",
        ),
    ),
    ids=("zipformer", "parakeet"),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_convolution_exports_as_tensorrt_plugin(
    tmp_path: Path,
    dtype: torch.dtype,
    convolution_type: type[ConvolutionModule] | type[ConformerConvolution],
    plugin_name: str,
    length_name: str,
) -> None:
    convolution = convolution_type(8, 5).eval().to(dtype)
    if convolution_type is ConvolutionModule:
        with torch.no_grad():
            convolution.depthwise_conv.weight.copy_(
                torch.arange(40, dtype=torch.float32).reshape(8, 1, 5) / 8
            )
            convolution.depthwise_conv.bias.copy_(torch.arange(8).float() / 8)
    onnx_path = tmp_path / "convolution.onnx"
    torch.onnx.export(
        convolution,
        (
            make_random_tensor((2, 17, 8), 31, dtype),
            torch.tensor([17, 9], dtype=torch.int32),
        ),
        onnx_path,
        input_names=("x", length_name),
        output_names=("output",),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    check_onnx_model_with_custom_plugins(model, {plugin_name: 1})
    (node,) = (node for node in model.graph.node if node.op_type == plugin_name)
    projected, lengths, weight_name, bias_name = node.input
    (convolved,) = node.output
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    } == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}
    assert [value.name for value in model.graph.input] == ["x", length_name]
    assert [value.name for value in model.graph.output] == ["output"]
    assert lengths == length_name
    assert convolved != "output"
    assert get_onnx_producer(model, projected).op_type == "Mul"
    assert [
        consumer.op_type for consumer in model.graph.node if convolved in consumer.input
    ] == ["MatMul"]
    for name in (projected, convolved, "output"):
        assert get_onnx_shape(model, name) == (2, 17, 8)
        assert get_onnx_element_type(model, name) == ONNX_DTYPES[dtype]
    assert get_onnx_shape(model, lengths) == (2,)
    assert get_onnx_element_type(model, lengths) == onnx.TensorProto.INT32
    initializers = {value.name: value for value in model.graph.initializer}
    for name, shape in ((weight_name, (5, 8)), (bias_name, (8,))):
        assert tuple(initializers[name].dims) == shape
        assert initializers[name].data_type == ONNX_DTYPES[dtype]

    if convolution_type is ConvolutionModule:
        for name, expected in (
            (weight_name, convolution.depthwise_conv.weight[:, 0].permute(1, 0)),
            (bias_name, convolution.depthwise_conv.bias),
        ):
            np.testing.assert_array_equal(
                onnx.numpy_helper.to_array(initializers[name]).astype(np.float32),
                expected.detach().float().numpy(),
            )
        assert not any(node.op_type == "Softplus" for node in model.graph.node)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_parakeet_conformer_convolution_matches_depthwise_convolution(
    dtype: torch.dtype,
) -> None:
    torch.default_generator.manual_seed(5)
    convolution = ConformerConvolution(8, 5).eval().to(dtype)
    x = make_random_tensor((3, 17, 8), 32, dtype)
    output_lengths = torch.tensor((17, 13, 9), dtype=torch.int32)
    padding_mask = torch.zeros(3, 17, dtype=torch.bool)
    padding_mask[1, 13:] = True
    padding_mask[2, 9:] = True

    expected = torch.nn.functional.glu(convolution.pointwise_conv1(x), dim=2)
    expected = expected.masked_fill(padding_mask.unsqueeze(2), 0.0).permute(0, 2, 1)
    expected = torch.nn.functional.pad(
        expected, (convolution.padding, convolution.padding)
    )
    expected = convolution.depthwise_conv(expected)
    expected = torch.nn.functional.silu(convolution.batch_norm(expected)).permute(
        0, 2, 1
    )
    expected = convolution.pointwise_conv2(expected)

    actual = convolution(x, output_lengths)

    assert torch.isfinite(actual).all()
    atol, rtol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_parakeet_conformer_convolution_exports_exact_folded_plugin_inputs(
    tmp_path: Path,
) -> None:
    convolution = ConformerConvolution(8, 5).eval()
    with torch.no_grad():
        convolution.depthwise_conv.weight.copy_(
            torch.arange(40, dtype=torch.float32).reshape(8, 1, 5) / 20.0 - 0.5
        )
        convolution.batch_norm.weight.copy_(torch.linspace(0.5, 1.2, 8))
        convolution.batch_norm.bias.copy_(torch.linspace(-0.3, 0.4, 8))
        convolution.batch_norm.running_mean.copy_(torch.linspace(-0.2, 0.2, 8))
        convolution.batch_norm.running_var.copy_(torch.linspace(0.4, 1.1, 8))

    onnx_path = tmp_path / "conformer_convolution_contract.onnx"
    torch.onnx.export(
        convolution,
        (
            make_random_tensor((2, 17, 8), 43),
            torch.tensor((17, 9), dtype=torch.int32),
        ),
        onnx_path,
        input_names=("x", "output_lengths"),
        output_names=("output",),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.load(onnx_path)
    check_onnx_model_with_custom_plugins(
        model, {PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME: 1}
    )
    (custom_node,) = (
        node
        for node in model.graph.node
        if node.op_type == PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME
    )
    initializers = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    exported_weight = initializers[custom_node.input[2]]
    exported_bias = initializers[custom_node.input[3]]
    batch_norm_scale = convolution.batch_norm.weight / torch.sqrt(
        convolution.batch_norm.running_var + convolution.batch_norm.eps
    )
    expected_weight = (
        convolution.depthwise_conv.weight[:, 0] * batch_norm_scale.unsqueeze(1)
    ).permute(1, 0)
    expected_bias = convolution.batch_norm.bias - (
        convolution.batch_norm.running_mean * batch_norm_scale
    )

    assert exported_weight.shape == (5, 8)
    assert exported_bias.shape == (8,)
    torch.testing.assert_close(
        torch.from_numpy(exported_weight.copy()), expected_weight
    )
    torch.testing.assert_close(torch.from_numpy(exported_bias.copy()), expected_bias)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_parakeet_flash_attention_exports_as_tensorrt_plugin(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    attention = RelPositionMultiHeadAttention(3, 12).eval().to(dtype)
    with torch.no_grad():
        attention.pos_bias_u.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        attention.pos_bias_v.copy_(
            -torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        )
    inputs = (
        make_random_tensor((2, 7, 12), 34, dtype),
        make_random_tensor((1, 13, 12), 35, dtype),
        torch.full((2,), 7, dtype=torch.int32),
    )
    onnx_path = tmp_path / f"parakeet_flash_attention_{dtype}.onnx"
    torch.onnx.export(
        attention,
        inputs,
        onnx_path,
        input_names=("x", "pos_emb", "output_lengths"),
        output_names=("output",),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    check_onnx_model_with_custom_plugins(
        model, {PARAKEET_FLASH_ATTENTION_PLUGIN_NAME: 1}
    )
    (custom_node,) = (
        node
        for node in model.graph.node
        if node.op_type == PARAKEET_FLASH_ATTENTION_PLUGIN_NAME
    )
    assert len(custom_node.input) == 5
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    }
    assert attributes == {
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        "scale": pytest.approx(0.5),
    }
    assert [value.name for value in model.graph.input] == [
        "x",
        "pos_emb",
        "output_lengths",
    ]
    assert [value.name for value in model.graph.output] == ["output"]
    assert custom_node.input[0] != "x"
    assert custom_node.input[1] != "pos_emb"
    assert custom_node.input[4] == "output_lengths"
    assert custom_node.output[0] != "output"
    assert [
        get_onnx_producer(model, name).op_type for name in custom_node.input[:2]
    ] == ["MatMul", "MatMul"]
    assert [
        node.op_type for node in model.graph.node if custom_node.output[0] in node.input
    ] == ["MatMul"]
    assert [get_onnx_shape(model, name) for name in custom_node.input[:2]] == [
        (2, 7, 36),
        (1, 13, 12),
    ]
    assert get_onnx_shape(model, custom_node.input[4]) == (2,)
    for name in (*custom_node.input[:4], *custom_node.output):
        assert get_onnx_element_type(model, name) == ONNX_DTYPES[dtype]
    assert get_onnx_element_type(model, custom_node.input[4]) == onnx.TensorProto.INT32
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    for input_name, expected in zip(
        custom_node.input[2:4],
        (attention.pos_bias_u, attention.pos_bias_v),
        strict=True,
    ):
        initializer = initializers[input_name]
        assert tuple(initializer.dims) == (3, 4)
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(initializer).astype(np.float32),
            expected.detach().float().numpy(),
        )
    assert get_onnx_shape(model, custom_node.output[0]) == (2, 7, 12)
    assert get_onnx_shape(model, "output") == (2, 7, 12)


def test_parakeet_encoder_exports_fixed_batch_with_dynamic_time(
    tmp_path: Path,
) -> None:
    encoder = make_parakeet_encoder()
    audio = torch.zeros(2, 8000)
    audio_lengths = torch.full((2,), 8000, dtype=torch.int64)
    onnx_path = tmp_path / "parakeet_encoder.onnx"
    torch.onnx.export(
        encoder,
        (audio, audio_lengths),
        onnx_path,
        dynamic_shapes={
            "audio": {1: torch.export.Dim.DYNAMIC},
            "audio_lengths": {},
        },
        input_names=("audio", "audio_lengths"),
        output_names=("encoder_output", "encoder_output_lengths"),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    check_onnx_model_with_custom_plugins(
        model,
        {
            PARAKEET_FEATURE_PLUGIN_NAME: 1,
            PARAKEET_FLASH_ATTENTION_PLUGIN_NAME: 1,
            PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME: 1,
        },
    )
    feature_node = next(
        node
        for node in model.graph.node
        if node.op_type == PARAKEET_FEATURE_PLUGIN_NAME
    )
    assert len(feature_node.input) == 4
    assert list(feature_node.input[:2]) == ["audio", "audio_lengths"]
    feature_attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in feature_node.attribute
    }
    assert feature_attributes == {
        "eps": pytest.approx(1e-5),
        "frame_shift": 160,
        "log_eps": pytest.approx(2**-24),
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        "preemph": pytest.approx(0.97),
    }
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    assert tuple(initializers[feature_node.input[2]].dims) == (512,)
    assert tuple(initializers[feature_node.input[3]].dims) == (257, 16)
    assert initializers[feature_node.input[2]].data_type == onnx.TensorProto.FLOAT
    assert initializers[feature_node.input[3]].data_type == onnx.TensorProto.FLOAT
    convolution_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    assert len(convolution_nodes) == 3
    assert sorted(
        next(
            (
                onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
                if attribute.name == "group"
            ),
            1,
        )
        for node in convolution_nodes
    ) == [1, 4, 4]
    assert {tuple(initializers[node.input[1]].dims) for node in convolution_nodes} == {
        (4, 1, 3, 3)
    }

    assert_encoder_onnx_interface(model, 16, onnx.TensorProto.FLOAT16)


def test_zipformer_encoder_exports_dynamic_plugin_contract(tmp_path: Path) -> None:
    encoder = make_zipformer(dtype=torch.float16)
    audio_lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    audio = add_zipformer_right_context(
        make_random_tensor((2, int(audio_lengths.max())), 36), audio_lengths
    )
    onnx_path = tmp_path / "zipformer_encoder.onnx"
    torch.onnx.export(
        encoder,
        (audio, audio_lengths),
        onnx_path,
        dynamic_shapes={
            "audio": {1: torch.export.Dim.DYNAMIC},
            "audio_lengths": {},
        },
        input_names=("audio", "audio_lengths"),
        output_names=("encoder_output", "encoder_output_lengths"),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    check_onnx_model_with_custom_plugins(
        model,
        {
            ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME: 18,
            ZIPFORMER_CONVOLUTION_PLUGIN_NAME: 12,
            ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME: 6,
            ZIPFORMER_FEATURE_PLUGIN_NAME: 1,
            ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME: 1,
            ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME: 6,
            ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME: 6,
        },
    )
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    convolution_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    assert len(convolution_nodes) == 4
    assert sorted(
        next(
            (
                onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
                if attribute.name == "group"
            ),
            1,
        )
        for node in convolution_nodes
    ) == [1, 1, 1, 8]
    assert {tuple(initializers[node.input[1]].dims) for node in convolution_nodes} == {
        (2, 1, 3, 3),
        (4, 2, 3, 3),
        (8, 1, 7, 7),
        (8, 4, 3, 3),
    }
    assert {initializers[node.input[1]].data_type for node in convolution_nodes} == {
        onnx.TensorProto.FLOAT16
    }

    feature_node = next(
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_FEATURE_PLUGIN_NAME
    )
    assert (
        get_onnx_element_type(model, feature_node.output[0]) == onnx.TensorProto.FLOAT
    )

    assembly_node = next(
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME
    )
    assert len(assembly_node.input) == 6
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in assembly_node.attribute
    } == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}

    assembly_time = get_onnx_shape(model, assembly_node.input[0])[1]
    assert isinstance(assembly_time, str)
    assert [get_onnx_shape(model, name) for name in assembly_node.input] == [
        (2, assembly_time, channels) for channels in ZIPFORMER_ENCODER_DIMS
    ]
    assert get_onnx_shape(model, assembly_node.output[0]) == (2, assembly_time, 20)
    for name in (*assembly_node.input, *assembly_node.output):
        assert get_onnx_element_type(model, name) == onnx.TensorProto.FLOAT16

    assert_encoder_onnx_interface(model, 10, onnx.TensorProto.FLOAT)


def test_partitioned_zipformer_encoder_matches_unsplit() -> None:
    encoder = make_zipformer()
    partitioned_encoder = make_zipformer(subsampling_batch_partitions=4)
    partitioned_encoder.load_state_dict(encoder.state_dict())
    audio = make_random_tensor((5, 3200), 37)
    lengths = torch.full((5,), 3200, dtype=torch.int64)
    audio = add_zipformer_right_context(audio, lengths)
    partition_batch_sizes: list[int] = []
    with (
        partitioned_encoder.subsampling.conv3.register_forward_pre_hook(
            lambda _module, inputs: partition_batch_sizes.append(inputs[0].size(0))
        ),
        torch.inference_mode(),
    ):
        expected = encoder(audio, lengths)
        actual = partitioned_encoder(audio, lengths)

    assert_balanced_partition_sizes(partition_batch_sizes, 5, 4)
    assert expected[0].shape == actual[0].shape == (5, 3, 10)
    assert expected[1].tolist() == actual[1].tolist() == [3, 3, 3, 3, 3]
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("batch_partitions", (1, 3))
def test_parakeet_subsampling_matches_functional_reference(
    batch_partitions: int,
) -> None:
    module = make_fast_conformer(batch_partitions).pre_encode
    feature_lengths = torch.tensor((23, 16, 9, 0), dtype=torch.int32)
    features = make_random_tensor((4, 23, 16), 38)
    for index, length in enumerate(feature_lengths.tolist()):
        features[index, length:] = 0.0

    with torch.inference_mode():
        expected = reference_parakeet_subsampling(module, features, feature_lengths)
        actual = module(features, feature_lengths)

    assert actual[0].shape == (4, 3, 16)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    assert expected[1].tolist() == actual[1].tolist() == [3, 2, 2, 0]
    assert torch.isfinite(actual[0]).all()
    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)


def test_fast_conformer_matches_independent_sequences() -> None:
    encoder = make_fast_conformer()
    lengths = torch.tensor([37, 53], dtype=torch.int32)
    features = torch.zeros(2, int(lengths.max()), 16)
    features[0, : lengths[0]] = make_random_tensor((int(lengths[0]), 16), 38)
    features[1, : lengths[1]] = make_random_tensor((int(lengths[1]), 16), 39)

    with torch.inference_mode():
        actual, actual_lengths = encoder(features, lengths)
    assert actual.shape == (2, 7, 16)
    assert actual_lengths.dtype == torch.int32
    assert actual_lengths.tolist() == [5, 7]
    assert torch.isfinite(actual).all()

    for index, length in enumerate(lengths):
        with torch.inference_mode():
            expected, expected_lengths = encoder(
                features[index : index + 1, : int(length)],
                lengths[index : index + 1],
            )
        torch.testing.assert_close(actual_lengths[index], expected_lengths[0])
        torch.testing.assert_close(
            actual[index, : expected_lengths[0]],
            expected[0, : expected_lengths[0]],
            atol=5e-4,
            rtol=5e-4,
        )


@pytest.mark.parametrize(
    "length_values",
    (
        (29, 37, 45, 53),
        (29, 37, 41, 45, 49, 51, 53),
    ),
    ids=("even", "uneven"),
)
def test_partitioned_fast_conformer_subsampling_matches_unsplit(
    length_values: tuple[int, ...],
) -> None:
    encoder = make_fast_conformer()
    split_encoder = make_fast_conformer(subsampling_batch_partitions=4)
    split_encoder.load_state_dict(encoder.state_dict())
    lengths = torch.tensor(length_values, dtype=torch.int32)
    features = make_random_tensor((len(lengths), int(lengths.max()), 16), 39)
    partition_batch_sizes: list[int] = []
    with (
        split_encoder.pre_encode.conv1.register_forward_pre_hook(
            lambda _module, inputs: partition_batch_sizes.append(inputs[0].size(0))
        ),
        torch.inference_mode(),
    ):
        expected = encoder(features, lengths)
        actual = split_encoder(features, lengths)

    assert_balanced_partition_sizes(partition_batch_sizes, len(length_values), 4)
    expected_lengths = [(length + 7) // 8 for length in length_values]
    assert expected[1].tolist() == actual[1].tolist() == expected_lengths
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_parakeet_encoder_matches_independent_waveforms(dtype: torch.dtype) -> None:
    encoder = make_parakeet_encoder(dtype)
    floating_tensors = [
        (name, tensor)
        for name, tensor in (*encoder.named_parameters(), *encoder.named_buffers())
        if tensor.is_floating_point()
    ]
    assert floating_tensors
    for name, tensor in floating_tensors:
        expected_dtype = (
            torch.float32
            if name.startswith(("feature_extractor.", "encoder.pos_enc."))
            else dtype
        )
        assert tensor.dtype == expected_dtype, name

    audio_lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    audio = make_random_tensor((2, int(audio_lengths.max())), 40)
    with torch.inference_mode():
        actual, actual_lengths = encoder(audio, audio_lengths)

    assert actual.shape == (2, 5, 16)
    assert actual.dtype == dtype
    assert actual_lengths.dtype == torch.int32
    assert actual_lengths.tolist() == [3, 4]
    assert torch.isfinite(actual).all()

    atol, rtol = DTYPE_TOLERANCES[dtype]
    for index, audio_length in enumerate(audio_lengths.tolist()):
        with torch.inference_mode():
            expected, expected_lengths = encoder(
                audio[index : index + 1, :audio_length],
                audio_lengths[index : index + 1],
            )
        torch.testing.assert_close(actual_lengths[index], expected_lengths[0])
        torch.testing.assert_close(
            actual[index, : expected_lengths[0]],
            expected[0, : expected_lengths[0]],
            atol=atol,
            rtol=rtol,
        )


def test_zipformer_encoder_matches_independent_waveforms() -> None:
    encoder = make_zipformer()

    lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    raw_audio = make_random_tensor((2, int(lengths.max())), 41)
    audio = add_zipformer_right_context(raw_audio, lengths)

    with torch.inference_mode():
        actual, actual_lengths = encoder(audio, lengths)

    assert actual.shape == (2, 6, 10)
    assert actual.dtype == torch.float32
    assert actual_lengths.dtype == torch.int32
    assert actual_lengths.tolist() == [3, 6]
    assert torch.isfinite(actual).all()
    for index, audio_length in enumerate(lengths.tolist()):
        individual_audio = add_zipformer_right_context(
            raw_audio[index : index + 1, :audio_length],
            lengths[index : index + 1],
        )
        with torch.inference_mode():
            expected, expected_lengths = encoder(
                individual_audio,
                lengths[index : index + 1],
            )
        torch.testing.assert_close(actual_lengths[index], expected_lengths[0])
        torch.testing.assert_close(
            actual[index, : expected_lengths[0]],
            expected[0, : expected_lengths[0]],
            atol=1e-2,
            rtol=1e-2,
        )


def test_zipformer_encoder_ignores_audio_after_reflected_context() -> None:
    encoder = make_zipformer()
    lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    raw_audio = make_random_tensor((2, int(lengths.max())), 42)
    expected_audio = add_zipformer_right_context(raw_audio, lengths)
    corrupted_audio = expected_audio.clone()
    tail_start = int(lengths[0]) + 200
    corrupted_audio[0, tail_start:] = 1000 * make_random_tensor(
        (corrupted_audio.size(1) - tail_start,), 43
    )

    with torch.inference_mode():
        expected, expected_lengths = encoder(expected_audio, lengths)
        actual, actual_lengths = encoder(corrupted_audio, lengths)

    torch.testing.assert_close(actual_lengths, expected_lengths)
    for index, valid_length in enumerate(actual_lengths.tolist()):
        torch.testing.assert_close(
            actual[index, :valid_length],
            expected[index, :valid_length],
            atol=0.0,
            rtol=0.0,
        )
