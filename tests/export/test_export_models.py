#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Integration tests for exported Zipformer and Parakeet encoder modules."""

from collections import Counter
from collections.abc import Iterator
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
    ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
    ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.parakeet.attention import RelPositionMultiHeadAttention
from fast_gpu_asr.export.model.parakeet.parakeet import (
    ConformerConvolution,
    FastConformer,
    ParakeetTDTEncoder,
)
from fast_gpu_asr.export.model.zipformer.activation import SwooshL, SwooshR
from fast_gpu_asr.export.model.zipformer.subsampling import BiasNorm
from fast_gpu_asr.export.model.zipformer.zipformer import (
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


@pytest.fixture(autouse=True)
def isolate_torch_rng() -> Iterator[None]:
    """Make random model initialization deterministic and local to each test."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a deterministic CPU tensor without changing global RNG state."""

    return torch.randn(
        shape,
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    )


def get_onnx_element_type(model: onnx.ModelProto, name: str) -> int:
    """Return a graph value or initializer element type by name."""

    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name:
            return value.type.tensor_type.elem_type
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer.data_type
    raise AssertionError(f"ONNX value {name} has no type information.")


def make_fast_conformer(subsampling_batch_partitions: int = 1) -> FastConformer:
    """Build a deterministic compact Fast Conformer encoder."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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


def make_parakeet_encoder(dtype: torch.dtype = torch.float16) -> ParakeetTDTEncoder:
    """Build a compact Parakeet encoder with production frontend settings."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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
    """Build a deterministic compact six-stack Zipformer encoder."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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
            encoder_dims=[8, 12, 16, 20, 16, 12],
            num_encoder_layers=[1, 1, 1, 1, 1, 1],
            downsampling_factors=[1, 2, 4, 8, 4, 2],
            bypass_scales=[
                torch.ones(dimension) for dimension in (8, 12, 16, 20, 16, 12)
            ],
            num_heads=[1, 1, 1, 1, 1, 1],
            feedforward_dims=[16, 20, 24, 28, 24, 20],
            cnn_module_kernels=[3, 3, 3, 3, 3, 3],
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
    """Keep feature/output boundaries FP32 and the encoder at configured precision."""

    encoder = make_zipformer(dtype=dtype)
    first_convolution_dtype = torch.float16 if dtype == torch.bfloat16 else dtype

    assert {buffer.dtype for buffer in encoder.feature_extractor.buffers()} == {
        torch.float32
    }
    for name, tensor in encoder.state_dict().items():
        if name.startswith("projection_output."):
            expected_dtype = torch.float32
        elif name.startswith("subsampling.conv1."):
            expected_dtype = first_convolution_dtype
        else:
            expected_dtype = dtype
        assert tensor.dtype == expected_dtype, name


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_returns_fp32_output(dtype: torch.dtype) -> None:
    """Return FP32 encoder projections for every supported internal precision."""

    encoder = make_zipformer(dtype=dtype)
    lengths = torch.tensor([3200], dtype=torch.int64)
    audio = add_zipformer_right_context(make_random_tensor((1, 3200), 1), lengths)

    with torch.inference_mode():
        output, output_lengths = encoder(audio, lengths)

    assert output.dtype == torch.float32
    assert output_lengths.dtype == torch.int32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_ctc_returns_fp32_normalized_log_probs(
    dtype: torch.dtype,
) -> None:
    """Normalize CTC logits in FP32 without changing the projection values."""

    encoder = make_zipformer(dtype=dtype, use_ctc=True)
    lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    audio = torch.zeros(2, int(lengths.max()))
    audio[0, : lengths[0]] = make_random_tensor((int(lengths[0]),), 2)
    audio[1, : lengths[1]] = make_random_tensor((int(lengths[1]),), 3)
    audio = add_zipformer_right_context(audio, lengths)
    projected_outputs = []
    projection_hook = encoder.projection_output.register_forward_hook(
        lambda _module, _inputs, output: projected_outputs.append(output)
    )

    with torch.inference_mode():
        output, output_lengths = encoder(audio, lengths)
    projection_hook.remove()

    assert output_lengths.tolist() == [3, 6]
    assert len(projected_outputs) == 1
    assert encoder.projection_output.weight.dtype == torch.float32
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output, torch.nn.functional.log_softmax(projected_outputs[0], dim=2)
    )
    torch.testing.assert_close(
        torch.logsumexp(output, dim=2),
        torch.zeros_like(output[:, :, 0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_zipformer_convolution_lengths_match_masked_fill() -> None:
    """Check length-based masking against the original masked convolution."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        module = ConvolutionModule(12, 5)
    x = make_random_tensor((3, 17, 12), 10)
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
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_zipformer_modules_use_checkpoint_swoosh_variants() -> None:
    """Keep each checkpointed branch paired with its trained Swoosh variant."""

    subsampling = make_zipformer().subsampling

    assert isinstance(subsampling.conv_activation, SwooshR)
    assert isinstance(subsampling.convnext_activation, SwooshL)
    assert isinstance(ConvolutionModule(8, 3).activation, SwooshR)
    assert isinstance(FeedforwardModule(8, 12).activation, SwooshL)


def test_zipformer_feedforward_matches_independent_swoosh_l_reference() -> None:
    """Match the feed-forward projections and an independent Swoosh-L formula."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
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


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_zipformer_convolution_exports_as_tensorrt_plugin(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    """Check that ONNX retains the fused Zipformer convolution custom node."""

    inputs = (
        make_random_tensor((2, 17, 8), 31, dtype),
        torch.full((2,), 17, dtype=torch.int32),
    )
    onnx_path = tmp_path / f"zipformer_convolution_{dtype}.onnx"
    torch.onnx.export(
        ConvolutionModule(8, 5).eval().to(dtype),
        inputs,
        onnx_path,
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_CONVOLUTION_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert (custom_node.domain, custom_node.op_type) == (
        "",
        ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
    )
    assert len(custom_node.input) == 4
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    } == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}
    for name in (*custom_node.input[:1], *custom_node.input[2:], *custom_node.output):
        assert get_onnx_element_type(model, name) == ONNX_DTYPES[dtype]
    assert get_onnx_element_type(model, custom_node.input[1]) == onnx.TensorProto.INT32
    assert not any(node.op_type == "Softplus" for node in model.graph.node)
    output_shape = model.graph.output[0].type.tensor_type.shape
    assert [dimension.dim_value for dimension in output_shape.dim] == [2, 17, 8]


def add_zipformer_right_context(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    right_padding_samples: int = 200,
) -> torch.Tensor:
    """Append reflected waveform context for the Zipformer frontend."""

    padded = torch.zeros(
        audio.size(0),
        audio.size(1) + right_padding_samples,
        dtype=audio.dtype,
    )
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        padded[index, :length] = audio[index, :length]
        reflected_samples = min(length, right_padding_samples)
        padded[index, length : length + reflected_samples] = torch.flip(
            audio[index, length - reflected_samples : length],
            dims=(0,),
        )
        if reflected_samples < right_padding_samples:
            padding_start = length + reflected_samples
            padded[index, padding_start : length + right_padding_samples] = audio[
                index, 0
            ]
    return padded


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_conformer_convolution_matches_depthwise_convolution(
    dtype: torch.dtype,
) -> None:
    """Check eager masking, depthwise convolution, BatchNorm, and SiLU."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
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
    expected = convolution.activation(convolution.batch_norm(expected)).permute(0, 2, 1)
    expected = convolution.pointwise_conv2(expected)

    torch.testing.assert_close(convolution(x, output_lengths), expected)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_conformer_convolution_exports_as_tensorrt_plugin(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    """Check that ONNX retains the typed Conformer convolution custom node."""

    inputs = (
        make_random_tensor((2, 17, 8), 33, dtype),
        torch.full((2,), 17, dtype=torch.int32),
    )
    onnx_path = tmp_path / f"conformer_convolution_{dtype}.onnx"
    torch.onnx.export(
        ConformerConvolution(8, 5).eval().to(dtype),
        inputs,
        onnx_path,
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert (custom_node.domain, custom_node.op_type) == (
        "",
        PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
    )
    for name in (*custom_node.input[:1], *custom_node.input[2:], *custom_node.output):
        assert get_onnx_element_type(model, name) == ONNX_DTYPES[dtype]
    assert get_onnx_element_type(model, custom_node.input[1]) == onnx.TensorProto.INT32
    output_shape = model.graph.output[0].type.tensor_type.shape
    assert [dimension.dim_value for dimension in output_shape.dim] == [2, 17, 8]


def test_conformer_convolution_exports_exact_folded_plugin_inputs(
    tmp_path: Path,
) -> None:
    """Verify BatchNorm folding, weight layout, and custom-node input order."""

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
        opset_version=ONNX_OPSET_VERSION,
    )

    graph = onnx.load(onnx_path).graph
    (custom_node,) = (
        node
        for node in graph.node
        if node.op_type == PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME
    )
    assert custom_node.input[1] == "output_lengths"
    assert len(custom_node.input) == 4
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    }
    assert attributes == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}

    initializers = {
        initializer.name: np.asarray(onnx.numpy_helper.to_array(initializer))
        for initializer in graph.initializer
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
    """Check that ONNX retains the fused Parakeet attention custom node."""

    attention = RelPositionMultiHeadAttention(3, 12).eval().to(dtype)
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
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == PARAKEET_FLASH_ATTENTION_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert (custom_node.domain, custom_node.op_type) == (
        "",
        PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
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
    for name in (*custom_node.input[:4], *custom_node.output):
        assert get_onnx_element_type(model, name) == ONNX_DTYPES[dtype]
    assert get_onnx_element_type(model, custom_node.input[4]) == onnx.TensorProto.INT32
    output_shape = model.graph.output[0].type.tensor_type.shape
    assert [dimension.dim_value for dimension in output_shape.dim] == [2, 7, 12]


def test_parakeet_encoder_exports_fixed_batch_with_dynamic_time(
    tmp_path: Path,
) -> None:
    """Preserve the production encoder's fixed-batch, dynamic-time ONNX contract."""

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
    custom_op_types = {
        PARAKEET_FEATURE_PLUGIN_NAME,
        PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
        PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
    }
    custom_nodes = Counter(
        (node.domain, node.op_type)
        for node in model.graph.node
        if node.op_type in custom_op_types
    )
    assert custom_nodes == Counter(
        {
            ("", PARAKEET_FEATURE_PLUGIN_NAME): 1,
            ("", PARAKEET_FLASH_ATTENTION_PLUGIN_NAME): 1,
            ("", PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME): 1,
        }
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
    graph_inputs = {value.name: value for value in model.graph.input}
    graph_outputs = {value.name: value for value in model.graph.output}
    assert tuple(graph_inputs) == ("audio", "audio_lengths")
    assert tuple(graph_outputs) == ("encoder_output", "encoder_output_lengths")
    assert graph_inputs["audio"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert (
        graph_inputs["audio_lengths"].type.tensor_type.elem_type
        == onnx.TensorProto.INT64
    )
    assert (
        graph_outputs["encoder_output"].type.tensor_type.elem_type
        == onnx.TensorProto.FLOAT16
    )
    assert (
        graph_outputs["encoder_output_lengths"].type.tensor_type.elem_type
        == onnx.TensorProto.INT32
    )
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

    audio_shape = graph_inputs["audio"].type.tensor_type.shape
    assert audio_shape.dim[0].dim_value == 2
    assert audio_shape.dim[1].dim_param
    assert [
        dimension.dim_value
        for dimension in graph_inputs["audio_lengths"].type.tensor_type.shape.dim
    ] == [2]
    encoder_output_shape = graph_outputs["encoder_output"].type.tensor_type.shape
    assert encoder_output_shape.dim[0].dim_value == 2
    assert encoder_output_shape.dim[1].dim_param
    assert encoder_output_shape.dim[2].dim_value == 16
    assert [
        dimension.dim_value
        for dimension in graph_outputs[
            "encoder_output_lengths"
        ].type.tensor_type.shape.dim
    ] == [2]


def test_zipformer_encoder_exports_output_assembly_contract(tmp_path: Path) -> None:
    """Check the top-level Zipformer output-assembly ONNX contract."""

    encoder = make_zipformer(dtype=torch.float32)
    audio_lengths = torch.tensor([3200], dtype=torch.int64)
    audio = add_zipformer_right_context(
        make_random_tensor((1, 3200), 36), audio_lengths
    )
    onnx_path = tmp_path / "zipformer_output_assembly.onnx"
    torch.onnx.export(
        encoder,
        (audio, audio_lengths),
        onnx_path,
        input_names=("audio", "audio_lengths"),
        output_names=("encoder_output", "encoder_output_lengths"),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert custom_node.domain == ""
    assert len(custom_node.input) == 6
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    } == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}

    shapes = {
        value.name: tuple(
            dimension.dim_value for dimension in value.type.tensor_type.shape.dim
        )
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    assert [shapes[name] for name in custom_node.input] == [
        (1, 6, channels) for channels in (8, 12, 16, 20, 16, 12)
    ]
    assert shapes[custom_node.output[0]] == (1, 6, 20)
    for name in (*custom_node.input, *custom_node.output):
        assert get_onnx_element_type(model, name) == onnx.TensorProto.FLOAT


def test_partitioned_zipformer_subsampling_matches_unsplit() -> None:
    """Keep full Zipformer outputs unchanged by subsampling batch partitioning."""

    encoder = make_zipformer()
    partitioned_encoder = make_zipformer(subsampling_batch_partitions=4)
    partitioned_encoder.load_state_dict(encoder.state_dict())
    audio = make_random_tensor((4, 3200), 37)
    lengths = torch.full((4,), 3200, dtype=torch.int32)
    audio = add_zipformer_right_context(audio, lengths)

    with torch.inference_mode():
        expected = encoder(audio, lengths)
        actual = partitioned_encoder(audio, lengths)

    torch.testing.assert_close(actual[0], expected[0], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual[1], expected[1])


def test_fast_conformer_matches_independent_sequences() -> None:
    """Prevent padded feature tails from affecting valid batched outputs."""

    encoder = make_fast_conformer()
    lengths = torch.tensor([37, 53], dtype=torch.int32)
    features = torch.zeros(2, int(lengths.max()), 16)
    features[0, : lengths[0]] = make_random_tensor((int(lengths[0]), 16), 38)
    features[1, : lengths[1]] = make_random_tensor((int(lengths[1]), 16), 39)

    with torch.inference_mode():
        actual, actual_lengths = encoder(features, lengths)
    assert actual_lengths.dtype == torch.int32

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
    """Match unsplit subsampling for exact and uneven nonempty partitions."""

    encoder = make_fast_conformer()
    split_encoder = make_fast_conformer(subsampling_batch_partitions=4)
    split_encoder.load_state_dict(encoder.state_dict())
    lengths = torch.tensor(length_values, dtype=torch.int32)
    features = make_random_tensor((len(lengths), int(lengths.max()), 16), 39)

    with torch.inference_mode():
        expected = encoder(features, lengths)
        actual = split_encoder(features, lengths)

    torch.testing.assert_close(actual[0], expected[0], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=FLOAT_DTYPE_IDS)
def test_parakeet_encoder_matches_independent_waveforms(dtype: torch.dtype) -> None:
    """Preserve precision, lengths, and valid outputs in a padded waveform batch."""

    encoder = make_parakeet_encoder(dtype)
    assert {buffer.dtype for buffer in encoder.feature_extractor.buffers()} == {
        torch.float32
    }
    assert {parameter.dtype for parameter in encoder.encoder.parameters()} == {dtype}

    audio_lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    audio = make_random_tensor((2, int(audio_lengths.max())), 40)
    with torch.inference_mode():
        actual, actual_lengths = encoder(audio, audio_lengths)

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
    """Prevent padded waveform tails from affecting valid Zipformer outputs."""

    encoder = make_zipformer()

    lengths = torch.tensor([3200, 5171], dtype=torch.int64)
    raw_audio = make_random_tensor((2, int(lengths.max())), 41)
    audio = add_zipformer_right_context(raw_audio, lengths)
    audio[0, int(lengths[0]) + 200 :] = make_random_tensor(
        (audio.size(1) - int(lengths[0]) - 200,), 42
    )

    with torch.inference_mode():
        actual, actual_lengths = encoder(audio, lengths)

    assert actual_lengths.tolist() == [3, 6]
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
