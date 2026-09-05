#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU contract tests for Zipformer convolutional subsampling."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from fast_gpu_asr.constants import ONNX_OPSET_VERSION, ZERO_LOG
from fast_gpu_asr.export.model.zipformer.subsampling import BiasNorm, Conv2dSubsampling

DTYPE_TOLERANCES = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 1e-2}


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create deterministic random data without changing the global RNG state.

    Parameters
    ----------
    shape : tuple[int, ...]
        Tensor dimensions.
    seed : int
        Seed for a private CPU generator.
    dtype : torch.dtype
        Type to which the FP32 samples are converted.

    Returns
    -------
    torch.Tensor
        Standard-normal CPU samples in the requested shape and dtype.
    """

    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


def fill_right_padding(features: torch.Tensor, lengths: torch.Tensor) -> None:
    """Fill frames beyond each valid input length with the production pad value.

    Parameters
    ----------
    features : torch.Tensor
        Features of shape ``(batch, frames, mel_bins)``, modified in place.
    lengths : torch.Tensor
        Valid frame counts of shape ``(batch,)``; negative counts are treated
        as zero and counts beyond capacity leave the utterance unchanged.
    """

    for index, length in enumerate(lengths.tolist()):
        features[index, max(0, length) :] = ZERO_LOG


def make_subsampling(
    input_dim: int = 16,
    output_dim: int = 12,
    layer1_channels: int = 2,
    layer2_channels: int = 4,
    layer3_channels: int = 8,
    batch_partitions: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Conv2dSubsampling:
    """Create a compact deterministic subsampling module for CPU tests.

    Parameters
    ----------
    input_dim : int
        Input mel-bin count.
    output_dim : int
        Projected feature width.
    layer1_channels : int
        First convolution's output channels.
    layer2_channels : int
        Second convolution's output channels.
    layer3_channels : int
        Third convolution's output channels.
    batch_partitions : int
        Number of partitions used by subsampling convolutions.
    dtype : torch.dtype
        Module precision; BF16 retains the production FP16 first convolution.

    Returns
    -------
    Conv2dSubsampling
        Evaluation-mode module with initialized BiasNorm scale and no change
        to the caller's RNG state.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        module = Conv2dSubsampling(
            input_dim=input_dim,
            output_dim=output_dim,
            layer1_channels=layer1_channels,
            layer2_channels=layer2_channels,
            layer3_channels=layer3_channels,
            batch_partitions=batch_partitions,
        ).eval()

    module.to(dtype)
    if dtype == torch.bfloat16:
        # Production keeps the first convolution in FP16 for BF16 encoders.
        module.conv1.to(torch.float16)
    module.out_norm.scale.fill_(1.0)
    return module


def get_onnx_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return fixed dimensions as integers and symbolic dimensions as strings.

    Parameters
    ----------
    value : onnx.ValueInfoProto
        Tensor value information from an exported graph.

    Returns
    -------
    tuple[int | str, ...]
        Dimensions in their original axis order.
    """

    return tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


def reference_subsampling(
    module: Conv2dSubsampling, features: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate subsampling without calling module ``forward`` methods.

    Parameters
    ----------
    module : Conv2dSubsampling
        Source of convolution, projection, and normalization parameters.
    features : torch.Tensor
        CPU features of shape ``(batch, frames, mel_bins)``.
    lengths : torch.Tensor
        Input frame counts of shape ``(batch,)``, clamped after subsampling.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Normalized projected features and bounded output frame counts.
    """

    output = features.unsqueeze(1)
    for layer, stride, padding in (
        (module.conv1, 1, (0, 1)),
        (module.conv2, 2, 0),
        (module.conv3, (1, 2), 0),
    ):
        output = torch.nn.functional.conv2d(
            output.to(layer.weight.dtype),
            layer.weight,
            layer.bias,
            stride=stride,
            padding=padding,
        )
        output = (
            torch.nn.functional.softplus(output - 1.0) - 0.08 * output - 0.313261687
        )

    output_frame_capacity = (features.size(1) - 7) // 2
    output_lengths = torch.clamp((lengths - 7) // 2, min=0, max=output_frame_capacity)
    valid_frames = torch.arange(
        output_frame_capacity, dtype=output_lengths.dtype, device=output_lengths.device
    ).unsqueeze(0) < output_lengths.unsqueeze(1)

    output = output * valid_frames.unsqueeze(1).unsqueeze(3)
    bypass = output
    output = torch.nn.functional.conv2d(
        output,
        module.depthwise_conv.weight,
        module.depthwise_conv.bias,
        padding=3,
        groups=output.size(1),
    )
    output = torch.nn.functional.linear(
        output.permute(0, 2, 3, 1),
        module.pointwise_conv1.weight,
        module.pointwise_conv1.bias,
    )
    output = torch.nn.functional.softplus(output - 4.0) - 0.08 * output - 0.035
    output = torch.nn.functional.linear(
        output, module.pointwise_conv2.weight, module.pointwise_conv2.bias
    ).permute(0, 3, 1, 2)
    output = bypass + output

    batch_size, channels, num_frames, frequency_dim = output.shape
    output = output.permute(0, 2, 1, 3).reshape(
        batch_size, num_frames, channels * frequency_dim
    )
    output = torch.nn.functional.linear(output, module.out.weight, module.out.bias)
    output_dtype = output.dtype
    output = output.float()
    rms = (
        torch.mean((output - module.out_norm.bias.float()) ** 2, dim=2, keepdim=True)
        .sqrt()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    output = (output * module.out_norm.scale.float() / rms).to(output_dtype)

    return output, output_lengths


def test_zipformer_subsampling_preserves_checkpoint_state_layout() -> None:
    module = make_subsampling()

    assert {
        name: tuple(tensor.shape) for name, tensor in module.state_dict().items()
    } == {
        "conv1.weight": (2, 1, 3, 3),
        "conv1.bias": (2,),
        "conv2.weight": (4, 2, 3, 3),
        "conv2.bias": (4,),
        "conv3.weight": (8, 4, 3, 3),
        "conv3.bias": (8,),
        "depthwise_conv.weight": (8, 1, 7, 7),
        "depthwise_conv.bias": (8,),
        "pointwise_conv1.weight": (24, 8),
        "pointwise_conv1.bias": (24,),
        "pointwise_conv2.weight": (8, 24),
        "pointwise_conv2.bias": (8,),
        "out.weight": (12, 24),
        "out.bias": (12,),
        "out_norm.scale": (),
        "out_norm.bias": (12,),
    }


def test_zipformer_subsampling_accepts_minimum_spatial_shape() -> None:
    module = make_subsampling(input_dim=7)
    features = make_random_tensor((1, 9, 7), seed=1)
    lengths = torch.tensor([9], dtype=torch.int32)

    with torch.inference_mode():
        expected_output, _ = reference_subsampling(module, features, lengths)
        output, output_lengths = module(features, lengths)

    assert output.shape == (1, 1, 12)
    assert output_lengths.tolist() == [1]
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6)


def test_zipformer_subsampling_supports_configurable_dimensions() -> None:
    module = make_subsampling(
        input_dim=17,
        output_dim=7,
        layer1_channels=3,
        layer2_channels=5,
        layer3_channels=6,
    )
    features = make_random_tensor((2, 12, 17), seed=12)
    lengths = torch.tensor((12, 9), dtype=torch.int32)

    with torch.inference_mode():
        expected_output, _ = reference_subsampling(module, features, lengths)
        output, output_lengths = module(features, lengths)

    assert output.shape == (2, 2, 7)
    assert output_lengths.tolist() == [2, 1]
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPE_TOLERANCES, ids=str)
@pytest.mark.parametrize(
    "partition_sizes",
    ((4,), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 3)),
    ids=("unpartitioned", "batch3", "batch4", "batch5", "batch7"),
)
def test_zipformer_subsampling_matches_functional_reference(
    partition_sizes: tuple[int, ...], dtype: torch.dtype
) -> None:
    batch_size = sum(partition_sizes)
    module = make_subsampling(batch_partitions=len(partition_sizes), dtype=dtype)
    lengths = torch.tensor((23, 20, 11, 8, 9, 10, 12)[:batch_size], dtype=torch.int32)
    features = make_random_tensor((batch_size, 23, 16), seed=2, dtype=dtype)
    fill_right_padding(features, lengths)
    original_features = features.clone()
    original_lengths = lengths.clone()
    original_state = {
        name: tensor.clone() for name, tensor in module.state_dict().items()
    }

    observed_sizes: dict[torch.nn.Module, list[int]] = {
        module.conv3: [],
        module.depthwise_conv: [],
    }
    hooks = [
        layer.register_forward_pre_hook(
            lambda layer, inputs: observed_sizes[layer].append(inputs[0].size(0))
        )
        for layer in observed_sizes
    ]
    try:
        with torch.inference_mode():
            expected = reference_subsampling(module, features, lengths)
            actual = module(features, lengths)
    finally:
        for hook in hooks:
            hook.remove()

    for layer, sizes in observed_sizes.items():
        assert sorted(sizes) == sorted(partition_sizes), layer
    assert torch.equal(features, original_features)
    assert torch.equal(lengths, original_lengths)
    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, original_state[name]), name
    assert actual[0].shape == (batch_size, 8, 12)
    assert actual[0].dtype == dtype
    assert torch.isfinite(actual[0]).all()
    tolerance = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(actual[0], expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_zipformer_subsampling_length_boundaries() -> None:
    module = make_subsampling()
    lengths = torch.tensor((-100, 8, 9, 10, 11, 12, 22, 23, 24, 100), dtype=torch.int32)
    features = make_random_tensor((len(lengths), 23, 16), seed=4)
    fill_right_padding(features, lengths)

    with torch.inference_mode():
        expected_output, _ = reference_subsampling(module, features, lengths)
        output, output_lengths = module(features, lengths)

    assert output.shape == (len(lengths), 8, 12)
    assert output_lengths.dtype == torch.int32
    assert output_lengths.tolist() == [0, 0, 1, 1, 2, 2, 7, 8, 8, 8]
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-6)


def test_partitioned_zipformer_subsampling_onnx_matches_dynamic_time(
    tmp_path: Path,
) -> None:
    module = make_subsampling(batch_partitions=3)
    num_frames = torch.export.Dim("num_frames", min=9, max=41)
    onnx_path = tmp_path / "zipformer_subsampling.onnx"
    example_features = make_random_tensor((4, 23, 16), seed=5)
    example_lengths = torch.tensor((23, 21, 17, 9), dtype=torch.int32)
    fill_right_padding(example_features, example_lengths)

    torch.onnx.export(
        module,
        (example_features, example_lengths),
        onnx_path,
        input_names=("features", "feature_lengths"),
        output_names=("output", "output_lengths"),
        dynamic_shapes=({1: num_frames}, {}),
        opset_version=ONNX_OPSET_VERSION,
    )
    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.checker.check_model(model)
    assert [opset.version for opset in model.opset_import if opset.domain == ""] == [
        ONNX_OPSET_VERSION
    ]
    assert [value.name for value in model.graph.input] == [
        "features",
        "feature_lengths",
    ]
    assert [value.name for value in model.graph.output] == ["output", "output_lengths"]
    features_shape = get_onnx_shape(model.graph.input[0])
    output_shape = get_onnx_shape(model.graph.output[0])
    assert features_shape == (4, "num_frames", 16)
    assert output_shape == (4, output_shape[1], 12)
    assert isinstance(output_shape[1], str)
    assert get_onnx_shape(model.graph.input[1]) == (4,)
    assert get_onnx_shape(model.graph.output[1]) == (4,)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.input[1].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[1].type.tensor_type.elem_type == onnx.TensorProto.INT32

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    initializer_shapes = {
        initializer.name: tuple(initializer.dims)
        for initializer in model.graph.initializer
    }
    for layer in (module.conv3, module.depthwise_conv):
        partition_nodes = [
            node
            for node in model.graph.node
            if node.op_type == "Conv"
            and initializer_shapes.get(node.input[1]) == tuple(layer.weight.shape)
        ]
        batch_sizes = [
            get_onnx_shape(value_info[node.input[0]])[0] for node in partition_nodes
        ]
        assert sorted(batch_sizes) == [1, 1, 2], layer
    evaluator = ReferenceEvaluator(model)

    for frame_count, input_lengths in (
        (9, (9, 9, 9, 9)),
        (10, (10, 9, 10, 9)),
        (11, (11, 10, 11, 9)),
        (24, (24, 23, 11, 9)),
        (31, (31, 30, 11, 9)),
        (41, (41, 40, 11, 9)),
        (23, (-100, 8, 24, 100)),
    ):
        features = make_random_tensor((4, frame_count, 16), seed=100 + frame_count)
        lengths = torch.tensor(input_lengths, dtype=torch.int32)
        fill_right_padding(features, lengths)
        with torch.inference_mode():
            expected_output, expected_lengths = module(
                features.clone(), lengths.clone()
            )
        actual_output, actual_lengths = evaluator.run(
            None, {"features": features.numpy(), "feature_lengths": lengths.numpy()}
        )
        np.testing.assert_allclose(
            actual_output,
            expected_output.numpy(),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"frame_count={frame_count}",
        )
        np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())


@pytest.mark.parametrize("batch_partitions", (1, 3))
def test_zipformer_subsampling_valid_prefix_is_invariant_to_right_padding(
    batch_partitions: int,
) -> None:
    module = make_subsampling(batch_partitions=batch_partitions)
    compact_features = torch.full((3, 24, 16), ZERO_LOG)
    extended_features = torch.full((3, 41, 16), ZERO_LOG)
    short_prefix = make_random_tensor((23, 16), seed=6)
    compact_features[0, :23] = short_prefix
    extended_features[0, :23] = short_prefix
    compact_features[1:] = make_random_tensor((2, 24, 16), seed=7)
    extended_features[1, :40] = make_random_tensor((40, 16), seed=8)
    extended_features[2] = make_random_tensor((41, 16), seed=9)
    compact_lengths = torch.tensor((23, 24, 24), dtype=torch.int32)
    extended_lengths = torch.tensor((23, 40, 41), dtype=torch.int32)

    with torch.inference_mode():
        compact, compact_output_lengths = module(compact_features, compact_lengths)
        extended, extended_output_lengths = module(extended_features, extended_lengths)

    assert compact_output_lengths.tolist() == [8, 8, 8]
    assert extended_output_lengths.tolist() == [8, 16, 17]
    valid_length = compact_output_lengths[0].item()
    torch.testing.assert_close(
        compact[0, :valid_length], extended[0, :valid_length], atol=1e-6, rtol=1e-6
    )


def test_bias_norm_exposes_converted_checkpoint_buffers() -> None:
    module = BiasNorm(12)
    state = {"scale": torch.tensor(1.3), "bias": make_random_tensor((12,), seed=10)}

    assert dict(module.named_parameters()) == {}
    assert set(dict(module.named_buffers())) == set(state)
    assert not any(buffer.requires_grad for buffer in module.buffers())
    module.load_state_dict(state)
    torch.testing.assert_close(module.state_dict(), state, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPE_TOLERANCES, ids=str)
def test_bias_norm_matches_float32_reference(dtype: torch.dtype) -> None:
    module = BiasNorm(12).to(dtype)
    module.scale.fill_(1.3)
    module.bias.copy_(make_random_tensor((12,), seed=13, dtype=dtype))
    inputs = make_random_tensor((2, 3, 12), seed=12, dtype=dtype)
    original_inputs = inputs.clone()
    original_scale = module.scale.clone()
    original_bias = module.bias.clone()
    expected = inputs.float() * module.scale.float()
    rms = torch.mean(
        (inputs.float() - module.bias.float()) ** 2, dim=2, keepdim=True
    ).sqrt()
    expected = (expected / rms.clamp_min(torch.finfo(torch.float32).tiny)).to(dtype)

    actual = module(inputs)

    assert torch.equal(inputs, original_inputs)
    assert torch.equal(module.scale, original_scale)
    assert torch.equal(module.bias, original_bias)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPE_TOLERANCES, ids=str)
@pytest.mark.parametrize("value", (0.0, 256.0), ids=("zero-rms", "large-magnitude"))
def test_bias_norm_handles_extreme_magnitudes(dtype: torch.dtype, value: float) -> None:
    module = BiasNorm(12).to(dtype)
    module.scale.fill_(1.0)
    inputs = torch.full((2, 3, 12), value, dtype=dtype)

    actual = module(inputs)

    expected = torch.full_like(inputs, 0.0 if value == 0 else 1.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
