#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU contract tests for Zipformer convolutional subsampling."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from fast_gpu_asr.constants import ZERO_LOG
from fast_gpu_asr.export.model.zipformer.subsampling import (
    BiasNorm,
    Conv2dSubsampling,
)

DTYPE_TOLERANCES = {
    torch.float32: (1e-6, 1e-6),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create deterministic random data without changing the global RNG state."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def make_subsampling(
    input_dim: int = 16,
    batch_partitions: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Conv2dSubsampling:
    """Create a compact deterministic subsampling module for CPU tests."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = Conv2dSubsampling(
            input_dim=input_dim,
            output_dim=12,
            layer1_channels=2,
            layer2_channels=4,
            layer3_channels=8,
            batch_partitions=batch_partitions,
        ).eval()

    module.to(dtype)
    if dtype == torch.bfloat16:
        # Production keeps the first convolution in FP16 for BF16 encoders.
        module.conv1.to(torch.float16)
    module.out_norm.scale.fill_(1.0)
    return module


def reference_subsampling(
    module: Conv2dSubsampling, features: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the subsampling equations without calling module ``forward`` methods."""

    output = torch.nn.functional.conv2d(
        features.unsqueeze(1).to(module.conv1.weight.dtype),
        module.conv1.weight,
        module.conv1.bias,
        stride=module.conv1.stride,
        padding=module.conv1.padding,
    )
    output = (
        torch.nn.functional.softplus(output - 1.0) - 0.08 * output - 0.313261687
    ).to(module.conv2.weight.dtype)
    output = torch.nn.functional.conv2d(
        output,
        module.conv2.weight,
        module.conv2.bias,
        stride=module.conv2.stride,
    )
    output = torch.nn.functional.softplus(output - 1.0) - 0.08 * output - 0.313261687

    output_lengths = torch.clamp((lengths - 7) // 2, min=0, max=output.size(2) - 2)
    valid_frames = torch.arange(
        output.size(2) - 2,
        dtype=output_lengths.dtype,
        device=output_lengths.device,
    ).unsqueeze(0) < output_lengths.unsqueeze(1)

    output = torch.nn.functional.conv2d(
        output,
        module.conv3.weight,
        module.conv3.bias,
        stride=module.conv3.stride,
    )
    output = torch.nn.functional.softplus(output - 1.0) - 0.08 * output - 0.313261687
    output = output * valid_frames.unsqueeze(1).unsqueeze(3)
    bypass = output
    output = torch.nn.functional.conv2d(
        output,
        module.depthwise_conv.weight,
        module.depthwise_conv.bias,
        padding=module.depthwise_conv.padding,
        groups=module.depthwise_conv.groups,
    )
    output = torch.nn.functional.linear(
        output.permute(0, 2, 3, 1),
        module.pointwise_conv1.weight,
        module.pointwise_conv1.bias,
    )
    output = torch.nn.functional.softplus(output - 4.0) - 0.08 * output - 0.035
    output = torch.nn.functional.linear(
        output,
        module.pointwise_conv2.weight,
        module.pointwise_conv2.bias,
    ).permute(0, 3, 1, 2)
    output = bypass + output

    batch_size, channels, num_frames, frequency_dim = output.shape
    output = output.permute(0, 2, 1, 3).reshape(
        batch_size, num_frames, channels * frequency_dim
    )
    output = torch.nn.functional.linear(output, module.out.weight, module.out.bias)
    output_dtype = output.dtype
    output = output.float()
    rms = torch.mean(
        (output - module.out_norm.bias.float()) ** 2, dim=2, keepdim=True
    ).sqrt()
    rms = rms.clamp_min(torch.finfo(torch.float32).tiny)
    output = (output * module.out_norm.scale.float() / rms).to(output_dtype)

    return output, output_lengths


def test_zipformer_subsampling_accepts_minimum_spatial_shape() -> None:
    module = make_subsampling(input_dim=7)
    features = make_random_tensor((2, 9, 7), seed=1)
    lengths = torch.tensor([9, 9], dtype=torch.int32)

    output, output_lengths = module(features, lengths)

    assert output.shape == (2, 1, 12)
    assert output.dtype == torch.float32
    assert output_lengths.dtype == torch.int32
    assert output_lengths.tolist() == [1, 1]
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_partitions", (1, 3))
def test_zipformer_subsampling_matches_functional_reference(
    batch_partitions: int,
) -> None:
    module = make_subsampling(batch_partitions=batch_partitions)
    lengths = torch.tensor((23, 20, 11, 9), dtype=torch.int32)
    features = make_random_tensor((4, 23, 16), seed=2)
    for index, length in enumerate(lengths.tolist()):
        features[index, length:] = ZERO_LOG

    with torch.inference_mode():
        expected = reference_subsampling(module, features, lengths)
        actual = module(features, lengths)

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
    ids=("fp32", "fp16", "bf16"),
)
@pytest.mark.parametrize("batch_size", (4, 5, 7))
def test_partitioned_zipformer_subsampling_handles_remainder_batch(
    batch_size: int, dtype: torch.dtype
) -> None:
    expected_module = make_subsampling(dtype=dtype)
    partitioned_module = make_subsampling(batch_partitions=3, dtype=dtype)
    partitioned_module.load_state_dict(expected_module.state_dict())
    features = make_random_tensor((batch_size, 23, 16), seed=3, dtype=dtype)
    lengths = torch.linspace(9, 23, batch_size, dtype=torch.int32).flip(0)
    for index, length in enumerate(lengths.tolist()):
        features[index, length:] = ZERO_LOG

    with torch.inference_mode():
        expected = expected_module(features, lengths)
        actual = partitioned_module(features, lengths)

    atol, rtol = DTYPE_TOLERANCES[dtype]
    assert actual[0].dtype == dtype
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_zipformer_subsampling_length_boundaries() -> None:
    module = make_subsampling()
    lengths = torch.tensor((-100, 8, 9, 10, 11, 12, 22, 23, 24, 100), dtype=torch.int32)

    output, output_lengths = module(
        torch.full((len(lengths), 23, 16), ZERO_LOG), lengths
    )

    assert output.shape == (len(lengths), 8, 12)
    assert output_lengths.tolist() == [0, 0, 1, 1, 2, 2, 7, 8, 8, 8]


def test_partitioned_zipformer_subsampling_onnx_matches_dynamic_time(
    tmp_path: Path,
) -> None:
    module = make_subsampling(batch_partitions=3)
    num_frames = torch.export.Dim("num_frames", min=9, max=41)
    onnx_path = tmp_path / "zipformer_subsampling.onnx"
    example_features = make_random_tensor((4, 23, 16), seed=5)
    example_lengths = torch.tensor((23, 21, 17, 9), dtype=torch.int32)
    for index, length in enumerate(example_lengths.tolist()):
        example_features[index, length:] = ZERO_LOG

    torch.onnx.export(
        module,
        (example_features, example_lengths),
        onnx_path,
        input_names=("features", "feature_lengths"),
        output_names=("output", "output_lengths"),
        dynamic_shapes=({1: num_frames}, {}),
        opset_version=20,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)

    for current_num_frames in (9, 10, 11, 24, 31, 41):
        lengths = torch.tensor(
            (
                current_num_frames,
                max(9, current_num_frames - 1),
                min(current_num_frames, 11),
                9,
            ),
            dtype=torch.int32,
        )
        features = make_random_tensor(
            (4, current_num_frames, 16), seed=100 + current_num_frames
        )
        for index, length in enumerate(lengths.tolist()):
            features[index, length:] = ZERO_LOG

        with torch.inference_mode():
            expected_output, expected_lengths = module(features, lengths)
        actual_output, actual_lengths = evaluator.run(
            None,
            {
                "features": features.numpy(),
                "feature_lengths": lengths.numpy(),
            },
        )

        np.testing.assert_allclose(
            actual_output, expected_output.numpy(), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())


@pytest.mark.parametrize("batch_partitions", (1, 3))
def test_zipformer_subsampling_valid_prefix_is_invariant_to_right_padding(
    batch_partitions: int,
) -> None:
    """Keep a short utterance independent of batch neighbors and padding extent."""

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

    assert compact_output_lengths[0] == extended_output_lengths[0]
    valid_length = compact_output_lengths[0]
    torch.testing.assert_close(
        compact[0, :valid_length],
        extended[0, :valid_length],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
    ids=("fp32", "fp16", "bf16"),
)
def test_bias_norm_preserves_configured_dtype(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    module = BiasNorm(12).to(dtype)
    module.scale.fill_(1.3)
    module.bias.copy_(make_random_tensor((12,), seed=10, dtype=dtype))
    inputs = make_random_tensor((3, 17, 12), seed=11, dtype=dtype)
    expected = inputs.float() * module.scale.float()
    rms = torch.mean(
        (inputs.float() - module.bias.float()) ** 2, dim=2, keepdim=True
    ).sqrt()
    expected = (expected / rms.clamp_min(torch.finfo(torch.float32).tiny)).to(dtype)

    actual = module(inputs)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_bias_norm_uses_stable_float32_accumulation(dtype: torch.dtype) -> None:
    module = BiasNorm(12).to(dtype)
    module.scale.fill_(1.0)
    module.bias.zero_()

    large_output = module(torch.full((2, 3, 12), 256.0, dtype=dtype))
    zero_output = module(torch.zeros(2, 3, 12, dtype=dtype))

    torch.testing.assert_close(
        large_output, torch.ones_like(large_output), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        zero_output, torch.zeros_like(zero_output), atol=0.0, rtol=0.0
    )
