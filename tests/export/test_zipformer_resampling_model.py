#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Numerical and ONNX tests for Zipformer temporal resampling modules."""

from pathlib import Path

import onnx
import pytest
import torch

from fast_gpu_asr.constants import (
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.zipformer.zipformer import (
    SimpleDownsample,
    SimpleUpsample,
)

DTYPE_TOLERANCES = {
    torch.float32: 1e-6,
    torch.float16: 3e-3,
    torch.bfloat16: 3e-2,
}
ONNX_DTYPES = {
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}
RESAMPLING_CASES = tuple(
    pytest.param(
        factor, sequence_length, id=f"factor-{factor}-frames-{sequence_length}"
    )
    for factor in (2, 4, 8)
    for sequence_length in (factor - 1, factor, factor + 1, 31)
)


class Resampling(torch.nn.Module):
    """Compose factor-four resampling modules for ONNX contract tests."""

    def __init__(self) -> None:
        """Initialize deterministic aggregation and bypass weights."""

        super().__init__()
        self.downsample = SimpleDownsample(4)
        self.downsample.weights.fill_(0.25)
        self.upsample = SimpleUpsample(4)
        self.register_buffer("bypass_scale", torch.linspace(0.0, 1.0, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``x``, restore its time axis, and apply the bypass."""

        x_later = self.downsample(x)
        return self.upsample(x, x_later, self.bypass_scale)


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype
) -> torch.Tensor:
    """Create deterministic random data without changing the global RNG state."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(dtype)


def get_onnx_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return fixed dimensions as integers and symbolic dimensions as strings."""

    return tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
    ids=("fp32", "fp16", "bf16"),
)
def test_factor_one_downsample_is_identity_and_upsample_blends(
    dtype: torch.dtype,
) -> None:
    """Preserve time while still combining distinct early and later values."""

    x_early = make_random_tensor((2, 7, 5), seed=1, dtype=dtype)
    x_later = make_random_tensor((2, 7, 5), seed=2, dtype=dtype)
    bypass_scale = torch.linspace(0.0, 1.0, 5).to(dtype)

    downsampled = SimpleDownsample(1).to(dtype)(x_early)
    actual = SimpleUpsample(1)(x_early, x_later, bypass_scale)
    expected = (
        x_early.float() + (x_later.float() - x_early.float()) * bypass_scale.float()
    ).to(dtype)

    assert downsampled.shape == x_early.shape
    assert downsampled.dtype == dtype
    assert actual.shape == x_early.shape
    assert actual.dtype == dtype
    tolerance = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(downsampled, x_early, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
    ids=("fp32", "fp16", "bf16"),
)
@pytest.mark.parametrize(("factor", "sequence_length"), RESAMPLING_CASES)
def test_resampling_modules_match_reference(
    dtype: torch.dtype, factor: int, sequence_length: int
) -> None:
    """Check every model factor, partial final groups, and supported dtype."""

    seed = factor * 100 + sequence_length
    x_early = make_random_tensor((3, sequence_length, 7), seed=seed, dtype=dtype)
    weights = torch.softmax(
        make_random_tensor((factor, 1), seed=seed + 1, dtype=torch.float32), dim=0
    ).to(dtype)
    output_length = (sequence_length + factor - 1) // factor
    x_later = make_random_tensor((3, output_length, 7), seed=seed + 2, dtype=dtype)
    bypass_scale = torch.linspace(0.0, 1.0, 7).to(dtype)

    downsample = SimpleDownsample(factor).to(dtype)
    downsample.weights.copy_(weights)
    with torch.inference_mode():
        actual_downsample = downsample(x_early)
        actual_output = SimpleUpsample(factor)(x_early, x_later, bypass_scale)

    expected_downsample = torch.zeros((3, output_length, 7), dtype=torch.float32)
    first_frames = torch.arange(output_length) * factor
    for offset in range(factor):
        frame_indexes = (first_frames + offset).clamp_max(sequence_length - 1)
        expected_downsample += (
            x_early[:, frame_indexes].float() * weights[offset].float()
        )
    expected_downsample = expected_downsample.to(dtype)

    later_indexes = torch.arange(sequence_length) // factor
    repeated_later = x_later[:, later_indexes]
    expected_output = (
        x_early.float()
        + (repeated_later.float() - x_early.float()) * bypass_scale.float()
    ).to(dtype)

    assert actual_downsample.shape == (3, output_length, 7)
    assert actual_downsample.dtype == dtype
    assert actual_output.shape == x_early.shape
    assert actual_output.dtype == dtype
    tolerance = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(
        actual_downsample, expected_downsample, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_output, expected_output, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
    ids=("fp32", "fp16", "bf16"),
)
def test_resampling_exports_dynamic_tensorrt_plugin_contract(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    """Retain dynamic time, plugin wiring, attributes, shapes, and dtypes in ONNX."""

    onnx_path = tmp_path / f"resampling_{str(dtype).removeprefix('torch.')}.onnx"
    num_frames = torch.export.Dim("num_frames", min=1, max=65)
    model = Resampling().eval().to(dtype)
    example = make_random_tensor((2, 17, 8), seed=3, dtype=dtype)
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        input_names=("x",),
        output_names=("output",),
        dynamic_shapes=({1: num_frames},),
        opset_version=20,
    )

    onnx_model = onnx.load(onnx_path)
    # TensorRT plugin nodes intentionally use the default ONNX domain and have no
    # ONNX schemas. Check an equivalent copy in a custom domain so the checker can
    # still validate the surrounding graph structure and tensor references.
    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(onnx_model)
    for node in checker_model.graph.node:
        node.domain = TENSORRT_PLUGIN_NAMESPACE
    checker_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(checker_model)
    graph = onnx_model.graph
    assert [(node.domain, node.op_type) for node in graph.node] == [
        ("", ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME),
        ("", ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME),
    ]
    downsample_node, upsample_node = graph.node

    initializers = {initializer.name: initializer for initializer in graph.initializer}
    assert downsample_node.input[0] == graph.input[0].name
    assert downsample_node.input[1] in initializers
    assert list(upsample_node.input[:2]) == [
        graph.input[0].name,
        downsample_node.output[0],
    ]
    assert upsample_node.input[2] in initializers
    assert graph.output[0].name == upsample_node.output[0]
    assert set(initializers) == {downsample_node.input[1], upsample_node.input[2]}

    expected_dtype = ONNX_DTYPES[dtype]
    assert initializers[downsample_node.input[1]].data_type == expected_dtype
    assert tuple(initializers[downsample_node.input[1]].dims) == (4, 1)
    assert initializers[upsample_node.input[2]].data_type == expected_dtype
    assert tuple(initializers[upsample_node.input[2]].dims) == (8,)

    value_info = {
        value.name: value for value in (*graph.input, *graph.value_info, *graph.output)
    }
    assert value_info[graph.input[0].name].type.tensor_type.elem_type == expected_dtype
    assert (
        value_info[downsample_node.output[0]].type.tensor_type.elem_type
        == expected_dtype
    )
    assert value_info[graph.output[0].name].type.tensor_type.elem_type == expected_dtype
    assert get_onnx_shape(graph.input[0]) == (2, "num_frames", 8)
    downsample_shape = get_onnx_shape(value_info[downsample_node.output[0]])
    assert downsample_shape[0::2] == (2, 8)
    assert str(downsample_shape[1]).replace(" ", "") == "((num_frames+3)//4)"
    assert get_onnx_shape(graph.output[0]) == (2, "num_frames", 8)

    for node in (downsample_node, upsample_node):
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes == {
            "factor": 4,
            "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        }
