#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Numerical and ONNX tests for Zipformer temporal resampling modules."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import sympy
import torch

from fast_gpu_asr.constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.zipformer.zipformer import (
    SimpleDownsample,
    SimpleUpsample,
)

EAGER_DTYPE_CASES = (
    pytest.param(torch.float32, 1e-6, id="fp32"),
    pytest.param(torch.float16, 3e-3, id="fp16"),
    pytest.param(torch.bfloat16, 3e-2, id="bf16"),
)
ONNX_DTYPES = {
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}
PLUGIN_OP_TYPES = (
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
RESAMPLING_CASES = tuple(
    pytest.param(
        factor, sequence_length, id=f"factor-{factor}-frames-{sequence_length}"
    )
    for factor, sequence_lengths in (
        (2, (1, 2, 3, 31)),
        (3, (1, 2, 3, 4, 31)),
        (4, (1, 3, 4, 5, 31)),
        (8, (1, 7, 8, 9, 31)),
    )
    for sequence_length in sequence_lengths
)


class Resampling(torch.nn.Module):
    """Compose resampling modules for ONNX contract tests."""

    def __init__(self, factor: int) -> None:
        """Initialize deterministic aggregation and bypass weights.

        Parameters
        ----------
        factor : int
            Positive temporal downsampling and upsampling factor.
        """

        super().__init__()
        self.downsample = SimpleDownsample(factor)
        self.downsample.weights.fill_(1.0 / factor)
        self.upsample = SimpleUpsample(factor)
        self.register_buffer("bypass_scale", torch.linspace(0.0, 1.0, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``x``, restore its time axis, and apply the bypass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, frames, 8)`` matching the module dtype.

        Returns
        -------
        torch.Tensor
            Blended features with the same shape and dtype as ``x``.
        """

        return self.upsample(x, self.downsample(x), self.bypass_scale)


def make_random_tensor(
    shape: tuple[int, ...], seed: int, dtype: torch.dtype
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


def test_resampling_modules_expose_converted_checkpoint_state_layout() -> None:
    downsample = SimpleDownsample(4)
    upsample = SimpleUpsample(4)
    converted_weights = torch.softmax(torch.arange(4, dtype=torch.float32), dim=0)
    converted_weights = converted_weights.unsqueeze(1)

    assert list(downsample.parameters()) == []
    assert [name for name, _ in downsample.named_buffers()] == ["weights"]
    assert not downsample.weights.requires_grad
    downsample.load_state_dict({"weights": converted_weights})
    torch.testing.assert_close(downsample.weights, converted_weights, atol=0, rtol=0)
    assert list(upsample.parameters()) == []
    assert list(upsample.buffers()) == []


@pytest.mark.parametrize(("dtype", "tolerance"), EAGER_DTYPE_CASES)
def test_factor_one_downsample_is_identity_and_upsample_blends(
    dtype: torch.dtype, tolerance: float
) -> None:
    x_early = make_random_tensor((2, 7, 5), 1, dtype)
    x_later = make_random_tensor((2, 7, 5), 2, dtype)
    bypass_scale = torch.linspace(0.0, 1.0, 5).to(dtype)
    inputs = (x_early, x_later, bypass_scale)
    original_inputs = tuple(value.clone() for value in inputs)
    expected = (
        x_early.float() + (x_later.float() - x_early.float()) * bypass_scale.float()
    ).to(dtype)

    downsampled = SimpleDownsample(1).to(dtype)(x_early)
    actual = SimpleUpsample(1)(*inputs)

    torch.testing.assert_close(inputs, original_inputs, atol=0, rtol=0)
    torch.testing.assert_close(downsampled, x_early, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(("dtype", "tolerance"), EAGER_DTYPE_CASES)
@pytest.mark.parametrize(("factor", "sequence_length"), RESAMPLING_CASES)
def test_resampling_modules_match_reference(
    dtype: torch.dtype, tolerance: float, factor: int, sequence_length: int
) -> None:
    seed = factor * 100 + sequence_length
    x_early = make_random_tensor((3, sequence_length, 7), seed, dtype)
    weights = torch.softmax(
        make_random_tensor((factor, 1), seed + 1, torch.float32), dim=0
    ).to(dtype)
    output_length = (sequence_length + factor - 1) // factor
    x_later = make_random_tensor((3, output_length, 7), seed + 2, dtype)
    bypass_scale = torch.linspace(0.0, 1.0, 7).to(dtype)
    downsample = SimpleDownsample(factor).to(dtype)
    downsample.weights.copy_(weights)
    inputs = (x_early, x_later, bypass_scale)
    original_inputs = tuple(value.clone() for value in inputs)

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

    actual_downsample = downsample(x_early)
    actual_output = SimpleUpsample(factor)(x_early, x_later, bypass_scale)

    torch.testing.assert_close(inputs, original_inputs, atol=0, rtol=0)
    torch.testing.assert_close(downsample.weights, weights, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_downsample, expected_downsample, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_output, expected_output, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    ("factor", "dtype"),
    (
        pytest.param(1, torch.float32, id="factor-1-fp32"),
        pytest.param(3, torch.float32, id="factor-3-fp32"),
        pytest.param(4, torch.float32, id="factor-4-fp32"),
        pytest.param(4, torch.float16, id="factor-4-fp16"),
        pytest.param(4, torch.bfloat16, id="factor-4-bf16"),
    ),
)
def test_resampling_exports_dynamic_tensorrt_plugin_contract(
    tmp_path: Path, factor: int, dtype: torch.dtype
) -> None:
    onnx_path = tmp_path / "resampling.onnx"
    num_frames = torch.export.Dim("num_frames", min=1, max=65)
    model = Resampling(factor).eval().to(dtype)
    example = make_random_tensor((2, 17, 8), 3, dtype)
    torch.onnx.export(
        model,
        (example,),
        onnx_path,
        input_names=("x",),
        output_names=("output",),
        dynamic_shapes=({1: num_frames},),
        opset_version=ONNX_OPSET_VERSION,
    )

    onnx_model = onnx.load(onnx_path)
    assert [
        opset.version for opset in onnx_model.opset_import if opset.domain == ""
    ] == [ONNX_OPSET_VERSION]
    graph = onnx_model.graph
    expected_plugins = PLUGIN_OP_TYPES if factor > 1 else PLUGIN_OP_TYPES[1:]
    assert [(node.domain, node.op_type) for node in graph.node] == [
        ("", plugin) for plugin in expected_plugins
    ]
    upsample_node = graph.node[-1]
    assert [value.name for value in graph.input] == ["x"]
    assert [value.name for value in graph.output] == ["output"]
    assert len(upsample_node.input) == 3
    assert list(upsample_node.output) == ["output"]

    expected_dtype = ONNX_DTYPES[dtype]
    for value in (graph.input[0], graph.output[0]):
        assert value.type.tensor_type.elem_type == expected_dtype
        assert get_onnx_shape(value) == (2, "num_frames", 8)

    downsample_output = "x"
    expected_initializers = [(upsample_node.input[2], model.bypass_scale)]
    if factor > 1:
        downsample_node = graph.node[0]
        assert len(downsample_node.input) == 2
        assert downsample_node.input[0] == "x"
        assert len(downsample_node.output) == 1
        downsample_output = downsample_node.output[0]
        expected_initializers.append(
            (downsample_node.input[1], model.downsample.weights)
        )
        value_info = {value.name: value for value in graph.value_info}
        value = value_info[downsample_output]
        assert value.type.tensor_type.elem_type == expected_dtype
        shape = get_onnx_shape(value)
        assert shape[0::2] == (2, 8)
        assert isinstance(shape[1], str)
        frames = sympy.Symbol("num_frames", integer=True)
        expression = sympy.sympify(shape[1], locals={"num_frames": frames})
        assert sympy.simplify(expression - ((frames + factor - 1) // factor)) == 0
    assert list(upsample_node.input[:2]) == ["x", downsample_output]

    initializers = {value.name: value for value in graph.initializer}
    assert set(initializers) == {name for name, _ in expected_initializers}
    for name, tensor in expected_initializers:
        assert initializers[name].data_type == expected_dtype
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(initializers[name]).astype(np.float32),
            tensor.float().numpy(),
            strict=True,
        )

    for node in graph.node:
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes == {
            "factor": factor,
            "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        }

    # Check graph structure without requiring ONNX schemas for TensorRT plugins.
    for node in graph.node:
        node.domain = TENSORRT_PLUGIN_NAMESPACE
    onnx_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(onnx_model)
