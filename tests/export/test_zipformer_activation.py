#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the Zipformer Swoosh activation functions."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from fast_gpu_asr.constants import ONNX_OPSET_VERSION
from fast_gpu_asr.export.model.zipformer.activation import SwooshL, SwooshR

SWOOSH_CASES = ((SwooshL, 4.0, 0.035), (SwooshR, 1.0, 0.313261687))


@pytest.mark.parametrize(
    ("activation_type", "shift", "offset"),
    SWOOSH_CASES,
)
def test_swoosh_matches_independent_closed_form(
    activation_type: type[torch.nn.Module], shift: float, offset: float
) -> None:
    values = torch.tensor(
        [-100.0, -10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0, 100.0],
        dtype=torch.float64,
    )
    shifted = values - shift
    softplus = torch.maximum(shifted, torch.zeros_like(shifted)) + torch.log1p(
        torch.exp(-torch.abs(shifted))
    )
    expected = softplus - 0.08 * values - offset

    actual = activation_type()(values)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    ((torch.float32, 2e-6), (torch.float16, 5e-4), (torch.bfloat16, 4e-3)),
)
def test_swoosh_supported_dtypes_match_float64_oracle(
    activation_type: type[torch.nn.Module],
    shift: float,
    offset: float,
    dtype: torch.dtype,
    atol: float,
) -> None:
    values = torch.linspace(-20.0, 20.0, 257, dtype=dtype)
    values64 = values.to(torch.float64)
    shifted = values64 - shift
    expected = (
        torch.maximum(shifted, torch.zeros_like(shifted))
        + torch.log1p(torch.exp(-torch.abs(shifted)))
        - 0.08 * values64
        - offset
    ).to(dtype)

    actual = activation_type()(values)

    torch.testing.assert_close(actual, expected, rtol=torch.finfo(dtype).eps, atol=atol)


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
@pytest.mark.parametrize(
    ("dtype", "onnx_dtype"),
    (
        (torch.float32, onnx.TensorProto.FLOAT),
        (torch.float16, onnx.TensorProto.FLOAT16),
        (torch.bfloat16, onnx.TensorProto.BFLOAT16),
    ),
)
def test_swoosh_onnx_contract(
    activation_type: type[torch.nn.Module],
    dtype: torch.dtype,
    onnx_dtype: int,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / f"{activation_type.__name__}_{dtype}.onnx"
    example = torch.randn(2, 3, 4, dtype=dtype)
    batch_size = torch.export.Dim("batch_size", min=1, max=4)
    num_rows = torch.export.Dim("num_rows", min=1, max=5)
    num_columns = torch.export.Dim("num_columns", min=1, max=6)
    torch.onnx.export(
        activation_type().eval(),
        (example,),
        onnx_path,
        input_names=("input",),
        output_names=("output",),
        dynamic_shapes={
            "x": {
                0: batch_size,
                1: num_rows,
                2: num_columns,
            }
        },
        opset_version=ONNX_OPSET_VERSION,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    input_type = model.graph.input[0].type.tensor_type
    output_type = model.graph.output[0].type.tensor_type
    input_shape = tuple(
        dimension.dim_param or dimension.dim_value for dimension in input_type.shape.dim
    )
    output_shape = tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in output_type.shape.dim
    )

    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"
    assert input_type.elem_type == onnx_dtype
    assert output_type.elem_type == onnx_dtype
    assert input_shape == ("batch_size", "num_rows", "num_columns")
    assert output_shape == input_shape
    assert sum(node.op_type == "Softplus" for node in model.graph.node) == 1
    assert all(node.domain == "" for node in model.graph.node)


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float32, 1e-6, 2e-6), (torch.float16, 2e-3, 2e-3)),
)
def test_swoosh_onnx_reference_matches_eager_for_dynamic_shape(
    activation_type: type[torch.nn.Module],
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / f"{activation_type.__name__}.onnx"
    example = torch.randn(2, 3, 4, dtype=dtype)
    torch.onnx.export(
        activation_type().eval(),
        (example,),
        onnx_path,
        input_names=("input",),
        output_names=("output",),
        dynamic_shapes={
            "x": {
                0: torch.export.Dim.DYNAMIC,
                1: torch.export.Dim.DYNAMIC,
                2: torch.export.Dim.DYNAMIC,
            }
        },
        opset_version=ONNX_OPSET_VERSION,
    )
    model = onnx.load(onnx_path)
    evaluator = ReferenceEvaluator(model)

    for shape in ((1, 2, 3), (3, 4, 5)):
        num_values = shape[0] * shape[1] * shape[2]
        values = torch.linspace(-8.0, 8.0, num_values, dtype=dtype).reshape(shape)
        (actual,) = evaluator.run(None, {"input": values.numpy()})
        expected = activation_type()(values)

        assert actual.dtype == values.numpy().dtype
        np.testing.assert_allclose(actual, expected.numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_swoosh_preserves_shape_and_dtype(
    activation_type: type[torch.nn.Module], dtype: torch.dtype
) -> None:
    values = torch.linspace(-8.0, 8.0, 48, dtype=dtype).reshape(2, 3, 8)
    values = values.permute(0, 2, 1)
    expected = activation_type()(values.contiguous())

    actual = activation_type()(values)

    assert not values.is_contiguous()
    assert actual.shape == values.shape
    assert actual.dtype == dtype
    assert actual.device == values.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("activation_type", "shift"),
    tuple((activation_type, shift) for activation_type, shift, _ in SWOOSH_CASES),
)
def test_swoosh_gradient_matches_closed_form(
    activation_type: type[torch.nn.Module], shift: float
) -> None:
    values = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64, requires_grad=True)

    activation_type()(values).sum().backward()

    assert values.grad is not None
    expected = torch.sigmoid(values.detach() - shift) - 0.08
    torch.testing.assert_close(values.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
def test_swoosh_accepts_empty_input(activation_type: type[torch.nn.Module]) -> None:
    values = torch.empty(2, 0, 3)

    actual = activation_type()(values)

    assert actual.shape == values.shape
    assert actual.dtype == values.dtype
    assert actual.device == values.device
    assert actual.numel() == 0


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_swoosh_large_finite_inputs_remain_finite(
    activation_type: type[torch.nn.Module], dtype: torch.dtype
) -> None:
    limit = torch.finfo(dtype).max
    values = torch.tensor([-limit, limit], dtype=dtype)

    actual = activation_type()(values)

    assert torch.isfinite(actual).all()
    assert torch.all(actual > 0.0)
    assert actual[1] > actual[0]
