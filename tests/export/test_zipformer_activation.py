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


def calculate_swoosh_reference(
    values: torch.Tensor, shift: float, offset: float
) -> torch.Tensor:
    """Evaluate Swoosh with a stable formula independent of the production code.

    Parameters
    ----------
    values : torch.Tensor
        Floating-point inputs; FP64 is used for the high-precision oracle.
    shift : float
        Input shift applied before softplus.
    offset : float
        Constant subtracted from the final activation.

    Returns
    -------
    torch.Tensor
        Reference activations with the same shape and dtype as ``values``.
    """

    shifted = values - shift
    softplus = torch.maximum(shifted, torch.zeros_like(shifted)) + torch.log1p(
        torch.exp(-torch.abs(shifted))
    )
    return softplus - 0.08 * values - offset


def tensor_to_numpy(tensor: torch.Tensor) -> np.typing.NDArray:
    """Convert a CPU tensor to NumPy while preserving BF16 storage.

    Parameters
    ----------
    tensor : torch.Tensor
        CPU tensor without gradient tracking.

    Returns
    -------
    np.typing.NDArray
        Same-shape view with the corresponding ONNX-compatible NumPy dtype.
        BF16 values are reinterpreted bitwise, not numerically cast.
    """

    if tensor.dtype == torch.bfloat16:
        numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16)
        return tensor.view(torch.uint16).numpy().view(numpy_dtype)
    return tensor.numpy()


@pytest.mark.parametrize("activation_type", (SwooshL, SwooshR))
def test_swoosh_is_checkpoint_stateless(activation_type: type[torch.nn.Module]) -> None:
    activation = activation_type()

    assert tuple(activation.named_parameters()) == ()
    assert tuple(activation.named_buffers()) == ()
    assert activation.state_dict() == {}


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
def test_swoosh_matches_independent_closed_form(
    activation_type: type[torch.nn.Module], shift: float, offset: float
) -> None:
    values = torch.tensor(
        (
            -100.0,
            -10.0,
            -4.0,
            -1.0,
            0.0,
            1.0,
            shift - 1e-6,
            shift,
            shift + 1e-6,
            4.0,
            10.0,
            100.0,
        ),
        dtype=torch.float64,
    )
    expected = calculate_swoosh_reference(values, shift, offset)

    actual = activation_type()(values)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
@pytest.mark.parametrize(
    ("dtype", "atol"),
    ((torch.float32, 2e-6), (torch.float16, 5e-4), (torch.bfloat16, 4e-3)),
    ids=("fp32", "fp16", "bf16"),
)
def test_swoosh_supported_dtypes_match_float64_oracle(
    activation_type: type[torch.nn.Module],
    shift: float,
    offset: float,
    dtype: torch.dtype,
    atol: float,
) -> None:
    values = torch.linspace(-20.0, 20.0, 255, dtype=dtype).reshape(3, 5, 17)
    values = values.transpose(1, 2)
    original = values.clone()
    expected = calculate_swoosh_reference(values.to(torch.float64), shift, offset).to(
        dtype
    )

    actual = activation_type()(values)

    assert not values.is_contiguous()
    assert torch.equal(values, original)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=torch.finfo(dtype).eps, atol=atol)


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
@pytest.mark.parametrize(
    ("dtype", "onnx_dtype", "rtol", "atol"),
    (
        (torch.float32, onnx.TensorProto.FLOAT, 1e-6, 2e-6),
        (torch.float16, onnx.TensorProto.FLOAT16, 2e-3, 2e-3),
        (torch.bfloat16, onnx.TensorProto.BFLOAT16, 1e-2, 4e-3),
    ),
    ids=("fp32", "fp16", "bf16"),
)
def test_swoosh_onnx_contract(
    activation_type: type[torch.nn.Module],
    shift: float,
    offset: float,
    dtype: torch.dtype,
    onnx_dtype: int,
    rtol: float,
    atol: float,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "activation.onnx"
    example = torch.zeros(2, 3, 4, dtype=dtype)
    torch.onnx.export(
        activation_type().eval(),
        (example,),
        onnx_path,
        input_names=("input",),
        output_names=("output",),
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch_size", min=1, max=4),
                1: torch.export.Dim("num_rows", min=1, max=5),
                2: torch.export.Dim("num_columns", min=1, max=6),
            }
        },
        opset_version=ONNX_OPSET_VERSION,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    assert [opset.version for opset in model.opset_import if opset.domain == ""] == [
        ONNX_OPSET_VERSION
    ]
    assert [value.name for value in model.graph.input] == ["input"]
    assert [value.name for value in model.graph.output] == ["output"]
    for value in (model.graph.input[0], model.graph.output[0]):
        tensor_type = value.type.tensor_type
        assert tensor_type.elem_type == onnx_dtype
        assert tuple(dim.dim_param for dim in tensor_type.shape.dim) == (
            "batch_size",
            "num_rows",
            "num_columns",
        )
    assert sum(node.op_type == "Softplus" for node in model.graph.node) == 1
    assert all(node.domain == "" for node in model.graph.node)

    evaluator = ReferenceEvaluator(model)

    for shape in ((1, 2, 3), (3, 4, 5)):
        num_values = shape[0] * shape[1] * shape[2]
        values = torch.linspace(-8.0, 8.0, num_values, dtype=dtype).reshape(shape)
        input_values = tensor_to_numpy(values)
        (actual,) = evaluator.run(None, {"input": input_values})
        expected = calculate_swoosh_reference(
            values.to(torch.float64), shift, offset
        ).to(dtype)
        assert actual.shape == shape
        assert actual.dtype == input_values.dtype
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.float().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"shape={shape}",
        )


@pytest.mark.parametrize(("activation_type", "shift"), ((SwooshL, 4.0), (SwooshR, 1.0)))
def test_swoosh_gradient_matches_closed_form(
    activation_type: type[torch.nn.Module], shift: float
) -> None:
    values = torch.linspace(-8.0, 8.0, 17, dtype=torch.float64, requires_grad=True)

    activation_type()(values).sum().backward()

    assert values.grad is not None
    expected = torch.sigmoid(values.detach() - shift) - 0.08
    torch.testing.assert_close(values.grad, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
@pytest.mark.parametrize("shape", ((), (2, 0, 3)), ids=("scalar", "empty"))
def test_swoosh_accepts_scalar_and_empty_inputs(
    activation_type: type[torch.nn.Module],
    shift: float,
    offset: float,
    shape: tuple[int, ...],
) -> None:
    values = torch.full(shape, 0.25, dtype=torch.float32)
    expected = calculate_swoosh_reference(values.to(torch.float64), shift, offset).to(
        values.dtype
    )

    actual = activation_type()(values)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("activation_type", "shift", "offset"), SWOOSH_CASES)
@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
    ids=("fp32", "fp16", "bf16"),
)
def test_swoosh_large_finite_inputs_match_asymptotes(
    activation_type: type[torch.nn.Module],
    shift: float,
    offset: float,
    dtype: torch.dtype,
) -> None:
    limit = torch.finfo(dtype).max
    values = torch.tensor([-limit, limit], dtype=dtype)
    expected = calculate_swoosh_reference(values.to(torch.float64), shift, offset).to(
        dtype
    )

    actual = activation_type()(values)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=torch.finfo(dtype).eps, atol=0.0)
