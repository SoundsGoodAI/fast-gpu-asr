#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the Zipformer predictor cache and exported joiner module."""

from itertools import product
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from fast_gpu_asr.constants import ONNX_OPSET_VERSION
from fast_gpu_asr.export.model.zipformer.decoder import Decoder, Joiner

DEFAULT_DIMENSIONS = (5, 8, 6)
ALTERNATE_DIMENSIONS = (7, 6, 9)


def make_decoder(
    context_size: int = 2,
    dtype: torch.dtype = torch.float32,
    dimensions: tuple[int, int, int] = DEFAULT_DIMENSIONS,
) -> Decoder:
    """Build a seeded predictor from vocabulary, decoder, and joiner dimensions.

    Parameters
    ----------
    context_size : int
        Number of preceding tokens consumed by the predictor.
    dtype : torch.dtype
        Predictor parameter precision.
    dimensions : tuple[int, int, int]
        Vocabulary size, decoder width, and joiner width, respectively.

    Returns
    -------
    Decoder
        CPU predictor in evaluation mode with unchanged global RNG state.
    """

    vocab_size, decoder_dim, joiner_dim = dimensions
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(41)
        decoder = Decoder(vocab_size, decoder_dim, joiner_dim, context_size, dtype)
    return decoder.eval()


def make_joiner(
    dtype: torch.dtype = torch.float32, vocab_size: int = 5, joiner_dim: int = 6
) -> Joiner:
    """Build a seeded joiner without changing the caller's RNG state.

    Parameters
    ----------
    dtype : torch.dtype
        Joiner parameter precision.
    vocab_size : int
        Number of output token classes.
    joiner_dim : int
        Width of the projected encoder and decoder inputs.

    Returns
    -------
    Joiner
        Deterministic CPU joiner in evaluation mode.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(43)
        joiner = Joiner(joiner_dim, vocab_size, dtype)
    return joiner.eval()


@pytest.mark.parametrize("context_size", (1, 2))
@pytest.mark.parametrize("dimensions", (DEFAULT_DIMENSIONS, ALTERNATE_DIMENSIONS))
def test_predictor_matches_reference(
    context_size: int, dimensions: tuple[int, int, int]
) -> None:
    vocab_size, decoder_dim, _ = dimensions
    decoder = make_decoder(context_size, dimensions=dimensions)
    context_rows = [
        [-1] * context_size,
        list(range(context_size)),
        list(range(context_size, 0, -1)),
        [vocab_size - 1] * context_size,
    ]
    if context_size > 1:
        context_rows.append([-1, *range(context_size - 1)])
    contexts = torch.tensor(context_rows, dtype=torch.int32)
    embeddings = torch.nn.functional.embedding(
        contexts.clamp_min(0), decoder.embedding.weight
    ) * (contexts.unsqueeze(2) >= 0)
    if context_size > 1:
        features = torch.nn.functional.conv1d(
            embeddings.permute(0, 2, 1), decoder.conv.weight, groups=decoder_dim // 4
        )[:, :, 0].relu()
    else:
        features = embeddings[:, 0]
    expected = features @ decoder.decoder_proj.weight.T + decoder.decoder_proj.bias
    snapshot = contexts.clone()

    actual = decoder(contexts)

    assert torch.equal(contexts, snapshot)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("context_size", (1, 2))
@pytest.mark.parametrize("dimensions", (DEFAULT_DIMENSIONS, ALTERNATE_DIMENSIONS))
def test_predictor_checkpoint_layout(
    context_size: int, dimensions: tuple[int, int, int]
) -> None:
    vocab_size, decoder_dim, joiner_dim = dimensions
    decoder = make_decoder(context_size, dimensions=dimensions)
    expected_shapes = {
        "embedding.weight": (vocab_size, decoder_dim),
        "decoder_proj.weight": (joiner_dim, decoder_dim),
        "decoder_proj.bias": (joiner_dim,),
    }
    if context_size > 1:
        groups = decoder_dim // 4
        assert decoder.conv.groups == groups
        expected_shapes["conv.weight"] = (decoder_dim, decoder_dim // groups, 2)
    else:
        assert isinstance(decoder.conv, torch.nn.Identity)

    assert {
        name: tuple(tensor.shape) for name, tensor in decoder.state_dict().items()
    } == expected_shapes


def assert_context_lookup(decoder: Decoder, chunk_size: int) -> None:
    """Check cache row order, bounded batching, precision, and unchanged weights.

    Parameters
    ----------
    decoder : Decoder
        Small predictor whose complete context vocabulary fits in memory.
    chunk_size : int
        Maximum number of contexts permitted in each predictor call.

    Notes
    -----
    Expected values are computed with reversed context enumeration, then
    indexed using the runtime's radix formula to detect cache-order mistakes.
    The temporary forward hook is removed before returning.
    """

    contexts = torch.tensor(
        list(product(range(-1, decoder.vocab_size), repeat=decoder.context_size)),
        dtype=torch.int32,
    ).flip(0)
    indexes = torch.zeros(len(contexts), dtype=torch.int64)
    for position in range(decoder.context_size):
        indexes = indexes * (decoder.vocab_size + 1) + contexts[:, position] + 1
    snapshot = {name: value.clone() for name, value in decoder.state_dict().items()}
    expected = decoder(contexts)
    batch_sizes = []

    with decoder.register_forward_pre_hook(
        lambda _module, inputs: batch_sizes.append(len(inputs[0]))
    ):
        lookup = decoder.make_context_lookup(chunk_size)

    assert sum(batch_sizes) == len(contexts)
    assert all(0 < size <= chunk_size for size in batch_sizes)
    assert lookup.shape == (len(contexts), decoder.decoder_proj.out_features)
    assert lookup.device == decoder.embedding.weight.device
    assert lookup.is_contiguous()
    assert not lookup.requires_grad
    assert torch.isfinite(lookup).all()
    torch.testing.assert_close(lookup[indexes], expected)
    torch.testing.assert_close(decoder.state_dict(), snapshot, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("chunk_size", (1, 7, 36, 100))
@pytest.mark.parametrize("context_size", (1, 2))
def test_context_lookup_matches_runtime_index(
    context_size: int, chunk_size: int, dtype: torch.dtype
) -> None:
    decoder = make_decoder(context_size, dtype)

    assert {parameter.dtype for parameter in decoder.parameters()} == {dtype}
    assert_context_lookup(decoder, chunk_size)


def test_context_lookup_honors_nondefault_dimensions() -> None:
    assert_context_lookup(make_decoder(dimensions=ALTERNATE_DIMENSIONS), 7)


@pytest.mark.parametrize("chunk_size", (0, -1, 1.0))
def test_context_lookup_rejects_invalid_chunk_size(chunk_size: int | float) -> None:
    decoder = make_decoder()

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        decoder.make_context_lookup(chunk_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("dimensions", (DEFAULT_DIMENSIONS, ALTERNATE_DIMENSIONS))
def test_joiner_matches_log_probability_reference(
    dtype: torch.dtype, dimensions: tuple[int, int, int]
) -> None:
    vocab_size, _, joiner_dim = dimensions
    joiner = make_joiner(dtype, vocab_size, joiner_dim)
    generator = torch.Generator().manual_seed(44)
    inputs = tuple(
        torch.randn(3, joiner_dim, dtype=dtype, generator=generator) for _ in range(2)
    )
    logits = torch.nn.functional.linear(
        torch.tanh(inputs[0] + inputs[1]),
        joiner.output_proj.weight,
        joiner.output_proj.bias,
    )
    expected = torch.log_softmax(logits.float(), dim=1)
    snapshots = tuple(value.clone() for value in inputs)

    actual = joiner(*inputs)

    torch.testing.assert_close(inputs, snapshots, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dimensions", (DEFAULT_DIMENSIONS, ALTERNATE_DIMENSIONS))
def test_joiner_checkpoint_layout(dimensions: tuple[int, int, int]) -> None:
    vocab_size, _, joiner_dim = dimensions
    joiner = make_joiner(vocab_size=vocab_size, joiner_dim=joiner_dim)

    assert {
        name: tuple(tensor.shape) for name, tensor in joiner.state_dict().items()
    } == {
        "output_proj.weight": (vocab_size, joiner_dim),
        "output_proj.bias": (vocab_size,),
    }


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_joiner_normalizes_low_precision_logits_in_float32(dtype: torch.dtype) -> None:
    joiner = make_joiner(dtype)
    with torch.no_grad():
        joiner.output_proj.weight.zero_()
        joiner.output_proj.bias.copy_(
            torch.tensor((-4.0, -2.0, -1.0, -0.5, 0.0), dtype=dtype)
        )
    expected = torch.log_softmax(joiner.output_proj.bias.float(), dim=0)
    low_precision = torch.log_softmax(joiner.output_proj.bias, dim=0).float()
    assert not torch.allclose(low_precision, expected)

    actual = joiner(torch.zeros(2, 6, dtype=dtype), torch.zeros(2, 6, dtype=dtype))

    torch.testing.assert_close(actual, expected.expand(2, 5))


@pytest.mark.parametrize(
    "dtype,onnx_dtype,rtol,atol",
    [
        (torch.float32, onnx.TensorProto.FLOAT, 1e-5, 1e-6),
        (torch.float16, onnx.TensorProto.FLOAT16, 5e-4, 5e-4),
        (torch.bfloat16, onnx.TensorProto.BFLOAT16, 2e-3, 2e-3),
    ],
    ids=("fp32", "fp16", "bf16"),
)
def test_joiner_onnx_contract(
    tmp_path: Path, dtype: torch.dtype, onnx_dtype: int, rtol: float, atol: float
) -> None:
    joiner = make_joiner(dtype)
    decoder_input = torch.linspace(-1.5, 1.5, 24, dtype=dtype).reshape(4, 6)
    encoder_output = torch.linspace(0.75, -0.5, 24, dtype=dtype).reshape(4, 6)
    onnx_path = tmp_path / "zipformer_joiner.onnx"

    with torch.inference_mode():
        expected = joiner(decoder_input, encoder_output).numpy()
        torch.onnx.export(
            joiner,
            (decoder_input, encoder_output),
            onnx_path,
            input_names=("decoder_input", "encoder_output"),
            output_names=("tokens_log_prob",),
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.checker.check_model(model)
    assert {opset.domain: opset.version for opset in model.opset_import}[""] == (
        ONNX_OPSET_VERSION
    )
    inputs = {value.name: value.type.tensor_type for value in model.graph.input}
    outputs = {value.name: value.type.tensor_type for value in model.graph.output}
    assert tuple(inputs) == ("decoder_input", "encoder_output")
    assert tuple(outputs) == ("tokens_log_prob",)
    assert {
        name: (tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim))
        for name, tensor in (inputs | outputs).items()
    } == {
        "decoder_input": (onnx_dtype, (4, 6)),
        "encoder_output": (onnx_dtype, (4, 6)),
        "tokens_log_prob": (onnx.TensorProto.FLOAT, (4, 5)),
    }
    floating_dtypes = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }
    assert {
        initializer.data_type
        for initializer in model.graph.initializer
        if initializer.data_type in floating_dtypes
    } == {onnx_dtype}

    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    (log_softmax,) = [
        node for node in model.graph.node if "tokens_log_prob" in node.output
    ]
    assert log_softmax.op_type == "LogSoftmax"
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in log_softmax.attribute
    } == {"axis": 1}
    assert value_types[log_softmax.input[0]] == onnx.TensorProto.FLOAT

    numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_dtype)
    (actual,) = ReferenceEvaluator(model).run(
        None,
        {
            "decoder_input": decoder_input.float().numpy().astype(numpy_dtype),
            "encoder_output": encoder_output.float().numpy().astype(numpy_dtype),
        },
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
