#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the exported Zipformer predictor and joiner modules."""

from itertools import product
from pathlib import Path

import onnx
import pytest
import torch

from fast_gpu_asr.constants import ONNX_OPSET_VERSION
from fast_gpu_asr.export.model.zipformer.decoder import Decoder, Joiner

DECODER_DIM = 8
JOINER_DIM = 6
VOCAB_SIZE = 5


def make_decoder(context_size: int = 2, dtype: torch.dtype = torch.float32) -> Decoder:
    """Construct a deterministic compact predictor."""

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(41)
        decoder = Decoder(
            vocab_size=VOCAB_SIZE,
            decoder_dim=DECODER_DIM,
            joiner_dim=JOINER_DIM,
            context_size=context_size,
            dtype=dtype,
        )
    return decoder.eval()


def make_joiner(dtype: torch.dtype = torch.float32) -> Joiner:
    """Construct a deterministic compact joiner without changing global RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(43)
        joiner = Joiner(joiner_dim=JOINER_DIM, vocab_size=VOCAB_SIZE, dtype=dtype)
    return joiner.eval()


@pytest.mark.parametrize("context_size", (1, 2))
def test_zipformer_predictor_matches_hand_computed_reference(
    context_size: int,
) -> None:
    """Match the Icefall predictor calculation for supported context sizes."""

    decoder = make_decoder(context_size)
    context_rows = [
        [-1] * context_size,
        list(range(context_size)),
        list(range(context_size, 0, -1)),
    ]
    if context_size > 1:
        context_rows.append([-1, *range(context_size - 1)])
    contexts = torch.tensor(context_rows, dtype=torch.int32)
    embeddings = torch.nn.functional.embedding(
        contexts.clamp_min(0), decoder.embedding.weight
    )
    embeddings = embeddings * (contexts.unsqueeze(2) >= 0)
    if context_size > 1:
        embeddings = torch.nn.functional.conv1d(
            embeddings.permute(0, 2, 1),
            decoder.conv.weight,
            groups=decoder.conv.groups,
        ).permute(0, 2, 1)
        features = torch.relu(embeddings[:, 0])
    else:
        features = embeddings[:, 0]
    expected = torch.nn.functional.linear(
        features, decoder.decoder_proj.weight, decoder.decoder_proj.bias
    )

    actual = decoder(contexts)

    torch.testing.assert_close(actual, expected)


def test_zipformer_bigram_predictor_matches_icefall_state_dict() -> None:
    """Keep the context-one predictor parameter-compatible with Icefall."""

    decoder = make_decoder(context_size=1)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(42)
        expected_state = {
            "embedding.weight": torch.randn_like(decoder.embedding.weight),
            "decoder_proj.weight": torch.randn_like(decoder.decoder_proj.weight),
            "decoder_proj.bias": torch.randn_like(decoder.decoder_proj.bias),
        }

    assert isinstance(decoder.conv, torch.nn.Identity)
    assert set(decoder.state_dict()) == set(expected_state)
    decoder.load_state_dict(expected_state, strict=True)
    for name, value in expected_state.items():
        torch.testing.assert_close(decoder.state_dict()[name], value)


def test_zipformer_trigram_predictor_matches_icefall_architecture() -> None:
    """Keep grouped convolution metadata and checkpoint names compatible."""

    decoder = make_decoder(context_size=2)

    assert isinstance(decoder.conv, torch.nn.Conv1d)
    assert decoder.conv.in_channels == DECODER_DIM
    assert decoder.conv.out_channels == DECODER_DIM
    assert decoder.conv.kernel_size == (2,)
    assert decoder.conv.groups == DECODER_DIM // 4
    assert decoder.conv.bias is None
    assert tuple(decoder.conv.weight.shape) == (
        DECODER_DIM,
        4,
        2,
    )
    assert set(decoder.state_dict()) == {
        "embedding.weight",
        "conv.weight",
        "decoder_proj.weight",
        "decoder_proj.bias",
    }


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("chunk_size", (1, 7, 36, 100))
@pytest.mark.parametrize("context_size", (1, 2))
def test_zipformer_context_lookup_matches_runtime_base_index(
    context_size: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    """Map every predictor context to the row selected by runtime indexing."""

    decoder = make_decoder(context_size, dtype)
    contexts = torch.tensor(
        tuple(product(range(-1, VOCAB_SIZE), repeat=context_size)),
        dtype=torch.int32,
    )
    indexes = torch.zeros(len(contexts), dtype=torch.int64)
    for position in range(context_size):
        indexes = indexes * (VOCAB_SIZE + 1) + contexts[:, position] + 1

    lookup = decoder.make_context_lookup(chunk_size)
    expected = decoder(contexts)

    assert lookup.shape == ((VOCAB_SIZE + 1) ** context_size, JOINER_DIM)
    assert lookup.dtype == dtype
    assert lookup.device == decoder.embedding.weight.device
    assert lookup.is_contiguous()
    assert not lookup.requires_grad
    assert torch.isfinite(lookup).all()
    assert expected.dtype == dtype
    torch.testing.assert_close(lookup[indexes], expected)


@pytest.mark.parametrize("chunk_size", (0, -1, 1.0, True))
def test_zipformer_context_lookup_rejects_invalid_chunk_size(
    chunk_size: int | float | bool,
) -> None:
    """Reject chunk sizes that cannot advance context-table construction."""

    decoder = make_decoder()

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        decoder.make_context_lookup(chunk_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_zipformer_joiner_matches_log_probability_reference(
    dtype: torch.dtype,
) -> None:
    """Match an explicit joiner projection and FP32 normalization."""

    joiner = make_joiner(dtype)
    generator = torch.Generator().manual_seed(44)
    decoder_output = torch.randn(3, JOINER_DIM, dtype=dtype, generator=generator)
    encoder_output = torch.randn(3, JOINER_DIM, dtype=dtype, generator=generator)
    logits = torch.nn.functional.linear(
        torch.tanh(encoder_output + decoder_output),
        joiner.output_proj.weight,
        joiner.output_proj.bias,
    )
    expected = torch.log_softmax(logits.float(), dim=1)

    actual = joiner(decoder_output, encoder_output)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual.exp().sum(dim=1), torch.ones(actual.size(0)), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_zipformer_joiner_normalizes_low_precision_logits_in_float32(
    dtype: torch.dtype,
) -> None:
    """Preserve low-probability detail by casting logits before LogSoftmax."""

    joiner = make_joiner(dtype)
    with torch.no_grad():
        joiner.output_proj.weight.zero_()
        joiner.output_proj.bias.copy_(
            torch.tensor((-4.0, -2.0, -1.0, -0.5, 0.0), dtype=dtype)
        )
    expected = torch.log_softmax(joiner.output_proj.bias.float(), dim=0)
    low_precision = torch.log_softmax(joiner.output_proj.bias, dim=0).float()
    assert not torch.allclose(low_precision, expected)

    actual = joiner(
        torch.zeros(2, JOINER_DIM, dtype=dtype),
        torch.zeros(2, JOINER_DIM, dtype=dtype),
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.expand_as(actual))
    torch.testing.assert_close(
        torch.logsumexp(actual, dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("dtype", "onnx_dtype"),
    (
        (torch.float32, onnx.TensorProto.FLOAT),
        (torch.float16, onnx.TensorProto.FLOAT16),
        (torch.bfloat16, onnx.TensorProto.BFLOAT16),
    ),
)
def test_zipformer_joiner_onnx_contract(
    tmp_path: Path,
    dtype: torch.dtype,
    onnx_dtype: int,
) -> None:
    """Preserve fixed TensorRT shapes and FP32 ONNX normalization."""

    joiner = make_joiner(dtype)
    onnx_path = tmp_path / f"zipformer_joiner_{dtype}.onnx"

    with torch.inference_mode():
        torch.onnx.export(
            joiner,
            (
                torch.zeros(4, JOINER_DIM, dtype=dtype),
                torch.zeros(4, JOINER_DIM, dtype=dtype),
            ),
            onnx_path,
            input_names=("decoder_input", "encoder_output"),
            output_names=("tokens_log_prob",),
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.checker.check_model(model)
    inputs = {value.name: value.type.tensor_type for value in model.graph.input}
    outputs = {value.name: value.type.tensor_type for value in model.graph.output}
    tensors = inputs | outputs

    assert tuple(inputs) == ("decoder_input", "encoder_output")
    assert tuple(outputs) == ("tokens_log_prob",)
    assert {
        name: tuple(dimension.dim_value for dimension in tensor.shape.dim)
        for name, tensor in tensors.items()
    } == {
        "decoder_input": (4, JOINER_DIM),
        "encoder_output": (4, JOINER_DIM),
        "tokens_log_prob": (4, VOCAB_SIZE),
    }
    assert {name: tensor.elem_type for name, tensor in tensors.items()} == {
        "decoder_input": onnx_dtype,
        "encoder_output": onnx_dtype,
        "tokens_log_prob": onnx.TensorProto.FLOAT,
    }

    value_types = {
        value.name: value.type.tensor_type
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    producers = {
        output_name: node for node in model.graph.node for output_name in node.output
    }
    log_softmax = producers["tokens_log_prob"]
    assert log_softmax.op_type == "LogSoftmax"
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in log_softmax.attribute
    } == {"axis": 1}
    assert value_types[log_softmax.input[0]].elem_type == onnx.TensorProto.FLOAT
