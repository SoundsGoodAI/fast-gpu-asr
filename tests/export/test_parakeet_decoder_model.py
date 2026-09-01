#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Numerical and export tests for the PyTorch Parakeet TDT decoder."""

from pathlib import Path

import onnx
import pytest
import torch

from fast_gpu_asr.constants import ONNX_OPSET_VERSION
from fast_gpu_asr.export.model.parakeet.decoder import Decoder

DECODER_DIM = 12
ENCODER_DIM = 16
JOINER_DIM = 10
NUM_EXTRA_OUTPUTS = 5
PRED_RNN_LAYERS = 2
VOCAB_SIZE = 17
TOKEN_OUTPUT_DIM = VOCAB_SIZE + 1


def make_decoder(
    dtype: torch.dtype = torch.float16,
    pred_rnn_layers: int = PRED_RNN_LAYERS,
) -> Decoder:
    """Create a deterministic compact Parakeet decoder for CPU tests."""

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        decoder = Decoder(
            vocab_size=VOCAB_SIZE,
            encoder_dim=ENCODER_DIM,
            decoder_dim=DECODER_DIM,
            joiner_dim=JOINER_DIM,
            pred_rnn_layers=pred_rnn_layers,
            num_extra_outputs=NUM_EXTRA_OUTPUTS,
            dtype=dtype,
        )
    return decoder.eval()


def test_tdt_decoder_matches_independent_hypotheses() -> None:
    """Keep batched hypotheses independent across outputs and recurrent states."""

    generator = torch.Generator().manual_seed(1)
    decoder = make_decoder()
    num_hypotheses = 5
    encoder_output = torch.randn(
        num_hypotheses,
        ENCODER_DIM,
        dtype=torch.float16,
        generator=generator,
    )
    targets = torch.tensor(((0,), (VOCAB_SIZE,), (7,), (3,), (12,)), dtype=torch.int32)
    state_1 = torch.randn(
        PRED_RNN_LAYERS,
        num_hypotheses,
        DECODER_DIM,
        dtype=torch.float16,
        generator=generator,
    )
    state_2 = torch.randn(
        PRED_RNN_LAYERS,
        num_hypotheses,
        DECODER_DIM,
        dtype=torch.float16,
        generator=generator,
    )

    actual = decoder(encoder_output, targets, state_1, state_2)
    expected = [
        decoder(
            encoder_output[index : index + 1],
            targets[index : index + 1],
            state_1[:, index : index + 1],
            state_2[:, index : index + 1],
        )
        for index in range(num_hypotheses)
    ]

    independent_outputs = zip(*expected, strict=True)
    for actual_output, output_parts, dimension in zip(
        actual, independent_outputs, (0, 0, 1, 1), strict=True
    ):
        torch.testing.assert_close(
            actual_output,
            torch.cat(output_parts, dim=dimension),
        )


@pytest.mark.parametrize("pred_rnn_layers", (1, PRED_RNN_LAYERS))
@pytest.mark.parametrize("num_hypotheses", (1, 3))
def test_tdt_decoder_matches_manual_single_step_lstm(
    pred_rnn_layers: int, num_hypotheses: int
) -> None:
    """Match an explicit LSTM and joiner calculation."""

    generator = torch.Generator().manual_seed(11)
    decoder = make_decoder(torch.float32, pred_rnn_layers)
    encoder_output = torch.randn(num_hypotheses, ENCODER_DIM, generator=generator)
    targets = torch.tensor(
        (7, 0, VOCAB_SIZE)[:num_hypotheses], dtype=torch.int32
    ).unsqueeze(1)
    state_1 = torch.randn(
        pred_rnn_layers, num_hypotheses, DECODER_DIM, generator=generator
    )
    state_2 = torch.randn(
        pred_rnn_layers, num_hypotheses, DECODER_DIM, generator=generator
    )

    layer_input = decoder.embedding(targets[:, 0])
    expected_hidden = []
    expected_cell = []
    for layer in range(pred_rnn_layers):
        gates = torch.nn.functional.linear(
            layer_input,
            getattr(decoder.lstm, f"weight_ih_l{layer}"),
            getattr(decoder.lstm, f"bias_ih_l{layer}"),
        ) + torch.nn.functional.linear(
            state_1[layer],
            getattr(decoder.lstm, f"weight_hh_l{layer}"),
            getattr(decoder.lstm, f"bias_hh_l{layer}"),
        )
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * state_2[layer] + torch.sigmoid(
            input_gate
        ) * torch.tanh(cell_gate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        expected_hidden.append(hidden)
        expected_cell.append(cell)
        layer_input = hidden

    joiner = decoder.output_proj(
        torch.relu(
            decoder.encoder_proj(encoder_output) + decoder.decoder_proj(layer_input)
        )
    )
    expected = (
        torch.log_softmax(joiner[:, :TOKEN_OUTPUT_DIM], dim=1),
        torch.log_softmax(joiner[:, TOKEN_OUTPUT_DIM:], dim=1),
        torch.stack(expected_hidden),
        torch.stack(expected_cell),
    )

    actual = decoder(encoder_output, targets, state_1, state_2)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)


def test_tdt_decoder_uses_blank_as_prediction_padding() -> None:
    """Initialize the final token ID as the zero prediction-padding embedding."""

    decoder = make_decoder(torch.float32)

    assert decoder.embedding.num_embeddings == TOKEN_OUTPUT_DIM
    assert decoder.embedding.padding_idx == VOCAB_SIZE
    torch.testing.assert_close(
        decoder.embedding(torch.tensor([VOCAB_SIZE], dtype=torch.int32)),
        torch.zeros(1, DECODER_DIM),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_tdt_decoder_preserves_precision_and_shape_contract(
    dtype: torch.dtype,
) -> None:
    """Return FP32 log probabilities and model-precision recurrent states."""

    decoder = make_decoder(dtype)
    encoder_output = torch.zeros(2, ENCODER_DIM, dtype=dtype)
    targets = torch.tensor(((0,), (VOCAB_SIZE,)), dtype=torch.int32)
    state_1 = torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype)
    state_2 = torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype)

    token_log_probs, duration_log_probs, output_state_1, output_state_2 = decoder(
        encoder_output, targets, state_1, state_2
    )

    assert {parameter.dtype for parameter in decoder.parameters()} == {dtype}
    assert token_log_probs.shape == (2, TOKEN_OUTPUT_DIM)
    assert token_log_probs.dtype == torch.float32
    assert duration_log_probs.shape == (2, NUM_EXTRA_OUTPUTS)
    assert duration_log_probs.dtype == torch.float32
    assert output_state_1.shape == (PRED_RNN_LAYERS, 2, DECODER_DIM)
    assert output_state_1.dtype == dtype
    assert output_state_2.shape == (PRED_RNN_LAYERS, 2, DECODER_DIM)
    assert output_state_2.dtype == dtype
    for output in (
        token_log_probs,
        duration_log_probs,
        output_state_1,
        output_state_2,
    ):
        assert torch.isfinite(output).all()
    for log_probs in (token_log_probs, duration_log_probs):
        torch.testing.assert_close(
            torch.logsumexp(log_probs, dim=1),
            torch.zeros(2),
            atol=1e-6,
            rtol=0.0,
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_tdt_decoder_normalizes_low_precision_logits_in_float32(
    dtype: torch.dtype,
) -> None:
    """Cast low-precision logits before normalization.

    This preserves log-probability detail in the exported decoder outputs.
    """

    decoder = make_decoder(dtype)
    with torch.no_grad():
        decoder.output_proj.weight.zero_()
        decoder.output_proj.bias.copy_(
            torch.linspace(
                -40.0,
                40.0,
                decoder.output_proj.out_features,
                dtype=dtype,
            )
        )

    token_log_probs, duration_log_probs, _, _ = decoder(
        torch.zeros(2, ENCODER_DIM, dtype=dtype),
        torch.ones(2, 1, dtype=torch.int32),
        torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype),
        torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype),
    )
    logits = decoder.output_proj.bias.to(torch.float32)

    torch.testing.assert_close(
        token_log_probs,
        torch.log_softmax(logits[:TOKEN_OUTPUT_DIM], dim=0).expand_as(token_log_probs),
    )
    torch.testing.assert_close(
        duration_log_probs,
        torch.log_softmax(logits[TOKEN_OUTPUT_DIM:], dim=0).expand_as(
            duration_log_probs
        ),
    )
    torch.testing.assert_close(
        torch.logsumexp(token_log_probs, dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.logsumexp(duration_log_probs, dim=1),
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
def test_tdt_decoder_onnx_contract(
    tmp_path: Path, dtype: torch.dtype, onnx_dtype: int
) -> None:
    """Preserve the fixed-capacity TensorRT interface and FP32 normalization."""

    decoder = make_decoder(dtype)
    onnx_path = tmp_path / f"parakeet_decoder_{dtype}.onnx"

    with torch.inference_mode():
        torch.onnx.export(
            decoder,
            (
                torch.zeros(2, ENCODER_DIM, dtype=dtype),
                torch.zeros(2, 1, dtype=torch.int32),
                torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype),
                torch.zeros(PRED_RNN_LAYERS, 2, DECODER_DIM, dtype=dtype),
            ),
            onnx_path,
            input_names=(
                "encoder_output",
                "targets",
                "input_states_1",
                "input_states_2",
            ),
            output_names=(
                "token_log_probs",
                "duration_log_probs",
                "output_states_1",
                "output_states_2",
            ),
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.checker.check_model(model)
    inputs = {value.name: value.type.tensor_type for value in model.graph.input}
    outputs = {value.name: value.type.tensor_type for value in model.graph.output}

    assert tuple(inputs) == (
        "encoder_output",
        "targets",
        "input_states_1",
        "input_states_2",
    )
    assert tuple(outputs) == (
        "token_log_probs",
        "duration_log_probs",
        "output_states_1",
        "output_states_2",
    )
    tensors = inputs | outputs
    assert {
        name: tuple(dimension.dim_value for dimension in tensor.shape.dim)
        for name, tensor in tensors.items()
    } == {
        "encoder_output": (2, ENCODER_DIM),
        "targets": (2, 1),
        "input_states_1": (PRED_RNN_LAYERS, 2, DECODER_DIM),
        "input_states_2": (PRED_RNN_LAYERS, 2, DECODER_DIM),
        "token_log_probs": (2, TOKEN_OUTPUT_DIM),
        "duration_log_probs": (2, NUM_EXTRA_OUTPUTS),
        "output_states_1": (PRED_RNN_LAYERS, 2, DECODER_DIM),
        "output_states_2": (PRED_RNN_LAYERS, 2, DECODER_DIM),
    }
    assert {name: tensor.elem_type for name, tensor in tensors.items()} == {
        "encoder_output": onnx_dtype,
        "targets": onnx.TensorProto.INT32,
        "input_states_1": onnx_dtype,
        "input_states_2": onnx_dtype,
        "token_log_probs": onnx.TensorProto.FLOAT,
        "duration_log_probs": onnx.TensorProto.FLOAT,
        "output_states_1": onnx_dtype,
        "output_states_2": onnx_dtype,
    }

    value_types = {
        value.name: value.type.tensor_type
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    producers = {
        output_name: node for node in model.graph.node for output_name in node.output
    }
    for output_name in ("token_log_probs", "duration_log_probs"):
        log_softmax = producers[output_name]
        assert log_softmax.op_type == "LogSoftmax"
        assert {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in log_softmax.attribute
        } == {"axis": 1}
        assert value_types[log_softmax.input[0]].elem_type == onnx.TensorProto.FLOAT
