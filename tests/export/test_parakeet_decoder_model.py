#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Numerical and export tests for the PyTorch Parakeet TDT decoder."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from fast_gpu_asr.constants import ONNX_OPSET_VERSION
from fast_gpu_asr.export.model.parakeet.decoder import Decoder

DEFAULT_CONFIG = {
    "vocab_size": 17,
    "encoder_dim": 16,
    "decoder_dim": 12,
    "joiner_dim": 10,
    "num_extra_outputs": 5,
}
ALTERNATE_CONFIG = {
    "vocab_size": 7,
    "encoder_dim": 11,
    "decoder_dim": 9,
    "joiner_dim": 6,
    "num_extra_outputs": 3,
}
ONNX_INPUT_NAMES = (
    "encoder_output",
    "targets",
    "input_states_1",
    "input_states_2",
)
ONNX_OUTPUT_NAMES = (
    "token_log_probs",
    "duration_log_probs",
    "output_states_1",
    "output_states_2",
)


def make_decoder(
    dtype: torch.dtype = torch.float16,
    pred_rnn_layers: int = 2,
    config: dict[str, int] = DEFAULT_CONFIG,
) -> Decoder:
    """Create a seeded decoder without changing the caller's RNG state.

    Parameters
    ----------
    dtype : torch.dtype
        Decoder parameter and recurrent-state precision.
    pred_rnn_layers : int
        Number of predictor LSTM layers.
    config : dict[str, int]
        Vocabulary and feature dimensions, passed without mutation.

    Returns
    -------
    Decoder
        Deterministic CPU decoder in evaluation mode.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        decoder = Decoder(**config, pred_rnn_layers=pred_rnn_layers, dtype=dtype)
    return decoder.eval()


def reorder_lstm_gates_for_onnx(
    tensor: torch.Tensor,
) -> np.typing.NDArray[np.float32]:
    """Return PyTorch IFGO gates in ONNX IOFC order as float32 NumPy values.

    Parameters
    ----------
    tensor : torch.Tensor
        CPU LSTM weight or bias with four equal gate blocks along axis zero.

    Returns
    -------
    np.typing.NDArray[np.float32]
        Detached copy with input, output, forget, and cell gates in that order.
    """

    input_gate, forget_gate, cell_gate, output_gate = tensor.detach().float().chunk(4)
    return torch.cat((input_gate, output_gate, forget_gate, cell_gate)).numpy()


@pytest.mark.parametrize("pred_rnn_layers", (1, 2))
@pytest.mark.parametrize("config", (DEFAULT_CONFIG, ALTERNATE_CONFIG))
def test_checkpoint_layout_and_blank_embedding(
    pred_rnn_layers: int, config: dict[str, int]
) -> None:
    decoder = make_decoder(torch.float32, pred_rnn_layers, config)
    vocab_size = config["vocab_size"]
    decoder_dim = config["decoder_dim"]
    joiner_dim = config["joiner_dim"]
    num_outputs = vocab_size + 1 + config["num_extra_outputs"]
    expected_shapes = {
        "embedding.weight": (vocab_size + 1, decoder_dim),
        "decoder_proj.weight": (joiner_dim, decoder_dim),
        "decoder_proj.bias": (joiner_dim,),
        "encoder_proj.weight": (joiner_dim, config["encoder_dim"]),
        "encoder_proj.bias": (joiner_dim,),
        "output_proj.weight": (num_outputs, joiner_dim),
        "output_proj.bias": (num_outputs,),
    }
    for layer in range(pred_rnn_layers):
        expected_shapes.update(
            {
                f"lstm.weight_ih_l{layer}": (4 * decoder_dim, decoder_dim),
                f"lstm.weight_hh_l{layer}": (4 * decoder_dim, decoder_dim),
                f"lstm.bias_ih_l{layer}": (4 * decoder_dim,),
                f"lstm.bias_hh_l{layer}": (4 * decoder_dim,),
            }
        )

    assert {
        name: tuple(tensor.shape) for name, tensor in decoder.state_dict().items()
    } == expected_shapes
    assert decoder.embedding.padding_idx == vocab_size
    assert torch.count_nonzero(decoder.embedding.weight[vocab_size]) == 0


def test_batched_hypotheses_are_independent() -> None:
    decoder = make_decoder()
    generator = torch.Generator().manual_seed(1)
    encoder_output = torch.randn(5, 16, dtype=torch.float16, generator=generator)
    targets = torch.tensor(((0,), (17,), (7,), (3,), (12,)), dtype=torch.int32)
    states = tuple(
        torch.randn(2, 5, 12, dtype=torch.float16, generator=generator)
        for _ in range(2)
    )
    inputs = (encoder_output, targets, *states)
    snapshots = tuple(value.clone() for value in inputs)

    actual = decoder(*inputs)

    torch.testing.assert_close(inputs, snapshots, rtol=0.0, atol=0.0)
    individual = [
        decoder(
            encoder_output[index : index + 1],
            targets[index : index + 1],
            states[0][:, index : index + 1],
            states[1][:, index : index + 1],
        )
        for index in range(5)
    ]
    expected = tuple(
        torch.cat(parts, dim=dimension)
        for parts, dimension in zip(
            zip(*individual, strict=True), (0, 0, 1, 1), strict=True
        )
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs, snapshots, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("pred_rnn_layers", (1, 2))
@pytest.mark.parametrize("num_hypotheses", (1, 3))
@pytest.mark.parametrize("config", (DEFAULT_CONFIG, ALTERNATE_CONFIG))
def test_decoder_matches_manual_lstm_step(
    pred_rnn_layers: int, num_hypotheses: int, config: dict[str, int]
) -> None:
    decoder = make_decoder(torch.float32, pred_rnn_layers, config)
    vocab_size = config["vocab_size"]
    decoder_dim = config["decoder_dim"]
    generator = torch.Generator().manual_seed(11)
    encoder_output = torch.randn(
        num_hypotheses, config["encoder_dim"], generator=generator
    )
    targets = torch.tensor(
        (vocab_size // 2, 0, vocab_size)[:num_hypotheses], dtype=torch.int32
    ).unsqueeze(1)
    state_1, state_2 = (
        torch.randn(pred_rnn_layers, num_hypotheses, decoder_dim, generator=generator)
        for _ in range(2)
    )

    layer_input = decoder.embedding(targets[:, 0])
    expected_hidden, expected_cell = [], []
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

    logits = decoder.output_proj(
        torch.relu(
            decoder.encoder_proj(encoder_output) + decoder.decoder_proj(layer_input)
        )
    )
    expected = (
        torch.log_softmax(logits[:, : vocab_size + 1], dim=1),
        torch.log_softmax(logits[:, vocab_size + 1 :], dim=1),
        torch.stack(expected_hidden),
        torch.stack(expected_cell),
    )

    actual = decoder(encoder_output, targets, state_1, state_2)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_decoder_precision_and_shape_contract(dtype: torch.dtype) -> None:
    decoder = make_decoder(dtype)

    outputs = decoder(
        torch.zeros(2, 16, dtype=dtype),
        torch.tensor(((0,), (17,)), dtype=torch.int32),
        torch.zeros(2, 2, 12, dtype=dtype),
        torch.zeros(2, 2, 12, dtype=dtype),
    )

    assert {parameter.dtype for parameter in decoder.parameters()} == {dtype}
    assert [(output.dtype, tuple(output.shape)) for output in outputs] == [
        (torch.float32, (2, 18)),
        (torch.float32, (2, 5)),
        (dtype, (2, 2, 12)),
        (dtype, (2, 2, 12)),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)
    for log_probs in outputs[:2]:
        torch.testing.assert_close(
            torch.logsumexp(log_probs, dim=1), torch.zeros(2), atol=1e-6, rtol=0.0
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_decoder_normalizes_low_precision_logits_in_float32(dtype: torch.dtype) -> None:
    decoder = make_decoder(dtype)
    with torch.no_grad():
        decoder.output_proj.weight.zero_()
        decoder.output_proj.bias.copy_(torch.linspace(-40.0, 40.0, 23, dtype=dtype))

    outputs = decoder(
        torch.zeros(2, 16, dtype=dtype),
        torch.ones(2, 1, dtype=torch.int32),
        torch.zeros(2, 2, 12, dtype=dtype),
        torch.zeros(2, 2, 12, dtype=dtype),
    )

    for actual, bias in zip(
        outputs[:2], decoder.output_proj.bias.split((18, 5)), strict=True
    ):
        expected = torch.log_softmax(bias.float(), dim=0)
        assert not torch.allclose(torch.log_softmax(bias, dim=0).float(), expected)
        torch.testing.assert_close(actual, expected.expand(2, len(bias)))
        torch.testing.assert_close(
            torch.logsumexp(actual, dim=1), torch.zeros(2), atol=1e-6, rtol=0.0
        )


@pytest.mark.parametrize(
    "dtype,onnx_dtype",
    [
        (torch.float32, onnx.TensorProto.FLOAT),
        (torch.float16, onnx.TensorProto.FLOAT16),
        (torch.bfloat16, onnx.TensorProto.BFLOAT16),
    ],
    ids=("fp32", "fp16", "bf16"),
)
@pytest.mark.parametrize("pred_rnn_layers", (1, 2))
def test_decoder_onnx_contract(
    tmp_path: Path, dtype: torch.dtype, onnx_dtype: int, pred_rnn_layers: int
) -> None:
    decoder = make_decoder(dtype, pred_rnn_layers)
    onnx_path = tmp_path / "parakeet_decoder.onnx"
    state_shape = (pred_rnn_layers, 2, 12)

    with torch.inference_mode():
        torch.onnx.export(
            decoder,
            (
                torch.zeros(2, 16, dtype=dtype),
                torch.zeros(2, 1, dtype=torch.int32),
                torch.zeros(state_shape, dtype=dtype),
                torch.zeros(state_shape, dtype=dtype),
            ),
            onnx_path,
            input_names=ONNX_INPUT_NAMES,
            output_names=ONNX_OUTPUT_NAMES,
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    onnx.checker.check_model(model)
    inputs = {value.name: value.type.tensor_type for value in model.graph.input}
    outputs = {value.name: value.type.tensor_type for value in model.graph.output}
    assert [opset.version for opset in model.opset_import if opset.domain == ""] == [
        ONNX_OPSET_VERSION
    ]
    assert tuple(inputs) == ONNX_INPUT_NAMES
    assert tuple(outputs) == ONNX_OUTPUT_NAMES
    assert {
        name: (tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim))
        for name, tensor in (inputs | outputs).items()
    } == {
        "encoder_output": (onnx_dtype, (2, 16)),
        "targets": (onnx.TensorProto.INT32, (2, 1)),
        "input_states_1": (onnx_dtype, state_shape),
        "input_states_2": (onnx_dtype, state_shape),
        "token_log_probs": (onnx.TensorProto.FLOAT, (2, 18)),
        "duration_log_probs": (onnx.TensorProto.FLOAT, (2, 5)),
        "output_states_1": (onnx_dtype, state_shape),
        "output_states_2": (onnx_dtype, state_shape),
    }
    floating_dtypes = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }
    assert {
        value.data_type
        for value in model.graph.initializer
        if value.data_type in floating_dtypes
    } == {onnx_dtype}
    initializers = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    lstm_nodes = [node for node in model.graph.node if node.op_type == "LSTM"]
    assert len(lstm_nodes) == pred_rnn_layers
    for layer, node in enumerate(lstm_nodes):
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes["hidden_size"] == 12
        assert attributes.get("direction", b"forward") == b"forward"
        assert attributes.get("input_forget", 0) == 0
        assert attributes.get("layout", 0) == 0
        for name, connection in zip(node.input[1:3], ("ih", "hh"), strict=True):
            expected = reorder_lstm_gates_for_onnx(
                getattr(decoder.lstm, f"weight_{connection}_l{layer}")
            )
            np.testing.assert_array_equal(
                initializers[name].astype(np.float32), expected[np.newaxis]
            )

        actual_bias = initializers[node.input[3]].astype(np.float32)
        input_bias, recurrent_bias = (
            reorder_lstm_gates_for_onnx(
                getattr(decoder.lstm, f"bias_{connection}_l{layer}")
            )
            for connection in ("ih", "hh")
        )
        np.testing.assert_array_equal(
            actual_bias[:, :48] + actual_bias[:, 48:],
            (input_bias + recurrent_bias)[np.newaxis],
        )

    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    producers = {name: node for node in model.graph.node for name in node.output}
    for output_name, state_index in (("output_states_1", 1), ("output_states_2", 2)):
        state_outputs = tuple(node.output[state_index] for node in lstm_nodes)
        if pred_rnn_layers == 1:
            assert state_outputs == (output_name,)
        else:
            concat = producers[output_name]
            assert concat.op_type == "Concat"
            assert {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in concat.attribute
            } == {"axis": 0}
            assert tuple(concat.input) == state_outputs

    slice_data_inputs = set()
    for output_name, expected_start, expected_end in (
        ("token_log_probs", 0, 18),
        ("duration_log_probs", 18, 23),
    ):
        log_softmax = producers[output_name]
        assert log_softmax.op_type == "LogSoftmax"
        assert {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in log_softmax.attribute
        } == {"axis": 1}
        assert value_types[log_softmax.input[0]] == onnx.TensorProto.FLOAT

        slice_output = log_softmax.input[0]
        if producers[slice_output].op_type == "Cast":
            slice_output = producers[slice_output].input[0]
        slice_node = producers[slice_output]
        assert slice_node.op_type == "Slice"
        data_input, *slice_inputs = slice_node.input
        slice_data_inputs.add(data_input)
        start, end, axis, step = (initializers[name].item() for name in slice_inputs)
        assert (start, min(end, 23), axis, step) == (expected_start, expected_end, 1, 1)
    assert len(slice_data_inputs) == 1
