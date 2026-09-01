#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Runtime and validation tests for the Parakeet TDT decoder."""

from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast

import cupy as cp
import cupyx as cpx
import numpy as np
import pytest
import tensorrt as trt

from fast_gpu_asr.constants import INT32_MAX
from fast_gpu_asr.decoder import parakeet_decoder
from fast_gpu_asr.decoder.gpu_kernels import (
    TDT_BEAM_SEARCH_KERNEL,
    TDT_FINALIZE_KERNEL,
    TDT_PREPARE_INPUTS_KERNEL,
    TDT_SELECT_TOKENS_KERNEL,
)
from fast_gpu_asr.decoder.parakeet_decoder import ParakeetModifiedBeamSearchDecoder
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError

PARAKEET_SEARCH_BUFFER_PAIRS = (
    ("hypothesis_scores", "next_scores"),
    ("hypothesis_lengths", "next_lengths"),
    ("time_indexes", "next_time_indexes"),
    ("last_tokens", "next_last_tokens"),
    ("symbols_at_timestep", "next_symbols_at_timestep"),
    ("hypothesis_nodes", "next_nodes"),
    ("hypothesis_hashes", "next_hashes"),
    ("state_1", "next_state_1"),
    ("state_2", "next_state_2"),
)


@pytest.mark.cuda
def test_parakeet_token_selection_scans_full_vocabulary() -> None:
    """Select the top beam candidates from every nonblank vocabulary token."""

    vocab_size = 7
    beam = 2
    threads = 256
    token_log_probs = cp.array(
        [[-5.0, -4.0, -3.0, -2.0, -1.0, 0.75, 1.25, -10.0]],
        dtype=np.float32,
    )
    top_token_scores = cp.empty((1, beam), dtype=np.float32)
    top_token_indexes = cp.empty((1, beam), dtype=np.int32)
    shared_memory_bytes = vocab_size * np.dtype(np.float32).itemsize + (
        threads // 32
    ) * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)

    TDT_SELECT_TOKENS_KERNEL(
        (1,),
        (threads,),
        (
            token_log_probs,
            cp.array([0.0], dtype=np.float32),
            cp.array([0], dtype=np.int32),
            cp.array([1], dtype=np.int32),
            top_token_scores,
            top_token_indexes,
            np.int32(vocab_size),
            np.int32(beam),
        ),
        shared_mem=shared_memory_bytes,
    )

    np.testing.assert_allclose(top_token_scores.get(), [[1.25, 0.75]])
    np.testing.assert_array_equal(top_token_indexes.get(), [[6, 5]])


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_parakeet_token_selection_keeps_nonfinite_indexes_in_bounds(
    invalid_score: float,
) -> None:
    """Select distinct real token indexes when every score is non-finite."""

    vocab_size = 3
    beam = 2
    threads = 256
    top_token_scores = cp.empty((1, beam), dtype=np.float32)
    top_token_indexes = cp.empty((1, beam), dtype=np.int32)
    shared_memory_bytes = vocab_size * np.dtype(np.float32).itemsize + (
        threads // 32
    ) * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)

    TDT_SELECT_TOKENS_KERNEL(
        (1,),
        (threads,),
        (
            cp.full((1, vocab_size + 1), invalid_score, dtype=np.float32),
            cp.array([0.0], dtype=np.float32),
            cp.array([0], dtype=np.int32),
            cp.array([1], dtype=np.int32),
            top_token_scores,
            top_token_indexes,
            np.int32(vocab_size),
            np.int32(beam),
        ),
        shared_mem=shared_memory_bytes,
    )

    np.testing.assert_array_equal(top_token_scores.get(), [[-np.inf, -np.inf]])
    np.testing.assert_array_equal(top_token_indexes.get(), [[0, 1]])


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("encoder_dtype", "encoder_dtype_code"),
    (
        pytest.param(np.dtype(np.float32), np.int32(0), id="encoder-fp32"),
        pytest.param(np.dtype(np.float16), np.int32(1), id="encoder-fp16"),
        pytest.param(cp.dtype("bfloat16"), np.int32(2), id="encoder-bf16"),
    ),
)
@pytest.mark.parametrize(
    ("decoder_dtype", "decoder_dtype_code"),
    (
        pytest.param(np.dtype(np.float32), np.int32(0), id="decoder-fp32"),
        pytest.param(np.dtype(np.float16), np.int32(1), id="decoder-fp16"),
        pytest.param(cp.dtype("bfloat16"), np.int32(2), id="decoder-bf16"),
    ),
)
def test_parakeet_prepare_inputs_converts_precision(
    encoder_dtype: np.dtype,
    encoder_dtype_code: np.int32,
    decoder_dtype: np.dtype,
    decoder_dtype_code: np.int32,
) -> None:
    """Convert active encoder frames and clear inactive decoder beam slots."""

    encoder_output = (
        cp.arange(16, dtype=cp.float32).reshape(1, 2, 8).astype(encoder_dtype)
    )
    encoder_input = cp.empty((2, 8), dtype=decoder_dtype)
    targets = cp.empty((2, 1), dtype=cp.int32)
    TDT_PREPARE_INPUTS_KERNEL(
        (2,),
        (256,),
        (
            encoder_output,
            cp.array([2], dtype=cp.int32),
            cp.array([0.0, -cp.inf], dtype=cp.float32),
            cp.array([1, 0], dtype=cp.int32),
            cp.array([7, 8], dtype=cp.int32),
            encoder_input,
            targets,
            np.int32(1),
            np.int32(2),
            np.int32(8),
            np.int32(2),
            encoder_dtype_code,
            decoder_dtype_code,
        ),
    )

    np.testing.assert_array_equal(
        encoder_input[0].astype(cp.float32).get(),
        encoder_output[0, 1].astype(cp.float32).get(),
    )
    np.testing.assert_array_equal(
        encoder_input[1].astype(cp.float32).get(), np.zeros(8, dtype=np.float32)
    )
    np.testing.assert_array_equal(targets.get(), np.array([[7], [0]], dtype=np.int32))


@pytest.mark.cuda
@pytest.mark.parametrize("output_length", (-1, 10))
def test_parakeet_prepare_inputs_clamps_invalid_output_length(
    output_length: int,
) -> None:
    """Do not keep a hypothesis active outside the available encoder frames."""

    encoder_input = cp.full((1, 4), -1.0, dtype=cp.float32)
    targets = cp.full((1, 1), -1, dtype=cp.int32)
    TDT_PREPARE_INPUTS_KERNEL(
        (1,),
        (256,),
        (
            cp.arange(8, dtype=cp.float32).reshape(1, 2, 4),
            cp.array([output_length], dtype=cp.int32),
            cp.array([0.0], dtype=cp.float32),
            cp.array([2], dtype=cp.int32),
            cp.array([7], dtype=cp.int32),
            encoder_input,
            targets,
            np.int32(1),
            np.int32(2),
            np.int32(4),
            np.int32(1),
            np.int32(0),
            np.int32(0),
        ),
    )

    np.testing.assert_array_equal(
        encoder_input.get(), np.zeros((1, 4), dtype=np.float32)
    )
    np.testing.assert_array_equal(targets.get(), np.zeros((1, 1), dtype=np.int32))


def make_fake_parakeet_decoder(
    beam: int = 1, batch_size: int = 1
) -> ParakeetModifiedBeamSearchDecoder:
    """Create a decoder without loading TensorRT."""

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    decoder.device = cp.cuda.Device(0)
    decoder.batch_size = batch_size
    decoder.beam = beam
    decoder.decoder_capacity = batch_size * beam
    decoder.encoder_dim = 3
    decoder.blank_id = 2
    decoder.durations_array = cp.array([0, 1], dtype=np.int32)
    decoder.positive_duration_indexes_array = cp.array([1], dtype=np.int32)
    decoder.max_symbols_per_timestep = 10
    decoder.encoder_frame_shift_sec = 0.08
    decoder.blank_penalty = 0.0
    decoder.state_layers = 1
    decoder.state_hidden_dim = 3
    decoder.kernel_dtype_map = {
        np.dtype(np.float32): np.int32(0),
        np.dtype(np.float16): np.int32(1),
        cp.dtype("bfloat16"): np.int32(2),
    }
    decoder.state_dtype = np.int32(0)
    decoder.encoder_input_dtype = np.int32(0)
    decoder.prepare_inputs_threads = 256
    decoder.token_selection_threads = 512
    decoder.beam_search_threads = 256
    beam_search_reductions = decoder.beam_search_threads // 32
    decoder.stream = cp.cuda.Stream(non_blocking=True)
    candidate_count = beam * (
        decoder.durations_array.size * beam
        + decoder.positive_duration_indexes_array.size
    )
    decoder.beam_search_shared_memory_bytes = (
        candidate_count * np.dtype(np.float32).itemsize
        + beam_search_reductions
        * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
        + beam * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )
    token_selection_reductions = decoder.token_selection_threads // 32
    decoder.token_selection_shared_memory_bytes = decoder.blank_id * np.dtype(
        np.float32
    ).itemsize + token_selection_reductions * (
        np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize
    )
    decoder.cuda_graph = None
    decoder.cuda_graph_signature = None
    decoder.cuda_graph_supported = False

    capacity = decoder.decoder_capacity
    search_shape = (batch_size, beam)
    decoder.encoder_input = cp.empty((capacity, 3), dtype=np.float32)
    decoder.targets = cp.empty((capacity, 1), dtype=np.int32)
    decoder.state_1 = cp.empty((1, capacity, 3), dtype=np.float32)
    decoder.state_2 = cp.empty((1, capacity, 3), dtype=np.float32)
    decoder.token_log_probs = cp.empty(
        (capacity, decoder.blank_id + 1), dtype=np.float32
    )
    decoder.duration_log_probs = cp.empty((capacity, 2), dtype=np.float32)
    decoder.output_state_1 = cp.empty((1, capacity, 3), dtype=np.float32)
    decoder.output_state_2 = cp.empty((1, capacity, 3), dtype=np.float32)
    decoder.next_state_1 = cp.empty_like(decoder.state_1)
    decoder.next_state_2 = cp.empty_like(decoder.state_2)
    decoder.hypothesis_scores = cp.empty(search_shape, dtype=np.float32)
    decoder.next_scores = cp.empty_like(decoder.hypothesis_scores)
    decoder.hypothesis_lengths = cp.empty(search_shape, dtype=np.int32)
    decoder.next_lengths = cp.empty_like(decoder.hypothesis_lengths)
    decoder.time_indexes = cp.empty(capacity, dtype=np.int32)
    decoder.next_time_indexes = cp.empty_like(decoder.time_indexes)
    decoder.last_tokens = cp.empty(capacity, dtype=np.int32)
    decoder.next_last_tokens = cp.empty_like(decoder.last_tokens)
    decoder.symbols_at_timestep = cp.empty(capacity, dtype=np.int32)
    decoder.next_symbols_at_timestep = cp.empty_like(decoder.symbols_at_timestep)
    decoder.parent_indexes = cp.empty(capacity, dtype=np.int32)
    decoder.use_output_state = cp.empty(capacity, dtype=np.uint8)
    decoder.search_output_lengths = cp.empty(batch_size, dtype=np.int32)
    decoder.active_flags = cp.empty(batch_size, dtype=np.int32)
    decoder.active_flags_host = cpx.empty_pinned(batch_size, dtype=np.int32)
    decoder.runtime_dimensions = cp.empty(2, dtype=np.int32)
    decoder.runtime_dimensions_host = cpx.empty_pinned(2, dtype=np.int32)
    decoder.top_token_scores = cp.empty((capacity, beam), dtype=np.float32)
    decoder.top_token_indexes = cp.empty((capacity, beam), dtype=np.int32)
    decoder.completed_scores = cp.empty(batch_size, dtype=np.float32)
    decoder.completed_lengths = cp.empty(batch_size, dtype=np.int32)
    decoder.completed_nodes = cp.empty(batch_size, dtype=np.int32)
    decoder.output_lengths = cp.empty(batch_size, dtype=np.int32)
    decoder.output_lengths_host = cpx.empty_pinned(batch_size, dtype=np.int32)
    decoder.token_capacity = 0
    decoder.hypothesis_nodes = cp.empty(search_shape, dtype=np.int32)
    decoder.next_nodes = cp.empty(search_shape, dtype=np.int32)
    decoder.hypothesis_hashes = cp.empty(search_shape, dtype=np.uint64)
    decoder.next_hashes = cp.empty(search_shape, dtype=np.uint64)
    decoder.node_counts = cp.empty(batch_size, dtype=np.int32)
    decoder.node_parents = None
    decoder.node_tokens = None
    decoder.node_timestamps = None
    decoder.output_tokens = None
    decoder.output_timestamps = None
    decoder.output_tokens_host = None
    decoder.output_timestamps_host = None
    return decoder


@pytest.mark.cuda
def test_parakeet_decoder_rejects_recurrent_state_binding_failure() -> None:
    """Stop before execution when TensorRT rejects either recurrent input."""

    decoder = make_fake_parakeet_decoder()

    class RejectingContext:
        def __init__(self) -> None:
            self.binding_names: list[str] = []

        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            self.binding_names.append(name)
            return False

        def execute_async_v3(self, stream_ptr: int) -> bool:
            raise AssertionError(
                "Decoder execution must not start after binding failure"
            )

    context = RejectingContext()
    decoder.decoder = context
    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder(
            cp.zeros((1, 1, 3), dtype=np.float32),
            cp.array([1], dtype=np.int32),
        )

    assert context.binding_names == ["input_states_1", "input_states_2"]


@pytest.mark.cuda
def test_parakeet_decoder_restores_buffers_after_runtime_binding_failure() -> None:
    """Restore canonical ping-pong buffers after a search-step binding failure."""

    decoder = make_fake_parakeet_decoder()
    original_scores = decoder.hypothesis_scores
    original_state_1 = decoder.state_1
    original_state_2 = decoder.state_2

    class RejectingContext:
        def __init__(self) -> None:
            self.binding_calls = 0

        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            self.binding_calls += 1
            return self.binding_calls < 3

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, 0.0, -2.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array([[-2.0, 0.0]], dtype=np.float32)
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = RejectingContext()
    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder(
            cp.zeros((1, 1, 3), dtype=np.float32),
            cp.array([1], dtype=np.int32),
        )

    assert decoder.hypothesis_scores is original_scores
    assert decoder.state_1 is original_state_1
    assert decoder.state_2 is original_state_2


@pytest.mark.cuda
def test_parakeet_decoder_returns_empty_results_for_zero_frames() -> None:
    """Return one empty result per actual utterance when no frames exist."""

    decoder = make_fake_parakeet_decoder(batch_size=2)

    token_ids, timestamps = decoder(
        cp.empty((1, 0, 3), dtype=np.float32),
        cp.zeros(1, dtype=np.int32),
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_reports_tensorrt_execution_failure() -> None:
    """Raise an inference error when TensorRT rejects decoder execution."""

    decoder = make_fake_parakeet_decoder()

    class FailingContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            return False

    decoder.decoder = FailingContext()
    with pytest.raises(ASRInferenceError, match="TensorRT decoder execution failed"):
        decoder(
            cp.zeros((1, 1, 3), dtype=np.float32),
            cp.ones(1, dtype=np.int32),
        )


@pytest.mark.cuda
def test_parakeet_decoder_restores_buffers_after_execution_failure() -> None:
    """Restore ping-pong ownership when execution fails after one search step."""

    decoder = make_fake_parakeet_decoder()
    decoder.max_symbols_per_timestep = 1
    original_scores = decoder.hypothesis_scores
    original_state_1 = decoder.state_1
    original_state_2 = decoder.state_2

    class FailingContext:
        def __init__(self) -> None:
            self.calls = 0

        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            self.calls += 1
            if self.calls > 1:
                return False
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, 0.0, -2.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array([[0.0, -10.0]], dtype=np.float32)
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = FailingContext()
    with pytest.raises(ASRInferenceError, match="TensorRT decoder execution failed"):
        decoder(
            cp.zeros((1, 1, 3), dtype=np.float32),
            cp.ones(1, dtype=np.int32),
        )

    assert decoder.hypothesis_scores is original_scores
    assert decoder.state_1 is original_state_1
    assert decoder.state_2 is original_state_2


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_parakeet_decoder_handles_nonfinite_search_scores(
    invalid_score: float,
) -> None:
    """Keep both Parakeet top-k stages in bounds for non-finite model output."""

    decoder = make_fake_parakeet_decoder(beam=2)

    class FakeContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs.fill(invalid_score)
            decoder.duration_log_probs.fill(invalid_score)
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.array([1], dtype=np.int32),
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_emits_token_with_positive_duration() -> None:
    """Emit a nonblank token and advance by its selected positive duration."""

    decoder = make_fake_parakeet_decoder()

    class FakeContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, 0.0, -2.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array(
                [[-2.0, 0.0]],
                dtype=np.float32,
            )
            decoder.output_state_1[...] = decoder.state_1 + 1.0
            decoder.output_state_2[...] = decoder.state_2 + 1.0
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.array([1], dtype=np.int32),
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.0]])


@pytest.mark.cuda
def test_parakeet_decoder_blank_advances_without_emitting_token() -> None:
    """Advance blank candidates using only positive duration outputs."""

    decoder = make_fake_parakeet_decoder()

    class BlankContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, -11.0, 0.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array(
                [[0.0, 0.0]],
                dtype=np.float32,
            )
            decoder.output_state_1[...] = decoder.state_1 + 1.0
            decoder.output_state_2[...] = decoder.state_2 + 1.0
            return True

    decoder.decoder = BlankContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 2, 3), dtype=np.float32),
        cp.array([2], dtype=np.int32),
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_limits_zero_duration_token_emissions() -> None:
    """Force frame advancement after the configured zero-duration token limit."""

    decoder = make_fake_parakeet_decoder()

    class ZeroDurationTokenContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, 0.0, -2.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array(
                [[0.0, -2.0]],
                dtype=np.float32,
            )
            decoder.output_state_1[...] = decoder.state_1 + 1.0
            decoder.output_state_2[...] = decoder.state_2 + 1.0
            return True

    decoder.decoder = ZeroDurationTokenContext()
    decoder.max_symbols_per_timestep = 2
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.array([1], dtype=np.int32),
    )

    assert token_ids == [[1, 1]]
    np.testing.assert_allclose(timestamps, [[0.0, 0.0]])


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("blank_penalty", "expected_tokens"),
    ((0.0, []), (0.2, [1])),
    ids=("no-penalty", "with-penalty"),
)
def test_parakeet_decoder_applies_blank_penalty(
    blank_penalty: float,
    expected_tokens: list[int],
) -> None:
    """Apply blank penalty while ranking joint token-duration candidates."""

    decoder = make_fake_parakeet_decoder()
    decoder.blank_penalty = blank_penalty

    class FakeContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-8.0, 0.0, 0.1]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array([[-8.0, 0.0]], dtype=np.float32)
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.ones(1, dtype=np.int32),
    )

    assert token_ids == [expected_tokens]
    assert timestamps == ([[0.0]] if expected_tokens else [[]])


@pytest.mark.cuda
def test_parakeet_decoder_keeps_batch_token_candidates_separate() -> None:
    """Use each utterance's token candidates instead of those from batch item zero."""

    decoder = make_fake_parakeet_decoder(batch_size=2)

    class FakeContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = cp.array(
                [[-10.0, 0.0, -2.0], [0.0, -10.0, -2.0]], dtype=np.float32
            )
            decoder.duration_log_probs[...] = cp.array(
                [[-2.0, 0.0], [-2.0, 0.0]], dtype=np.float32
            )
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((2, 1, 3), dtype=np.float32),
        cp.array([1, 1], dtype=np.int32),
    )

    assert token_ids == [[1], [0]]
    np.testing.assert_allclose(timestamps, [[0.0], [0.0]])


@pytest.mark.cuda
def test_parakeet_decoder_reuses_cuda_graph_across_frame_shapes() -> None:
    """Reuse one captured graph across compatible batch and frame dimensions."""

    decoder = make_fake_parakeet_decoder(batch_size=2)
    decoder.cuda_graph_supported = True
    token_log_probs = cp.array([[-10.0, 0.0, -10.0]], dtype=np.float32)
    duration_log_probs = cp.array([[-10.0, 0.0]], dtype=np.float32)

    class FakeContext:
        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs[...] = token_log_probs
            decoder.duration_log_probs[...] = duration_log_probs
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    decoder.decoder = FakeContext()
    encoder_output = cp.zeros((2, 20, 3), dtype=np.float32)
    first_token_ids, _ = decoder(encoder_output, cp.array([20, 15], dtype=np.int32))
    captured_graph = decoder.cuda_graph
    output_tokens_host = decoder.output_tokens_host
    assert captured_graph is not None
    assert output_tokens_host is not None

    second_encoder_output = encoder_output.reshape(-1)[:36].reshape(1, 12, 3)
    second_token_ids, _ = decoder(
        second_encoder_output,
        cp.array([12], dtype=np.int32),
    )

    assert [len(tokens) for tokens in first_token_ids] == [20, 15]
    assert len(second_token_ids[0]) == 12
    assert decoder.cuda_graph is captured_graph
    assert decoder.output_tokens_host is output_tokens_host


@pytest.mark.cuda
def test_parakeet_decoder_length_normalizes_completed_hypotheses() -> None:
    """Select the completed TDT history with the best normalized score."""

    decoder = make_fake_parakeet_decoder(beam=2)

    class FakeContext:
        def __init__(self) -> None:
            self.calls = 0

        def set_tensor_address(self, name: str, address: int) -> bool:
            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs.fill(-10.0)
            decoder.duration_log_probs.fill(-10.0)
            if self.calls == 0:
                decoder.token_log_probs[:, 1] = 0.0
                decoder.token_log_probs[:, decoder.blank_id] = 0.1
                decoder.duration_log_probs[:, 0] = -0.2
                decoder.duration_log_probs[:, 1] = -0.4
            else:
                decoder.token_log_probs[:, 0] = 0.0
                decoder.duration_log_probs[:, 1] = -0.15
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            self.calls += 1
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.array([1], dtype=np.int32),
    )

    assert token_ids == [[1, 0]]
    np.testing.assert_allclose(timestamps, [[0.0, 0.0]])


@pytest.mark.cuda
def test_parakeet_finalize_length_normalizes_active_hypotheses() -> None:
    """Select the best normalized active beam when no path completed."""

    output_tokens = cp.empty((1, 3), dtype=np.int32)
    output_timestamps = cp.empty((1, 3), dtype=np.float32)
    output_lengths = cp.empty(1, dtype=np.int32)

    TDT_FINALIZE_KERNEL(
        (1,),
        (1,),
        (
            cp.array([[-1.0, -1.2]], dtype=np.float32),
            cp.array([[0, 1]], dtype=np.int32),
            cp.array([[1, 2]], dtype=np.int32),
            cp.array([-np.inf], dtype=np.float32),
            cp.array([-1], dtype=np.int32),
            cp.array([0], dtype=np.int32),
            cp.array([-1, 0], dtype=np.int32),
            cp.array([3, 4], dtype=np.int32),
            cp.array([0.0, 0.08], dtype=np.float32),
            output_tokens,
            output_timestamps,
            output_lengths,
            np.int32(3),
            np.int32(2),
        ),
    )

    np.testing.assert_array_equal(output_lengths.get(), [2])
    np.testing.assert_array_equal(output_tokens.get()[0, :2], [3, 4])
    np.testing.assert_allclose(output_timestamps.get()[0, :2], [0.0, 0.08])


@pytest.mark.cuda
def test_parakeet_kernel_merges_duplicate_blank_histories() -> None:
    """Merge duplicate blank histories while retaining alternate time advances."""

    beam = 2
    duration_count = 1
    token_stride = 4
    threads = 256
    candidate_count = beam * (duration_count * beam + duration_count)
    shared_memory_bytes = (
        candidate_count * np.dtype(np.float32).itemsize
        + threads // 32 * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
        + beam * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )

    token_log_probs = cp.full((beam, 3), -100.0, dtype=np.float32)
    token_log_probs[:, 2] = 0.0
    duration_log_probs = cp.zeros((beam, duration_count), dtype=np.float32)
    top_token_scores = cp.full((beam, beam), -100.0, dtype=np.float32)
    top_token_indexes = cp.array([[0, 1], [0, 1]], dtype=np.int32)
    scores = cp.array([[0.0, -0.2]], dtype=np.float32)
    nodes = cp.full((1, beam), -1, dtype=np.int32)
    hashes = cp.zeros((1, beam), dtype=np.uint64)
    lengths = cp.zeros((1, beam), dtype=np.int32)
    time_indexes = cp.zeros(beam, dtype=np.int32)
    last_tokens = cp.full(beam, 2, dtype=np.int32)
    symbols = cp.zeros(beam, dtype=np.int32)
    next_scores = cp.empty_like(scores)
    next_nodes = cp.empty_like(nodes)
    next_hashes = cp.empty_like(hashes)
    next_lengths = cp.empty_like(lengths)
    next_time_indexes = cp.empty_like(time_indexes)
    next_last_tokens = cp.empty_like(last_tokens)
    next_symbols = cp.empty_like(symbols)
    parent_indexes = cp.empty(beam, dtype=np.int32)
    use_output_state = cp.empty(beam, dtype=np.uint8)
    node_parents = cp.empty(beam * token_stride, dtype=np.int32)
    node_tokens = cp.empty(beam * token_stride, dtype=np.int32)
    node_timestamps = cp.empty(beam * token_stride, dtype=np.float32)
    node_counts = cp.zeros(1, dtype=np.int32)
    completed_scores = cp.full(1, -np.inf, dtype=np.float32)
    completed_nodes = cp.full(1, -1, dtype=np.int32)
    completed_lengths = cp.zeros(1, dtype=np.int32)
    active_flags = cp.empty(1, dtype=np.int32)
    output_lengths = cp.array([2], dtype=np.int32)
    durations = cp.array([1], dtype=np.int32)
    positive_duration_indexes = cp.array([0], dtype=np.int32)
    state_1 = cp.zeros((1, beam, 4), dtype=np.float32)
    state_2 = cp.zeros_like(state_1)
    output_state_1 = cp.ones_like(state_1)
    output_state_2 = cp.ones_like(state_2)
    next_state_1 = cp.empty_like(state_1)
    next_state_2 = cp.empty_like(state_2)
    encoder_output = cp.zeros((1, 2, 4), dtype=np.float32)
    encoder_input = cp.empty((beam, 4), dtype=np.float32)
    targets = cp.empty((beam, 1), dtype=np.int32)
    runtime_dimensions = cp.array([1, 2], dtype=np.int32)

    TDT_BEAM_SEARCH_KERNEL(
        (1,),
        (threads,),
        (
            token_log_probs,
            duration_log_probs,
            top_token_scores,
            top_token_indexes,
            scores,
            nodes,
            hashes,
            lengths,
            time_indexes,
            last_tokens,
            symbols,
            next_scores,
            next_nodes,
            next_hashes,
            next_lengths,
            next_time_indexes,
            next_last_tokens,
            next_symbols,
            parent_indexes,
            use_output_state,
            node_parents,
            node_tokens,
            node_timestamps,
            node_counts,
            completed_scores,
            completed_nodes,
            completed_lengths,
            active_flags,
            output_lengths,
            durations,
            positive_duration_indexes,
            state_1,
            state_2,
            output_state_1,
            output_state_2,
            next_state_1,
            next_state_2,
            encoder_output,
            encoder_input,
            targets,
            np.int32(4),
            np.int32(1),
            runtime_dimensions,
            np.int32(4),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(token_stride),
            np.int32(beam),
            np.int32(duration_count),
            np.int32(1),
            np.int32(2),
            np.int32(10),
            np.float32(0.0),
            np.float32(0.08),
        ),
        shared_mem=shared_memory_bytes,
    )

    np.testing.assert_allclose(
        next_scores.get()[0, 0],
        np.logaddexp(np.float32(0.0), np.float32(-0.2)),
    )
    np.testing.assert_array_equal(next_lengths.get(), [[0, 0]])
    np.testing.assert_array_equal(next_time_indexes.get(), [1, 2])
    np.testing.assert_array_equal(active_flags.get(), [1])


@pytest.mark.cuda
def test_parakeet_kernel_preserves_distinct_active_symbol_counts() -> None:
    """Keep active TDT states whose future force-advance behavior differs."""

    beam = 2
    duration_count = 2
    token_stride = 4
    threads = 256
    candidate_count = beam * (duration_count * beam + 1)
    shared_memory_bytes = (
        candidate_count * np.dtype(np.float32).itemsize
        + threads // 32 * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
        + beam * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )

    token_log_probs = cp.full((beam, 3), -100.0, dtype=np.float32)
    duration_log_probs = cp.array([[-100.0, 0.0], [0.0, -100.0]], dtype=np.float32)
    top_token_scores = cp.array([[0.0, -100.0], [-0.1, -100.0]], dtype=np.float32)
    top_token_indexes = cp.array([[1, 0], [1, 0]], dtype=np.int32)
    scores = cp.zeros((1, beam), dtype=np.float32)
    nodes = cp.full((1, beam), -1, dtype=np.int32)
    hashes = cp.full((1, beam), 7, dtype=np.uint64)
    lengths = cp.ones((1, beam), dtype=np.int32)
    time_indexes = cp.array([0, 1], dtype=np.int32)
    last_tokens = cp.full(beam, 1, dtype=np.int32)
    symbols = cp.array([0, 0], dtype=np.int32)
    next_scores = cp.empty_like(scores)
    next_nodes = cp.empty_like(nodes)
    next_hashes = cp.empty_like(hashes)
    next_lengths = cp.empty_like(lengths)
    next_time_indexes = cp.empty_like(time_indexes)
    next_last_tokens = cp.empty_like(last_tokens)
    next_symbols = cp.empty_like(symbols)
    parent_indexes = cp.empty(beam, dtype=np.int32)
    use_output_state = cp.empty(beam, dtype=np.uint8)
    node_parents = cp.empty(beam * token_stride, dtype=np.int32)
    node_tokens = cp.empty(beam * token_stride, dtype=np.int32)
    node_timestamps = cp.empty(beam * token_stride, dtype=np.float32)
    node_counts = cp.zeros(1, dtype=np.int32)
    completed_scores = cp.full(1, -np.inf, dtype=np.float32)
    completed_nodes = cp.full(1, -1, dtype=np.int32)
    completed_lengths = cp.zeros(1, dtype=np.int32)
    active_flags = cp.empty(1, dtype=np.int32)
    output_lengths = cp.array([3], dtype=np.int32)
    durations = cp.array([0, 1], dtype=np.int32)
    positive_duration_indexes = cp.array([1], dtype=np.int32)
    state_1 = cp.zeros((1, beam, 4), dtype=np.float32)
    state_2 = cp.zeros_like(state_1)
    output_state_1 = cp.ones_like(state_1)
    output_state_2 = cp.ones_like(state_2)
    next_state_1 = cp.empty_like(state_1)
    next_state_2 = cp.empty_like(state_2)
    encoder_output = cp.zeros((1, 3, 4), dtype=np.float32)
    encoder_input = cp.empty((beam, 4), dtype=np.float32)
    targets = cp.empty((beam, 1), dtype=np.int32)
    runtime_dimensions = cp.array([1, 3], dtype=np.int32)

    TDT_BEAM_SEARCH_KERNEL(
        (1,),
        (threads,),
        (
            token_log_probs,
            duration_log_probs,
            top_token_scores,
            top_token_indexes,
            scores,
            nodes,
            hashes,
            lengths,
            time_indexes,
            last_tokens,
            symbols,
            next_scores,
            next_nodes,
            next_hashes,
            next_lengths,
            next_time_indexes,
            next_last_tokens,
            next_symbols,
            parent_indexes,
            use_output_state,
            node_parents,
            node_tokens,
            node_timestamps,
            node_counts,
            completed_scores,
            completed_nodes,
            completed_lengths,
            active_flags,
            output_lengths,
            durations,
            positive_duration_indexes,
            state_1,
            state_2,
            output_state_1,
            output_state_2,
            next_state_1,
            next_state_2,
            encoder_output,
            encoder_input,
            targets,
            np.int32(4),
            np.int32(1),
            runtime_dimensions,
            np.int32(4),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(token_stride),
            np.int32(beam),
            np.int32(duration_count),
            np.int32(1),
            np.int32(2),
            np.int32(10),
            np.float32(0.0),
            np.float32(0.08),
        ),
        shared_mem=shared_memory_bytes,
    )

    np.testing.assert_allclose(next_scores.get(), [[0.0, -0.1]])
    np.testing.assert_array_equal(next_lengths.get(), [[2, 2]])
    np.testing.assert_array_equal(next_time_indexes.get(), [1, 1])
    np.testing.assert_array_equal(next_symbols.get(), [0, 1])
    np.testing.assert_array_equal(active_flags.get(), [1])


class NullCudaContext:
    """Provide CUDA device and stream protocols without accessing a GPU."""

    attributes = {"MultiProcessorCount": 1}
    ptr = 117

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id

    def __enter__(self) -> "NullCudaContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


class FakeCudaArray:
    """Expose array metadata for validation paths without allocating storage."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=True)


class FakeDeviceBuffer:
    """Expose a stable device pointer without allocating CUDA memory."""

    def __init__(self, pointer: int) -> None:
        self.data = SimpleNamespace(ptr=pointer)


class RejectedProfileContext:
    """Record and reject TensorRT optimization-profile selection."""

    def __init__(self) -> None:
        self.profile_calls: list[tuple[int, int]] = []

    def set_optimization_profile_async(
        self, profile_index: int, stream_pointer: int
    ) -> bool:
        self.profile_calls.append((profile_index, stream_pointer))
        return False


class FakeParakeetEngine:
    """Expose only Parakeet metadata queried before context initialization."""

    shapes = {
        "encoder_output": (2, 4),
        "targets": (2, 1),
        "input_states_1": (1, 2, 3),
        "token_log_probs": (2, 8),
        "duration_log_probs": (2, 2),
    }
    dtypes = {
        "encoder_output": trt.float32,
        "input_states_1": trt.float32,
    }

    def __init__(self, context: RejectedProfileContext | None) -> None:
        self.context = context
        self.shape_requests: list[str] = []
        self.dtype_requests: list[str] = []

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        self.shape_requests.append(name)
        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        self.dtype_requests.append(name)
        return self.dtypes[name]

    def create_execution_context(self) -> RejectedProfileContext | None:
        return self.context


def make_parakeet_validation_decoder() -> ParakeetModifiedBeamSearchDecoder:
    """Construct only the state needed by pre-CUDA Parakeet validation."""

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    decoder.batch_size = 2
    decoder.decoder_capacity = 2
    decoder.encoder_dim = 4
    decoder.max_symbols_per_timestep = 1
    return decoder


def construct_parakeet_decoder(stream: NullCudaContext) -> None:
    """Initialize a Parakeet decoder through its context-setup boundary."""

    ParakeetModifiedBeamSearchDecoder(
        Path("decoder.trt"),
        batch_size=2,
        blank_id=7,
        durations=(0, 1),
        max_symbols_per_timestep=10,
        encoder_frame_shift_sec=0.08,
        blank_penalty=0.0,
        device_id=0,
        stream=cast(cp.cuda.Stream, stream),
    )


def test_parakeet_decoder_swaps_all_search_buffers_and_rebinds_states() -> None:
    """Swap every paired search buffer and rebind both recurrent inputs."""

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    original_buffers: dict[str, FakeDeviceBuffer] = {}
    for pointer, name in enumerate(
        (name for pair in PARAKEET_SEARCH_BUFFER_PAIRS for name in pair), start=1
    ):
        buffer = FakeDeviceBuffer(pointer)
        setattr(decoder, name, buffer)
        original_buffers[name] = buffer

    class RecordingContext:
        def __init__(self) -> None:
            self.bindings: dict[str, int] = {}

        def set_tensor_address(self, name: str, address: int) -> bool:
            self.bindings[name] = address
            return True

    context = RecordingContext()
    decoder.decoder = cast(trt.IExecutionContext, context)
    decoder.swap_buffers()

    for current_name, next_name in PARAKEET_SEARCH_BUFFER_PAIRS:
        assert getattr(decoder, current_name) is original_buffers[next_name]
        assert getattr(decoder, next_name) is original_buffers[current_name]
    assert context.bindings == {
        "input_states_1": decoder.state_1.data.ptr,
        "input_states_2": decoder.state_2.data.ptr,
    }


@pytest.mark.parametrize("rejected_name", ("input_states_1", "input_states_2"))
def test_parakeet_decoder_attempts_both_recurrent_bindings(
    rejected_name: str,
) -> None:
    """Attempt both recurrent bindings before reporting either rejection."""

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    for pointer, name in enumerate(
        (name for pair in PARAKEET_SEARCH_BUFFER_PAIRS for name in pair), start=1
    ):
        setattr(decoder, name, FakeDeviceBuffer(pointer))

    class RejectingContext:
        def __init__(self) -> None:
            self.binding_names: list[str] = []

        def set_tensor_address(self, name: str, address: int) -> bool:
            assert address > 0
            self.binding_names.append(name)
            return name != rejected_name

    context = RejectingContext()
    decoder.decoder = cast(trt.IExecutionContext, context)

    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder.swap_buffers()

    assert context.binding_names == ["input_states_1", "input_states_2"]


@pytest.mark.parametrize(
    ("encoder_output", "encoder_output_lengths", "message"),
    (
        pytest.param(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            "rank-3",
            id="rank",
        ),
        pytest.param(
            np.zeros((0, 3, 4), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            "batch capacity",
            id="empty-batch",
        ),
        pytest.param(
            np.zeros((3, 3, 4), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            "batch capacity",
            id="oversized-batch",
        ),
        pytest.param(
            np.zeros((2, 3, 5), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            "dimension 4",
            id="encoder-dimension",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.int32),
            np.zeros(2, dtype=np.int32),
            "float16, float32, or bfloat16",
            id="encoder-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 8), dtype=np.float32)[:, :, ::2],
            np.zeros(2, dtype=np.int32),
            "contiguous",
            id="noncontiguous-encoder",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            "encoder output lengths",
            id="length-shape",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            "int32 encoder output lengths",
            id="length-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(4, dtype=np.int32)[::2],
            "contiguous int32 encoder output lengths",
            id="noncontiguous-lengths",
        ),
    ),
)
def test_parakeet_decoder_rejects_malformed_inputs(
    encoder_output: np.ndarray,
    encoder_output_lengths: np.ndarray,
    message: str,
) -> None:
    """Reject malformed Parakeet encoder outputs before CUDA execution."""

    decoder = make_parakeet_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, encoder_output_lengths),
        )


def test_parakeet_decoder_rejects_int32_history_capacity_overflow() -> None:
    """Reject search histories that exceed int32 kernel indexing."""

    decoder = make_parakeet_validation_decoder()
    decoder.max_symbols_per_timestep = INT32_MAX
    encoder_output = np.zeros((2, 2, 4), dtype=np.float32)
    output_lengths = np.full(2, 2, dtype=np.int32)

    with pytest.raises(ASRInferenceError, match="signed 32-bit kernel indexing"):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, output_lengths),
        )


def test_parakeet_decoder_rejects_int32_encoder_capacity_overflow() -> None:
    """Reject encoder tensors that exceed int32 kernel indexing."""

    decoder = make_parakeet_validation_decoder()
    max_frames = INT32_MAX // (decoder.batch_size * decoder.encoder_dim) + 1
    encoder_output = FakeCudaArray(
        (decoder.batch_size, max_frames, decoder.encoder_dim),
        np.dtype(np.float32),
    )
    output_lengths = FakeCudaArray((decoder.batch_size,), np.dtype(np.int32))

    with pytest.raises(ASRInferenceError, match="signed 32-bit kernel indexing"):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, output_lengths),
        )


def test_parakeet_decoder_rejects_missing_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a model-specific error when TensorRT cannot create a context."""

    stream = NullCudaContext()
    engine = FakeParakeetEngine(None)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeParakeetEngine:
        loaded_paths.append(engine_path)
        return engine

    monkeypatch.setattr(parakeet_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(parakeet_decoder, "get_engine", load_engine)

    with pytest.raises(ASRInitializationError) as error:
        construct_parakeet_decoder(stream)

    assert str(error.value) == (
        "TensorRT could not create the Parakeet decoder execution context."
    )
    assert loaded_paths == [Path("decoder.trt")]
    assert sorted(engine.shape_requests) == sorted(engine.shapes)
    assert sorted(engine.dtype_requests) == sorted(engine.dtypes)


def test_parakeet_decoder_rejects_profile_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a decoder context that cannot select optimization profile zero."""

    stream = NullCudaContext()
    context = RejectedProfileContext()
    engine = FakeParakeetEngine(context)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeParakeetEngine:
        loaded_paths.append(engine_path)
        return engine

    monkeypatch.setattr(parakeet_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(parakeet_decoder, "get_engine", load_engine)

    with pytest.raises(ASRInitializationError) as error:
        construct_parakeet_decoder(stream)

    assert str(error.value) == (
        "TensorRT could not select Parakeet decoder optimization profile 0."
    )
    assert loaded_paths == [Path("decoder.trt")]
    assert sorted(engine.shape_requests) == sorted(engine.shapes)
    assert sorted(engine.dtype_requests) == sorted(engine.dtypes)
    assert context.profile_calls == [(0, stream.ptr)]
