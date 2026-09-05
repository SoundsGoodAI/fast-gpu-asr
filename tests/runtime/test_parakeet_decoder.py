#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Runtime and validation tests for the Parakeet TDT decoder."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast
from unittest.mock import Mock

import cupy as cp
import cupyx as cpx
import numpy as np
import pytest
import tensorrt as trt

from fast_gpu_asr.constants import INT32_MAX, TDT_SEARCH_CHUNK_STEPS
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


def expected_token_selection_shared_memory_bytes(vocab_size: int, threads: int) -> int:
    """Return storage for token scores and one reduction pair per warp.

    Parameters
    ----------
    vocab_size : int
        Number of nonblank token scores reduced by each block.
    threads : int
        Number of CUDA threads in the token-selection block.

    Returns
    -------
    int
        Required dynamic shared-memory size in bytes.
    """

    return vocab_size * np.dtype(np.float32).itemsize + threads // 32 * (
        np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize
    )


def expected_beam_search_shared_memory_bytes(
    beam: int, duration_count: int, positive_duration_count: int, threads: int
) -> int:
    """Return storage for expanded candidates, reductions, and selected pairs.

    Parameters
    ----------
    beam : int
        Number of active hypotheses retained per utterance.
    duration_count : int
        Total number of TDT duration classes.
    positive_duration_count : int
        Number of duration classes that advance encoder time.
    threads : int
        Number of CUDA threads in the beam-search block.

    Returns
    -------
    int
        Required dynamic shared-memory size in bytes.
    """

    candidate_count = beam * (duration_count * beam + positive_duration_count)
    return (
        candidate_count * np.dtype(np.float32).itemsize
        + threads // 32 * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
        + beam * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )


@pytest.mark.cuda
def test_parakeet_token_selection_scans_full_vocabulary() -> None:
    beam = 2
    threads = 512
    vocab_size = threads + 7
    token_log_probs = cp.full((1, vocab_size + 1), -10.0, dtype=np.float32)
    token_log_probs[0, threads + 5] = 0.75
    token_log_probs[0, threads + 6] = 1.25
    top_token_scores = cp.empty((1, beam), dtype=np.float32)
    top_token_indexes = cp.empty((1, beam), dtype=np.int32)
    shared_memory_bytes = expected_token_selection_shared_memory_bytes(
        vocab_size, threads
    )

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
    np.testing.assert_array_equal(top_token_indexes.get(), [[threads + 6, threads + 5]])


@pytest.mark.cuda
def test_parakeet_token_selection_excludes_blank_column() -> None:
    vocab_size = 3
    beam = 2
    threads = 256
    top_token_scores = cp.empty((1, beam), dtype=np.float32)
    top_token_indexes = cp.empty((1, beam), dtype=np.int32)

    TDT_SELECT_TOKENS_KERNEL(
        (1,),
        (threads,),
        (
            cp.array([[0.25, 0.5, -1.0, 100.0]], dtype=np.float32),
            cp.array([0.0], dtype=np.float32),
            cp.array([0], dtype=np.int32),
            cp.array([1], dtype=np.int32),
            top_token_scores,
            top_token_indexes,
            np.int32(vocab_size),
            np.int32(beam),
        ),
        shared_mem=expected_token_selection_shared_memory_bytes(vocab_size, threads),
    )

    np.testing.assert_allclose(top_token_scores.get(), [[0.5, 0.25]])
    np.testing.assert_array_equal(top_token_indexes.get(), [[1, 0]])


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_parakeet_token_selection_keeps_nonfinite_indexes_in_bounds(
    invalid_score: float,
) -> None:
    vocab_size = 3
    beam = 2
    threads = 256
    top_token_scores = cp.empty((1, beam), dtype=np.float32)
    top_token_indexes = cp.empty((1, beam), dtype=np.int32)
    shared_memory_bytes = expected_token_selection_shared_memory_bytes(
        vocab_size, threads
    )

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
        pytest.param(
            cp.dtype("bfloat16"), np.int32(2), marks=pytest.mark.sm80, id="encoder-bf16"
        ),
    ),
)
@pytest.mark.parametrize(
    ("decoder_dtype", "decoder_dtype_code"),
    (
        pytest.param(np.dtype(np.float32), np.int32(0), id="decoder-fp32"),
        pytest.param(np.dtype(np.float16), np.int32(1), id="decoder-fp16"),
        pytest.param(
            cp.dtype("bfloat16"), np.int32(2), marks=pytest.mark.sm80, id="decoder-bf16"
        ),
    ),
)
def test_parakeet_prepare_inputs_converts_precision(
    encoder_dtype: np.dtype,
    encoder_dtype_code: np.int32,
    decoder_dtype: np.dtype,
    decoder_dtype_code: np.int32,
) -> None:
    encoder_output = (
        cp.arange(16, dtype=cp.float32).reshape(1, 2, 8).astype(encoder_dtype)
    )
    encoder_input = cp.full((2, 8), -1.0, dtype=decoder_dtype)
    targets = cp.full((2, 1), -1, dtype=cp.int32)
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


def make_fake_parakeet_decoder(
    beam: int = 1,
    batch_size: int = 1,
    durations: tuple[int, ...] = (0, 1),
    state_dtype: np.dtype | None = None,
    state_hidden_dim: int = 3,
    state_layers: int = 1,
) -> ParakeetModifiedBeamSearchDecoder:
    """Create a Parakeet decoder without loading TensorRT.

    Parameters
    ----------
    beam : int
        Number of active hypotheses retained per utterance.
    batch_size : int
        Maximum number of utterances decoded together.
    durations : tuple[int, ...]
        TDT encoder-frame advances indexed by duration score.
    state_dtype : np.dtype | None
        Recurrent-state dtype; defaults to ``np.float32``.
    state_hidden_dim : int
        Width of each recurrent-state vector.
    state_layers : int
        Number of recurrent-state layers.

    Returns
    -------
    ParakeetModifiedBeamSearchDecoder
        Decoder with CUDA search buffers initialized and no TensorRT context.
    """

    if state_dtype is None:
        state_dtype = np.dtype(np.float32)

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    decoder.device = cp.cuda.Device(0)
    decoder.batch_size = batch_size
    decoder.beam = beam
    decoder.decoder_capacity = batch_size * beam
    decoder.encoder_dim = 3
    decoder.blank_id = 2
    decoder.durations_array = cp.array(durations, dtype=np.int32)
    decoder.positive_duration_indexes_array = cp.array(
        [index for index, duration in enumerate(durations) if duration > 0],
        dtype=np.int32,
    )
    decoder.max_symbols_per_timestep = 10
    decoder.encoder_frame_shift_sec = 0.08
    decoder.blank_penalty = 0.0
    decoder.state_layers = state_layers
    decoder.state_hidden_dim = state_hidden_dim
    decoder.kernel_dtype_map = {
        np.dtype(np.float32): np.int32(0),
        np.dtype(np.float16): np.int32(1),
        cp.dtype("bfloat16"): np.int32(2),
    }
    decoder.state_dtype = decoder.kernel_dtype_map[state_dtype]
    decoder.encoder_input_dtype = np.int32(0)
    decoder.prepare_inputs_threads = 256
    decoder.token_selection_threads = 512
    decoder.beam_search_threads = 256
    decoder.stream = cp.cuda.get_current_stream()
    decoder.beam_search_shared_memory_bytes = expected_beam_search_shared_memory_bytes(
        beam,
        decoder.durations_array.size,
        decoder.positive_duration_indexes_array.size,
        decoder.beam_search_threads,
    )
    decoder.token_selection_shared_memory_bytes = (
        expected_token_selection_shared_memory_bytes(
            decoder.blank_id, decoder.token_selection_threads
        )
    )
    decoder.cuda_graph = None
    decoder.cuda_graph_signature = None
    decoder.cuda_graph_supported = False

    capacity = decoder.decoder_capacity
    search_shape = (batch_size, beam)
    decoder.encoder_input = cp.empty((capacity, 3), dtype=np.float32)
    decoder.targets = cp.empty((capacity, 1), dtype=np.int32)
    decoder.state_1 = cp.empty(
        (state_layers, capacity, state_hidden_dim), dtype=state_dtype
    )
    decoder.state_2 = cp.empty(
        (state_layers, capacity, state_hidden_dim), dtype=state_dtype
    )
    decoder.token_log_probs = cp.empty(
        (capacity, decoder.blank_id + 1), dtype=np.float32
    )
    decoder.duration_log_probs = cp.empty((capacity, len(durations)), dtype=np.float32)
    decoder.output_state_1 = cp.empty_like(decoder.state_1)
    decoder.output_state_2 = cp.empty_like(decoder.state_2)
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


def get_parakeet_search_buffers(
    decoder: ParakeetModifiedBeamSearchDecoder,
) -> dict[str, cp.ndarray]:
    """Return every current and next search buffer by attribute name.

    Parameters
    ----------
    decoder : ParakeetModifiedBeamSearchDecoder
        Decoder whose ping-pong buffers are inspected.

    Returns
    -------
    dict[str, cp.ndarray]
        Mapping from each current and next buffer name to its CUDA array.
    """

    return {
        name: cast(cp.ndarray, getattr(decoder, name))
        for pair in PARAKEET_SEARCH_BUFFER_PAIRS
        for name in pair
    }


class RuntimeDecoderContext:
    """Run a fake TensorRT decoder through a supplied per-step callback."""

    def __init__(
        self,
        decoder: ParakeetModifiedBeamSearchDecoder,
        execute: Callable[[int], bool],
        rejected_binding_call: int | None = None,
    ) -> None:
        """Initialize a callback-backed TensorRT execution context.

        Parameters
        ----------
        decoder : ParakeetModifiedBeamSearchDecoder
            Decoder whose stream and output buffers are used by the callback.
        execute : Callable[[int], bool]
            Function invoked with the zero-based decoder call index.
        rejected_binding_call : int | None
            One-based recurrent-state binding call that should fail, when set.
        """

        self.decoder = decoder
        self.execute = execute
        self.rejected_binding_call = rejected_binding_call
        self.binding_names: list[str] = []
        self.calls = 0

    def set_tensor_address(self, name: str, address: int) -> bool:
        """Record a recurrent-state binding and optionally reject it.

        Parameters
        ----------
        name : str
            Recurrent-state tensor name being rebound.
        address : int
            CUDA device address assigned to the tensor.

        Returns
        -------
        bool
            ``False`` only for the configured binding-call index.
        """

        assert name in {"input_states_1", "input_states_2"}
        assert address > 0
        self.binding_names.append(name)
        return len(self.binding_names) != self.rejected_binding_call

    def execute_async_v3(self, stream_ptr: int) -> bool:
        """Invoke the scripted decoder step on the expected CUDA stream.

        Parameters
        ----------
        stream_ptr : int
            CUDA stream pointer supplied by the decoder.

        Returns
        -------
        bool
            Execution result returned by the configured callback.
        """

        assert stream_ptr == self.decoder.stream.ptr
        result = self.execute(self.calls)
        self.calls += 1
        return result


def install_static_decoder_context(
    decoder: ParakeetModifiedBeamSearchDecoder,
    token_scores: list[float] | list[list[float]],
    duration_scores: list[float] | list[list[float]],
) -> RuntimeDecoderContext:
    """Install a context that returns fixed scores and recurrent states.

    Parameters
    ----------
    decoder : ParakeetModifiedBeamSearchDecoder
        Decoder receiving the fake TensorRT context.
    token_scores : list[float] | list[list[float]]
        Token scores for one row or every decoder-capacity row.
    duration_scores : list[float] | list[list[float]]
        Duration scores for one row or every decoder-capacity row.

    Returns
    -------
    RuntimeDecoderContext
        Installed callback-backed TensorRT execution context.
    """

    token_scores_array = cp.array(token_scores, dtype=np.float32)
    duration_scores_array = cp.array(duration_scores, dtype=np.float32)
    if token_scores_array.ndim == 1:
        token_scores_array = cp.broadcast_to(
            token_scores_array, (decoder.decoder_capacity, token_scores_array.size)
        )
    if duration_scores_array.ndim == 1:
        duration_scores_array = cp.broadcast_to(
            duration_scores_array,
            (decoder.decoder_capacity, duration_scores_array.size),
        )

    def execute(_call: int) -> bool:
        """Write fixed scores and preserve recurrent state for one step.

        Parameters
        ----------
        _call : int
            Zero-based invocation accepted for the callback contract.

        Returns
        -------
        bool
            Always ``True`` to report successful TensorRT execution.
        """

        decoder.token_log_probs[...] = token_scores_array
        decoder.duration_log_probs[...] = duration_scores_array
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    context = RuntimeDecoderContext(decoder, execute)
    decoder.decoder = context
    return context


@pytest.mark.cuda
def test_parakeet_decoder_rejects_recurrent_state_binding_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    execute = Mock(side_effect=AssertionError("decoder execution must not start"))
    context = RuntimeDecoderContext(decoder, execute, rejected_binding_call=1)
    decoder.decoder = context

    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32))
    decoder.stream.synchronize()

    assert context.binding_names == ["input_states_1", "input_states_2"]
    execute.assert_not_called()


@pytest.mark.cuda
def test_parakeet_decoder_restores_buffers_after_runtime_binding_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    original_buffers = get_parakeet_search_buffers(decoder)
    context = install_static_decoder_context(decoder, [-10.0, 0.0, -2.0], [-2.0, 0.0])
    context.rejected_binding_call = 3

    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32))
    decoder.stream.synchronize()

    assert len(context.binding_names) == 6
    for name, original_buffer in original_buffers.items():
        assert getattr(decoder, name) is original_buffer


@pytest.mark.cuda
def test_parakeet_decoder_returns_empty_results_for_zero_frames() -> None:
    decoder = make_fake_parakeet_decoder(batch_size=2)

    token_ids, timestamps = decoder(
        cp.empty((1, 0, 3), dtype=np.float32), cp.zeros(1, dtype=np.int32)
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_reports_tensorrt_execution_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.decoder = RuntimeDecoderContext(decoder, lambda _call: False)

    with pytest.raises(ASRInferenceError, match="TensorRT decoder execution failed"):
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32))
    decoder.stream.synchronize()


@pytest.mark.cuda
def test_parakeet_decoder_propagates_non_capture_driver_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = make_fake_parakeet_decoder()

    class FakeDriverError(RuntimeError):
        """Expose a CUDA driver status through a deterministic test error."""

        def __init__(self, status: int) -> None:
            """Initialize the error from one CUDA driver status.

            Parameters
            ----------
            status : int
                CUDA driver error code exposed through ``status``.
            """

            super().__init__(f"CUDA driver error {status}")
            self.status = status

    failure = FakeDriverError(901)
    failing_kernel = Mock(side_effect=failure)
    monkeypatch.setattr(
        parakeet_decoder.cp.cuda.driver, "CUDADriverError", FakeDriverError
    )
    monkeypatch.setattr(parakeet_decoder, "TDT_SELECT_TOKENS_KERNEL", failing_kernel)
    install_static_decoder_context(decoder, [-10.0, 0.0, -2.0], [-2.0, 0.0])

    with pytest.raises(FakeDriverError) as error:
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32))
    decoder.stream.synchronize()

    assert error.value is failure
    failing_kernel.assert_called_once()


@pytest.mark.cuda
def test_parakeet_decoder_restores_buffers_after_execution_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.max_symbols_per_timestep = 1
    original_buffers = get_parakeet_search_buffers(decoder)

    def execute(call: int) -> bool:
        """Succeed once, then simulate TensorRT execution failure.

        Parameters
        ----------
        call : int
            Zero-based decoder invocation index.

        Returns
        -------
        bool
            ``True`` for the first invocation and ``False`` thereafter.
        """

        if call:
            return False
        decoder.token_log_probs[...] = cp.array([[-10.0, 0.0, -2.0]], dtype=np.float32)
        decoder.duration_log_probs[...] = cp.array([[0.0, -10.0]], dtype=np.float32)
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    with pytest.raises(ASRInferenceError, match="TensorRT decoder execution failed"):
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32))
    decoder.stream.synchronize()

    for name, original_buffer in original_buffers.items():
        assert getattr(decoder, name) is original_buffer


@pytest.mark.cuda
def test_parakeet_decoder_restores_buffers_after_captured_binding_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.cuda_graph_supported = True
    decoder.max_symbols_per_timestep = 20
    original_buffers = get_parakeet_search_buffers(decoder)

    class RejectingContext:
        """Reject one recurrent-state binding during CUDA graph capture."""

        def __init__(self) -> None:
            """Initialize captured binding history."""

            self.capture_bindings: list[str] = []

        def set_tensor_address(self, name: str, address: int) -> bool:
            """Record capture-time bindings and reject the first state input.

            Parameters
            ----------
            name : str
                Recurrent-state tensor name being rebound.
            address : int
                CUDA device address assigned to the tensor.

            Returns
            -------
            bool
                ``False`` for ``input_states_1`` during capture, otherwise ``True``.
            """

            assert name in {"input_states_1", "input_states_2"}
            assert address > 0
            if cp.cuda.runtime.streamIsCapturing(decoder.stream.ptr):
                self.capture_bindings.append(name)
                return name != "input_states_1"
            return True

        def execute_async_v3(self, stream_ptr: int) -> bool:
            """Populate deterministic decoder outputs on the expected stream.

            Parameters
            ----------
            stream_ptr : int
                CUDA stream pointer supplied by the decoder.

            Returns
            -------
            bool
                Always ``True`` after writing scores and recurrent states.
            """

            assert stream_ptr == decoder.stream.ptr
            decoder.token_log_probs.fill(-10.0)
            decoder.token_log_probs[:, 1] = 0.0
            decoder.duration_log_probs.fill(-10.0)
            decoder.duration_log_probs[:, 0] = 0.0
            decoder.output_state_1[...] = decoder.state_1
            decoder.output_state_2[...] = decoder.state_2
            return True

    context = RejectingContext()
    decoder.decoder = context
    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder(cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32))
    decoder.stream.synchronize()

    assert context.capture_bindings == ["input_states_1", "input_states_2"]
    for name, original_buffer in original_buffers.items():
        assert getattr(decoder, name) is original_buffer


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_parakeet_decoder_handles_nonfinite_search_scores(invalid_score: float) -> None:
    decoder = make_fake_parakeet_decoder(beam=2)
    install_static_decoder_context(
        decoder, [[invalid_score] * 3] * 2, [[invalid_score] * 2] * 2
    )
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32)
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_emits_token_with_positive_duration() -> None:
    decoder = make_fake_parakeet_decoder()
    install_static_decoder_context(decoder, [-10.0, 0.0, -2.0], [-2.0, 0.0])
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32)
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.0]])


@pytest.mark.cuda
def test_parakeet_decoder_timestamps_token_after_blank_advance() -> None:
    decoder = make_fake_parakeet_decoder(durations=(0, 1, 2))

    def execute(call: int) -> bool:
        """Script a blank advance followed by one token emission.

        Parameters
        ----------
        call : int
            Zero-based decoder invocation index.

        Returns
        -------
        bool
            Always ``True`` after writing the invocation-specific scores.
        """

        decoder.token_log_probs.fill(-1000.0 if call == 0 else -10.0)
        decoder.duration_log_probs.fill(-10.0)
        if call == 0:
            decoder.token_log_probs[:, decoder.blank_id] = 0.0
            decoder.duration_log_probs[:, 0] = 100.0
            decoder.duration_log_probs[:, 2] = 0.0
        else:
            decoder.token_log_probs[:, 1] = 0.0
            decoder.duration_log_probs[:, 1] = 0.0
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    token_ids, timestamps = decoder(
        cp.zeros((1, 3, 3), dtype=np.float32), cp.array([3], dtype=np.int32)
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.16]])


@pytest.mark.cuda
def test_parakeet_decoder_does_not_leak_history_across_calls() -> None:
    decoder = make_fake_parakeet_decoder()
    emit_token = True

    def execute(_call: int) -> bool:
        """Select a token or blank according to the enclosing test state.

        Parameters
        ----------
        _call : int
            Zero-based invocation accepted for the callback contract.

        Returns
        -------
        bool
            Always ``True`` after writing deterministic scores and states.
        """

        decoder.token_log_probs.fill(-10.0)
        decoder.token_log_probs[:, 1 if emit_token else decoder.blank_id] = 0.0
        decoder.duration_log_probs.fill(-10.0)
        decoder.duration_log_probs[:, 1] = 0.0
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    encoder_output = cp.zeros((1, 1, 3), dtype=np.float32)
    output_lengths = cp.ones(1, dtype=np.int32)

    first_token_ids, first_timestamps = decoder(encoder_output, output_lengths)
    emit_token = False
    second_token_ids, second_timestamps = decoder(encoder_output, output_lengths)

    assert first_token_ids == [[1]]
    np.testing.assert_allclose(first_timestamps, [[0.0]])
    assert second_token_ids == [[]]
    assert second_timestamps == [[]]


@pytest.mark.cuda
def test_parakeet_decoder_uses_configured_duration_values() -> None:
    decoder = make_fake_parakeet_decoder(durations=(0, 2))
    encoder_inputs: list[np.typing.NDArray[np.float32]] = []

    def execute(_call: int) -> bool:
        """Record prepared encoder input and emit a two-frame token.

        Parameters
        ----------
        _call : int
            Zero-based invocation accepted for the callback contract.

        Returns
        -------
        bool
            Always ``True`` after writing deterministic scores and states.
        """

        encoder_inputs.append(decoder.encoder_input.get())
        decoder.token_log_probs[...] = cp.array([[-10.0, 0.0, -2.0]], dtype=np.float32)
        decoder.duration_log_probs[...] = cp.array([[-10.0, 0.0]], dtype=np.float32)
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    encoder_output = cp.arange(9, dtype=np.float32).reshape(1, 3, 3)
    token_ids, timestamps = decoder(encoder_output, cp.array([3], dtype=np.int32))

    assert token_ids == [[1, 1]]
    np.testing.assert_allclose(timestamps, [[0.0, 0.16]])
    assert len(encoder_inputs) == TDT_SEARCH_CHUNK_STEPS
    np.testing.assert_array_equal(encoder_inputs[0][0], encoder_output[0, 0].get())
    np.testing.assert_array_equal(encoder_inputs[1][0], encoder_output[0, 2].get())


@pytest.mark.cuda
def test_parakeet_decoder_clamps_runtime_output_lengths() -> None:
    decoder = make_fake_parakeet_decoder(batch_size=2)
    install_static_decoder_context(
        decoder, [[-10.0, 0.0, -2.0]] * 2, [[-10.0, 0.0]] * 2
    )
    token_ids, timestamps = decoder(
        cp.zeros((2, 2, 3), dtype=np.float32), cp.array([-1, 10], dtype=np.int32)
    )

    assert token_ids == [[], [1, 1]]
    assert timestamps[0] == []
    np.testing.assert_allclose(timestamps[1], [0.0, 0.08])


@pytest.mark.cuda
def test_parakeet_decoder_clears_inactive_partial_batch_inputs() -> None:
    decoder = make_fake_parakeet_decoder(beam=2, batch_size=3)
    encoder_inputs: list[np.typing.NDArray[np.float32]] = []
    target_inputs: list[np.typing.NDArray[np.int32]] = []

    def execute(call: int) -> bool:
        """Record first-step inputs and advance every active hypothesis.

        Parameters
        ----------
        call : int
            Zero-based decoder invocation index.

        Returns
        -------
        bool
            Always ``True`` after writing blank and duration scores.
        """

        if call == 0:
            encoder_inputs.append(decoder.encoder_input.get())
            target_inputs.append(decoder.targets.get())
        decoder.token_log_probs.fill(-10.0)
        decoder.token_log_probs[:, decoder.blank_id] = 0.0
        decoder.duration_log_probs.fill(-10.0)
        decoder.duration_log_probs[:, 1] = 0.0
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    decoder.encoder_input.fill(-1.0)
    decoder.targets.fill(-1)
    token_ids, timestamps = decoder(
        cp.array([[[1.0, 2.0, 3.0]]], dtype=np.float32), cp.array([1], dtype=np.int32)
    )

    assert token_ids == [[]]
    assert timestamps == [[]]
    np.testing.assert_array_equal(encoder_inputs[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(encoder_inputs[0][1:], 0.0)
    np.testing.assert_array_equal(
        target_inputs[0], np.array([[2], [0], [0], [0], [0], [0]], dtype=np.int32)
    )
    np.testing.assert_array_equal(decoder.search_output_lengths.get(), [1, 0, 0])


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("state_dtype", "state_hidden_dim", "state_layers"),
    (
        pytest.param(np.dtype(np.float32), 3, 1, id="fp32-scalar"),
        pytest.param(np.dtype(np.float32), 4, 2, id="fp32-packed-multilayer"),
        pytest.param(np.dtype(np.float16), 3, 1, id="fp16-scalar"),
        pytest.param(np.dtype(np.float16), 8, 2, id="fp16-packed-multilayer"),
        pytest.param(
            cp.dtype("bfloat16"), 3, 1, marks=pytest.mark.sm80, id="bf16-scalar"
        ),
        pytest.param(
            cp.dtype("bfloat16"),
            8,
            2,
            marks=pytest.mark.sm80,
            id="bf16-packed-multilayer",
        ),
    ),
)
def test_parakeet_decoder_routes_recurrent_state_by_emission(
    state_dtype: np.dtype, state_hidden_dim: int, state_layers: int
) -> None:
    decoder = make_fake_parakeet_decoder(
        state_dtype=state_dtype,
        state_hidden_dim=state_hidden_dim,
        state_layers=state_layers,
    )
    state_values = cp.arange(decoder.state_1.size, dtype=np.float32).reshape(
        decoder.state_1.shape
    )

    state_snapshots: list[
        tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]
    ] = []

    def execute(call: int) -> bool:
        """Record recurrent state and alternate token and blank outputs.

        Parameters
        ----------
        call : int
            Zero-based decoder invocation index.

        Returns
        -------
        bool
            Always ``True`` after writing scores and replacement states.
        """

        state_snapshots.append(
            (
                decoder.state_1.astype(cp.float32).get(),
                decoder.state_2.astype(cp.float32).get(),
            )
        )
        decoder.token_log_probs.fill(-10.0)
        decoder.duration_log_probs.fill(-10.0)
        if call == 0:
            decoder.token_log_probs[:, 1] = 0.0
            decoder.duration_log_probs[:, 0] = 0.0
        else:
            decoder.token_log_probs[:, decoder.blank_id] = 0.0
            decoder.duration_log_probs[:, 1] = 0.0
        decoder.output_state_1[...] = state_values + 3.0 + call
        decoder.output_state_2[...] = state_values + 7.0 + call
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    token_ids, timestamps = decoder(
        cp.zeros((1, 2, 3), dtype=np.float32), cp.array([2], dtype=np.int32)
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.0]])
    assert len(state_snapshots) == TDT_SEARCH_CHUNK_STEPS
    np.testing.assert_array_equal(state_snapshots[0][0], 0.0)
    np.testing.assert_array_equal(state_snapshots[0][1], 0.0)
    expected_values = state_values.get()
    for snapshot in state_snapshots[1:3]:
        np.testing.assert_array_equal(snapshot[0], expected_values + 3.0)
        np.testing.assert_array_equal(snapshot[1], expected_values + 7.0)


@pytest.mark.cuda
def test_parakeet_decoder_blank_advances_without_emitting_token() -> None:
    decoder = make_fake_parakeet_decoder()
    context = install_static_decoder_context(decoder, [-10.0, -11.0, 0.0], [0.0, 0.0])
    token_ids, timestamps = decoder(
        cp.zeros((1, 2, 3), dtype=np.float32), cp.array([2], dtype=np.int32)
    )

    assert token_ids == [[]]
    assert timestamps == [[]]
    assert context.calls == TDT_SEARCH_CHUNK_STEPS


@pytest.mark.cuda
def test_parakeet_decoder_limits_zero_duration_token_emissions() -> None:
    decoder = make_fake_parakeet_decoder()
    original_buffers = get_parakeet_search_buffers(decoder)
    install_static_decoder_context(decoder, [-10.0, 0.0, -2.0], [0.0, -2.0])
    decoder.max_symbols_per_timestep = 2
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32)
    )

    assert token_ids == [[1, 1]]
    np.testing.assert_allclose(timestamps, [[0.0, 0.0]])
    for name, original_buffer in original_buffers.items():
        assert getattr(decoder, name) is original_buffer


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("blank_penalty", "expected_tokens"),
    ((0.0, []), (0.2, [1])),
    ids=("no-penalty", "with-penalty"),
)
def test_parakeet_decoder_applies_blank_penalty(
    blank_penalty: float, expected_tokens: list[int]
) -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.blank_penalty = blank_penalty
    install_static_decoder_context(decoder, [-8.0, 0.0, 0.1], [-8.0, 0.0])
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32)
    )

    assert token_ids == [expected_tokens]
    assert timestamps == ([[0.0]] if expected_tokens else [[]])


@pytest.mark.cuda
def test_parakeet_decoder_keeps_batch_token_candidates_separate() -> None:
    decoder = make_fake_parakeet_decoder(batch_size=2)
    install_static_decoder_context(
        decoder, [[-10.0, 0.0, -2.0], [0.0, -10.0, -2.0]], [[-2.0, 0.0], [-2.0, 0.0]]
    )
    token_ids, timestamps = decoder(
        cp.zeros((2, 1, 3), dtype=np.float32), cp.array([1, 1], dtype=np.int32)
    )

    assert token_ids == [[1], [0]]
    np.testing.assert_allclose(timestamps, [[0.0], [0.0]])


@pytest.mark.cuda
def test_parakeet_decoder_reuses_cuda_graph_across_frame_shapes() -> None:
    decoder = make_fake_parakeet_decoder(batch_size=2)
    decoder.cuda_graph_supported = True
    context = install_static_decoder_context(decoder, [-10.0, 0.0, -10.0], [-10.0, 0.0])
    encoder_output = cp.zeros((2, 20, 3), dtype=np.float32)
    first_token_ids, first_timestamps = decoder(
        encoder_output, cp.array([20, 15], dtype=np.int32)
    )
    captured_graph = decoder.cuda_graph
    output_tokens_host = decoder.output_tokens_host
    captured_calls = context.calls
    assert captured_graph is not None
    assert output_tokens_host is not None

    second_encoder_output = encoder_output.reshape(-1)[:36].reshape(1, 12, 3)
    second_token_ids, second_timestamps = decoder(
        second_encoder_output, cp.array([12], dtype=np.int32)
    )

    assert first_token_ids == [[1] * 20, [1] * 15]
    np.testing.assert_allclose(first_timestamps[0], np.arange(20) * 0.08)
    np.testing.assert_allclose(first_timestamps[1], np.arange(15) * 0.08)
    assert second_token_ids == [[1] * 12]
    np.testing.assert_allclose(second_timestamps, [np.arange(12) * 0.08])
    assert decoder.cuda_graph is captured_graph
    assert decoder.output_tokens_host is output_tokens_host
    assert context.calls == captured_calls

    replacement_output = cp.zeros_like(second_encoder_output)
    third_token_ids, third_timestamps = decoder(
        replacement_output, cp.array([12], dtype=np.int32)
    )

    assert third_token_ids == second_token_ids
    assert third_timestamps == second_timestamps
    assert decoder.cuda_graph is not captured_graph
    assert context.calls > captured_calls


@pytest.mark.cuda
def test_parakeet_decoder_retries_after_capture_execution_failure() -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.cuda_graph_supported = True
    decoder.max_symbols_per_timestep = 20
    original_buffers = get_parakeet_search_buffers(decoder)

    capture_calls = 0

    def execute(_call: int) -> bool:
        """Fail the second TensorRT execution recorded during graph capture.

        Parameters
        ----------
        _call : int
            Zero-based invocation accepted for the callback contract.

        Returns
        -------
        bool
            ``False`` only for the second capture-time invocation.
        """

        nonlocal capture_calls
        decoder.token_log_probs.fill(-10.0)
        decoder.token_log_probs[:, 1] = 0.0
        decoder.duration_log_probs.fill(-10.0)
        decoder.duration_log_probs[:, 0] = 0.0
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        if cp.cuda.runtime.streamIsCapturing(decoder.stream.ptr):
            capture_calls += 1
            return capture_calls != 2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        token_ids, timestamps = decoder(
            cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32)
        )

    assert token_ids == [[1] * 20]
    np.testing.assert_allclose(timestamps, [[0.0] * 20])
    assert capture_calls == 2
    assert not decoder.cuda_graph_supported
    assert decoder.cuda_graph is None
    assert decoder.cuda_graph_signature is None
    for name, original_buffer in original_buffers.items():
        assert getattr(decoder, name) is original_buffer

    subsequent_token_ids, subsequent_timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32)
    )
    assert subsequent_token_ids == [[1] * 20]
    np.testing.assert_allclose(subsequent_timestamps, [[0.0] * 20])
    assert decoder.cuda_graph is None


@pytest.mark.cuda
def test_parakeet_decoder_recovers_from_invalidated_cuda_capture() -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.cuda_graph_supported = True
    decoder.max_symbols_per_timestep = 20

    capture_invalidations = 0

    def execute(_call: int) -> bool:
        """Invalidate CUDA capture while returning valid decoder outputs.

        Parameters
        ----------
        _call : int
            Zero-based invocation accepted for the callback contract.

        Returns
        -------
        bool
            Always ``True`` after exercising capture invalidation.
        """

        nonlocal capture_invalidations
        decoder.token_log_probs.fill(-10.0)
        decoder.token_log_probs[:, 1] = 0.0
        decoder.duration_log_probs.fill(-10.0)
        decoder.duration_log_probs[:, 0] = 0.0
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        if cp.cuda.runtime.streamIsCapturing(decoder.stream.ptr):
            with pytest.raises(cp.cuda.runtime.CUDARuntimeError) as error:
                cp.cuda.runtime.streamSynchronize(decoder.stream.ptr)
            assert error.value.status == 900
            capture_invalidations += 1
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        token_ids, timestamps = decoder(
            cp.zeros((1, 1, 3), dtype=np.float32), cp.ones(1, dtype=np.int32)
        )

    assert token_ids == [[1] * 20]
    np.testing.assert_allclose(timestamps, [[0.0] * 20])
    assert capture_invalidations == 1
    assert not decoder.cuda_graph_supported
    assert decoder.cuda_graph is None
    assert decoder.cuda_graph_signature is None


@pytest.mark.cuda
def test_parakeet_decoder_propagates_non_invalidation_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = make_fake_parakeet_decoder()
    decoder.cuda_graph_supported = True
    decoder.max_symbols_per_timestep = 20

    class FakeCaptureError(RuntimeError):
        """Expose a CUDA runtime status through a deterministic test error."""

        def __init__(self, status: int) -> None:
            """Initialize the error from one CUDA runtime status.

            Parameters
            ----------
            status : int
                CUDA runtime error code exposed through ``status``.
            """

            super().__init__(f"CUDA capture error {status}")
            self.status = status

    class FailingCaptureStream(cp.cuda.Stream):
        """Finish graph capture and then report an unexpected CUDA error."""

        def end_capture(self) -> cp.cuda.graph.Graph:
            """End stream capture and raise the deterministic runtime error.

            Raises
            ------
            FakeCaptureError
                Always raised after CuPy finishes the active capture.
            """

            super().end_capture()
            raise FakeCaptureError(17)

    monkeypatch.setattr(
        parakeet_decoder.cp.cuda.runtime, "CUDARuntimeError", FakeCaptureError
    )
    decoder.stream = FailingCaptureStream(non_blocking=True)
    install_static_decoder_context(decoder, [-10.0, 0.0, -10.0], [0.0, -10.0])
    encoder_output = cp.zeros((1, 1, 3), dtype=np.float32)
    output_lengths = cp.ones(1, dtype=np.int32)
    cp.cuda.get_current_stream().synchronize()

    with pytest.raises(FakeCaptureError) as error:
        decoder(encoder_output, output_lengths)
    decoder.stream.synchronize()

    assert error.value.status == 17
    assert decoder.cuda_graph_supported
    assert decoder.cuda_graph is None


@pytest.mark.cuda
def test_parakeet_decoder_grows_and_reuses_output_buffers() -> None:
    decoder = make_fake_parakeet_decoder()
    install_static_decoder_context(decoder, [-10.0, 0.0, -10.0], [-10.0, 0.0])
    backing_output = cp.zeros((1, 3, 3), dtype=np.float32)
    buffer_names = (
        "node_parents",
        "node_tokens",
        "node_timestamps",
        "output_tokens",
        "output_timestamps",
        "output_tokens_host",
        "output_timestamps_host",
    )

    short_tokens, _ = decoder(backing_output[:, :1], cp.array([1], dtype=np.int32))
    short_buffers = [getattr(decoder, name) for name in buffer_names]
    assert decoder.token_capacity == 11

    long_tokens, _ = decoder(backing_output, cp.array([3], dtype=np.int32))
    long_buffers = [getattr(decoder, name) for name in buffer_names]
    assert decoder.token_capacity == 33

    final_tokens, _ = decoder(backing_output[:, :1], cp.array([1], dtype=np.int32))

    assert short_tokens == [[1]]
    assert long_tokens == [[1, 1, 1]]
    assert final_tokens == [[1]]
    for name, short_buffer, long_buffer in zip(
        buffer_names, short_buffers, long_buffers, strict=True
    ):
        assert long_buffer is not short_buffer
        assert getattr(decoder, name) is long_buffer


@pytest.mark.cuda
def test_parakeet_decoder_length_normalizes_completed_hypotheses() -> None:
    decoder = make_fake_parakeet_decoder(beam=2)

    def execute(call: int) -> bool:
        """Script competing completed hypotheses with different lengths.

        Parameters
        ----------
        call : int
            Zero-based decoder invocation index.

        Returns
        -------
        bool
            Always ``True`` after writing invocation-specific scores.
        """

        decoder.token_log_probs.fill(-10.0)
        decoder.duration_log_probs.fill(-10.0)
        if call == 0:
            decoder.token_log_probs[:, 1] = 0.0
            decoder.token_log_probs[:, decoder.blank_id] = 0.1
            decoder.duration_log_probs[:, 0] = -0.2
            decoder.duration_log_probs[:, 1] = -0.4
        else:
            decoder.token_log_probs[:, 0] = 0.0
            decoder.duration_log_probs[:, 1] = -0.15
        decoder.output_state_1[...] = decoder.state_1
        decoder.output_state_2[...] = decoder.state_2
        return True

    decoder.decoder = RuntimeDecoderContext(decoder, execute)
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32), cp.array([1], dtype=np.int32)
    )

    assert token_ids == [[1, 0]]
    np.testing.assert_allclose(timestamps, [[0.0, 0.0]])


@pytest.mark.cuda
def test_parakeet_finalize_length_normalizes_active_hypotheses() -> None:
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


def make_beam_search_buffers(
    beam: int, durations: tuple[int, ...], token_stride: int, num_frames: int
) -> SimpleNamespace:
    """Allocate a one-utterance TDT beam-search kernel fixture.

    Parameters
    ----------
    beam : int
        Number of active hypotheses represented by the buffers.
    durations : tuple[int, ...]
        TDT encoder-frame advances indexed by duration score.
    token_stride : int
        History capacity reserved for each hypothesis.
    num_frames : int
        Valid encoder frames available to the search.

    Returns
    -------
    SimpleNamespace
        CUDA arrays and scalar metadata required by ``TDT_BEAM_SEARCH_KERNEL``.
    """

    blank_id = beam
    duration_count = len(durations)
    search_shape = (1, beam)
    state_1 = cp.zeros((1, beam, 4), dtype=np.float32)
    state_2 = cp.zeros_like(state_1)
    scores = cp.zeros(search_shape, dtype=np.float32)
    nodes = cp.full(search_shape, -1, dtype=np.int32)
    hashes = cp.arange(1, beam + 1, dtype=np.uint64).reshape(search_shape)
    lengths = cp.zeros(search_shape, dtype=np.int32)
    time_indexes = cp.zeros(beam, dtype=np.int32)
    last_tokens = cp.full(beam, blank_id, dtype=np.int32)
    symbols = cp.zeros(beam, dtype=np.int32)

    return SimpleNamespace(
        beam=beam,
        blank_id=blank_id,
        token_stride=token_stride,
        token_log_probs=cp.full((beam, blank_id + 1), -100.0, dtype=np.float32),
        duration_log_probs=cp.zeros((beam, duration_count), dtype=np.float32),
        top_token_scores=cp.full((beam, beam), -100.0, dtype=np.float32),
        top_token_indexes=cp.tile(cp.arange(beam, dtype=np.int32), (beam, 1)),
        scores=scores,
        nodes=nodes,
        hashes=hashes,
        lengths=lengths,
        time_indexes=time_indexes,
        last_tokens=last_tokens,
        symbols=symbols,
        next_scores=cp.empty_like(scores),
        next_nodes=cp.empty_like(nodes),
        next_hashes=cp.empty_like(hashes),
        next_lengths=cp.empty_like(lengths),
        next_time_indexes=cp.empty_like(time_indexes),
        next_last_tokens=cp.empty_like(last_tokens),
        next_symbols=cp.empty_like(symbols),
        parent_indexes=cp.empty(beam, dtype=np.int32),
        use_output_state=cp.empty(beam, dtype=np.uint8),
        node_parents=cp.empty(beam * token_stride, dtype=np.int32),
        node_tokens=cp.empty(beam * token_stride, dtype=np.int32),
        node_timestamps=cp.empty(beam * token_stride, dtype=np.float32),
        node_counts=cp.zeros(1, dtype=np.int32),
        completed_scores=cp.full(1, -np.inf, dtype=np.float32),
        completed_nodes=cp.full(1, -1, dtype=np.int32),
        completed_lengths=cp.zeros(1, dtype=np.int32),
        active_flags=cp.empty(1, dtype=np.int32),
        output_lengths=cp.array([num_frames], dtype=np.int32),
        durations=cp.array(durations, dtype=np.int32),
        positive_duration_indexes=cp.array(
            [index for index, duration in enumerate(durations) if duration > 0],
            dtype=np.int32,
        ),
        state_1=state_1,
        state_2=state_2,
        output_state_1=state_1 + 1.0,
        output_state_2=state_2 + 1.0,
        next_state_1=cp.empty_like(state_1),
        next_state_2=cp.empty_like(state_2),
        encoder_output=cp.zeros((1, num_frames, 4), dtype=np.float32),
        encoder_input=cp.empty((beam, 4), dtype=np.float32),
        targets=cp.empty((beam, 1), dtype=np.int32),
        runtime_dimensions=cp.array([1, num_frames], dtype=np.int32),
    )


def run_beam_search(buffers: SimpleNamespace, threads: int = 256) -> None:
    """Launch the TDT beam-search kernel for a prepared fixture.

    Parameters
    ----------
    buffers : SimpleNamespace
        CUDA arrays and scalar metadata created by ``make_beam_search_buffers``.
    threads : int
        Number of CUDA threads launched for the utterance.
    """

    duration_count = buffers.durations.size
    positive_duration_count = buffers.positive_duration_indexes.size
    TDT_BEAM_SEARCH_KERNEL(
        (1,),
        (threads,),
        (
            buffers.token_log_probs,
            buffers.duration_log_probs,
            buffers.top_token_scores,
            buffers.top_token_indexes,
            buffers.scores,
            buffers.nodes,
            buffers.hashes,
            buffers.lengths,
            buffers.time_indexes,
            buffers.last_tokens,
            buffers.symbols,
            buffers.next_scores,
            buffers.next_nodes,
            buffers.next_hashes,
            buffers.next_lengths,
            buffers.next_time_indexes,
            buffers.next_last_tokens,
            buffers.next_symbols,
            buffers.parent_indexes,
            buffers.use_output_state,
            buffers.node_parents,
            buffers.node_tokens,
            buffers.node_timestamps,
            buffers.node_counts,
            buffers.completed_scores,
            buffers.completed_nodes,
            buffers.completed_lengths,
            buffers.active_flags,
            buffers.output_lengths,
            buffers.durations,
            buffers.positive_duration_indexes,
            buffers.state_1,
            buffers.state_2,
            buffers.output_state_1,
            buffers.output_state_2,
            buffers.next_state_1,
            buffers.next_state_2,
            buffers.encoder_output,
            buffers.encoder_input,
            buffers.targets,
            np.int32(4),
            np.int32(1),
            buffers.runtime_dimensions,
            np.int32(4),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(buffers.token_stride),
            np.int32(buffers.beam),
            np.int32(duration_count),
            np.int32(positive_duration_count),
            np.int32(buffers.blank_id),
            np.int32(10),
            np.float32(0.0),
            np.float32(0.08),
        ),
        shared_mem=expected_beam_search_shared_memory_bytes(
            buffers.beam, duration_count, positive_duration_count, threads
        ),
    )


@pytest.mark.cuda
def test_parakeet_beam_search_scans_beyond_first_thread_block() -> None:
    beam = 16
    buffers = make_beam_search_buffers(beam, (1,), token_stride=2, num_frames=2)
    assert beam * (beam + 1) > 256

    buffers.token_log_probs[:, buffers.blank_id] = 0.0
    buffers.token_log_probs[-1, buffers.blank_id] = 5.0
    buffers.state_1 = cp.repeat(
        cp.arange(beam, dtype=np.float32)[None, :, None], 4, axis=2
    )
    buffers.state_2 = buffers.state_1 + 100.0
    buffers.output_state_1 = buffers.state_1 + 1000.0
    buffers.output_state_2 = buffers.state_2 + 1000.0
    run_beam_search(buffers)

    np.testing.assert_allclose(buffers.next_scores.get()[0, 0], 5.0)
    assert buffers.parent_indexes.get()[0] == beam - 1
    assert buffers.next_hashes.get()[0, 0] == beam
    assert buffers.next_time_indexes.get()[0] == 1
    assert buffers.use_output_state.get()[0] == 0
    np.testing.assert_array_equal(buffers.next_state_1.get()[0, 0], beam - 1)
    np.testing.assert_array_equal(buffers.next_state_2.get()[0, 0], beam + 99)
    np.testing.assert_array_equal(buffers.node_counts.get(), [0])
    np.testing.assert_array_equal(buffers.active_flags.get(), [1])


@pytest.mark.cuda
def test_parakeet_beam_search_merges_duplicate_blank_histories() -> None:
    buffers = make_beam_search_buffers(2, (1,), token_stride=4, num_frames=2)
    buffers.token_log_probs[:, buffers.blank_id] = 0.0
    buffers.scores[...] = cp.array([[0.0, -0.2]], dtype=np.float32)
    buffers.hashes.fill(0)
    run_beam_search(buffers)

    np.testing.assert_allclose(
        buffers.next_scores.get()[0, 0], np.logaddexp(np.float32(0.0), np.float32(-0.2))
    )
    assert np.isneginf(buffers.next_scores.get()[0, 1])
    np.testing.assert_array_equal(buffers.next_lengths.get(), [[0, 0]])
    np.testing.assert_array_equal(buffers.next_time_indexes.get(), [1, 2])
    np.testing.assert_array_equal(buffers.parent_indexes.get(), [0, -1])
    np.testing.assert_array_equal(buffers.use_output_state.get(), [0, 0])
    np.testing.assert_array_equal(buffers.node_counts.get(), [0])
    np.testing.assert_array_equal(buffers.active_flags.get(), [1])


@pytest.mark.cuda
def test_parakeet_beam_search_preserves_distinct_active_symbol_counts() -> None:
    buffers = make_beam_search_buffers(2, (0, 1), token_stride=4, num_frames=3)
    buffers.duration_log_probs[...] = cp.array(
        [[-100.0, 0.0], [0.0, -100.0]], dtype=np.float32
    )
    buffers.top_token_scores[...] = cp.array(
        [[0.0, -100.0], [-0.1, -100.0]], dtype=np.float32
    )
    buffers.top_token_indexes[...] = cp.array([[1, 0], [1, 0]], dtype=np.int32)
    buffers.hashes.fill(7)
    buffers.lengths.fill(1)
    buffers.time_indexes[...] = cp.array([0, 1], dtype=np.int32)
    buffers.last_tokens.fill(1)
    buffers.state_1[...] = cp.array([[[10.0] * 4, [20.0] * 4]], dtype=np.float32)
    buffers.state_2[...] = cp.array([[[30.0] * 4, [40.0] * 4]], dtype=np.float32)
    buffers.output_state_1[...] = buffers.state_1 + 1.0
    buffers.output_state_2[...] = buffers.state_2 + 1.0
    run_beam_search(buffers)

    np.testing.assert_allclose(buffers.next_scores.get(), [[0.0, -0.1]])
    np.testing.assert_array_equal(buffers.next_lengths.get(), [[2, 2]])
    np.testing.assert_array_equal(buffers.next_time_indexes.get(), [1, 1])
    np.testing.assert_array_equal(buffers.next_symbols.get(), [0, 1])
    np.testing.assert_array_equal(buffers.parent_indexes.get(), [0, 1])
    np.testing.assert_array_equal(buffers.use_output_state.get(), [1, 1])
    np.testing.assert_array_equal(buffers.next_state_1.get()[0, :, 0], [11.0, 21.0])
    np.testing.assert_array_equal(buffers.next_state_2.get()[0, :, 0], [31.0, 41.0])
    np.testing.assert_array_equal(buffers.active_flags.get(), [1])


class NullCudaContext:
    """Provide CUDA device and stream protocols without accessing a GPU."""

    attributes = {"MultiProcessorCount": 1}
    ptr = 117

    def __init__(self, device_id: int = 0) -> None:
        """Initialize a no-op context for one synthetic CUDA device.

        Parameters
        ----------
        device_id : int
            Device identifier exposed to the decoder under test.
        """

        self.device_id = device_id

    def __enter__(self) -> "NullCudaContext":
        """Enter the no-op CUDA context.

        Returns
        -------
        NullCudaContext
            Current context instance.
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the no-op CUDA context.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class leaving the context, when present.
        exc_value : BaseException | None
            Exception instance leaving the context, when present.
        traceback : TracebackType | None
            Exception traceback leaving the context, when present.
        """

        pass


class RecordingCudaContext:
    """Record entry and exit for a fake CUDA device or stream context."""

    ptr = 117

    def __init__(self, name: str, events: list[str]) -> None:
        """Initialize a named context backed by an event log.

        Parameters
        ----------
        name : str
            Context label included in recorded entry and exit events.
        events : list[str]
            Mutable event log shared with the test.
        """

        self.name = name
        self.events = events

    def __enter__(self) -> "RecordingCudaContext":
        """Record context entry.

        Returns
        -------
        RecordingCudaContext
            Current context instance.
        """

        self.events.append(f"enter_{self.name}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record context exit.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception class leaving the context, when present.
        exc_value : BaseException | None
            Exception instance leaving the context, when present.
        traceback : TracebackType | None
            Exception traceback leaving the context, when present.
        """

        self.events.append(f"exit_{self.name}")


class FakeCudaArray:
    """Expose array metadata for validation paths without allocating storage."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        """Initialize array metadata used by decoder validation.

        Parameters
        ----------
        shape : tuple[int, ...]
            Dimensions reported by the fake CUDA array.
        dtype : np.dtype
            Element dtype reported by the fake CUDA array.
        """

        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=True)


class FakeDeviceBuffer:
    """Expose a stable device pointer without allocating CUDA memory."""

    def __init__(self, pointer: int) -> None:
        """Initialize a fake allocation with one device pointer.

        Parameters
        ----------
        pointer : int
            Synthetic CUDA address exposed through ``data.ptr``.
        """

        self.data = SimpleNamespace(ptr=pointer)


class RecordingParakeetContext:
    """Record TensorRT profile selection and fixed-buffer bindings."""

    def __init__(
        self, rejected_binding: str | None = None, profile_accepted: bool = True
    ) -> None:
        """Initialize configurable profile and binding outcomes.

        Parameters
        ----------
        rejected_binding : str | None
            Tensor name whose address assignment should fail, when present.
        profile_accepted : bool
            Whether optimization-profile selection succeeds.
        """

        self.rejected_binding = rejected_binding
        self.profile_accepted = profile_accepted
        self.profile_calls: list[tuple[int, int]] = []
        self.bindings: dict[str, int] = {}

    def set_optimization_profile_async(
        self, profile_index: int, stream_pointer: int
    ) -> bool:
        """Record an asynchronous optimization-profile request.

        Parameters
        ----------
        profile_index : int
            TensorRT optimization-profile index.
        stream_pointer : int
            CUDA stream pointer associated with profile selection.

        Returns
        -------
        bool
            Configured profile-selection result.
        """

        self.profile_calls.append((profile_index, stream_pointer))
        return self.profile_accepted

    def set_tensor_address(self, name: str, address: int) -> bool:
        """Record one fixed tensor binding and optionally reject it.

        Parameters
        ----------
        name : str
            TensorRT tensor name being bound.
        address : int
            Device address assigned to the tensor.

        Returns
        -------
        bool
            ``False`` only when ``name`` is the configured rejected binding.
        """

        self.bindings[name] = address
        return name != self.rejected_binding


class FakeParakeetEngine:
    """Expose only Parakeet metadata queried before context initialization."""

    shapes = {
        "encoder_output": (2, 4),
        "targets": (2, 1),
        "input_states_1": (2, 2, 3),
        "token_log_probs": (2, 8),
        "duration_log_probs": (2, 2),
    }

    def __init__(
        self,
        context: RecordingParakeetContext | None,
        encoder_dtype: trt.DataType = trt.float32,
        state_dtype: trt.DataType = trt.float32,
    ) -> None:
        """Initialize fake engine metadata and execution context.

        Parameters
        ----------
        context : RecordingParakeetContext | None
            Context returned during decoder initialization, or ``None`` to
            simulate context-creation failure.
        encoder_dtype : trt.DataType
            Floating-point dtype reported for encoder input.
        state_dtype : trt.DataType
            Floating-point dtype reported for recurrent state.
        """

        self.context = context
        self.dtypes = {"encoder_output": encoder_dtype, "input_states_1": state_dtype}
        self.shape_requests: list[str] = []
        self.dtype_requests: list[str] = []

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Record and return the fixed shape of one decoder tensor.

        Parameters
        ----------
        name : str
            TensorRT tensor name.

        Returns
        -------
        tuple[int, ...]
            Static dimensions configured for ``name``.
        """

        self.shape_requests.append(name)
        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        """Record and return one configured floating-point dtype.

        Parameters
        ----------
        name : str
            Decoder tensor whose dtype is requested.

        Returns
        -------
        trt.DataType
            Configured TensorRT dtype for ``name``.
        """

        self.dtype_requests.append(name)
        return self.dtypes[name]

    def create_execution_context(self) -> RecordingParakeetContext | None:
        """Return the configured fake TensorRT execution context.

        Returns
        -------
        RecordingParakeetContext | None
            Context supplied at engine construction.
        """

        return self.context


def make_parakeet_validation_decoder() -> ParakeetModifiedBeamSearchDecoder:
    """Construct only the state needed by pre-CUDA Parakeet validation.

    Returns
    -------
    ParakeetModifiedBeamSearchDecoder
        Uninitialized decoder carrying valid shape and capacity metadata.
    """

    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    decoder.batch_size = 2
    decoder.decoder_capacity = 2
    decoder.encoder_dim = 4
    decoder.max_symbols_per_timestep = 1
    return decoder


def test_parakeet_decoder_initializes_inside_requested_cuda_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    requested_device_ids: list[int] = []
    failure = RuntimeError("engine load failed")
    stream = RecordingCudaContext("stream", events)

    def make_device(device_id: int) -> RecordingCudaContext:
        """Record the requested device and return its fake CUDA context.

        Parameters
        ----------
        device_id : int
            CUDA device identifier requested by the decoder.

        Returns
        -------
        RecordingCudaContext
            Device context backed by the shared event log.
        """

        requested_device_ids.append(device_id)
        return RecordingCudaContext("device", events)

    def fail_engine_load(engine_path: Path) -> None:
        """Raise the configured engine-load failure inside active contexts.

        Parameters
        ----------
        engine_path : Path
            TensorRT engine path requested by the decoder.

        Raises
        ------
        RuntimeError
            Always raises the test's configured failure instance.
        """

        assert engine_path == Path("decoder.trt")
        assert events == ["enter_device", "enter_stream"]
        raise failure

    monkeypatch.setattr(parakeet_decoder.cp.cuda, "Device", make_device)
    monkeypatch.setattr(parakeet_decoder, "get_engine", fail_engine_load)

    with pytest.raises(RuntimeError) as error:
        ParakeetModifiedBeamSearchDecoder(
            Path("decoder.trt"),
            batch_size=2,
            blank_id=7,
            durations=(0, 1),
            max_symbols_per_timestep=10,
            encoder_frame_shift_sec=0.08,
            blank_penalty=0.0,
            device_id=4,
            stream=cast(cp.cuda.Stream, stream),
        )

    assert error.value is failure
    assert requested_device_ids == [4]
    assert events == ["enter_device", "enter_stream", "exit_stream", "exit_device"]


def test_parakeet_decoder_zero_frame_call_uses_cuda_contexts() -> None:
    events: list[str] = []
    decoder = make_parakeet_validation_decoder()
    decoder.device = cast(cp.cuda.Device, RecordingCudaContext("device", events))
    decoder.stream = cast(cp.cuda.Stream, RecordingCudaContext("stream", events))

    token_ids, timestamps = decoder(
        cast(cp.ndarray, np.empty((1, 0, 4), dtype=np.float32)),
        cast(cp.ndarray, np.zeros(1, dtype=np.int32)),
    )

    assert token_ids == [[]]
    assert timestamps == [[]]
    assert events == ["enter_device", "enter_stream", "exit_stream", "exit_device"]


def construct_parakeet_decoder(stream: cp.cuda.Stream | NullCudaContext) -> None:
    """Initialize a Parakeet decoder through its context-setup boundary.

    Parameters
    ----------
    stream : cp.cuda.Stream | NullCudaContext
        Real or no-op CUDA stream supplied to constructor tests.
    """

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

    context = RecordingParakeetContext()
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
def test_parakeet_decoder_attempts_both_recurrent_bindings(rejected_name: str) -> None:
    decoder = ParakeetModifiedBeamSearchDecoder.__new__(
        ParakeetModifiedBeamSearchDecoder
    )
    for pointer, name in enumerate(
        (name for pair in PARAKEET_SEARCH_BUFFER_PAIRS for name in pair), start=1
    ):
        setattr(decoder, name, FakeDeviceBuffer(pointer))

    context = RecordingParakeetContext(rejected_name)
    decoder.decoder = cast(trt.IExecutionContext, context)

    with pytest.raises(ASRInferenceError, match="recurrent-state input"):
        decoder.swap_buffers()

    assert list(context.bindings) == ["input_states_1", "input_states_2"]


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
    encoder_output: np.typing.NDArray[np.generic],
    encoder_output_lengths: np.typing.NDArray[np.generic],
    message: str,
) -> None:
    decoder = make_parakeet_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, encoder_output), cast(cp.ndarray, encoder_output_lengths)
        )


def test_parakeet_decoder_rejects_int32_history_capacity_overflow() -> None:
    decoder = make_parakeet_validation_decoder()
    decoder.max_symbols_per_timestep = INT32_MAX
    encoder_output = np.zeros((2, 2, 4), dtype=np.float32)
    output_lengths = np.full(2, 2, dtype=np.int32)

    with pytest.raises(ASRInferenceError, match="signed 32-bit kernel indexing"):
        decoder(cast(cp.ndarray, encoder_output), cast(cp.ndarray, output_lengths))


def test_parakeet_decoder_rejects_int32_encoder_capacity_overflow() -> None:
    decoder = make_parakeet_validation_decoder()
    max_frames = INT32_MAX // (decoder.batch_size * decoder.encoder_dim) + 1
    encoder_output = FakeCudaArray(
        (decoder.batch_size, max_frames, decoder.encoder_dim), np.dtype(np.float32)
    )
    output_lengths = FakeCudaArray((decoder.batch_size,), np.dtype(np.int32))

    with pytest.raises(ASRInferenceError, match="signed 32-bit kernel indexing"):
        decoder(cast(cp.ndarray, encoder_output), cast(cp.ndarray, output_lengths))


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("encoder_trt_dtype", "state_trt_dtype"),
    (
        pytest.param(trt.float32, trt.float16, id="fp32-encoder-fp16-state"),
        pytest.param(trt.float16, trt.float32, id="fp16-encoder-fp32-state"),
        pytest.param(trt.bfloat16, trt.bfloat16, marks=pytest.mark.sm80, id="bf16"),
    ),
)
def test_parakeet_decoder_initializes_precisions_and_fixed_bindings(
    monkeypatch: pytest.MonkeyPatch,
    encoder_trt_dtype: trt.DataType,
    state_trt_dtype: trt.DataType,
) -> None:
    dtype_map = {
        trt.float32: (np.dtype(np.float32), np.int32(0)),
        trt.float16: (np.dtype(np.float16), np.int32(1)),
        trt.bfloat16: (cp.dtype("bfloat16"), np.int32(2)),
    }
    encoder_array_dtype, encoder_dtype_code = dtype_map[encoder_trt_dtype]
    state_array_dtype, state_dtype_code = dtype_map[state_trt_dtype]
    stream = cp.cuda.get_current_stream()
    context = RecordingParakeetContext()
    engine = FakeParakeetEngine(context, encoder_trt_dtype, state_trt_dtype)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeParakeetEngine:
        """Record the requested path and return the configured fake engine.

        Parameters
        ----------
        engine_path : Path
            TensorRT engine path requested by the decoder.

        Returns
        -------
        FakeParakeetEngine
            Engine configured by the enclosing test.
        """

        loaded_paths.append(engine_path)
        return engine

    monkeypatch.setattr(parakeet_decoder, "get_engine", load_engine)
    decoder = ParakeetModifiedBeamSearchDecoder(
        Path("decoder.trt"),
        batch_size=2,
        blank_id=7,
        durations=(0, 1),
        max_symbols_per_timestep=10,
        encoder_frame_shift_sec=0.08,
        blank_penalty=0.0,
        device_id=0,
        stream=stream,
    )

    assert loaded_paths == [Path("decoder.trt")]
    assert sorted(engine.shape_requests) == sorted(engine.shapes)
    assert sorted(engine.dtype_requests) == sorted(engine.dtypes)
    assert context.profile_calls == [(0, stream.ptr)]
    assert context.bindings == {
        "encoder_output": decoder.encoder_input.data.ptr,
        "targets": decoder.targets.data.ptr,
        "input_states_1": decoder.state_1.data.ptr,
        "input_states_2": decoder.state_2.data.ptr,
        "token_log_probs": decoder.token_log_probs.data.ptr,
        "duration_log_probs": decoder.duration_log_probs.data.ptr,
        "output_states_1": decoder.output_state_1.data.ptr,
        "output_states_2": decoder.output_state_2.data.ptr,
    }
    assert decoder.device.id == 0
    assert decoder.stream is stream
    assert decoder.decoder is context
    assert (
        decoder.batch_size,
        decoder.beam,
        decoder.decoder_capacity,
        decoder.encoder_dim,
        decoder.state_layers,
        decoder.state_hidden_dim,
    ) == (2, 1, 2, 4, 2, 3)
    assert (
        decoder.blank_id,
        decoder.max_symbols_per_timestep,
        decoder.encoder_frame_shift_sec,
        decoder.blank_penalty,
    ) == (7, 10, 0.08, 0.0)
    assert decoder.beam_search_shared_memory_bytes == (
        expected_beam_search_shared_memory_bytes(
            decoder.beam,
            decoder.duration_log_probs.shape[1],
            decoder.positive_duration_indexes_array.size,
            decoder.beam_search_threads,
        )
    )
    assert decoder.token_selection_shared_memory_bytes == (
        expected_token_selection_shared_memory_bytes(
            decoder.blank_id, decoder.token_selection_threads
        )
    )
    assert (decoder.cuda_graph, decoder.cuda_graph_signature) == (None, None)
    assert decoder.cuda_graph_supported

    expected_shapes = {
        "encoder_input": (2, 4),
        "targets": (2, 1),
        "state_1": (2, 2, 3),
        "next_state_1": (2, 2, 3),
        "token_log_probs": (2, 8),
        "top_token_scores": (2, 1),
        "duration_log_probs": (2, 2),
        "output_state_1": (2, 2, 3),
        "hypothesis_scores": (2, 1),
        "time_indexes": (2,),
        "search_output_lengths": (2,),
        "runtime_dimensions": (2,),
        "completed_scores": (2,),
        "output_lengths": (2,),
        "hypothesis_nodes": (2, 1),
        "hypothesis_hashes": (2, 1),
        "node_counts": (2,),
    }
    for name, expected_shape in expected_shapes.items():
        assert getattr(decoder, name).shape == expected_shape

    assert decoder.encoder_input.dtype == encoder_array_dtype
    assert {
        decoder.state_1.dtype,
        decoder.state_2.dtype,
        decoder.next_state_1.dtype,
        decoder.next_state_2.dtype,
        decoder.output_state_1.dtype,
        decoder.output_state_2.dtype,
    } == {state_array_dtype}
    assert decoder.encoder_input_dtype == encoder_dtype_code
    assert decoder.state_dtype == state_dtype_code
    np.testing.assert_array_equal(decoder.durations_array.get(), [0, 1])
    np.testing.assert_array_equal(decoder.positive_duration_indexes_array.get(), [1])
    assert decoder.token_capacity == 0
    assert all(
        getattr(decoder, name) is None
        for name in (
            "node_parents",
            "node_tokens",
            "node_timestamps",
            "output_tokens",
            "output_timestamps",
            "output_tokens_host",
            "output_timestamps_host",
        )
    )


@pytest.mark.cuda
def test_parakeet_decoder_derives_wide_beam_capacity_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = cp.cuda.get_current_stream()
    context = RecordingParakeetContext()
    engine = FakeParakeetEngine(context)
    engine.shapes = {
        "encoder_output": (6, 4),
        "targets": (6, 1),
        "input_states_1": (2, 6, 3),
        "token_log_probs": (6, 8),
        "duration_log_probs": (6, 2),
    }
    monkeypatch.setattr(parakeet_decoder, "get_engine", lambda _path: engine)

    decoder = ParakeetModifiedBeamSearchDecoder(
        Path("wide-decoder.trt"),
        batch_size=2,
        blank_id=7,
        durations=(0, 1),
        max_symbols_per_timestep=10,
        encoder_frame_shift_sec=0.08,
        blank_penalty=0.25,
        device_id=0,
        stream=stream,
    )

    assert decoder.beam == 3
    assert decoder.decoder_capacity == 6
    assert decoder.top_token_scores.shape == (6, 3)
    assert decoder.top_token_indexes.shape == (6, 3)
    assert decoder.hypothesis_scores.shape == (2, 3)
    assert decoder.hypothesis_lengths.shape == (2, 3)
    assert decoder.hypothesis_nodes.shape == (2, 3)
    assert decoder.hypothesis_hashes.shape == (2, 3)
    assert decoder.time_indexes.shape == (6,)
    assert decoder.state_1.shape == (2, 6, 3)
    assert decoder.beam_search_shared_memory_bytes == (
        expected_beam_search_shared_memory_bytes(
            beam=3,
            duration_count=2,
            positive_duration_count=1,
            threads=decoder.beam_search_threads,
        )
    )
    assert decoder.blank_penalty == 0.25


@pytest.mark.cuda
@pytest.mark.parametrize(
    "rejected_binding",
    (
        "encoder_output",
        "targets",
        "input_states_1",
        "input_states_2",
        "token_log_probs",
        "duration_log_probs",
        "output_states_1",
        "output_states_2",
    ),
)
def test_parakeet_decoder_reports_fixed_tensor_binding_failure(
    monkeypatch: pytest.MonkeyPatch, rejected_binding: str
) -> None:
    stream = cp.cuda.get_current_stream()
    context = RecordingParakeetContext(rejected_binding)
    engine = FakeParakeetEngine(context)
    monkeypatch.setattr(parakeet_decoder, "get_engine", lambda _path: engine)

    with pytest.raises(ASRInitializationError) as error:
        construct_parakeet_decoder(stream)

    assert str(error.value) == (
        f"TensorRT rejected the Parakeet decoder tensor {rejected_binding}."
    )
    assert context.bindings[rejected_binding] > 0


def test_parakeet_decoder_rejects_missing_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = NullCudaContext()
    engine = FakeParakeetEngine(None)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeParakeetEngine:
        """Record the requested path and return the context-free engine.

        Parameters
        ----------
        engine_path : Path
            TensorRT engine path requested by the decoder.

        Returns
        -------
        FakeParakeetEngine
            Engine configured without an execution context.
        """

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
    stream = NullCudaContext()
    context = RecordingParakeetContext(profile_accepted=False)
    engine = FakeParakeetEngine(context)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeParakeetEngine:
        """Record the requested path and return the profile-rejecting engine.

        Parameters
        ----------
        engine_path : Path
            TensorRT engine path requested by the decoder.

        Returns
        -------
        FakeParakeetEngine
            Engine whose context rejects profile selection.
        """

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
