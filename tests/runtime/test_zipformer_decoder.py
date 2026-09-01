#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Runtime and validation tests for Zipformer CTC and RNN-T decoders."""

from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast

import cupy as cp
import numpy as np
import pytest

from fast_gpu_asr.constants import INT32_MAX
from fast_gpu_asr.decoder import zipformer_decoder
from fast_gpu_asr.decoder.gpu_kernels import (
    ZIPFORMER_FINALIZE_KERNEL,
    get_zipformer_beam_search_kernels,
)
from fast_gpu_asr.decoder.zipformer_decoder import (
    CTCGreedyDecoder,
    ZipformerModifiedBeamSearchDecoder,
)
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError


def initialize_zipformer_search_buffers(
    decoder: ZipformerModifiedBeamSearchDecoder,
    vocab_size: int,
) -> None:
    """Initialize buffers normally created by the TensorRT-backed constructor."""

    decoder.kernel_dtype_map = {
        np.dtype(np.float32): np.int32(0),
        np.dtype(np.float16): np.int32(1),
        cp.dtype("bfloat16"): np.int32(2),
    }
    (
        decoder.register_beam_search,
        decoder.shared_beam_search,
        decoder.beam_search_threads,
    ) = get_zipformer_beam_search_kernels(
        decoder.beam,
        vocab_size,
        decoder.context_size,
    )
    decoder.beam_search_register_batch_limit = 64
    decoder.cuda_graph = None
    decoder.cuda_graph_signature = None
    decoder.cuda_graph_supported = False
    search_shape = (decoder.batch_size, decoder.beam)
    decoder.output_tokens = None
    decoder.output_timestamps = None
    decoder.output_lengths = cp.empty(decoder.batch_size, dtype=np.int32)
    decoder.output_tokens_host = None
    decoder.output_timestamps_host = None
    decoder.output_lengths_host = None
    decoder.search_output_lengths = cp.empty(decoder.batch_size, dtype=np.int32)
    decoder.hypothesis_scores = cp.empty(search_shape, dtype=np.float32)
    decoder.next_scores = cp.empty_like(decoder.hypothesis_scores)
    decoder.hypothesis_lengths = cp.empty(search_shape, dtype=np.int32)
    decoder.next_lengths = cp.empty_like(decoder.hypothesis_lengths)
    decoder.hypothesis_hashes = cp.empty(search_shape, dtype=np.uint64)
    decoder.next_hashes = cp.empty_like(decoder.hypothesis_hashes)
    decoder.hypothesis_nodes = cp.empty(search_shape, dtype=np.int32)
    decoder.next_nodes = cp.empty_like(decoder.hypothesis_nodes)
    decoder.node_counts = cp.empty(decoder.batch_size, dtype=np.int32)
    initial_contexts = np.zeros(
        (decoder.batch_size, decoder.beam, decoder.context_size), dtype=np.int32
    )
    initial_contexts[:, 0, :] = -1
    initial_contexts[:, 0, decoder.context_size - 1] = decoder.blank_id
    initial_contexts = initial_contexts.reshape(
        decoder.batch_size * decoder.beam, decoder.context_size
    )
    decoder.initial_contexts = cp.array(initial_contexts)
    initial_lookup_indexes = np.zeros(decoder.batch_size * decoder.beam, dtype=np.int64)
    for position in range(decoder.context_size):
        initial_lookup_indexes *= vocab_size + 1
        initial_lookup_indexes += initial_contexts[:, position] + 1
    decoder.initial_decoder_input = decoder.context_lookup[
        cp.array(initial_lookup_indexes)
    ]
    decoder.frame_capacity = 0
    decoder.node_parents = None
    decoder.node_tokens = None
    decoder.node_timestamps = None


def make_fake_zipformer_decoder(
    *,
    batch_size: int = 1,
    beam: int = 1,
    context_size: int = 1,
    encoder_dim: int = 3,
    vocab_size: int = 4,
    decoder_dtype: np.dtype | None = None,
    sequential_context_lookup: bool = False,
) -> ZipformerModifiedBeamSearchDecoder:
    """Create a GPU decoder without loading a TensorRT engine."""

    if decoder_dtype is None:
        decoder_dtype = np.dtype(np.float16)

    decoder = ZipformerModifiedBeamSearchDecoder.__new__(
        ZipformerModifiedBeamSearchDecoder
    )
    decoder.device = cp.cuda.Device(0)
    decoder.batch_size = batch_size
    decoder.beam = beam
    decoder.context_size = context_size
    decoder.encoder_dim = encoder_dim
    decoder.blank_id = 0
    decoder.encoder_frame_shift_sec = 0.04
    decoder.blank_penalty = 0.0
    decoder.stream = cp.cuda.Stream(non_blocking=True)

    decoder_capacity = batch_size * beam
    decoder.decoder_input = cp.empty(
        (decoder_capacity, encoder_dim), dtype=decoder_dtype
    )
    decoder.contexts = cp.empty((decoder_capacity, context_size), dtype=np.int32)
    context_rows = (vocab_size + 1) ** context_size
    if sequential_context_lookup:
        decoder.context_lookup = (
            cp.arange(context_rows * encoder_dim, dtype=np.float32)
            .reshape(context_rows, encoder_dim)
            .astype(decoder_dtype)
        )
    else:
        decoder.context_lookup = cp.zeros(
            (context_rows, encoder_dim), dtype=decoder_dtype
        )
    decoder.encoder_input = cp.empty(
        (decoder_capacity, encoder_dim), dtype=decoder_dtype
    )
    decoder.tokens_log_prob = cp.empty((decoder_capacity, vocab_size), dtype=np.float32)
    initialize_zipformer_search_buffers(decoder, vocab_size)
    return decoder


@pytest.mark.cuda
def test_ctc_greedy_collapses_repeats_and_blanks() -> None:
    """Collapse repeated labels while preserving repeats separated by blank."""

    decoder = CTCGreedyDecoder(
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=cp.cuda.Stream(non_blocking=True),
    )
    paths = cp.array(
        [
            [0, 1, 1, 0, 1, 2],
            [2, 2, 0, 3, 3, 3],
        ],
    )
    log_probs = cp.eye(4, dtype=cp.float32)[paths]

    token_ids, timestamps = decoder(
        log_probs,
        cp.array([6, 4], dtype=np.int32),
    )

    assert token_ids == [[1, 1, 2], [2, 3]]
    np.testing.assert_allclose(timestamps[0], [0.04, 0.16, 0.20])
    np.testing.assert_allclose(timestamps[1], [0.0, 0.12])


@pytest.mark.cuda
def test_ctc_greedy_applies_blank_penalty_before_argmax() -> None:
    """Apply the configured blank penalty before selecting each frame label."""

    decoder = CTCGreedyDecoder(
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.2,
        device_id=0,
        stream=cp.cuda.Stream(non_blocking=True),
    )
    log_probs = cp.array([[[0.1, 0.0]]], dtype=cp.float32)

    token_ids, timestamps = decoder(log_probs, cp.array([1], dtype=np.int32))

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps[0], [0.0])


@pytest.mark.cuda
def test_ctc_greedy_clamps_invalid_output_lengths() -> None:
    """Clamp output lengths to the valid frame interval for each utterance."""

    decoder = CTCGreedyDecoder(
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=cp.cuda.Stream(non_blocking=True),
    )
    log_probs = cp.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    token_ids, timestamps = decoder(log_probs, cp.array([10, -1], dtype=np.int32))

    assert token_ids == [[1], []]
    np.testing.assert_allclose(timestamps[0], [0.0])
    assert timestamps[1] == []


@pytest.mark.cuda
def test_ctc_greedy_returns_empty_results_for_zero_frames() -> None:
    """Return one empty result per utterance when no encoder frames exist."""

    decoder = CTCGreedyDecoder(
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=cp.cuda.Stream(non_blocking=True),
    )

    token_ids, timestamps = decoder(
        cp.empty((2, 0, 4), dtype=np.float32),
        cp.zeros(2, dtype=np.int32),
    )

    assert token_ids == [[], []]
    assert timestamps == [[], []]


@pytest.mark.cuda
def test_zipformer_register_and_shared_search_match() -> None:
    """Produce identical search state with register and shared candidate storage."""

    batch_size = 2
    beam = 6
    vocab_size = 16
    context_size = 2
    encoder_dim = 8
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, context_size
    )
    assert register_launch is not None

    rng = np.random.default_rng(0)
    log_probs = cp.array(
        rng.normal(size=(batch_size * beam, vocab_size)).astype(np.float32)
    )
    hypothesis_scores = cp.array(rng.normal(size=(batch_size, beam)).astype(np.float32))
    hypothesis_nodes = cp.full((batch_size, beam), -1, dtype=np.int32)
    hypothesis_lengths = cp.zeros((batch_size, beam), dtype=np.int32)
    hypothesis_hashes = cp.arange(batch_size * beam, dtype=np.uint64).reshape(
        batch_size, beam
    )
    initial_contexts = cp.array(
        rng.integers(
            0,
            vocab_size,
            size=(batch_size * beam, context_size),
            dtype=np.int32,
        )
    )
    output_lengths = cp.array([10, -1], dtype=np.int32)
    encoder_output = cp.zeros((batch_size, 1, encoder_dim), dtype=np.float32)
    context_lookup = cp.zeros(
        ((vocab_size + 1) ** context_size, encoder_dim), dtype=np.float32
    )

    def run_search(launch: tuple[cp.RawKernel, int]) -> tuple[np.ndarray, ...]:
        kernel, shared_memory_bytes = launch
        contexts = initial_contexts.copy()
        next_scores = cp.empty_like(hypothesis_scores)
        next_nodes = cp.empty_like(hypothesis_nodes)
        next_lengths = cp.empty_like(hypothesis_lengths)
        next_hashes = cp.empty_like(hypothesis_hashes)
        node_elements = batch_size * beam
        node_parents = cp.full(node_elements, -2, dtype=np.int32)
        node_tokens = cp.full(node_elements, -2, dtype=np.int32)
        node_timestamps = cp.full(node_elements, -2.0, dtype=np.float32)
        node_counts = cp.zeros(batch_size, dtype=np.int32)

        kernel(
            (batch_size,),
            (threads,),
            (
                log_probs,
                encoder_output,
                cp.empty((batch_size * beam, encoder_dim), dtype=np.float32),
                context_lookup,
                cp.empty((batch_size * beam, encoder_dim), dtype=np.float32),
                contexts,
                hypothesis_scores,
                hypothesis_nodes,
                hypothesis_lengths,
                hypothesis_hashes,
                next_scores,
                next_nodes,
                next_lengths,
                next_hashes,
                node_parents,
                node_tokens,
                node_timestamps,
                node_counts,
                output_lengths,
                np.int32(0),
                np.int32(1),
                np.int32(encoder_dim),
                np.int32(0),
                np.int32(0),
                np.int32(0),
                np.int32(0),
                np.float32(0.1),
                np.float32(0.04),
            ),
            shared_mem=shared_memory_bytes,
        )
        return tuple(
            array.get()
            for array in (
                next_scores,
                next_nodes,
                next_lengths,
                next_hashes,
                contexts,
                node_parents,
                node_tokens,
                node_timestamps,
                node_counts,
            )
        )

    register_results = run_search(register_launch)
    shared_results = run_search(shared_launch)
    for register_result, shared_result in zip(
        register_results, shared_results, strict=True
    ):
        np.testing.assert_array_equal(register_result, shared_result)


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_zipformer_search_keeps_nonfinite_candidates_in_bounds(
    invalid_score: float,
) -> None:
    """Keep real candidate indexes distinct when every token score is non-finite."""

    beam = 2
    vocab_size = 3
    context_size = 1
    encoder_dim = 4
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, context_size
    )
    assert register_launch is not None

    def run_search(launch: tuple[cp.RawKernel, int]) -> tuple[np.ndarray, ...]:
        kernel, shared_memory_bytes = launch
        next_scores = cp.empty((1, beam), dtype=np.float32)
        next_nodes = cp.empty((1, beam), dtype=np.int32)
        next_lengths = cp.empty((1, beam), dtype=np.int32)
        next_hashes = cp.empty((1, beam), dtype=np.uint64)
        node_counts = cp.zeros(1, dtype=np.int32)
        contexts = cp.zeros((beam, context_size), dtype=np.int32)
        kernel(
            (1,),
            (threads,),
            (
                cp.full((beam, vocab_size), invalid_score, dtype=np.float32),
                cp.zeros((1, 1, encoder_dim), dtype=np.float32),
                cp.empty((beam, encoder_dim), dtype=np.float32),
                cp.zeros(
                    ((vocab_size + 1) ** context_size, encoder_dim),
                    dtype=np.float32,
                ),
                cp.empty((beam, encoder_dim), dtype=np.float32),
                contexts,
                cp.array([[0.0, -np.inf]], dtype=np.float32),
                cp.full((1, beam), -1, dtype=np.int32),
                cp.zeros((1, beam), dtype=np.int32),
                cp.zeros((1, beam), dtype=np.uint64),
                next_scores,
                next_nodes,
                next_lengths,
                next_hashes,
                cp.full(beam, -1, dtype=np.int32),
                cp.full(beam, -1, dtype=np.int32),
                cp.zeros(beam, dtype=np.float32),
                node_counts,
                cp.array([1], dtype=np.int32),
                np.int32(0),
                np.int32(1),
                np.int32(encoder_dim),
                np.int32(0),
                np.int32(0),
                np.int32(0),
                np.int32(0),
                np.float32(0.0),
                np.float32(0.04),
            ),
            shared_mem=shared_memory_bytes,
        )
        return (
            next_scores.get(),
            next_nodes.get(),
            next_lengths.get(),
            next_hashes.get(),
            contexts.get(),
            node_counts.get(),
        )

    register_results = run_search(register_launch)
    shared_results = run_search(shared_launch)
    for register_result, shared_result in zip(
        register_results, shared_results, strict=True
    ):
        np.testing.assert_array_equal(register_result, shared_result)
    np.testing.assert_array_equal(register_results[1], [[-1, 0]])
    np.testing.assert_array_equal(register_results[-1], [1])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "use_register_search",
    (False, True),
    ids=("shared", "register"),
)
def test_zipformer_search_merges_duplicate_histories(
    use_register_search: bool,
) -> None:
    """Merge equal token histories with log-add-exp in both search kernels."""

    beam = 2
    vocab_size = 3
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, 1
    )
    launch = register_launch if use_register_search else shared_launch
    assert launch is not None
    kernel, shared_memory_bytes = launch

    next_scores = cp.empty((1, beam), dtype=np.float32)
    next_nodes = cp.empty((1, beam), dtype=np.int32)
    next_lengths = cp.empty((1, beam), dtype=np.int32)
    next_hashes = cp.empty((1, beam), dtype=np.uint64)
    node_parents = cp.array([-1, -2, -2, -2], dtype=np.int32)
    node_tokens = cp.array([1, -2, -2, -2], dtype=np.int32)
    node_timestamps = cp.array([0.0, -2.0, -2.0, -2.0], dtype=np.float32)
    node_counts = cp.array([1], dtype=np.int32)

    kernel(
        (1,),
        (threads,),
        (
            cp.array([[-0.1, -5.0, -6.0], [-5.0, -0.2, -6.0]], dtype=np.float32),
            cp.zeros((1, 2, 1), dtype=np.float32),
            cp.empty((beam, 1), dtype=np.float32),
            cp.zeros((vocab_size + 1, 1), dtype=np.float32),
            cp.empty((beam, 1), dtype=np.float32),
            cp.array([[1], [0]], dtype=np.int32),
            cp.array([[-0.2, -0.3]], dtype=np.float32),
            cp.array([[0, -1]], dtype=np.int32),
            cp.array([[1, 0]], dtype=np.int32),
            cp.array([[2, 0]], dtype=np.uint64),
            next_scores,
            next_nodes,
            next_lengths,
            next_hashes,
            node_parents,
            node_tokens,
            node_timestamps,
            node_counts,
            cp.array([2], dtype=np.int32),
            np.int32(1),
            np.int32(2),
            np.int32(1),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.int32(0),
            np.float32(0.0),
            np.float32(0.04),
        ),
        shared_mem=shared_memory_bytes,
    )

    expected_score = np.logaddexp(np.float32(-0.3), np.float32(-0.5))
    np.testing.assert_allclose(next_scores.get()[0, 0], expected_score, atol=1e-7)
    np.testing.assert_array_equal(next_nodes.get()[0, 0], 0)
    np.testing.assert_array_equal(next_lengths.get()[0, 0], 1)
    np.testing.assert_array_equal(node_counts.get(), [1])


@pytest.mark.cuda
def test_zipformer_finalize_uses_length_normalized_score() -> None:
    """Select the completed history with the best length-normalized score."""

    output_tokens = cp.empty((1, 3), dtype=np.int32)
    output_timestamps = cp.empty((1, 3), dtype=np.float32)
    output_lengths = cp.empty(1, dtype=np.int32)

    ZIPFORMER_FINALIZE_KERNEL(
        (1,),
        (1,),
        (
            cp.array([[-1.0, -1.2]], dtype=np.float32),
            cp.array([[0, 1]], dtype=np.int32),
            cp.array([[1, 2]], dtype=np.int32),
            cp.array([-1, 0], dtype=np.int32),
            cp.array([3, 4], dtype=np.int32),
            cp.array([0.0, 0.04], dtype=np.float32),
            output_tokens,
            output_timestamps,
            output_lengths,
            np.int32(3),
            np.int32(2),
            np.int32(2),
        ),
    )

    np.testing.assert_array_equal(output_lengths.get(), [2])
    np.testing.assert_array_equal(output_tokens.get()[0, :2], [3, 4])
    np.testing.assert_allclose(output_timestamps.get()[0, :2], [0.0, 0.04])


@pytest.mark.cuda
def test_zipformer_decoder_returns_empty_results_for_zero_frames() -> None:
    """Return one empty result for each actual zero-frame utterance."""

    decoder = make_fake_zipformer_decoder(batch_size=2)

    token_ids, timestamps = decoder(
        cp.empty((1, 0, 3), dtype=np.float32),
        cp.zeros(1, dtype=np.int32),
    )

    assert token_ids == [[]]
    assert timestamps == [[]]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("blank_penalty", "expected_tokens"),
    ((0.0, []), (0.2, [1])),
    ids=("no-penalty", "with-penalty"),
)
def test_zipformer_decoder_applies_blank_penalty(
    blank_penalty: float,
    expected_tokens: list[int],
) -> None:
    """Apply blank penalty while ranking full RNN-T search candidates."""

    decoder = make_fake_zipformer_decoder()
    decoder.blank_penalty = blank_penalty

    class FakeContext:
        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.tokens_log_prob[...] = cp.array(
                [[0.1, 0.0, -8.0, -8.0]], dtype=np.float32
            )
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.ones(1, dtype=np.int32),
    )

    assert token_ids == [expected_tokens]
    assert timestamps == ([[0.0]] if expected_tokens else [[]])


@pytest.mark.cuda
def test_zipformer_decoder_reports_tensorrt_execution_failure() -> None:
    """Raise an inference error when TensorRT rejects decoder execution."""

    decoder = make_fake_zipformer_decoder()

    class FailingContext:
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
@pytest.mark.parametrize(
    "use_register_search",
    (False, True),
    ids=("shared", "register"),
)
def test_zipformer_decoder_batches_unequal_lengths(
    use_register_search: bool,
) -> None:
    """Decode unequal valid lengths through both kernel and CUDA graph paths."""

    decoder = make_fake_zipformer_decoder(
        batch_size=2,
        beam=2,
        context_size=1,
        encoder_dim=3,
        vocab_size=4,
    )
    decoder.beam_search_register_batch_limit = 64 if use_register_search else 0
    decoder.cuda_graph_supported = True

    class FakeContext:
        def __init__(self) -> None:
            self.calls = 0

        def set_tensor_address(self, name: str, address: int) -> None:
            assert name == "decoder_input"
            assert address > 0

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.tokens_log_prob.fill(-8.0)
            if self.calls % 2 == 0:
                decoder.tokens_log_prob[:, 1] = 0.0
                decoder.tokens_log_prob[:, 0] = -1.0
            else:
                decoder.tokens_log_prob[:, 0] = 0.0
            self.calls += 1
            return True

    decoder.decoder = FakeContext()
    encoder_output = cp.zeros((2, 2, 3), dtype=np.float32)
    encoder_output_lengths = cp.array([2, 1], dtype=np.int32)
    for _ in range(3):
        token_ids, timestamps = decoder(encoder_output, encoder_output_lengths)
        assert token_ids == [[1], [1]]
        np.testing.assert_allclose(timestamps, [[0.0], [0.0]])

    assert decoder.cuda_graph is not None
    assert decoder.decoder.calls == 4


@pytest.mark.cuda
@pytest.mark.parametrize(
    "encoder_dtype",
    (np.dtype(np.float32), np.dtype(np.float16), cp.dtype("bfloat16")),
    ids=("encoder-fp32", "encoder-fp16", "encoder-bf16"),
)
@pytest.mark.parametrize(
    "decoder_dtype",
    (np.dtype(np.float32), np.dtype(np.float16), cp.dtype("bfloat16")),
    ids=("decoder-fp32", "decoder-fp16", "decoder-bf16"),
)
def test_zipformer_decoder_converts_encoder_precision(
    encoder_dtype: np.dtype, decoder_dtype: np.dtype
) -> None:
    """Convert every supported encoder dtype to every decoder engine dtype."""

    decoder = make_fake_zipformer_decoder(
        encoder_dim=8,
        decoder_dtype=decoder_dtype,
        sequential_context_lookup=True,
    )

    class FakeContext:
        def __init__(self) -> None:
            self.calls = 0

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.tokens_log_prob.fill(-8.0)
            decoder.tokens_log_prob[:, 1 if self.calls == 0 else 0] = 0.0
            self.calls += 1
            return True

    decoder.decoder = FakeContext()
    encoder_output = (
        cp.arange(16, dtype=np.float32).astype(encoder_dtype).reshape(1, 2, 8)
    )
    token_ids, timestamps = decoder(
        encoder_output,
        cp.array([2], dtype=np.int32),
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.0]])
    np.testing.assert_array_equal(
        decoder.encoder_input.astype(cp.float32).get(),
        np.arange(8, 16, dtype=np.float32).reshape(1, 8),
    )
    np.testing.assert_array_equal(
        decoder.decoder_input.astype(cp.float32).get(),
        np.arange(16, 24, dtype=np.float32).reshape(1, 8),
    )


@pytest.mark.cuda
def test_zipformer_beam_one_decoder_keeps_search_on_gpu() -> None:
    """Decode beam-one RNN-T search entirely through the GPU search path."""

    decoder = make_fake_zipformer_decoder(
        batch_size=2,
        context_size=2,
        encoder_dim=3,
        vocab_size=4,
    )

    class FakeContext:
        def __init__(self) -> None:
            self.calls = 0

        def set_tensor_address(self, name: str, address: int) -> None:
            assert name == "decoder_input"
            assert address > 0

        def execute_async_v3(self, stream_ptr: int) -> bool:
            assert stream_ptr == decoder.stream.ptr
            decoder.tokens_log_prob.fill(-8.0)
            if self.calls == 0:
                decoder.tokens_log_prob[:, 1] = 0.0
                decoder.tokens_log_prob[:, 0] = -1.0
            else:
                decoder.tokens_log_prob[:, 0] = 0.0
            self.calls += 1
            return True

    decoder.decoder = FakeContext()
    token_ids, timestamps = decoder(
        cp.zeros((2, 2, 3), dtype=np.float32),
        cp.array([2, 1], dtype=np.int32),
    )

    assert token_ids == [[1], [1]]
    np.testing.assert_allclose(timestamps, [[0.0], [0.0]])


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


class RejectedProfileContext:
    """Record and reject TensorRT optimization-profile selection."""

    def __init__(self) -> None:
        self.profile_calls: list[tuple[int, int]] = []

    def set_optimization_profile_async(
        self, profile_index: int, stream_pointer: int
    ) -> bool:
        self.profile_calls.append((profile_index, stream_pointer))
        return False


class FakeZipformerEngine:
    """Expose only Zipformer metadata queried before context initialization."""

    shapes = {
        "decoder_input": (2, 4),
        "encoder_output": (2, 4),
        "tokens_log_prob": (2, 8),
    }

    def __init__(self, context: RejectedProfileContext | None) -> None:
        self.context = context
        self.shape_requests: list[str] = []

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        self.shape_requests.append(name)
        return self.shapes[name]

    def create_execution_context(self) -> RejectedProfileContext | None:
        return self.context


def make_ctc_validation_decoder() -> CTCGreedyDecoder:
    """Construct only the state needed by pre-CUDA CTC validation."""

    decoder = CTCGreedyDecoder.__new__(CTCGreedyDecoder)
    decoder.blank_id = 0
    return decoder


def make_zipformer_validation_decoder() -> ZipformerModifiedBeamSearchDecoder:
    """Construct only the state needed by pre-CUDA RNN-T validation."""

    decoder = ZipformerModifiedBeamSearchDecoder.__new__(
        ZipformerModifiedBeamSearchDecoder
    )
    decoder.batch_size = 2
    decoder.beam = 1
    decoder.encoder_dim = 4
    return decoder


def construct_zipformer_decoder(stream: NullCudaContext) -> None:
    """Initialize a Zipformer decoder through its context-setup boundary."""

    ZipformerModifiedBeamSearchDecoder(
        Path("decoder.trt"),
        batch_size=2,
        context_size=2,
        vocab_size=8,
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=cast(cp.cuda.Stream, stream),
    )


@pytest.mark.parametrize(
    ("log_probs", "output_lengths", "message"),
    (
        pytest.param(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            "rank-3",
            id="rank",
        ),
        pytest.param(
            np.zeros((0, 3, 4), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            "At least one CTC utterance",
            id="empty-batch",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.int32),
            np.zeros(2, dtype=np.int32),
            "float16, float32, or bfloat16",
            id="log-probability-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(3, dtype=np.int32),
            "output lengths",
            id="length-shape",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            "output lengths",
            id="length-dtype",
        ),
        pytest.param(
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros(4, dtype=np.int32)[::2],
            "contiguous int32 output lengths",
            id="noncontiguous-lengths",
        ),
    ),
)
def test_ctc_decoder_rejects_malformed_inputs(
    log_probs: np.ndarray,
    output_lengths: np.ndarray,
    message: str,
) -> None:
    """Reject malformed CTC inputs before entering a CUDA context."""

    decoder = make_ctc_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, log_probs),
            cast(cp.ndarray, output_lengths),
        )


def test_ctc_decoder_rejects_int32_frame_overflow() -> None:
    """Reject frame counts that CUDA kernels cannot index with int32 values."""

    log_probs = FakeCudaArray((1, INT32_MAX + 1, 2), np.dtype(np.float32))
    output_lengths = FakeCudaArray((1,), np.dtype(np.int32))

    with pytest.raises(ASRInferenceError, match="CTC frame count exceeds"):
        make_ctc_validation_decoder()(
            cast(cp.ndarray, log_probs),
            cast(cp.ndarray, output_lengths),
        )


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
def test_zipformer_decoder_rejects_malformed_inputs(
    encoder_output: np.ndarray,
    encoder_output_lengths: np.ndarray,
    message: str,
) -> None:
    """Reject malformed Zipformer encoder outputs before CUDA execution."""

    decoder = make_zipformer_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, encoder_output_lengths),
        )


def test_zipformer_decoder_rejects_int32_history_capacity_overflow() -> None:
    """Reject token-history buffers that exceed int32 kernel indexing."""

    decoder = make_zipformer_validation_decoder()
    decoder.beam = 2
    max_frames = INT32_MAX // (decoder.batch_size * decoder.beam) + 1
    encoder_output = FakeCudaArray(
        (1, max_frames, decoder.encoder_dim), np.dtype(np.float32)
    )
    output_lengths = FakeCudaArray((1,), np.dtype(np.int32))

    with pytest.raises(ASRInferenceError, match="token histories exceed"):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, output_lengths),
        )


def test_zipformer_decoder_rejects_int32_encoder_index_overflow() -> None:
    """Reject encoder tensors that exceed int32 kernel indexing."""

    decoder = make_zipformer_validation_decoder()
    max_frames = INT32_MAX // (decoder.batch_size * decoder.encoder_dim) + 1
    encoder_output = FakeCudaArray(
        (decoder.batch_size, max_frames, decoder.encoder_dim),
        np.dtype(np.float32),
    )
    output_lengths = FakeCudaArray((decoder.batch_size,), np.dtype(np.int32))

    with pytest.raises(ASRInferenceError, match="encoder output exceeds"):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, output_lengths),
        )


def test_zipformer_decoder_rejects_missing_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a model-specific error when TensorRT cannot create a context."""

    stream = NullCudaContext()
    engine = FakeZipformerEngine(None)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeZipformerEngine:
        loaded_paths.append(engine_path)
        return engine

    monkeypatch.setattr(zipformer_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(zipformer_decoder, "get_engine", load_engine)

    with pytest.raises(ASRInitializationError) as error:
        construct_zipformer_decoder(stream)

    assert str(error.value) == (
        "TensorRT could not create the Zipformer decoder execution context."
    )
    assert loaded_paths == [Path("decoder.trt")]
    assert sorted(engine.shape_requests) == sorted(engine.shapes)


def test_zipformer_decoder_rejects_profile_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a decoder context that cannot select optimization profile zero."""

    stream = NullCudaContext()
    context = RejectedProfileContext()
    engine = FakeZipformerEngine(context)
    loaded_paths: list[Path] = []

    def load_engine(engine_path: Path) -> FakeZipformerEngine:
        loaded_paths.append(engine_path)
        return engine

    monkeypatch.setattr(zipformer_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(zipformer_decoder, "get_engine", load_engine)

    with pytest.raises(ASRInitializationError) as error:
        construct_zipformer_decoder(stream)

    assert str(error.value) == (
        "TensorRT could not select Zipformer decoder optimization profile 0."
    )
    assert loaded_paths == [Path("decoder.trt")]
    assert sorted(engine.shape_requests) == sorted(engine.shapes)
    assert context.profile_calls == [(0, stream.ptr)]
