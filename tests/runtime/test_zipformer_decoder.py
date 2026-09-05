#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Runtime and validation tests for Zipformer CTC and RNN-T decoders."""

from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch

from fast_gpu_asr.constants import INT32_MAX, ZIPFORMER_DECODER_CONTEXTS_FILE
from fast_gpu_asr.decoder import gpu_kernels, zipformer_decoder
from fast_gpu_asr.decoder.gpu_kernels import (
    ZIPFORMER_FINALIZE_KERNEL,
    get_zipformer_beam_search_kernels,
)
from fast_gpu_asr.decoder.zipformer_decoder import (
    CTCGreedyDecoder,
    ZipformerModifiedBeamSearchDecoder,
)
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError


class ScriptedDecoderContext:
    """Populate decoder scores from a repeating script of TensorRT outputs."""

    def __init__(
        self,
        decoder: ZipformerModifiedBeamSearchDecoder,
        outputs: tuple[np.typing.ArrayLike, ...] = (),
        rejected_call: int = -1,
    ) -> None:
        """Initialize a scripted TensorRT execution context.

        Parameters
        ----------
        decoder : ZipformerModifiedBeamSearchDecoder
            Decoder whose output tensor receives the scripted scores.
        outputs : tuple[np.typing.ArrayLike, ...]
            Score rows replayed cyclically across decoder invocations.
        rejected_call : int
            Zero-based invocation that reports execution failure, or ``-1``
            when every invocation succeeds.
        """

        self.decoder = decoder
        with decoder.device, decoder.stream:
            self.outputs = tuple(
                cp.asarray(output, dtype=np.float32) for output in outputs
            )
        self.rejected_call = rejected_call
        self.calls = 0

    def execute_async_v3(self, stream_ptr: int) -> bool:
        """Write the next scripted output and report execution status.

        Parameters
        ----------
        stream_ptr : int
            CUDA stream pointer supplied by the decoder.

        Returns
        -------
        bool
            ``False`` only for the configured rejected invocation.
        """

        assert stream_ptr == self.decoder.stream.ptr
        call = self.calls
        self.calls += 1
        if self.outputs:
            cp.copyto(
                self.decoder.tokens_log_prob,
                self.outputs[call % len(self.outputs)],
            )
        return call != self.rejected_call


def make_ctc_decoder(blank_id: int = 0, blank_penalty: float = 0.0) -> CTCGreedyDecoder:
    """Create a CTC decoder on the test's current CUDA stream.

    Parameters
    ----------
    blank_id : int
        Vocabulary ID interpreted as CTC blank.
    blank_penalty : float
        Score subtracted from the blank token before greedy selection.

    Returns
    -------
    CTCGreedyDecoder
        Decoder configured for 40 ms encoder frames on CUDA device zero.
    """

    return CTCGreedyDecoder(
        blank_id=blank_id,
        encoder_frame_shift_sec=0.04,
        blank_penalty=blank_penalty,
        device_id=0,
        stream=cp.cuda.get_current_stream(),
    )


def initialize_zipformer_search_buffers(
    decoder: ZipformerModifiedBeamSearchDecoder,
    vocab_size: int,
) -> None:
    """Initialize buffers normally created by the TensorRT-backed constructor.

    Parameters
    ----------
    decoder : ZipformerModifiedBeamSearchDecoder
        Partially initialized decoder that receives search state and output buffers.
    vocab_size : int
        Number of token-score columns produced by the decoder engine.
    """

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
    ].astype(decoder.decoder_input.dtype, copy=False)
    decoder.frame_capacity = 0
    decoder.node_parents = None
    decoder.node_tokens = None
    decoder.node_timestamps = None


def make_fake_zipformer_decoder(
    batch_size: int = 1,
    beam: int = 1,
    context_size: int = 1,
    encoder_dim: int = 3,
    vocab_size: int = 4,
    blank_id: int = 0,
    decoder_dtype: np.dtype | None = None,
    context_dtype: np.dtype | None = None,
    sequential_context_lookup: bool = False,
) -> ZipformerModifiedBeamSearchDecoder:
    """Create a GPU decoder without loading a TensorRT engine.

    Parameters
    ----------
    batch_size : int
        Maximum number of utterances decoded together.
    beam : int
        Number of active hypotheses retained per utterance.
    context_size : int
        Number of preceding tokens represented by each predictor cache row.
    encoder_dim : int
        Width of encoder and cached predictor vectors.
    vocab_size : int
        Number of decoder token-score columns.
    blank_id : int
        Vocabulary ID interpreted as transducer blank.
    decoder_dtype : np.dtype | None
        Dtype used by decoder input buffers; defaults to ``np.float16``.
    context_dtype : np.dtype | None
        Dtype used by the predictor context cache; defaults to ``decoder_dtype``.
    sequential_context_lookup : bool
        Populate cache rows with distinguishable values instead of zeros.

    Returns
    -------
    ZipformerModifiedBeamSearchDecoder
        Decoder with CUDA search buffers initialized and no TensorRT context.
    """

    if decoder_dtype is None:
        decoder_dtype = np.dtype(np.float16)
    if context_dtype is None:
        context_dtype = decoder_dtype

    decoder = ZipformerModifiedBeamSearchDecoder.__new__(
        ZipformerModifiedBeamSearchDecoder
    )
    decoder.device = cp.cuda.Device(0)
    decoder.batch_size = batch_size
    decoder.beam = beam
    decoder.context_size = context_size
    decoder.encoder_dim = encoder_dim
    decoder.blank_id = blank_id
    decoder.encoder_frame_shift_sec = 0.04
    decoder.blank_penalty = 0.0
    decoder.stream = cp.cuda.get_current_stream()

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
            .astype(context_dtype)
        )
    else:
        decoder.context_lookup = cp.zeros(
            (context_rows, encoder_dim), dtype=context_dtype
        )
    decoder.encoder_input = cp.empty(
        (decoder_capacity, encoder_dim), dtype=decoder_dtype
    )
    decoder.tokens_log_prob = cp.empty((decoder_capacity, vocab_size), dtype=np.float32)
    initialize_zipformer_search_buffers(decoder, vocab_size)
    return decoder


@pytest.mark.parametrize(
    ("beam", "vocab_size", "register_supported"),
    (
        pytest.param(8, 512, True, id="register-upper-bound"),
        pytest.param(8, 513, False, id="register-storage-limit"),
        pytest.param(9, 1, False, id="register-beam-limit"),
    ),
)
def test_zipformer_beam_search_factory_builds_matching_launches(
    monkeypatch: pytest.MonkeyPatch,
    beam: int,
    vocab_size: int,
    register_supported: bool,
) -> None:
    created_kernels: list[SimpleNamespace] = []

    def create_kernel(
        source: str,
        name: str,
        options: tuple[str, ...],
        backend: str,
    ) -> SimpleNamespace:
        """Record one CuPy kernel compilation request.

        Parameters
        ----------
        source : str
            CUDA source passed to ``cp.RawKernel``.
        name : str
            Entry-point name requested from the source.
        options : tuple[str, ...]
            NVCC compiler options for the specialization.
        backend : str
            CuPy compiler backend selected by the factory.

        Returns
        -------
        SimpleNamespace
            Recorded kernel metadata used as a lightweight stand-in.
        """

        assert source == gpu_kernels.ZIPFORMER_BEAM_SEARCH_SOURCE
        kernel = SimpleNamespace(name=name, options=options, backend=backend)
        created_kernels.append(kernel)
        return kernel

    monkeypatch.setattr(gpu_kernels.cp, "RawKernel", create_kernel)
    get_zipformer_beam_search_kernels.cache_clear()
    try:
        launches = get_zipformer_beam_search_kernels(beam, vocab_size, 2)
        cached_launches = get_zipformer_beam_search_kernels(beam, vocab_size, 2)
    finally:
        get_zipformer_beam_search_kernels.cache_clear()

    register_launch, shared_launch, threads = launches
    register_memory = threads // 32 * 8 + beam * 8 + beam * 2 * 4
    shared_memory = register_memory + beam * vocab_size * 4

    assert cached_launches is launches
    assert threads == gpu_kernels.ZIPFORMER_BEAM_SEARCH_THREADS
    assert shared_launch[1] == shared_memory
    assert (register_launch is not None) is register_supported
    if register_launch is not None:
        assert register_launch[1] == register_memory

    expected_definitions = {
        f"-DZIPFORMER_BEAM={beam}",
        f"-DZIPFORMER_VOCAB_SIZE={vocab_size}",
        "-DZIPFORMER_CONTEXT_SIZE=2",
        f"-DZIPFORMER_BEAM_SEARCH_THREADS={threads}",
    }
    assert len(created_kernels) == 1 + int(register_supported)
    assert all(kernel.name == "zipformer_beam_search" for kernel in created_kernels)
    assert all(kernel.backend == "nvcc" for kernel in created_kernels)
    assert all(
        expected_definitions <= set(kernel.options) for kernel in created_kernels
    )
    assert any(
        "-DZIPFORMER_REGISTER_TOPK=0" in kernel.options for kernel in created_kernels
    )
    assert (
        any(
            "-DZIPFORMER_REGISTER_TOPK=1" in kernel.options
            for kernel in created_kernels
        )
        is register_supported
    )


@pytest.mark.parametrize(
    "threads",
    (
        pytest.param(0, id="zero"),
        pytest.param(31, id="not-warp-aligned"),
        pytest.param(1056, id="above-cuda-limit"),
        pytest.param(512.0, id="not-an-integer"),
    ),
)
def test_zipformer_beam_search_factory_rejects_invalid_thread_count(
    monkeypatch: pytest.MonkeyPatch,
    threads: int | float,
) -> None:
    monkeypatch.setattr(gpu_kernels, "ZIPFORMER_BEAM_SEARCH_THREADS", threads)
    get_zipformer_beam_search_kernels.cache_clear()
    try:
        with pytest.raises(ValueError, match="positive multiple of 32"):
            get_zipformer_beam_search_kernels(1, 1, 1)
    finally:
        get_zipformer_beam_search_kernels.cache_clear()


def test_zipformer_beam_search_factory_rejects_shared_memory_overflow() -> None:
    get_zipformer_beam_search_kernels.cache_clear()
    try:
        with pytest.raises(ValueError, match="dynamic shared memory exceeds"):
            get_zipformer_beam_search_kernels(1, 1, INT32_MAX // 4)
    finally:
        get_zipformer_beam_search_kernels.cache_clear()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(np.dtype(np.float32), id="fp32"),
        pytest.param(np.dtype(np.float16), id="fp16"),
        pytest.param(cp.dtype("bfloat16"), marks=pytest.mark.sm80, id="bf16"),
    ),
)
def test_ctc_greedy_collapses_repeats_and_blanks(dtype: np.dtype) -> None:
    decoder = make_ctc_decoder()
    paths = cp.array(
        [
            [0, 1, 1, 0, 1, 2],
            [2, 2, 0, 3, 3, 3],
        ],
    )
    log_probs = cp.eye(4, dtype=cp.float32)[paths].astype(dtype)

    token_ids, timestamps = decoder(
        log_probs,
        cp.array([6, 4], dtype=np.int32),
    )

    assert token_ids == [[1, 1, 2], [2, 3]]
    np.testing.assert_allclose(timestamps[0], [0.04, 0.16, 0.20])
    np.testing.assert_allclose(timestamps[1], [0.0, 0.12])


@pytest.mark.cuda
def test_ctc_greedy_applies_blank_penalty_before_argmax() -> None:
    decoder = make_ctc_decoder(blank_penalty=0.2)
    log_probs = cp.array([[[0.1, 0.0]]], dtype=cp.float32)

    token_ids, _ = decoder(log_probs, cp.array([1], dtype=np.int32))

    assert token_ids == [[1]]


@pytest.mark.cuda
def test_ctc_greedy_supports_nonzero_blank_id() -> None:
    decoder = make_ctc_decoder(blank_id=2)
    paths = cp.array([[2, 1, 1, 2, 1]])
    log_probs = cp.eye(3, dtype=cp.float32)[paths]

    token_ids, timestamps = decoder(log_probs, cp.array([5], dtype=np.int32))

    assert token_ids == [[1, 1]]
    np.testing.assert_allclose(timestamps, [[0.04, 0.16]])


@pytest.mark.cuda
def test_ctc_greedy_clamps_invalid_output_lengths() -> None:
    decoder = make_ctc_decoder()
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
def test_ctc_greedy_reuses_buffers_without_leaking_results() -> None:
    decoder = make_ctc_decoder()
    first_paths = cp.array([[1, 0, 2, 0], [2, 2, 0, 0]])
    first_tokens, first_timestamps = decoder(
        cp.eye(4, dtype=cp.float32)[first_paths],
        cp.full(2, 4, dtype=np.int32),
    )
    allocated_buffers = (
        decoder.emitted_tokens,
        decoder.emitted_timestamps,
        decoder.emitted_lengths,
        decoder.emitted_tokens_host,
        decoder.emitted_timestamps_host,
        decoder.emitted_lengths_host,
    )

    second_paths = cp.array([[3, 0, 1, 0], [0, 0, 0, 0]])
    second_tokens, second_timestamps = decoder(
        cp.eye(4, dtype=cp.float32)[second_paths],
        cp.full(2, 4, dtype=np.int32),
    )

    assert first_tokens == [[1, 2], [2]]
    np.testing.assert_allclose(first_timestamps[0], [0.0, 0.08])
    np.testing.assert_allclose(first_timestamps[1], [0.0])
    assert second_tokens == [[3, 1], []]
    np.testing.assert_allclose(second_timestamps[0], [0.0, 0.08])
    assert second_timestamps[1] == []
    current_buffers = (
        decoder.emitted_tokens,
        decoder.emitted_timestamps,
        decoder.emitted_lengths,
        decoder.emitted_tokens_host,
        decoder.emitted_timestamps_host,
        decoder.emitted_lengths_host,
    )
    assert all(
        current is allocated
        for current, allocated in zip(current_buffers, allocated_buffers, strict=True)
    )


@pytest.mark.cuda
def test_ctc_greedy_returns_empty_results_for_zero_frames() -> None:
    decoder = make_ctc_decoder()

    token_ids, timestamps = decoder(
        cp.empty((2, 0, 4), dtype=np.float32),
        cp.zeros(2, dtype=np.int32),
    )

    assert token_ids == [[], []]
    assert timestamps == [[], []]


@pytest.mark.cuda
def test_zipformer_register_and_shared_search_match() -> None:
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

    def run_search(
        launch: tuple[cp.RawKernel, int],
    ) -> tuple[np.typing.NDArray[np.generic], ...]:
        """Run one beam-search tactic and return its host-visible state.

        Parameters
        ----------
        launch : tuple[cp.RawKernel, int]
            Specialized kernel and its required dynamic shared-memory size.

        Returns
        -------
        tuple[np.typing.NDArray[np.generic], ...]
            Search outputs, contexts, history nodes, and per-utterance node counts.
        """

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
    np.testing.assert_array_equal(register_results[0][1], hypothesis_scores.get()[1])
    np.testing.assert_array_equal(register_results[1][1], [-1] * beam)
    np.testing.assert_array_equal(register_results[2][1], [0] * beam)
    np.testing.assert_array_equal(
        register_results[4][beam:], initial_contexts.get()[beam:]
    )
    assert register_results[8][1] == 0


@pytest.mark.cuda
@pytest.mark.parametrize(
    "use_register_search", (False, True), ids=("shared", "register")
)
def test_zipformer_search_updates_distinct_parent_histories(
    use_register_search: bool,
) -> None:
    beam = 2
    vocab_size = 4
    context_size = 2
    encoder_dim = 4
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, context_size
    )
    assert register_launch is not None
    kernel, shared_memory_bytes = (
        register_launch if use_register_search else shared_launch
    )

    encoder_input = cp.empty((beam, encoder_dim), dtype=np.float32)
    decoder_input = cp.empty_like(encoder_input)
    contexts = cp.array([[1, 2], [3, 1]], dtype=np.int32)
    next_scores = cp.empty((1, beam), dtype=np.float32)
    next_nodes = cp.empty((1, beam), dtype=np.int32)
    next_lengths = cp.empty((1, beam), dtype=np.int32)
    next_hashes = cp.empty((1, beam), dtype=np.uint64)
    node_parents = cp.full(beam * 3, -2, dtype=np.int32)
    node_tokens = cp.full(beam * 3, -2, dtype=np.int32)
    node_timestamps = cp.full(beam * 3, -2.0, dtype=np.float32)
    node_counts = cp.zeros(1, dtype=np.int32)
    context_lookup = cp.arange(
        (vocab_size + 1) ** context_size * encoder_dim,
        dtype=np.float32,
    ).reshape(-1, encoder_dim)

    kernel(
        (1,),
        (threads,),
        (
            cp.array(
                [
                    [-0.4, -5.0, -5.0, -0.1],
                    [-0.6, -5.0, 0.0, -5.0],
                ],
                dtype=np.float32,
            ),
            cp.arange(12, dtype=np.float32).reshape(1, 3, encoder_dim),
            encoder_input,
            context_lookup,
            decoder_input,
            contexts,
            cp.array([[0.0, -0.2]], dtype=np.float32),
            cp.full((1, beam), -1, dtype=np.int32),
            cp.zeros((1, beam), dtype=np.int32),
            cp.zeros((1, beam), dtype=np.uint64),
            next_scores,
            next_nodes,
            next_lengths,
            next_hashes,
            node_parents,
            node_tokens,
            node_timestamps,
            node_counts,
            cp.array([3], dtype=np.int32),
            np.int32(1),
            np.int32(3),
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

    np.testing.assert_allclose(next_scores.get(), [[-0.1, -0.2]], atol=1e-7)
    np.testing.assert_array_equal(next_nodes.get(), [[0, 1]])
    np.testing.assert_array_equal(next_lengths.get(), [[1, 1]])
    np.testing.assert_array_equal(contexts.get(), [[2, 3], [1, 2]])
    np.testing.assert_array_equal(node_parents.get()[:2], [-1, -1])
    np.testing.assert_array_equal(node_tokens.get()[:2], [3, 2])
    np.testing.assert_allclose(node_timestamps.get()[:2], [0.04, 0.04])
    np.testing.assert_array_equal(node_counts.get(), [2])
    np.testing.assert_array_equal(
        encoder_input.get(),
        np.tile(np.arange(8, 12, dtype=np.float32), (beam, 1)),
    )
    np.testing.assert_array_equal(
        decoder_input.get(),
        np.vstack(
            (
                np.arange(76, 80, dtype=np.float32),
                np.arange(52, 56, dtype=np.float32),
            )
        ),
    )


@pytest.mark.cuda
@pytest.mark.parametrize("invalid_score", (-np.inf, np.nan))
def test_zipformer_search_keeps_nonfinite_candidates_in_bounds(
    invalid_score: float,
) -> None:
    beam = 2
    vocab_size = 3
    context_size = 1
    encoder_dim = 4
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, context_size
    )
    assert register_launch is not None

    def run_search(
        launch: tuple[cp.RawKernel, int],
    ) -> tuple[np.typing.NDArray[np.generic], ...]:
        """Run one beam-search tactic against non-finite candidate scores.

        Parameters
        ----------
        launch : tuple[cp.RawKernel, int]
            Specialized kernel and its required dynamic shared-memory size.

        Returns
        -------
        tuple[np.typing.NDArray[np.generic], ...]
            Host copies of bounded search outputs and history state.
        """

        kernel, shared_memory_bytes = launch
        next_scores = cp.empty((1, beam), dtype=np.float32)
        next_nodes = cp.empty((1, beam), dtype=np.int32)
        next_lengths = cp.empty((1, beam), dtype=np.int32)
        next_hashes = cp.empty((1, beam), dtype=np.uint64)
        node_counts = cp.zeros(1, dtype=np.int32)
        contexts = cp.zeros((beam, context_size), dtype=np.int32)
        node_parents = cp.full(beam, -2, dtype=np.int32)
        node_tokens = cp.full(beam, -2, dtype=np.int32)
        node_timestamps = cp.full(beam, -2.0, dtype=np.float32)
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
                node_parents,
                node_tokens,
                node_timestamps,
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
        return tuple(
            array.get()
            for array in (
                next_scores,
                next_nodes,
                next_lengths,
                contexts,
                node_counts,
            )
        )

    register_results = run_search(register_launch)
    shared_results = run_search(shared_launch)
    for register_result, shared_result in zip(
        register_results, shared_results, strict=True
    ):
        np.testing.assert_array_equal(register_result, shared_result)
    np.testing.assert_array_equal(register_results[0], [[-np.inf, -np.inf]])
    np.testing.assert_array_equal(register_results[1], [[-1, 0]])
    np.testing.assert_array_equal(register_results[2], [[0, 1]])
    np.testing.assert_array_equal(register_results[3], [[0], [1]])
    np.testing.assert_array_equal(register_results[4], [1])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "use_register_search", (False, True), ids=("shared", "register")
)
def test_zipformer_search_merges_and_resorts_duplicate_histories(
    use_register_search: bool,
) -> None:
    beam = 3
    vocab_size = 3
    register_launch, shared_launch, threads = get_zipformer_beam_search_kernels(
        beam, vocab_size, 1
    )
    assert register_launch is not None
    kernel, shared_memory_bytes = (
        register_launch if use_register_search else shared_launch
    )

    next_scores = cp.empty((1, beam), dtype=np.float32)
    next_nodes = cp.empty((1, beam), dtype=np.int32)
    next_lengths = cp.empty((1, beam), dtype=np.int32)
    next_hashes = cp.empty((1, beam), dtype=np.uint64)
    node_parents = cp.array([-1, -2, -2, -2, -2, -2], dtype=np.int32)
    node_tokens = cp.array([1, -2, -2, -2, -2, -2], dtype=np.int32)
    node_timestamps = cp.array([0.0, -2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    node_counts = cp.array([1], dtype=np.int32)
    contexts = cp.array([[1], [0], [0]], dtype=np.int32)

    kernel(
        (1,),
        (threads,),
        (
            cp.array(
                [
                    [-0.2, -8.0, -0.1],
                    [-8.0, -0.3, -8.0],
                    [-8.0, -8.0, -8.0],
                ],
                dtype=np.float32,
            ),
            cp.zeros((1, 2, 1), dtype=np.float32),
            cp.empty((beam, 1), dtype=np.float32),
            cp.zeros((vocab_size + 1, 1), dtype=np.float32),
            cp.empty((beam, 1), dtype=np.float32),
            contexts,
            cp.array([[0.0, 0.0, -np.inf]], dtype=np.float32),
            cp.array([[0, -1, -1]], dtype=np.int32),
            cp.array([[1, 0, 0]], dtype=np.int32),
            cp.array([[2, 0, 0]], dtype=np.uint64),
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

    expected_score = np.logaddexp(np.float32(-0.2), np.float32(-0.3))
    scores = next_scores.get()
    np.testing.assert_allclose(scores[0, 0], expected_score, atol=1e-7)
    np.testing.assert_allclose(scores[0, 1], -0.1, atol=1e-7)
    assert np.isneginf(scores[0, 2])
    np.testing.assert_array_equal(next_nodes.get(), [[0, 1, -1]])
    np.testing.assert_array_equal(next_lengths.get(), [[1, 2, 0]])
    np.testing.assert_array_equal(contexts.get(), [[1], [2], [0]])
    np.testing.assert_array_equal(node_parents.get()[:2], [-1, 0])
    np.testing.assert_array_equal(node_tokens.get()[:2], [1, 2])
    np.testing.assert_allclose(node_timestamps.get()[:2], [0.0, 0.04])
    np.testing.assert_array_equal(node_counts.get(), [2])


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("scores", "expected_tokens", "expected_timestamps"),
    (
        pytest.param((-1.0, -1.2), (3, 4), (0.0, 0.04), id="length-normalization"),
        pytest.param((-1.0, -1.5), (3,), (0.0,), id="context-size-denominator"),
    ),
)
def test_zipformer_finalize_uses_icefall_length_normalization(
    scores: tuple[float, float],
    expected_tokens: tuple[int, ...],
    expected_timestamps: tuple[float, ...],
) -> None:
    output_tokens = cp.empty((1, 3), dtype=np.int32)
    output_timestamps = cp.empty((1, 3), dtype=np.float32)
    output_lengths = cp.empty(1, dtype=np.int32)

    ZIPFORMER_FINALIZE_KERNEL(
        (1,),
        (1,),
        (
            cp.array([scores], dtype=np.float32),
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

    expected_length = len(expected_tokens)
    np.testing.assert_array_equal(output_lengths.get(), [expected_length])
    np.testing.assert_array_equal(
        output_tokens.get()[0, :expected_length], expected_tokens
    )
    np.testing.assert_allclose(
        output_timestamps.get()[0, :expected_length], expected_timestamps
    )


@pytest.mark.cuda
def test_zipformer_decoder_returns_empty_results_for_zero_frames() -> None:
    decoder = make_fake_zipformer_decoder(batch_size=2)

    token_ids, timestamps = decoder(
        cp.empty((2, 0, 3), dtype=np.float32),
        cp.zeros(2, dtype=np.int32),
    )

    assert token_ids == [[], []]
    assert timestamps == [[], []]


@pytest.mark.cuda
def test_zipformer_decoder_clamps_invalid_output_lengths() -> None:
    decoder = make_fake_zipformer_decoder(batch_size=2)
    decoder.decoder = ScriptedDecoderContext(decoder, ([-8.0, 0.0, -8.0, -8.0],))
    token_ids, timestamps = decoder(
        cp.zeros((2, 2, 3), dtype=np.float32),
        cp.array([10, -1], dtype=np.int32),
    )

    assert token_ids == [[1, 1], []]
    np.testing.assert_allclose(timestamps[0], [0.0, 0.04])
    assert timestamps[1] == []


@pytest.mark.cuda
def test_zipformer_decoder_does_not_leak_results_across_calls() -> None:
    decoder = make_fake_zipformer_decoder(batch_size=2)
    decoder.decoder = ScriptedDecoderContext(decoder, ([-8.0, 0.0, -8.0, -8.0],))
    first_tokens, first_timestamps = decoder(
        cp.zeros((2, 3, 3), dtype=np.float32),
        cp.array([3, 2], dtype=np.int32),
    )
    empty_tokens, empty_timestamps = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.zeros(1, dtype=np.int32),
    )
    final_tokens, final_timestamps = decoder(
        cp.zeros((2, 4, 3), dtype=np.float32),
        cp.array([4, 1], dtype=np.int32),
    )

    assert first_tokens == [[1, 1, 1], [1, 1]]
    np.testing.assert_allclose(first_timestamps[0], [0.0, 0.04, 0.08])
    np.testing.assert_allclose(first_timestamps[1], [0.0, 0.04])
    assert empty_tokens == [[]]
    assert empty_timestamps == [[]]
    assert final_tokens == [[1, 1, 1, 1], [1]]
    np.testing.assert_allclose(final_timestamps[0], [0.0, 0.04, 0.08, 0.12])
    np.testing.assert_allclose(final_timestamps[1], [0.0])


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
    decoder = make_fake_zipformer_decoder()
    decoder.blank_penalty = blank_penalty
    decoder.decoder = ScriptedDecoderContext(decoder, ([0.1, 0.0, -8.0, -8.0],))
    token_ids, _ = decoder(
        cp.zeros((1, 1, 3), dtype=np.float32),
        cp.ones(1, dtype=np.int32),
    )

    assert token_ids == [expected_tokens]


@pytest.mark.cuda
def test_zipformer_decoder_supports_nonzero_blank_id() -> None:
    decoder = make_fake_zipformer_decoder(blank_id=3)
    decoder.decoder = ScriptedDecoderContext(
        decoder,
        (
            [-8.0, -8.0, -8.0, 0.0],
            [-8.0, 0.0, -8.0, -8.0],
        ),
    )
    token_ids, timestamps = decoder(
        cp.zeros((1, 2, 3), dtype=np.float32),
        cp.array([2], dtype=np.int32),
    )

    assert token_ids == [[1]]
    np.testing.assert_allclose(timestamps, [[0.04]])


@pytest.mark.cuda
def test_zipformer_decoder_keeps_batch_candidates_separate() -> None:
    decoder = make_fake_zipformer_decoder(batch_size=2)
    decoder.decoder = ScriptedDecoderContext(
        decoder,
        ([[-1.0, 0.0, -8.0, -8.0], [-1.0, -8.0, 0.0, -8.0]],),
    )
    token_ids, _ = decoder(
        cp.zeros((2, 1, 3), dtype=np.float32),
        cp.ones(2, dtype=np.int32),
    )

    assert token_ids == [[1], [2]]


@pytest.mark.cuda
def test_zipformer_decoder_clears_inactive_encoder_slots() -> None:
    decoder = make_fake_zipformer_decoder(batch_size=3, beam=2)
    decoder.decoder = ScriptedDecoderContext(
        decoder,
        (
            [-8.0, 0.0, -8.0, -8.0],
            [0.0, -8.0, -8.0, -8.0],
        ),
    )
    encoder_output = cp.arange(6, dtype=np.float32).reshape(1, 2, 3)
    token_ids, _ = decoder(
        encoder_output,
        cp.array([2], dtype=np.int32),
    )

    assert token_ids == [[1]]
    staged_encoder = decoder.encoder_input.reshape(3, 2, 3).get()
    np.testing.assert_array_equal(
        staged_encoder[0], np.tile(np.arange(3, 6, dtype=np.float32), (2, 1))
    )
    np.testing.assert_array_equal(staged_encoder[1:], np.zeros((2, 2, 3)))


@pytest.mark.cuda
def test_zipformer_decoder_reports_tensorrt_execution_failure() -> None:
    decoder = make_fake_zipformer_decoder()
    decoder.decoder = ScriptedDecoderContext(decoder, rejected_call=0)
    with pytest.raises(ASRInferenceError, match="TensorRT decoder execution failed"):
        decoder(
            cp.zeros((1, 1, 3), dtype=np.float32),
            cp.ones(1, dtype=np.int32),
        )


@pytest.mark.cuda
def test_zipformer_decoder_falls_back_after_captured_execution_failure() -> None:
    decoder = make_fake_zipformer_decoder()
    decoder.cuda_graph_supported = True
    decoder.decoder = ScriptedDecoderContext(
        decoder,
        ([-8.0, 0.0, -8.0, -8.0],),
        rejected_call=3,
    )
    encoder_output = cp.zeros((1, 2, 3), dtype=np.float32)
    encoder_output_lengths = cp.full(1, 2, dtype=np.int32)

    first_tokens, _ = decoder(encoder_output, encoder_output_lengths)
    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        retried_tokens, _ = decoder(encoder_output, encoder_output_lengths)

    assert first_tokens == [[1, 1]]
    assert retried_tokens == [[1, 1]]
    assert decoder.decoder.calls == 6
    assert not decoder.cuda_graph_supported
    assert decoder.cuda_graph is None
    assert decoder.cuda_graph_signature is None


@pytest.mark.cuda
@pytest.mark.parametrize("status", (901, 999), ids=("invalidated", "unexpected"))
def test_zipformer_decoder_handles_cuda_capture_errors(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    class CaptureError(RuntimeError):
        """Carry the CUDA status exposed by CuPy capture failures."""

        def __init__(self, status: int) -> None:
            """Initialize an error with one CUDA runtime status.

            Parameters
            ----------
            status : int
                CUDA error code exposed through the ``status`` attribute.
            """

            super().__init__(f"CUDA capture error {status}")
            self.status = status

    class InvalidatingStream(cp.cuda.Stream):
        """Finish capture, then report CUDA's invalidated-capture status."""

        def end_capture(self) -> cp.cuda.graph.Graph:
            """End stream capture and raise the configured CUDA error.

            Raises
            ------
            CaptureError
                Always raised after CuPy finishes the active capture.
            """

            super().end_capture()
            raise CaptureError(status)

    monkeypatch.setattr(
        zipformer_decoder.cp.cuda.runtime,
        "CUDARuntimeError",
        CaptureError,
    )
    decoder = make_fake_zipformer_decoder()
    decoder.stream = InvalidatingStream(non_blocking=True)
    decoder.cuda_graph_supported = True
    decoder.decoder = ScriptedDecoderContext(decoder, ([-8.0, 0.0, -8.0, -8.0],))
    encoder_output = cp.zeros((1, 1, 3), dtype=np.float32)
    output_lengths = cp.ones(1, dtype=np.int32)
    cp.cuda.get_current_stream().synchronize()
    decoder(encoder_output, output_lengths)

    if status != 901:
        with pytest.raises(CaptureError, match=str(status)):
            decoder(encoder_output, output_lengths)
        return

    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        token_ids, _ = decoder(encoder_output, output_lengths)

    assert token_ids == [[1]]
    assert not decoder.cuda_graph_supported
    assert decoder.cuda_graph is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "use_register_search",
    (False, True),
    ids=("shared", "register"),
)
def test_zipformer_decoder_uses_live_graph_inputs_and_invalidates_changed_buffers(
    use_register_search: bool,
) -> None:
    decoder = make_fake_zipformer_decoder(
        batch_size=2,
        beam=2,
        context_size=1,
        encoder_dim=3,
        vocab_size=4,
    )
    decoder.beam_search_register_batch_limit = 64 if use_register_search else 0
    decoder.cuda_graph_supported = True
    assert decoder.register_beam_search is not None

    kernel_calls = {"register": 0, "shared": 0}

    def record_launch(
        name: str, launch: tuple[cp.RawKernel, int]
    ) -> tuple[cp.RawKernel, int]:
        """Wrap one search tactic and count its kernel launches.

        Parameters
        ----------
        name : str
            Counter key identifying the register or shared-memory tactic.
        launch : tuple[cp.RawKernel, int]
            Kernel and dynamic shared-memory size returned by the factory.

        Returns
        -------
        tuple[cp.RawKernel, int]
            Counting kernel wrapper and the unchanged shared-memory size.
        """

        kernel, shared_memory_bytes = launch

        def recording_kernel(*args: Any, **kwargs: Any) -> Any:
            """Count and forward one CuPy kernel invocation.

            Parameters
            ----------
            *args : Any
                Positional launch arguments forwarded to the wrapped kernel.
            **kwargs : Any
                Keyword launch arguments forwarded to the wrapped kernel.

            Returns
            -------
            Any
                Return value produced by the wrapped CuPy kernel.
            """

            kernel_calls[name] += 1
            return kernel(*args, **kwargs)

        return cast(cp.RawKernel, recording_kernel), shared_memory_bytes

    decoder.register_beam_search = record_launch(
        "register", decoder.register_beam_search
    )
    decoder.shared_beam_search = record_launch("shared", decoder.shared_beam_search)

    decoder.decoder = ScriptedDecoderContext(
        decoder,
        (
            [-1.0, 0.0, -8.0, -8.0],
            [-1.0, -8.0, 0.0, -8.0],
        ),
    )
    encoder_output = cp.zeros((2, 2, 3), dtype=np.float32)
    encoder_output_lengths = cp.array([2, 1], dtype=np.int32)
    for _ in range(3):
        token_ids, _ = decoder(encoder_output, encoder_output_lengths)
        assert token_ids == [[1, 2], [1]]

    captured_graph = decoder.cuda_graph
    assert captured_graph is not None
    assert decoder.decoder.calls == 4

    with decoder.stream:
        encoder_output[...] = cp.arange(12, dtype=np.float32).reshape(2, 2, 3)
        encoder_output_lengths[...] = cp.array([1, 2], dtype=np.int32)
    token_ids, _ = decoder(encoder_output, encoder_output_lengths)

    assert token_ids == [[1], [1, 2]]
    np.testing.assert_array_equal(
        decoder.encoder_input.reshape(2, 2, 3).get(),
        [
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            [[9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
        ],
    )
    assert decoder.cuda_graph is captured_graph
    assert decoder.decoder.calls == 4

    with decoder.stream:
        replacement_output = cp.zeros_like(encoder_output)
        replacement_lengths = cp.array([2, 1], dtype=np.int32)
    token_ids, _ = decoder(replacement_output, replacement_lengths)

    assert token_ids == [[1, 2], [1]]
    assert decoder.cuda_graph is None
    assert decoder.decoder.calls == 6
    selected_kernel = "register" if use_register_search else "shared"
    unselected_kernel = "shared" if use_register_search else "register"
    assert kernel_calls[selected_kernel] > 0
    assert kernel_calls[unselected_kernel] == 0


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("encoder_dtype", "context_dtype", "decoder_dtype"),
    (
        pytest.param(
            np.dtype(np.float32),
            np.dtype(np.float32),
            np.dtype(np.float32),
            id="all-fp32",
        ),
        pytest.param(
            np.dtype(np.float16),
            np.dtype(np.float16),
            np.dtype(np.float16),
            id="all-fp16",
        ),
        pytest.param(
            cp.dtype("bfloat16"),
            cp.dtype("bfloat16"),
            cp.dtype("bfloat16"),
            marks=pytest.mark.sm80,
            id="all-bf16",
        ),
        pytest.param(
            np.dtype(np.float32),
            np.dtype(np.float16),
            cp.dtype("bfloat16"),
            marks=pytest.mark.sm80,
            id="fp32-fp16-to-bf16",
        ),
        pytest.param(
            np.dtype(np.float16),
            cp.dtype("bfloat16"),
            np.dtype(np.float32),
            marks=pytest.mark.sm80,
            id="fp16-bf16-to-fp32",
        ),
        pytest.param(
            cp.dtype("bfloat16"),
            np.dtype(np.float32),
            np.dtype(np.float16),
            marks=pytest.mark.sm80,
            id="bf16-fp32-to-fp16",
        ),
    ),
)
def test_zipformer_decoder_converts_inputs_to_engine_precision(
    encoder_dtype: np.dtype,
    context_dtype: np.dtype,
    decoder_dtype: np.dtype,
) -> None:
    decoder = make_fake_zipformer_decoder(
        context_size=2,
        encoder_dim=8,
        decoder_dtype=decoder_dtype,
        context_dtype=context_dtype,
        sequential_context_lookup=True,
    )
    decoder.decoder = ScriptedDecoderContext(
        decoder,
        (
            [-8.0, 0.0, -8.0, -8.0],
            [0.0, -8.0, -8.0, -8.0],
        ),
    )
    encoder_output = (
        cp.arange(16, dtype=np.float32).astype(encoder_dtype).reshape(1, 2, 8)
    )
    token_ids, _ = decoder(
        encoder_output,
        cp.array([2], dtype=np.int32),
    )

    assert token_ids == [[1]]
    np.testing.assert_array_equal(
        decoder.encoder_input.astype(cp.float32).get(),
        np.arange(8, 16, dtype=np.float32).reshape(1, 8),
    )
    np.testing.assert_array_equal(
        decoder.decoder_input.astype(cp.float32).get(),
        np.arange(56, 64, dtype=np.float32).reshape(1, 8),
    )


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
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Exit the no-op CUDA context.

        Parameters
        ----------
        _exc_type : type[BaseException] | None
            Exception class leaving the context, when present.
        _exc_value : BaseException | None
            Exception instance leaving the context, when present.
        _traceback : TracebackType | None
            Exception traceback leaving the context, when present.
        """

        pass


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


class RecordingZipformerContext:
    """Record profile selection and fixed TensorRT tensor bindings."""

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


class FakeZipformerEngine:
    """Expose only Zipformer metadata queried before context initialization."""

    shapes = {
        "decoder_input": (2, 4),
        "encoder_output": (2, 4),
        "tokens_log_prob": (2, 8),
    }

    def __init__(
        self,
        context: RecordingZipformerContext | None,
        tensor_dtype: trt.DataType = trt.float32,
    ) -> None:
        """Initialize fake engine metadata and execution context.

        Parameters
        ----------
        context : RecordingZipformerContext | None
            Context returned during decoder initialization, or ``None`` to
            simulate context-creation failure.
        tensor_dtype : trt.DataType
            Floating-point dtype reported for decoder inputs.
        """

        self.context = context
        self.tensor_dtype = tensor_dtype

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return the fixed shape of one decoder tensor.

        Parameters
        ----------
        name : str
            TensorRT tensor name.

        Returns
        -------
        tuple[int, ...]
            Static dimensions configured for ``name``.
        """

        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        """Return the configured floating-point decoder dtype.

        Parameters
        ----------
        name : str
            Decoder input tensor whose dtype is requested.

        Returns
        -------
        trt.DataType
            Configured TensorRT floating-point dtype.
        """

        assert name in ("decoder_input", "encoder_output")
        return self.tensor_dtype

    def create_execution_context(
        self,
    ) -> RecordingZipformerContext | None:
        """Return the configured fake TensorRT execution context.

        Returns
        -------
        RecordingZipformerContext | None
            Context supplied at engine construction.
        """

        return self.context


def make_ctc_validation_decoder() -> CTCGreedyDecoder:
    """Construct only the state needed by pre-CUDA CTC validation.

    Returns
    -------
    CTCGreedyDecoder
        Uninitialized decoder carrying a valid blank-token ID.
    """

    decoder = CTCGreedyDecoder.__new__(CTCGreedyDecoder)
    decoder.blank_id = 0
    return decoder


def make_zipformer_validation_decoder() -> ZipformerModifiedBeamSearchDecoder:
    """Construct only the state needed by pre-CUDA RNN-T validation.

    Returns
    -------
    ZipformerModifiedBeamSearchDecoder
        Uninitialized decoder carrying valid shape and capacity metadata.
    """

    decoder = ZipformerModifiedBeamSearchDecoder.__new__(
        ZipformerModifiedBeamSearchDecoder
    )
    decoder.batch_size = 2
    decoder.beam = 1
    decoder.encoder_dim = 4
    return decoder


def construct_zipformer_decoder(
    stream: NullCudaContext,
) -> ZipformerModifiedBeamSearchDecoder:
    """Initialize a Zipformer decoder through its TensorRT setup boundary.

    Parameters
    ----------
    stream : NullCudaContext
        No-op stream supplied to constructor validation tests.

    Returns
    -------
    ZipformerModifiedBeamSearchDecoder
        Decoder initialized against the monkeypatched TensorRT engine.
    """

    return ZipformerModifiedBeamSearchDecoder(
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
    log_probs: np.typing.NDArray[np.generic],
    output_lengths: np.typing.NDArray[np.generic],
    message: str,
) -> None:
    decoder = make_ctc_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, log_probs),
            cast(cp.ndarray, output_lengths),
        )


def test_ctc_decoder_rejects_int32_frame_overflow() -> None:
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
    encoder_output: np.typing.NDArray[np.generic],
    encoder_output_lengths: np.typing.NDArray[np.generic],
    message: str,
) -> None:
    decoder = make_zipformer_validation_decoder()

    with pytest.raises(ASRInferenceError, match=message):
        decoder(
            cast(cp.ndarray, encoder_output),
            cast(cp.ndarray, encoder_output_lengths),
        )


def test_zipformer_decoder_rejects_int32_history_capacity_overflow() -> None:
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


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("context_dtype", "engine_dtype"),
    (
        pytest.param(torch.float32, trt.float16, id="context-fp32-engine-fp16"),
        pytest.param(
            torch.float16,
            trt.bfloat16,
            marks=pytest.mark.sm80,
            id="context-fp16-engine-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            trt.float32,
            marks=pytest.mark.sm80,
            id="context-bf16-engine-fp32",
        ),
        pytest.param(
            torch.bfloat16,
            trt.bfloat16,
            marks=pytest.mark.sm80,
            id="context-bf16-engine-bf16",
        ),
    ),
)
def test_zipformer_decoder_initializes_context_cache_and_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_dtype: torch.dtype,
    engine_dtype: trt.DataType,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    context_lookup = torch.arange(81 * 4, dtype=torch.float32).reshape(81, 4) / 7.0
    torch.save(
        context_lookup.to(context_dtype),
        tmp_path / ZIPFORMER_DECODER_CONTEXTS_FILE,
    )
    context = RecordingZipformerContext()
    engine = FakeZipformerEngine(context, engine_dtype)
    monkeypatch.setattr(zipformer_decoder, "get_engine", lambda _path: engine)
    stream = cp.cuda.get_current_stream()
    decoder = ZipformerModifiedBeamSearchDecoder(
        engine_path,
        batch_size=2,
        context_size=2,
        vocab_size=8,
        blank_id=0,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=stream,
    )

    expected_context_dtype = (
        cp.dtype("bfloat16")
        if context_dtype == torch.bfloat16
        else np.dtype(np.float16 if context_dtype == torch.float16 else np.float32)
    )
    expected_engine_dtype = (
        cp.dtype("bfloat16")
        if engine_dtype == trt.bfloat16
        else np.dtype(np.float16 if engine_dtype == trt.float16 else np.float32)
    )
    engine_torch_dtype = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
    }[engine_dtype]
    expected_context_lookup = context_lookup.to(context_dtype).to(torch.float32).numpy()
    expected_initial_decoder_input = (
        context_lookup.to(context_dtype)
        .to(engine_torch_dtype)
        .to(torch.float32)
        .numpy()
    )
    assert context.profile_calls == [(0, stream.ptr)]
    assert context.bindings == {
        "decoder_input": decoder.decoder_input.data.ptr,
        "encoder_output": decoder.encoder_input.data.ptr,
        "tokens_log_prob": decoder.tokens_log_prob.data.ptr,
    }
    assert decoder.decoder_input.dtype == expected_engine_dtype
    assert decoder.encoder_input.dtype == expected_engine_dtype
    assert decoder.context_lookup.dtype == expected_context_dtype
    assert decoder.initial_decoder_input.dtype == expected_engine_dtype
    assert decoder.tokens_log_prob.dtype == np.float32
    np.testing.assert_array_equal(
        decoder.context_lookup.astype(cp.float32).get(),
        expected_context_lookup,
    )
    np.testing.assert_array_equal(
        decoder.initial_contexts.get(),
        [[-1, 0], [-1, 0]],
    )
    np.testing.assert_array_equal(
        decoder.initial_decoder_input.astype(cp.float32).get(),
        np.tile(expected_initial_decoder_input[1], (2, 1)),
    )


@pytest.mark.cuda
def test_zipformer_decoder_initializes_every_wide_beam_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_path = tmp_path / "decoder.trt"
    context_lookup = torch.arange(81 * 4, dtype=torch.float32).reshape(81, 4)
    torch.save(context_lookup, tmp_path / ZIPFORMER_DECODER_CONTEXTS_FILE)
    context = RecordingZipformerContext()
    engine = FakeZipformerEngine(context)
    engine.shapes = {
        "decoder_input": (6, 4),
        "encoder_output": (6, 4),
        "tokens_log_prob": (6, 8),
    }
    monkeypatch.setattr(zipformer_decoder, "get_engine", lambda _path: engine)

    decoder = ZipformerModifiedBeamSearchDecoder(
        engine_path,
        batch_size=2,
        context_size=2,
        vocab_size=8,
        blank_id=3,
        encoder_frame_shift_sec=0.04,
        blank_penalty=0.0,
        device_id=0,
        stream=cp.cuda.get_current_stream(),
    )

    assert decoder.beam == 3
    np.testing.assert_array_equal(
        decoder.initial_contexts.get(),
        [
            [-1, 3],
            [0, 0],
            [0, 0],
            [-1, 3],
            [0, 0],
            [0, 0],
        ],
    )
    np.testing.assert_array_equal(
        decoder.initial_decoder_input.get(),
        context_lookup[[4, 10, 10, 4, 10, 10]].numpy(),
    )
    assert decoder.hypothesis_scores.shape == (2, 3)
    assert decoder.contexts.shape == (6, 2)


def test_zipformer_decoder_rejects_missing_execution_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = NullCudaContext()
    engine = FakeZipformerEngine(None)
    monkeypatch.setattr(zipformer_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(zipformer_decoder, "get_engine", lambda _path: engine)

    with pytest.raises(
        ASRInitializationError, match="could not create the Zipformer decoder"
    ):
        construct_zipformer_decoder(stream)


def test_zipformer_decoder_rejects_profile_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = NullCudaContext()
    context = RecordingZipformerContext(profile_accepted=False)
    engine = FakeZipformerEngine(context)
    monkeypatch.setattr(zipformer_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(zipformer_decoder, "get_engine", lambda _path: engine)

    with pytest.raises(ASRInitializationError, match="optimization profile 0"):
        construct_zipformer_decoder(stream)

    assert context.profile_calls == [(0, stream.ptr)]


@pytest.mark.parametrize(
    "rejected_binding",
    ("decoder_input", "encoder_output", "tokens_log_prob"),
)
def test_zipformer_decoder_rejects_tensor_binding_failure(
    monkeypatch: pytest.MonkeyPatch,
    rejected_binding: str,
) -> None:
    stream = NullCudaContext()
    context = RecordingZipformerContext(rejected_binding)
    engine = FakeZipformerEngine(context)

    def allocate(_shape: tuple[int, ...], dtype: type[np.float32]) -> SimpleNamespace:
        """Return a pointer-bearing stand-in for one CuPy allocation.

        Parameters
        ----------
        _shape : tuple[int, ...]
            Allocation shape accepted for compatibility with ``cp.empty``.
        dtype : type[np.float32]
            Requested NumPy scalar type.

        Returns
        -------
        SimpleNamespace
            Allocation stand-in exposing ``data.ptr``.
        """

        assert dtype is np.float32
        return SimpleNamespace(data=SimpleNamespace(ptr=1))

    monkeypatch.setattr(zipformer_decoder.cp.cuda, "Device", NullCudaContext)
    monkeypatch.setattr(zipformer_decoder.cp, "empty", allocate)
    monkeypatch.setattr(zipformer_decoder, "get_engine", lambda _path: engine)

    with pytest.raises(ASRInitializationError, match=rejected_binding):
        construct_zipformer_decoder(stream)

    binding_order = ["decoder_input", "encoder_output", "tokens_log_prob"]
    rejected_index = binding_order.index(rejected_binding)
    assert list(context.bindings) == binding_order[: rejected_index + 1]
