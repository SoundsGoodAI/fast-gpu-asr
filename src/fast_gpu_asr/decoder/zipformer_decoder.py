#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Batched TensorRT CTC and transducer decoding for Zipformer outputs."""

from pathlib import Path
from warnings import warn

import cupy as cp
import cupyx as cpx
import numpy as np
import tensorrt as trt
import torch

from ..constants import INT32_MAX, ZIPFORMER_DECODER_CONTEXTS_FILE
from ..utils import ASRInferenceError, ASRInitializationError, get_engine
from .gpu_kernels import (
    CTC_COLLAPSE_KERNEL,
    ZIPFORMER_FINALIZE_KERNEL,
    get_zipformer_beam_search_kernels,
)


class CTCGreedyDecoder:
    """Collapse batched Zipformer CTC argmax paths into token sequences.

    The decoder keeps all frame-level work on the shared CUDA stream. It first
    computes the argmax path for every frame, then launches one CUDA block per
    utterance to remove repeated tokens and blanks. Token IDs, timestamps, and
    emitted lengths are written to reusable device buffers and copied back to
    pinned host buffers only after the valid output width is known.
    """

    def __init__(
        self,
        blank_id: int,
        encoder_frame_shift_sec: float,
        blank_penalty: float,
        device_id: int,
        stream: cp.cuda.Stream,
    ) -> None:
        """Initialize the decoder.

        Parameters
        ----------
        blank_id : int
            CTC blank token ID.
        encoder_frame_shift_sec : float
            Time shift in seconds between adjacent encoder frames.
        blank_penalty : float
            Value subtracted from blank-token log probabilities before greedy
            path selection.
        device_id : int
            CUDA device ordinal used for inference.
        stream : cp.cuda.Stream
            CUDA stream shared with the encoder.
        """

        self.device = cp.cuda.Device(device_id)
        self.blank_id = blank_id
        self.encoder_frame_shift_sec = encoder_frame_shift_sec
        self.blank_penalty = blank_penalty
        self.stream = stream

        self.emitted_tokens: cp.ndarray | None = None
        self.emitted_timestamps: cp.ndarray | None = None
        self.emitted_lengths: cp.ndarray | None = None

        self.emitted_tokens_host: np.typing.NDArray[np.int32] | None = None
        self.emitted_timestamps_host: np.typing.NDArray[np.float32] | None = None
        self.emitted_lengths_host: np.typing.NDArray[np.int32] | None = None

    def __call__(
        self, log_probs: cp.ndarray, output_lengths: cp.ndarray
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Decode padded CTC log probabilities.

        Parameters
        ----------
        log_probs : cp.ndarray
            FP32, FP16, or BF16 CTC log probabilities with shape
            ``(batch_size, num_frames, vocab_size)``. A nonzero blank penalty
            adjusts the blank column in place before path selection.
        output_lengths : cp.ndarray
            Contiguous CUDA ``int32`` valid frame counts with shape
            ``(batch_size,)``.

        Returns
        -------
        tuple[list[list[int]], list[list[float]]]
            Collapsed non-blank token IDs and corresponding token start
            timestamps in seconds.

        Raises
        ------
        ASRInferenceError
            Raised for an empty batch, malformed decoder inputs, or an
            unexpectedly unavailable reusable output buffer.
        """

        if log_probs.ndim != 3:
            raise ASRInferenceError(
                f"Expected rank-3 CTC log probabilities, got shape {log_probs.shape}."
            )

        batch_size, num_frames = log_probs.shape[:2]
        if batch_size == 0:
            raise ASRInferenceError("At least one CTC utterance is required.")
        if num_frames > INT32_MAX:
            raise ASRInferenceError(
                "CTC frame count exceeds signed 32-bit kernel indexing: "
                f"{num_frames} frames, limit={INT32_MAX}."
            )
        if (
            log_probs.dtype not in (np.float16, np.float32, cp.dtype("bfloat16"))
            or output_lengths.shape != (batch_size,)
            or output_lengths.dtype != np.int32
            or not output_lengths.flags.c_contiguous
        ):
            raise ASRInferenceError(
                "Expected float16, float32, or bfloat16 CTC log probabilities and "
                f"contiguous int32 output lengths with shape {(batch_size,)}, got "
                f"log-probability dtype {log_probs.dtype} and output lengths with "
                f"shape {output_lengths.shape} and dtype {output_lengths.dtype}."
            )

        with self.device, self.stream:
            if self.blank_penalty != 0.0:
                log_probs[:, :, self.blank_id] -= self.blank_penalty
            paths = cp.argmax(log_probs, axis=2).astype(cp.int32, copy=False)

            output_shape = (batch_size, num_frames)
            if self.emitted_tokens is None or self.emitted_tokens.shape != output_shape:
                self.emitted_tokens = cp.empty(output_shape, dtype=np.int32)
                self.emitted_timestamps = cp.empty(output_shape, dtype=np.float32)
            if self.emitted_lengths is None or self.emitted_lengths.shape != (
                batch_size,
            ):
                self.emitted_lengths = cp.empty(batch_size, dtype=np.int32)
                self.emitted_lengths_host = cpx.empty_pinned(batch_size, dtype=np.int32)

            if (
                self.emitted_tokens is None
                or self.emitted_timestamps is None
                or self.emitted_lengths is None
                or self.emitted_lengths_host is None
            ):
                raise ASRInferenceError("CTC output buffers were not initialized.")

            CTC_COLLAPSE_KERNEL(
                (batch_size,),
                (1,),
                (
                    paths,
                    output_lengths,
                    self.emitted_tokens,
                    self.emitted_timestamps,
                    self.emitted_lengths,
                    np.int32(num_frames),
                    np.int32(self.blank_id),
                    np.float32(self.encoder_frame_shift_sec),
                ),
                stream=self.stream,
            )
            self.emitted_lengths.get(
                out=self.emitted_lengths_host, stream=self.stream, blocking=False
            )
            self.stream.synchronize()

            max_emitted = int(self.emitted_lengths_host.max())
            host_shape = (batch_size, max_emitted)
            if (
                self.emitted_tokens_host is None
                or self.emitted_tokens_host.shape != host_shape
            ):
                self.emitted_tokens_host = cpx.empty_pinned(host_shape, dtype=np.int32)
                self.emitted_timestamps_host = cpx.empty_pinned(
                    host_shape, dtype=np.float32
                )

            if self.emitted_tokens_host is None or self.emitted_timestamps_host is None:
                raise ASRInferenceError("CTC host output buffers were not initialized.")

            if max_emitted > 0:
                self.emitted_tokens[:, :max_emitted].get(
                    out=self.emitted_tokens_host, stream=self.stream, blocking=False
                )
                self.emitted_timestamps[:, :max_emitted].get(
                    out=self.emitted_timestamps_host, stream=self.stream, blocking=False
                )
                self.stream.synchronize()

        token_ids = []
        timestamps = []
        for index, length in enumerate(self.emitted_lengths_host):
            token_ids.append(self.emitted_tokens_host[index, :length].tolist())
            timestamps.append(self.emitted_timestamps_host[index, :length].tolist())

        return token_ids, timestamps


class ZipformerModifiedBeamSearchDecoder:
    """Decode fixed-capacity batches with modified beam search.

    Decoder engines with ``beam=1`` (corresponding to an RNN-T greedy decoder) use
    the same modified beam-search kernel as wider beams, which keeps RNNT
    decoding semantics in one implementation. "Modified" follows Icefall: each
    frame emits at most one symbol, candidates come from the current
    hypothesis-by-vocabulary table, duplicate token histories are merged, and
    the top ``beam`` histories are retained. Token histories use compact GPU
    backpointers, and final selection applies length-normalized log probability.
    """

    def __init__(
        self,
        engine_path: Path,
        batch_size: int,
        context_size: int,
        vocab_size: int,
        blank_id: int,
        encoder_frame_shift_sec: float,
        blank_penalty: float,
        device_id: int,
        stream: cp.cuda.Stream,
    ) -> None:
        """Initialize the TensorRT decoder and reusable search buffers.

        Parameters
        ----------
        engine_path : Path
            Path to the stateless Zipformer decoder TensorRT engine.
        batch_size : int
            Fixed utterance capacity of the paired encoder engine.
        context_size : int
            Number of previous tokens consumed by the stateless predictor.
        vocab_size : int
            Number of output tokens, including blank.
        blank_id : int
            Transducer blank token ID.
        encoder_frame_shift_sec : float
            Time shift in seconds between adjacent encoder frames.
        blank_penalty : float
            Value subtracted from blank log probabilities.
        device_id : int
            CUDA device ordinal used for inference.
        stream : cp.cuda.Stream
            CUDA stream shared with the encoder.
        """

        self.device = cp.cuda.Device(device_id)
        self.stream = stream
        with self.device, self.stream:
            engine = get_engine(engine_path)

            decoder_shape = tuple(engine.get_tensor_shape("decoder_input"))
            encoder_shape = tuple(engine.get_tensor_shape("encoder_output"))
            output_shape = tuple(engine.get_tensor_shape("tokens_log_prob"))

            self.batch_size = batch_size
            self.beam = decoder_shape[0] // batch_size
            self.context_size = context_size
            self.encoder_dim = encoder_shape[1]
            self.blank_id = blank_id
            self.encoder_frame_shift_sec = encoder_frame_shift_sec
            self.blank_penalty = blank_penalty

            (
                self.register_beam_search,
                self.shared_beam_search,
                self.beam_search_threads,
            ) = get_zipformer_beam_search_kernels(
                self.beam, vocab_size, self.context_size
            )

            # Register-local candidate lists favor small grids. Once roughly half
            # the SMs have work, shared candidates preserve occupancy more reliably.
            self.beam_search_register_batch_limit = max(
                1, self.device.attributes["MultiProcessorCount"] // 2
            )

            self.kernel_dtype_map = {
                np.dtype(np.float32): np.int32(0),
                np.dtype(np.float16): np.int32(1),
                cp.dtype("bfloat16"): np.int32(2),
            }

            self.decoder = engine.create_execution_context()
            if self.decoder is None:
                raise ASRInitializationError(
                    "TensorRT could not create the Zipformer decoder execution context."
                )
            if not self.decoder.set_optimization_profile_async(0, self.stream.ptr):
                raise ASRInitializationError(
                    "TensorRT could not select Zipformer decoder optimization "
                    "profile 0."
                )

            self.cuda_graph: cp.cuda.graph.Graph | None = None
            self.cuda_graph_signature: tuple[int, ...] | None = None
            self.cuda_graph_supported = True

            self.decoder_input = cp.empty(
                decoder_shape,
                dtype=(
                    cp.dtype("bfloat16")
                    if engine.get_tensor_dtype("decoder_input") == trt.bfloat16
                    else trt.nptype(engine.get_tensor_dtype("decoder_input"))
                ),
            )
            self.encoder_input = cp.empty(
                encoder_shape,
                dtype=(
                    cp.dtype("bfloat16")
                    if engine.get_tensor_dtype("encoder_output") == trt.bfloat16
                    else trt.nptype(engine.get_tensor_dtype("encoder_output"))
                ),
            )
            self.tokens_log_prob = cp.empty(output_shape, dtype=np.float32)

            for name, binding in (
                ("decoder_input", self.decoder_input),
                ("encoder_output", self.encoder_input),
                ("tokens_log_prob", self.tokens_log_prob),
            ):
                if not self.decoder.set_tensor_address(name, binding.data.ptr):
                    raise ASRInitializationError(
                        f"TensorRT rejected the Zipformer decoder tensor {name}."
                    )

            context_lookup_path = engine_path.parent / ZIPFORMER_DECODER_CONTEXTS_FILE
            context_lookup = torch.load(
                context_lookup_path, map_location="cpu", weights_only=True
            )
            if context_lookup.dtype == torch.bfloat16:
                self.context_lookup = cp.array(
                    context_lookup.to(torch.float32).numpy(), dtype=cp.dtype("bfloat16")
                )
            else:
                self.context_lookup = cp.array(context_lookup.numpy())

            initial_contexts = np.zeros(
                (self.batch_size, self.beam, self.context_size), dtype=np.int32
            )
            initial_contexts[:, 0, :] = -1
            initial_contexts[:, 0, self.context_size - 1] = self.blank_id
            initial_contexts = initial_contexts.reshape(
                self.batch_size * self.beam, self.context_size
            )
            self.initial_contexts = cp.array(initial_contexts)
            self.contexts = cp.empty_like(self.initial_contexts)

            initial_lookup_indexes = np.zeros(
                self.batch_size * self.beam, dtype=np.int64
            )
            # Cache rows encode contexts in base vocab_size + 1; adding one maps
            # the -1 start sentinel to zero and token IDs to positive digits.
            for position in range(self.context_size):
                initial_lookup_indexes = (
                    initial_lookup_indexes * (vocab_size + 1)
                    + initial_contexts[:, position]
                    + 1
                )
            self.initial_decoder_input = self.context_lookup[
                cp.array(initial_lookup_indexes)
            ]

            self.output_tokens: cp.ndarray | None = None
            self.output_timestamps: cp.ndarray | None = None
            self.output_lengths = cp.empty(self.batch_size, dtype=np.int32)

            self.output_tokens_host: np.typing.NDArray[np.int32] | None = None
            self.output_timestamps_host: np.typing.NDArray[np.float32] | None = None
            self.output_lengths_host: np.typing.NDArray[np.int32] | None = None

            search_shape = (self.batch_size, self.beam)
            self.search_output_lengths = cp.empty(self.batch_size, dtype=np.int32)
            self.hypothesis_scores = cp.empty(search_shape, dtype=np.float32)
            self.next_scores = cp.empty_like(self.hypothesis_scores)
            self.hypothesis_lengths = cp.empty(search_shape, dtype=np.int32)
            self.next_lengths = cp.empty_like(self.hypothesis_lengths)
            self.hypothesis_hashes = cp.empty(search_shape, dtype=np.uint64)
            self.next_hashes = cp.empty_like(self.hypothesis_hashes)
            self.hypothesis_nodes = cp.empty(search_shape, dtype=np.int32)
            self.next_nodes = cp.empty_like(self.hypothesis_nodes)
            self.node_counts = cp.empty(self.batch_size, dtype=np.int32)

            self.frame_capacity = 0
            self.node_parents: cp.ndarray | None = None
            self.node_tokens: cp.ndarray | None = None
            self.node_timestamps: cp.ndarray | None = None

    def __call__(
        self, encoder_output: cp.ndarray, encoder_output_lengths: cp.ndarray
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Decode actual utterances in one fixed-capacity encoder batch.

        Parameters
        ----------
        encoder_output : cp.ndarray
            Contiguous FP32, FP16, or BF16 CUDA encoder embeddings with shape
            ``(actual_batch, num_frames, encoder_dim)``. Frames are converted to
            the decoder engine's floating-point dtype while being staged.
        encoder_output_lengths : cp.ndarray
            Contiguous CUDA ``int32`` valid encoder lengths with shape
            ``(actual_batch,)``.

        Returns
        -------
        tuple[list[list[int]], list[list[float]]]
            Best token IDs and corresponding token start timestamps in seconds
            for each actual utterance.

        Raises
        ------
        ASRInferenceError
            Raised for an empty or oversized batch, incompatible encoder output,
            unavailable reusable buffers, or TensorRT execution failure.

        Warns
        -----
        RuntimeWarning
            Warned when CUDA graph capture fails and subsequent decoder calls
            fall back to ordinary TensorRT and CUDA execution.
        """

        if encoder_output.ndim != 3:
            raise ASRInferenceError(
                f"Expected rank-3 encoder output, got shape {encoder_output.shape}."
            )
        actual_batch_size = encoder_output.shape[0]
        if not 0 < actual_batch_size <= self.batch_size:
            raise ASRInferenceError(
                f"Decoder batch capacity is {self.batch_size}, got {actual_batch_size}."
            )
        if (
            encoder_output.shape[2] != self.encoder_dim
            or encoder_output.dtype
            not in (np.float16, np.float32, cp.dtype("bfloat16"))
            or not encoder_output.flags.c_contiguous
        ):
            raise ASRInferenceError(
                "Expected contiguous rank-3 encoder output with dimension "
                f"{self.encoder_dim} and float16, float32, or bfloat16 values, "
                f"got shape {encoder_output.shape} and dtype {encoder_output.dtype}."
            )
        if (
            encoder_output_lengths.shape != (actual_batch_size,)
            or encoder_output_lengths.dtype != np.int32
            or not encoder_output_lengths.flags.c_contiguous
        ):
            raise ASRInferenceError(
                "Expected contiguous int32 encoder output lengths with shape "
                f"{(actual_batch_size,)}, got shape {encoder_output_lengths.shape} "
                f"and dtype {encoder_output_lengths.dtype}."
            )

        max_frames = encoder_output.shape[1]
        history_elements = self.batch_size * self.beam * max_frames
        if history_elements > INT32_MAX:
            raise ASRInferenceError(
                "Zipformer token histories exceed signed 32-bit kernel indexing: "
                f"{history_elements} elements, limit={INT32_MAX}."
            )
        encoder_elements = actual_batch_size * max_frames * self.encoder_dim
        if encoder_elements > INT32_MAX:
            raise ASRInferenceError(
                "Zipformer encoder output exceeds signed 32-bit kernel indexing: "
                f"{encoder_elements} elements, limit={INT32_MAX}."
            )

        with self.device, self.stream:
            if max_frames == 0:
                return (
                    [[] for _ in range(actual_batch_size)],
                    [[] for _ in range(actual_batch_size)],
                )

            if max_frames > self.frame_capacity:
                num_hypotheses = self.batch_size * self.beam
                node_elements = num_hypotheses * max_frames
                output_elements = self.batch_size * max_frames

                self.node_parents = cp.empty(node_elements, dtype=np.int32)
                self.node_tokens = cp.empty(node_elements, dtype=np.int32)
                self.node_timestamps = cp.empty(node_elements, dtype=np.float32)
                self.output_tokens = cp.empty(output_elements, dtype=np.int32)
                self.output_timestamps = cp.empty(output_elements, dtype=np.float32)
                self.frame_capacity = max_frames

                self.cuda_graph, self.cuda_graph_signature = None, None

            output_lengths = self.search_output_lengths

            node_parents = self.node_parents
            node_tokens = self.node_tokens
            node_timestamps = self.node_timestamps
            if node_parents is None or node_tokens is None or node_timestamps is None:
                raise ASRInferenceError(
                    "Zipformer search buffers were not initialized."
                )
            if self.output_tokens is None or self.output_timestamps is None:
                raise ASRInferenceError(
                    "Zipformer device output buffers were not initialized."
                )

            contexts = self.contexts
            encoder_output_dtype = self.kernel_dtype_map[encoder_output.dtype]
            encoder_input_dtype = self.kernel_dtype_map[self.encoder_input.dtype]
            context_lookup_dtype = self.kernel_dtype_map[self.context_lookup.dtype]

            # Captured graphs retain external buffer addresses and layouts.
            graph_signature = (
                int(encoder_output.data.ptr),
                *encoder_output.shape,
                *encoder_output.strides,
                int(encoder_output_dtype),
                int(encoder_output_lengths.data.ptr),
                *encoder_output_lengths.shape,
                *encoder_output_lengths.strides,
            )
            signature_changed = graph_signature != self.cuda_graph_signature
            if signature_changed:
                self.cuda_graph = None

            if self.cuda_graph_supported and self.cuda_graph is not None:
                self.cuda_graph.launch(self.stream)
            else:
                should_capture = self.cuda_graph_supported and not signature_changed
                # Failed capture discards recorded work, so retry once normally.
                capture_attempts = (True, False) if should_capture else (False,)
                for capture in capture_attempts:
                    if capture:
                        self.stream.synchronize()
                        self.stream.begin_capture()

                    output_lengths.fill(0)
                    output_lengths[:actual_batch_size] = encoder_output_lengths

                    self.hypothesis_scores.fill(-np.inf)
                    self.hypothesis_scores[:, 0] = 0.0
                    self.hypothesis_lengths.fill(0)
                    self.hypothesis_hashes.fill(0)
                    self.hypothesis_nodes.fill(-1)
                    self.node_counts.fill(0)

                    cp.copyto(contexts, self.initial_contexts)
                    cp.copyto(self.decoder_input, self.initial_decoder_input)

                    encoder_input = self.encoder_input.reshape(
                        self.batch_size, self.beam, self.encoder_dim
                    )
                    if actual_batch_size != self.batch_size:
                        encoder_input.fill(0.0)
                    encoder_input[:actual_batch_size] = encoder_output[
                        :actual_batch_size, 0, None, :
                    ]

                    hypothesis_scores = self.hypothesis_scores
                    hypothesis_nodes = self.hypothesis_nodes
                    hypothesis_lengths = self.hypothesis_lengths
                    hypothesis_hashes = self.hypothesis_hashes

                    next_scores = self.next_scores
                    next_nodes = self.next_nodes
                    next_lengths = self.next_lengths
                    next_hashes = self.next_hashes

                    beam_search_kernel, beam_search_shared_memory_bytes = (
                        self.shared_beam_search
                    )
                    if (
                        self.register_beam_search is not None
                        and actual_batch_size <= self.beam_search_register_batch_limit
                    ):
                        beam_search_kernel, beam_search_shared_memory_bytes = (
                            self.register_beam_search
                        )

                    executed = True
                    for frame_index in range(max_frames):
                        if not self.decoder.execute_async_v3(self.stream.ptr):
                            executed = False
                            break

                        beam_search_kernel(
                            (actual_batch_size,),
                            (self.beam_search_threads,),
                            (
                                self.tokens_log_prob,
                                encoder_output,
                                self.encoder_input,
                                self.context_lookup,
                                self.decoder_input,
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
                                self.node_counts,
                                output_lengths,
                                np.int32(frame_index),
                                np.int32(max_frames),
                                np.int32(self.encoder_dim),
                                encoder_output_dtype,
                                encoder_input_dtype,
                                context_lookup_dtype,
                                np.int32(self.blank_id),
                                np.float32(self.blank_penalty),
                                np.float32(self.encoder_frame_shift_sec),
                            ),
                            shared_mem=beam_search_shared_memory_bytes,
                            stream=self.stream,
                        )

                        hypothesis_scores, next_scores = next_scores, hypothesis_scores
                        hypothesis_nodes, next_nodes = next_nodes, hypothesis_nodes
                        hypothesis_lengths, next_lengths = (
                            next_lengths,
                            hypothesis_lengths,
                        )
                        hypothesis_hashes, next_hashes = next_hashes, hypothesis_hashes

                    if capture:
                        try:
                            captured_graph = self.stream.end_capture()
                        except cp.cuda.runtime.CUDARuntimeError as error:
                            if error.status != 901:  # cudaErrorStreamCaptureInvalidated
                                raise
                            captured_graph = None

                        if executed and captured_graph is not None:
                            self.cuda_graph = captured_graph
                            self.cuda_graph.upload(self.stream)
                            self.cuda_graph.launch(self.stream)
                            break

                        warn(
                            "CUDA graph capture failed; decoder inference "
                            "will continue without CUDA graph replay.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self.cuda_graph_supported = False
                        self.cuda_graph, self.cuda_graph_signature = None, None
                    elif not executed:
                        raise ASRInferenceError("TensorRT decoder execution failed.")
                    elif self.cuda_graph_supported:
                        self.cuda_graph_signature = graph_signature

            output_shape = (actual_batch_size, max_frames)
            output_elements = actual_batch_size * max_frames
            output_tokens = self.output_tokens[:output_elements].reshape(output_shape)
            output_timestamps = self.output_timestamps[:output_elements].reshape(
                output_shape
            )
            if (
                self.output_tokens_host is None
                or self.output_tokens_host.shape != output_shape
            ):
                self.output_tokens_host = cpx.empty_pinned(output_shape, dtype=np.int32)
                self.output_timestamps_host = cpx.empty_pinned(
                    output_shape, dtype=np.float32
                )
            if self.output_lengths_host is None or self.output_lengths_host.shape != (
                actual_batch_size,
            ):
                self.output_lengths_host = cpx.empty_pinned(
                    actual_batch_size, dtype=np.int32
                )

            if (
                self.output_tokens_host is None
                or self.output_timestamps_host is None
                or self.output_lengths_host is None
            ):
                raise ASRInferenceError(
                    "Zipformer output buffers were not initialized."
                )

            # Search writes into the alternate buffers and swaps local references
            # after every frame, so parity identifies the buffers holding final state.
            if max_frames % 2 == 0:
                hypothesis_scores = self.hypothesis_scores
                hypothesis_nodes = self.hypothesis_nodes
                hypothesis_lengths = self.hypothesis_lengths
            else:
                hypothesis_scores = self.next_scores
                hypothesis_nodes = self.next_nodes
                hypothesis_lengths = self.next_lengths

            ZIPFORMER_FINALIZE_KERNEL(
                (actual_batch_size,),
                (1,),
                (
                    hypothesis_scores,
                    hypothesis_nodes,
                    hypothesis_lengths,
                    node_parents,
                    node_tokens,
                    node_timestamps,
                    output_tokens,
                    output_timestamps,
                    self.output_lengths,
                    np.int32(max_frames),
                    np.int32(self.beam),
                    np.int32(self.context_size),
                ),
                stream=self.stream,
            )

            output_tokens.get(
                out=self.output_tokens_host, stream=self.stream, blocking=False
            )
            output_timestamps.get(
                out=self.output_timestamps_host, stream=self.stream, blocking=False
            )
            self.output_lengths[:actual_batch_size].get(
                out=self.output_lengths_host, stream=self.stream, blocking=False
            )
            self.stream.synchronize()

        token_ids = []
        timestamps = []
        for length, tokens, token_timestamps in zip(
            self.output_lengths_host,
            self.output_tokens_host,
            self.output_timestamps_host,
            strict=True,
        ):
            token_ids.append(tokens[:length].tolist())
            timestamps.append(token_timestamps[:length].tolist())

        return token_ids, timestamps
