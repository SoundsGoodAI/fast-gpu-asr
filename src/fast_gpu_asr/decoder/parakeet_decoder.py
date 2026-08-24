#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Batched TensorRT modified beam search for Parakeet TDT models."""

from pathlib import Path
from warnings import warn

import cupy as cp
import cupyx as cpx
import numpy as np
import tensorrt as trt

from ..constants import INT32_MAX, TDT_SEARCH_CHUNK_STEPS
from ..utils import ASRInferenceError, get_engine
from .gpu_kernels import (
    TDT_BEAM_SEARCH_KERNEL,
    TDT_FINALIZE_KERNEL,
    TDT_PREPARE_INPUTS_KERNEL,
    TDT_SELECT_TOKENS_KERNEL,
)
from .transcription import DecoderResult


class ParakeetDecoder:
    """Decode fixed-capacity Parakeet batches with TDT modified beam search.

    Decoder engines with ``beam=1`` use the same search path as wider beams,
    so greedy decoding does not require a second implementation. Each active
    hypothesis expands over nonblank token-duration combinations and
    blank-duration advances. Retained duplicate histories at the same
    encoder position are merged, token histories use compact GPU backpointers,
    recurrent states are routed on the device, and final selection applies
    length-normalized log probability. The host only polls completion
    periodically and copies the selected histories.
    """

    def __init__(
        self,
        engine_path: Path,
        batch_size: int,
        blank_id: int,
        durations: tuple[int, ...],
        max_symbols_per_timestep: int,
        encoder_frame_shift_sec: float,
        blank_penalty: float,
        device_id: int,
        stream: cp.cuda.Stream,
    ) -> None:
        """Initialize the TensorRT decoder and reusable GPU search buffers.

        Parameters
        ----------
        engine_path : Path
            Path to the Parakeet TDT decoder TensorRT engine.
        batch_size : int
            Fixed utterance capacity of the paired encoder engine.
        blank_id : int
            TDT blank token ID.
        durations : tuple[int, ...]
            Frame advances represented by decoder duration outputs.
        max_symbols_per_timestep : int
            Maximum consecutive zero-duration token emissions.
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

            encoder_shape = tuple(engine.get_tensor_shape("encoder_output"))
            encoder_dtype = engine.get_tensor_dtype("encoder_output")
            encoder_dtype = (
                cp.dtype("bfloat16")
                if encoder_dtype == trt.bfloat16
                else np.dtype(trt.nptype(encoder_dtype))
            )
            target_shape = tuple(engine.get_tensor_shape("targets"))
            state_shape = tuple(engine.get_tensor_shape("input_states_1"))
            state_dtype = engine.get_tensor_dtype("input_states_1")
            state_dtype = (
                cp.dtype("bfloat16")
                if state_dtype == trt.bfloat16
                else np.dtype(trt.nptype(state_dtype))
            )
            token_shape = tuple(engine.get_tensor_shape("token_log_probs"))
            duration_shape = tuple(engine.get_tensor_shape("duration_log_probs"))

            self.batch_size = batch_size
            self.beam = encoder_shape[0] // batch_size
            self.decoder_capacity = encoder_shape[0]
            self.encoder_dim = encoder_shape[1]
            self.blank_id = blank_id
            positive_duration_indexes = tuple(
                index for index, duration in enumerate(durations) if duration > 0
            )
            self.max_symbols_per_timestep = max_symbols_per_timestep
            self.encoder_frame_shift_sec = encoder_frame_shift_sec
            self.blank_penalty = blank_penalty
            self.state_layers = state_shape[0]
            self.state_hidden_dim = state_shape[2]
            self.kernel_dtype_map = {
                np.dtype(np.float32): np.int32(0),
                np.dtype(np.float16): np.int32(1),
                cp.dtype("bfloat16"): np.int32(2),
            }
            self.state_dtype = self.kernel_dtype_map[state_dtype]
            self.encoder_input_dtype = self.kernel_dtype_map[encoder_dtype]
            self.prepare_inputs_threads = 256
            self.token_selection_threads = 512
            self.beam_search_threads = 256
            candidate_count = self.beam * (
                len(durations) * self.beam + len(positive_duration_indexes)
            )
            self.beam_search_shared_memory_bytes = (
                candidate_count * np.dtype(np.float32).itemsize
                + self.beam_search_threads
                // 32
                * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
                + self.beam
                * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
            )
            self.token_selection_shared_memory_bytes = self.blank_id * np.dtype(
                np.float32
            ).itemsize + self.token_selection_threads // 32 * (
                np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize
            )

            self.decoder = engine.create_execution_context()
            self.decoder.set_optimization_profile_async(0, self.stream.ptr)
            self.cuda_graph: cp.cuda.graph.Graph | None = None
            self.cuda_graph_signature: tuple[int, ...] | None = None
            self.cuda_graph_supported = True

            self.encoder_input = cp.empty(encoder_shape, dtype=encoder_dtype)
            self.targets = cp.empty(target_shape, dtype=np.int32)
            self.state_1 = cp.empty(state_shape, dtype=state_dtype)
            self.state_2 = cp.empty(state_shape, dtype=state_dtype)
            self.next_state_1 = cp.empty_like(self.state_1)
            self.next_state_2 = cp.empty_like(self.state_2)
            self.token_log_probs = cp.empty(token_shape, dtype=np.float32)

            top_token_shape = (self.decoder_capacity, self.beam)
            self.top_token_scores = cp.empty(top_token_shape, dtype=np.float32)
            self.top_token_indexes = cp.empty(top_token_shape, dtype=np.int32)
            self.duration_log_probs = cp.empty(duration_shape, dtype=np.float32)
            self.output_state_1 = cp.empty(state_shape, dtype=state_dtype)
            self.output_state_2 = cp.empty(state_shape, dtype=state_dtype)

            search_shape = (batch_size, self.beam)
            self.hypothesis_scores = cp.empty(search_shape, dtype=np.float32)
            self.next_scores = cp.empty_like(self.hypothesis_scores)
            self.hypothesis_lengths = cp.empty(search_shape, dtype=np.int32)
            self.next_lengths = cp.empty_like(self.hypothesis_lengths)
            self.time_indexes = cp.empty(encoder_shape[0], dtype=np.int32)
            self.next_time_indexes = cp.empty_like(self.time_indexes)
            self.last_tokens = cp.empty(encoder_shape[0], dtype=np.int32)
            self.next_last_tokens = cp.empty_like(self.last_tokens)
            self.symbols_at_timestep = cp.empty(encoder_shape[0], dtype=np.int32)
            self.next_symbols_at_timestep = cp.empty_like(self.symbols_at_timestep)
            self.parent_indexes = cp.empty(encoder_shape[0], dtype=np.int32)
            self.use_output_state = cp.empty(encoder_shape[0], dtype=np.uint8)
            self.search_output_lengths = cp.empty(batch_size, dtype=np.int32)
            self.active_flags = cp.empty(batch_size, dtype=np.int32)
            self.active_flags_host = cpx.empty_pinned(batch_size, dtype=np.int32)
            # Device-resident dimensions let one captured search graph serve
            # different actual batch sizes and temporal shapes.
            self.runtime_dimensions = cp.empty(2, dtype=np.int32)
            self.runtime_dimensions_host = cpx.empty_pinned(2, dtype=np.int32)

            self.durations_array = cp.array(durations, dtype=np.int32)
            self.positive_duration_indexes_array = cp.array(
                positive_duration_indexes, dtype=np.int32
            )

            self.completed_scores = cp.empty(batch_size, dtype=np.float32)
            self.completed_lengths = cp.empty(batch_size, dtype=np.int32)
            self.completed_nodes = cp.empty(batch_size, dtype=np.int32)
            self.output_lengths = cp.empty(batch_size, dtype=np.int32)
            self.output_lengths_host = cpx.empty_pinned(batch_size, dtype=np.int32)

            self.token_capacity = 0
            self.hypothesis_nodes = cp.empty(search_shape, dtype=np.int32)
            self.next_nodes = cp.empty_like(self.hypothesis_nodes)
            self.hypothesis_hashes = cp.empty(search_shape, dtype=np.uint64)
            self.next_hashes = cp.empty_like(self.hypothesis_hashes)
            self.node_counts = cp.empty(batch_size, dtype=np.int32)
            self.node_parents: cp.ndarray | None = None
            self.node_tokens: cp.ndarray | None = None
            self.node_timestamps: cp.ndarray | None = None
            self.output_tokens: cp.ndarray | None = None
            self.output_timestamps: cp.ndarray | None = None
            self.output_tokens_host: np.typing.NDArray[np.int32] | None = None
            self.output_timestamps_host: np.typing.NDArray[np.float32] | None = None

            for name, binding in (
                ("encoder_output", self.encoder_input),
                ("targets", self.targets),
                ("input_states_1", self.state_1),
                ("input_states_2", self.state_2),
                ("token_log_probs", self.token_log_probs),
                ("duration_log_probs", self.duration_log_probs),
                ("output_states_1", self.output_state_1),
                ("output_states_2", self.output_state_2),
            ):
                self.decoder.set_tensor_address(name, binding.data.ptr)

    def __call__(
        self, encoder_output: cp.ndarray, encoder_output_lengths: cp.ndarray
    ) -> DecoderResult:
        """Decode actual utterances in one fixed-capacity encoder batch.

        Parameters
        ----------
        encoder_output : cp.ndarray
            Contiguous FP32, FP16, or BF16 CUDA encoder embeddings with shape
            ``(actual_batch, num_frames, encoder_dim)``.
        encoder_output_lengths : cp.ndarray
            Contiguous CUDA ``int32`` valid encoder lengths with shape
            ``(actual_batch,)``.

        Returns
        -------
        DecoderResult
            Best token IDs and token start timestamps in seconds for each
            actual utterance.

        Raises
        ------
        ASRInferenceError
            Raised for malformed inputs, a batch exceeding decoder capacity,
            unavailable reusable buffers, or TensorRT execution failure.

        Warns
        -----
        RuntimeWarning
            Warned when CUDA graph capture fails and subsequent decoder calls
            fall back to ordinary TensorRT and CUDA execution.
        """

        if encoder_output.ndim != 3:
            raise ASRInferenceError(
                "Expected a rank-3 Parakeet encoder output, got shape "
                f"{encoder_output.shape}.",
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
                f"got shape {encoder_output.shape} and dtype {encoder_output.dtype}.",
            )
        if (
            encoder_output_lengths.shape != (actual_batch_size,)
            or encoder_output_lengths.dtype != np.int32
            or not encoder_output_lengths.flags.c_contiguous
        ):
            raise ASRInferenceError(
                "Expected contiguous int32 encoder output lengths with shape "
                f"{(actual_batch_size,)}, got shape {encoder_output_lengths.shape} "
                f"and dtype {encoder_output_lengths.dtype}.",
            )

        with self.device, self.stream:
            max_frames = encoder_output.shape[1]
            if max_frames == 0:
                return DecoderResult(
                    token_ids=[[] for _ in range(actual_batch_size)],
                    timestamps=[[] for _ in range(actual_batch_size)],
                )

            max_steps = max_frames * (self.max_symbols_per_timestep + 1)
            node_elements = self.decoder_capacity * max_steps
            encoder_elements = self.batch_size * max_frames * self.encoder_dim
            if max(node_elements, encoder_elements) > INT32_MAX:
                raise ASRInferenceError(
                    "Parakeet decoder buffers exceed signed 32-bit kernel "
                    f"indexing: encoder capacity={encoder_elements}, "
                    f"history capacity={node_elements}, limit={INT32_MAX}.",
                )

            if self.token_capacity < max_steps:
                output_shape = (self.batch_size, max_steps)
                self.node_parents = cp.empty(node_elements, dtype=np.int32)
                self.node_tokens = cp.empty(node_elements, dtype=np.int32)
                self.node_timestamps = cp.empty(node_elements, dtype=np.float32)
                self.output_tokens = cp.empty(output_shape, dtype=np.int32)
                self.output_timestamps = cp.empty(output_shape, dtype=np.float32)
                self.token_capacity = max_steps
                self.cuda_graph, self.cuda_graph_signature = None, None

            if any(
                buffer is None
                for buffer in (
                    self.node_parents,
                    self.node_tokens,
                    self.node_timestamps,
                    self.output_tokens,
                    self.output_timestamps,
                )
            ):
                raise ASRInferenceError("TDT token buffers were not initialized.")

            node_parents = self.node_parents
            node_tokens = self.node_tokens
            node_timestamps = self.node_timestamps
            output_tokens = self.output_tokens
            output_timestamps = self.output_timestamps
            duration_count = self.durations_array.size
            positive_duration_count = self.positive_duration_indexes_array.size

            self.search_output_lengths.fill(0)
            self.search_output_lengths[:actual_batch_size] = encoder_output_lengths
            self.hypothesis_scores.fill(-np.inf)
            self.hypothesis_scores[:actual_batch_size, 0] = 0.0
            self.hypothesis_lengths.fill(0)
            self.hypothesis_nodes.fill(-1)
            self.hypothesis_hashes.fill(0)
            self.node_counts.fill(0)
            self.time_indexes.fill(0)
            self.last_tokens.fill(self.blank_id)
            self.symbols_at_timestep.fill(0)
            self.state_1.fill(0.0)
            self.state_2.fill(0.0)
            self.completed_scores.fill(-np.inf)
            self.completed_lengths.fill(0)
            self.completed_nodes.fill(-1)

            encoder_output_dtype = self.kernel_dtype_map[encoder_output.dtype]
            self.runtime_dimensions_host[0] = actual_batch_size
            self.runtime_dimensions_host[1] = max_frames
            self.runtime_dimensions.set(
                self.runtime_dimensions_host, stream=self.stream
            )

            # Search kernels read batch and frame counts from runtime_dimensions.
            # With contiguous encoder storage, its pointer and dtype plus the
            # history stride fully identify all addresses captured by the graph.
            graph_signature = (
                int(encoder_output.data.ptr),
                int(encoder_output_dtype),
                self.token_capacity,
            )
            signature_changed = graph_signature != self.cuda_graph_signature
            if signature_changed:
                self.cuda_graph = None

            self.decoder.set_tensor_address("input_states_1", self.state_1.data.ptr)
            self.decoder.set_tensor_address("input_states_2", self.state_2.data.ptr)
            TDT_PREPARE_INPUTS_KERNEL(
                (self.decoder_capacity,),
                (self.prepare_inputs_threads,),
                (
                    encoder_output,
                    self.search_output_lengths,
                    self.hypothesis_scores,
                    self.time_indexes,
                    self.last_tokens,
                    self.encoder_input,
                    self.targets,
                    np.int32(actual_batch_size),
                    np.int32(encoder_output.shape[1]),
                    np.int32(self.encoder_dim),
                    np.int32(self.beam),
                    encoder_output_dtype,
                    self.encoder_input_dtype,
                ),
                stream=self.stream,
            )

            graph_warmed = not signature_changed
            steps_executed = 0
            # An even chunk amortizes launches while restoring ping-pong buffers to
            # their canonical identities after graph replay. This cadence performed
            # best across batch-one and batch-256 decoder benchmarks.
            for chunk_start in range(0, max_steps, TDT_SEARCH_CHUNK_STEPS):
                chunk_steps = min(TDT_SEARCH_CHUNK_STEPS, max_steps - chunk_start)

                replay_graph = (
                    chunk_steps == TDT_SEARCH_CHUNK_STEPS
                    and self.cuda_graph_supported
                    and self.cuda_graph is not None
                )
                if replay_graph:
                    self.cuda_graph.launch(self.stream)
                else:
                    should_capture = (
                        chunk_steps == TDT_SEARCH_CHUNK_STEPS
                        and self.cuda_graph_supported
                        and graph_warmed
                    )
                    # Failed capture discards recorded work, so retry the chunk once.
                    capture_attempts = (True, False) if should_capture else (False,)
                    for capture in capture_attempts:
                        if capture:
                            self.stream.synchronize()
                            self.stream.begin_capture()

                        executed = True
                        for _ in range(chunk_steps):
                            if not self.decoder.execute_async_v3(self.stream.ptr):
                                executed = False
                                break

                            TDT_SELECT_TOKENS_KERNEL(
                                (self.decoder_capacity,),
                                (self.token_selection_threads,),
                                (
                                    self.token_log_probs,
                                    self.hypothesis_scores,
                                    self.time_indexes,
                                    self.search_output_lengths,
                                    self.top_token_scores,
                                    self.top_token_indexes,
                                    np.int32(self.blank_id),
                                    np.int32(self.beam),
                                ),
                                shared_mem=self.token_selection_shared_memory_bytes,
                                stream=self.stream,
                            )

                            TDT_BEAM_SEARCH_KERNEL(
                                (self.batch_size,),
                                (self.beam_search_threads,),
                                (
                                    self.token_log_probs,
                                    self.duration_log_probs,
                                    self.top_token_scores,
                                    self.top_token_indexes,
                                    self.hypothesis_scores,
                                    self.hypothesis_nodes,
                                    self.hypothesis_hashes,
                                    self.hypothesis_lengths,
                                    self.time_indexes,
                                    self.last_tokens,
                                    self.symbols_at_timestep,
                                    self.next_scores,
                                    self.next_nodes,
                                    self.next_hashes,
                                    self.next_lengths,
                                    self.next_time_indexes,
                                    self.next_last_tokens,
                                    self.next_symbols_at_timestep,
                                    self.parent_indexes,
                                    self.use_output_state,
                                    node_parents,
                                    node_tokens,
                                    node_timestamps,
                                    self.node_counts,
                                    self.completed_scores,
                                    self.completed_nodes,
                                    self.completed_lengths,
                                    self.active_flags,
                                    self.search_output_lengths,
                                    self.durations_array,
                                    self.positive_duration_indexes_array,
                                    self.state_1,
                                    self.state_2,
                                    self.output_state_1,
                                    self.output_state_2,
                                    self.next_state_1,
                                    self.next_state_2,
                                    encoder_output,
                                    self.encoder_input,
                                    self.targets,
                                    np.int32(self.state_hidden_dim),
                                    np.int32(self.state_layers),
                                    self.runtime_dimensions,
                                    np.int32(self.encoder_dim),
                                    self.state_dtype,
                                    encoder_output_dtype,
                                    self.encoder_input_dtype,
                                    np.int32(self.token_capacity),
                                    np.int32(self.beam),
                                    np.int32(duration_count),
                                    np.int32(positive_duration_count),
                                    np.int32(self.blank_id),
                                    np.int32(self.max_symbols_per_timestep),
                                    np.float32(self.blank_penalty),
                                    np.float32(self.encoder_frame_shift_sec),
                                ),
                                shared_mem=self.beam_search_shared_memory_bytes,
                                stream=self.stream,
                            )

                            self.hypothesis_scores, self.next_scores = (
                                self.next_scores,
                                self.hypothesis_scores,
                            )
                            self.hypothesis_lengths, self.next_lengths = (
                                self.next_lengths,
                                self.hypothesis_lengths,
                            )
                            self.time_indexes, self.next_time_indexes = (
                                self.next_time_indexes,
                                self.time_indexes,
                            )
                            self.last_tokens, self.next_last_tokens = (
                                self.next_last_tokens,
                                self.last_tokens,
                            )
                            self.symbols_at_timestep, self.next_symbols_at_timestep = (
                                self.next_symbols_at_timestep,
                                self.symbols_at_timestep,
                            )
                            self.hypothesis_nodes, self.next_nodes = (
                                self.next_nodes,
                                self.hypothesis_nodes,
                            )
                            self.hypothesis_hashes, self.next_hashes = (
                                self.next_hashes,
                                self.hypothesis_hashes,
                            )
                            self.state_1, self.next_state_1 = (
                                self.next_state_1,
                                self.state_1,
                            )
                            self.state_2, self.next_state_2 = (
                                self.next_state_2,
                                self.state_2,
                            )

                            self.decoder.set_tensor_address(
                                "input_states_1", self.state_1.data.ptr
                            )
                            self.decoder.set_tensor_address(
                                "input_states_2", self.state_2.data.ptr
                            )

                        if capture:
                            try:
                                captured_graph = self.stream.end_capture()
                            except cp.cuda.runtime.CUDARuntimeError as error:
                                # 901 is cudaErrorStreamCaptureInvalidated.
                                if error.status != 901:
                                    raise
                                captured_graph = None

                            if executed and captured_graph is not None:
                                self.cuda_graph = captured_graph
                                self.cuda_graph_signature = graph_signature
                                self.cuda_graph.upload(self.stream)
                                self.cuda_graph.launch(self.stream)
                                break

                            warn(
                                "CUDA graph capture failed; Parakeet decoder "
                                "inference will continue without graph replay.",
                                RuntimeWarning,
                                stacklevel=2,
                            )

                            self.cuda_graph_supported = False
                            self.cuda_graph, self.cuda_graph_signature = None, None
                        elif not executed:
                            raise ASRInferenceError(
                                "TensorRT decoder execution failed."
                            )
                        else:
                            if self.cuda_graph_supported:
                                self.cuda_graph_signature = graph_signature
                                graph_warmed = True

                steps_executed += chunk_steps

                # TDT durations make max_steps a safety bound rather than an exact
                # loop count. Stop once every actual utterance has completed.
                self.active_flags[:actual_batch_size].get(
                    out=self.active_flags_host[:actual_batch_size],
                    stream=self.stream,
                    blocking=False,
                )
                self.stream.synchronize()

                if not self.active_flags_host[:actual_batch_size].any():
                    break

            # One thread follows each selected backpointer chain serially.
            TDT_FINALIZE_KERNEL(
                (actual_batch_size,),
                (1,),
                (
                    self.hypothesis_scores,
                    self.hypothesis_nodes,
                    self.hypothesis_lengths,
                    self.completed_scores,
                    self.completed_nodes,
                    self.completed_lengths,
                    node_parents,
                    node_tokens,
                    node_timestamps,
                    output_tokens,
                    output_timestamps,
                    self.output_lengths,
                    np.int32(self.token_capacity),
                    np.int32(self.beam),
                ),
                stream=self.stream,
            )

            # Full graph chunks contain an even number of ping-pong steps. A
            # final partial chunk may not, so restore canonical buffer identities
            # after the finalize launch has captured the current pointers.
            if steps_executed % 2:
                self.hypothesis_scores, self.next_scores = (
                    self.next_scores,
                    self.hypothesis_scores,
                )
                self.hypothesis_lengths, self.next_lengths = (
                    self.next_lengths,
                    self.hypothesis_lengths,
                )
                self.time_indexes, self.next_time_indexes = (
                    self.next_time_indexes,
                    self.time_indexes,
                )
                self.last_tokens, self.next_last_tokens = (
                    self.next_last_tokens,
                    self.last_tokens,
                )
                self.symbols_at_timestep, self.next_symbols_at_timestep = (
                    self.next_symbols_at_timestep,
                    self.symbols_at_timestep,
                )
                self.hypothesis_nodes, self.next_nodes = (
                    self.next_nodes,
                    self.hypothesis_nodes,
                )
                self.hypothesis_hashes, self.next_hashes = (
                    self.next_hashes,
                    self.hypothesis_hashes,
                )
                self.state_1, self.next_state_1 = self.next_state_1, self.state_1
                self.state_2, self.next_state_2 = self.next_state_2, self.state_2
                self.decoder.set_tensor_address("input_states_1", self.state_1.data.ptr)
                self.decoder.set_tensor_address("input_states_2", self.state_2.data.ptr)

            self.output_lengths[:actual_batch_size].get(
                out=self.output_lengths_host[:actual_batch_size],
                stream=self.stream,
                blocking=False,
            )
            self.stream.synchronize()

            max_emitted = int(self.output_lengths_host[:actual_batch_size].max())
            host_shape = (actual_batch_size, max_emitted)
            host_elements = actual_batch_size * max_emitted
            if (
                self.output_tokens_host is None
                or self.output_tokens_host.size < host_elements
            ):
                self.output_tokens_host = cpx.empty_pinned(
                    host_elements, dtype=np.int32
                )
                self.output_timestamps_host = cpx.empty_pinned(
                    host_elements, dtype=np.float32
                )

            if self.output_tokens_host is None or self.output_timestamps_host is None:
                raise ASRInferenceError("TDT host output buffers were not initialized.")

            output_tokens_host = self.output_tokens_host[:host_elements].reshape(
                host_shape
            )
            output_timestamps_host = self.output_timestamps_host[
                :host_elements
            ].reshape(host_shape)
            if max_emitted > 0:
                output_tokens[:actual_batch_size, :max_emitted].get(
                    out=output_tokens_host, stream=self.stream, blocking=False
                )
                output_timestamps[:actual_batch_size, :max_emitted].get(
                    out=output_timestamps_host, stream=self.stream, blocking=False
                )
                self.stream.synchronize()

        token_ids = []
        timestamps = []
        for length, tokens, token_timestamps in zip(
            self.output_lengths_host[:actual_batch_size],
            output_tokens_host,
            output_timestamps_host,
            strict=True,
        ):
            token_ids.append(tokens[:length].tolist())
            timestamps.append(token_timestamps[:length].tolist())

        return DecoderResult(token_ids=token_ids, timestamps=timestamps)
