#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Batched raw-audio TensorRT encoder using CuPy buffers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import prod
from os import cpu_count
from pathlib import Path
from warnings import warn

import cupy as cp
import cupyx as cpx
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime

from ..constants import AUDIO_SAMPLES_PER_WORKER
from ..utils import ASRInferenceError, ASRInitializationError, get_engine, get_names


class Encoder:
    """Prepare audio and run a merged feature extractor + encoder engine."""

    def __init__(
        self,
        engine_path: Path,
        sample_rate: int,
        device_id: int,
        stream: cp.cuda.Stream,
        right_padding_samples: int,
    ) -> None:
        """Initialize the fixed-batch TensorRT encoder.

        Parameters
        ----------
        engine_path : Path
            Path to the Zipformer or Parakeet encoder engine.
        sample_rate : int
            Sampling rate expected by the model.
        device_id : int
            CUDA device ordinal used for inference.
        stream : cp.cuda.Stream
            CUDA stream shared with downstream decoders.
        right_padding_samples : int
            Reflected waveform samples appended by the runtime for optimized
            Zipformer feature extraction. Parakeet uses zero.
        """

        self.device = cp.cuda.Device(device_id)
        with self.device:
            engine = get_engine(engine_path)
            input_names, output_names = get_names(engine)
            min_shape, _, max_shape = engine.get_tensor_profile_shape("audio", 0)

            self.batch_size = min_shape[0]
            self.min_samples = min_shape[1] - right_padding_samples
            self.max_samples = max_shape[1] - right_padding_samples
            self.right_padding_samples = right_padding_samples
            self.sample_rate = sample_rate
            self.dtypes = {
                name: cp.dtype("bfloat16")
                if engine.get_tensor_dtype(name) == trt.bfloat16
                else trt.nptype(engine.get_tensor_dtype(name))
                for name in input_names + output_names
            }
            self.stream = stream

            self.encoder = engine.create_execution_context(
                trt.ExecutionContextAllocationStrategy.USER_MANAGED
            )
            if self.encoder is None:
                raise ASRInitializationError(
                    "TensorRT could not create the encoder execution context."
                )
            if not self.encoder.set_optimization_profile_async(0, self.stream.ptr):
                raise ASRInitializationError(
                    "TensorRT could not select encoder optimization profile 0."
                )

            self.aux_streams = [
                cp.cuda.Stream(null=False, non_blocking=True, ptds=False)
                for _ in range(engine.num_aux_streams)
            ]
            self.context_memory: cp.cuda.Memory | None = None
            self.context_memory_size = 0

            lengths_shape = tuple(engine.get_tensor_shape("audio_lengths"))
            self.lengths = cp.empty(lengths_shape, dtype=self.dtypes["audio_lengths"])
            self.lengths_host = cpx.empty_pinned(
                lengths_shape, dtype=self.dtypes["audio_lengths"]
            )
            self.output_lengths = cp.empty(
                tuple(engine.get_tensor_shape("encoder_output_lengths")),
                dtype=self.dtypes["encoder_output_lengths"],
            )
            self.audio: cp.ndarray | None = None
            self.audio_host: np.typing.NDArray[np.float32] | None = None
            self.encoder_output: cp.ndarray | None = None
            self.cuda_graph: cp.cuda.graph.Graph | None = None
            self.cuda_graph_shape: tuple[int, ...] | None = None
            self.cuda_graph_supported = True
            self.host_transfer_event = cp.cuda.Event(disable_timing=True)
            self.host_transfer_pending = False
            audio_copy_workers = min(
                self.batch_size,
                cpu_count() or 1,
                (
                    self.batch_size * (self.max_samples + self.right_padding_samples)
                    + AUDIO_SAMPLES_PER_WORKER
                    - 1
                )
                // AUDIO_SAMPLES_PER_WORKER,
            )
            self.audio_copy_pool = ThreadPoolExecutor(
                max_workers=audio_copy_workers, thread_name_prefix="fast-gpu-asr-audio"
            )

    def copy_audio_range(
        self,
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        """Copy a range of waveforms into the pinned encoder input buffer.

        Each destination row receives the waveform, its reflected right
        context, and trailing zeros. If a waveform is shorter than the
        configured right context, its first sample fills the remaining context.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Prepared one-dimensional waveforms for the actual input batch.
        audio_host : np.typing.NDArray[np.float32]
            Two-dimensional pinned host buffer for the fixed-capacity batch.
        start : int
            Inclusive index of the first waveform to copy.
        end : int
            Exclusive index after the last waveform to copy.
        """

        for index in range(start, end):
            waveform = audios[index]
            waveform_samples = len(waveform)
            np.copyto(audio_host[index, :waveform_samples], waveform)
            reflected_samples = min(waveform_samples, self.right_padding_samples)
            np.copyto(
                audio_host[
                    index, waveform_samples : waveform_samples + reflected_samples
                ],
                waveform[waveform_samples - reflected_samples :][::-1],
            )
            padding_end = waveform_samples + self.right_padding_samples
            if reflected_samples < self.right_padding_samples:
                padding_start = waveform_samples + reflected_samples
                audio_host[index, padding_start:padding_end].fill(waveform[0])
            audio_host[index, padding_end:].fill(0.0)

    def __call__(
        self, audios: list[np.typing.NDArray[np.float32]]
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Encode one partial or complete fixed-capacity audio batch.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            One-dimensional normalized mono waveforms. The list length must not
            exceed the fixed TensorRT batch size.

        Returns
        -------
        tuple[cp.ndarray, cp.ndarray]
            CUDA encoder embeddings and valid encoder lengths for the actual
            input utterances. Both arrays are views of reusable internal buffers.

        Raises
        ------
        ASRInferenceError
            Raised for an empty batch, malformed audio, profile overflow, or
            TensorRT execution failure.

        Warns
        -----
        RuntimeWarning
            Warned when CUDA graph capture fails and subsequent encoder calls
            fall back to ordinary TensorRT execution.

        Notes
        -----
        Execution is asynchronous on ``stream``. Consume the outputs on that
        stream, or explicitly wait for it before using another stream or reading
        results on the host. A later call may overwrite the returned views; copy
        them before that call if they must be retained. Calls on one instance must
        be serialized, as they are by :class:`fast_gpu_asr.ASR`.
        """

        batch_size = len(audios)
        if not 0 < batch_size <= self.batch_size:
            raise ASRInferenceError(
                f"Expected 1 to {self.batch_size} audio waveforms, got {batch_size}."
            )

        prepared_audios = []
        lengths = []
        for audio in audios:
            if audio.ndim != 1 or audio.size == 0:
                raise ASRInferenceError(
                    f"Expected non-empty one-dimensional mono audio, got {audio.shape}."
                )
            audio = audio.astype(np.float32, copy=False)
            prepared_audios.append(audio)
            lengths.append(len(audio))

        num_samples = max(self.min_samples, max(lengths))
        if num_samples > self.max_samples:
            max_seconds = self.max_samples / self.sample_rate
            raise ASRInferenceError(
                f"Audio exceeds the {max_seconds:.3f}-second TensorRT profile."
            )

        with self.device, self.stream:
            if self.host_transfer_pending:
                self.host_transfer_event.synchronize()
                self.host_transfer_pending = False

            audio_shape = (self.batch_size, num_samples + self.right_padding_samples)
            audio_elements = prod(audio_shape)
            shape_changed = audio_shape != self.cuda_graph_shape
            if shape_changed and self.cuda_graph is not None:
                self.stream.synchronize()
                self.cuda_graph = None
                if self.cuda_graph_shape is not None and audio_elements > prod(
                    self.cuda_graph_shape
                ):
                    (status,) = runtime.cudaDeviceGraphMemTrim(self.device.id)
                    if status != runtime.cudaError_t.cudaSuccess:
                        raise ASRInferenceError(
                            "Failed to trim CUDA graph memory on device "
                            f"{self.device.id}: error {status}."
                        )

            if self.audio is None or self.audio.size < audio_elements:
                if self.audio is not None:
                    # TensorRT may still be reading the previous input allocation.
                    self.stream.synchronize()
                self.cuda_graph, self.cuda_graph_shape = None, None
                self.audio = cp.empty(audio_elements, dtype=self.dtypes["audio"])
                self.audio_host = cpx.empty_pinned(
                    audio_elements, dtype=self.dtypes["audio"]
                )
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

            if self.audio is None or self.audio_host is None:
                raise ASRInferenceError("Encoder audio buffers were not initialized.")

            audio = self.audio[:audio_elements].reshape(audio_shape)
            audio_host = self.audio_host[:audio_elements].reshape(audio_shape)
            np.copyto(self.lengths_host[:batch_size], lengths)
            self.lengths_host[batch_size:].fill(self.min_samples)
            audio_copy_workers = min(
                batch_size,
                cpu_count() or 1,
                (batch_size * audio_shape[1] + AUDIO_SAMPLES_PER_WORKER - 1)
                // AUDIO_SAMPLES_PER_WORKER,
            )

            if batch_size < self.batch_size:
                audio[batch_size:].fill(0.0)

            transfer_enqueued = False
            try:
                if audio_copy_workers > 1:
                    first_error: Exception | None = None
                    chunk_size = (
                        batch_size + audio_copy_workers - 1
                    ) // audio_copy_workers
                    futures = {
                        self.audio_copy_pool.submit(
                            self.copy_audio_range,
                            prepared_audios,
                            audio_host,
                            start,
                            min(start + chunk_size, batch_size),
                        ): (start, min(start + chunk_size, batch_size))
                        for start in range(0, batch_size, chunk_size)
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                            if first_error is None:
                                start, end = futures[future]
                                audio[start:end].set(
                                    audio_host[start:end], stream=self.stream
                                )
                                transfer_enqueued = True
                        except Exception as error:
                            if first_error is None:
                                first_error = error

                    if first_error is not None:
                        raise first_error
                else:
                    self.copy_audio_range(prepared_audios, audio_host, 0, batch_size)
                    audio[:batch_size].set(audio_host[:batch_size], stream=self.stream)
                    transfer_enqueued = True

                self.lengths.set(self.lengths_host, stream=self.stream)
                transfer_enqueued = True
                self.host_transfer_event.record(self.stream)
                self.host_transfer_pending = True
            except:
                # Drain any queued DMA before the pinned staging buffer is reusable.
                if transfer_enqueued:
                    self.stream.synchronize()
                self.host_transfer_pending = False
                raise

            if not self.encoder.set_input_shape("audio", audio_shape):
                raise ASRInferenceError(
                    f"TensorRT rejected encoder input shape {audio_shape}."
                )

            context_memory_size = self.encoder.update_device_memory_size_for_shapes()
            if context_memory_size > self.context_memory_size:
                if self.context_memory is not None:
                    # TensorRT may still use the previous context allocation.
                    self.stream.synchronize()
                self.cuda_graph, self.cuda_graph_shape = None, None
                self.context_memory = cp.cuda.Memory(context_memory_size)
                self.context_memory_size = context_memory_size
                self.encoder.set_device_memory(
                    self.context_memory.ptr, self.context_memory_size
                )

            output_shape = tuple(self.encoder.get_tensor_shape("encoder_output"))
            output_elements = prod(output_shape)
            if (
                self.encoder_output is None
                or self.encoder_output.size < output_elements
            ):
                if self.encoder_output is not None:
                    # TensorRT or the decoder may still use the previous allocation.
                    self.stream.synchronize()
                self.cuda_graph, self.cuda_graph_shape = None, None
                self.encoder_output = cp.empty(
                    output_elements, dtype=self.dtypes["encoder_output"]
                )
                cp.get_default_memory_pool().free_all_blocks()

            if self.encoder_output is None:
                raise ASRInferenceError("Encoder output buffer was not initialized.")

            encoder_output = self.encoder_output[:output_elements].reshape(output_shape)

            for name, binding in (
                ("audio", audio),
                ("audio_lengths", self.lengths),
                ("encoder_output", encoder_output),
                ("encoder_output_lengths", self.output_lengths),
            ):
                if not self.encoder.set_tensor_address(name, binding.data.ptr):
                    raise ASRInferenceError(
                        f"TensorRT rejected encoder tensor address for {name}."
                    )

            if self.aux_streams:
                self.encoder.set_aux_streams(
                    [aux_stream.ptr for aux_stream in self.aux_streams]
                )

            # Warm each new shape once, capture its next execution, then replay it.
            if not self.cuda_graph_supported or shape_changed:
                if not self.encoder.execute_async_v3(self.stream.ptr):
                    raise ASRInferenceError("TensorRT encoder execution failed.")
                if self.cuda_graph_supported:
                    self.cuda_graph_shape = audio_shape
            elif self.cuda_graph is not None:
                self.cuda_graph.launch(self.stream)
            else:
                self.stream.synchronize()
                self.stream.begin_capture()
                executed = self.encoder.execute_async_v3(self.stream.ptr)
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
                else:
                    warn(
                        "CUDA graph capture failed; encoder inference "
                        "will continue without CUDA graph replay.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.cuda_graph_supported = False
                    self.cuda_graph, self.cuda_graph_shape = None, None
                    captured_graph = None
                    (status,) = runtime.cudaDeviceGraphMemTrim(self.device.id)
                    if status != runtime.cudaError_t.cudaSuccess:
                        raise ASRInferenceError(
                            "Failed to trim CUDA graph memory on device "
                            f"{self.device.id}: error {status}."
                        )
                    if not self.encoder.execute_async_v3(self.stream.ptr):
                        raise ASRInferenceError("TensorRT encoder execution failed.")

            return encoder_output[:batch_size], self.output_lengths[:batch_size]
