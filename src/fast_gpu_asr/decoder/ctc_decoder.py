#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Batched GPU CTC decoding shared by Zipformer and Parakeet."""

import cupy as cp
import cupyx as cpx
import numpy as np

from ..constants import INT32_MAX
from ..utils import ASRInferenceError
from .gpu_kernels import CTC_COLLAPSE_KERNEL


class CTCGreedyDecoder:
    """Collapse batched CTC argmax paths into token sequences for either model family.

    The decoder keeps all frame-level work on the shared CUDA stream. Token IDs,
    timestamps, and emitted lengths are written to reusable device buffers and
    copied back to pinned host buffers only after the valid output width is known.

    CuPy computes the argmax path for every frame. One CUDA block per utterance
    then removes repeated tokens and blanks.
    """

    def __init__(
        self,
        blank_id: int,
        encoder_frame_shift_sec: float,
        blank_penalty: float,
        device: cp.cuda.Device,
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
        device : cp.cuda.Device
            CUDA device shared by the encoder and decoder.
        stream : cp.cuda.Stream
            CUDA stream shared with the encoder.
        """

        self.device = device
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
            adjusts the blank column in place. Path selection uses the updated
            scores in the input dtype.
        output_lengths : cp.ndarray
            Contiguous CUDA ``int32`` valid frame counts with shape
            ``(batch_size,)``.

        Returns
        -------
        tuple[list[list[int]], list[list[float]]]
            Collapsed non-blank token IDs and corresponding token start
            timestamps in seconds, in input order.

        Raises
        ------
        ASRInferenceError
            Raised for an empty batch, malformed decoder inputs, or an
            unexpectedly unavailable reusable output buffer.

        Notes
        -----
        Inputs must be ready on ``stream``. The method waits for its host outputs
        before returning. Calls on one instance must be serialized because its
        device and pinned host buffers are reused.
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

            if self.blank_penalty != 0.0:
                log_probs[:, :, self.blank_id] -= self.blank_penalty
            paths = cp.argmax(log_probs, axis=2).astype(cp.int32, copy=False)

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
