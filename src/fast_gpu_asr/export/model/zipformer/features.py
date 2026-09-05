#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Export-friendly Kaldi log-mel extraction for Zipformer models."""

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZERO_LOG,
    ZIPFORMER_FEATURE_PLUGIN_NAME,
)


class FeatureExtractor(torch.nn.Module):
    """Compute batched Kaldi-compatible log-mel features.

    Dynamo ONNX export combines framing, DC removal, pre-emphasis, the Povey
    window, the real Fourier transform, and mel projection in one native
    cuFFT-based TensorRT plugin. Eager execution computes the equivalent
    operations with PyTorch.
    """

    def __init__(
        self,
        samp_freq: int,
        frame_shift_ms: int,
        frame_length_ms: int,
        n_mels: int,
        preemph: float,
        low_freq: int,
        high_freq: int,
        min_frames: int,
    ) -> None:
        """Initialize the fixed window and mel filterbank.

        Parameters
        ----------
        samp_freq : int
            Model sampling rate in Hertz.
        frame_shift_ms : int
            Hop size between neighboring frames in milliseconds.
        frame_length_ms : int
            Analysis window size in milliseconds.
        n_mels : int
            Number of mel filterbank channels.
        preemph : float
            The waveform pre-emphasis coefficient.
        low_freq : int
            Lower mel filterbank frequency in Hertz.
        high_freq : int
            Upper mel filterbank frequency in Hertz.
        min_frames : int
            Minimum feature length accepted by the Zipformer encoder.
        """

        super().__init__()

        self.frame_length = frame_length_ms * samp_freq // 1000
        self.frame_shift = frame_shift_ms * samp_freq // 1000
        self.left_padding = self.frame_length // 2 - self.frame_shift // 2
        self.min_frames = min_frames
        self.preemph = preemph
        self.n_mels = n_mels
        self.n_fft = 2 ** (self.frame_length - 1).bit_length()
        self.log_eps = torch.finfo(torch.float32).eps
        self.zero_log = ZERO_LOG

        frame_positions = torch.arange(self.frame_length, dtype=torch.float32)

        povey_window = (
            0.5
            - 0.5
            * torch.cos(2.0 * torch.pi * frame_positions / (self.frame_length - 1))
        ) ** 0.85
        self.register_buffer("window", povey_window, persistent=False)

        mel_low = 1127.0 * torch.log(
            1.0 + torch.tensor(low_freq, dtype=torch.float32) / 700.0
        )
        mel_high = 1127.0 * torch.log(
            1.0 + torch.tensor(high_freq, dtype=torch.float32) / 700.0
        )
        mel_delta = (mel_high - mel_low) / (n_mels + 1)
        # Kaldi excludes the Nyquist bin, so the final filterbank row stays zero.
        fft_frequencies = (
            torch.arange(self.n_fft // 2, dtype=torch.float32) * samp_freq / self.n_fft
        )
        fft_mels = 1127.0 * torch.log(1.0 + fft_frequencies / 700.0)

        left = mel_low + torch.arange(n_mels, dtype=torch.float32) * mel_delta
        center = left + mel_delta
        right = center + mel_delta
        lower = (fft_mels.unsqueeze(1) - left) / (center - left)
        upper = (right - fft_mels.unsqueeze(1)) / (right - center)
        mel_filterbank = torch.nn.functional.pad(
            torch.minimum(lower, upper).clamp_min(0.0), (0, 0, 0, 1)
        )
        self.register_buffer("mel_filterbank", mel_filterbank, persistent=False)

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract padded filterbanks and valid feature lengths.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded normalized waveforms with shape
            ``(batch_size, num_samples + frame_length // 2)``. The runtime
            appends the reflected right context after each valid waveform.
        audio_lengths : torch.Tensor[torch.int64]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.int32]]
            ``torch.float32`` log-mel features with shape
            ``(batch_size, num_frames, n_mels)`` and ``torch.int32`` valid
            frame counts with shape ``(batch_size,)``.
        """

        if torch.onnx.is_in_onnx_export():
            num_frames = (
                audio.shape[1] + self.left_padding - self.frame_length
            ) // self.frame_shift + 1
            features, feature_lengths = torch.onnx.ops.symbolic_multi_out(
                ZIPFORMER_FEATURE_PLUGIN_NAME,
                (audio, audio_lengths, self.window, self.mel_filterbank),
                {
                    "frame_length": self.frame_length,
                    "frame_shift": self.frame_shift,
                    "left_padding": self.left_padding,
                    "preemph": self.preemph,
                    "min_frames": self.min_frames,
                    "zero_log": self.zero_log,
                    "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE,
                },
                dtypes=(torch.float32, torch.int32),
                shapes=((audio.shape[0], num_frames, self.n_mels), audio_lengths.shape),
                version=ONNX_OPSET_VERSION,
            )
            return features, feature_lengths

        left_context = torch.flip(audio[:, : self.left_padding], dims=(1,))
        frames = torch.cat((left_context, audio), dim=1).unfold(
            1, self.frame_length, self.frame_shift
        )
        frames = frames - frames.mean(dim=2, keepdim=True)
        frames = torch.cat(
            (
                (1.0 - self.preemph) * frames[:, :, :1],
                frames[:, :, 1:] - self.preemph * frames[:, :, :-1],
            ),
            dim=2,
        )
        spectrum = torch.fft.rfft(frames * self.window, n=self.n_fft)
        power_spectrum = spectrum.real**2 + spectrum.imag**2
        features = torch.log(
            torch.clamp(power_spectrum @ self.mel_filterbank, min=self.log_eps)
        )

        audio_lengths = audio_lengths.clamp(min=0, max=audio.size(1))
        feature_lengths = (audio_lengths + self.frame_shift // 2) // self.frame_shift
        feature_lengths = torch.clamp(
            feature_lengths, min=self.min_frames, max=features.size(1)
        ).to(torch.int32)
        padding_mask = torch.arange(
            features.size(1), dtype=feature_lengths.dtype, device=feature_lengths.device
        ).unsqueeze(0) >= feature_lengths.unsqueeze(1)
        features = features.masked_fill(padding_mask.unsqueeze(2), self.zero_log)

        return features, feature_lengths
