#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2018 Ryan Leary
# Adapted for batched TensorRT export; see the upstream notices in NOTICE.
"""Export-friendly log-mel extraction for NVIDIA Parakeet TDT models."""

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    PARAKEET_FEATURE_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
)


class FeatureExtractor(torch.nn.Module):
    """Compute batched log-mel features compatible with NeMo.

    Both supported Parakeet TDT checkpoints use 16 kHz mono PCM, 25 ms Hann
    windows, a 10 ms frame shift, a 512-point FFT, power spectra, Slaney mel
    normalization, 0.97 pre-emphasis, no inference-time dither, and per-feature
    utterance normalization. Dynamo ONNX export emits the complete frontend as
    one native cuFFT-based TensorRT plugin, while eager execution computes the
    equivalent operations with PyTorch.
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
    ) -> None:
        """Initialize the centered Hann window and Slaney mel filterbank.

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
            Waveform pre-emphasis coefficient.
        low_freq : int
            Lower mel filterbank frequency in Hertz.
        high_freq : int
            Upper mel filterbank frequency in Hertz.
        """

        super().__init__()

        win_length = frame_length_ms * samp_freq // 1000
        self.hop_length = frame_shift_ms * samp_freq // 1000

        self.preemph = preemph
        # Round the window length up to the next power-of-two FFT size.
        self.n_fft = 2 ** (win_length - 1).bit_length()
        self.eps = 1e-5
        self.log_eps = 2**-24

        window = torch.zeros(self.n_fft, dtype=torch.float32)
        window_start = (self.n_fft - win_length) // 2
        window[window_start : window_start + win_length] = torch.hann_window(
            win_length, periodic=False, dtype=torch.float32
        )
        self.register_buffer("window", window, persistent=False)

        linear_hz_per_mel = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_hz_per_mel
        logstep = torch.log(torch.tensor(6.4, dtype=torch.float64)) / 27.0
        frequencies = torch.tensor([low_freq, high_freq], dtype=torch.float64)

        min_mel, max_mel = torch.where(
            frequencies >= min_log_hz,
            min_log_mel + torch.log(frequencies / min_log_hz) / logstep,
            frequencies / linear_hz_per_mel,
        )
        mel_frequencies = min_mel + (max_mel - min_mel) * torch.linspace(
            0.0, 1.0, n_mels + 2, dtype=torch.float64
        )
        mel_frequencies = torch.where(
            mel_frequencies >= min_log_mel,
            min_log_hz * torch.exp(logstep * (mel_frequencies - min_log_mel)),
            linear_hz_per_mel * mel_frequencies,
        )
        mel_diffs = mel_frequencies[1:] - mel_frequencies[:-1]

        ramps = torch.linspace(
            0.0, samp_freq / 2.0, self.n_fft // 2 + 1, dtype=torch.float64
        ).unsqueeze(1) - mel_frequencies.unsqueeze(0)
        lower = ramps[:, :-2] / mel_diffs[:-1].unsqueeze(0)
        upper = -ramps[:, 2:] / mel_diffs[1:].unsqueeze(0)
        mel_filterbank = torch.clamp(torch.minimum(lower, upper), min=0.0)
        mel_filterbank = mel_filterbank * (
            2.0 / (mel_frequencies[2 : n_mels + 2] - mel_frequencies[:n_mels])
        ).unsqueeze(0)

        self.register_buffer(
            "mel_filterbank", mel_filterbank.to(torch.float32), persistent=False
        )

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized log-mel features and valid frame counts.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded mono waveforms with shape ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int64]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.int32]]
            ``torch.float32`` normalized features with shape
            ``(batch_size, num_frames, n_mels)`` and ``torch.int32`` valid
            frame counts with shape ``(batch_size,)``. The STFT emits one extra
            right-edge frame; it is zeroed and excluded from the valid lengths to
            match Parakeet inference.
        """

        if torch.onnx.is_in_onnx_export():
            num_frames = audio.shape[1] // self.hop_length + 1
            features, feature_lengths = torch.onnx.ops.symbolic_multi_out(
                PARAKEET_FEATURE_PLUGIN_NAME,
                (audio, audio_lengths, self.window, self.mel_filterbank),
                {
                    "frame_shift": self.hop_length,
                    "preemph": self.preemph,
                    "log_eps": self.log_eps,
                    "eps": self.eps,
                    "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE,
                },
                dtypes=(torch.float32, torch.int32),
                shapes=(
                    (audio.shape[0], num_frames, self.mel_filterbank.shape[1]),
                    audio_lengths.shape,
                ),
                version=ONNX_OPSET_VERSION,
            )
            return features, feature_lengths

        audio_lengths = audio_lengths.clamp(min=0, max=audio.size(1))
        audio = torch.cat(
            (audio[:, :1], audio[:, 1:] - self.preemph * audio[:, : audio.size(1) - 1]),
            dim=1,
        )
        sample_mask = torch.arange(
            audio.size(1), dtype=audio_lengths.dtype, device=audio_lengths.device
        ).unsqueeze(0) >= audio_lengths.unsqueeze(1)
        audio = audio.masked_fill(sample_mask, 0.0)

        frames = torch.nn.functional.pad(
            audio, (self.n_fft // 2, self.n_fft // 2)
        ).unfold(1, self.n_fft, self.hop_length)
        spectrum = torch.fft.rfft(frames * self.window, n=self.n_fft)
        power_spectrum = spectrum.real**2 + spectrum.imag**2
        features = torch.log(power_spectrum @ self.mel_filterbank + self.log_eps)

        feature_lengths = (audio_lengths // self.hop_length).to(torch.int32)
        frame_mask = torch.arange(
            features.size(1),
            dtype=feature_lengths.dtype,
            device=feature_lengths.device,
        ).unsqueeze(0) >= feature_lengths.unsqueeze(1)
        valid_frames = (~frame_mask).unsqueeze(2).to(features.dtype)
        normalization_lengths = feature_lengths.clamp_min(1).unsqueeze(1).unsqueeze(2)
        # Center before reducing so a constant feature sequence has an exactly
        # representable zero mean, matching the plugin's Welford accumulator.
        offset = features[:, :1]
        means = (
            offset
            + torch.sum((features - offset) * valid_frames, dim=1, keepdim=True)
            / normalization_lengths
        )
        features = (features - means).masked_fill(frame_mask.unsqueeze(2), 0.0)
        stds = (
            torch.sqrt(
                torch.sum(features**2, dim=1, keepdim=True)
                / (normalization_lengths - 1).clamp_min(1)
            )
            + self.eps
        )

        features = (features / stds).masked_fill(
            (feature_lengths < 2).unsqueeze(1).unsqueeze(2), 0.0
        )

        return features, feature_lengths
