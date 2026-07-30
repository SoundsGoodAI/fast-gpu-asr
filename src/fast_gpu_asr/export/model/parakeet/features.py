#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""
Audio features compatible with NVIDIA Parakeet TDT models.

Both supported Parakeet TDT checkpoints use the same preprocessor:
16 kHz mono PCM, 25 ms Hann windows, 10 ms frame shift, 512-point FFT,
log energies, 0.97 pre-emphasis, no inference-time dither, and per-feature
utterance normalization.

The implementation keeps shared parameters fixed in ``__init__`` instead of exposing
unused generic arguments. The STFT is written as a fixed Fourier ``conv1d`` so
the module exports cleanly to ONNX.
"""

import torch


class FeatureExtractor(torch.nn.Module):
    """
    Feature extractor for Parakeet TDT.
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
        device: torch.device,
    ) -> None:
        """
        FeatureExtractor initialization.

        Parameters
        ----------
        samp_freq : int
            The model sampling rate in Hertz.
        frame_shift_ms : int
            The hop size between neighboring frames in milliseconds.
        frame_length_ms : int
            The analysis window size in milliseconds.
        n_mels : int
            The number of mel filterbank channels.
        preemph : float
            The waveform pre-emphasis coefficient.
        low_freq : int
            The lower mel filterbank frequency in Hertz.
        high_freq : int
            The upper mel filterbank frequency in Hertz.
        device : torch.device
            The device used to store the fixed Fourier kernels and mel filterbank.
            Should be either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        win_length = frame_length_ms * samp_freq // 1000
        self.hop_length = frame_shift_ms * samp_freq // 1000

        self.preemph = preemph
        # Round the window length up to the next power-of-two FFT size.
        self.n_fft = 2 ** (win_length - 1).bit_length()
        self.eps = 1e-5
        self.log_eps = 2**-24

        window = torch.zeros(self.n_fft, dtype=torch.float32, device=device)
        window_start = (self.n_fft - win_length) // 2
        window[window_start : window_start + win_length] = torch.hann_window(
            win_length, periodic=False, dtype=torch.float32, device=device
        )

        frame_positions = torch.arange(
            self.n_fft, dtype=torch.float32, device=device
        ).unsqueeze(0)
        frequencies = torch.arange(
            self.n_fft // 2 + 1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        angles = 2.0 * torch.pi * frequencies * frame_positions / self.n_fft
        self.fourier_kernels = torch.cat(
            (torch.cos(angles) * window, -torch.sin(angles) * window), dim=0
        ).unsqueeze(1)

        linear_hz_per_mel = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / linear_hz_per_mel
        logstep = (
            torch.log(torch.tensor(6.4, dtype=torch.float64, device=device)) / 27.0
        )
        frequencies = torch.tensor(
            [low_freq, high_freq], dtype=torch.float64, device=device
        )

        min_mel, max_mel = torch.where(
            frequencies >= min_log_hz,
            min_log_mel + torch.log(frequencies / min_log_hz) / logstep,
            frequencies / linear_hz_per_mel,
        )
        mel_frequencies = min_mel + (max_mel - min_mel) * torch.linspace(
            0.0, 1.0, n_mels + 2, dtype=torch.float64, device=device
        )
        mel_frequencies = torch.where(
            mel_frequencies >= min_log_mel,
            min_log_hz * torch.exp(logstep * (mel_frequencies - min_log_mel)),
            linear_hz_per_mel * mel_frequencies,
        )
        mel_diffs = mel_frequencies[1:] - mel_frequencies[:-1]

        ramps = torch.linspace(
            0.0,
            samp_freq / 2.0,
            self.n_fft // 2 + 1,
            dtype=torch.float64,
            device=device,
        ).unsqueeze(1) - mel_frequencies.unsqueeze(0)
        lower = ramps[:, :-2] / mel_diffs[:-1].unsqueeze(0)
        upper = -ramps[:, 2:] / mel_diffs[1:].unsqueeze(0)
        mel_filterbank = torch.clamp(torch.minimum(lower, upper), min=0.0)
        mel_filterbank = mel_filterbank * (
            2.0 / (mel_frequencies[2 : n_mels + 2] - mel_frequencies[:n_mels])
        ).unsqueeze(0)

        self.mel_filterbank = mel_filterbank.to(torch.float32)

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the FeatureExtractor module.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded waveforms with shape ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int32]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Normalized features with shape
            ``(batch_size, num_frames, n_mels)`` and ``torch.int32`` valid
            frame counts with shape ``(batch_size,)``. The STFT emits one extra
            right-edge frame; it is zeroed and excluded from the valid lengths to
            match Parakeet inference.
        """

        audio = torch.cat(
            (audio[:, :1], audio[:, 1:] - self.preemph * audio[:, : audio.size(1) - 1]),
            dim=1,
        )

        sample_mask = torch.arange(
            audio.size(1),
            dtype=audio_lengths.dtype,
            device=audio_lengths.device,
        ).unsqueeze(0) >= audio_lengths.unsqueeze(1)
        audio = audio.masked_fill(sample_mask, 0.0).unsqueeze(1)

        audio = torch.nn.functional.pad(audio, (self.n_fft // 2, self.n_fft // 2))
        fourier = torch.nn.functional.conv1d(
            audio, self.fourier_kernels, stride=self.hop_length
        )
        real, imag = torch.chunk(fourier, 2, dim=1)
        power_spectrum = (real**2 + imag**2).permute(0, 2, 1)
        features = torch.log(power_spectrum @ self.mel_filterbank + self.log_eps)

        feature_lengths = audio_lengths // self.hop_length
        frame_mask = (
            torch.arange(
                features.size(1),
                dtype=feature_lengths.dtype,
                device=feature_lengths.device,
            ).unsqueeze(0)
            >= feature_lengths.unsqueeze(1)
        ).unsqueeze(2)
        valid_frames = (~frame_mask).to(features.dtype)
        means = torch.sum(features * valid_frames, dim=1, keepdim=True)
        means = means / feature_lengths.unsqueeze(1).unsqueeze(2)
        features = (features - means).masked_fill(frame_mask, 0.0)
        stds = (
            torch.sqrt(
                torch.sum(features**2, dim=1, keepdim=True)
                / (feature_lengths.unsqueeze(1).unsqueeze(2) - 1),
            )
            + self.eps
        )
        features = features / stds

        return features, feature_lengths
