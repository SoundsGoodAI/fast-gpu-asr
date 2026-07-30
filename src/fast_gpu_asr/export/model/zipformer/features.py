#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Export-friendly Kaldi filterbank extraction for Zipformer models."""

import torch

from fast_gpu_asr.constants import ZERO_LOG


class FeatureExtractor(torch.nn.Module):
    """Compute batched Kaldi-compatible log-mel filterbank features.

    DC removal, pre-emphasis, the Povey window, and the real Fourier transform
    are combined into one fixed ``Conv1d`` kernel. This avoids materializing
    overlapping waveform frames and keeps the complete frontend exportable to
    ONNX and TensorRT.
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
        device: torch.device,
    ) -> None:
        """Initialize fixed Fourier and mel filterbank kernels.

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
        device : torch.device
            The device used to store the fixed Fourier kernels and mel
            filterbank. Should be either ``torch.device("cpu")`` or
            ``torch.device("cuda")``.
        """

        super().__init__()

        frame_length = frame_length_ms * samp_freq // 1000
        self.frame_shift = frame_shift_ms * samp_freq // 1000

        self.left_padding = frame_length // 2 - self.frame_shift // 2
        self.right_padding = frame_length // 2
        self.min_frames = min_frames
        self.log_eps = 2**-24
        self.zero_log = ZERO_LOG

        # Round the window length up to the next power-of-two FFT size.
        n_fft = 2 ** (frame_length - 1).bit_length()
        frame_positions = torch.arange(frame_length, dtype=torch.float32, device=device)
        frequencies = torch.arange(
            n_fft // 2 + 1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        angles = 2.0 * torch.pi * frequencies * frame_positions / n_fft
        fourier_kernels = torch.cat((torch.cos(angles), -torch.sin(angles)))

        povey_window = (
            0.5 - 0.5 * torch.cos(2.0 * torch.pi * frame_positions / (frame_length - 1))
        ) ** 0.85
        windowed_kernels = fourier_kernels * povey_window

        kernels = windowed_kernels.clone()
        kernels[:, 0] = (1.0 - preemph) * windowed_kernels[
            :, 0
        ] - preemph * windowed_kernels[:, 1]
        kernels[:, 1 : frame_length - 1] = (
            windowed_kernels[:, 1 : frame_length - 1]
            - preemph * windowed_kernels[:, 2:frame_length]
        )
        kernels[:, frame_length - 1] = windowed_kernels[:, frame_length - 1]
        kernels = kernels - kernels.mean(dim=1, keepdim=True)
        self.fourier_kernels = kernels.unsqueeze(1)

        mel_low = 1127.0 * torch.log(
            1.0 + torch.tensor(low_freq, dtype=torch.float32, device=device) / 700.0
        )
        mel_high = 1127.0 * torch.log(
            1.0 + torch.tensor(high_freq, dtype=torch.float32, device=device) / 700.0
        )
        mel_delta = (mel_high - mel_low) / (n_mels + 1)
        fft_frequencies = (
            torch.arange(n_fft // 2, dtype=torch.float32, device=device)
            * samp_freq
            / n_fft
        )
        fft_mels = 1127.0 * torch.log(1.0 + fft_frequencies / 700.0)

        mel_filterbank = torch.zeros(
            n_fft // 2 + 1, n_mels, dtype=torch.float32, device=device
        )
        for mel_index in range(n_mels):
            left = mel_low + mel_index * mel_delta
            center = left + mel_delta
            right = center + mel_delta
            weights = torch.where(
                (fft_mels > left) & (fft_mels < right),
                torch.where(
                    fft_mels <= center,
                    (fft_mels - left) / (center - left),
                    (right - fft_mels) / (right - center),
                ),
                0.0,
            )
            mel_filterbank[: n_fft // 2, mel_index] = weights
        self.mel_filterbank = mel_filterbank

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract padded filterbanks and valid feature lengths.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded normalized waveforms with shape
            ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int32]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Log-mel features with shape
            ``(batch_size, num_frames, n_mels)`` and ``torch.int32`` valid
            frame counts with shape ``(batch_size,)``.
        """

        sample_indexes = torch.arange(
            -self.left_padding, audio.size(1) + self.right_padding, device=audio.device
        ).unsqueeze(0)
        sample_indexes = sample_indexes.expand(audio.size(0), sample_indexes.size(1))
        sample_indexes = torch.where(
            sample_indexes < 0, -sample_indexes - 1, sample_indexes
        )
        sample_indexes = torch.where(
            sample_indexes >= audio_lengths.unsqueeze(1),
            2 * audio_lengths.unsqueeze(1) - 1 - sample_indexes,
            sample_indexes,
        )
        sample_indexes = torch.clamp(sample_indexes, min=0)
        reflected_audio = torch.gather(audio, 1, sample_indexes).unsqueeze(1)

        fourier = torch.nn.functional.conv1d(
            reflected_audio, self.fourier_kernels, stride=self.frame_shift
        )
        real, imaginary = torch.chunk(fourier, 2, dim=1)
        power_spectrum = (real**2 + imaginary**2).permute(0, 2, 1)
        features = torch.log(
            torch.clamp(power_spectrum @ self.mel_filterbank, min=self.log_eps)
        )

        feature_lengths = (audio_lengths + self.frame_shift // 2) // self.frame_shift
        feature_lengths = torch.clamp(feature_lengths, min=self.min_frames)
        padding_mask = torch.arange(
            features.size(1),
            dtype=feature_lengths.dtype,
            device=feature_lengths.device,
        ).unsqueeze(0) >= feature_lengths.unsqueeze(1)
        features = features.masked_fill(padding_mask.unsqueeze(2), self.zero_log)

        return features, feature_lengths
