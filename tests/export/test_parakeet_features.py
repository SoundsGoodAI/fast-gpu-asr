#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for the eager Parakeet log-mel feature extractor."""

import math

import pytest
import torch

from fast_gpu_asr.export.model.parakeet.features import FeatureExtractor

SAMPLE_RATE = 16_000
FRAME_SHIFT_MS = 10
FRAME_LENGTH_MS = 25
HOP_LENGTH = FRAME_SHIFT_MS * SAMPLE_RATE // 1000
WINDOW_LENGTH = FRAME_LENGTH_MS * SAMPLE_RATE // 1000
N_FFT = 512
NUM_MELS = 16
PREEMPH = 0.97
LOW_FREQUENCY = 0
HIGH_FREQUENCY = 8000
LOG_EPS = 2**-24
NORMALIZATION_EPS = 1e-5


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Build a compact extractor with production timing and frequency bounds."""

    return FeatureExtractor(
        samp_freq=SAMPLE_RATE,
        frame_shift_ms=FRAME_SHIFT_MS,
        frame_length_ms=FRAME_LENGTH_MS,
        n_mels=NUM_MELS,
        preemph=PREEMPH,
        low_freq=LOW_FREQUENCY,
        high_freq=HIGH_FREQUENCY,
    ).eval()


def extract_reference_features(
    waveform: torch.Tensor,
    mel_filterbank: torch.Tensor,
) -> torch.Tensor:
    """Extract normalized features through PyTorch's independent STFT path."""

    preemphasized = torch.cat(
        (waveform[:1], waveform[1:] - PREEMPH * waveform[:-1]),
    )
    spectrum = torch.stft(
        preemphasized,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WINDOW_LENGTH,
        window=torch.hann_window(WINDOW_LENGTH, periodic=False),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    features = torch.log(
        spectrum.abs().square().transpose(0, 1) @ mel_filterbank + LOG_EPS,
    )
    valid_frames = waveform.numel() // HOP_LENGTH
    normalized = torch.zeros_like(features)
    if valid_frames >= 2:
        valid_features = features[:valid_frames]
        means = valid_features.mean(dim=0)
        stds = (
            torch.sqrt(
                torch.sum((valid_features - means) ** 2, dim=0) / (valid_frames - 1),
            )
            + NORMALIZATION_EPS
        )
        normalized[:valid_frames] = (valid_features - means) / stds
    return normalized


def test_parakeet_analysis_window_matches_nemo(
    feature_extractor: FeatureExtractor,
) -> None:
    """Use NeMo's centered, nonperiodic 25 ms Hann window and 10 ms hop."""

    expected_window = torch.zeros(N_FFT)
    window_start = (N_FFT - WINDOW_LENGTH) // 2
    expected_window[window_start : window_start + WINDOW_LENGTH] = torch.hann_window(
        WINDOW_LENGTH,
        periodic=False,
    )

    assert feature_extractor.hop_length == HOP_LENGTH
    assert feature_extractor.n_fft == N_FFT
    torch.testing.assert_close(
        feature_extractor.window,
        expected_window,
        atol=0.0,
        rtol=0.0,
    )


def test_parakeet_slaney_filterbank_matches_scalar_construction(
    feature_extractor: FeatureExtractor,
) -> None:
    """Match a scalar Slaney filterbank construction independent of PyTorch ops."""

    linear_hz_per_mel = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_hz_per_mel
    logstep = math.log(6.4) / 27.0
    mel_low = 0.0
    mel_high = min_log_mel + math.log(HIGH_FREQUENCY / min_log_hz) / logstep
    mel_points = [
        mel_low + index * (mel_high - mel_low) / (NUM_MELS + 1)
        for index in range(NUM_MELS + 2)
    ]
    frequencies = [
        (
            linear_hz_per_mel * mel
            if mel < min_log_mel
            else min_log_hz * math.exp(logstep * (mel - min_log_mel))
        )
        for mel in mel_points
    ]
    expected = torch.zeros_like(feature_extractor.mel_filterbank)
    for fft_bin in range(N_FFT // 2 + 1):
        frequency = fft_bin * SAMPLE_RATE / N_FFT
        for mel_bin in range(NUM_MELS):
            lower, center, upper = frequencies[mel_bin : mel_bin + 3]
            lower_slope = (frequency - lower) / (center - lower)
            upper_slope = (upper - frequency) / (upper - center)
            expected[fft_bin, mel_bin] = (
                max(0.0, min(lower_slope, upper_slope)) * 2.0 / (upper - lower)
            )

    torch.testing.assert_close(
        feature_extractor.mel_filterbank, expected, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize(
    "waveform_name",
    ("random", "first-impulse", "last-impulse", "alternating"),
)
def test_parakeet_features_match_independent_stft_reference(
    feature_extractor: FeatureExtractor,
    waveform_name: str,
) -> None:
    """Match NeMo framing, pre-emphasis, power, log, and normalization."""

    num_samples = 3200
    if waveform_name == "random":
        waveform = torch.randn(num_samples, generator=torch.Generator().manual_seed(2))
    elif waveform_name == "first-impulse":
        waveform = torch.zeros(num_samples)
        waveform[0] = 1.0
    elif waveform_name == "last-impulse":
        waveform = torch.zeros(num_samples)
        waveform[-1] = 1.0
    else:
        waveform = torch.tensor((-1.0, 1.0)).repeat(num_samples // 2)

    actual, actual_lengths = feature_extractor(
        waveform.unsqueeze(0),
        torch.tensor((num_samples,), dtype=torch.int64),
    )
    expected = extract_reference_features(
        waveform,
        feature_extractor.mel_filterbank,
    )

    assert actual_lengths.tolist() == [num_samples // HOP_LENGTH]
    torch.testing.assert_close(actual[0], expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "length_values",
    (
        pytest.param((3200, 5171), id="regular-lengths"),
        pytest.param((0, 160, 320, 5171), id="mixed-short-and-regular"),
    ),
)
def test_parakeet_features_are_batch_and_padding_invariant(
    feature_extractor: FeatureExtractor,
    length_values: tuple[int, ...],
) -> None:
    """Match individual extraction despite mixed lengths and padded tails."""

    lengths = torch.tensor(length_values, dtype=torch.int64)
    audio = torch.randn(
        len(length_values),
        int(lengths.max()),
        generator=torch.Generator().manual_seed(5),
    )

    actual, actual_lengths = feature_extractor(audio, lengths)

    assert actual.dtype == torch.float32
    assert actual_lengths.dtype == torch.int32
    assert actual_lengths.tolist() == [length // HOP_LENGTH for length in length_values]
    for index, length in enumerate(lengths):
        expected, expected_lengths = feature_extractor(
            audio[index : index + 1, :length],
            lengths[index : index + 1],
        )
        valid_frames = int(expected_lengths[0])
        assert int(actual_lengths[index]) == valid_frames
        torch.testing.assert_close(
            actual[index, :valid_frames],
            expected[0, :valid_frames],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.count_nonzero(actual[index, valid_frames:]) == 0


def test_parakeet_features_normalize_all_valid_frames(
    feature_extractor: FeatureExtractor,
) -> None:
    """Normalize each mel channel to zero mean and unbiased unit deviation."""

    audio = torch.randn(1, 6400, generator=torch.Generator().manual_seed(7))
    audio_lengths = torch.tensor([6400], dtype=torch.int64)

    features, feature_lengths = feature_extractor(audio, audio_lengths)

    normalized_features = features[0, : feature_lengths[0]]
    torch.testing.assert_close(
        normalized_features.mean(dim=0),
        torch.zeros(NUM_MELS),
        atol=2e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        normalized_features.std(dim=0),
        torch.ones(NUM_MELS),
        atol=2e-4,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "num_samples",
    (
        pytest.param(320, id="two-frames"),
        pytest.param(4000, id="short-utterance"),
        pytest.param(640000, id="maximum-export-duration"),
    ),
)
def test_parakeet_silence_normalizes_to_zero(
    feature_extractor: FeatureExtractor, num_samples: int
) -> None:
    """Keep silent features finite and exactly zero through normalization."""

    features, feature_lengths = feature_extractor(
        torch.zeros(1, num_samples),
        torch.tensor((num_samples,), dtype=torch.int64),
    )

    assert features.shape == (1, num_samples // HOP_LENGTH + 1, NUM_MELS)
    assert feature_lengths.tolist() == [num_samples // HOP_LENGTH]
    torch.testing.assert_close(features, torch.zeros_like(features), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("declared_length", "expected_frames"),
    (
        pytest.param(torch.iinfo(torch.int64).min, 0, id="int64-min"),
        pytest.param(-1, 0, id="negative"),
        pytest.param(0, 0, id="zero"),
        pytest.param(159, 0, id="below-first-hop"),
        pytest.param(160, 1, id="first-hop"),
        pytest.param(319, 1, id="below-second-hop"),
        pytest.param(320, 2, id="second-hop"),
        pytest.param(10000, 25, id="above-physical-length"),
        pytest.param(torch.iinfo(torch.int64).max, 25, id="int64-max"),
    ),
)
def test_parakeet_features_clamp_lengths_and_zero_short_utterances(
    feature_extractor: FeatureExtractor,
    declared_length: int,
    expected_frames: int,
) -> None:
    """Clamp malformed lengths and zero sequences too short to normalize."""

    audio = torch.randn(1, 4000, generator=torch.Generator().manual_seed(11))

    features, feature_lengths = feature_extractor(
        audio, torch.tensor([declared_length], dtype=torch.int64)
    )

    assert feature_lengths.tolist() == [expected_frames]
    assert torch.isfinite(features).all()
    if expected_frames < 2:
        assert torch.count_nonzero(features) == 0
    else:
        assert torch.count_nonzero(features[:, expected_frames:]) == 0


def test_parakeet_features_ignore_invalid_tail_and_padding_extent(
    feature_extractor: FeatureExtractor,
) -> None:
    """Keep valid normalized frames independent of padded sample contents."""

    valid_length = 320
    valid_audio = torch.randn(
        valid_length,
        generator=torch.Generator().manual_seed(29),
    )
    short_padded = torch.cat((valid_audio, torch.full((321,), 1e6))).unsqueeze(0)
    long_padded = torch.cat((valid_audio, torch.full((680,), -1e6))).unsqueeze(0)
    lengths = torch.tensor((valid_length,), dtype=torch.int64)

    short_features, short_lengths = feature_extractor(short_padded, lengths)
    long_features, long_lengths = feature_extractor(long_padded, lengths)

    assert short_lengths.tolist() == long_lengths.tolist() == [2]
    torch.testing.assert_close(short_features[:, :2], long_features[:, :2])
    assert torch.count_nonzero(short_features[:, 2:]) == 0
    assert torch.count_nonzero(long_features[:, 2:]) == 0
