#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Reference and ONNX tests for the Parakeet log-mel feature extractor."""

import math
from pathlib import Path

import numpy as np
import onnx
import pytest
import sympy
import torch

from fast_gpu_asr.constants import (
    ONNX_OPSET_VERSION,
    PARAKEET_FEATURE_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
)
from fast_gpu_asr.export.model.parakeet.features import FeatureExtractor

FEATURE_CONFIG = {
    "samp_freq": 16_000,
    "frame_shift_ms": 10,
    "frame_length_ms": 25,
    "n_mels": 16,
    "preemph": 0.97,
    "low_freq": 0,
    "high_freq": 8000,
}
CUSTOM_CONFIG = FEATURE_CONFIG | {
    "frame_shift_ms": 12,
    "frame_length_ms": 31,
    "n_mels": 12,
    "preemph": 0.5,
    "low_freq": 80,
    "high_freq": 7600,
}


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create a compact frontend with the production timing and mel bounds.

    Returns
    -------
    FeatureExtractor
        Fresh CPU frontend in evaluation mode with 16 mel bins.
    """

    return FeatureExtractor(**FEATURE_CONFIG).eval()


def make_slaney_filterbank(
    sample_rate: int = 16_000,
    n_fft: int = 512,
    num_mels: int = 16,
    low_frequency: int = 0,
    high_frequency: int = 8000,
) -> torch.Tensor:
    """Construct scalar Slaney weights independently of the vectorized frontend.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hertz.
    n_fft : int
        FFT size defining the frequency-bin spacing.
    num_mels : int
        Number of triangular mel filters.
    low_frequency : int
        Lower filterbank bound in Hertz.
    high_frequency : int
        Upper filterbank bound in Hertz.

    Returns
    -------
    torch.Tensor
        FP32 weights of shape ``(n_fft // 2 + 1, num_mels)``, normalized by
        each triangular filter's bandwidth in Hertz.
    """

    linear_hz_per_mel = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_hz_per_mel
    logstep = math.log(6.4) / 27.0
    mel_low, mel_high = (
        frequency / linear_hz_per_mel
        if frequency < min_log_hz
        else min_log_mel + math.log(frequency / min_log_hz) / logstep
        for frequency in (low_frequency, high_frequency)
    )
    mel_points = [
        mel_low + index * (mel_high - mel_low) / (num_mels + 1)
        for index in range(num_mels + 2)
    ]
    frequencies = [
        linear_hz_per_mel * mel
        if mel < min_log_mel
        else min_log_hz * math.exp(logstep * (mel - min_log_mel))
        for mel in mel_points
    ]
    filterbank = torch.zeros(n_fft // 2 + 1, num_mels)
    for fft_bin in range(n_fft // 2 + 1):
        frequency = fft_bin * sample_rate / n_fft
        for mel_bin in range(num_mels):
            lower, center, upper = frequencies[mel_bin : mel_bin + 3]
            lower_slope = (frequency - lower) / (center - lower)
            upper_slope = (upper - frequency) / (upper - center)
            filterbank[fft_bin, mel_bin] = (
                max(0.0, min(lower_slope, upper_slope)) * 2.0 / (upper - lower)
            )
    return filterbank


@pytest.fixture(scope="module")
def reference_filterbank() -> torch.Tensor:
    """Build the shared, read-only scalar reference for the default mel bounds.

    Returns
    -------
    torch.Tensor
        FP32 filterbank of shape ``(257, 16)``; callers must not mutate it.
    """

    return make_slaney_filterbank()


def extract_reference_features(
    waveform: torch.Tensor,
    mel_filterbank: torch.Tensor,
    hop_length: int = 160,
    window_length: int = 400,
    preemph: float = 0.97,
) -> torch.Tensor:
    """Extract one unpadded waveform through PyTorch's independent STFT path.

    Parameters
    ----------
    waveform : torch.Tensor
        Unpadded one-dimensional FP32 CPU waveform.
    mel_filterbank : torch.Tensor
        Frequency-major mel weights matching the FFT size.
    hop_length : int
        Frame shift in samples.
    window_length : int
        Symmetric Hann window length in samples.
    preemph : float
        Preemphasis coefficient applied before the STFT.

    Returns
    -------
    torch.Tensor
        Time-major log-mel features normalized with sample standard deviation
        over valid frames only. The extra STFT edge frame and utterances with
        fewer than two valid frames are zeroed.
    """

    preemphasized = torch.cat((waveform[:1], waveform[1:] - preemph * waveform[:-1]))
    spectrum = torch.stft(
        preemphasized,
        n_fft=2 ** (window_length - 1).bit_length(),
        hop_length=hop_length,
        win_length=window_length,
        window=torch.hann_window(window_length, periodic=False),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    features = torch.log(spectrum.abs().square().T @ mel_filterbank + 2**-24)
    valid_frames = waveform.numel() // hop_length
    normalized = torch.zeros_like(features)
    if valid_frames >= 2:
        valid = features[:valid_frames]
        normalized[:valid_frames] = (valid - valid.mean(dim=0)) / (
            valid.std(dim=0, correction=1) + 1e-5
        )
    return normalized


def test_feature_buffers_are_nonpersistent(feature_extractor: FeatureExtractor) -> None:
    buffers = dict(feature_extractor.named_buffers())

    assert set(buffers) == {"window", "mel_filterbank"}
    assert feature_extractor.state_dict() == {}
    assert buffers["window"].shape == (512,)
    assert buffers["mel_filterbank"].shape == (257, 16)
    assert all(buffer.dtype == torch.float32 for buffer in buffers.values())


@pytest.mark.parametrize(
    "frame_length_ms,expected_n_fft", [(25, 512), (32, 512), (33, 1024)]
)
def test_analysis_window_and_fft_size(
    frame_length_ms: int, expected_n_fft: int
) -> None:
    extractor = FeatureExtractor(
        **(FEATURE_CONFIG | {"frame_length_ms": frame_length_ms})
    ).eval()
    window_length = frame_length_ms * 16
    padding = (expected_n_fft - window_length) // 2
    expected_window = torch.nn.functional.pad(
        torch.hann_window(window_length, periodic=False), (padding, padding)
    )

    assert extractor.hop_length == 160
    assert extractor.n_fft == expected_n_fft
    torch.testing.assert_close(extractor.window, expected_window, atol=0.0, rtol=0.0)


def test_slaney_filterbank_matches_scalar_reference(
    feature_extractor: FeatureExtractor, reference_filterbank: torch.Tensor
) -> None:
    torch.testing.assert_close(
        feature_extractor.mel_filterbank, reference_filterbank, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize(
    "waveform_name", ["random", "dc", "first-impulse", "last-impulse", "alternating"]
)
def test_features_match_stft_reference(
    feature_extractor: FeatureExtractor,
    reference_filterbank: torch.Tensor,
    waveform_name: str,
) -> None:
    waveform = torch.zeros(3200)
    if waveform_name == "random":
        waveform = torch.randn(3200, generator=torch.Generator().manual_seed(2))
    elif waveform_name == "dc":
        waveform.fill_(0.25)
    elif waveform_name == "first-impulse":
        waveform[0] = 1.0
    elif waveform_name == "last-impulse":
        waveform[-1] = 1.0
    else:
        waveform[::2], waveform[1::2] = -1.0, 1.0

    features, lengths = feature_extractor(
        waveform.unsqueeze(0), torch.tensor([3200], dtype=torch.int64)
    )

    assert lengths.tolist() == [20]
    torch.testing.assert_close(
        features[0],
        extract_reference_features(waveform, reference_filterbank),
        atol=2e-5,
        rtol=2e-5,
    )


def test_features_honor_constructor_configuration() -> None:
    extractor = FeatureExtractor(**CUSTOM_CONFIG).eval()
    waveform = torch.randn(5003, generator=torch.Generator().manual_seed(3))
    filterbank = make_slaney_filterbank(
        num_mels=12, low_frequency=80, high_frequency=7600
    )

    features, lengths = extractor(
        waveform.unsqueeze(0), torch.tensor([5003], dtype=torch.int64)
    )
    expected = extract_reference_features(
        waveform, filterbank, hop_length=192, window_length=496, preemph=0.5
    )

    assert extractor.hop_length == 192
    assert extractor.n_fft == 512
    assert features.shape == (1, 27, 12)
    assert lengths.tolist() == [26]
    torch.testing.assert_close(
        extractor.mel_filterbank, filterbank, rtol=1e-6, atol=1e-9
    )
    torch.testing.assert_close(features[0], expected, atol=2e-5, rtol=2e-5)


def test_features_are_batch_and_padding_invariant(
    feature_extractor: FeatureExtractor,
) -> None:
    sample_counts = [0, 160, 320, 3200, 5171]
    lengths = torch.tensor(sample_counts, dtype=torch.int64)
    audio = torch.randn(5, 5171, generator=torch.Generator().manual_seed(5))

    features, feature_lengths = feature_extractor(audio, lengths)

    assert features.shape == (5, 33, 16)
    assert features.dtype == torch.float32
    assert feature_lengths.dtype == torch.int32
    assert feature_lengths.tolist() == [0, 1, 2, 20, 32]
    for index, samples in enumerate(sample_counts):
        expected, expected_lengths = feature_extractor(
            audio[index : index + 1, :samples], lengths[index : index + 1]
        )
        frames = samples // 160
        assert expected_lengths.tolist() == [frames]
        torch.testing.assert_close(
            features[index, :frames], expected[0, :frames], atol=2e-5, rtol=2e-5
        )
        assert torch.count_nonzero(features[index, frames:]) == 0
        assert torch.count_nonzero(expected[:, frames:]) == 0


@pytest.mark.parametrize("num_samples", [320, 6400], ids=["two-frames", "regular"])
def test_valid_features_have_zero_mean_and_sample_std(
    feature_extractor: FeatureExtractor, num_samples: int
) -> None:
    audio = torch.randn(1, num_samples, generator=torch.Generator().manual_seed(17))
    features, lengths = feature_extractor(
        audio, torch.tensor([num_samples], dtype=torch.int64)
    )

    assert lengths.tolist() == [num_samples // 160]
    valid = features[0, : lengths[0]]
    torch.testing.assert_close(valid.mean(dim=0), torch.zeros(16), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        valid.std(dim=0, correction=1),
        torch.ones(16),
        atol=5e-5 if num_samples == 320 else 2e-4,
        rtol=0.0,
    )
    assert torch.count_nonzero(features[:, lengths[0] :]) == 0


@pytest.mark.parametrize("num_samples", [320, 4000, 640_000])
def test_silence_normalizes_to_zero(
    feature_extractor: FeatureExtractor, num_samples: int
) -> None:
    features, lengths = feature_extractor(
        torch.zeros(1, num_samples), torch.tensor([num_samples], dtype=torch.int64)
    )

    assert features.shape == (1, num_samples // 160 + 1, 16)
    assert lengths.tolist() == [num_samples // 160]
    assert torch.count_nonzero(features) == 0


@pytest.mark.parametrize(
    "declared_length,expected_frames",
    [
        pytest.param(torch.iinfo(torch.int64).min, 0, id="int64-min"),
        (-1, 0),
        (0, 0),
        (159, 0),
        (160, 1),
        (319, 1),
        (320, 2),
        (4000, 25),
        (10000, 25),
        pytest.param(torch.iinfo(torch.int64).max, 25, id="int64-max"),
    ],
)
def test_lengths_are_clamped_and_short_utterances_are_zeroed(
    feature_extractor: FeatureExtractor,
    reference_filterbank: torch.Tensor,
    declared_length: int,
    expected_frames: int,
) -> None:
    audio = torch.randn(1, 4000, generator=torch.Generator().manual_seed(11))

    features, lengths = feature_extractor(
        audio, torch.tensor([declared_length], dtype=torch.int64)
    )

    assert features.shape == (1, 26, 16)
    assert lengths.tolist() == [expected_frames]
    assert torch.count_nonzero(features[:, expected_frames:]) == 0
    if expected_frames < 2:
        assert torch.count_nonzero(features) == 0
    else:
        clamped_length = max(0, min(declared_length, audio.size(1)))
        expected = extract_reference_features(
            audio[0, :clamped_length], reference_filterbank
        )
        torch.testing.assert_close(
            features[0, :expected_frames],
            expected[:expected_frames],
            atol=2e-5,
            rtol=2e-5,
        )


def test_features_ignore_invalid_tail_and_padding_extent(
    feature_extractor: FeatureExtractor, reference_filterbank: torch.Tensor
) -> None:
    waveform = torch.randn(321, generator=torch.Generator().manual_seed(29))
    expected = extract_reference_features(waveform, reference_filterbank)
    lengths = torch.tensor([321], dtype=torch.int64)

    for padding_size, padding_value in ((321, 1e6), (680, -1e6)):
        audio = torch.cat((waveform, torch.full((padding_size,), padding_value)))
        features, feature_lengths = feature_extractor(audio.unsqueeze(0), lengths)

        assert features.shape == (1, audio.numel() // 160 + 1, 16)
        assert feature_lengths.tolist() == [2]
        torch.testing.assert_close(features[0, :2], expected[:2], atol=2e-5, rtol=2e-5)
        assert torch.count_nonzero(features[:, 2:]) == 0


def test_features_export_as_native_plugin(tmp_path: Path) -> None:
    extractor = FeatureExtractor(**CUSTOM_CONFIG).eval()
    onnx_path = tmp_path / "parakeet_features.onnx"
    num_samples = torch.export.Dim("num_samples", min=384, max=6400)

    torch.onnx.export(
        extractor,
        (torch.zeros(2, 5003), torch.tensor([5003, 3840], dtype=torch.int64)),
        onnx_path,
        dynamic_shapes={"audio": {1: num_samples}, "audio_lengths": {}},
        input_names=("audio", "audio_lengths"),
        output_names=("features", "feature_lengths"),
        opset_version=ONNX_OPSET_VERSION,
    )

    model = onnx.load(onnx_path)
    graph = model.graph
    assert [opset.version for opset in model.opset_import if opset.domain == ""] == [
        ONNX_OPSET_VERSION
    ]
    assert [(node.domain, node.op_type) for node in graph.node] == [
        ("", PARAKEET_FEATURE_PLUGIN_NAME)
    ]
    node = graph.node[0]
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    } == {
        "eps": pytest.approx(1e-5),
        "frame_shift": 192,
        "log_eps": pytest.approx(2**-24),
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        "preemph": pytest.approx(0.5),
    }
    assert [value.name for value in graph.input] == ["audio", "audio_lengths"]
    assert len(node.input) == 4
    assert list(node.input[:2]) == ["audio", "audio_lengths"]
    assert list(node.output) == ["features", "feature_lengths"]
    assert [value.name for value in graph.output] == list(node.output)

    initializers = {value.name: value for value in graph.initializer}
    assert set(initializers) == set(node.input[2:])
    expected_window = torch.nn.functional.pad(
        torch.hann_window(496, periodic=False), (8, 8)
    )
    expected_filterbank = make_slaney_filterbank(
        num_mels=12, low_frequency=80, high_frequency=7600
    )
    for name, expected, atol, rtol in (
        (node.input[2], expected_window, 0.0, 0.0),
        (node.input[3], expected_filterbank, 1e-9, 1e-6),
    ):
        initializer = initializers[name]
        assert initializer.data_type == onnx.TensorProto.FLOAT
        assert tuple(initializer.dims) == tuple(expected.shape)
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(initializer),
            expected.numpy(),
            atol=atol,
            rtol=rtol,
        )

    tensor_types = {
        value.name: value.type.tensor_type for value in (*graph.input, *graph.output)
    }
    assert {name: value.elem_type for name, value in tensor_types.items()} == {
        "audio": onnx.TensorProto.FLOAT,
        "audio_lengths": onnx.TensorProto.INT64,
        "features": onnx.TensorProto.FLOAT,
        "feature_lengths": onnx.TensorProto.INT32,
    }
    shapes = {
        name: tuple(dim.dim_param or dim.dim_value for dim in value.shape.dim)
        for name, value in tensor_types.items()
    }
    assert shapes["audio"] == (2, "num_samples")
    assert shapes["audio_lengths"] == shapes["feature_lengths"] == (2,)
    batch, frames, channels = shapes["features"]
    assert (batch, channels) == (2, 12)
    assert isinstance(frames, str)
    samples = sympy.Symbol("num_samples", integer=True)
    frame_expression = sympy.sympify(frames, locals={"num_samples": samples})
    assert sympy.simplify(frame_expression - (samples // 192 + 1)) == 0

    # The native plugin has no ONNX schema; relabel only for structural checking.
    node.domain = TENSORRT_PLUGIN_NAMESPACE
    model.opset_import.append(onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1))
    onnx.checker.check_model(model)
