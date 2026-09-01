#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Reference and ONNX tests for the Zipformer feature extractor."""

from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnx
import pytest
import torch

from fast_gpu_asr.constants import (
    TENSORRT_PLUGIN_NAMESPACE,
    ZERO_LOG,
    ZIPFORMER_FEATURE_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.zipformer.features import FeatureExtractor

SAMPLE_RATE = 16_000
FRAME_SHIFT_MS = 10
FRAME_LENGTH_MS = 25
FRAME_SHIFT_SAMPLES = FRAME_SHIFT_MS * SAMPLE_RATE // 1000
FRAME_LENGTH_SAMPLES = FRAME_LENGTH_MS * SAMPLE_RATE // 1000
LEFT_PADDING_SAMPLES = FRAME_LENGTH_SAMPLES // 2 - FRAME_SHIFT_SAMPLES // 2
RIGHT_PADDING_SAMPLES = FRAME_LENGTH_SAMPLES // 2
NUM_MELS = 80
PREEMPH = 0.97
LOW_FREQUENCY = 20
HIGH_FREQUENCY = 7600
MIN_FRAMES = 9
NEXT_FRAME_BOUNDARY = (MIN_FRAMES + 1) * FRAME_SHIFT_SAMPLES - FRAME_SHIFT_SAMPLES // 2
FEATURE_CONFIG = {
    "frame_opts": {
        "samp_freq": SAMPLE_RATE,
        "frame_shift_ms": FRAME_SHIFT_MS,
        "frame_length_ms": FRAME_LENGTH_MS,
        "dither": 0.0,
        "preemph_coeff": PREEMPH,
        "window_type": "povey",
        "blackman_coeff": 0.42,
        "snip_edges": False,
    },
    "mel_opts": {
        "num_bins": NUM_MELS,
        "low_freq": LOW_FREQUENCY,
        "high_freq": HIGH_FREQUENCY,
    },
}


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create the production-compatible Zipformer frontend."""

    return FeatureExtractor(
        samp_freq=SAMPLE_RATE,
        frame_shift_ms=FRAME_SHIFT_MS,
        frame_length_ms=FRAME_LENGTH_MS,
        n_mels=NUM_MELS,
        preemph=PREEMPH,
        low_freq=LOW_FREQUENCY,
        high_freq=HIGH_FREQUENCY,
        min_frames=MIN_FRAMES,
    ).eval()


def add_right_context(
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
) -> torch.Tensor:
    """Append the reflected right context expected by the exported frontend."""

    padded = torch.zeros(
        audio.size(0),
        audio.size(1) + RIGHT_PADDING_SAMPLES,
        dtype=audio.dtype,
    )
    for index, length_tensor in enumerate(audio_lengths):
        length = int(length_tensor)
        padded[index, :length] = audio[index, :length]
        reflected_samples = min(length, RIGHT_PADDING_SAMPLES)
        padded[index, length : length + reflected_samples] = torch.flip(
            audio[index, length - reflected_samples : length],
            dims=(0,),
        )
        if reflected_samples < RIGHT_PADDING_SAMPLES:
            padding_start = length + reflected_samples
            padded[index, padding_start : length + RIGHT_PADDING_SAMPLES] = audio[
                index, 0
            ]
    return padded


def extract_native(
    audio: np.typing.NDArray[np.float32],
) -> torch.Tensor:
    """Extract one waveform with Kaldi's independent reference frontend."""

    options = knf.FbankOptions.from_dict(FEATURE_CONFIG)
    fbank = knf.OnlineFbank(options)
    fbank.accept_waveform(SAMPLE_RATE, audio)
    fbank.input_finished()
    return torch.from_numpy(
        np.stack(
            [fbank.get_frame(index) for index in range(fbank.num_frames_ready)],
        ),
    )


def get_onnx_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return fixed dimensions as integers and symbolic dimensions as strings."""

    return tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


def test_zipformer_features_match_kaldi_native_fbank(
    feature_extractor: FeatureExtractor,
) -> None:
    """Match Kaldi and fill a mixed-length batch's padded tail with ``ZERO_LOG``."""

    generator = np.random.default_rng(0)
    audio_lengths = torch.tensor([SAMPLE_RATE, 23789], dtype=torch.int64)
    audio = torch.zeros(2, int(audio_lengths.max()), dtype=torch.float32)
    waveforms = [
        generator.normal(0.0, 0.05, int(length)).astype(np.float32)
        for length in audio_lengths.tolist()
    ]
    for index, waveform in enumerate(waveforms):
        audio[index, : len(waveform)] = torch.from_numpy(waveform)

    actual, actual_lengths = feature_extractor(
        add_right_context(audio, audio_lengths), audio_lengths
    )
    expected_features = [extract_native(waveform) for waveform in waveforms]

    assert actual.shape == (2, 149, NUM_MELS)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual_lengths.dtype == torch.int32
    assert actual_lengths.tolist() == [100, 149]
    for index, expected in enumerate(expected_features):
        assert expected.shape == (int(actual_lengths[index]), NUM_MELS)
        torch.testing.assert_close(
            actual[index, : len(expected)],
            expected,
            atol=5e-4,
            rtol=1e-4,
        )
    torch.testing.assert_close(
        actual[0, actual_lengths[0] :],
        torch.full_like(actual[0, actual_lengths[0] :], ZERO_LOG),
        atol=5e-5,
        rtol=0.0,
    )


def test_zipformer_features_use_kaldi_float32_energy_floor(
    feature_extractor: FeatureExtractor,
) -> None:
    """Match Kaldi's finite float32 energy floor for a silent waveform."""

    audio_lengths = torch.tensor([SAMPLE_RATE], dtype=torch.int64)
    audio = torch.zeros(1, int(audio_lengths[0]), dtype=torch.float32)
    actual, actual_lengths = feature_extractor(
        add_right_context(audio, audio_lengths), audio_lengths
    )
    expected = extract_native(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert actual.shape == (1, len(expected), NUM_MELS)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual_lengths.tolist() == [len(expected)]
    torch.testing.assert_close(
        actual[0, : len(expected)],
        expected,
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "waveform_name",
    (
        "dc",
        "first-impulse",
        "last-impulse",
        "alternating",
        "low-edge-tone",
        "high-edge-tone",
    ),
)
def test_zipformer_features_match_kaldi_on_adversarial_waveforms(
    feature_extractor: FeatureExtractor,
    waveform_name: str,
) -> None:
    """Match Kaldi where framing, preemphasis, and mel edges are conspicuous."""

    sample_count = 3200
    if waveform_name == "dc":
        waveform = np.full(sample_count, 0.25, dtype=np.float32)
    elif waveform_name == "first-impulse":
        waveform = np.zeros(sample_count, dtype=np.float32)
        waveform[0] = 1.0
    elif waveform_name == "last-impulse":
        waveform = np.zeros(sample_count, dtype=np.float32)
        waveform[-1] = 1.0
    elif waveform_name == "alternating":
        waveform = np.tile(np.array((-1.0, 1.0), dtype=np.float32), sample_count // 2)
    else:
        frequency = (
            float(LOW_FREQUENCY)
            if waveform_name == "low-edge-tone"
            else float(HIGH_FREQUENCY)
        )
        time = np.arange(sample_count, dtype=np.float32) / np.float32(SAMPLE_RATE)
        waveform = np.sin(2.0 * np.pi * frequency * time).astype(np.float32)

    audio_lengths = torch.tensor((sample_count,), dtype=torch.int64)
    audio = torch.from_numpy(waveform).unsqueeze(0)
    actual, actual_lengths = feature_extractor(
        add_right_context(audio, audio_lengths), audio_lengths
    )
    expected = extract_native(waveform)
    relative_tolerance = (
        2e-3 if waveform_name in ("alternating", "high-edge-tone") else 1e-4
    )

    assert actual.shape == (1, len(expected), NUM_MELS)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert actual_lengths.tolist() == [len(expected)]
    torch.testing.assert_close(
        actual[0, : len(expected)],
        expected,
        atol=5e-4,
        rtol=relative_tolerance,
    )


@pytest.mark.parametrize(
    "declared_samples,expected_frames",
    (
        pytest.param(torch.iinfo(torch.int64).min, MIN_FRAMES, id="int64-min"),
        pytest.param(0, MIN_FRAMES, id="zero"),
        pytest.param(
            NEXT_FRAME_BOUNDARY - 1,
            MIN_FRAMES,
            id="below-rounding-boundary",
        ),
        pytest.param(
            NEXT_FRAME_BOUNDARY,
            MIN_FRAMES + 1,
            id="at-rounding-boundary",
        ),
        pytest.param(1800, 10, id="physical-length"),
        pytest.param(10_000, 10, id="above-physical-length"),
        pytest.param(torch.iinfo(torch.int64).max, 10, id="int64-max"),
    ),
)
def test_zipformer_feature_lengths_are_clamped_and_rounded(
    feature_extractor: FeatureExtractor,
    declared_samples: int,
    expected_frames: int,
) -> None:
    """Clamp malformed lengths and preserve Kaldi's nearest-frame boundaries."""

    audio_samples = 1800
    audio = torch.linspace(-0.1, 0.1, audio_samples).unsqueeze(0)
    audio_lengths = torch.tensor([declared_samples], dtype=torch.int64)

    features, feature_lengths = feature_extractor(audio, audio_lengths)

    assert features.shape == (1, 10, NUM_MELS)
    assert features.dtype == torch.float32
    assert feature_lengths.dtype == torch.int32
    assert feature_lengths.tolist() == [expected_frames]
    torch.testing.assert_close(
        features[0, expected_frames:],
        torch.full_like(features[0, expected_frames:], ZERO_LOG),
        atol=5e-5,
        rtol=0.0,
    )


def test_zipformer_features_export_as_native_plugin(
    tmp_path: Path, feature_extractor: FeatureExtractor
) -> None:
    """Preserve the complete dynamic TensorRT feature-plugin ONNX contract."""

    audio = torch.zeros(2, 3400, dtype=torch.float32)
    audio_lengths = torch.full((2,), 3200, dtype=torch.int64)
    onnx_path = tmp_path / "zipformer_features.onnx"
    num_samples = torch.export.Dim("num_samples", min=1600, max=6400)

    torch.onnx.export(
        feature_extractor,
        (audio, audio_lengths),
        onnx_path,
        dynamic_shapes={
            "audio": {1: num_samples},
            "audio_lengths": {},
        },
        input_names=("audio", "audio_lengths"),
        output_names=("features", "feature_lengths"),
        opset_version=20,
    )

    model = onnx.load(onnx_path)
    graph = model.graph
    assert [(node.domain, node.op_type) for node in graph.node] == [
        ("", ZIPFORMER_FEATURE_PLUGIN_NAME)
    ]
    custom_node = graph.node[0]

    # TensorRT plugin nodes intentionally use the default ONNX domain and have no
    # ONNX schemas. Check an equivalent copy in a custom domain so the checker can
    # still validate the graph structure and tensor references.
    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(model)
    checker_model.graph.node[0].domain = TENSORRT_PLUGIN_NAMESPACE
    checker_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(checker_model)

    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    }
    assert attributes == {
        "frame_length": FRAME_LENGTH_SAMPLES,
        "frame_shift": FRAME_SHIFT_SAMPLES,
        "left_padding": LEFT_PADDING_SAMPLES,
        "min_frames": MIN_FRAMES,
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        "preemph": pytest.approx(PREEMPH),
        "zero_log": pytest.approx(ZERO_LOG),
    }
    assert [value.name for value in graph.input] == ["audio", "audio_lengths"]
    assert len(custom_node.input) == 4
    assert list(custom_node.input[:2]) == ["audio", "audio_lengths"]
    assert list(custom_node.output) == ["features", "feature_lengths"]
    assert [value.name for value in graph.output] == list(custom_node.output)

    initializers = {initializer.name: initializer for initializer in graph.initializer}
    assert set(initializers) == set(custom_node.input[2:])
    window = initializers[custom_node.input[2]]
    mel_filterbank = initializers[custom_node.input[3]]
    assert (window.data_type, tuple(window.dims)) == (
        onnx.TensorProto.FLOAT,
        (FRAME_LENGTH_SAMPLES,),
    )
    assert (mel_filterbank.data_type, tuple(mel_filterbank.dims)) == (
        onnx.TensorProto.FLOAT,
        (feature_extractor.n_fft // 2 + 1, NUM_MELS),
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(window), feature_extractor.window.numpy()
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(mel_filterbank),
        feature_extractor.mel_filterbank.numpy(),
    )

    value_info = {
        value.name: value for value in (*graph.input, *graph.value_info, *graph.output)
    }
    assert value_info["audio"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert (
        value_info["audio_lengths"].type.tensor_type.elem_type == onnx.TensorProto.INT64
    )
    assert value_info["features"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert (
        value_info["feature_lengths"].type.tensor_type.elem_type
        == onnx.TensorProto.INT32
    )
    assert get_onnx_shape(value_info["audio"]) == (2, "num_samples")
    assert get_onnx_shape(value_info["audio_lengths"]) == (2,)
    feature_shape = get_onnx_shape(value_info["features"])
    assert feature_shape[0::2] == (2, NUM_MELS)
    frame_offset = FRAME_LENGTH_SAMPLES - LEFT_PADDING_SAMPLES
    assert (
        str(feature_shape[1]).replace(" ", "")
        == f"(((num_samples-{frame_offset})//{FRAME_SHIFT_SAMPLES}))+1"
    )
    assert get_onnx_shape(value_info["feature_lengths"]) == (2,)
