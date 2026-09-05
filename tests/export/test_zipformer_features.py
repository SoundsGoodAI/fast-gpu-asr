#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Reference and ONNX tests for the Zipformer feature extractor."""

from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnx
import pytest
import sympy
import torch

from fast_gpu_asr.constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZERO_LOG,
    ZIPFORMER_FEATURE_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.zipformer.features import FeatureExtractor

FEATURE_CONFIG = {
    "samp_freq": 16_000,
    "frame_shift_ms": 10,
    "frame_length_ms": 25,
    "n_mels": 80,
    "preemph": 0.97,
    "low_freq": 20,
    "high_freq": 7600,
    "min_frames": 9,
}
CUSTOM_CONFIG = FEATURE_CONFIG | {
    "frame_shift_ms": 8,
    "frame_length_ms": 16,
    "n_mels": 13,
    "preemph": 0.5,
    "low_freq": 80,
    "high_freq": 7000,
    "min_frames": 11,
}


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create the production-configured Zipformer frontend.

    Returns
    -------
    FeatureExtractor
        Fresh CPU frontend in evaluation mode with 80 mel bins.
    """

    return FeatureExtractor(**FEATURE_CONFIG).eval()


def make_audio_batch(
    waveforms: list[torch.Tensor], right_padding: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch waveforms of at least ``right_padding`` samples with reflected tails.

    Parameters
    ----------
    waveforms : list[torch.Tensor]
        Unpadded one-dimensional CPU waveforms sharing a dtype.
    right_padding : int
        Number of final samples to reflect at each utterance boundary.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Batch padded with reflected context followed by zeros, and INT64
        sample counts excluding both forms of padding.
    """

    padded = [
        torch.cat((waveform, waveform.flip(0)[:right_padding]))
        for waveform in waveforms
    ]
    return (
        torch.nn.utils.rnn.pad_sequence(padded, batch_first=True),
        torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.int64),
    )


def extract_native(
    waveform: torch.Tensor, config: dict[str, int | float] = FEATURE_CONFIG
) -> torch.Tensor:
    """Extract one waveform with Kaldi's independent reference frontend.

    Parameters
    ----------
    waveform : torch.Tensor
        Unpadded one-dimensional FP32 CPU waveform with at least one frame.
    config : dict[str, int | float]
        Timing, mel bounds, and preemphasis settings, read without mutation.

    Returns
    -------
    torch.Tensor
        Time-major FP32 Kaldi features without minimum-frame batch padding.
    """

    options = knf.FbankOptions.from_dict(
        {
            "frame_opts": {
                "samp_freq": config["samp_freq"],
                "frame_shift_ms": config["frame_shift_ms"],
                "frame_length_ms": config["frame_length_ms"],
                "preemph_coeff": config["preemph"],
                "dither": 0.0,
                "window_type": "povey",
                "snip_edges": False,
            },
            "mel_opts": {
                "num_bins": config["n_mels"],
                "low_freq": config["low_freq"],
                "high_freq": config["high_freq"],
            },
        }
    )
    fbank = knf.OnlineFbank(options)
    fbank.accept_waveform(config["samp_freq"], waveform.numpy())
    fbank.input_finished()
    return torch.from_numpy(
        np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)])
    )


def test_feature_buffers_are_nonpersistent(feature_extractor: FeatureExtractor) -> None:
    buffers = dict(feature_extractor.named_buffers())

    assert set(buffers) == {"window", "mel_filterbank"}
    assert feature_extractor.state_dict() == {}
    assert feature_extractor.n_fft == 512
    assert buffers["window"].shape == (400,)
    assert buffers["mel_filterbank"].shape == (257, 80)
    assert all(buffer.dtype == torch.float32 for buffer in buffers.values())
    assert torch.count_nonzero(buffers["mel_filterbank"][256]) == 0


@pytest.mark.parametrize("frame_length_ms,expected_n_fft", [(32, 512), (33, 1024)])
def test_fft_size_crosses_power_of_two_boundary(
    frame_length_ms: int, expected_n_fft: int
) -> None:
    extractor = FeatureExtractor(
        **(FEATURE_CONFIG | {"frame_length_ms": frame_length_ms})
    ).eval()

    assert extractor.frame_length == frame_length_ms * 16
    assert extractor.n_fft == expected_n_fft
    assert extractor.window.shape == (frame_length_ms * 16,)
    assert extractor.mel_filterbank.shape == (expected_n_fft // 2 + 1, 80)


@pytest.mark.parametrize(
    "sample_counts,expected_lengths",
    [
        pytest.param((16000, 23789), [100, 149], id="mixed-length-batch"),
        pytest.param((1360,), [9], id="minimum-profile"),
    ],
)
def test_features_match_kaldi(
    feature_extractor: FeatureExtractor,
    sample_counts: tuple[int, ...],
    expected_lengths: list[int],
) -> None:
    generator = np.random.default_rng(0)
    waveforms = [
        torch.from_numpy(generator.normal(0.0, 0.05, count).astype(np.float32))
        for count in sample_counts
    ]

    features, lengths = feature_extractor(*make_audio_batch(waveforms))

    assert features.shape == (len(waveforms), max(expected_lengths), 80)
    assert lengths.dtype == torch.int32
    assert lengths.tolist() == expected_lengths
    for index, waveform in enumerate(waveforms):
        expected = extract_native(waveform)
        torch.testing.assert_close(
            features[index, : lengths[index]], expected, atol=5e-4, rtol=1e-4
        )
        assert (features[index, lengths[index] :] == ZERO_LOG).all()


def test_features_honor_constructor_configuration() -> None:
    extractor = FeatureExtractor(**CUSTOM_CONFIG).eval()
    waveform = torch.from_numpy(
        np.random.default_rng(17).normal(0.0, 0.05, 5003).astype(np.float32)
    )
    audio, audio_lengths = make_audio_batch([waveform], right_padding=128)

    features, lengths = extractor(audio, audio_lengths)
    expected = extract_native(waveform, CUSTOM_CONFIG)

    assert lengths.tolist() == [39]
    torch.testing.assert_close(features[0], expected, atol=5e-4, rtol=1e-4)

    minimum_features, minimum_lengths = extractor(
        audio, torch.zeros_like(audio_lengths)
    )
    assert minimum_lengths.tolist() == [11]
    torch.testing.assert_close(
        minimum_features[:, :11], features[:, :11], atol=0.0, rtol=0.0
    )
    assert (minimum_features[:, 11:] == ZERO_LOG).all()


@pytest.mark.parametrize(
    "waveform_name",
    [
        "silence",
        "dc",
        "first-impulse",
        "last-impulse",
        "alternating",
        "low-edge-tone",
        "high-edge-tone",
    ],
)
def test_features_match_kaldi_on_adversarial_waveforms(
    feature_extractor: FeatureExtractor, waveform_name: str
) -> None:
    sample_count = 16000 if waveform_name == "silence" else 3200
    waveform = np.zeros(sample_count, dtype=np.float32)
    if waveform_name == "dc":
        waveform.fill(0.25)
    elif waveform_name == "first-impulse":
        waveform[0] = 1.0
    elif waveform_name == "last-impulse":
        waveform[sample_count - 1] = 1.0
    elif waveform_name == "alternating":
        waveform[::2], waveform[1::2] = -1.0, 1.0
    elif waveform_name in ("low-edge-tone", "high-edge-tone"):
        frequency = 20.0 if waveform_name == "low-edge-tone" else 7600.0
        time = np.arange(sample_count, dtype=np.float32) / np.float32(16000)
        waveform = np.sin(2.0 * np.pi * frequency * time).astype(np.float32)
    waveform = torch.from_numpy(waveform)

    features, lengths = feature_extractor(*make_audio_batch([waveform]))
    expected = extract_native(waveform)
    atol, rtol = 5e-4, 1e-4
    if waveform_name == "silence":
        atol, rtol = 1e-6, 0.0
    elif waveform_name in ("alternating", "high-edge-tone"):
        # Near-floor bins in these signals are sensitive to CPU FFT rounding.
        rtol = 3e-3

    assert features.shape == (1, len(expected), 80)
    assert lengths.tolist() == [len(expected)]
    torch.testing.assert_close(features[0], expected, atol=atol, rtol=rtol)


def test_valid_features_ignore_trailing_padding_extent(
    feature_extractor: FeatureExtractor,
) -> None:
    waveform = torch.randn(3201, generator=torch.Generator().manual_seed(23))
    audio, lengths = make_audio_batch([waveform])
    heavily_padded = torch.cat((audio, torch.full((1, 1600), 1e6)), dim=1)

    minimal_features, minimal_lengths = feature_extractor(audio, lengths)
    padded_features, padded_lengths = feature_extractor(heavily_padded, lengths)

    assert minimal_lengths.tolist() == padded_lengths.tolist() == [20]
    assert minimal_features.shape == (1, 20, 80)
    assert padded_features.shape == (1, 30, 80)
    torch.testing.assert_close(
        minimal_features[0], extract_native(waveform), atol=5e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        padded_features[:, :20], minimal_features, atol=2e-5, rtol=2e-5
    )
    assert (padded_features[:, 20:] == ZERO_LOG).all()


@pytest.mark.parametrize(
    "declared_samples,expected_frames",
    [
        pytest.param(torch.iinfo(torch.int64).min, 9, id="int64-min"),
        pytest.param(0, 9, id="zero"),
        pytest.param(1519, 9, id="below-rounding-boundary"),
        pytest.param(1520, 10, id="at-rounding-boundary"),
        pytest.param(1800, 10, id="physical-length"),
        pytest.param(10000, 10, id="above-physical-length"),
        pytest.param(torch.iinfo(torch.int64).max, 10, id="int64-max"),
    ],
)
def test_feature_lengths_are_clamped_and_rounded(
    feature_extractor: FeatureExtractor, declared_samples: int, expected_frames: int
) -> None:
    audio = torch.linspace(-0.1, 0.1, 1800).unsqueeze(0)
    reference_features, reference_lengths = feature_extractor(
        audio, torch.tensor([1800], dtype=torch.int64)
    )

    features, lengths = feature_extractor(
        audio, torch.tensor([declared_samples], dtype=torch.int64)
    )

    assert reference_lengths.tolist() == [10]
    assert features.shape == (1, 10, 80)
    assert lengths.dtype == torch.int32
    assert lengths.tolist() == [expected_frames]
    torch.testing.assert_close(
        features[:, :expected_frames],
        reference_features[:, :expected_frames],
        atol=0.0,
        rtol=0.0,
    )
    assert (features[:, expected_frames:] == ZERO_LOG).all()


def test_features_export_as_native_plugin(tmp_path: Path) -> None:
    extractor = FeatureExtractor(**CUSTOM_CONFIG).eval()
    onnx_path = tmp_path / "zipformer_features.onnx"
    num_samples = torch.export.Dim("num_samples", min=1472, max=6400)

    torch.onnx.export(
        extractor,
        (torch.zeros(2, 3400), torch.full((2,), 3200, dtype=torch.int64)),
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
        ("", ZIPFORMER_FEATURE_PLUGIN_NAME)
    ]
    node = graph.node[0]
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    } == {
        "frame_length": 256,
        "frame_shift": 128,
        "left_padding": 64,
        "min_frames": 11,
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
        "preemph": pytest.approx(0.5),
        "zero_log": pytest.approx(ZERO_LOG),
    }
    assert [value.name for value in graph.input] == ["audio", "audio_lengths"]
    assert len(node.input) == 4
    assert list(node.input[:2]) == ["audio", "audio_lengths"]
    assert list(node.output) == ["features", "feature_lengths"]
    assert [value.name for value in graph.output] == list(node.output)

    initializers = {value.name: value for value in graph.initializer}
    assert set(initializers) == set(node.input[2:])
    for name, expected, shape in (
        (node.input[2], extractor.window, (256,)),
        (node.input[3], extractor.mel_filterbank, (129, 13)),
    ):
        initializer = initializers[name]
        assert initializer.data_type == onnx.TensorProto.FLOAT
        assert tuple(initializer.dims) == shape
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(initializer), expected.numpy()
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
    assert (batch, channels) == (2, 13)
    assert isinstance(frames, str)
    samples = sympy.Symbol("num_samples", integer=True)
    frame_expression = sympy.sympify(frames, locals={"num_samples": samples})
    assert sympy.simplify(frame_expression - ((samples - 192) // 128 + 1)) == 0

    # TensorRT uses the default domain without an ONNX schema. Relabel only for
    # structural checking after verifying the exported plugin contract above.
    node.domain = TENSORRT_PLUGIN_NAMESPACE
    model.opset_import.append(onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1))
    onnx.checker.check_model(model)
