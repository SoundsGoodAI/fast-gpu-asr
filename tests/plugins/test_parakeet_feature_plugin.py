#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet feature plugin."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import (
    PARAKEET_FEATURE_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
)
from fast_gpu_asr.export.model.parakeet.features import FeatureExtractor

pytestmark = pytest.mark.cuda

PLUGIN_NAME = PARAKEET_FEATURE_PLUGIN_NAME
PLUGIN_VERSION = "1"
SAMPLE_RATE = 16000
FRAME_SHIFT = 160
NUM_FEATURES = 128
PREEMPH = 0.97
LOG_EPS = 2**-24
NORMALIZATION_EPS = 1e-5
PROFILE_SHAPES = ((1, 640), (3, 4000), (256, 4000))
FFT_LENGTH = 512
MEL_FREQUENCIES = FFT_LENGTH // 2 + 1
MAX_FREQUENCIES = (48 << 10) // np.dtype(np.complex64).itemsize
MAX_FFT_LENGTH = 2 * (MAX_FREQUENCIES - 1)
FEATURE_RTOL = 3e-4
FEATURE_ATOL = 3e-2
FEATURE_RMSE_ATOL = 2e-3
FIELD_NAMES = ("frame_shift", "preemph", "log_eps", "eps")
INT32_SENTINEL = np.iinfo(np.int32).min

type PluginCreatorFixture = tuple[list[ctypes.CDLL], trt.IPluginCreatorV3One]
type FeatureEngine = tuple[trt.Runtime, trt.ICudaEngine, FeatureExtractor]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


def make_extractor(
    frame_shift_ms: int = 10,
    frame_length_ms: int = 25,
    n_mels: int = NUM_FEATURES,
    preemph: float = PREEMPH,
    low_freq: int = 0,
    high_freq: int = 8000,
) -> FeatureExtractor:
    """Create the Parakeet frontend represented by the native plugin.

    Parameters
    ----------
    frame_shift_ms : int
        Frame hop in milliseconds.
    frame_length_ms : int
        Analysis window length in milliseconds.
    n_mels : int
        Number of mel-frequency output bins.
    preemph : float
        Preemphasis coefficient applied before spectral analysis.
    low_freq : int
        Lower mel-filterbank frequency in Hz.
    high_freq : int
        Upper mel-filterbank frequency in Hz.

    Returns
    -------
    FeatureExtractor
        CPU FP32 frontend in evaluation mode.
    """

    return FeatureExtractor(
        samp_freq=SAMPLE_RATE,
        frame_shift_ms=frame_shift_ms,
        frame_length_ms=frame_length_ms,
        n_mels=n_mels,
        preemph=preemph,
        low_freq=low_freq,
        high_freq=high_freq,
    ).eval()


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory) -> PluginCreatorFixture:
    """Compile and register Zipformer before the Parakeet feature plugin.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for isolated, module-scoped compiled libraries.

    Returns
    -------
    PluginCreatorFixture
        Library handle(s) and the registered creator, retained for dependent
        engines.
    """

    libraries = []
    # Production loads both DSOs with RTLD_GLOBAL. Preserve that order here to
    # catch accidental C++ symbol collisions between the two feature plugins.
    for source_name, initializer_name in (
        ("zipformer_feature_plugin.cu", "initFastGpuAsrZipformerFeaturePlugin"),
        ("parakeet_feature_plugin.cu", "initFastGpuAsrParakeetFeaturePlugin"),
    ):
        libraries.append(
            compile_and_load_plugin(
                tmp_path_factory,
                source_name,
                initializer_name,
                ("cublas", "cudart", "cufft"),
            )
        )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return libraries, creator


def make_fields(
    overrides: dict[str, np.typing.NDArray] | None = None,
    extractor: FeatureExtractor | None = None,
) -> list[trt.PluginField]:
    """Create typed fields from frontend parameters, allowing malformed test values.

    Parameters
    ----------
    overrides : dict[str, np.typing.NDArray] | None
        Replacement serialized field values; malformed values are allowed for
        negative tests.
    extractor : FeatureExtractor or None
        Frontend supplying constants and parameters; None uses the default test
        frontend.

    Returns
    -------
    list[trt.PluginField]
        Typed fields ready for a PluginFieldCollection.
    """

    values = {
        "frame_shift": np.array(
            [FRAME_SHIFT if extractor is None else extractor.hop_length],
            dtype=np.int32,
        ),
        "preemph": np.array(
            [PREEMPH if extractor is None else extractor.preemph],
            dtype=np.float32,
        ),
        "log_eps": np.array(
            [LOG_EPS if extractor is None else extractor.log_eps],
            dtype=np.float32,
        ),
        "eps": np.array(
            [NORMALIZATION_EPS if extractor is None else extractor.eps],
            dtype=np.float32,
        ),
    }
    values.update(overrides or {})
    return [
        trt.PluginField(
            name, value, getattr(trt.PluginFieldType, value.dtype.name.upper())
        )
        for name, value in values.items()
    ]


def make_plugin(
    creator: trt.IPluginCreatorV3One,
    overrides: dict[str, np.typing.NDArray] | None = None,
    extractor: FeatureExtractor | None = None,
) -> trt.IPluginV3:
    """Create a valid Parakeet feature plugin.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    overrides : dict[str, np.typing.NDArray] | None
        Replacement serialized field values; malformed values are allowed for
        negative tests.
    extractor : FeatureExtractor or None
        Frontend supplying constants and parameters; None uses the default test
        frontend.

    Returns
    -------
    trt.IPluginV3
        New plugin configured for the build phase.
    """

    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(make_fields(overrides, extractor)),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def set_profile_shape(
    profile: trt.IOptimizationProfile,
    name: str,
    min_shape: tuple[int, ...],
    opt_shape: tuple[int, ...],
    max_shape: tuple[int, ...],
) -> None:
    """Set and read back one dynamic profile to reject setup false positives.

    Parameters
    ----------
    profile : trt.IOptimizationProfile
        Profile receiving the bounds, which are read back to verify test setup.
    name : str
        Tensor name used for the optimization profile.
    min_shape : tuple[int, ...]
        Minimum input shape.
    opt_shape : tuple[int, ...]
        Optimum input shape used during tactic selection.
    max_shape : tuple[int, ...]
        Maximum input shape.
    """

    profile.set_shape(name, min_shape, opt_shape, max_shape)
    assert tuple(map(tuple, profile.get_shape(name))) == (
        min_shape,
        opt_shape,
        max_shape,
    )


def build_feature_engine(
    creator: trt.IPluginCreatorV3One,
    extractor: FeatureExtractor,
    profile_shapes: tuple[tuple[int, int], ...],
    length_batches: tuple[int, ...] | None = None,
) -> tuple[trt.Runtime, trt.ICudaEngine] | None:
    """Build and deserialize a dynamic frontend, or return None on profile rejection.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    profile_shapes : tuple[tuple[int, int], ...]
        Minimum, optimum, and maximum (batch, audio_samples) input shapes.
    length_batches : tuple[int, ...] or None
        Independent min/opt/max batch bounds for the valid-length input.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine] | None
        Deserialized engine with its owning runtime, or None on build rejection.
    """

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    audio = network.add_input("audio", trt.float32, (-1, -1))
    audio_lengths = network.add_input("audio_lengths", trt.int64, (-1,))
    window = network.add_constant(extractor.window.shape, extractor.window.numpy())
    mel = network.add_constant(
        extractor.mel_filterbank.shape, extractor.mel_filterbank.numpy()
    )
    assert all(tensor is not None for tensor in (audio, audio_lengths, window, mel))
    layer = network.add_plugin_v3(
        [audio, audio_lengths, window.get_output(0), mel.get_output(0)],
        [],
        make_plugin(creator, extractor=extractor),
    )
    assert layer is not None
    for index, name in enumerate(("features", "feature_lengths")):
        output = layer.get_output(index)
        output.name = name
        network.mark_output(output)

    profile = builder.create_optimization_profile()
    set_profile_shape(profile, "audio", *profile_shapes)
    if length_batches is None:
        length_batches = tuple(batch for batch, _ in profile_shapes)
    set_profile_shape(profile, "audio_lengths", *((batch,) for batch in length_batches))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 3
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        "audio": (trt.TensorIOMode.INPUT, trt.float32),
        "audio_lengths": (trt.TensorIOMode.INPUT, trt.int64),
        "features": (trt.TensorIOMode.OUTPUT, trt.float32),
        "feature_lengths": (trt.TensorIOMode.OUTPUT, trt.int32),
    }
    assert {
        name: (engine.get_tensor_mode(name), engine.get_tensor_dtype(name))
        for name in engine
    } == expected_io
    return runtime, engine


@pytest.fixture(scope="module")
def feature_engine(plugin_creator: PluginCreatorFixture) -> FeatureEngine:
    """Build the shared default frontend and retain its TensorRT runtime.

    Parameters
    ----------
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    FeatureEngine
        Owning runtime, reusable dynamic engine, and matching eager frontend.
    """

    _, creator = plugin_creator
    extractor = make_extractor()
    result = build_feature_engine(creator, extractor, PROFILE_SHAPES)
    assert result is not None
    return *result, extractor


def make_audio(
    lengths: np.typing.NDArray[np.int64], audio_samples: int, seed: int = 0
) -> np.typing.NDArray[np.float32]:
    """Create padded waveforms whose invalid tails must not affect features.

    Parameters
    ----------
    lengths : np.typing.NDArray[np.int64]
        INT64 valid sample counts, one per waveform.
    audio_samples : int
        Physical sample count per waveform, including padding.
    seed : int
        Local random-generator seed; does not change global NumPy or Torch state.

    Returns
    -------
    np.typing.NDArray[np.float32]
        FP32 audio in (batch, audio_samples) layout, with conspicuous invalid tails.
    """

    assert np.all(lengths >= 0)
    assert np.all(lengths <= audio_samples)
    rng = np.random.default_rng(seed + audio_samples + int(lengths.sum()))
    audio = rng.normal(0.0, 0.05, (len(lengths), audio_samples)).astype(np.float32)
    for index, length in enumerate(lengths):
        audio[index, length:] = 10.0
    return audio


@dataclass
class FeatureRun:
    """Device buffers and execution state retained after one inference."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    audio: cp.ndarray
    lengths: cp.ndarray
    features: cp.ndarray
    feature_lengths: cp.ndarray


def prepare_run(
    engine: trt.ICudaEngine,
    audio: np.typing.NDArray,
    lengths: np.typing.NDArray,
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    extractor: FeatureExtractor | None = None,
) -> FeatureRun:
    """Bind inputs and sentinel-filled outputs without running inference.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized engine whose runtime must remain alive during execution.
    audio : np.typing.NDArray
        FP32 padded audio with shape (batch, audio_samples).
    lengths : np.typing.NDArray
        INT64 valid sample counts, one per waveform.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    extractor : FeatureExtractor or None
        Frontend supplying constants and parameters; None uses the default test
        frontend.

    Returns
    -------
    FeatureRun
        Bound buffers and execution state; no inference has been enqueued.
    """

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert audio.dtype == np.float32 and lengths.dtype == np.int64
    assert context.set_input_shape("audio", audio.shape)
    assert context.set_input_shape("audio_lengths", lengths.shape)
    assert context.infer_shapes() == []
    frame_shift = FRAME_SHIFT if extractor is None else extractor.hop_length
    num_features = (
        NUM_FEATURES if extractor is None else extractor.mel_filterbank.shape[1]
    )
    feature_shape = tuple(context.get_tensor_shape("features"))
    assert feature_shape == (
        audio.shape[0],
        audio.shape[1] // frame_shift + 1,
        num_features,
    )
    assert tuple(context.get_tensor_shape("feature_lengths")) == lengths.shape

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        audio_device = cp.array(audio)
        lengths_device = cp.array(lengths)
        features = cp.full(feature_shape, cp.nan, dtype=cp.float32)
        feature_lengths = cp.full(lengths.shape, INT32_SENTINEL, dtype=cp.int32)
        for name, buffer in (
            ("audio", audio_device),
            ("audio_lengths", lengths_device),
            ("features", features),
            ("feature_lengths", feature_lengths),
        ):
            assert context.set_tensor_address(name, buffer.data.ptr)
    return FeatureRun(
        context, stream, audio_device, lengths_device, features, feature_lengths
    )


def run_engine(
    engine: trt.ICudaEngine,
    audio: np.typing.NDArray,
    lengths: np.typing.NDArray,
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    extractor: FeatureExtractor | None = None,
) -> FeatureRun:
    """Enqueue inference; output comparisons synchronize the retained stream.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized engine whose runtime must remain alive during execution.
    audio : np.typing.NDArray
        FP32 padded audio with shape (batch, audio_samples).
    lengths : np.typing.NDArray
        INT64 valid sample counts, one per waveform.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    extractor : FeatureExtractor or None
        Frontend supplying constants and parameters; None uses the default test
        frontend.

    Returns
    -------
    FeatureRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    run = prepare_run(engine, audio, lengths, context, stream, extractor)
    with run.stream:
        assert run.context.execute_async_v3(run.stream.ptr)
    return run


def assert_run_matches_pytorch(
    run: FeatureRun,
    extractor: FeatureExtractor,
    audio: np.typing.NDArray,
    lengths: np.typing.NDArray,
    atol: float = FEATURE_ATOL,
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Check untouched inputs, eager feature parity, and exact zero padding.

    Parameters
    ----------
    run : FeatureRun
        Bound device buffers and the context/stream that own their pending work.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    audio : np.typing.NDArray
        FP32 padded audio with shape (batch, audio_samples).
    lengths : np.typing.NDArray
        INT64 valid sample counts, one per waveform.
    atol : float
        Maximum elementwise absolute error; the independent RMSE bound still
        applies.

    Returns
    -------
    tuple[np.typing.NDArray, np.typing.NDArray]
        Completed FP32 features and INT32 valid frame counts copied to the host.
    """

    run.stream.synchronize()
    actual = cp.asnumpy(run.features)
    actual_lengths = cp.asnumpy(run.feature_lengths)
    np.testing.assert_array_equal(cp.asnumpy(run.audio), audio)
    np.testing.assert_array_equal(cp.asnumpy(run.lengths), lengths)
    with torch.inference_mode():
        expected, expected_lengths = extractor(
            torch.from_numpy(audio), torch.from_numpy(lengths)
        )
    expected = expected.numpy()
    assert actual.dtype == np.float32 and actual_lengths.dtype == np.int32
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())
    # FAST_16BF mel GEMM can produce sparse outliers; also bound the aggregate error.
    np.testing.assert_allclose(actual, expected, rtol=FEATURE_RTOL, atol=atol)
    valid_frames = np.arange(actual.shape[1]) < actual_lengths[:, np.newaxis]
    differences = (actual - expected)[valid_frames].astype(np.float64)
    if differences.size:
        assert np.sqrt(np.mean(differences**2)) < FEATURE_RMSE_ATOL
    np.testing.assert_array_equal(actual[~valid_frames], 0.0)
    np.testing.assert_array_equal(actual[actual_lengths < 2], 0.0)
    return actual, actual_lengths


def assert_valid_features_are_normalized(
    features: np.typing.NDArray[np.float32],
    feature_lengths: np.typing.NDArray[np.int32],
) -> None:
    """Check mean and sample standard deviation over valid nonsilent frames.

    Parameters
    ----------
    features : np.typing.NDArray[np.float32]
        Normalized features with shape (batch, time, mel_bins).
    feature_lengths : np.typing.NDArray[np.int32]
        Valid feature-frame counts for each utterance.
    """

    for utterance, feature_length in zip(features, feature_lengths, strict=True):
        if feature_length < 2:
            continue

        valid_features = utterance[:feature_length].astype(np.float64)
        np.testing.assert_allclose(
            valid_features.mean(axis=0),
            0.0,
            rtol=0.0,
            atol=1e-4,
        )
        standard_deviations = valid_features.std(axis=0, ddof=1)
        assert np.all((standard_deviations > 0.99) & (standard_deviations < 1.001))


def feature_input_specs(
    audio_shape: tuple[int, ...] = (1, 640),
    length_shape: tuple[int, ...] = (1,),
    window_shape: tuple[int, ...] = (FFT_LENGTH,),
    mel_shape: tuple[int, ...] = (MEL_FREQUENCIES, NUM_FEATURES),
    dtypes: tuple[trt.DataType, ...] | None = None,
) -> tuple[InputSpec, ...]:
    """Create one static four-input TensorRT plugin contract.

    Parameters
    ----------
    audio_shape : tuple[int, ...]
        Shape of the audio input.
    length_shape : tuple[int, ...]
        Shape of the valid-length input.
    window_shape : tuple[int, ...]
        Shape of the frontend window input.
    mel_shape : tuple[int, ...]
        Shape of the frequency-by-mel filterbank input.
    dtypes : tuple[trt.DataType, ...] or None
        Input dtypes in binding order; None selects this helper's default contract.

    Returns
    -------
    tuple[InputSpec, ...]
        Input-ordered dtype/shape pairs, without building or allocating an engine.
    """

    if dtypes is None:
        dtypes = (trt.float32, trt.int64, trt.float32, trt.float32)
    return tuple(
        zip(
            dtypes,
            (audio_shape, length_shape, window_shape, mel_shape),
            strict=True,
        )
    )


INVALID_CONTRACT_CASES = (
    pytest.param(feature_input_specs(audio_shape=(640,)), id="audio-rank"),
    pytest.param(feature_input_specs(length_shape=(1, 1)), id="length-rank"),
    pytest.param(feature_input_specs(window_shape=(1, FFT_LENGTH)), id="window-rank"),
    pytest.param(
        feature_input_specs(mel_shape=(MEL_FREQUENCIES,)),
        id="mel-rank",
    ),
    pytest.param(
        feature_input_specs(dtypes=(trt.float16, trt.int64, trt.float32, trt.float32)),
        id="audio-dtype",
    ),
    pytest.param(
        feature_input_specs(dtypes=(trt.float32, trt.int32, trt.float32, trt.float32)),
        id="length-dtype",
    ),
    pytest.param(
        feature_input_specs(dtypes=(trt.float32, trt.int64, trt.float16, trt.float32)),
        id="window-dtype",
    ),
    pytest.param(
        feature_input_specs(dtypes=(trt.float32, trt.int64, trt.float32, trt.float16)),
        id="mel-dtype",
    ),
    pytest.param(feature_input_specs(length_shape=(2,)), id="batch-mismatch"),
    pytest.param(
        feature_input_specs(
            window_shape=(1,),
            mel_shape=(1, NUM_FEATURES),
        ),
        id="window-too-short",
    ),
    pytest.param(
        feature_input_specs(
            window_shape=(FFT_LENGTH - 1,),
            mel_shape=(FFT_LENGTH // 2, NUM_FEATURES),
        ),
        id="window-odd",
    ),
    pytest.param(
        feature_input_specs(mel_shape=(MEL_FREQUENCIES - 1, NUM_FEATURES)),
        id="frequency-mismatch",
    ),
    pytest.param(
        feature_input_specs(
            window_shape=(MAX_FFT_LENGTH + 2,),
            mel_shape=(MAX_FREQUENCIES + 1, NUM_FEATURES),
        ),
        id="too-many-frequencies",
    ),
    pytest.param(feature_input_specs()[:3], id="missing-input"),
    pytest.param(
        feature_input_specs() + ((trt.float32, (1,)),),
        id="extra-input",
    ),
)
WORKSPACE_OVERFLOW_SPECS = feature_input_specs(
    audio_shape=(260, 640_200),
    length_shape=(260,),
)


def build_static_contract(
    creator: trt.IPluginCreatorV3One,
    input_specs: tuple[InputSpec, ...],
) -> tuple[bool, trt.IHostMemory | None]:
    """Build one static plugin contract and report whether its layer was added.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.

    Returns
    -------
    tuple[bool, trt.IHostMemory | None]
        Whether TensorRT added the layer, followed by serialized bytes or None on
        build rejection.
    """

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = [
        network.add_input(f"input_{index}", dtype, shape)
        for index, (dtype, shape) in enumerate(input_specs)
    ]
    assert all(tensor is not None for tensor in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    if layer is None:
        return False, None
    for index, name in enumerate(("features", "feature_lengths")):
        output = layer.get_output(index)
        output.name = name
        network.mark_output(output)
    config = builder.create_builder_config()
    # Keep the builder cap above the plugin's signed-32-bit workspace limit so
    # this test exercises the plugin's own overflow guard.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    return True, builder.build_serialized_network(network, config)


@pytest.mark.parametrize(
    ("samples", "length_values", "atol"),
    (
        pytest.param(640, (480,), FEATURE_ATOL, id="profile-min"),
        pytest.param(1920, (1600, 800), FEATURE_ATOL, id="intermediate"),
        pytest.param(4000, (3840, 1920, 320), FEATURE_ATOL, id="profile-opt"),
        pytest.param(4000, (0, 160, 320) + (3840,) * 253, 7e-3, id="profile-max"),
    ),
)
def test_parakeet_feature_plugin_matches_pytorch(
    feature_engine: FeatureEngine,
    samples: int,
    length_values: tuple[int, ...],
    atol: float,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array(length_values, dtype=np.int64)
    audio = make_audio(lengths, samples)
    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
        atol,
    )
    np.testing.assert_array_equal(actual_lengths, lengths // FRAME_SHIFT)
    assert_valid_features_are_normalized(actual, actual_lengths)


def test_parakeet_feature_plugin_honors_nondefault_serialized_frontend(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    extractor = make_extractor(
        frame_shift_ms=8,
        frame_length_ms=15,
        n_mels=40,
        preemph=0.25,
        low_freq=100,
        high_freq=7600,
    )
    extractor.log_eps = 1e-2
    extractor.eps = 0.1
    result = build_feature_engine(creator, extractor, ((1, 640), (2, 2048), (3, 4096)))
    assert result is not None
    _runtime, engine = result
    lengths = np.array((3968, 2176), dtype=np.int64)
    audio = make_audio(lengths, 4096, seed=11)
    assert extractor.n_fft == 256
    assert extractor.mel_filterbank.shape == (129, 40)
    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths, extractor=extractor),
        extractor,
        audio,
        lengths,
    )
    assert actual.shape == (2, 33, 40)
    np.testing.assert_array_equal(actual_lengths, (31, 17))


def test_parakeet_feature_plugin_ignores_trailing_padding_extent(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    valid_length = 1601
    lengths = np.array((valid_length,), dtype=np.int64)
    valid_audio = (
        np.random.default_rng(31).normal(0.0, 0.05, valid_length).astype(np.float32)
    )
    short_audio = np.full((1, 1920), 1e6, dtype=np.float32)
    long_audio = np.full((1, 4000), -1e6, dtype=np.float32)
    short_audio[0, :valid_length] = valid_audio
    long_audio[0, :valid_length] = valid_audio

    short_run = run_engine(engine, short_audio, lengths)
    short_features, short_lengths = assert_run_matches_pytorch(
        short_run, extractor, short_audio, lengths
    )
    long_run = run_engine(engine, long_audio, lengths, context=short_run.context)
    long_features, long_lengths = assert_run_matches_pytorch(
        long_run, extractor, long_audio, lengths
    )

    np.testing.assert_array_equal(short_lengths, (10,))
    np.testing.assert_array_equal(long_lengths, short_lengths)
    np.testing.assert_allclose(
        long_features[0, :10],
        short_features[0, :10],
        rtol=FEATURE_RTOL,
        atol=2e-3,
    )


def test_parakeet_feature_plugin_reuses_context_across_shape_changes(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        (640, (480,)),
        (4000, (3840, 1920, 320)),
        (4000, (3681,)),
        (1920, (1600, 800)),
        (1920, (1441, 960)),
        (640, (320,)),
    )
    for case_index, (samples, lengths) in enumerate(shape_cases):
        stream = streams[case_index % len(streams)]
        lengths_array = np.array(lengths, dtype=np.int64)
        audio = make_audio(lengths_array, samples, seed=case_index)
        run = run_engine(engine, audio, lengths_array, context, stream)
        assert run.context is context and run.stream is stream
        assert_run_matches_pytorch(run, extractor, audio, lengths_array)


def test_parakeet_feature_plugin_supports_concurrent_contexts(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    host_cases = []
    for audio_samples, length_values in (
        (640, (480,)),
        (4000, (3840, 1920, 320)),
    ):
        lengths = np.array(length_values, dtype=np.int64)
        host_cases.append((make_audio(lengths, audio_samples), lengths))

    runs = [run_engine(engine, audio, lengths) for audio, lengths in host_cases]
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr
    for run, (audio, lengths) in zip(runs, host_cases, strict=True):
        assert_run_matches_pytorch(run, extractor, audio, lengths)


def test_parakeet_feature_plugin_rejects_runtime_batch_mismatch(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, _ = feature_engine
    run = prepare_run(
        engine,
        np.zeros((2, 1920), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
    )
    with run.stream:
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert not executed
    assert bool(cp.isnan(run.features).all())
    assert bool((run.feature_lengths == INT32_SENTINEL).all())


def test_parakeet_feature_plugin_full_scale_pcm_is_finite(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((4000,), dtype=np.int64)
    audio = np.tile(np.array((-1.0, 1.0), dtype=np.float32), 2000).reshape(1, -1)

    assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
        atol=7e-3,
    )


@pytest.mark.parametrize("batch_size", (3, 256), ids=("small", "profile-maximum"))
def test_parakeet_feature_plugin_returns_zero_for_silence(
    feature_engine: FeatureEngine,
    batch_size: int,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.full(batch_size, 4000, dtype=np.int64)
    if batch_size == 3:
        lengths[:2] = (640, 1920)
    audio = np.zeros((batch_size, 4000), dtype=np.float32)

    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
    )

    np.testing.assert_array_equal(
        actual_lengths,
        (lengths // FRAME_SHIFT).astype(np.int32),
    )
    np.testing.assert_array_equal(actual, 0.0)


def test_parakeet_feature_plugin_ignores_nonfinite_padding(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((640, 1280, 1920), dtype=np.int64)
    audio = make_audio(lengths, 1920, seed=29)
    audio[0, 640:] = np.nan
    audio[1, 1280::2] = np.inf
    audio[1, 1281::2] = -np.inf

    assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
    )


def test_parakeet_feature_plugin_supports_cuda_graphs(
    feature_engine: FeatureEngine,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((1600, 800), dtype=np.int64)
    audio = make_audio(lengths, 1920, seed=1)
    run = run_engine(engine, audio, lengths)
    assert_run_matches_pytorch(run, extractor, audio, lengths)

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for seed, length_values in ((2, (1441, 960)), (3, (1760, 641))):
        replay_lengths = np.array(length_values, dtype=np.int64)
        replay_audio = make_audio(replay_lengths, 1920, seed=seed)
        with run.stream:
            run.audio.set(replay_audio, stream=run.stream)
            run.lengths.set(replay_lengths, stream=run.stream)
            run.features.fill(cp.nan)
            run.feature_lengths.fill(INT32_SENTINEL)
            graph.launch(run.stream)
        assert_run_matches_pytorch(
            run, extractor, replay_audio, replay_lengths, atol=3e-3
        )


@pytest.mark.parametrize(
    ("samples", "length_values", "expected_lengths", "seed"),
    (
        pytest.param(4000, (-1, 160, 10000), (0, 1, 25), 17, id="malformed"),
        pytest.param(640, (0, 1, 159), (0, 0, 0), 19, id="all-short"),
        pytest.param(
            799,
            (
                0,
                159,
                160,
                161,
                319,
                320,
                321,
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
            ),
            (0, 0, 1, 1, 1, 2, 2, 0, 4),
            23,
            id="hop-boundaries-and-int64-limits",
        ),
    ),
)
def test_parakeet_feature_plugin_clamps_lengths_and_zeroes_short_utterances(
    feature_engine: FeatureEngine,
    samples: int,
    length_values: tuple[int, ...],
    expected_lengths: tuple[int, ...],
    seed: int,
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array(length_values, dtype=np.int64)
    audio = (
        np.random.default_rng(seed)
        .normal(
            0.0,
            0.05,
            (len(lengths), samples),
        )
        .astype(np.float32)
    )
    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
    )
    np.testing.assert_array_equal(actual_lengths, expected_lengths)
    for utterance, length in zip(actual, actual_lengths, strict=True):
        if length >= 2:
            assert np.count_nonzero(utterance[:length]) > 0


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_parakeet_feature_plugin_rejects_invalid_contracts(
    plugin_creator: PluginCreatorFixture,
    input_specs: tuple[InputSpec, ...],
) -> None:
    _, creator = plugin_creator
    _, serialized_engine = build_static_contract(creator, input_specs)
    assert serialized_engine is None


@pytest.mark.parametrize(
    "input_specs",
    (
        pytest.param(feature_input_specs(), id="default"),
        pytest.param(
            feature_input_specs(
                window_shape=(2,),
                mel_shape=(2, NUM_FEATURES),
            ),
            id="minimum-even-fft",
        ),
        pytest.param(
            feature_input_specs(
                window_shape=(MAX_FFT_LENGTH,),
                mel_shape=(MAX_FREQUENCIES, NUM_FEATURES),
            ),
            id="maximum-shared-memory-fft",
        ),
    ),
)
def test_parakeet_feature_plugin_accepts_valid_static_contract(
    plugin_creator: PluginCreatorFixture,
    input_specs: tuple[InputSpec, ...],
) -> None:
    _, creator = plugin_creator
    _, serialized_engine = build_static_contract(creator, input_specs)
    assert serialized_engine is not None


def test_parakeet_feature_plugin_rejects_workspace_overflow(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    layer_added, serialized_engine = build_static_contract(
        creator,
        WORKSPACE_OVERFLOW_SPECS,
    )

    assert layer_added
    assert serialized_engine is None


@pytest.mark.parametrize(
    "length_batches",
    (
        pytest.param((2, 3, 256), id="min"),
        pytest.param((1, 2, 256), id="opt"),
        pytest.param((1, 3, 255), id="max"),
    ),
)
def test_parakeet_feature_plugin_rejects_invalid_profile_endpoints(
    plugin_creator: PluginCreatorFixture,
    length_batches: tuple[int, ...],
) -> None:
    _, creator = plugin_creator
    result = build_feature_engine(
        creator, make_extractor(), PROFILE_SHAPES, length_batches
    )
    assert result is None


def test_parakeet_feature_creator_exposes_complete_contract(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert (creator.name, creator.plugin_version, creator.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    expected_fields = {
        "frame_shift": (trt.PluginFieldType.INT32, 1),
        "preemph": (trt.PluginFieldType.FLOAT32, 1),
        "log_eps": (trt.PluginFieldType.FLOAT32, 1),
        "eps": (trt.PluginFieldType.FLOAT32, 1),
    }
    assert len(creator.field_names) == len(expected_fields)
    assert {
        field.name: (field.type, field.size) for field in creator.field_names
    } == expected_fields

    plugin = make_plugin(creator)
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None and build is not None and runtime is not None
    assert (core.plugin_name, core.plugin_version, core.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    assert build.num_outputs == 2


@pytest.mark.parametrize(
    ("overrides", "equivalent"),
    (
        pytest.param({}, True, id="identical"),
        pytest.param(
            {"frame_shift": np.array([128], dtype=np.int32)}, False, id="frame-shift"
        ),
        pytest.param(
            {"preemph": np.array([0.25], dtype=np.float32)}, False, id="preemph"
        ),
        pytest.param(
            {"log_eps": np.array([1e-2], dtype=np.float32)}, False, id="log-eps"
        ),
        pytest.param({"eps": np.array([0.1], dtype=np.float32)}, False, id="eps"),
    ),
)
def test_parakeet_feature_timing_cache_depends_on_frontend(
    plugin_creator: PluginCreatorFixture,
    overrides: dict[str, np.typing.NDArray],
    equivalent: bool,
) -> None:
    _, creator = plugin_creator
    plugins = (make_plugin(creator), make_plugin(creator, overrides))
    cache_ids = [
        plugin.get_capability_interface(trt.PluginCapabilityType.BUILD).timing_cache_id
        for plugin in plugins
    ]
    assert all(cache_ids)
    assert (cache_ids[0] == cache_ids[1]) is equivalent


@pytest.mark.parametrize("name", FIELD_NAMES)
@pytest.mark.parametrize("problem", ("missing", "wrong-type"))
def test_parakeet_feature_creator_requires_correctly_typed_fields(
    plugin_creator: PluginCreatorFixture,
    name: str,
    problem: str,
) -> None:
    _, creator = plugin_creator
    if problem == "missing":
        fields = [field for field in make_fields() if field.name != name]
    else:
        value = (
            np.array([float(FRAME_SHIFT)], dtype=np.float32)
            if name == "frame_shift"
            else np.array([1], dtype=np.int32)
        )
        fields = make_fields({name: value})
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


@pytest.mark.parametrize("name", ("frame_shift", "eps"), ids=("integer", "float"))
@pytest.mark.parametrize("count", (0, 2), ids=("empty", "multiple"))
def test_parakeet_feature_creator_rejects_non_scalar_field(
    plugin_creator: PluginCreatorFixture,
    name: str,
    count: int,
) -> None:
    _, creator = plugin_creator
    dtype = np.int32 if name == "frame_shift" else np.float32
    value = FRAME_SHIFT if name == "frame_shift" else NORMALIZATION_EPS
    fields = make_fields({name: np.full(count, value, dtype=dtype)})
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_parakeet_feature_creator_rejects_duplicate_field(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    fields = make_fields()
    fields.append(fields[0])
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_parakeet_feature_creator_accepts_reordered_and_unknown_fields(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    fields = make_fields({"implementation_metadata": np.array([1], dtype=np.int32)})
    fields.reverse()
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None


@pytest.mark.parametrize(
    ("frame_shift", "preemph", "positive_value"),
    (
        pytest.param(
            1,
            np.float32(0.0),
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            id="lower",
        ),
        pytest.param(
            np.iinfo(np.int32).max,
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.finfo(np.float32).max,
            id="upper",
        ),
    ),
)
def test_parakeet_feature_creator_accepts_valid_parameter_boundaries(
    plugin_creator: PluginCreatorFixture,
    frame_shift: int,
    preemph: np.float32,
    positive_value: np.float32,
) -> None:
    _, creator = plugin_creator
    make_plugin(
        creator,
        {
            "frame_shift": np.array([frame_shift], dtype=np.int32),
            "preemph": np.array([preemph], dtype=np.float32),
            "log_eps": np.array([positive_value], dtype=np.float32),
            "eps": np.array([positive_value], dtype=np.float32),
        },
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("frame_shift", 0),
        ("preemph", -0.1),
        ("preemph", 1.0),
        ("preemph", np.nan),
        ("preemph", np.inf),
        ("log_eps", -1.0),
        ("log_eps", 0.0),
        ("log_eps", np.nan),
        ("log_eps", np.inf),
        ("eps", -1.0),
        ("eps", 0.0),
        ("eps", np.nan),
        ("eps", np.inf),
    ),
)
def test_parakeet_feature_creator_rejects_invalid_parameter(
    plugin_creator: PluginCreatorFixture,
    name: str,
    value: int | float,
) -> None:
    _, creator = plugin_creator
    dtype = np.int32 if name == "frame_shift" else np.float32
    fields = make_fields({name: np.array([value], dtype=dtype)})
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None
