#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer feature plugin."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE, ZERO_LOG
from fast_gpu_asr.export.model.zipformer.features import FeatureExtractor

pytestmark = pytest.mark.cuda

PLUGIN_NAME = "zipformer_feature_extractor"
PLUGIN_VERSION = "1"
SAMPLE_RATE = 16000
FRAME_LENGTH = 400
FRAME_SHIFT = 160
LEFT_PADDING = 120
RIGHT_PADDING = 200
MIN_FRAMES = 9
NUM_FEATURES = 80
PREEMPH = 0.97
NEXT_FRAME_BOUNDARY = (MIN_FRAMES + 1) * FRAME_SHIFT - FRAME_SHIFT // 2
MIN_AUDIO_SAMPLES = (MIN_FRAMES - 1) * FRAME_SHIFT + FRAME_LENGTH - LEFT_PADDING
FEATURE_RTOL = 2e-4
FEATURE_ATOL = 7e-3
FEATURE_RMSE_ATOL = 2e-3
FULL_SCALE_FEATURE_ATOL = 1.1e-2
PROFILE_SHAPES = ((1, 1800), (2, 3400), (256, 5000))
FFT_LENGTH = 1 << (FRAME_LENGTH - 1).bit_length()
MEL_FREQUENCIES = FFT_LENGTH // 2 + 1
PORTABLE_SHARED_MEMORY_BYTES = 48 << 10
WARPS_PER_BLOCK = 256 // 32
MAX_FRAME_LENGTH = (
    PORTABLE_SHARED_MEMORY_BYTES - WARPS_PER_BLOCK * np.dtype(np.float32).itemsize
) // np.dtype(np.float32).itemsize
MAX_FREQUENCIES = PORTABLE_SHARED_MEMORY_BYTES // np.dtype(np.complex64).itemsize
FIELD_NAMES = (
    "frame_length",
    "frame_shift",
    "left_padding",
    "min_frames",
    "preemph",
    "zero_log",
)
INTEGER_FIELD_NAMES = FIELD_NAMES[:4]
INT32_SENTINEL = np.iinfo(np.int32).min

type InputSpec = tuple[trt.DataType, tuple[int, ...]]


def make_extractor(
    frame_shift_ms: int = 10,
    frame_length_ms: int = 25,
    n_mels: int = NUM_FEATURES,
    preemph: float = PREEMPH,
    low_freq: int = 20,
    high_freq: int = 7600,
    min_frames: int = MIN_FRAMES,
) -> FeatureExtractor:
    """Create the Zipformer frontend represented by the native plugin.

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
    min_frames : int
        Minimum reported valid frame count for the Zipformer frontend.

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
        min_frames=min_frames,
    ).eval()


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register the current Zipformer feature plugin source.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for isolated, module-scoped compiled libraries.

    Returns
    -------
    tuple
        Library handle(s) and the registered creator, retained for dependent
        engines.
    """

    library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_feature_plugin.cu",
        "initFastGpuAsrZipformerFeaturePlugin",
        ("cublas", "cudart", "cufft"),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return library, creator


def make_fields(extractor=None, **overrides) -> list[trt.PluginField]:
    """Encode frontend scalars as TensorRT fields, with optional replacements.

    Parameters
    ----------
    extractor : FeatureExtractor or None
        Frontend supplying constants and parameters; None uses the default test
        frontend.
    **overrides : int or float
        Keyword replacements for serialized frontend scalar fields.

    Returns
    -------
    list[trt.PluginField]
        Typed fields ready for a PluginFieldCollection.
    """

    values = {
        "frame_length": FRAME_LENGTH,
        "frame_shift": FRAME_SHIFT,
        "left_padding": LEFT_PADDING,
        "min_frames": MIN_FRAMES,
        "preemph": PREEMPH,
        "zero_log": ZERO_LOG,
    }
    if extractor is not None:
        values = {name: getattr(extractor, name) for name in values}
    values.update(overrides)
    fields = []
    for name, value in values.items():
        if name in INTEGER_FIELD_NAMES:
            dtype, field_type = np.int32, trt.PluginFieldType.INT32
        else:
            dtype, field_type = np.float32, trt.PluginFieldType.FLOAT32
        fields.append(trt.PluginField(name, np.array([value], dtype=dtype), field_type))
    return fields


def make_plugin(creator, fields=None):
    """Create a plugin, leaving acceptance or rejection to the caller.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    fields : list[trt.PluginField] or None
        Explicit fields; None uses the default frontend parameters.

    Returns
    -------
    trt.IPluginV3 or None
        Created plugin, or None when the creator rejects the supplied fields.
    """

    return creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(make_fields() if fields is None else fields),
        trt.TensorRTPhase.BUILD,
    )


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


def add_feature_plugin_layer(
    network: trt.INetworkDefinition,
    creator: trt.IPluginCreatorV3One,
    extractor: FeatureExtractor,
    audio: trt.ITensor,
    audio_lengths: trt.ITensor,
) -> trt.IPluginV3Layer:
    """Add one extractor's constant inputs and mark both plugin outputs.

    Parameters
    ----------
    network : trt.INetworkDefinition
        Strongly typed network receiving constants and marked plugin outputs.
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    audio : trt.ITensor
        FP32 network input of shape (batch, audio_samples).
    audio_lengths : trt.ITensor
        INT64 network input containing valid sample counts.

    Returns
    -------
    trt.IPluginV3Layer
        Added plugin layer with named features and feature_lengths marked as
        outputs.
    """

    window_layer = network.add_constant(
        extractor.window.shape, extractor.window.numpy()
    )
    mel_layer = network.add_constant(
        extractor.mel_filterbank.shape, extractor.mel_filterbank.numpy()
    )
    assert window_layer is not None and mel_layer is not None
    plugin = make_plugin(creator, make_fields(extractor))
    assert plugin is not None
    layer = network.add_plugin_v3(
        [audio, audio_lengths, window_layer.get_output(0), mel_layer.get_output(0)],
        [],
        plugin,
    )
    assert layer is not None
    for index, name in enumerate(("features", "feature_lengths")):
        output = layer.get_output(index)
        output.name = name
        network.mark_output(output)
    return layer


def build_feature_engine(
    creator: trt.IPluginCreatorV3One,
    extractor: FeatureExtractor,
    profile_shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[trt.Runtime, trt.ICudaEngine]:
    """Build and deserialize a dynamic engine for one frontend configuration.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    profile_shapes : tuple[tuple[int, int], ...]
        Minimum, optimum, and maximum (batch, audio_samples) input shapes.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine]
        Deserialized engine with its owning runtime.
    """

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    audio = network.add_input("audio", trt.float32, (-1, -1))
    audio_lengths = network.add_input("audio_lengths", trt.int64, (-1,))
    assert audio is not None and audio_lengths is not None
    add_feature_plugin_layer(network, creator, extractor, audio, audio_lengths)

    profile = builder.create_optimization_profile()
    set_profile_shape(profile, "audio", *profile_shapes)
    set_profile_shape(
        profile, "audio_lengths", *((shape[0],) for shape in profile_shapes)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 3
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        "audio": (trt.TensorIOMode.INPUT, trt.float32),
        "audio_lengths": (trt.TensorIOMode.INPUT, trt.int64),
        "features": (trt.TensorIOMode.OUTPUT, trt.float32),
        "feature_lengths": (trt.TensorIOMode.OUTPUT, trt.int32),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return runtime, engine


@pytest.fixture(scope="module")
def feature_engine(plugin_creator):
    """Build a dynamic TensorRT engine around the native feature plugin.

    Parameters
    ----------
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine, FeatureExtractor]
        Owning runtime, reusable dynamic engine, and matching eager frontend.
    """

    _, creator = plugin_creator
    extractor = make_extractor()
    runtime, engine = build_feature_engine(creator, extractor, PROFILE_SHAPES)
    return runtime, engine, extractor


def make_padded_audio(
    lengths: np.typing.NDArray[np.int64],
    audio_samples: int,
    right_padding: int = RIGHT_PADDING,
) -> np.typing.NDArray[np.float32]:
    """Create waveforms with the reflected right context expected at runtime.

    Parameters
    ----------
    lengths : np.typing.NDArray[np.int64]
        INT64 valid sample counts, one per waveform.
    audio_samples : int
        Physical sample count per waveform, including padding.
    right_padding : int
        Reflected context samples appended to each waveform before zero padding.

    Returns
    -------
    np.typing.NDArray[np.float32]
        FP32 audio with reflected right context followed by zero padding.
    """

    assert np.all(lengths > 0)
    assert right_padding >= 0
    assert np.all(lengths + right_padding <= audio_samples)
    rng = np.random.default_rng(1000 + audio_samples + int(lengths.sum()))
    audio = np.zeros((len(lengths), audio_samples), dtype=np.float32)
    for index, length_value in enumerate(lengths):
        length = int(length_value)
        waveform = rng.normal(0.0, 0.05, length).astype(np.float32)
        audio[index, :length] = waveform
        reflected = min(length, right_padding)
        audio[index, length : length + reflected] = waveform[::-1][:reflected]
        if reflected < right_padding:
            audio[index, length + reflected : length + right_padding] = waveform[0]
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


def run_engine(
    engine: trt.ICudaEngine,
    extractor: FeatureExtractor,
    audio: np.typing.NDArray[np.float32],
    lengths: np.typing.NDArray[np.int64],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
) -> FeatureRun:
    """Execute with sentinel outputs on one explicitly ordered CUDA stream.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized engine whose runtime must remain alive during execution.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    audio : np.typing.NDArray
        FP32 padded audio with shape (batch, audio_samples).
    lengths : np.typing.NDArray[np.int64]
        INT64 valid sample counts, one per waveform.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.

    Returns
    -------
    FeatureRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert audio.dtype == np.float32
    assert lengths.dtype == np.int64
    assert audio.ndim == 2
    assert lengths.shape == (audio.shape[0],)
    assert context.set_input_shape("audio", audio.shape)
    assert context.set_input_shape("audio_lengths", lengths.shape)
    expected_frames = (
        audio.shape[1] + extractor.left_padding - extractor.frame_length
    ) // extractor.frame_shift + 1
    feature_shape = tuple(context.get_tensor_shape("features"))
    feature_length_shape = tuple(context.get_tensor_shape("feature_lengths"))
    assert feature_shape == (audio.shape[0], expected_frames, extractor.n_mels)
    assert feature_length_shape == lengths.shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        audio_device = cp.asarray(audio)
        lengths_device = cp.asarray(lengths)
        features_device = cp.full(feature_shape, cp.nan, dtype=cp.float32)
        feature_lengths_device = cp.full(
            feature_length_shape, INT32_SENTINEL, dtype=cp.int32
        )
        assert context.set_tensor_address("audio", audio_device.data.ptr)
        assert context.set_tensor_address("audio_lengths", lengths_device.data.ptr)
        assert context.set_tensor_address("features", features_device.data.ptr)
        assert context.set_tensor_address(
            "feature_lengths", feature_lengths_device.data.ptr
        )
        assert context.execute_async_v3(stream.ptr)
    return FeatureRun(
        context,
        stream,
        audio_device,
        lengths_device,
        features_device,
        feature_lengths_device,
    )


def assert_run_matches_pytorch(
    run: FeatureRun,
    extractor: FeatureExtractor,
    audio: np.typing.NDArray[np.float32],
    lengths: np.typing.NDArray[np.int64],
    atol: float = FEATURE_ATOL,
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Compare one native run with the independent eager implementation.

    Parameters
    ----------
    run : FeatureRun
        Bound device buffers and the context/stream that own their pending work.
    extractor : FeatureExtractor
        Eager frontend providing window, mel filterbank, and serialized parameters.
    audio : np.typing.NDArray
        FP32 padded audio with shape (batch, audio_samples).
    lengths : np.typing.NDArray[np.int64]
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
    actual, actual_lengths = cp.asnumpy(run.features), cp.asnumpy(run.feature_lengths)
    np.testing.assert_array_equal(cp.asnumpy(run.audio), audio)
    np.testing.assert_array_equal(cp.asnumpy(run.lengths), lengths)
    with torch.inference_mode():
        expected, expected_lengths = extractor(
            torch.from_numpy(audio), torch.from_numpy(lengths)
        )

    np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())
    # TensorRT may select FAST_16BF for the mel GEMM. It retains FP32
    # accumulation but introduces a small reduced-input-precision drift.
    expected_array = expected.numpy()
    np.testing.assert_allclose(actual, expected_array, rtol=FEATURE_RTOL, atol=atol)
    valid_frames = np.arange(actual.shape[1]) < actual_lengths[:, np.newaxis]
    differences = (actual - expected_array)[valid_frames].astype(np.float64)
    assert np.sqrt(np.mean(differences**2)) < FEATURE_RMSE_ATOL
    np.testing.assert_allclose(
        actual[~valid_frames], extractor.zero_log, rtol=0, atol=5e-5
    )
    return actual, actual_lengths


def feature_input_specs(
    audio_shape: tuple[int, ...] = (1, 1800),
    length_shape: tuple[int, ...] = (1,),
    window_shape: tuple[int, ...] = (FRAME_LENGTH,),
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
        zip(dtypes, (audio_shape, length_shape, window_shape, mel_shape), strict=True)
    )


INVALID_CONTRACT_CASES = (
    pytest.param(feature_input_specs(audio_shape=(1800,)), id="audio-rank"),
    pytest.param(feature_input_specs(length_shape=(1, 1)), id="length-rank"),
    pytest.param(feature_input_specs(window_shape=(1, FRAME_LENGTH)), id="window-rank"),
    pytest.param(feature_input_specs(mel_shape=(MEL_FREQUENCIES,)), id="mel-rank"),
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
        feature_input_specs(window_shape=(FRAME_LENGTH - 1,)), id="window-length"
    ),
    pytest.param(
        feature_input_specs(audio_shape=(1, LEFT_PADDING - 1)),
        id="too-short-for-left-context",
    ),
    pytest.param(
        feature_input_specs(audio_shape=(1, MIN_AUDIO_SAMPLES - 1)),
        id="below-minimum-frame-boundary",
    ),
    pytest.param(
        feature_input_specs(mel_shape=(1, NUM_FEATURES)), id="too-few-frequencies"
    ),
    pytest.param(
        feature_input_specs(mel_shape=(FRAME_LENGTH // 2, NUM_FEATURES)),
        id="fft-shorter-than-window",
    ),
    pytest.param(
        feature_input_specs(mel_shape=(MAX_FREQUENCIES + 1, NUM_FEATURES)),
        id="too-many-frequencies",
    ),
    pytest.param(feature_input_specs()[:3], id="missing-input"),
    pytest.param(feature_input_specs() + ((trt.float32, (1,)),), id="extra-input"),
)


def build_static_contract(
    creator: trt.IPluginCreatorV3One,
    input_specs: tuple[InputSpec, ...],
    plugin_overrides: dict[str, int | float] | None = None,
) -> tuple[bool, trt.IHostMemory | None]:
    """Build one static plugin contract and report whether its layer was added.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.
    plugin_overrides : dict[str, int | float] or None
        Serialized frontend parameter replacements for this contract.

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
    plugin = make_plugin(creator, make_fields(**(plugin_overrides or {})))
    assert plugin is not None
    layer = network.add_plugin_v3(inputs, [], plugin)
    if layer is None:
        return False, None
    for index, name in enumerate(("features", "feature_lengths")):
        output = layer.get_output(index)
        output.name = name
        network.mark_output(output)
    config = builder.create_builder_config()
    # Keep the builder cap above the plugin's signed-32-bit workspace limit so
    # that the overflow case cannot pass because of an unrelated config limit.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    return True, builder.build_serialized_network(network, config)


@pytest.mark.parametrize(
    "audio_samples,lengths",
    (
        pytest.param(1800, (1600,), id="profile-min"),
        pytest.param(3400, (3200, 1723), id="profile-opt"),
        pytest.param(5000, (4800, 3001, 800), id="long-shape"),
    ),
)
def test_feature_plugin_matches_pytorch(feature_engine, audio_samples, lengths) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array(lengths, dtype=np.int64)
    audio = make_padded_audio(lengths, audio_samples)
    assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths), extractor, audio, lengths
    )


def test_feature_plugin_matches_pytorch_at_maximum_batch(feature_engine) -> None:
    _, engine, extractor = feature_engine
    lengths = np.full(PROFILE_SHAPES[2][0], 4800, dtype=np.int64)
    lengths[:4] = (1, NEXT_FRAME_BOUNDARY - 1, NEXT_FRAME_BOUNDARY, 3001)
    audio = make_padded_audio(lengths, PROFILE_SHAPES[2][1])

    _, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths), extractor, audio, lengths
    )

    np.testing.assert_array_equal(
        actual_lengths[:4], (MIN_FRAMES, MIN_FRAMES, MIN_FRAMES + 1, 19)
    )


def test_feature_plugin_preserves_nondefault_serialized_parameters(
    plugin_creator,
) -> None:
    _, creator = plugin_creator
    extractor = make_extractor(
        frame_shift_ms=8,
        frame_length_ms=16,
        n_mels=17,
        preemph=0.5,
        low_freq=80,
        high_freq=7000,
        min_frames=11,
    )
    extractor.zero_log = -17.25
    profile_shapes = ((1, 1600), (2, 1800), (3, 2200))
    _runtime, engine = build_feature_engine(creator, extractor, profile_shapes)
    lengths = np.array((1500, 1200), dtype=np.int64)
    audio = make_padded_audio(
        lengths, profile_shapes[1][1], right_padding=extractor.frame_length // 2
    )

    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths), extractor, audio, lengths
    )

    assert actual.shape == (2, 13, 17)
    np.testing.assert_array_equal(actual_lengths, (12, 11))


def test_feature_plugin_ignores_trailing_padding_extent(feature_engine) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((PROFILE_SHAPES[0][1] - RIGHT_PADDING,), dtype=np.int64)
    compact_audio = make_padded_audio(lengths, PROFILE_SHAPES[0][1])
    context = engine.create_execution_context()
    assert context is not None

    compact, compact_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, compact_audio, lengths, context=context),
        extractor,
        compact_audio,
        lengths,
    )

    extended_audio = np.empty((1, PROFILE_SHAPES[2][1]), dtype=np.float32)
    extended_audio[:, : compact_audio.shape[1]] = compact_audio
    extended_audio[:, compact_audio.shape[1] :] = np.resize(
        np.array((-1e4, 1e4), dtype=np.float32),
        extended_audio.shape[1] - compact_audio.shape[1],
    )
    extended, extended_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, extended_audio, lengths, context=context),
        extractor,
        extended_audio,
        lengths,
    )

    np.testing.assert_array_equal(extended_lengths, compact_lengths)
    np.testing.assert_allclose(
        extended[:, : compact.shape[1]], compact, rtol=FEATURE_RTOL, atol=FEATURE_ATOL
    )


def test_feature_plugin_reuses_context_across_shape_changes(feature_engine) -> None:
    _, engine, extractor = feature_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)

    shape_cases = (
        (1800, (1600,)),
        (1800, (1600, 1200, 800)),
        (5000, (4800,)),
        (1880, (1680, 1000)),
        (5000, (4800, 3001, 800)),
        (1800, (1521,)),
    )
    for case_index, (audio_samples, lengths) in enumerate(shape_cases):
        stream = streams[case_index % len(streams)]
        lengths_array = np.array(lengths, dtype=np.int64)
        audio = make_padded_audio(lengths_array, audio_samples)
        run = run_engine(
            engine, extractor, audio, lengths_array, context=context, stream=stream
        )
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_pytorch(run, extractor, audio, lengths_array)


def test_feature_plugin_supports_concurrent_contexts(feature_engine) -> None:
    _, engine, extractor = feature_engine
    host_cases = []
    for audio_samples, length_values in ((1800, (1600,)), (5000, (4800, 3001, 800))):
        lengths = np.array(length_values, dtype=np.int64)
        host_cases.append((make_padded_audio(lengths, audio_samples), lengths))

    runs = tuple(
        run_engine(engine, extractor, audio, lengths) for audio, lengths in host_cases
    )
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr

    for run, (audio, lengths) in zip(runs, host_cases, strict=True):
        assert_run_matches_pytorch(run, extractor, audio, lengths)


def test_feature_plugin_rejects_runtime_batch_mismatch(feature_engine) -> None:
    _, engine, _ = feature_engine
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("audio", (2, 3400))
    assert context.set_input_shape("audio_lengths", (1,))
    feature_shape = tuple(context.get_tensor_shape("features"))
    feature_length_shape = tuple(context.get_tensor_shape("feature_lengths"))
    assert feature_shape == (2, 20, NUM_FEATURES)
    assert feature_length_shape == (1,)
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        audio = cp.zeros((2, 3400), dtype=cp.float32)
        lengths = cp.zeros((1,), dtype=cp.int64)
        features = cp.full(feature_shape, cp.nan, dtype=cp.float32)
        feature_lengths = cp.full(feature_length_shape, INT32_SENTINEL, dtype=cp.int32)
        for name, value in (
            ("audio", audio),
            ("audio_lengths", lengths),
            ("features", features),
            ("feature_lengths", feature_lengths),
        ):
            assert context.set_tensor_address(name, value.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    assert bool(cp.isnan(features).all())
    assert bool((feature_lengths == INT32_SENTINEL).all())


def test_feature_plugin_full_scale_pcm_is_finite(feature_engine) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((4000,), dtype=np.int64)
    waveform = np.tile(np.array((-1.0, 1.0), dtype=np.float32), 2000)
    audio = np.concatenate((waveform, waveform[-RIGHT_PADDING:][::-1])).reshape(1, -1)

    actual, _ = assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths),
        extractor,
        audio,
        lengths,
        atol=FULL_SCALE_FEATURE_ATOL,
    )
    assert np.isfinite(actual).all()


def test_feature_plugin_supports_cuda_graphs(feature_engine) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((3200, 1723), dtype=np.int64)
    audio = make_padded_audio(lengths, 3400)
    run = run_engine(engine, extractor, audio, lengths)
    run.stream.synchronize()

    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)
    for replay_lengths in (
        np.array((3001, 1200), dtype=np.int64),
        np.array((2800, 1601), dtype=np.int64),
    ):
        replay_audio = make_padded_audio(replay_lengths, 3400)
        with run.stream:
            run.audio.set(replay_audio, stream=run.stream)
            run.lengths.set(replay_lengths, stream=run.stream)
            run.features.fill(cp.nan)
            run.feature_lengths.fill(INT32_SENTINEL)
            graph.launch(run.stream)

        assert_run_matches_pytorch(run, extractor, replay_audio, replay_lengths)


@pytest.mark.parametrize(
    "declared_samples,expected_frames",
    (
        pytest.param(np.iinfo(np.int64).min, MIN_FRAMES, id="int64-min"),
        pytest.param(0, MIN_FRAMES, id="zero"),
        pytest.param(NEXT_FRAME_BOUNDARY - 1, MIN_FRAMES, id="below-rounding-boundary"),
        pytest.param(NEXT_FRAME_BOUNDARY, MIN_FRAMES + 1, id="at-rounding-boundary"),
        pytest.param(1800, MIN_FRAMES + 1, id="physical-length"),
        pytest.param(10_000, MIN_FRAMES + 1, id="above-physical-length"),
        pytest.param(np.iinfo(np.int64).max, MIN_FRAMES + 1, id="int64-max"),
    ),
)
def test_feature_plugin_clamps_and_rounds_lengths(
    feature_engine, declared_samples: int, expected_frames: int
) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((declared_samples,), dtype=np.int64)
    audio = np.linspace(-0.1, 0.1, 1800, dtype=np.float32).reshape(1, -1)
    _, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths), extractor, audio, lengths
    )

    np.testing.assert_array_equal(actual_lengths, (expected_frames,))


def test_feature_plugin_distinguishes_energy_floor_from_padding(feature_engine) -> None:
    _, engine, extractor = feature_engine
    lengths = np.array((NEXT_FRAME_BOUNDARY - 1,), dtype=np.int64)
    audio = np.zeros((1, PROFILE_SHAPES[0][1]), dtype=np.float32)

    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, extractor, audio, lengths), extractor, audio, lengths
    )

    np.testing.assert_array_equal(actual_lengths, (MIN_FRAMES,))
    energy_floor = np.log(np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        actual[0, :MIN_FRAMES], energy_floor, rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_feature_plugin_rejects_invalid_contracts(
    plugin_creator, input_specs: tuple[InputSpec, ...]
) -> None:
    _, creator = plugin_creator
    _, serialized_engine = build_static_contract(creator, input_specs)
    assert serialized_engine is None


@pytest.mark.parametrize(
    "input_specs,plugin_overrides",
    (
        pytest.param(
            feature_input_specs(audio_shape=(1, MIN_AUDIO_SAMPLES)),
            None,
            id="minimum-frame-boundary",
        ),
        pytest.param(feature_input_specs(), None, id="default-contract"),
        pytest.param(
            feature_input_specs(
                audio_shape=(1, 2), window_shape=(2,), mel_shape=(2, NUM_FEATURES)
            ),
            {"frame_length": 2, "frame_shift": 1, "left_padding": 0, "min_frames": 1},
            id="minimum-kernel-shapes",
        ),
        pytest.param(
            feature_input_specs(
                audio_shape=(1, MAX_FRAME_LENGTH),
                window_shape=(MAX_FRAME_LENGTH,),
                mel_shape=(MAX_FREQUENCIES, NUM_FEATURES),
            ),
            {"frame_length": MAX_FRAME_LENGTH, "min_frames": 1},
            id="maximum-shared-memory-kernels",
        ),
    ),
)
def test_feature_plugin_accepts_valid_static_contract(
    plugin_creator,
    input_specs: tuple[InputSpec, ...],
    plugin_overrides: dict[str, int | float] | None,
) -> None:
    _, creator = plugin_creator
    layer_added, serialized_engine = build_static_contract(
        creator, input_specs, plugin_overrides
    )

    assert layer_added
    assert serialized_engine is not None


def test_feature_plugin_rejects_workspace_overflow(plugin_creator) -> None:
    _, creator = plugin_creator
    layer_added, serialized_engine = build_static_contract(
        creator, feature_input_specs(audio_shape=(260, 640_200), length_shape=(260,))
    )

    assert layer_added
    assert serialized_engine is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "opt", "max"))
def test_feature_plugin_rejects_invalid_profile_endpoints(
    plugin_creator, invalid_endpoint: str
) -> None:
    _, creator = plugin_creator
    extractor = make_extractor()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    audio = network.add_input("audio", trt.float32, (-1, -1))
    audio_lengths = network.add_input("audio_lengths", trt.int64, (-1,))
    assert audio is not None and audio_lengths is not None
    add_feature_plugin_layer(network, creator, extractor, audio, audio_lengths)

    min_batch, opt_batch, max_batch = (shape[0] for shape in PROFILE_SHAPES)
    length_batches = {
        "min": (min_batch + 1, opt_batch, max_batch),
        "opt": (min_batch, opt_batch + 1, max_batch),
        "max": (min_batch, opt_batch, max_batch - 1),
    }[invalid_endpoint]
    profile = builder.create_optimization_profile()
    set_profile_shape(profile, "audio", *PROFILE_SHAPES)
    set_profile_shape(
        profile, "audio_lengths", *((batch_size,) for batch_size in length_batches)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


@pytest.mark.parametrize("missing_name", FIELD_NAMES)
def test_feature_creator_requires_every_field(plugin_creator, missing_name) -> None:
    _, creator = plugin_creator
    fields = [field for field in make_fields() if field.name != missing_name]
    assert make_plugin(creator, fields) is None


def test_feature_creator_exposes_complete_contract(plugin_creator) -> None:
    _, creator = plugin_creator
    fields = tuple(creator.field_names)

    assert creator.name == PLUGIN_NAME
    assert creator.plugin_version == PLUGIN_VERSION
    assert creator.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert len(fields) == len(FIELD_NAMES)
    assert {field.name: (field.type, field.size) for field in fields} == {
        **{name: (trt.PluginFieldType.INT32, 1) for name in INTEGER_FIELD_NAMES},
        "preemph": (trt.PluginFieldType.FLOAT32, 1),
        "zero_log": (trt.PluginFieldType.FLOAT32, 1),
    }

    plugin = make_plugin(creator)
    assert plugin is not None
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None
    assert core.plugin_name == PLUGIN_NAME
    assert core.plugin_version == PLUGIN_VERSION
    assert core.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert build is not None
    assert build.num_outputs == 2
    assert runtime is not None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="equivalent"),
        pytest.param({"frame_length": 512}, id="frame-length"),
        pytest.param({"frame_shift": 128}, id="frame-shift"),
        pytest.param({"left_padding": 80}, id="left-padding"),
        pytest.param({"min_frames": 10}, id="minimum-frames"),
        pytest.param({"preemph": 0.5}, id="preemphasis"),
        pytest.param({"zero_log": -17.25}, id="zero-log"),
    ],
)
def test_feature_timing_cache_identity(plugin_creator, overrides) -> None:
    _, creator = plugin_creator
    default_plugin = make_plugin(creator)
    alternate_plugin = make_plugin(creator, make_fields(**overrides))
    assert default_plugin is not None and alternate_plugin is not None
    default = default_plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    alternate = alternate_plugin.get_capability_interface(
        trt.PluginCapabilityType.BUILD
    )

    assert default is not None and alternate is not None
    assert default.timing_cache_id and alternate.timing_cache_id
    assert (default.timing_cache_id != alternate.timing_cache_id) == bool(overrides)


def test_feature_creator_rejects_duplicate_field(plugin_creator) -> None:
    _, creator = plugin_creator
    fields = make_fields()
    fields.append(fields[0])
    assert make_plugin(creator, fields) is None


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_feature_creator_rejects_wrong_field_type(plugin_creator, name) -> None:
    _, creator = plugin_creator
    if name in INTEGER_FIELD_NAMES:
        dtype, field_type = np.float32, trt.PluginFieldType.FLOAT32
    else:
        dtype, field_type = np.int32, trt.PluginFieldType.INT32
    fields = make_fields()
    index = FIELD_NAMES.index(name)
    # Preserve valid parameter bytes so rejection depends only on the type tag.
    value = fields[index].data.view(dtype).copy()
    fields[index] = trt.PluginField(name, value, field_type)
    assert make_plugin(creator, fields) is None


@pytest.mark.parametrize("name", ["frame_length", "zero_log"], ids=["integer", "float"])
@pytest.mark.parametrize("count", [0, 2], ids=["empty", "multiple"])
def test_feature_creator_rejects_non_scalar_field(plugin_creator, name, count) -> None:
    _, creator = plugin_creator
    fields = make_fields()
    index = FIELD_NAMES.index(name)
    field = fields[index]
    fields[index] = trt.PluginField(name, np.repeat(field.data, count), field.type)
    assert make_plugin(creator, fields) is None


def test_feature_creator_accepts_reordered_and_unknown_fields(plugin_creator) -> None:
    _, creator = plugin_creator
    fields = make_fields()
    fields.append(
        trt.PluginField(
            "implementation_metadata",
            np.array([1], dtype=np.int32),
            trt.PluginFieldType.INT32,
        )
    )
    assert make_plugin(creator, fields[::-1]) is not None


@pytest.mark.parametrize(
    "frame_shift,left_padding,preemph,zero_log",
    [
        pytest.param(1, 0, 0.0, -np.finfo(np.float32).max, id="lower"),
        pytest.param(
            2,
            1,
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.finfo(np.float32).max,
            id="upper",
        ),
    ],
)
def test_feature_creator_accepts_valid_parameter_boundaries(
    plugin_creator, frame_shift, left_padding, preemph, zero_log
) -> None:
    _, creator = plugin_creator
    fields = make_fields(
        frame_length=2,
        frame_shift=frame_shift,
        left_padding=left_padding,
        min_frames=1,
        preemph=preemph,
        zero_log=zero_log,
    )
    assert make_plugin(creator, fields) is not None


@pytest.mark.parametrize(
    "name,value",
    [
        pytest.param("frame_length", 1, id="frame-length-too-small"),
        pytest.param("frame_length", MAX_FRAME_LENGTH + 1, id="frame-length-too-large"),
        pytest.param("frame_shift", 0, id="frame-shift-too-small"),
        pytest.param("frame_shift", FRAME_LENGTH + 1, id="frame-shift-exceeds-frame"),
        pytest.param("left_padding", -1, id="left-padding-negative"),
        pytest.param("left_padding", FRAME_LENGTH, id="left-padding-reaches-frame"),
        pytest.param("min_frames", 0, id="min-frames-too-small"),
        pytest.param("preemph", -0.1, id="preemph-negative"),
        pytest.param("preemph", 1.0, id="preemph-one"),
        pytest.param("preemph", np.nan, id="preemph-nan"),
        pytest.param("preemph", np.inf, id="preemph-infinite"),
        pytest.param("zero_log", np.nan, id="zero-log-nan"),
        pytest.param("zero_log", np.inf, id="zero-log-infinite"),
        pytest.param("zero_log", -np.inf, id="zero-log-negative-infinite"),
    ],
)
def test_feature_creator_rejects_invalid_parameter(plugin_creator, name, value) -> None:
    _, creator = plugin_creator
    assert make_plugin(creator, make_fields(**{name: value})) is None
