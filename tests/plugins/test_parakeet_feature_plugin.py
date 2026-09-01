#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet feature plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE
from fast_gpu_asr.export.model.parakeet.features import FeatureExtractor

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.cuda

PLUGIN_NAME = "parakeet_feature_extractor"
PLUGIN_VERSION = "1"
SAMPLE_RATE = 16000
FRAME_SHIFT = 160
NUM_FEATURES = 128
PROFILE_SHAPES = ((1, 640), (3, 4000), (256, 4000))
FFT_LENGTH = 512
MEL_FREQUENCIES = FFT_LENGTH // 2 + 1
FEATURE_RTOL = 3e-4
FEATURE_ATOL = 3e-2
FEATURE_RMSE_ATOL = 2e-3
FIELD_NAMES = ("frame_shift", "preemph", "log_eps", "eps")
INT32_SENTINEL = np.iinfo(np.int32).min


def make_extractor() -> FeatureExtractor:
    """Create the Parakeet frontend represented by the native plugin."""

    return FeatureExtractor(
        samp_freq=SAMPLE_RATE,
        frame_shift_ms=10,
        frame_length_ms=25,
        n_mels=NUM_FEATURES,
        preemph=0.97,
        low_freq=0,
        high_freq=8000,
    ).eval()


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register Zipformer before the Parakeet feature plugin."""

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
    overrides: dict[str, tuple[np.ndarray, trt.PluginFieldType]] | None = None,
) -> list[trt.PluginField]:
    """Create a complete feature-plugin field list with optional replacements."""

    values = {
        "frame_shift": (
            np.array([FRAME_SHIFT], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        "preemph": (np.array([0.97], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        "log_eps": (
            np.array([2**-24], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        "eps": (np.array([1e-5], dtype=np.float32), trt.PluginFieldType.FLOAT32),
    }
    if overrides is not None:
        values.update(overrides)
    return [
        trt.PluginField(name, value, field_type)
        for name, (value, field_type) in values.items()
    ]


def make_plugin(creator):
    """Create a valid Parakeet feature plugin."""

    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(make_fields()),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def add_feature_plugin_layer(
    network,
    creator,
    extractor: FeatureExtractor,
    audio,
    audio_lengths,
):
    """Add the production constant inputs and mark both plugin outputs."""

    window_layer = network.add_constant(
        extractor.window.shape,
        extractor.window.numpy(),
    )
    mel_layer = network.add_constant(
        extractor.mel_filterbank.shape,
        extractor.mel_filterbank.numpy(),
    )
    assert window_layer is not None and mel_layer is not None
    layer = network.add_plugin_v3(
        [audio, audio_lengths, window_layer.get_output(0), mel_layer.get_output(0)],
        [],
        make_plugin(creator),
    )
    assert layer is not None
    features = layer.get_output(0)
    feature_lengths = layer.get_output(1)
    features.name = "features"
    feature_lengths.name = "feature_lengths"
    network.mark_output(features)
    network.mark_output(feature_lengths)
    return layer


@pytest.fixture(scope="module")
def feature_engine(plugin_creator):
    """Build a dynamic TensorRT engine around the native feature plugin."""

    _, creator = plugin_creator
    extractor = make_extractor()
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
    profile.set_shape("audio", *PROFILE_SHAPES)
    profile.set_shape(
        "audio_lengths",
        (PROFILE_SHAPES[0][0],),
        (PROFILE_SHAPES[1][0],),
        (PROFILE_SHAPES[2][0],),
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
    return runtime, engine, extractor


def make_audio(
    lengths: np.typing.NDArray[np.int64], audio_samples: int, seed: int = 0
) -> np.typing.NDArray[np.float32]:
    """Create padded waveforms whose invalid tails must not affect features."""

    rng = np.random.default_rng(seed + audio_samples + int(lengths.sum()))
    audio = rng.normal(0.0, 0.05, (len(lengths), audio_samples)).astype(np.float32)
    for index, length in enumerate(lengths):
        audio[index, int(length) :] = 10.0
    return audio


@dataclass
class FeatureRun:
    """Device buffers and execution state retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    audio: cp.ndarray
    lengths: cp.ndarray
    features: cp.ndarray
    feature_lengths: cp.ndarray


def run_engine(
    engine,
    audio: np.typing.NDArray[np.float32],
    lengths: np.typing.NDArray[np.int64],
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
) -> FeatureRun:
    """Execute with sentinel outputs on one explicitly ordered CUDA stream."""

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert audio.dtype == np.float32
    assert lengths.dtype == np.int64
    assert audio.ndim == 2
    assert lengths.shape == (audio.shape[0],)
    assert context.set_input_shape("audio", audio.shape)
    assert context.set_input_shape("audio_lengths", lengths.shape)
    feature_shape = tuple(context.get_tensor_shape("features"))
    feature_length_shape = tuple(context.get_tensor_shape("feature_lengths"))
    assert feature_shape == (
        audio.shape[0],
        audio.shape[1] // FRAME_SHIFT + 1,
        NUM_FEATURES,
    )
    assert feature_length_shape == lengths.shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        audio_device = cp.asarray(audio)
        lengths_device = cp.asarray(lengths)
        features_device = cp.full(
            feature_shape,
            cp.nan,
            dtype=cp.float32,
        )
        feature_lengths_device = cp.full(
            feature_length_shape,
            INT32_SENTINEL,
            dtype=cp.int32,
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


def collect_outputs(run: FeatureRun) -> tuple[np.ndarray, np.ndarray]:
    """Synchronize one run and copy both outputs to NumPy."""

    run.stream.synchronize()
    return cp.asnumpy(run.features), cp.asnumpy(run.feature_lengths)


def assert_run_matches_pytorch(
    run: FeatureRun,
    extractor: FeatureExtractor,
    audio: np.typing.NDArray[np.float32],
    lengths: np.typing.NDArray[np.int64],
    *,
    atol: float = FEATURE_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare one native run with the independent eager implementation."""

    actual, actual_lengths = collect_outputs(run)
    with torch.inference_mode():
        expected, expected_lengths = extractor(
            torch.from_numpy(audio), torch.from_numpy(lengths)
        )

    assert actual.dtype == np.float32
    assert actual_lengths.dtype == np.int32
    np.testing.assert_array_equal(actual_lengths, expected_lengths.numpy())
    # TensorRT may select FAST_16BF for the mel GEMM. FP32 accumulation
    # preserves model accuracy, but reduced-precision inputs can create sparse
    # normalized-feature outliers relative to strict FP32 PyTorch output.
    expected_array = expected.numpy()
    np.testing.assert_allclose(
        actual,
        expected_array,
        rtol=FEATURE_RTOL,
        atol=atol,
    )
    valid_frames = np.arange(actual.shape[1]) < actual_lengths[:, np.newaxis]
    valid_differences = (actual - expected_array)[valid_frames]
    if valid_differences.size:
        assert np.sqrt(np.mean(valid_differences.astype(np.float64) ** 2)) < (
            FEATURE_RMSE_ATOL
        )
    return actual, actual_lengths


def feature_input_specs(
    *,
    audio_shape: tuple[int, ...] = (1, 640),
    length_shape: tuple[int, ...] = (1,),
    window_shape: tuple[int, ...] = (FFT_LENGTH,),
    mel_shape: tuple[int, ...] = (MEL_FREQUENCIES, NUM_FEATURES),
    dtypes: tuple[object, ...] | None = None,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Create one static four-input TensorRT plugin contract."""

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
            window_shape=(12_288,),
            mel_shape=(6_145, NUM_FEATURES),
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
    creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> tuple[bool, object | None]:
    """Build one static plugin contract and report whether its layer was added."""

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
    "audio_shape,lengths",
    (
        pytest.param((1, 640), (480,), id="profile-min"),
        pytest.param((2, 1920), (1600, 800), id="intermediate"),
        pytest.param((3, 4000), (3840, 1920, 320), id="profile-opt"),
    ),
)
def test_parakeet_feature_plugin_matches_pytorch(
    feature_engine, audio_shape: tuple[int, int], lengths: tuple[int, ...]
) -> None:
    """Compare dynamic native features and valid lengths with eager PyTorch."""

    _, engine, extractor = feature_engine
    lengths_array = np.array(lengths, dtype=np.int64)
    audio = make_audio(lengths_array, audio_shape[1])
    actual, actual_lengths = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths_array),
        extractor,
        audio,
        lengths_array,
    )
    for index, length in enumerate(actual_lengths):
        np.testing.assert_array_equal(actual[index, int(length) :], 0.0)


def test_parakeet_feature_plugin_matches_pytorch_at_large_batch(feature_engine) -> None:
    """Exercise adaptive normalization at the production profile maximum."""

    _, engine, extractor = feature_engine
    lengths = np.full(256, 3840, dtype=np.int64)
    audio = make_audio(lengths, 4000)
    assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
        atol=7e-3,
    )


def test_parakeet_feature_plugin_reuses_context_across_shape_changes(
    feature_engine,
) -> None:
    """Rebuild cuFFT plans across shapes and alternating CUDA streams."""

    _, engine, extractor = feature_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        ((1, 640), (480,)),
        ((3, 4000), (3840, 1920, 320)),
        ((1, 4000), (3681,)),
        ((2, 1920), (1600, 800)),
        ((2, 1920), (1441, 960)),
        ((1, 640), (320,)),
    )
    for case_index, (audio_shape, lengths) in enumerate(shape_cases):
        stream = streams[case_index % len(streams)]
        lengths_array = np.array(lengths, dtype=np.int64)
        audio = make_audio(lengths_array, audio_shape[1], seed=case_index)
        run = run_engine(
            engine,
            audio,
            lengths_array,
            context=context,
            stream=stream,
        )
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_pytorch(run, extractor, audio, lengths_array)


def test_parakeet_feature_plugin_supports_concurrent_contexts(
    feature_engine,
) -> None:
    """Keep per-context cuFFT and cuBLAS state independent across streams."""

    _, engine, extractor = feature_engine
    host_cases = []
    for audio_samples, length_values in (
        (640, (480,)),
        (4000, (3840, 1920, 320)),
    ):
        lengths = np.array(length_values, dtype=np.int64)
        host_cases.append((make_audio(lengths, audio_samples), lengths))

    runs = tuple(
        run_engine(
            engine,
            audio,
            lengths,
            context=engine.create_execution_context(),
            stream=cp.cuda.Stream(non_blocking=True),
        )
        for audio, lengths in host_cases
    )
    assert runs[0].context is not runs[1].context
    assert runs[0].stream is not runs[1].stream
    for run, (audio, lengths) in zip(runs, host_cases, strict=True):
        assert_run_matches_pytorch(run, extractor, audio, lengths)


def test_parakeet_feature_plugin_rejects_runtime_batch_mismatch(
    feature_engine,
) -> None:
    """Reject concrete shapes that disagree inside a valid dynamic profile."""

    _, engine, _ = feature_engine
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("audio", (2, 1920))
    assert context.set_input_shape("audio_lengths", (1,))
    feature_shape = tuple(context.get_tensor_shape("features"))
    feature_length_shape = tuple(context.get_tensor_shape("feature_lengths"))
    assert feature_shape == (2, 13, NUM_FEATURES)
    assert feature_length_shape == (1,)
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        audio = cp.zeros((2, 1920), dtype=cp.float32)
        lengths = cp.zeros((1,), dtype=cp.int64)
        features = cp.full(feature_shape, cp.nan, dtype=cp.float32)
        feature_lengths = cp.full(
            feature_length_shape,
            INT32_SENTINEL,
            dtype=cp.int32,
        )
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


def test_parakeet_feature_plugin_full_scale_pcm_is_finite(feature_engine) -> None:
    """Keep full-scale high-frequency PCM finite through the mel projection."""

    _, engine, extractor = feature_engine
    lengths = np.array((4000,), dtype=np.int64)
    audio = np.tile(np.array((-1.0, 1.0), dtype=np.float32), 2000).reshape(1, -1)

    actual, _ = assert_run_matches_pytorch(
        run_engine(engine, audio, lengths),
        extractor,
        audio,
        lengths,
        atol=7e-3,
    )
    assert np.isfinite(actual).all()


def test_parakeet_feature_plugin_supports_cuda_graphs(feature_engine) -> None:
    """Capture and replay feature extraction on a non-default CUDA stream."""

    _, engine, extractor = feature_engine
    lengths = np.array((1600, 800), dtype=np.int64)
    audio = make_audio(lengths, 1920, seed=1)
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    run = run_engine(
        engine,
        audio,
        lengths,
        context=context,
        stream=stream,
    )
    run.stream.synchronize()

    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)

    for seed, replay_lengths in (
        (2, np.array((1441, 960), dtype=np.int64)),
        (3, np.array((1760, 641), dtype=np.int64)),
    ):
        replay_audio = make_audio(replay_lengths, 1920, seed=seed)
        with run.stream:
            run.audio.set(replay_audio, stream=run.stream)
            run.lengths.set(replay_lengths, stream=run.stream)
            run.features.fill(cp.nan)
            run.feature_lengths.fill(INT32_SENTINEL)
            graph.launch(run.stream)
        assert_run_matches_pytorch(
            run,
            extractor,
            replay_audio,
            replay_lengths,
            atol=3e-3,
        )


def test_parakeet_feature_plugin_clamps_malformed_lengths(feature_engine) -> None:
    """Keep device-side invalid lengths bounded and deterministic."""

    _, engine, _ = feature_engine
    lengths = np.array((-1, 160, 10000), dtype=np.int64)
    audio = np.zeros((3, 4000), dtype=np.float32)
    actual, actual_lengths = collect_outputs(run_engine(engine, audio, lengths))

    np.testing.assert_array_equal(actual_lengths, np.array((0, 1, 25), np.int32))
    np.testing.assert_array_equal(actual, 0.0)


def test_parakeet_feature_plugin_clamps_hop_boundary_lengths(feature_engine) -> None:
    """Clamp extreme lengths and preserve exact 160-sample hop boundaries."""

    _, engine, _ = feature_engine
    lengths = np.array(
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
        dtype=np.int64,
    )
    audio = np.zeros((len(lengths), 799), dtype=np.float32)

    actual, actual_lengths = collect_outputs(run_engine(engine, audio, lengths))

    np.testing.assert_array_equal(
        actual_lengths, np.array((0, 0, 1, 1, 1, 2, 2, 0, 4), dtype=np.int32)
    )
    assert np.isfinite(actual).all()
    for index, length in enumerate(actual_lengths):
        np.testing.assert_array_equal(actual[index, int(length) :], 0.0)


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_parakeet_feature_plugin_rejects_invalid_contracts(
    plugin_creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid input counts, ranks, dtypes, and shape relationships."""

    _, creator = plugin_creator
    _, serialized_engine = build_static_contract(creator, input_specs)
    assert serialized_engine is None


def test_parakeet_feature_plugin_rejects_workspace_overflow(
    plugin_creator,
) -> None:
    """Reject a layer whose required workspace exceeds signed 32-bit offsets."""

    _, creator = plugin_creator
    layer_added, serialized_engine = build_static_contract(
        creator,
        WORKSPACE_OVERFLOW_SPECS,
    )

    assert layer_added
    assert serialized_engine is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "opt", "max"))
def test_parakeet_feature_plugin_rejects_invalid_profile_endpoints(
    plugin_creator,
    invalid_endpoint: str,
) -> None:
    """Validate cross-input shapes at every dynamic profile endpoint."""

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

    length_batches = {
        "min": (2, 3, 256),
        "opt": (1, 2, 256),
        "max": (1, 3, 255),
    }[invalid_endpoint]
    profile = builder.create_optimization_profile()
    profile.set_shape("audio", *PROFILE_SHAPES)
    profile.set_shape(
        "audio_lengths",
        *((batch_size,) for batch_size in length_batches),
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


@pytest.mark.parametrize("missing_name", FIELD_NAMES)
def test_parakeet_feature_creator_requires_every_field(
    plugin_creator,
    missing_name: str,
) -> None:
    """Reject plugin creation when one required attribute is absent."""

    _, creator = plugin_creator
    fields = [field for field in make_fields() if field.name != missing_name]
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_parakeet_feature_creator_rejects_duplicate_field(plugin_creator) -> None:
    """Reject ambiguous duplicate feature attributes."""

    _, creator = plugin_creator
    fields = make_fields()
    fields.append(fields[0])
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


@pytest.mark.parametrize(
    "name,value,field_type",
    (
        pytest.param(
            "frame_shift",
            np.array([float(FRAME_SHIFT)], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
            id="integer-as-float",
        ),
        pytest.param(
            "preemph",
            np.array([1], dtype=np.int32),
            trt.PluginFieldType.INT32,
            id="float-as-integer",
        ),
    ),
)
def test_parakeet_feature_creator_rejects_wrong_field_type(
    plugin_creator,
    name: str,
    value: np.ndarray,
    field_type,
) -> None:
    """Reject a field whose declared TensorRT type does not match its value."""

    _, creator = plugin_creator
    fields = make_fields({name: (value, field_type)})
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


@pytest.mark.parametrize(
    "values",
    (
        pytest.param(np.array([], dtype=np.int32), id="empty"),
        pytest.param(
            np.array([FRAME_SHIFT, FRAME_SHIFT], dtype=np.int32), id="multiple"
        ),
    ),
)
def test_parakeet_feature_creator_rejects_non_scalar_field(
    plugin_creator,
    values: np.ndarray,
) -> None:
    """Reject required fields containing anything other than one value."""

    _, creator = plugin_creator
    fields = make_fields(
        {
            "frame_shift": (
                values,
                trt.PluginFieldType.INT32,
            )
        }
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_parakeet_feature_creator_accepts_reordered_and_unknown_fields(
    plugin_creator,
) -> None:
    """Parse required fields by name and ignore unrelated metadata."""

    _, creator = plugin_creator
    fields = make_fields(
        {
            "implementation_metadata": (
                np.array([1], dtype=np.int32),
                trt.PluginFieldType.INT32,
            )
        }
    )
    fields.reverse()
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None


@pytest.mark.parametrize(
    "preemph",
    (
        pytest.param(np.float32(0.0), id="minimum-preemphasis"),
        pytest.param(
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            id="below-maximum-preemphasis",
        ),
    ),
)
def test_parakeet_feature_creator_accepts_valid_parameter_boundaries(
    plugin_creator,
    preemph: np.float32,
) -> None:
    """Accept the finite parameter values immediately inside each boundary."""

    _, creator = plugin_creator
    smallest_positive = np.nextafter(np.float32(0.0), np.float32(1.0))
    fields = make_fields(
        {
            "frame_shift": (
                np.array([1], dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            "preemph": (
                np.array([preemph], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            "log_eps": (
                np.array([smallest_positive], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            "eps": (
                np.array([smallest_positive], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
        }
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None


@pytest.mark.parametrize(
    "name,value,field_type",
    (
        ("frame_shift", 0, trt.PluginFieldType.INT32),
        ("preemph", -0.1, trt.PluginFieldType.FLOAT32),
        ("preemph", 1.0, trt.PluginFieldType.FLOAT32),
        ("preemph", np.nan, trt.PluginFieldType.FLOAT32),
        ("preemph", np.inf, trt.PluginFieldType.FLOAT32),
        ("log_eps", -1.0, trt.PluginFieldType.FLOAT32),
        ("log_eps", 0.0, trt.PluginFieldType.FLOAT32),
        ("log_eps", np.nan, trt.PluginFieldType.FLOAT32),
        ("log_eps", np.inf, trt.PluginFieldType.FLOAT32),
        ("eps", -1.0, trt.PluginFieldType.FLOAT32),
        ("eps", 0.0, trt.PluginFieldType.FLOAT32),
        ("eps", np.nan, trt.PluginFieldType.FLOAT32),
        ("eps", np.inf, trt.PluginFieldType.FLOAT32),
    ),
)
def test_parakeet_feature_creator_rejects_invalid_parameter(
    plugin_creator, name: str, value: int | float, field_type
) -> None:
    """Reject feature parameters outside the supported numerical domain."""

    dtype = np.int32 if field_type == trt.PluginFieldType.INT32 else np.float32
    fields = make_fields({name: (np.array([value], dtype=dtype), field_type)})
    _, creator = plugin_creator
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None
