#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer attention-value plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.cuda


PLUGIN_NAME = "zipformer_attention_value"
PLUGIN_VERSION = "1"
SHAPE_CASES = (
    pytest.param(1, 1, id="minimum"),
    pytest.param(1, 3, id="shorter-than-warp"),
    pytest.param(2, 17, id="profile-optimum"),
    pytest.param(3, 65, id="profile-maximum"),
)


@dataclass(frozen=True)
class DTypeCase:
    """TensorRT, CuPy, and comparison settings for one numeric dtype."""

    name: str
    trt_dtype: object
    cupy_dtype: object
    tolerance: float


DTYPE_CASES = (
    DTypeCase("fp32", trt.float32, cp.float32, 1e-2),
    DTypeCase("fp16", trt.float16, cp.float16, 1e-2),
    pytest.param(
        DTypeCase("bf16", trt.bfloat16, cp.dtype("bfloat16"), 3e-2),
        marks=pytest.mark.sm80,
    ),
)


@dataclass(frozen=True)
class LayerCase:
    """One attention/value head layout embedded in the shared test engine."""

    name: str
    attention_heads: int
    value_heads: int
    channels: int

    @property
    def attention_name(self) -> str:
        return f"{self.name}_attention"

    @property
    def value_name(self) -> str:
        return f"{self.name}_value"

    @property
    def output_name(self) -> str:
        return f"{self.name}_output"


LAYER_CASES = (
    LayerCase("self", attention_heads=4, value_heads=4, channels=48),
    LayerCase("nonlinear", attention_heads=4, value_heads=1, channels=64),
    # A non-12 head dimension forces the general multi-head cuBLAS path.
    LayerCase("general_multihead", attention_heads=2, value_heads=2, channels=10),
)


@dataclass(frozen=True)
class AttentionEngine:
    """One deserialized engine and the contract used to build it."""

    runtime: object
    engine: object
    dtype_case: DTypeCase
    layers: tuple[LayerCase, ...]


@dataclass(frozen=True)
class HostLayerInputs:
    """Host inputs for one plugin layer."""

    layer: LayerCase
    attention: np.ndarray
    value: np.ndarray


@dataclass
class DeviceLayerRun:
    """Device buffers retained after launching one plugin layer."""

    layer: LayerCase
    attention: cp.ndarray
    value: cp.ndarray
    output: cp.ndarray


@dataclass
class EngineRun:
    """Execution state and device buffers retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    layers: tuple[DeviceLayerRun, ...]


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register the current attention-value plugin source."""

    library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_attention_value_plugin.cu",
        "initFastGpuAsrZipformerAttentionValuePlugin",
        ("cublas", "cudart"),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return library, creator


def make_plugin(creator, value_heads: int):
    """Create an attention-value plugin with the requested value-head count."""

    value_heads_field = np.array([value_heads], dtype=np.int32)
    field = trt.PluginField(
        "num_heads",
        value_heads_field,
        trt.PluginFieldType.INT32,
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def add_attention_value_layer(network, creator, dtype, layer_case: LayerCase) -> None:
    """Add one dynamic attention-value plugin layer to a TensorRT network."""

    attention = network.add_input(
        layer_case.attention_name,
        dtype,
        (-1, layer_case.attention_heads, -1, -1),
    )
    value = network.add_input(
        layer_case.value_name,
        dtype,
        (-1, -1, layer_case.channels),
    )
    assert attention is not None and value is not None

    layer = network.add_plugin_v3(
        [attention, value],
        [],
        make_plugin(creator, layer_case.value_heads),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = layer_case.output_name
    network.mark_output(output)


def build_attention_engine(
    creator,
    dtype_case: DTypeCase,
    layers: tuple[LayerCase, ...] = LAYER_CASES,
) -> AttentionEngine:
    """Build a dynamic engine covering production and guaranteed fallback layouts."""

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    for layer_case in layers:
        add_attention_value_layer(network, creator, dtype_case.trt_dtype, layer_case)

    profile = builder.create_optimization_profile()
    for layer_case in layers:
        profile.set_shape(
            layer_case.attention_name,
            (1, layer_case.attention_heads, 1, 1),
            (2, layer_case.attention_heads, 17, 17),
            (3, layer_case.attention_heads, 65, 65),
        )
        profile.set_shape(
            layer_case.value_name,
            (1, 1, layer_case.channels),
            (2, 17, layer_case.channels),
            (3, 65, layer_case.channels),
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
        name: (mode, dtype_case.trt_dtype)
        for layer_case in layers
        for name, mode in (
            (layer_case.attention_name, trt.TensorIOMode.INPUT),
            (layer_case.value_name, trt.TensorIOMode.INPUT),
            (layer_case.output_name, trt.TensorIOMode.OUTPUT),
        )
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return AttentionEngine(runtime, engine, dtype_case, layers)


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=lambda case: case.name)
def attention_engine(request, plugin_creator) -> AttentionEngine:
    """Build one shared-layout engine for every supported numeric dtype."""

    _, creator = plugin_creator
    return build_attention_engine(creator, request.param)


def make_attention(
    batch_size: int,
    sequence_length: int,
    attention_heads: int,
    seed: int,
) -> np.ndarray:
    """Create distinct normalized attention matrices for every batch and head."""

    rng = np.random.default_rng(seed)
    logits = rng.normal(
        size=(batch_size, attention_heads, sequence_length, sequence_length)
    ).astype(np.float32)
    logits -= logits.max(axis=3, keepdims=True)
    attention = np.exp(logits)
    return attention / attention.sum(axis=3, keepdims=True)


def make_host_inputs(
    layers: tuple[LayerCase, ...],
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> tuple[HostLayerInputs, ...]:
    """Create deterministic, layout-specific inputs for every engine layer."""

    rng = np.random.default_rng(seed)
    return tuple(
        HostLayerInputs(
            layer_case,
            make_attention(
                batch_size,
                sequence_length,
                layer_case.attention_heads,
                seed + 101 * (index + 1),
            ),
            rng.normal(
                0.0,
                0.5,
                (batch_size, sequence_length, layer_case.channels),
            ).astype(np.float32),
        )
        for index, layer_case in enumerate(layers)
    )


def reference_attention_value(
    attention: np.ndarray,
    value: np.ndarray,
    layer_case: LayerCase,
) -> np.ndarray:
    """Evaluate the public NHTT-by-NTC attention/value contract in NumPy."""

    batch_size, sequence_length, channels = value.shape
    assert attention.shape == (
        batch_size,
        layer_case.attention_heads,
        sequence_length,
        sequence_length,
    )
    assert channels == layer_case.channels
    assert channels % layer_case.value_heads == 0
    head_dim = channels // layer_case.value_heads
    values_by_head = value.reshape(
        batch_size,
        sequence_length,
        layer_case.value_heads,
        head_dim,
    )
    return np.einsum(
        "nhqk,nkhd->nqhd",
        attention[:, : layer_case.value_heads],
        values_by_head,
        optimize=True,
    ).reshape(value.shape)


def prepare_engine_run(
    attention_engine: AttentionEngine,
    host_inputs: tuple[HostLayerInputs, ...],
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Resolve shapes and bind sentinel-backed buffers without executing."""

    assert tuple(inputs.layer for inputs in host_inputs) == attention_engine.layers
    if context is None:
        context = attention_engine.engine.create_execution_context()
    assert context is not None
    for inputs in host_inputs:
        assert context.set_input_shape(
            inputs.layer.attention_name, inputs.attention.shape
        )
        assert context.set_input_shape(inputs.layer.value_name, inputs.value.shape)
    for inputs in host_inputs:
        assert tuple(context.get_tensor_shape(inputs.layer.output_name)) == (
            inputs.value.shape
        )

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    device_runs = []
    with stream:
        for inputs in host_inputs:
            attention = cp.asarray(
                inputs.attention,
                dtype=attention_engine.dtype_case.cupy_dtype,
            )
            value = cp.asarray(
                inputs.value,
                dtype=attention_engine.dtype_case.cupy_dtype,
            )
            output = cp.full(
                inputs.value.shape,
                cp.nan,
                dtype=attention_engine.dtype_case.cupy_dtype,
            )
            for name, buffer in (
                (inputs.layer.attention_name, attention),
                (inputs.layer.value_name, value),
                (inputs.layer.output_name, output),
            ):
                assert context.set_tensor_address(name, buffer.data.ptr)
            device_runs.append(DeviceLayerRun(inputs.layer, attention, value, output))
    return EngineRun(context, stream, tuple(device_runs))


def run_engine(
    attention_engine: AttentionEngine,
    host_inputs: tuple[HostLayerInputs, ...],
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
    synchronize: bool = True,
) -> EngineRun:
    """Execute all plugin layers and retain quantized inputs and outputs."""

    run = prepare_engine_run(
        attention_engine,
        host_inputs,
        context=context,
        stream=stream,
    )
    with run.stream:
        assert run.context.execute_async_v3(run.stream.ptr)
    if synchronize:
        run.stream.synchronize()
    return run


def assert_run_matches_reference(
    run: EngineRun,
    dtype_case: DTypeCase,
) -> None:
    """Compare every layer with the independent quantization-aware oracle."""

    run.stream.synchronize()
    for layer_run in run.layers:
        actual = cp.asnumpy(layer_run.output).astype(np.float32)
        attention = cp.asnumpy(layer_run.attention).astype(np.float32)
        value = cp.asnumpy(layer_run.value).astype(np.float32)
        assert actual.shape == value.shape
        assert np.isfinite(actual).all()
        expected = reference_attention_value(attention, value, layer_run.layer)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=dtype_case.tolerance,
            atol=dtype_case.tolerance,
        )


@pytest.mark.parametrize(("batch_size", "sequence_length"), SHAPE_CASES)
def test_attention_value_plugin_matches_reference(
    attention_engine: AttentionEngine,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Compare self, nonlinear, and fallback layouts across all dtypes."""

    host_inputs = make_host_inputs(
        attention_engine.layers,
        batch_size,
        sequence_length,
        seed=1000 + batch_size * 100 + sequence_length,
    )
    assert_run_matches_reference(
        run_engine(attention_engine, host_inputs),
        attention_engine.dtype_case,
    )


def test_attention_value_plugin_supports_cuda_graph_replay(
    attention_engine: AttentionEngine,
) -> None:
    """Replay changed inputs accurately for every configured layout."""

    batch_size, sequence_length = 2, 17
    host_inputs = make_host_inputs(
        attention_engine.layers,
        batch_size,
        sequence_length,
        seed=2026,
    )
    run = run_engine(attention_engine, host_inputs)

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for replay in range(2):
        replay_inputs = make_host_inputs(
            attention_engine.layers,
            batch_size,
            sequence_length,
            seed=3000 + replay,
        )
        with run.stream:
            for layer_run, inputs in zip(run.layers, replay_inputs, strict=True):
                cp.copyto(
                    layer_run.attention,
                    cp.asarray(
                        inputs.attention,
                        dtype=attention_engine.dtype_case.cupy_dtype,
                    ),
                )
                cp.copyto(
                    layer_run.value,
                    cp.asarray(
                        inputs.value,
                        dtype=attention_engine.dtype_case.cupy_dtype,
                    ),
                )
                layer_run.output.fill(cp.nan)
            graph.launch(run.stream)

        assert_run_matches_reference(run, attention_engine.dtype_case)


def test_attention_value_plugin_reuses_context_across_shapes_and_streams(
    attention_engine: AttentionEngine,
) -> None:
    """Reuse one context across profile boundaries and alternating streams."""

    context = attention_engine.engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    assert streams[0].ptr != streams[1].ptr

    for index, (batch_size, sequence_length) in enumerate(((1, 1), (3, 65), (1, 3))):
        run = run_engine(
            attention_engine,
            make_host_inputs(
                attention_engine.layers,
                batch_size,
                sequence_length,
                seed=4000 + index,
            ),
            context=context,
            stream=streams[index % len(streams)],
        )
        assert run.context is context
        assert run.stream is streams[index % len(streams)]
        assert_run_matches_reference(run, attention_engine.dtype_case)


def test_attention_value_plugin_supports_concurrent_contexts(
    attention_engine: AttentionEngine,
) -> None:
    """Keep independent cuBLAS state for overlapping execution contexts."""

    first = run_engine(
        attention_engine,
        make_host_inputs(attention_engine.layers, 1, 17, seed=5001),
        synchronize=False,
    )
    second = run_engine(
        attention_engine,
        make_host_inputs(attention_engine.layers, 3, 65, seed=5002),
        synchronize=False,
    )

    assert first.context is not second.context
    assert first.stream.ptr != second.stream.ptr
    assert_run_matches_reference(first, attention_engine.dtype_case)
    assert_run_matches_reference(second, attention_engine.dtype_case)


@pytest.mark.parametrize(
    "invalid_relationship",
    ("batch", "sequence", "non-square-attention"),
)
def test_attention_value_plugin_rejects_runtime_shape_mismatch(
    attention_engine: AttentionEngine,
    invalid_relationship: str,
) -> None:
    """Reject invalid concrete shapes that each fit the dynamic profile."""

    host_inputs = list(make_host_inputs(attention_engine.layers, 2, 17, seed=6000))
    invalid_inputs = host_inputs[0]
    if invalid_relationship == "batch":
        attention = invalid_inputs.attention
        value = invalid_inputs.value[:1]
    elif invalid_relationship == "sequence":
        attention = invalid_inputs.attention
        value = invalid_inputs.value[:, :16]
    else:
        assert invalid_relationship == "non-square-attention"
        attention = invalid_inputs.attention[..., :16]
        value = invalid_inputs.value
    host_inputs[0] = HostLayerInputs(
        invalid_inputs.layer,
        attention,
        value,
    )
    run = prepare_engine_run(attention_engine, tuple(host_inputs))
    with run.stream:
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    assert not executed
    assert bool(cp.isnan(run.layers[0].output).all())


def attention_value_input_specs(
    *,
    attention_shape: tuple[int, ...] = (2, 4, 7, 7),
    value_shape: tuple[int, ...] = (2, 7, 48),
    attention_dtype=trt.float16,
    value_dtype=trt.float16,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Create one static two-input TensorRT plugin contract."""

    return (
        (attention_dtype, attention_shape),
        (value_dtype, value_shape),
    )


INVALID_CONTRACT_CASES = (
    pytest.param(
        4,
        attention_value_input_specs(attention_shape=(2, 4, 7)),
        id="attention-rank",
    ),
    pytest.param(
        4,
        attention_value_input_specs(value_shape=(2, 7)),
        id="value-rank",
    ),
    pytest.param(
        4,
        attention_value_input_specs(attention_shape=(1, 4, 7, 7)),
        id="batch-mismatch",
    ),
    pytest.param(
        4,
        attention_value_input_specs(attention_shape=(2, 4, 7, 8)),
        id="non-square-attention",
    ),
    pytest.param(
        4,
        attention_value_input_specs(value_shape=(2, 8, 48)),
        id="sequence-mismatch",
    ),
    pytest.param(
        4,
        attention_value_input_specs(attention_shape=(2, 3, 7, 7)),
        id="too-few-attention-heads",
    ),
    pytest.param(
        4,
        attention_value_input_specs(attention_shape=(2, 5, 7, 7)),
        id="extra-multihead-attention-head",
    ),
    pytest.param(
        4,
        attention_value_input_specs(value_shape=(2, 7, 50)),
        id="channels-not-divisible-by-heads",
    ),
    pytest.param(
        4,
        attention_value_input_specs(
            attention_dtype=trt.int32,
            value_dtype=trt.int32,
        ),
        id="unsupported-dtype",
    ),
    pytest.param(
        4,
        attention_value_input_specs(value_dtype=trt.float32),
        id="mixed-dtypes",
    ),
    pytest.param(
        4,
        attention_value_input_specs()[:1],
        id="missing-input",
    ),
    pytest.param(
        4,
        attention_value_input_specs() + ((trt.float16, (1,)),),
        id="extra-input",
    ),
)


def build_static_contract(
    creator,
    value_heads: int,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> object | None:
    """Attempt to build one static attention-value plugin contract."""

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
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator, value_heads))
    if layer is None:
        return None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    return builder.build_serialized_network(network, config)


@pytest.mark.parametrize(("value_heads", "input_specs"), INVALID_CONTRACT_CASES)
def test_attention_value_plugin_rejects_invalid_contracts(
    plugin_creator,
    value_heads: int,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid counts, ranks, dtypes, and shape relationships."""

    _, creator = plugin_creator
    assert build_static_contract(creator, value_heads, input_specs) is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "opt", "max"))
@pytest.mark.parametrize("invalid_relationship", ("batch", "sequence"))
def test_attention_value_plugin_rejects_invalid_profile_endpoints(
    plugin_creator,
    invalid_endpoint: str,
    invalid_relationship: str,
) -> None:
    """Validate cross-input shapes at every dynamic profile endpoint."""

    _, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    attention = network.add_input("attention", trt.float16, (-1, 4, -1, -1))
    value = network.add_input("value", trt.float16, (-1, -1, 48))
    assert attention is not None and value is not None
    layer = network.add_plugin_v3(
        [attention, value],
        [],
        make_plugin(creator, value_heads=4),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    attention_shapes = (
        (1, 4, 1, 1),
        (2, 4, 17, 17),
        (3, 4, 65, 65),
    )
    value_batches = (
        {
            "min": (2, 2, 3),
            "opt": (1, 1, 3),
            "max": (1, 2, 2),
        }[invalid_endpoint]
        if invalid_relationship == "batch"
        else (1, 2, 3)
    )
    value_lengths = (
        {
            "min": (2, 17, 65),
            "opt": (1, 16, 65),
            "max": (1, 17, 64),
        }[invalid_endpoint]
        if invalid_relationship == "sequence"
        else (1, 17, 65)
    )
    value_shapes = tuple(
        (batch_size, sequence_length, 48)
        for batch_size, sequence_length in zip(
            value_batches,
            value_lengths,
            strict=True,
        )
    )
    profile = builder.create_optimization_profile()
    profile.set_shape("attention", *attention_shapes)
    profile.set_shape("value", *value_shapes)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


@pytest.mark.parametrize(
    ("values", "field_type"),
    (
        pytest.param(
            np.array([1.0], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
            id="wrong-type",
        ),
        pytest.param(
            np.array([], dtype=np.int32),
            trt.PluginFieldType.INT32,
            id="empty",
        ),
        pytest.param(
            np.array([1, 4], dtype=np.int32),
            trt.PluginFieldType.INT32,
            id="multiple",
        ),
    ),
)
def test_attention_value_creator_rejects_malformed_num_heads(
    plugin_creator,
    values: np.ndarray,
    field_type,
) -> None:
    """Reject a required field with the wrong type or cardinality."""

    _, creator = plugin_creator
    field = trt.PluginField("num_heads", values, field_type)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


@pytest.mark.parametrize("value_heads", (0, -1))
def test_attention_value_creator_rejects_nonpositive_num_heads(
    plugin_creator,
    value_heads: int,
) -> None:
    """Reject nonpositive value-head counts during plugin creation."""

    _, creator = plugin_creator
    value = np.array([value_heads], dtype=np.int32)
    field = trt.PluginField("num_heads", value, trt.PluginFieldType.INT32)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_attention_value_creator_requires_num_heads(plugin_creator) -> None:
    """Reject plugin creation when the required value-head field is absent."""

    _, creator = plugin_creator
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_attention_value_creator_rejects_duplicate_num_heads(plugin_creator) -> None:
    """Reject ambiguous duplicate value-head fields."""

    _, creator = plugin_creator
    values = (np.array([1], dtype=np.int32), np.array([4], dtype=np.int32))
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField("num_heads", value, trt.PluginFieldType.INT32)
            for value in values
        ]
    )
    plugin = creator.create_plugin(PLUGIN_NAME, fields, trt.TensorRTPhase.BUILD)
    assert plugin is None


def test_attention_value_creator_accepts_reordered_and_unknown_fields(
    plugin_creator,
) -> None:
    """Find the required field by name while ignoring parser metadata."""

    _, creator = plugin_creator
    metadata = np.array([17], dtype=np.int32)
    value_heads = np.array([2], dtype=np.int32)
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(
                "implementation_metadata",
                metadata,
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "num_heads",
                value_heads,
                trt.PluginFieldType.INT32,
            ),
        ]
    )
    plugin = creator.create_plugin(PLUGIN_NAME, fields, trt.TensorRTPhase.BUILD)
    assert plugin is not None
