#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer attention-value plugin."""

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
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
)

pytestmark = pytest.mark.cuda

PLUGIN_NAME = ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME
PLUGIN_VERSION = "1"
DTYPE_CASES = ("float32", "float16", pytest.param("bfloat16", marks=pytest.mark.sm80))


@dataclass(frozen=True)
class LayerCase:
    """One attention/value head layout embedded in the shared test engine."""

    name: str
    attention_heads: int
    value_heads: int
    channels: int

    @property
    def attention_name(self) -> str:
        """Return this layer's attention-weight binding name.

        Returns
        -------
        str
            Unique attention-weight input binding name for this layer case.
        """

        return f"{self.name}_attention"

    @property
    def value_name(self) -> str:
        """Return this layer's value binding name.

        Returns
        -------
        str
            Unique value input binding name for this layer case.
        """

        return f"{self.name}_value"

    @property
    def output_name(self) -> str:
        """Return this layer's output binding name.

        Returns
        -------
        str
            Unique output binding name for this layer case.
        """

        return f"{self.name}_output"


LAYER_CASES = (
    LayerCase("self", 4, 4, 48),
    LayerCase("self_8_heads", 8, 8, 96),
    LayerCase("nonlinear", 4, 1, 64),
    LayerCase("nonlinear_8_heads", 8, 1, 64),
    # A non-12 head dimension forces the general multi-head cuBLAS path.
    LayerCase("general_multihead", 2, 2, 10),
)


@dataclass(frozen=True)
class AttentionEngine:
    """Keep the runtime alive alongside the engine and its numeric dtype."""

    runtime: trt.Runtime
    engine: trt.ICudaEngine
    dtype: str


@dataclass
class EngineRun:
    """Retain the context, stream, and all bound buffers until execution finishes."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    buffers: dict[str, cp.ndarray]


type PluginCreatorFixture = tuple[ctypes.CDLL, trt.IPluginCreatorV3One]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory) -> PluginCreatorFixture:
    """Compile and register the current attention-value plugin source.

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


def make_plugin(creator: trt.IPluginCreatorV3One, value_heads: int) -> trt.IPluginV3:
    """Create an attention-value plugin with the requested value-head count.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    value_heads : int
        Number of value heads; one uses attention head zero for nonlinear attention.

    Returns
    -------
    trt.IPluginV3
        New plugin configured for the build phase.
    """

    value = np.array([value_heads], dtype=np.int32)
    field = trt.PluginField("num_heads", value, trt.PluginFieldType.INT32)
    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([field]), trt.TensorRTPhase.BUILD
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
    """Set and read back a profile so rejection tests cannot fail at setup.

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


@pytest.fixture(scope="module", params=DTYPE_CASES)
def attention_engine(
    request: pytest.FixtureRequest, plugin_creator: PluginCreatorFixture
) -> AttentionEngine:
    """Build all production layouts and a guaranteed cuBLAS fallback per dtype.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    AttentionEngine
        Deserialized engine with its owning runtime.
    """

    _, creator = plugin_creator
    dtype = getattr(trt, request.param)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    profile = builder.create_optimization_profile()
    for case in LAYER_CASES:
        attention = network.add_input(
            case.attention_name, dtype, (-1, case.attention_heads, -1, -1)
        )
        value = network.add_input(case.value_name, dtype, (-1, -1, case.channels))
        assert attention is not None and value is not None
        layer = network.add_plugin_v3(
            [attention, value], [], make_plugin(creator, case.value_heads)
        )
        assert layer is not None
        output = layer.get_output(0)
        output.name = case.output_name
        network.mark_output(output)
        set_profile_shape(
            profile,
            case.attention_name,
            (1, case.attention_heads, 1, 1),
            (2, case.attention_heads, 17, 17),
            (3, case.attention_heads, 65, 65),
        )
        set_profile_shape(
            profile,
            case.value_name,
            (1, 1, case.channels),
            (2, 17, case.channels),
            (3, 65, case.channels),
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
        name: mode
        for case in LAYER_CASES
        for name, mode in (
            (case.attention_name, trt.TensorIOMode.INPUT),
            (case.value_name, trt.TensorIOMode.INPUT),
            (case.output_name, trt.TensorIOMode.OUTPUT),
        )
    }
    assert {name: engine.get_tensor_mode(name) for name in engine} == expected_io
    assert all(engine.get_tensor_dtype(name) == dtype for name in engine)
    return AttentionEngine(runtime, engine, request.param)


def make_host_inputs(
    batch_size: int, sequence_length: int, seed: int
) -> dict[str, np.typing.NDArray]:
    """Create deterministic, distinct attention and value inputs for every layer.

    Parameters
    ----------
    batch_size : int
        Number of utterances in the batch.
    sequence_length : int
        Physical number of frames per utterance.
    seed : int
        Local random-generator seed; does not change global NumPy or Torch state.

    Returns
    -------
    dict[str, np.typing.NDArray]
        Per-layer normalized attention weights and distinct FP32 value activations.
    """

    rng = np.random.default_rng(seed)
    inputs = {}
    for index, case in enumerate(LAYER_CASES):
        attention_rng = np.random.default_rng(seed + 101 * (index + 1))
        logits = attention_rng.normal(
            size=(batch_size, case.attention_heads, sequence_length, sequence_length)
        ).astype(np.float32)
        logits -= logits.max(axis=3, keepdims=True)
        attention = np.exp(logits)
        inputs[case.attention_name] = attention / attention.sum(axis=3, keepdims=True)
        inputs[case.value_name] = rng.normal(
            0.0, 0.5, (batch_size, sequence_length, case.channels)
        ).astype(np.float32)
    return inputs


def reference_attention_value(
    attention: np.typing.NDArray, value: np.typing.NDArray, value_heads: int
) -> np.typing.NDArray:
    """Evaluate NHTT-by-NTC in NumPy; a single value head uses attention head zero.

    Parameters
    ----------
    attention : np.typing.NDArray
        Attention weights with shape (batch, heads, query_time, key_time).
    value : np.typing.NDArray
        Value activations with shape (batch, time, channels).
    value_heads : int
        Number of value heads; one uses attention head zero for nonlinear attention.

    Returns
    -------
    np.typing.NDArray
        FP32 weighted values in the original NTC value layout.
    """

    batch, frames, channels = value.shape
    values_by_head = value.reshape(batch, frames, value_heads, channels // value_heads)
    return np.einsum(
        "nhqk,nkhd->nqhd", attention[:, :value_heads], values_by_head, optimize=True
    ).reshape(value.shape)


def prepare_engine_run(
    attention_engine: AttentionEngine,
    host_inputs: dict[str, np.typing.NDArray],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Resolve shapes and bind NaN-filled outputs without executing the engine.

    Parameters
    ----------
    attention_engine : AttentionEngine
        Engine, owning runtime, and numeric/layout settings.
    host_inputs : dict[str, np.typing.NDArray]
        Host attention/value arrays indexed by the engine's input binding names.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.

    Returns
    -------
    EngineRun
        Bound buffers and execution state; no inference has been enqueued.
    """

    if context is None:
        context = attention_engine.engine.create_execution_context()
    assert context is not None
    for name, array in host_inputs.items():
        assert context.set_input_shape(name, array.shape)
    assert context.infer_shapes() == []

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        buffers = {
            name: cp.array(array, dtype=attention_engine.dtype)
            for name, array in host_inputs.items()
        }
        for case in LAYER_CASES:
            shape = tuple(context.get_tensor_shape(case.output_name))
            assert shape == host_inputs[case.value_name].shape
            buffers[case.output_name] = cp.full(
                shape, cp.nan, dtype=attention_engine.dtype
            )
        for name, buffer in buffers.items():
            assert context.set_tensor_address(name, buffer.data.ptr)
    return EngineRun(context, stream, buffers)


def run_engine(
    attention_engine: AttentionEngine,
    host_inputs: dict[str, np.typing.NDArray],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Enqueue inference; the caller must synchronize before reading the buffers.

    Parameters
    ----------
    attention_engine : AttentionEngine
        Engine, owning runtime, and numeric/layout settings.
    host_inputs : dict[str, np.typing.NDArray]
        Host attention/value arrays indexed by the engine's input binding names.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.

    Returns
    -------
    EngineRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    run = prepare_engine_run(attention_engine, host_inputs, context, stream)
    with run.stream:
        assert run.context.execute_async_v3(run.stream.ptr)
    return run


def assert_run_matches_reference(
    run: EngineRun,
    dtype: str,
    host_inputs: dict[str, np.typing.NDArray],
    exact: bool = False,
) -> None:
    """Check untouched inputs and outputs against an independent CPU dtype oracle.

    Parameters
    ----------
    run : EngineRun
        Bound device buffers and the context/stream that own their pending work.
    dtype : str
        Numeric storage dtype name, including bfloat16 where supported.
    host_inputs : dict[str, np.typing.NDArray]
        Host attention/value arrays indexed by the engine's input binding names.
    exact : bool
        Require exact equality for representable hand-computed inputs.
    """

    run.stream.synchronize()
    expected_inputs = {
        name: torch.from_numpy(array).to(getattr(torch, dtype)).float().numpy()
        for name, array in host_inputs.items()
    }
    for name, expected in expected_inputs.items():
        np.testing.assert_array_equal(
            cp.asnumpy(run.buffers[name]).astype(np.float32), expected
        )
    for case in LAYER_CASES:
        actual = cp.asnumpy(run.buffers[case.output_name]).astype(np.float32)
        expected = reference_attention_value(
            expected_inputs[case.attention_name],
            expected_inputs[case.value_name],
            case.value_heads,
        )
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all()
        if exact:
            np.testing.assert_array_equal(actual, expected)
        else:
            tolerance = 1e-2 if dtype == "bfloat16" else 2e-3
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"), ((1, 1), (1, 3), (2, 17), (3, 65))
)
def test_attention_value_plugin_matches_reference(
    attention_engine: AttentionEngine, batch_size: int, sequence_length: int
) -> None:
    inputs = make_host_inputs(
        batch_size, sequence_length, 1000 + batch_size * 100 + sequence_length
    )
    run = run_engine(attention_engine, inputs)
    assert_run_matches_reference(run, attention_engine.dtype, inputs)


def test_attention_value_plugin_preserves_query_key_and_head_layout(
    attention_engine: AttentionEngine,
) -> None:
    inputs = {}
    coefficients = np.array((0.0, 0.5, -1.0, 2.0), dtype=np.float32)
    for case in LAYER_CASES:
        attention = np.zeros((2, case.attention_heads, 3, 3), dtype=np.float32)
        batch, head, query = np.indices((2, case.attention_heads, 3))
        key = (2 * query + head + batch) % 3
        attention[batch, head, query, key] = coefficients[
            (query + 2 * head + 3 * batch) % 4
        ]
        batch, key, channel = np.indices((2, 3, case.channels))
        inputs[case.attention_name] = attention
        inputs[case.value_name] = ((7 * batch + 3 * key + channel) % 23 - 11).astype(
            np.float32
        )
    run = run_engine(attention_engine, inputs)
    assert_run_matches_reference(run, attention_engine.dtype, inputs, exact=True)


def test_attention_value_plugin_accumulates_products_in_fp32(
    attention_engine: AttentionEngine,
) -> None:
    values_by_key = np.array((2048.0, 1.0, 1.0, -2048.0), dtype=np.float32)
    inputs = {}
    for case in LAYER_CASES:
        inputs[case.attention_name] = np.ones(
            (1, case.attention_heads, 4, 4), dtype=np.float32
        )
        inputs[case.value_name] = np.broadcast_to(
            values_by_key.reshape(1, 4, 1), (1, 4, case.channels)
        ).copy()
    run = run_engine(attention_engine, inputs)
    run.stream.synchronize()
    for case in LAYER_CASES:
        np.testing.assert_array_equal(
            cp.asnumpy(run.buffers[case.output_name]).astype(np.float32),
            np.full((1, 4, case.channels), 2.0, dtype=np.float32),
        )


def test_attention_value_plugin_supports_cuda_graph_replay(
    attention_engine: AttentionEngine,
) -> None:
    inputs = make_host_inputs(2, 17, seed=2026)
    run = run_engine(attention_engine, inputs)
    assert_run_matches_reference(run, attention_engine.dtype, inputs)
    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for seed in (3000, 3001):
        inputs = make_host_inputs(2, 17, seed)
        with run.stream:
            for name, array in inputs.items():
                cp.copyto(
                    run.buffers[name], cp.array(array, dtype=attention_engine.dtype)
                )
            for case in LAYER_CASES:
                run.buffers[case.output_name].fill(cp.nan)
            graph.launch(run.stream)
        assert_run_matches_reference(run, attention_engine.dtype, inputs)


def test_attention_value_plugin_reuses_context_across_shapes_and_streams(
    attention_engine: AttentionEngine,
) -> None:
    context = attention_engine.engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    for index, (batch, frames) in enumerate(((1, 1), (3, 65), (1, 3))):
        inputs = make_host_inputs(batch, frames, seed=4000 + index)
        stream = streams[index % 2]
        run = run_engine(attention_engine, inputs, context, stream)
        assert run.context is context and run.stream is stream
        assert_run_matches_reference(run, attention_engine.dtype, inputs)


def test_attention_value_plugin_supports_concurrent_contexts(
    attention_engine: AttentionEngine,
) -> None:
    first_inputs = make_host_inputs(1, 17, seed=5001)
    second_inputs = make_host_inputs(3, 65, seed=5002)
    first = run_engine(attention_engine, first_inputs)
    second = run_engine(attention_engine, second_inputs)
    assert first.context is not second.context
    assert first.stream.ptr != second.stream.ptr
    assert_run_matches_reference(first, attention_engine.dtype, first_inputs)
    assert_run_matches_reference(second, attention_engine.dtype, second_inputs)


@pytest.mark.parametrize("relationship", ("batch", "sequence", "non-square-attention"))
def test_attention_value_plugin_rejects_runtime_shape_mismatch(
    attention_engine: AttentionEngine, relationship: str
) -> None:
    inputs = make_host_inputs(2, 17, seed=6000)
    case = LAYER_CASES[0]
    if relationship == "batch":
        inputs[case.value_name] = inputs[case.value_name][:1]
    elif relationship == "sequence":
        inputs[case.value_name] = inputs[case.value_name][:, :16]
    else:
        inputs[case.attention_name] = inputs[case.attention_name][..., :16]
    run = prepare_engine_run(attention_engine, inputs)
    with run.stream:
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert not executed
    assert bool(cp.isnan(run.buffers[case.output_name]).all())


def attention_value_input_specs(
    attention_shape: tuple[int, ...] = (2, 4, 7, 7),
    value_shape: tuple[int, ...] = (2, 7, 48),
    attention_dtype: trt.DataType = trt.float16,
    value_dtype: trt.DataType = trt.float16,
) -> tuple[InputSpec, ...]:
    """Describe the attention and value tensors for a build-contract test.

    Parameters
    ----------
    attention_shape : tuple[int, ...]
        Shape of the attention weights.
    value_shape : tuple[int, ...]
        Shape of the value activations.
    attention_dtype : trt.DataType
        TensorRT dtype of attention weights.
    value_dtype : trt.DataType
        TensorRT dtype of value activations.

    Returns
    -------
    tuple[InputSpec, ...]
        Input-ordered dtype/shape pairs, without building or allocating an engine.
    """

    return ((attention_dtype, attention_shape), (value_dtype, value_shape))


def build_contract(
    creator: trt.IPluginCreatorV3One,
    value_heads: int,
    input_specs: tuple[InputSpec, ...],
    profiles: tuple = (),
) -> trt.IHostMemory | None:
    """Build one plugin with optional min/opt/max profiles; return None on rejection.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    value_heads : int
        Number of value heads; one uses attention head zero for nonlinear attention.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.
    profiles : tuple
        Input-ordered tuples of min/opt/max shapes; empty means static.

    Returns
    -------
    trt.IHostMemory | None
        Serialized engine bytes, or None when TensorRT rejects the contract.
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
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator, value_heads))
    if layer is None:
        return None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if profiles:
        profile = builder.create_optimization_profile()
        for tensor, shapes in zip(inputs, profiles, strict=True):
            set_profile_shape(profile, tensor.name, *shapes)
        assert config.add_optimization_profile(profile) == 0
    return builder.build_serialized_network(network, config)


@pytest.mark.parametrize(
    ("value_heads", "attention_shape", "value_shape"),
    (
        pytest.param(1, (1, 1, 1, 1), (1, 1, 1), id="minimum"),
        pytest.param(4, (2, 4, 7, 7), (2, 7, 48), id="self-attention"),
        pytest.param(8, (2, 8, 7, 7), (2, 7, 96), id="self-attention-8-heads"),
        pytest.param(1, (2, 4, 7, 7), (2, 7, 64), id="nonlinear-attention"),
        pytest.param(1, (2, 8, 7, 7), (2, 7, 64), id="nonlinear-attention-8-heads"),
        pytest.param(2, (2, 2, 7, 7), (2, 7, 10), id="general-multihead"),
    ),
)
def test_attention_value_plugin_accepts_valid_static_contracts(
    plugin_creator: PluginCreatorFixture,
    value_heads: int,
    attention_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
) -> None:
    _, creator = plugin_creator
    specs = attention_value_input_specs(attention_shape, value_shape)
    assert build_contract(creator, value_heads, specs) is not None


@pytest.mark.parametrize(
    "input_specs",
    (
        pytest.param(attention_value_input_specs((2, 4, 7)), id="attention-rank"),
        pytest.param(attention_value_input_specs(value_shape=(2, 7)), id="value-rank"),
        pytest.param(
            attention_value_input_specs((0, 4, 7, 7), (0, 7, 48)), id="empty-batch"
        ),
        pytest.param(
            attention_value_input_specs((2, 4, 0, 0), (2, 0, 48)), id="empty-sequence"
        ),
        pytest.param(
            attention_value_input_specs((2, 0, 7, 7)), id="empty-attention-heads"
        ),
        pytest.param(
            attention_value_input_specs(value_shape=(2, 7, 0)),
            id="empty-value-channels",
        ),
        pytest.param(attention_value_input_specs((1, 4, 7, 7)), id="batch-mismatch"),
        pytest.param(
            attention_value_input_specs((2, 4, 7, 8)), id="non-square-attention"
        ),
        pytest.param(
            attention_value_input_specs(value_shape=(2, 8, 48)), id="sequence-mismatch"
        ),
        pytest.param(
            attention_value_input_specs((2, 3, 7, 7)), id="too-few-attention-heads"
        ),
        pytest.param(
            attention_value_input_specs((2, 5, 7, 7)),
            id="extra-multihead-attention-head",
        ),
        pytest.param(
            attention_value_input_specs(value_shape=(2, 7, 50)),
            id="channels-not-divisible-by-heads",
        ),
        pytest.param(
            attention_value_input_specs(
                attention_dtype=trt.int32, value_dtype=trt.int32
            ),
            id="unsupported-dtype",
        ),
        pytest.param(
            attention_value_input_specs(value_dtype=trt.float32), id="mixed-dtypes"
        ),
        pytest.param(attention_value_input_specs()[:1], id="missing-input"),
        pytest.param(
            attention_value_input_specs() + ((trt.float16, (1,)),), id="extra-input"
        ),
    ),
)
def test_attention_value_plugin_rejects_invalid_contracts(
    plugin_creator: PluginCreatorFixture, input_specs: tuple[InputSpec, ...]
) -> None:
    _, creator = plugin_creator
    assert build_contract(creator, 4, input_specs) is None


def test_attention_value_plugin_accepts_valid_profile(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    specs = attention_value_input_specs((-1, 4, -1, -1), (-1, -1, 48))
    profiles = (
        ((1, 4, 1, 1), (2, 4, 17, 17), (3, 4, 65, 65)),
        ((1, 1, 48), (2, 17, 48), (3, 65, 48)),
    )
    assert build_contract(creator, 4, specs, profiles) is not None


@pytest.mark.parametrize("endpoint", (0, 1, 2), ids=("min", "opt", "max"))
@pytest.mark.parametrize("relationship", ("batch", "sequence", "non-square-attention"))
def test_attention_value_plugin_rejects_invalid_profile_endpoints(
    plugin_creator: PluginCreatorFixture, endpoint: int, relationship: str
) -> None:
    _, creator = plugin_creator
    attention_shapes = [[1, 4, 1, 1], [2, 4, 17, 17], [3, 4, 65, 65]]
    value_shapes = [[1, 1, 48], [2, 17, 48], [3, 65, 48]]
    # Keep each input's profile monotonic; only the shape relationship is invalid.
    if relationship == "non-square-attention":
        attention_shapes[endpoint][3] += 1 if endpoint < 2 else -1
    else:
        dimension = 0 if relationship == "batch" else 1
        value_shapes[endpoint][dimension] += 1 if endpoint == 0 else -1
    profiles = (tuple(map(tuple, attention_shapes)), tuple(map(tuple, value_shapes)))
    specs = attention_value_input_specs((-1, 4, -1, -1), (-1, -1, 48))
    assert build_contract(creator, 4, specs, profiles) is None


def test_attention_value_creator_exposes_complete_contract(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert (creator.name, creator.plugin_version, creator.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    assert [(field.name, field.type, field.size) for field in creator.field_names] == [
        ("num_heads", trt.PluginFieldType.INT32, 1)
    ]

    plugin = make_plugin(creator, 4)
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None and build is not None and runtime is not None
    assert (core.plugin_name, core.plugin_version, core.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    assert build.num_outputs == 1


def test_attention_value_timing_cache_depends_on_head_layout(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    plugins = [make_plugin(creator, heads) for heads in (1, 2, 4, 8, 4)]
    cache_ids = [
        plugin.get_capability_interface(trt.PluginCapabilityType.BUILD).timing_cache_id
        for plugin in plugins
    ]
    assert all(cache_ids)
    assert len(set(cache_ids[:4])) == 4
    assert cache_ids[2] == cache_ids[4]


@pytest.mark.parametrize(
    ("values", "field_type"),
    (
        pytest.param(
            np.array([1.0], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
            id="wrong-type",
        ),
        pytest.param(
            np.array([], dtype=np.int32), trt.PluginFieldType.INT32, id="empty"
        ),
        pytest.param(
            np.array([1, 4], dtype=np.int32), trt.PluginFieldType.INT32, id="multiple"
        ),
        pytest.param(
            np.array([0], dtype=np.int32), trt.PluginFieldType.INT32, id="zero"
        ),
        pytest.param(
            np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32, id="negative"
        ),
    ),
)
def test_attention_value_creator_rejects_malformed_num_heads(
    plugin_creator: PluginCreatorFixture,
    values: np.typing.NDArray,
    field_type: trt.PluginFieldType,
) -> None:
    _, creator = plugin_creator
    field = trt.PluginField("num_heads", values, field_type)
    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([field]), trt.TensorRTPhase.BUILD
    )
    assert plugin is None


@pytest.mark.parametrize(
    ("field_specs", "valid"),
    (
        pytest.param((), False, id="missing"),
        pytest.param((("implementation_metadata", 17),), False, id="unknown-only"),
        pytest.param((("num_heads", 1), ("num_heads", 4)), False, id="duplicate"),
        pytest.param(
            (("implementation_metadata", 17), ("num_heads", 2)),
            True,
            id="reordered-and-unknown",
        ),
    ),
)
def test_attention_value_creator_validates_field_collection(
    plugin_creator: PluginCreatorFixture,
    field_specs: tuple[tuple[str, int], ...],
    valid: bool,
) -> None:
    _, creator = plugin_creator
    values = [np.array([value], dtype=np.int32) for _, value in field_specs]
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(name, value, trt.PluginFieldType.INT32)
            for (name, _), value in zip(field_specs, values, strict=True)
        ]
    )
    plugin = creator.create_plugin(PLUGIN_NAME, fields, trt.TensorRTPhase.BUILD)
    assert (plugin is not None) is valid
