#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer output-assembly plugin."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import (
    INT32_MAX,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
)

pytestmark = pytest.mark.cuda

PLUGIN_NAME = ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME
PLUGIN_VERSION = "1"
ENCODER_DIMS = (16, 32, 48, 64, 48, 32)
SHAPE_CASES = ((1, 1), (1, 3), (2, 17), (3, 65))
ENGINE_CASES = (
    pytest.param(("float32", ENCODER_DIMS), id="fp32"),
    pytest.param(("float16", ENCODER_DIMS), id="fp16"),
    pytest.param(("bfloat16", ENCODER_DIMS), id="bf16", marks=pytest.mark.sm80),
    pytest.param(("float32", (13, 1, 27, 20, 12, 4)), id="fp32-vector-boundary"),
    pytest.param(("float16", (13, 1, 27, 24, 16, 8)), id="fp16-vector-boundary"),
    pytest.param(("float16", (1, 2, 3, 8, 8, 8)), id="fp16-minimal-equal"),
    pytest.param(("float16", (1, 2, 3, 16, 16, 8)), id="fp16-empty-encoder4-band"),
    pytest.param(("float16", (1, 2, 3, 24, 8, 8)), id="fp16-empty-encoder5-band"),
)


@dataclass(frozen=True)
class AssemblyEngine:
    """Keep the runtime alive alongside the engine and its input layout."""

    runtime: trt.Runtime
    engine: trt.ICudaEngine
    dtype: np.dtype
    encoder_dims: tuple[int, ...]


type PluginCreatorFixture = tuple[ctypes.CDLL, trt.IPluginCreatorV3One]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory) -> PluginCreatorFixture:
    """Compile and register the current output-assembly plugin source.

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
        "zipformer_output_assembly_plugin.cu",
        "initFastGpuAsrZipformerOutputAssemblyPlugin",
        ("cudart",),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return library, creator


def make_plugin(creator: trt.IPluginCreatorV3One) -> trt.IPluginV3:
    """Create the field-free output-assembly plugin.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.

    Returns
    -------
    trt.IPluginV3
        New plugin configured for the build phase.
    """

    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
    )
    assert plugin is not None
    return plugin


def build_engine(
    creator: trt.IPluginCreatorV3One,
    input_specs: tuple[InputSpec, ...],
    profiles: tuple = (),
) -> tuple[trt.Runtime, trt.ICudaEngine] | None:
    """Build with optional min/opt/max profiles; return None on rejection.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.
    profiles : tuple
        Input-ordered tuples of min/opt/max shapes; empty means static.

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
    inputs = [
        network.add_input(f"encoder_{index + 1}", dtype, shape)
        for index, (dtype, shape) in enumerate(input_specs)
    ]
    assert all(tensor is not None for tensor in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
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
            profile.set_shape(tensor.name, *shapes)
            assert tuple(map(tuple, profile.get_shape(tensor.name))) == shapes
        assert config.add_optimization_profile(profile) == 0
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    return runtime, engine


def build_assembly_engine(
    creator: trt.IPluginCreatorV3One,
    dtype: str,
    encoder_dims: tuple[int, ...] = ENCODER_DIMS,
) -> AssemblyEngine:
    """Build a dynamic engine and verify its six-input, one-output signature.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    dtype : str
        Numeric storage dtype name, including bfloat16 where supported.
    encoder_dims : tuple[int, ...]
        Channel widths of the six consecutive encoder stacks.

    Returns
    -------
    AssemblyEngine
        Deserialized engine with its owning runtime.
    """

    trt_dtype = getattr(trt, dtype)
    specs = tuple((trt_dtype, (-1, -1, channels)) for channels in encoder_dims)
    profiles = tuple(
        ((1, 1, channels), (2, 17, channels), (3, 65, channels))
        for channels in encoder_dims
    )
    built = build_engine(creator, specs, profiles)
    assert built is not None
    runtime, engine = built
    expected_io = {
        **{f"encoder_{index + 1}": trt.TensorIOMode.INPUT for index in range(6)},
        "output": trt.TensorIOMode.OUTPUT,
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, mode in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == trt_dtype
    return AssemblyEngine(runtime, engine, cp.dtype(dtype), encoder_dims)


@pytest.fixture(scope="module", params=ENGINE_CASES)
def assembly_engine(
    request: pytest.FixtureRequest, plugin_creator: PluginCreatorFixture
) -> AssemblyEngine:
    """Build each dtype/layout once for all inference checks.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    AssemblyEngine
        Deserialized engine with its owning runtime.
    """

    _, creator = plugin_creator
    return build_assembly_engine(creator, *request.param)


@pytest.fixture(scope="module")
def alignment_engine(plugin_creator: PluginCreatorFixture) -> AssemblyEngine:
    """Share a representative FP16 engine across binding checks.

    Parameters
    ----------
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    AssemblyEngine
        Deserialized engine with its owning runtime.
    """

    _, creator = plugin_creator
    return build_assembly_engine(creator, "float16")


def make_inputs(
    batch_size: int, sequence_length: int, encoder_dims: tuple[int, ...], seed: int = 0
) -> tuple[np.typing.NDArray, ...]:
    """Create deterministic, distinct values for all six encoder stacks.

    Parameters
    ----------
    batch_size : int
        Number of utterances in the batch.
    sequence_length : int
        Physical number of frames per utterance.
    encoder_dims : tuple[int, ...]
        Channel widths of the six consecutive encoder stacks.
    seed : int
        Local random-generator seed; does not change global NumPy or Torch state.

    Returns
    -------
    tuple[np.typing.NDArray, ...]
        One independent FP32 NTC array per encoder stack.
    """

    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal(size=(batch_size, sequence_length, channels)).astype(np.float32)
        for channels in encoder_dims
    )


def expected_assembly(inputs: tuple[cp.ndarray, ...]) -> cp.ndarray:
    """Concatenate the surviving channel bands without using plugin metadata.

    Parameters
    ----------
    inputs : tuple[cp.ndarray, ...]
        Six device tensors in stack order; the final three supply the channel bands.

    Returns
    -------
    cp.ndarray
        Concatenated contributor bands, preserving source dtype and exact stored
        bits.

    Notes
    -----
    Operations are enqueued on the current CuPy stream; the caller owns ordering.
    """

    return cp.concatenate(
        (
            inputs[5],
            inputs[4][:, :, inputs[5].shape[2] :],
            inputs[3][:, :, inputs[4].shape[2] :],
        ),
        axis=2,
    )


@dataclass
class EngineRun:
    """Retain the context, stream, and bound device buffers until work completes."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    inputs: list[cp.ndarray]
    output: cp.ndarray


def prepare_engine_run(
    assembly: AssemblyEngine,
    host_inputs: tuple[np.typing.NDArray, ...],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Resolve dynamic shapes and bind buffers with a NaN output sentinel.

    Parameters
    ----------
    assembly : AssemblyEngine
        Engine, owning runtime, storage dtype, and six stack widths.
    host_inputs : tuple[np.typing.NDArray, ...]
        Six host stack outputs in NTC layout, ordered from encoder 1 through 6.
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
        context = assembly.engine.create_execution_context()
    assert context is not None
    assert len(host_inputs) == 6
    for index, values in enumerate(host_inputs):
        assert context.set_input_shape(f"encoder_{index + 1}", values.shape)
    assert context.infer_shapes() == []
    assert tuple(context.get_tensor_shape("output")) == host_inputs[3].shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        inputs = [cp.array(values, dtype=assembly.dtype) for values in host_inputs]
        output = cp.full(host_inputs[3].shape, cp.nan, dtype=assembly.dtype)
        for index, values in enumerate(inputs):
            assert context.set_tensor_address(f"encoder_{index + 1}", values.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
    return EngineRun(context, stream, inputs, output)


def run_engine(
    assembly: AssemblyEngine,
    host_inputs: tuple[np.typing.NDArray, ...],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Enqueue inference; the caller synchronizes before inspecting the buffers.

    Parameters
    ----------
    assembly : AssemblyEngine
        Engine, owning runtime, storage dtype, and six stack widths.
    host_inputs : tuple[np.typing.NDArray, ...]
        Six host stack outputs in NTC layout, ordered from encoder 1 through 6.
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

    run = prepare_engine_run(assembly, host_inputs, context, stream)
    assert run.context.execute_async_v3(run.stream.ptr)
    return run


def assert_run_matches_reference(
    run: EngineRun, host_inputs: tuple[np.typing.NDArray, ...]
) -> None:
    """Compare output and unchanged inputs bit-for-bit against fresh host copies.

    Parameters
    ----------
    run : EngineRun
        Bound device buffers and the context/stream that own their pending work.
    host_inputs : tuple[np.typing.NDArray, ...]
        Six host stack outputs in NTC layout, ordered from encoder 1 through 6.
    """

    with run.stream:
        expected_inputs = tuple(
            cp.array(values, dtype=run.output.dtype) for values in host_inputs
        )
        expected_output = expected_assembly(expected_inputs)
    run.stream.synchronize()
    for actual, expected in zip(run.inputs, expected_inputs, strict=True):
        cp.testing.assert_array_equal(actual.view(cp.uint8), expected.view(cp.uint8))
    cp.testing.assert_array_equal(
        run.output.view(cp.uint8), expected_output.view(cp.uint8)
    )


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_output_assembly_plugin_matches_reference(
    assembly_engine: AssemblyEngine, batch_size: int, sequence_length: int
) -> None:
    inputs = make_inputs(batch_size, sequence_length, assembly_engine.encoder_dims)
    run = run_engine(assembly_engine, inputs)
    assert_run_matches_reference(run, inputs)


def test_output_assembly_plugin_places_bands_at_exact_coordinates(
    alignment_engine: AssemblyEngine,
) -> None:
    frames = np.arange(2, dtype=np.float32)[:, None] * 80
    inputs = tuple(
        ((index + 1) * 200 + frames + np.arange(channels, dtype=np.float32))[None]
        for index, channels in enumerate(alignment_engine.encoder_dims)
    )
    run = run_engine(alignment_engine, inputs)
    expected = (
        np.concatenate(
            (
                1200 + np.arange(32, dtype=np.float32),
                1000 + np.arange(32, 48, dtype=np.float32),
                800 + np.arange(48, 64, dtype=np.float32),
            )
        )
        + frames
    )
    run.stream.synchronize()

    np.testing.assert_array_equal(cp.asnumpy(run.output), expected[None])


def test_output_assembly_plugin_supports_cuda_graphs(
    assembly_engine: AssemblyEngine,
) -> None:
    inputs = make_inputs(2, 17, assembly_engine.encoder_dims)
    run = run_engine(assembly_engine, inputs)
    assert_run_matches_reference(run, inputs)

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for seed in (1, 2):
        inputs = make_inputs(2, 17, assembly_engine.encoder_dims, seed)
        with run.stream:
            for destination, source in zip(run.inputs, inputs, strict=True):
                cp.copyto(destination, cp.array(source, dtype=assembly_engine.dtype))
            run.output.fill(cp.nan)
            graph.launch(run.stream)
        assert_run_matches_reference(run, inputs)


def test_output_assembly_plugin_reuses_context_across_dynamic_shapes(
    assembly_engine: AssemblyEngine,
) -> None:
    context = assembly_engine.engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)

    for seed, (batch, frames) in enumerate(
        (SHAPE_CASES[0], SHAPE_CASES[-1], SHAPE_CASES[0])
    ):
        inputs = make_inputs(batch, frames, assembly_engine.encoder_dims, seed)
        run = run_engine(assembly_engine, inputs, context, stream)
        assert run.context is context
        assert_run_matches_reference(run, inputs)


def test_output_assembly_plugin_supports_concurrent_contexts(
    assembly_engine: AssemblyEngine,
) -> None:
    first_inputs = make_inputs(1, 3, assembly_engine.encoder_dims, 1)
    second_inputs = make_inputs(3, 65, assembly_engine.encoder_dims, 2)
    first = run_engine(assembly_engine, first_inputs)
    second = run_engine(assembly_engine, second_inputs)

    assert first.context is not second.context
    assert first.stream.ptr != second.stream.ptr
    assert_run_matches_reference(first, first_inputs)
    assert_run_matches_reference(second, second_inputs)


def test_output_assembly_plugin_output_is_independent_of_dependency_values(
    assembly_engine: AssemblyEngine,
) -> None:
    inputs = tuple(
        np.full((1, 3, channels), sentinel, dtype=np.float32)
        for channels, sentinel in zip(
            assembly_engine.encoder_dims,
            (np.nan, np.inf, -np.inf, 4, 5, 6),
            strict=True,
        )
    )
    run = run_engine(assembly_engine, inputs)
    assert_run_matches_reference(run, inputs)


def test_output_assembly_plugin_preserves_contributor_bit_patterns(
    assembly_engine: AssemblyEngine,
) -> None:
    inputs = make_inputs(2, 17, assembly_engine.encoder_dims)
    run = prepare_engine_run(assembly_engine, inputs)
    if run.output.dtype.itemsize == 4:
        bit_dtype = cp.uint32
        patterns = (
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x00000001,
            0x7F7FFFFF,
            0x3F800000,
        )
    else:
        bit_dtype = cp.uint16
        patterns = (
            0x0000,
            0x8000,
            0x7C00,
            0xFC00,
            0x7E01,
            0x7F80,
            0xFF80,
            0x7FC1,
            0x0001,
            0x7BFF,
            0x7F7F,
            0x3C00,
            0x3F80,
        )

    with run.stream:
        patterns = cp.array(patterns, dtype=bit_dtype)
        for source_index, values in enumerate(run.inputs[3:]):
            bits = values.view(bit_dtype).reshape(-1)
            indexes = (
                cp.arange(bits.size, dtype=cp.int64) + source_index
            ) % patterns.size
            bits[...] = patterns[indexes]
        expected_sources = [values.view(bit_dtype).copy() for values in run.inputs[3:]]
        expected_bits = expected_assembly(
            tuple(values.view(bit_dtype) for values in run.inputs)
        )
        run.output.view(cp.uint8).fill(0xA5)
        assert run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    for actual, expected in zip(run.inputs[3:], expected_sources, strict=True):
        cp.testing.assert_array_equal(actual.view(bit_dtype), expected)
    cp.testing.assert_array_equal(run.output.view(bit_dtype), expected_bits)


@pytest.mark.parametrize("input_index", range(6), ids=lambda i: f"encoder-{i + 1}")
@pytest.mark.parametrize("axis", (0, 1), ids=("batch", "time"))
@pytest.mark.parametrize("delta", (-1, 1), ids=("shorter", "longer"))
def test_output_assembly_plugin_rejects_runtime_shape_mismatch(
    alignment_engine: AssemblyEngine, input_index: int, axis: int, delta: int
) -> None:
    inputs = list(make_inputs(2, 3, alignment_engine.encoder_dims))
    shape = list(inputs[input_index].shape)
    shape[axis] += delta
    inputs[input_index] = np.zeros(shape, dtype=np.float32)
    run = prepare_engine_run(alignment_engine, tuple(inputs))

    assert not run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert bool(cp.isnan(run.output).all())


def make_misaligned_copy(values: cp.ndarray) -> cp.ndarray:
    """Copy into an offset view, which retains its backing allocation.

    Parameters
    ----------
    values : cp.ndarray
        Input values to round or copy without modifying the original array.

    Returns
    -------
    cp.ndarray
        Same-shaped device view, offset one element into its owned backing
        allocation.

    Notes
    -----
    Operations are enqueued on the current CuPy stream; the caller owns ordering.
    """

    backing = cp.empty(values.size + 1, dtype=values.dtype)
    misaligned = backing[1:].reshape(values.shape)
    cp.copyto(misaligned, values)
    assert misaligned.data.ptr % 16 != 0
    return misaligned


@pytest.mark.parametrize(
    "binding",
    (
        "encoder_1",
        "encoder_2",
        "encoder_3",
        "encoder_4",
        "encoder_5",
        "encoder_6",
        "output",
    ),
)
def test_output_assembly_plugin_binding_alignment(
    alignment_engine: AssemblyEngine, binding: str
) -> None:
    inputs = make_inputs(1, 3, alignment_engine.encoder_dims)
    run = prepare_engine_run(alignment_engine, inputs)
    with run.stream:
        if binding == "output":
            run.output = make_misaligned_copy(run.output)
            buffer = run.output
        else:
            index = int(binding.removeprefix("encoder_")) - 1
            run.inputs[index] = make_misaligned_copy(run.inputs[index])
            buffer = run.inputs[index]
        assert run.context.set_tensor_address(binding, buffer.data.ptr)
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    if binding in ("encoder_1", "encoder_2", "encoder_3"):
        assert executed
        assert_run_matches_reference(run, inputs)
    else:
        assert not executed
        assert bool(cp.isnan(run.output).all())


def test_output_assembly_creator_exposes_parameter_free_contract(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert creator.name == PLUGIN_NAME
    assert creator.plugin_version == PLUGIN_VERSION
    assert creator.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert tuple(creator.field_names) == ()

    plugin = make_plugin(creator)
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None and build is not None and runtime is not None
    assert core.plugin_name == PLUGIN_NAME
    assert core.plugin_version == PLUGIN_VERSION
    assert core.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert build.num_outputs == 1


def test_output_assembly_creator_rejects_unexpected_fields(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    value = np.array([1], dtype=np.int32)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection(
            [trt.PluginField("unexpected", value, trt.PluginFieldType.INT32)]
        ),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def assembly_input_specs(
    encoder_dims: tuple[int, ...] = ENCODER_DIMS,
    batch_sizes: tuple[int, ...] = (1,) * 6,
    sequence_lengths: tuple[int, ...] = (3,) * 6,
    dtypes: tuple[trt.DataType, ...] | None = None,
    shape_overrides: dict[int, tuple[int, ...]] | None = None,
) -> tuple[InputSpec, ...]:
    """Create one static six-input TensorRT contract.

    Parameters
    ----------
    encoder_dims : tuple[int, ...]
        Channel widths of the six consecutive encoder stacks.
    batch_sizes : tuple[int, ...]
        Batch dimensions of the six stack inputs.
    sequence_lengths : tuple[int, ...]
        Time dimensions of the six stack inputs.
    dtypes : tuple[trt.DataType, ...] or None
        Input dtypes in binding order; None selects this helper's default contract.
    shape_overrides : dict[int, tuple[int, ...]] or None
        Zero-based input indices whose complete shapes should be replaced.

    Returns
    -------
    tuple[InputSpec, ...]
        Input-ordered dtype/shape pairs, without building or allocating an engine.
    """

    if dtypes is None:
        dtypes = (trt.float16,) * 6
    specs = [
        (dtypes[index], (batch_sizes[index], sequence_lengths[index], channels))
        for index, channels in enumerate(encoder_dims)
    ]
    for index, shape in (shape_overrides or {}).items():
        specs[index] = (dtypes[index], shape)
    return tuple(specs)


INVALID_CONTRACT_CASES = (
    *(
        pytest.param(
            assembly_input_specs(shape_overrides={input_index: (1, 3)}),
            id=f"encoder-{input_index + 1}-rank",
        )
        for input_index in range(6)
    ),
    pytest.param(assembly_input_specs(batch_sizes=(0,) * 6), id="empty-batch"),
    pytest.param(assembly_input_specs(sequence_lengths=(0,) * 6), id="empty-sequence"),
    *(
        pytest.param(
            assembly_input_specs(
                encoder_dims=tuple(
                    0 if index == input_index else channels
                    for index, channels in enumerate(ENCODER_DIMS)
                )
            ),
            id=f"encoder-{input_index + 1}-empty-channels",
        )
        for input_index in range(3)
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:3] + (0, 0, 0)),
        id="empty-encoder4-channels",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:3] + (64, 0, 0)),
        id="empty-encoder5-channels",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:3] + (64, 48, 0)),
        id="empty-encoder6-channels",
    ),
    pytest.param(
        assembly_input_specs(
            batch_sizes=(INT32_MAX,) * 6, sequence_lengths=(INT32_MAX,) * 6
        ),
        id="address-volume-overflow",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:3] + (66,) + ENCODER_DIMS[4:]),
        id="unaligned-output-fp16",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:4] + (50,) + ENCODER_DIMS[5:]),
        id="unaligned-encoder5-fp16",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=ENCODER_DIMS[:5] + (34,)),
        id="unaligned-encoder6-fp16",
    ),
    pytest.param(
        assembly_input_specs(
            encoder_dims=(16, 32, 48, 18, 12, 4), dtypes=(trt.float32,) * 6
        ),
        id="unaligned-output-fp32",
    ),
    pytest.param(
        assembly_input_specs(
            encoder_dims=ENCODER_DIMS[:4] + (50, 32), dtypes=(trt.float32,) * 6
        ),
        id="unaligned-encoder5-fp32",
    ),
    pytest.param(
        assembly_input_specs(
            encoder_dims=ENCODER_DIMS[:5] + (34,), dtypes=(trt.float32,) * 6
        ),
        id="unaligned-encoder6-fp32",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=(16, 32, 48, 64, 32, 48)),
        id="encoder5-narrower-than-encoder6",
    ),
    pytest.param(
        assembly_input_specs(encoder_dims=(16, 32, 48, 48, 64, 32)),
        id="encoder4-narrower-than-encoder5",
    ),
    pytest.param(
        assembly_input_specs(batch_sizes=(2, 1, 1, 1, 1, 1)), id="dependency-batch"
    ),
    pytest.param(
        assembly_input_specs(batch_sizes=(1, 1, 1, 2, 1, 1)), id="contributor-batch"
    ),
    pytest.param(
        assembly_input_specs(sequence_lengths=(3, 4, 3, 3, 3, 3)), id="dependency-time"
    ),
    pytest.param(
        assembly_input_specs(sequence_lengths=(3, 3, 3, 4, 3, 3)), id="contributor-time"
    ),
    pytest.param(assembly_input_specs(dtypes=(trt.int32,) * 6), id="unsupported-dtype"),
    *(
        pytest.param(
            assembly_input_specs(
                dtypes=tuple(
                    trt.float32 if index == input_index else trt.float16
                    for index in range(6)
                )
            ),
            id=f"encoder-{input_index + 1}-mixed-dtype",
        )
        for input_index in range(6)
    ),
    pytest.param(assembly_input_specs()[:5], id="missing-input"),
    pytest.param(
        assembly_input_specs() + ((trt.float16, (1, 3, 8)),), id="extra-input"
    ),
)


def test_static_contract_harness_accepts_valid_output_assembly(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    built_engine = build_engine(creator, assembly_input_specs())
    assert built_engine is not None
    _, engine = built_engine
    assert tuple(engine.get_tensor_shape("output")) == (1, 3, ENCODER_DIMS[3])


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_output_assembly_plugin_rejects_invalid_contracts(
    plugin_creator: PluginCreatorFixture, input_specs: tuple[InputSpec, ...]
) -> None:
    _, creator = plugin_creator
    assert build_engine(creator, input_specs) is None


@pytest.mark.parametrize("invalid_endpoint", range(3), ids=("min", "opt", "max"))
@pytest.mark.parametrize("input_index", range(6), ids=lambda i: f"encoder-{i + 1}")
@pytest.mark.parametrize("axis", (0, 1), ids=("batch", "time"))
def test_output_assembly_plugin_rejects_invalid_profile_endpoints(
    plugin_creator: PluginCreatorFixture,
    invalid_endpoint: int,
    input_index: int,
    axis: int,
) -> None:
    _, creator = plugin_creator
    input_specs = tuple((trt.float16, (-1, -1, channels)) for channels in ENCODER_DIMS)
    invalid_value = ((2, 4), (1, 18), (2, 64))[invalid_endpoint][axis]
    profiles = []
    for index, channels in enumerate(ENCODER_DIMS):
        shapes = [[1, 3, channels], [2, 17, channels], [3, 65, channels]]
        if index == input_index:
            shapes[invalid_endpoint][axis] = invalid_value
        profiles.append(tuple(map(tuple, shapes)))

    assert build_engine(creator, input_specs, tuple(profiles)) is None
