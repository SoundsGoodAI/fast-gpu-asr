#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer output-assembly plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.cuda

PLUGIN_NAME = "zipformer_output_assembly"
PLUGIN_VERSION = "1"
ENCODER_DIMS = (16, 32, 48, 64, 48, 32)
FP32_VECTOR_BOUNDARY_DIMS = (13, 1, 27, 20, 12, 4)
FP16_VECTOR_BOUNDARY_DIMS = (13, 1, 27, 24, 16, 8)
MINIMAL_EQUAL_DIMS = (1, 2, 3, 8, 8, 8)
EMPTY_ENCODER4_BAND_DIMS = (1, 2, 3, 16, 16, 8)
EMPTY_ENCODER5_BAND_DIMS = (1, 2, 3, 24, 8, 8)
SHAPE_CASES = ((1, 1), (1, 3), (2, 17), (3, 65))


@dataclass(frozen=True)
class EngineCase:
    """TensorRT dtype and channel layout for one dynamic engine."""

    name: str
    trt_dtype: object
    cupy_dtype: object
    encoder_dims: tuple[int, ...]


ENGINE_CASES = (
    EngineCase("fp32", trt.float32, cp.float32, ENCODER_DIMS),
    EngineCase("fp16", trt.float16, cp.float16, ENCODER_DIMS),
    pytest.param(
        EngineCase("bf16", trt.bfloat16, cp.dtype("bfloat16"), ENCODER_DIMS),
        marks=pytest.mark.sm80,
    ),
    EngineCase(
        "fp32-vector-boundary",
        trt.float32,
        cp.float32,
        FP32_VECTOR_BOUNDARY_DIMS,
    ),
    EngineCase(
        "fp16-vector-boundary",
        trt.float16,
        cp.float16,
        FP16_VECTOR_BOUNDARY_DIMS,
    ),
    EngineCase(
        "fp16-minimal-equal",
        trt.float16,
        cp.float16,
        MINIMAL_EQUAL_DIMS,
    ),
    EngineCase(
        "fp16-empty-encoder4-band",
        trt.float16,
        cp.float16,
        EMPTY_ENCODER4_BAND_DIMS,
    ),
    EngineCase(
        "fp16-empty-encoder5-band",
        trt.float16,
        cp.float16,
        EMPTY_ENCODER5_BAND_DIMS,
    ),
)


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register the current output-assembly plugin source."""

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


def make_plugin(creator):
    """Create the field-free output-assembly plugin."""

    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def build_assembly_engine(creator, engine_case: EngineCase):
    """Build and deserialize one dynamic output-assembly engine."""

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = []
    for index, channels in enumerate(engine_case.encoder_dims):
        tensor = network.add_input(
            f"encoder_{index + 1}",
            engine_case.trt_dtype,
            (-1, -1, channels),
        )
        assert tensor is not None
        inputs.append(tensor)

    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    for index, channels in enumerate(engine_case.encoder_dims):
        profile.set_shape(
            f"encoder_{index + 1}",
            (1, 1, channels),
            (2, 17, channels),
            (3, 65, channels),
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    assert engine.num_io_tensors == 7
    for index in range(6):
        assert engine.get_tensor_mode(f"encoder_{index + 1}") == trt.TensorIOMode.INPUT
    assert engine.get_tensor_mode("output") == trt.TensorIOMode.OUTPUT
    assert engine.get_tensor_dtype("output") == engine_case.trt_dtype
    return runtime, engine, engine_case


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def assembly_engine(request, plugin_creator):
    """Build a dynamic TensorRT engine for one supported dtype and layout."""

    _, creator = plugin_creator
    return build_assembly_engine(creator, request.param)


@pytest.fixture(scope="module")
def alignment_engine(plugin_creator):
    """Build one representative engine for misaligned-pointer checks."""

    _, creator = plugin_creator
    return build_assembly_engine(
        creator,
        EngineCase("fp16-alignment", trt.float16, cp.float16, ENCODER_DIMS),
    )


def make_inputs(
    batch_size: int,
    sequence_length: int,
    encoder_dims: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic, distinct values for all six encoder stacks."""

    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal(size=(batch_size, sequence_length, channels)).astype(np.float32)
        for channels in encoder_dims
    )


def expected_assembly(
    inputs: tuple[cp.ndarray, ...], encoder_dims: tuple[int, ...]
) -> cp.ndarray:
    """Assemble the three surviving channel bands directly."""

    return cp.concatenate(
        (
            inputs[5],
            inputs[4][:, :, encoder_dims[5] :],
            inputs[3][:, :, encoder_dims[4] :],
        ),
        axis=2,
    )


@dataclass
class EngineRun:
    """Execution state and buffers retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    inputs: tuple[cp.ndarray, ...]
    output: cp.ndarray


def run_engine(
    engine,
    engine_case: EngineCase,
    host_inputs: tuple[np.ndarray, ...],
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
    synchronize: bool = True,
) -> EngineRun:
    """Copy inputs and execute in order on one non-default CUDA stream."""

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert len(host_inputs) == 6
    for index, values in enumerate(host_inputs):
        assert context.set_input_shape(f"encoder_{index + 1}", values.shape)
    expected_output_shape = (
        host_inputs[3].shape[0],
        host_inputs[3].shape[1],
        engine_case.encoder_dims[3],
    )
    assert tuple(context.get_tensor_shape("output")) == expected_output_shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_inputs = tuple(
            cp.asarray(values, dtype=engine_case.cupy_dtype) for values in host_inputs
        )
        output = cp.full(
            expected_output_shape,
            cp.nan,
            dtype=engine_case.cupy_dtype,
        )
        for index, values in enumerate(device_inputs):
            assert context.set_tensor_address(f"encoder_{index + 1}", values.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        assert context.execute_async_v3(stream.ptr)
    if synchronize:
        stream.synchronize()
    return EngineRun(context, stream, device_inputs, output)


def assert_run_matches_reference(run: EngineRun, engine_case: EngineCase) -> None:
    """Compare one completed run against direct channel-band assembly."""

    run.stream.synchronize()
    cp.testing.assert_array_equal(
        run.output,
        expected_assembly(run.inputs, engine_case.encoder_dims),
    )


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_output_assembly_plugin_matches_reference(
    assembly_engine, batch_size: int, sequence_length: int
) -> None:
    """Compare dynamic FP32, FP16, and BF16 output against direct assembly."""

    _, engine, engine_case = assembly_engine
    host_inputs = make_inputs(
        batch_size,
        sequence_length,
        engine_case.encoder_dims,
        seed=1000 + batch_size * 100 + sequence_length,
    )
    run = run_engine(engine, engine_case, host_inputs)

    assert_run_matches_reference(run, engine_case)


def test_output_assembly_plugin_supports_cuda_graphs(assembly_engine) -> None:
    """Capture and replay output assembly on a non-default CUDA stream."""

    _, engine, engine_case = assembly_engine
    run = run_engine(
        engine,
        engine_case,
        make_inputs(2, 17, engine_case.encoder_dims, seed=2000),
    )

    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)
    with run.stream:
        run.inputs[3].fill(4)
        run.inputs[4].fill(5)
        run.inputs[5].fill(6)
        run.output.fill(cp.nan)
        graph.launch(run.stream)

    assert_run_matches_reference(run, engine_case)


def test_output_assembly_plugin_reuses_context_across_dynamic_shapes(
    assembly_engine,
) -> None:
    """Reuse one execution context while the dynamic N/T dimensions change."""

    _, engine, engine_case = assembly_engine
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)

    for iteration, (batch_size, sequence_length) in enumerate(
        (SHAPE_CASES[0], SHAPE_CASES[-1], SHAPE_CASES[0])
    ):
        run = run_engine(
            engine,
            engine_case,
            make_inputs(
                batch_size,
                sequence_length,
                engine_case.encoder_dims,
                seed=3000 + iteration,
            ),
            context=context,
            stream=stream,
        )

        assert run.context is context
        assert_run_matches_reference(run, engine_case)


def test_output_assembly_plugin_ignores_dependency_values(assembly_engine) -> None:
    """Keep stacks 1-3 as graph dependencies without reading their values."""

    _, engine, engine_case = assembly_engine
    host_inputs = tuple(
        np.full((1, 3, channels), index + 1, dtype=np.float32)
        for index, channels in enumerate(engine_case.encoder_dims)
    )
    run = run_engine(engine, engine_case, host_inputs)
    assert_run_matches_reference(run, engine_case)
    baseline = run.output.get()

    with run.stream:
        for values in run.inputs[:3]:
            values.fill(100)
        run.output.fill(cp.nan)
        assert run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    cp.testing.assert_array_equal(run.output, baseline)


def test_output_assembly_plugin_rejects_runtime_shape_mismatch(
    alignment_engine,
) -> None:
    """Reject concrete input shapes that disagree inside a valid profile."""

    _, engine, engine_case = alignment_engine
    host_inputs = list(make_inputs(2, 3, engine_case.encoder_dims, seed=3500))
    host_inputs[0] = host_inputs[0][:1]
    context = engine.create_execution_context()
    assert context is not None
    for index, values in enumerate(host_inputs):
        assert context.set_input_shape(f"encoder_{index + 1}", values.shape)
    output_shape = (2, 3, engine_case.encoder_dims[3])
    assert tuple(context.get_tensor_shape("output")) == output_shape
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        device_inputs = tuple(
            cp.asarray(values, dtype=engine_case.cupy_dtype) for values in host_inputs
        )
        output = cp.full(output_shape, cp.nan, dtype=engine_case.cupy_dtype)
        for index, values in enumerate(device_inputs):
            assert context.set_tensor_address(f"encoder_{index + 1}", values.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    assert bool(cp.isnan(output).all())


@pytest.mark.parametrize(
    "misaligned_binding",
    ("encoder_4", "encoder_5", "encoder_6", "output"),
)
def test_output_assembly_plugin_rejects_misaligned_bindings(
    alignment_engine,
    misaligned_binding: str,
) -> None:
    """Reject source and destination pointers that cannot hold aligned uint4 values."""

    _, engine, engine_case = alignment_engine
    host_inputs = make_inputs(1, 3, engine_case.encoder_dims, seed=4000)
    context = engine.create_execution_context()
    assert context is not None
    for index, values in enumerate(host_inputs):
        assert context.set_input_shape(f"encoder_{index + 1}", values.shape)
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        device_inputs = [
            cp.asarray(values, dtype=engine_case.cupy_dtype) for values in host_inputs
        ]
        input_backing = None
        if misaligned_binding.startswith("encoder_"):
            input_index = int(misaligned_binding.removeprefix("encoder_")) - 1
            input_backing = cp.empty(
                device_inputs[input_index].size + 1,
                dtype=engine_case.cupy_dtype,
            )
            misaligned_input = input_backing[1:].reshape(
                device_inputs[input_index].shape
            )
            cp.copyto(misaligned_input, device_inputs[input_index])
            device_inputs[input_index] = misaligned_input

        output_shape = (1, 3, engine_case.encoder_dims[3])
        output_backing = None
        if misaligned_binding == "output":
            output_backing = cp.empty(
                int(np.prod(output_shape)) + 1,
                dtype=engine_case.cupy_dtype,
            )
            output = output_backing[1:].reshape(output_shape)
            output.fill(cp.nan)
        else:
            output = cp.full(output_shape, cp.nan, dtype=engine_case.cupy_dtype)

        for index, values in enumerate(device_inputs):
            assert context.set_tensor_address(f"encoder_{index + 1}", values.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        assert context.get_tensor_address(misaligned_binding) % 16 != 0
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    assert bool(cp.isnan(output).all())


def test_output_assembly_creator_rejects_unexpected_fields(plugin_creator) -> None:
    """Reject attributes because output assembly has no serialized state."""

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
    *,
    encoder_dims: tuple[int, ...] = ENCODER_DIMS,
    batch_sizes: tuple[int, ...] = (1,) * 6,
    sequence_lengths: tuple[int, ...] = (3,) * 6,
    dtypes: tuple[object, ...] | None = None,
    shape_overrides: dict[int, tuple[int, ...]] | None = None,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Create one static six-input TensorRT contract."""

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
    pytest.param(
        assembly_input_specs(shape_overrides={0: (1, 3)}),
        id="dependency-rank",
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
            encoder_dims=(16, 32, 48, 18, 12, 4),
            dtypes=(trt.float32,) * 6,
        ),
        id="unaligned-output-fp32",
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
        assembly_input_specs(batch_sizes=(2, 1, 1, 1, 1, 1)),
        id="dependency-batch",
    ),
    pytest.param(
        assembly_input_specs(batch_sizes=(1, 1, 1, 2, 1, 1)),
        id="contributor-batch",
    ),
    pytest.param(
        assembly_input_specs(sequence_lengths=(3, 4, 3, 3, 3, 3)),
        id="dependency-time",
    ),
    pytest.param(
        assembly_input_specs(sequence_lengths=(3, 3, 3, 4, 3, 3)),
        id="contributor-time",
    ),
    pytest.param(
        assembly_input_specs(dtypes=(trt.int32,) * 6),
        id="unsupported-dtype",
    ),
    pytest.param(
        assembly_input_specs(
            dtypes=(trt.float16,) * 5 + (trt.float32,),
        ),
        id="mixed-dtypes",
    ),
    pytest.param(
        assembly_input_specs()[:5],
        id="missing-input",
    ),
    pytest.param(
        assembly_input_specs() + ((trt.float16, (1, 3, 8)),),
        id="extra-input",
    ),
)


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_output_assembly_plugin_rejects_invalid_contracts(
    plugin_creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid counts, ranks, dimensions, channel layouts, and dtypes."""

    _, creator = plugin_creator
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
        return
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert builder.build_serialized_network(network, config) is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "max"))
def test_output_assembly_plugin_rejects_invalid_profile_endpoints(
    plugin_creator,
    invalid_endpoint: str,
) -> None:
    """Validate dependency shapes at every dynamic optimization-profile endpoint."""

    _, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = [
        network.add_input(
            f"encoder_{index + 1}",
            trt.float16,
            (-1, -1, channels),
        )
        for index, channels in enumerate(ENCODER_DIMS)
    ]
    assert all(tensor is not None for tensor in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    for index, channels in enumerate(ENCODER_DIMS):
        minimum_batch = 2 if invalid_endpoint == "min" and index == 0 else 1
        maximum_batch = 2 if invalid_endpoint == "max" and index == 0 else 3
        profile.set_shape(
            f"encoder_{index + 1}",
            (minimum_batch, 3, channels),
            (2, 17, channels),
            (maximum_batch, 65, channels),
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None
