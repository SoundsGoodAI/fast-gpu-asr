#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet convolution plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = [pytest.mark.cuda, pytest.mark.sm80]

PLUGIN_NAME = "parakeet_conformer_convolution"
PLUGIN_VERSION = "1"
CHANNELS = 16
KERNEL_SIZE = 9
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
SHAPE_CASES = (
    pytest.param((1, 1, CHANNELS), (1,), id="one-frame"),
    pytest.param((1, 2, CHANNELS), (2,), id="two-frames"),
    pytest.param((1, 3, CHANNELS), (2,), id="three-frames"),
    pytest.param((1, 4, CHANNELS), (3,), id="four-frames"),
    pytest.param((2, 17, CHANNELS), (17, 5), id="mixed-lengths"),
    pytest.param(
        (3, 65, CHANNELS),
        (INT32_MIN, 34, INT32_MAX),
        id="clamped-length-extremes",
    ),
    pytest.param((1, 258, CHANNELS), (257,), id="thread-block-boundary"),
)


@dataclass(frozen=True)
class EngineCase:
    """Numeric type, channel layout, kernel size, and comparison tolerance."""

    name: str
    trt_dtype: object
    cupy_dtype: object
    channels: int
    kernel_size: int
    tolerance: float


ENGINE_CASES = (
    EngineCase("fp32", trt.float32, cp.float32, CHANNELS, KERNEL_SIZE, 2e-5),
    EngineCase("fp16", trt.float16, cp.float16, CHANNELS, KERNEL_SIZE, 2e-2),
    EngineCase(
        "bf16",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        CHANNELS,
        KERNEL_SIZE,
        8e-2,
    ),
)
ADDITIONAL_LAYOUT_CASES = (
    EngineCase("fp32-vector-boundary", trt.float32, cp.float32, 4, 1, 2e-5),
    EngineCase("fp16-vector-boundary", trt.float16, cp.float16, 2, 1, 2e-2),
    EngineCase(
        "bf16-vector-boundary",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        2,
        1,
        8e-2,
    ),
    EngineCase("fp16-pair-fallback", trt.float16, cp.float16, 6, 15, 2e-2),
    EngineCase(
        "bf16-pair-fallback",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        6,
        15,
        8e-2,
    ),
    EngineCase("fp16-adjacent-k15", trt.float16, cp.float16, 8, 15, 2e-2),
    EngineCase(
        "bf16-adjacent-k15",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        8,
        15,
        8e-2,
    ),
)


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Register the plugin after a library with colliding historical symbols."""

    output_assembly_library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_output_assembly_plugin.cu",
        "initFastGpuAsrZipformerOutputAssemblyPlugin",
        ("cudart",),
    )
    library = compile_and_load_plugin(
        tmp_path_factory,
        "parakeet_convolution_plugin.cu",
        "initFastGpuAsrParakeetConvolutionPlugin",
        ("cudart",),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return output_assembly_library, library, creator


def make_plugin(creator):
    """Create a field-free Parakeet convolution plugin."""

    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD
    )
    assert plugin is not None
    return plugin


def build_engine(
    creator,
    dtype,
    channels: int,
    kernel_size: int,
    min_shape: tuple[int, int, int],
    opt_shape: tuple[int, int, int],
    max_shape: tuple[int, int, int],
):
    """Build one dynamic engine for a concrete dtype and channel layout."""

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    x = network.add_input("x", dtype, (-1, -1, channels))
    valid_lengths = network.add_input("valid_lengths", trt.int32, (-1,))
    weight = network.add_input("weight", dtype, (kernel_size, channels))
    bias = network.add_input("bias", dtype, (channels,))
    assert x is not None and valid_lengths is not None
    assert weight is not None and bias is not None

    layer = network.add_plugin_v3(
        [x, valid_lengths, weight, bias], [], make_plugin(creator)
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    profile.set_shape("x", min_shape, opt_shape, max_shape)
    profile.set_shape(
        "valid_lengths", (min_shape[0],), (opt_shape[0],), (max_shape[0],)
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
        "x": (trt.TensorIOMode.INPUT, dtype),
        "valid_lengths": (trt.TensorIOMode.INPUT, trt.int32),
        "weight": (trt.TensorIOMode.INPUT, dtype),
        "bias": (trt.TensorIOMode.INPUT, dtype),
        "output": (trt.TensorIOMode.OUTPUT, dtype),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, tensor_dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == tensor_dtype
    return runtime, engine


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def convolution_engine(request, plugin_creator):
    """Build one dynamic TensorRT engine for each supported numeric dtype."""

    *_, creator = plugin_creator
    engine_case = request.param
    runtime, engine = build_engine(
        creator,
        engine_case.trt_dtype,
        engine_case.channels,
        engine_case.kernel_size,
        (1, 1, engine_case.channels),
        (2, 17, engine_case.channels),
        (3, 258, engine_case.channels),
    )
    return runtime, engine, engine_case


def make_inputs(
    shape: tuple[int, int, int],
    lengths: tuple[int, ...],
    kernel_size: int = KERNEL_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic inputs with conspicuous invalid tail values."""

    rng = np.random.default_rng(1000 + shape[0] * 100 + shape[1])
    x = rng.normal(0.0, 0.4, shape).astype(np.float32)
    lengths_array = np.array(lengths, dtype=np.int32)
    for index, length in enumerate(lengths_array):
        x[index, max(0, min(int(length), shape[1])) :] = 10.0
    channels = shape[2]
    weight = rng.normal(0.0, 0.2, (kernel_size, channels)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (channels,)).astype(np.float32)
    return x, lengths_array, weight, bias


@dataclass
class ConvolutionRun:
    """Device buffers and execution state retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    x: cp.ndarray
    valid_lengths: cp.ndarray
    weight: cp.ndarray
    bias: cp.ndarray
    output: cp.ndarray


def run_engine(
    engine,
    engine_case: EngineCase,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
) -> ConvolutionRun:
    """Execute with sentinel output on one explicitly ordered CUDA stream."""

    x, valid_lengths, weight, bias = inputs
    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert x.dtype == weight.dtype == bias.dtype == np.float32
    assert valid_lengths.dtype == np.int32
    assert x.shape[2] == engine_case.channels
    assert weight.shape == (engine_case.kernel_size, engine_case.channels)
    assert bias.shape == (engine_case.channels,)
    assert valid_lengths.shape == (x.shape[0],)
    assert context.set_input_shape("x", x.shape)
    assert context.set_input_shape("valid_lengths", valid_lengths.shape)
    output_shape = tuple(context.get_tensor_shape("output"))
    assert output_shape == x.shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x_device = cp.asarray(x, dtype=engine_case.cupy_dtype)
        lengths_device = cp.asarray(valid_lengths)
        weight_device = cp.asarray(weight, dtype=engine_case.cupy_dtype)
        bias_device = cp.asarray(bias, dtype=engine_case.cupy_dtype)
        output_device = cp.full(
            output_shape,
            cp.nan,
            dtype=engine_case.cupy_dtype,
        )
        for name, value in (
            ("x", x_device),
            ("valid_lengths", lengths_device),
            ("weight", weight_device),
            ("bias", bias_device),
            ("output", output_device),
        ):
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.execute_async_v3(stream.ptr)
    return ConvolutionRun(
        context,
        stream,
        x_device,
        lengths_device,
        weight_device,
        bias_device,
        output_device,
    )


def collect_run(
    run: ConvolutionRun,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronize and copy one output and its quantized inputs to NumPy."""

    run.stream.synchronize()
    output, x, weight, bias = (
        cp.asnumpy(value).astype(np.float32)
        for value in (run.output, run.x, run.weight, run.bias)
    )
    valid_lengths = cp.asnumpy(run.valid_lengths).astype(np.int32)
    return output, x, valid_lengths, weight, bias


def reference_convolution(
    x: np.ndarray,
    valid_lengths: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Evaluate the masked, folded depthwise convolution in FP32."""

    x_tensor = torch.from_numpy(x)
    lengths_tensor = torch.from_numpy(valid_lengths).clamp(0, x.shape[1])
    invalid = torch.arange(x.shape[1]).unsqueeze(0) >= lengths_tensor.unsqueeze(1)
    x_tensor = x_tensor.masked_fill(invalid.unsqueeze(2), 0.0).permute(0, 2, 1)
    output = torch.nn.functional.conv1d(
        x_tensor,
        torch.from_numpy(weight).permute(1, 0).unsqueeze(1),
        torch.from_numpy(bias),
        padding=weight.shape[0] // 2,
        groups=x.shape[2],
    )
    return torch.nn.functional.silu(output).permute(0, 2, 1).numpy()


def assert_run_matches_reference(
    run: ConvolutionRun,
    tolerance: float,
) -> np.ndarray:
    """Compare one native run with quantization-aware PyTorch convolution."""

    actual, x, valid_lengths, weight, bias = collect_run(run)
    expected = reference_convolution(x, valid_lengths, weight, bias)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    return actual


def convolution_input_specs(
    *,
    x_shape: tuple[int, ...] = (1, 17, CHANNELS),
    length_shape: tuple[int, ...] = (1,),
    weight_shape: tuple[int, ...] = (KERNEL_SIZE, CHANNELS),
    bias_shape: tuple[int, ...] = (CHANNELS,),
    dtypes: tuple[object, ...] | None = None,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Create one static four-input TensorRT plugin contract."""

    if dtypes is None:
        dtypes = (trt.float32, trt.int32, trt.float32, trt.float32)
    return tuple(
        zip(
            dtypes,
            (x_shape, length_shape, weight_shape, bias_shape),
            strict=True,
        )
    )


INVALID_CONTRACT_CASES = (
    pytest.param(convolution_input_specs(x_shape=(1, 17)), id="x-rank"),
    pytest.param(
        convolution_input_specs(length_shape=(1, 1)),
        id="length-rank",
    ),
    pytest.param(
        convolution_input_specs(weight_shape=(1, KERNEL_SIZE, CHANNELS)),
        id="weight-rank",
    ),
    pytest.param(
        convolution_input_specs(bias_shape=(1, CHANNELS)),
        id="bias-rank",
    ),
    pytest.param(
        convolution_input_specs(dtypes=(trt.int32,) * 4),
        id="unsupported-dtype",
    ),
    pytest.param(
        convolution_input_specs(
            dtypes=(trt.float32, trt.int64, trt.float32, trt.float32)
        ),
        id="length-dtype",
    ),
    pytest.param(
        convolution_input_specs(
            dtypes=(trt.float32, trt.int32, trt.float16, trt.float32)
        ),
        id="weight-dtype",
    ),
    pytest.param(
        convolution_input_specs(
            dtypes=(trt.float32, trt.int32, trt.float32, trt.float16)
        ),
        id="bias-dtype",
    ),
    pytest.param(
        convolution_input_specs(length_shape=(2,)),
        id="batch-mismatch",
    ),
    pytest.param(
        convolution_input_specs(weight_shape=(KERNEL_SIZE - 1, CHANNELS)),
        id="even-kernel",
    ),
    pytest.param(
        convolution_input_specs(weight_shape=(KERNEL_SIZE, CHANNELS - 4)),
        id="weight-channels",
    ),
    pytest.param(
        convolution_input_specs(bias_shape=(CHANNELS - 4,)),
        id="bias-channels",
    ),
    pytest.param(
        convolution_input_specs(
            x_shape=(1, 17, 6),
            weight_shape=(KERNEL_SIZE, 6),
            bias_shape=(6,),
        ),
        id="fp32-channel-alignment",
    ),
    pytest.param(
        convolution_input_specs(
            x_shape=(1, 17, 3),
            weight_shape=(KERNEL_SIZE, 3),
            bias_shape=(3,),
            dtypes=(trt.float16, trt.int32, trt.float16, trt.float16),
        ),
        id="fp16-channel-alignment",
    ),
    pytest.param(convolution_input_specs()[:3], id="missing-input"),
    pytest.param(
        convolution_input_specs() + ((trt.float32, (1,)),),
        id="extra-input",
    ),
)


def build_static_contract(
    creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> object | None:
    """Attempt to build one static convolution-plugin contract."""

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
        return None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    return builder.build_serialized_network(network, config)


@pytest.mark.parametrize("shape,lengths", SHAPE_CASES)
def test_parakeet_convolution_plugin_matches_reference(
    convolution_engine,
    shape: tuple[int, int, int],
    lengths: tuple[int, ...],
) -> None:
    """Compare dynamic masked convolution across all supported dtypes."""

    _, engine, engine_case = convolution_engine
    inputs = make_inputs(shape, lengths)
    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs),
        engine_case.tolerance,
    )


@pytest.mark.parametrize(
    "engine_case",
    ADDITIONAL_LAYOUT_CASES,
    ids=lambda case: case.name,
)
def test_parakeet_convolution_plugin_supports_additional_layouts(
    plugin_creator,
    engine_case: EngineCase,
) -> None:
    """Cover vector boundaries, both low-precision tactics, and odd kernels."""

    *_, creator = plugin_creator
    shape = (3, 17, engine_case.channels)
    _, engine = build_engine(
        creator,
        engine_case.trt_dtype,
        engine_case.channels,
        engine_case.kernel_size,
        (1, 1, engine_case.channels),
        (2, 9, engine_case.channels),
        shape,
    )
    inputs = make_inputs(shape, (-4, 9, 99), engine_case.kernel_size)
    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs),
        engine_case.tolerance,
    )


def test_parakeet_convolution_plugin_reuses_context_across_shapes_and_streams(
    convolution_engine,
) -> None:
    """Reuse one context while changing N/T dimensions and CUDA streams."""

    _, engine, engine_case = convolution_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        ((1, 1, engine_case.channels), (1,)),
        (
            (3, 65, engine_case.channels),
            (INT32_MIN, 34, INT32_MAX),
        ),
        ((1, 3, engine_case.channels), (2,)),
    )

    for index, (shape, lengths) in enumerate(shape_cases):
        stream = streams[index % len(streams)]
        run = run_engine(
            engine,
            engine_case,
            make_inputs(shape, lengths, engine_case.kernel_size),
            context=context,
            stream=stream,
        )
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_reference(run, engine_case.tolerance)


def test_parakeet_convolution_plugin_rejects_runtime_batch_mismatch(
    convolution_engine,
) -> None:
    """Reject concrete shapes that disagree inside a valid dynamic profile."""

    _, engine, engine_case = convolution_engine
    context = engine.create_execution_context()
    assert context is not None
    x_shape = (2, 17, engine_case.channels)
    assert context.set_input_shape("x", x_shape)
    assert context.set_input_shape("valid_lengths", (1,))
    output_shape = tuple(context.get_tensor_shape("output"))
    assert output_shape == x_shape
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:
        x = cp.zeros(x_shape, dtype=engine_case.cupy_dtype)
        valid_lengths = cp.zeros((1,), dtype=cp.int32)
        weight = cp.zeros(
            (engine_case.kernel_size, engine_case.channels),
            dtype=engine_case.cupy_dtype,
        )
        bias = cp.zeros(
            (engine_case.channels,),
            dtype=engine_case.cupy_dtype,
        )
        output = cp.full(output_shape, cp.nan, dtype=engine_case.cupy_dtype)
        for name, value in (
            ("x", x),
            ("valid_lengths", valid_lengths),
            ("weight", weight),
            ("bias", bias),
            ("output", output),
        ):
            assert context.set_tensor_address(name, value.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    assert bool(cp.isnan(output).all())


def test_parakeet_convolution_plugin_supports_cuda_graph_replay(
    convolution_engine,
) -> None:
    """Replay changed inputs accurately on a captured non-default stream."""

    _, engine, engine_case = convolution_engine
    inputs = make_inputs((2, 17, CHANNELS), (17, 5))
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    run = run_engine(
        engine,
        engine_case,
        inputs,
        context=context,
        stream=stream,
    )
    run.stream.synchronize()

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for replay in range(2):
        replay_inputs = make_inputs(
            (2, 17, CHANNELS),
            (8 + replay, 16 - replay),
            KERNEL_SIZE,
        )
        replay_x, replay_lengths, replay_weight, replay_bias = replay_inputs
        replay_x += np.float32(0.125 * (replay + 1))
        replay_weight *= np.float32(-1.0 + 0.25 * replay)
        replay_bias += np.float32(0.05 * (replay + 1))
        with run.stream:
            cp.copyto(
                run.x,
                cp.asarray(replay_x, dtype=engine_case.cupy_dtype),
            )
            cp.copyto(run.valid_lengths, cp.asarray(replay_lengths))
            cp.copyto(
                run.weight,
                cp.asarray(replay_weight, dtype=engine_case.cupy_dtype),
            )
            cp.copyto(
                run.bias,
                cp.asarray(replay_bias, dtype=engine_case.cupy_dtype),
            )
            run.output.fill(cp.nan)
            graph.launch(run.stream)

        assert_run_matches_reference(run, engine_case.tolerance)


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_parakeet_convolution_plugin_rejects_invalid_contracts(
    plugin_creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid counts, ranks, dtypes, shapes, and channel layouts."""

    *_, creator = plugin_creator
    assert build_static_contract(creator, input_specs) is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "opt", "max"))
def test_parakeet_convolution_plugin_rejects_invalid_profile_endpoints(
    plugin_creator,
    invalid_endpoint: str,
) -> None:
    """Validate cross-input batch shapes at every dynamic profile endpoint."""

    *_, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    x = network.add_input("x", trt.float16, (-1, -1, CHANNELS))
    valid_lengths = network.add_input("valid_lengths", trt.int32, (-1,))
    weight = network.add_input(
        "weight",
        trt.float16,
        (KERNEL_SIZE, CHANNELS),
    )
    bias = network.add_input("bias", trt.float16, (CHANNELS,))
    assert all(tensor is not None for tensor in (x, valid_lengths, weight, bias))
    layer = network.add_plugin_v3(
        [x, valid_lengths, weight, bias],
        [],
        make_plugin(creator),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    length_batches = {
        "min": (2, 2, 3),
        "opt": (1, 1, 3),
        "max": (1, 2, 2),
    }[invalid_endpoint]
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "x",
        (1, 1, CHANNELS),
        (2, 17, CHANNELS),
        (3, 65, CHANNELS),
    )
    profile.set_shape(
        "valid_lengths",
        *((batch_size,) for batch_size in length_batches),
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


def test_parakeet_convolution_creator_rejects_fields(plugin_creator) -> None:
    """Reject unexpected serialized fields for the field-free plugin."""

    *_, creator = plugin_creator
    field = trt.PluginField(
        "unexpected", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def test_parakeet_convolution_builds_with_folded_constants(plugin_creator) -> None:
    """Build the production layout with embedded FP16 weight and bias tensors."""

    *_, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    x = network.add_input("constant_x", trt.float16, (-1, -1, CHANNELS))
    valid_lengths = network.add_input("constant_lengths", trt.int32, (-1,))
    rng = np.random.default_rng(20260819)
    weight_values = rng.normal(0.0, 0.2, (KERNEL_SIZE, CHANNELS)).astype(np.float16)
    bias_values = rng.normal(0.0, 0.1, (CHANNELS,)).astype(np.float16)
    weight_layer = network.add_constant(weight_values.shape, weight_values)
    bias_layer = network.add_constant(bias_values.shape, bias_values)
    assert x is not None and valid_lengths is not None
    assert weight_layer is not None and bias_layer is not None

    layer = network.add_plugin_v3(
        [
            x,
            valid_lengths,
            weight_layer.get_output(0),
            bias_layer.get_output(0),
        ],
        [],
        make_plugin(creator),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "constant_output"
    network.mark_output(output)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "constant_x", (1, 1, CHANNELS), (2, 17, CHANNELS), (3, 65, CHANNELS)
    )
    profile.set_shape("constant_lengths", (1,), (2,), (3,))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        "constant_x": (trt.TensorIOMode.INPUT, trt.float16),
        "constant_lengths": (trt.TensorIOMode.INPUT, trt.int32),
        "constant_output": (trt.TensorIOMode.OUTPUT, trt.float16),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype

    x_values, lengths, _, _ = make_inputs((2, 17, CHANNELS), (17, 5))
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("constant_x", x_values.shape)
    assert context.set_input_shape("constant_lengths", lengths.shape)
    output_shape = tuple(context.get_tensor_shape("constant_output"))
    assert output_shape == x_values.shape
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x_device = cp.asarray(x_values, dtype=cp.float16)
        lengths_device = cp.asarray(lengths)
        output_device = cp.full(output_shape, cp.nan, dtype=cp.float16)
        for name, value in (
            ("constant_x", x_device),
            ("constant_lengths", lengths_device),
            ("constant_output", output_device),
        ):
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.execute_async_v3(stream.ptr)
    stream.synchronize()

    actual = cp.asnumpy(output_device).astype(np.float32)
    expected = reference_convolution(
        cp.asnumpy(x_device).astype(np.float32),
        lengths,
        weight_values.astype(np.float32),
        bias_values.astype(np.float32),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
