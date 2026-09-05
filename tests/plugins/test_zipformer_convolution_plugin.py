#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer convolution plugin."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

pytestmark = [pytest.mark.cuda, pytest.mark.sm80]

PLUGIN_NAME = "zipformer_convolution"
PLUGIN_VERSION = "1"
CHANNELS = 16
KERNEL_SIZE = 31
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
FP32_TOLERANCE = 2e-5
FP16_TOLERANCE = 3e-3
BF16_TOLERANCE = 3e-2
SHAPE_CASES = (
    pytest.param((1, 1, CHANNELS), (1,), id="one-frame"),
    pytest.param((1, 2, CHANNELS), (2,), id="two-frames"),
    pytest.param((1, 3, CHANNELS), (2,), id="three-frames"),
    pytest.param((1, 4, CHANNELS), (3,), id="four-frames"),
    pytest.param((2, 7, CHANNELS), (0, 1), id="zero-and-one-valid-frame"),
    pytest.param((2, 17, CHANNELS), (17, 5), id="mixed-lengths"),
    pytest.param(
        (3, 65, CHANNELS),
        (INT32_MIN, 34, INT32_MAX),
        id="clamped-length-extremes",
    ),
    pytest.param((1, 514, CHANNELS), (513,), id="thread-block-boundary"),
)

type CuPyDType = type[np.generic] | np.dtype[np.generic]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


@dataclass(frozen=True)
class EngineCase:
    """Numeric type, channel layout, kernel size, and comparison tolerance."""

    name: str
    trt_dtype: trt.DataType
    cupy_dtype: CuPyDType
    channels: int
    kernel_size: int
    tolerance: float


ENGINE_CASES = (
    EngineCase("fp32", trt.float32, cp.float32, CHANNELS, KERNEL_SIZE, FP32_TOLERANCE),
    EngineCase("fp16", trt.float16, cp.float16, CHANNELS, KERNEL_SIZE, FP16_TOLERANCE),
    EngineCase(
        "bf16",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        CHANNELS,
        KERNEL_SIZE,
        BF16_TOLERANCE,
    ),
)
ADDITIONAL_LAYOUT_CASES = (
    EngineCase("fp32-k15", trt.float32, cp.float32, 12, 15, FP32_TOLERANCE),
    EngineCase("fp16-k15", trt.float16, cp.float16, 24, 15, FP16_TOLERANCE),
    EngineCase("bf16-k15", trt.bfloat16, cp.dtype("bfloat16"), 24, 15, BF16_TOLERANCE),
    EngineCase(
        "fp16-four-channel-boundary", trt.float16, cp.float16, 4, 15, FP16_TOLERANCE
    ),
    EngineCase(
        "bf16-four-channel-boundary",
        trt.bfloat16,
        cp.dtype("bfloat16"),
        4,
        15,
        BF16_TOLERANCE,
    ),
    EngineCase("fp16-pair-fallback", trt.float16, cp.float16, 6, 15, FP16_TOLERANCE),
    EngineCase(
        "bf16-pair-fallback", trt.bfloat16, cp.dtype("bfloat16"), 6, 15, BF16_TOLERANCE
    ),
    EngineCase("fp32-vector-boundary", trt.float32, cp.float32, 4, 1, FP32_TOLERANCE),
    EngineCase("fp16-vector-boundary", trt.float16, cp.float16, 2, 1, FP16_TOLERANCE),
    EngineCase(
        "bf16-vector-boundary", trt.bfloat16, cp.dtype("bfloat16"), 2, 1, BF16_TOLERANCE
    ),
)
INPUT_NAMES = ("x", "valid_lengths", "weight", "bias")
VALID_INPUT_SPECS = {
    "x": (trt.float32, (1, 17, CHANNELS)),
    "valid_lengths": (trt.int32, (1,)),
    "weight": (trt.float32, (KERNEL_SIZE, CHANNELS)),
    "bias": (trt.float32, (CHANNELS,)),
}


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Register the plugin after a library with colliding historical symbols.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for isolated, module-scoped compiled libraries.

    Returns
    -------
    tuple
        Output-assembly library, convolution library, and registered creator.
    """

    output_assembly_library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_output_assembly_plugin.cu",
        "initFastGpuAsrZipformerOutputAssemblyPlugin",
        ("cudart",),
    )
    library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_convolution_plugin.cu",
        "initFastGpuAsrZipformerConvolutionPlugin",
        ("cudart",),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return output_assembly_library, library, creator


def make_plugin(creator: trt.IPluginCreatorV3One) -> trt.IPluginV3:
    """Create a field-free Zipformer convolution plugin.

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
    input_specs: dict[str, InputSpec],
    profiles: dict[str, tuple[tuple[int, ...], ...]] | None = None,
    constants: dict[str, np.typing.NDArray] | None = None,
) -> tuple[trt.Runtime, trt.ICudaEngine] | None:
    """Build and deserialize a contract, returning None when TensorRT rejects it.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : dict[str, InputSpec]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.
    profiles : dict[str, tuple[tuple[int, ...], ...]] | None
        Mapping of input names to min/opt/max shapes; empty means static.
    constants : dict or None
        Host weight/bias arrays to fold into the graph instead of runtime inputs.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine] | None
        Deserialized engine with its owning runtime, or None on build rejection.
    """

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    constants = constants or {}
    inputs = []
    for name, (dtype, shape) in input_specs.items():
        if name in constants:
            values = constants[name]
            # Explicit storage types preserve BF16, which NumPy cannot identify for TRT.
            weights = trt.Weights(dtype, values.ctypes.data, values.size)
            layer = network.add_constant(shape, weights)
            assert layer is not None
            tensor = layer.get_output(0)
        else:
            tensor = network.add_input(name, dtype, shape)
        assert tensor is not None
        inputs.append(tensor)

    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    if layer is None:
        return None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 3
    if profiles:
        profile = builder.create_optimization_profile()
        for name, shapes in profiles.items():
            profile.set_shape(name, *shapes)
            assert tuple(map(tuple, profile.get_shape(name))) == shapes
        assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        name: (trt.TensorIOMode.INPUT, dtype)
        for name, (dtype, _) in input_specs.items()
        if name not in constants
    }
    expected_io["output"] = (trt.TensorIOMode.OUTPUT, input_specs["x"][0])
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return runtime, engine


def build_convolution_engine(
    creator: trt.IPluginCreatorV3One,
    case: EngineCase,
    frames: tuple[int, int, int] = (1, 17, 514),
    length_batches: tuple[int, int, int] = (1, 2, 3),
    constants: dict[str, np.typing.NDArray] | None = None,
) -> tuple[trt.Runtime, trt.ICudaEngine] | None:
    """Build an N/T-dynamic engine with independent valid-length profile bounds.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    frames : tuple[int, int, int]
        Minimum, optimum, and maximum physical sequence lengths.
    length_batches : tuple[int, int, int]
        Independent min/opt/max batch bounds for the valid-length input.
    constants : dict or None
        Host weight/bias arrays to fold into the graph instead of runtime inputs.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine] | None
        Deserialized engine with its owning runtime, or None on build rejection.
    """

    input_specs = {
        "x": (case.trt_dtype, (-1, -1, case.channels)),
        "valid_lengths": (trt.int32, (-1,)),
        "weight": (case.trt_dtype, (case.kernel_size, case.channels)),
        "bias": (case.trt_dtype, (case.channels,)),
    }
    profiles = {
        "x": tuple(
            (batch, length, case.channels) for batch, length in enumerate(frames, 1)
        ),
        "valid_lengths": tuple((batch,) for batch in length_batches),
    }
    return build_engine(creator, input_specs, profiles, constants)


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def convolution_engine(request: pytest.FixtureRequest, plugin_creator):
    """Build one reusable dynamic engine per supported numeric dtype.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    tuple[trt.Runtime, trt.ICudaEngine, EngineCase]
        Owning runtime, reusable dynamic engine, and selected numeric/layout case.
    """

    *_, creator = plugin_creator
    result = build_convolution_engine(creator, request.param)
    assert result is not None
    runtime, engine = result
    return runtime, engine, request.param


def make_inputs(
    shape: tuple[int, int, int],
    lengths: tuple[int, ...],
    kernel_size: int = KERNEL_SIZE,
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """Create deterministic inputs with conspicuous invalid tail values.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Input dimensions in batch, time, channel order.
    lengths : tuple[int, ...]
        Declared valid frame counts, including extreme values used to test clamping.
    kernel_size : int
        Odd temporal kernel width; weights have shape (kernel_size, channels).

    Returns
    -------
    tuple of np.typing.NDArray
        FP32 NTC activations, INT32 lengths, FP32 weights, and FP32 bias.
    """

    rng = np.random.default_rng(1000 + shape[0] * 100 + shape[1])
    x = rng.normal(0.0, 0.4, shape).astype(np.float32)
    lengths_array = np.array(lengths, dtype=np.int32)
    for index, length in enumerate(lengths_array):
        x[index, max(0, min(int(length), shape[1])) :] = 10.0
    weight = rng.normal(0.0, 0.2, (kernel_size, shape[2])).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (shape[2],)).astype(np.float32)
    return x, lengths_array, weight, bias


@dataclass
class ConvolutionRun:
    """Own the buffers, context, and stream until asynchronous inference finishes."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    inputs: dict[str, cp.ndarray]
    output: cp.ndarray


def run_engine(
    engine: trt.ICudaEngine,
    engine_case: EngineCase,
    inputs: tuple[
        np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray
    ],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    execute: bool = True,
) -> ConvolutionRun:
    """Bind sentinel-filled buffers and optionally enqueue on an ordered stream.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized engine whose runtime must remain alive during execution.
    engine_case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    inputs : tuple
        Host activations, INT32 valid lengths, depthwise weights, and channel bias.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    execute : bool
        Whether to enqueue inference after binding; False permits rejection tests.

    Returns
    -------
    ConvolutionRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("x", inputs[0].shape)
    assert context.set_input_shape("valid_lengths", inputs[1].shape)
    output_shape = tuple(context.get_tensor_shape("output"))
    assert output_shape == inputs[0].shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    input_names = {
        engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
    }
    with stream:
        buffers = {
            name: cp.array(
                values,
                dtype=cp.int32 if name == "valid_lengths" else engine_case.cupy_dtype,
            )
            for name, values in zip(INPUT_NAMES, inputs, strict=True)
            if name in input_names
        }
        output = cp.full(output_shape, cp.nan, dtype=engine_case.cupy_dtype)
        for name, value in buffers.items():
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        if execute:
            assert context.execute_async_v3(stream.ptr)
    return ConvolutionRun(context, stream, buffers, output)


def quantize_input(
    values: np.typing.NDArray, cupy_dtype: CuPyDType
) -> np.typing.NDArray:
    """Round FP32 values with a CPU oracle for the engine storage dtype.

    Parameters
    ----------
    values : np.typing.NDArray
        Input values to round or copy without modifying the original array.
    cupy_dtype : CuPyDType
        Engine storage dtype whose rounding is reproduced on the CPU.

    Returns
    -------
    np.typing.NDArray
        Independent FP32 host values rounded to the requested storage precision.
    """

    storage_dtype = cp.dtype(cupy_dtype)
    if storage_dtype == cp.dtype(cp.float32):
        return np.array(values, dtype=np.float32, copy=True)
    if storage_dtype == cp.dtype(cp.float16):
        return values.astype(np.float16).astype(np.float32)
    if storage_dtype == cp.dtype("bfloat16"):
        return torch.from_numpy(values).to(torch.bfloat16).float().numpy()
    raise AssertionError(f"Unsupported test dtype: {storage_dtype}")


def reference_convolution(
    x: np.typing.NDArray,
    valid_lengths: np.typing.NDArray,
    weight: np.typing.NDArray,
    bias: np.typing.NDArray,
) -> np.typing.NDArray:
    """Evaluate masked depthwise convolution and Swoosh-R in FP32.

    Parameters
    ----------
    x : np.typing.NDArray
        FP32 activations with shape (batch, time, channels).
    valid_lengths : np.typing.NDArray
        Declared valid frame counts; negative and oversized values exercise
        clamping.
    weight : np.typing.NDArray
        FP32 weights with shape (kernel_size, channels).
    bias : np.typing.NDArray
        FP32 channel bias applied before Swoosh-R.

    Returns
    -------
    np.typing.NDArray
        FP32 NTC output after masked convolution and Swoosh-R.

    Notes
    -----
    Invalid input frames are zeroed before convolution. Output frames are not
    remasked, so the activation of the bias remains observable in empty rows.
    The oracle accumulates in FP32; per-case tolerances allow the native packed
    kernels' FP16/BF16 accumulation rounding.
    """

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
    output = torch.nn.functional.softplus(output - 1.0) - 0.08 * output - 0.313261687
    return output.permute(0, 2, 1).numpy()


def assert_run_matches_reference(
    run: ConvolutionRun,
    inputs: tuple[
        np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray
    ],
    engine_case: EngineCase,
) -> None:
    """Check input immutability and compare with independently rounded host inputs.

    Parameters
    ----------
    run : ConvolutionRun
        Bound device buffers and the context/stream that own their pending work.
    inputs : tuple
        Host activations, INT32 valid lengths, depthwise weights, and channel bias.
    engine_case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    """

    run.stream.synchronize()
    expected_inputs = {
        name: values
        if name == "valid_lengths"
        else quantize_input(values, engine_case.cupy_dtype)
        for name, values in zip(INPUT_NAMES, inputs, strict=True)
    }
    for name, values in run.inputs.items():
        np.testing.assert_array_equal(
            cp.asnumpy(values).astype(expected_inputs[name].dtype),
            expected_inputs[name],
        )
    expected = reference_convolution(*(expected_inputs[name] for name in INPUT_NAMES))
    np.testing.assert_allclose(
        cp.asnumpy(run.output).astype(np.float32),
        expected,
        rtol=engine_case.tolerance,
        atol=engine_case.tolerance,
    )


@pytest.mark.parametrize("shape,lengths", SHAPE_CASES)
def test_zipformer_convolution_plugin_matches_reference(
    convolution_engine,
    shape: tuple[int, int, int],
    lengths: tuple[int, ...],
) -> None:
    _, engine, case = convolution_engine
    inputs = make_inputs(shape, lengths, case.kernel_size)
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


def test_zipformer_convolution_plugin_suppresses_nonfinite_padding(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    inputs = make_inputs((2, 17, case.channels), (5, 12), case.kernel_size)
    inputs[0][0, 5:] = np.nan
    inputs[0][1, 12:] = np.inf
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


def test_zipformer_convolution_plugin_does_not_remask_output_frames(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    x, lengths, weight, bias = make_inputs(
        (1, 7, case.channels), (0,), case.kernel_size
    )
    weight.fill(0.0)
    bias.fill(2.0)
    inputs = x, lengths, weight, bias
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


def test_zipformer_convolution_plugin_handles_activation_extremes(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    values = np.array((-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0), dtype=np.float32)
    x = np.broadcast_to(
        values[np.newaxis, :, np.newaxis], (1, len(values), case.channels)
    )
    weight = np.zeros((case.kernel_size, case.channels), dtype=np.float32)
    weight[case.kernel_size // 2] = 1.0
    inputs = (
        x.copy(),
        np.array((len(values),), dtype=np.int32),
        weight,
        np.zeros(case.channels, dtype=np.float32),
    )
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


@pytest.mark.parametrize("case", ADDITIONAL_LAYOUT_CASES, ids=lambda case: case.name)
def test_zipformer_convolution_plugin_supports_additional_layouts(
    plugin_creator,
    case: EngineCase,
) -> None:
    *_, creator = plugin_creator
    result = build_convolution_engine(creator, case, (1, 129, 685))
    assert result is not None
    runtime, engine = result
    inputs = make_inputs((3, 685, case.channels), (-4, 343, 999), case.kernel_size)
    inputs[0][0] = np.nan
    inputs[0][1, 343:] = np.inf
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


def test_zipformer_convolution_plugin_reuses_context_across_shapes_and_streams(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        ((1, 1, case.channels), (1,)),
        ((3, 65, case.channels), (INT32_MIN, 34, INT32_MAX)),
        ((1, 3, case.channels), (2,)),
    )
    for index, (shape, lengths) in enumerate(shape_cases):
        stream = streams[index % len(streams)]
        inputs = make_inputs(shape, lengths, case.kernel_size)
        run = run_engine(engine, case, inputs, context, stream)
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_reference(run, inputs, case)


def test_zipformer_convolution_plugin_supports_concurrent_contexts(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    inputs = (
        make_inputs((3, 65, case.channels), (65, 34, 1), case.kernel_size),
        make_inputs((2, 17, case.channels), (5, 16), case.kernel_size),
    )
    runs = [run_engine(engine, case, values, execute=False) for values in inputs]
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr
    for run in runs:
        assert run.context.execute_async_v3(run.stream.ptr)
    for run, values in zip(runs, inputs, strict=True):
        assert_run_matches_reference(run, values, case)


def test_zipformer_convolution_plugin_rejects_runtime_batch_mismatch(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    inputs = make_inputs((2, 17, case.channels), (0,), case.kernel_size)
    run = run_engine(engine, case, inputs, execute=False)
    assert not run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert bool(cp.isnan(run.output).all())


def test_zipformer_convolution_plugin_supports_cuda_graph_replay(
    convolution_engine,
) -> None:
    _, engine, case = convolution_engine
    inputs = make_inputs((2, 17, case.channels), (17, 5), case.kernel_size)
    run = run_engine(engine, case, inputs)
    run.stream.synchronize()

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for replay in range(2):
        replay_inputs = make_inputs(
            (2, 17, case.channels), (8 + replay, 16 - replay), case.kernel_size
        )
        x, _, weight, bias = replay_inputs
        x += np.float32(0.125 * (replay + 1))
        weight *= np.float32(-1.0 + 0.25 * replay)
        bias += np.float32(0.05 * (replay + 1))
        with run.stream:
            for name, values in zip(INPUT_NAMES, replay_inputs, strict=True):
                buffer = run.inputs[name]
                cp.copyto(buffer, cp.array(values, dtype=buffer.dtype))
            run.output.fill(cp.nan)
            graph.launch(run.stream)
        assert_run_matches_reference(run, replay_inputs, case)


def test_zipformer_convolution_plugin_accepts_valid_static_contract(
    plugin_creator,
) -> None:
    *_, creator = plugin_creator
    result = build_engine(creator, VALID_INPUT_SPECS)
    assert result is not None
    runtime, engine = result
    inputs = make_inputs((1, 17, CHANNELS), (9,))
    assert_run_matches_reference(
        run_engine(engine, ENGINE_CASES[0], inputs), inputs, ENGINE_CASES[0]
    )


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({"x": (trt.float32, (1, 17))}, id="x-rank"),
        pytest.param({"x": (trt.float32, (1, 0, CHANNELS))}, id="empty-sequence"),
        pytest.param({"valid_lengths": (trt.int32, (1, 1))}, id="length-rank"),
        pytest.param(
            {"weight": (trt.float32, (1, KERNEL_SIZE, CHANNELS))}, id="weight-rank"
        ),
        pytest.param({"bias": (trt.float32, (1, CHANNELS))}, id="bias-rank"),
        pytest.param(
            {
                "x": (trt.int32, (1, 17, CHANNELS)),
                "weight": (trt.int32, (KERNEL_SIZE, CHANNELS)),
                "bias": (trt.int32, (CHANNELS,)),
            },
            id="unsupported-dtype",
        ),
        pytest.param({"valid_lengths": (trt.int64, (1,))}, id="length-dtype"),
        pytest.param(
            {"weight": (trt.float16, (KERNEL_SIZE, CHANNELS))}, id="weight-dtype"
        ),
        pytest.param({"bias": (trt.float16, (CHANNELS,))}, id="bias-dtype"),
        pytest.param({"valid_lengths": (trt.int32, (2,))}, id="batch-mismatch"),
        pytest.param(
            {"weight": (trt.float32, (KERNEL_SIZE - 1, CHANNELS))}, id="even-kernel"
        ),
        pytest.param(
            {"weight": (trt.float32, (KERNEL_SIZE, CHANNELS - 4))}, id="weight-channels"
        ),
        pytest.param({"bias": (trt.float32, (CHANNELS - 4,))}, id="bias-channels"),
    ),
)
def test_zipformer_convolution_plugin_rejects_invalid_contracts(
    plugin_creator,
    overrides: dict[str, InputSpec],
) -> None:
    *_, creator = plugin_creator
    assert build_engine(creator, VALID_INPUT_SPECS | overrides) is None


@pytest.mark.parametrize(
    "dtype,channels",
    ((trt.float32, 6), (trt.float16, 3), (trt.bfloat16, 3)),
    ids=("fp32", "fp16", "bf16"),
)
def test_zipformer_convolution_plugin_rejects_unaligned_channels(
    plugin_creator, dtype: trt.DataType, channels: int
) -> None:
    *_, creator = plugin_creator
    specs = VALID_INPUT_SPECS | {
        "x": (dtype, (1, 17, channels)),
        "weight": (dtype, (KERNEL_SIZE, channels)),
        "bias": (dtype, (channels,)),
    }
    assert build_engine(creator, specs) is None


@pytest.mark.parametrize("input_count", (3, 5))
def test_zipformer_convolution_plugin_rejects_invalid_input_count(
    plugin_creator, input_count: int
) -> None:
    *_, creator = plugin_creator
    specs = dict(VALID_INPUT_SPECS)
    if input_count == 3:
        del specs["bias"]
    else:
        specs["extra"] = (trt.float32, (1,))
    assert build_engine(creator, specs) is None


@pytest.mark.parametrize(
    "length_batches", ((2, 2, 3), (1, 1, 3), (1, 2, 2)), ids=("min", "opt", "max")
)
def test_zipformer_convolution_plugin_rejects_invalid_profile_endpoints(
    plugin_creator,
    length_batches: tuple[int, int, int],
) -> None:
    *_, creator = plugin_creator
    assert (
        build_convolution_engine(creator, ENGINE_CASES[1], (1, 17, 65), length_batches)
        is None
    )


def test_zipformer_convolution_creator_rejects_fields(plugin_creator) -> None:
    *_, creator = plugin_creator
    assert list(creator.field_names) == []
    field = trt.PluginField(
        "unexpected", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32
    )
    assert (
        creator.create_plugin(
            PLUGIN_NAME, trt.PluginFieldCollection([field]), trt.TensorRTPhase.BUILD
        )
        is None
    )


@pytest.mark.parametrize("case", ENGINE_CASES, ids=lambda case: case.name)
def test_zipformer_convolution_builds_with_constants(
    plugin_creator,
    case: EngineCase,
) -> None:
    *_, creator = plugin_creator
    rng = np.random.default_rng(20260819)
    weight = rng.normal(0.0, 0.2, (case.kernel_size, case.channels)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (case.channels,)).astype(np.float32)
    constants = {
        name: cp.asnumpy(cp.array(values, dtype=case.cupy_dtype))
        for name, values in (("weight", weight), ("bias", bias))
    }
    for name, values in (("weight", weight), ("bias", bias)):
        np.testing.assert_array_equal(
            constants[name].astype(np.float32), quantize_input(values, case.cupy_dtype)
        )
    result = build_convolution_engine(creator, case, (1, 17, 65), constants=constants)
    assert result is not None
    runtime, engine = result
    x, lengths, _, _ = make_inputs((2, 17, case.channels), (17, 5), case.kernel_size)
    inputs = x, lengths, weight, bias
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)
