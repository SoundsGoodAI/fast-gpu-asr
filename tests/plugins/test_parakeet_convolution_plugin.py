#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet convolution plugin."""

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

PLUGIN_NAME = "parakeet_conformer_convolution"
PLUGIN_VERSION = "1"
CHANNELS = 16
KERNEL_SIZE = 9
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
FP32_TOLERANCE = 2e-5
FP16_TOLERANCE = 3e-3
BF16_TOLERANCE = 3e-2
INPUT_NAMES = ("x", "valid_lengths", "weight", "bias")
SHAPE_CASES = (
    pytest.param((1, 1, CHANNELS), (1,), id="one-frame"),
    pytest.param((1, 2, CHANNELS), (2,), id="two-frames"),
    pytest.param((1, 3, CHANNELS), (2,), id="three-frames"),
    pytest.param((1, 4, CHANNELS), (3,), id="four-frames"),
    pytest.param((2, 17, CHANNELS), (17, 5), id="mixed-lengths"),
    pytest.param(
        (3, 65, CHANNELS), (INT32_MIN, 34, INT32_MAX), id="clamped-length-extremes"
    ),
    pytest.param((1, 258, CHANNELS), (257,), id="thread-block-boundary"),
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
    EngineCase("fp32-vector-boundary", trt.float32, cp.float32, 4, 1, FP32_TOLERANCE),
    EngineCase("fp16-vector-boundary", trt.float16, cp.float16, 2, 1, FP16_TOLERANCE),
    EngineCase(
        "bf16-vector-boundary", trt.bfloat16, cp.dtype("bfloat16"), 2, 1, BF16_TOLERANCE
    ),
    EngineCase("fp16-pair-fallback", trt.float16, cp.float16, 6, 15, FP16_TOLERANCE),
    EngineCase(
        "bf16-pair-fallback", trt.bfloat16, cp.dtype("bfloat16"), 6, 15, BF16_TOLERANCE
    ),
    EngineCase("fp16-adjacent-k15", trt.float16, cp.float16, 8, 15, FP16_TOLERANCE),
    EngineCase(
        "bf16-adjacent-k15", trt.bfloat16, cp.dtype("bfloat16"), 8, 15, BF16_TOLERANCE
    ),
)


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


def make_plugin(creator: trt.IPluginCreatorV3One) -> trt.IPluginV3:
    """Create a field-free Parakeet convolution plugin.

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
    case: EngineCase,
    max_frames: int = 258,
    length_batches: tuple[int, int, int] = (1, 2, 3),
    constants: dict[str, np.typing.NDArray] | None = None,
) -> tuple[trt.Runtime, trt.ICudaEngine] | None:
    """Build a dynamic engine with weight and bias as inputs or constants.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    max_frames : int
        Maximum physical sequence length in the dynamic profile.
    length_batches : tuple[int, int, int]
        Independent min/opt/max batch bounds for the valid-length input.
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
    input_specs = {
        "x": (case.trt_dtype, (-1, -1, case.channels)),
        "valid_lengths": (trt.int32, (-1,)),
        "weight": (case.trt_dtype, (case.kernel_size, case.channels)),
        "bias": (case.trt_dtype, (case.channels,)),
    }
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
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    profiles = {
        "x": (
            (1, 1, case.channels),
            (2, 17, case.channels),
            (3, max_frames, case.channels),
        ),
        "valid_lengths": tuple((batch,) for batch in length_batches),
    }
    profile = builder.create_optimization_profile()
    for name, shapes in profiles.items():
        profile.set_shape(name, *shapes)
        assert tuple(map(tuple, profile.get_shape(name))) == shapes
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
        name: (trt.TensorIOMode.INPUT, dtype)
        for name, (dtype, _) in input_specs.items()
        if name not in constants
    }
    expected_io["output"] = (trt.TensorIOMode.OUTPUT, case.trt_dtype)
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return runtime, engine


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def convolution_engine(request, plugin_creator):
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
    result = build_engine(creator, request.param)
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
    channels = shape[2]
    weight = rng.normal(0.0, 0.2, (kernel_size, channels)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (channels,)).astype(np.float32)
    return x, lengths_array, weight, bias


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


@dataclass
class ConvolutionRun:
    """Own the buffers, context, and stream until asynchronous inference finishes."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    inputs: dict[str, cp.ndarray]
    output: cp.ndarray


def run_engine(
    engine: trt.ICudaEngine,
    case: EngineCase,
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
    case : EngineCase
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

    x, valid_lengths, weight, bias = inputs
    assert x.dtype == weight.dtype == bias.dtype == np.float32
    assert valid_lengths.dtype == np.int32
    assert x.shape[2] == case.channels
    assert weight.shape == (case.kernel_size, case.channels)
    assert bias.shape == (case.channels,)
    if execute:
        assert valid_lengths.shape == (x.shape[0],)
    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("x", x.shape)
    assert context.set_input_shape("valid_lengths", valid_lengths.shape)
    output_shape = tuple(context.get_tensor_shape("output"))
    assert output_shape == x.shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    input_names = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index))
        == trt.TensorIOMode.INPUT
    }
    with stream:
        buffers = {
            name: cp.asarray(
                values, dtype=cp.int32 if name == "valid_lengths" else case.cupy_dtype
            )
            for name, values in zip(INPUT_NAMES, inputs, strict=True)
            if name in input_names
        }
        output = cp.full(output_shape, cp.nan, dtype=case.cupy_dtype)
        for name, value in buffers.items():
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        if execute:
            assert context.execute_async_v3(stream.ptr)
    return ConvolutionRun(context, stream, buffers, output)


def reference_convolution(
    x: np.typing.NDArray,
    valid_lengths: np.typing.NDArray,
    weight: np.typing.NDArray,
    bias: np.typing.NDArray,
) -> np.typing.NDArray:
    """Evaluate the masked, folded depthwise convolution in FP32.

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
        FP32 channel bias, already folded with batch normalization where applicable.

    Returns
    -------
    np.typing.NDArray
        FP32 NTC output after masked convolution and SiLU.

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
    return torch.nn.functional.silu(output).permute(0, 2, 1).numpy()


def assert_run_matches_reference(
    run: ConvolutionRun,
    inputs: tuple[
        np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray
    ],
    case: EngineCase,
) -> None:
    """Check input immutability and compare with independently rounded host inputs.

    Parameters
    ----------
    run : ConvolutionRun
        Bound device buffers and the context/stream that own their pending work.
    inputs : tuple
        Host activations, INT32 valid lengths, depthwise weights, and channel bias.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    """

    run.stream.synchronize()
    expected_inputs = {
        name: values
        if name == "valid_lengths"
        else quantize_input(values, case.cupy_dtype)
        for name, values in zip(INPUT_NAMES, inputs, strict=True)
    }
    for name, values in run.inputs.items():
        np.testing.assert_array_equal(
            cp.asnumpy(values).astype(expected_inputs[name].dtype),
            expected_inputs[name],
            err_msg=name,
        )
    expected = reference_convolution(*(expected_inputs[name] for name in INPUT_NAMES))
    np.testing.assert_allclose(
        cp.asnumpy(run.output).astype(np.float32),
        expected,
        rtol=case.tolerance,
        atol=case.tolerance,
    )


def convolution_input_specs(
    x_shape: tuple[int, ...] = (1, 17, CHANNELS),
    length_shape: tuple[int, ...] = (1,),
    weight_shape: tuple[int, ...] = (KERNEL_SIZE, CHANNELS),
    bias_shape: tuple[int, ...] = (CHANNELS,),
    dtypes: tuple[trt.DataType, ...] | None = None,
) -> tuple[InputSpec, ...]:
    """Create one static four-input TensorRT plugin contract.

    Parameters
    ----------
    x_shape : tuple[int, ...]
        Activation shape; malformed ranks and dimensions are allowed for negative
        tests.
    length_shape : tuple[int, ...]
        Shape of the valid-length input.
    weight_shape : tuple[int, ...]
        Shape of the depthwise weights.
    bias_shape : tuple[int, ...]
        Shape of the channel bias.
    dtypes : tuple[trt.DataType, ...] or None
        Input dtypes in binding order; None selects this helper's default contract.

    Returns
    -------
    tuple[InputSpec, ...]
        Input-ordered dtype/shape pairs, without building or allocating an engine.
    """

    if dtypes is None:
        dtypes = (trt.float32, trt.int32, trt.float32, trt.float32)
    return tuple(
        zip(dtypes, (x_shape, length_shape, weight_shape, bias_shape), strict=True)
    )


INVALID_CONTRACT_CASES = (
    pytest.param(convolution_input_specs(x_shape=(1, 17)), id="x-rank"),
    pytest.param(convolution_input_specs(length_shape=(1, 1)), id="length-rank"),
    pytest.param(
        convolution_input_specs(weight_shape=(1, KERNEL_SIZE, CHANNELS)),
        id="weight-rank",
    ),
    pytest.param(convolution_input_specs(bias_shape=(1, CHANNELS)), id="bias-rank"),
    pytest.param(
        convolution_input_specs(dtypes=(trt.int32,) * 4), id="unsupported-dtype"
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
    pytest.param(convolution_input_specs(length_shape=(2,)), id="batch-mismatch"),
    pytest.param(
        convolution_input_specs(weight_shape=(KERNEL_SIZE - 1, CHANNELS)),
        id="even-kernel",
    ),
    pytest.param(
        convolution_input_specs(weight_shape=(KERNEL_SIZE, CHANNELS - 4)),
        id="weight-channels",
    ),
    pytest.param(
        convolution_input_specs(bias_shape=(CHANNELS - 4,)), id="bias-channels"
    ),
    pytest.param(
        convolution_input_specs(
            x_shape=(1, 17, 6), weight_shape=(KERNEL_SIZE, 6), bias_shape=(6,)
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
    pytest.param(
        convolution_input_specs(
            x_shape=(1, 17, 3),
            weight_shape=(KERNEL_SIZE, 3),
            bias_shape=(3,),
            dtypes=(trt.bfloat16, trt.int32, trt.bfloat16, trt.bfloat16),
        ),
        id="bf16-channel-alignment",
    ),
    pytest.param(convolution_input_specs()[:3], id="missing-input"),
    pytest.param(convolution_input_specs() + ((trt.float32, (1,)),), id="extra-input"),
)


def build_static_contract(
    creator: trt.IPluginCreatorV3One, input_specs: tuple[InputSpec, ...]
) -> trt.IHostMemory | None:
    """Attempt to build one static convolution-plugin contract.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.

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
def test_convolution_plugin_matches_reference(
    convolution_engine, shape: tuple[int, int, int], lengths: tuple[int, ...]
) -> None:
    _, engine, engine_case = convolution_engine
    inputs = make_inputs(shape, lengths, engine_case.kernel_size)
    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs), inputs, engine_case
    )


def test_convolution_plugin_suppresses_nonfinite_padding(convolution_engine) -> None:
    _, engine, engine_case = convolution_engine
    inputs = make_inputs(
        (2, 17, engine_case.channels), (5, 12), engine_case.kernel_size
    )
    x, _, _, _ = inputs
    x[0, 5:] = np.nan
    x[1, 12:] = np.inf

    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs), inputs, engine_case
    )


def test_convolution_plugin_handles_activation_extremes(convolution_engine) -> None:
    _, engine, case = convolution_engine
    values = np.array((-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0), dtype=np.float32)
    x = np.broadcast_to(values[None, :, None], (1, len(values), case.channels)).copy()
    weight = np.zeros((case.kernel_size, case.channels), dtype=np.float32)
    weight[case.kernel_size // 2] = 1.0
    inputs = (
        x,
        np.array((len(values),), dtype=np.int32),
        weight,
        np.zeros(case.channels, dtype=np.float32),
    )
    assert_run_matches_reference(run_engine(engine, case, inputs), inputs, case)


@pytest.mark.parametrize(
    "engine_case", ADDITIONAL_LAYOUT_CASES, ids=lambda case: case.name
)
def test_convolution_plugin_supports_additional_layouts(
    plugin_creator, engine_case: EngineCase
) -> None:
    *_, creator = plugin_creator
    shape = (3, 259, engine_case.channels)
    result = build_engine(creator, engine_case, max_frames=shape[1])
    assert result is not None
    runtime, engine = result
    inputs = make_inputs(shape, (-4, 129, 999), engine_case.kernel_size)
    x, _, _, _ = inputs
    x[0] = np.nan
    x[1, 129:] = np.inf
    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs), inputs, engine_case
    )


def test_convolution_plugin_reuses_context_across_shapes_and_streams(
    convolution_engine,
) -> None:
    _, engine, engine_case = convolution_engine
    context = engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        ((1, 1, engine_case.channels), (1,)),
        ((3, 65, engine_case.channels), (INT32_MIN, 34, INT32_MAX)),
        ((1, 3, engine_case.channels), (2,)),
    )

    for index, (shape, lengths) in enumerate(shape_cases):
        stream = streams[index % len(streams)]
        inputs = make_inputs(shape, lengths, engine_case.kernel_size)
        run = run_engine(engine, engine_case, inputs, context=context, stream=stream)
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_reference(run, inputs, engine_case)


def test_convolution_plugin_supports_concurrent_contexts(convolution_engine) -> None:
    _, engine, engine_case = convolution_engine
    first_inputs = make_inputs(
        (3, 258, engine_case.channels), (258, 129, 1), engine_case.kernel_size
    )
    second_inputs = make_inputs(
        (2, 17, engine_case.channels), (5, 16), engine_case.kernel_size
    )
    first_run = run_engine(engine, engine_case, first_inputs)
    second_run = run_engine(engine, engine_case, second_inputs)

    assert first_run.context is not second_run.context
    assert first_run.stream.ptr != second_run.stream.ptr
    assert_run_matches_reference(first_run, first_inputs, engine_case)
    assert_run_matches_reference(second_run, second_inputs, engine_case)


def test_convolution_plugin_rejects_runtime_batch_mismatch(convolution_engine) -> None:
    _, engine, case = convolution_engine
    inputs = (
        np.zeros((2, 17, case.channels), dtype=np.float32),
        np.zeros((1,), dtype=np.int32),
        np.zeros((case.kernel_size, case.channels), dtype=np.float32),
        np.zeros((case.channels,), dtype=np.float32),
    )
    run = run_engine(engine, case, inputs, execute=False)
    executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    assert not executed
    assert bool(cp.isnan(run.output).all())


def test_convolution_plugin_supports_cuda_graph_replay(convolution_engine) -> None:
    _, engine, engine_case = convolution_engine
    inputs = make_inputs(
        (2, 17, engine_case.channels), (17, 5), engine_case.kernel_size
    )
    run = run_engine(engine, engine_case, inputs)
    run.stream.synchronize()

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for replay in range(2):
        replay_inputs = make_inputs(
            (2, 17, engine_case.channels),
            (8 + replay, 16 - replay),
            engine_case.kernel_size,
        )
        replay_x, replay_lengths, replay_weight, replay_bias = replay_inputs
        replay_x += np.float32(0.125 * (replay + 1))
        replay_weight *= np.float32(-1.0 + 0.25 * replay)
        replay_bias += np.float32(0.05 * (replay + 1))
        with run.stream:
            for name, values in zip(INPUT_NAMES, replay_inputs, strict=True):
                buffer = run.inputs[name]
                cp.copyto(buffer, cp.asarray(values, dtype=buffer.dtype))
            run.output.fill(cp.nan)
            graph.launch(run.stream)

        assert_run_matches_reference(run, replay_inputs, engine_case)


def test_convolution_plugin_accepts_valid_static_contract(plugin_creator) -> None:
    *_, creator = plugin_creator
    assert build_static_contract(creator, convolution_input_specs()) is not None


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_convolution_plugin_rejects_invalid_contracts(
    plugin_creator, input_specs: tuple[InputSpec, ...]
) -> None:
    *_, creator = plugin_creator
    assert build_static_contract(creator, input_specs) is None


@pytest.mark.parametrize("invalid_endpoint", ("min", "opt", "max"))
def test_convolution_plugin_rejects_invalid_profile_endpoints(
    plugin_creator, invalid_endpoint: str
) -> None:
    *_, creator = plugin_creator
    length_batches = {"min": (2, 2, 3), "opt": (1, 1, 3), "max": (1, 2, 2)}[
        invalid_endpoint
    ]
    assert build_engine(creator, ENGINE_CASES[1], 65, length_batches) is None


def test_convolution_creator_rejects_fields(plugin_creator) -> None:
    *_, creator = plugin_creator
    assert list(creator.field_names) == []
    field = trt.PluginField(
        "unexpected", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([field]), trt.TensorRTPhase.BUILD
    )
    assert plugin is None


@pytest.mark.parametrize("engine_case", ENGINE_CASES, ids=lambda case: case.name)
def test_convolution_builds_with_folded_constants(
    plugin_creator, engine_case: EngineCase
) -> None:
    *_, creator = plugin_creator
    rng = np.random.default_rng(20260819)
    weight = rng.normal(
        0.0, 0.2, (engine_case.kernel_size, engine_case.channels)
    ).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (engine_case.channels,)).astype(np.float32)
    constants = {
        name: cp.asnumpy(cp.asarray(values, dtype=engine_case.cupy_dtype))
        for name, values in (("weight", weight), ("bias", bias))
    }
    result = build_engine(creator, engine_case, max_frames=65, constants=constants)
    assert result is not None
    runtime, engine = result
    for name, values in (("weight", weight), ("bias", bias)):
        np.testing.assert_array_equal(
            constants[name].astype(np.float32),
            quantize_input(values, engine_case.cupy_dtype),
            err_msg=name,
        )
    x, lengths, _, _ = make_inputs(
        (2, 17, engine_case.channels), (17, 5), engine_case.kernel_size
    )
    inputs = x, lengths, weight, bias
    assert_run_matches_reference(
        run_engine(engine, engine_case, inputs), inputs, engine_case
    )
