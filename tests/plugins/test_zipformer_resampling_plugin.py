#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer temporal-resampling plugins."""

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
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)

pytestmark = pytest.mark.cuda


DOWNSAMPLE_NAME = ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME
UPSAMPLE_NAME = ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME
PLUGIN_VERSION = "1"
FACTORS = (1, 2, 3, 4, 8)
CHANNEL_CASES = {"even_multiblock": 514, "odd_multiblock": 513}
SHAPE_CASES = ((1, 1), (2, 3), (2, 6), (2, 8), (2, 17), (3, 65))

type PluginCreatorsFixture = tuple[
    ctypes.CDLL,
    dict[str, trt.IPluginCreatorV3One],
]


@dataclass(frozen=True)
class DTypeCase:
    """TensorRT, CuPy, Torch, and comparison settings for one dtype."""

    name: str
    trt_dtype: trt.DataType
    cupy_dtype: type[np.generic] | np.dtype[np.generic]
    torch_dtype: torch.dtype
    tolerance: float


DTYPE_CASES = (
    DTypeCase("fp32", trt.float32, cp.float32, torch.float32, 1e-5),
    DTypeCase("fp16", trt.float16, cp.float16, torch.float16, 3e-3),
    pytest.param(
        DTypeCase(
            "bf16",
            trt.bfloat16,
            cp.dtype("bfloat16"),
            torch.bfloat16,
            3e-2,
        ),
        marks=pytest.mark.sm80,
    ),
)


@dataclass(frozen=True)
class ResamplingEngine:
    """One deserialized engine and its numeric test contract."""

    runtime: trt.Runtime
    engine: trt.ICudaEngine
    output_names: tuple[str, ...]
    dtype_case: DTypeCase


@pytest.fixture(scope="module")
def plugin_creators(
    tmp_path_factory: pytest.TempPathFactory,
) -> PluginCreatorsFixture:
    """Compile and register the current Zipformer resampling plugin source.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for isolated, module-scoped compiled libraries.

    Returns
    -------
    PluginCreatorsFixture
        Compiled library handle and both registered resampling creators.
    """

    library = compile_and_load_plugin(
        tmp_path_factory,
        "zipformer_resampling_plugin.cu",
        "initFastGpuAsrZipformerResamplingPlugins",
        ("cudart",),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    downsample_creator = registry.get_creator(
        DOWNSAMPLE_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    upsample_creator = registry.get_creator(
        UPSAMPLE_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert downsample_creator is not None
    assert upsample_creator is not None
    return library, {
        DOWNSAMPLE_NAME: downsample_creator,
        UPSAMPLE_NAME: upsample_creator,
    }


def make_plugin(
    creator: trt.IPluginCreatorV3One,
    factor: int,
) -> trt.IPluginV3:
    """Create one resampling plugin with a fixed temporal factor.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    factor : int
        Temporal downsampling or upsampling factor.

    Returns
    -------
    trt.IPluginV3
        New plugin configured for the build phase.
    """

    factor_field = np.array([factor], dtype=np.int32)
    fields = trt.PluginFieldCollection(
        [trt.PluginField("factor", factor_field, trt.PluginFieldType.INT32)]
    )
    plugin = creator.create_plugin(creator.name, fields, trt.TensorRTPhase.BUILD)
    assert plugin is not None
    return plugin


def set_profile_shape(profile, name, minimum, optimum, maximum) -> None:
    """Verify profile bounds without relying on set_shape's version-dependent return.

    Parameters
    ----------
    profile : trt.IOptimizationProfile
        Profile receiving the bounds, which are read back to verify test setup.
    name : str
        Tensor name used for the optimization profile.
    minimum : tuple[int, ...]
        Minimum input shape.
    optimum : tuple[int, ...]
        Optimum input shape used during tactic selection.
    maximum : tuple[int, ...]
        Maximum input shape.
    """

    profile.set_shape(name, minimum, optimum, maximum)
    assert tuple(map(tuple, profile.get_shape(name))) == tuple(
        map(tuple, (minimum, optimum, maximum))
    )


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=lambda case: case.name)
def resampling_engine(
    request: pytest.FixtureRequest,
    plugin_creators: PluginCreatorsFixture,
) -> ResamplingEngine:
    """Build a dynamic engine covering scalar and vector resampling paths.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creators : PluginCreatorsFixture
        Compiled library handle and creators indexed by resampling plugin name.

    Returns
    -------
    ResamplingEngine
        Deserialized engine with its owning runtime.
    """

    _, creators = plugin_creators
    dtype_case: DTypeCase = request.param
    dtype = dtype_case.trt_dtype
    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    inputs = {}
    output_names = []
    for factor in FACTORS:
        name = f"weights_{factor}"
        inputs[name] = network.add_input(name, dtype, (factor, 1))
        assert inputs[name] is not None

    for case_name, channels in CHANNEL_CASES.items():
        input_name = f"input_{case_name}"
        scale_name = f"scale_{case_name}"
        inputs[input_name] = network.add_input(input_name, dtype, (-1, -1, channels))
        inputs[scale_name] = network.add_input(scale_name, dtype, (channels,))
        assert inputs[input_name] is not None and inputs[scale_name] is not None

        for factor in FACTORS:
            later_name = f"later_{case_name}_{factor}"
            inputs[later_name] = network.add_input(
                later_name, dtype, (-1, -1, channels)
            )
            assert inputs[later_name] is not None

            downsample = network.add_plugin_v3(
                [inputs[input_name], inputs[f"weights_{factor}"]],
                [],
                make_plugin(creators[DOWNSAMPLE_NAME], factor),
            )
            assert downsample is not None
            downsample_output = downsample.get_output(0)
            downsample_output.name = f"down_{case_name}_{factor}"
            network.mark_output(downsample_output)
            output_names.append(downsample_output.name)

            upsample = network.add_plugin_v3(
                [inputs[input_name], inputs[later_name], inputs[scale_name]],
                [],
                make_plugin(creators[UPSAMPLE_NAME], factor),
            )
            assert upsample is not None
            upsample_output = upsample.get_output(0)
            upsample_output.name = f"up_{case_name}_{factor}"
            network.mark_output(upsample_output)
            output_names.append(upsample_output.name)

    profile = builder.create_optimization_profile()
    for case_name, channels in CHANNEL_CASES.items():
        set_profile_shape(
            profile,
            f"input_{case_name}",
            (1, 1, channels),
            (2, 17, channels),
            (3, 65, channels),
        )
        for factor in FACTORS:
            set_profile_shape(
                profile,
                f"later_{case_name}_{factor}",
                (1, 1, channels),
                (2, (17 + factor - 1) // factor, channels),
                (3, (65 + factor - 1) // factor, channels),
            )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    config.builder_optimization_level = 3
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        **{name: (trt.TensorIOMode.INPUT, dtype) for name in inputs},
        **{name: (trt.TensorIOMode.OUTPUT, dtype) for name in output_names},
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, tensor_dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == tensor_dtype
    return ResamplingEngine(runtime, engine, tuple(output_names), dtype_case)


@dataclass
class EngineRun:
    """Device and host state retained from one plugin execution."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    device_inputs: dict[str, cp.ndarray]
    device_outputs: dict[str, cp.ndarray]
    inputs: dict[str, np.typing.NDArray]


def quantize(values: np.typing.NDArray, dtype_case: DTypeCase) -> np.typing.NDArray:
    """Round FP32 values with a CPU oracle for the engine storage dtype.

    Parameters
    ----------
    values : np.typing.NDArray
        Input values to round or copy without modifying the original array.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.

    Returns
    -------
    np.typing.NDArray
        Independent FP32 host values rounded to the requested storage precision.
    """

    source = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    return source.to(dtype_case.torch_dtype).float().numpy()


def downsample_reference(
    values: np.typing.NDArray,
    weights: np.typing.NDArray,
    factor: int,
    dtype_case: DTypeCase,
) -> np.typing.NDArray:
    """Apply last-frame-padded weighted downsampling with FP32 accumulation.

    Parameters
    ----------
    values : np.typing.NDArray
        Dtype-rounded full-rate activations with shape (batch, time, channels).
    weights : np.typing.NDArray
        Per-offset downsampling coefficients with shape (factor, 1).
    factor : int
        Temporal downsampling or upsampling factor.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.

    Returns
    -------
    np.typing.NDArray
        FP32 host output rounded to the case dtype, with ceil(time / factor) frames.
    """

    output_length = (values.shape[1] + factor - 1) // factor
    output = np.zeros(
        (values.shape[0], output_length, values.shape[2]), dtype=np.float32
    )
    for offset in range(factor):
        frames = np.minimum(
            np.arange(output_length, dtype=np.int64) * factor + offset,
            values.shape[1] - 1,
        )
        output += values[:, frames] * weights[offset]
    return quantize(output, dtype_case)


def upsample_reference(
    early: np.typing.NDArray,
    later: np.typing.NDArray,
    scale: np.typing.NDArray,
    factor: int,
    dtype_case: DTypeCase,
) -> np.typing.NDArray:
    """Repeat lower-rate frames and apply the per-channel bypass scale.

    Parameters
    ----------
    early : np.typing.NDArray
        Original full-rate tensor in (batch, time, channels) layout.
    later : np.typing.NDArray
        Lower-rate tensor with ceil(time / factor) frames.
    scale : np.typing.NDArray
        Per-channel bypass interpolation scale, not restricted to [0, 1].
    factor : int
        Temporal downsampling or upsampling factor.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.

    Returns
    -------
    np.typing.NDArray
        FP32 host output rounded to the case dtype, with the original time
        dimension.
    """

    repeated = np.repeat(later, factor, axis=1)[:, : early.shape[1]]
    return quantize(early + (repeated - early) * scale, dtype_case)


def assert_resampling_outputs_match_reference(
    run: EngineRun, dtype_case: DTypeCase
) -> None:
    """Check input immutability, output shapes, and every numerical path.

    Parameters
    ----------
    run : EngineRun
        Bound device buffers and the context/stream that own their pending work.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.

    Notes
    -----
    The caller must synchronize the run's stream before invoking this helper.
    """

    for name, expected in run.inputs.items():
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_inputs[name]).astype(np.float32),
            expected,
            err_msg=name,
        )
    for case_name in CHANNEL_CASES:
        early = run.inputs[f"input_{case_name}"]
        scale = run.inputs[f"scale_{case_name}"]
        for factor in FACTORS:
            expected_outputs = {
                f"down_{case_name}_{factor}": downsample_reference(
                    early, run.inputs[f"weights_{factor}"], factor, dtype_case
                ),
                f"up_{case_name}_{factor}": upsample_reference(
                    early,
                    run.inputs[f"later_{case_name}_{factor}"],
                    scale,
                    factor,
                    dtype_case,
                ),
            }
            for name, expected in expected_outputs.items():
                actual = cp.asnumpy(run.device_outputs[name]).astype(np.float32)
                assert actual.shape == expected.shape, name
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=dtype_case.tolerance,
                    atol=dtype_case.tolerance,
                    err_msg=name,
                )


def run_engine(
    resampling: ResamplingEngine,
    batch_size: int,
    sequence_length: int,
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    input_overrides: dict[str, np.typing.NDArray] | None = None,
) -> EngineRun:
    """Execute and check all paths, retaining buffers for context reuse or capture.

    Parameters
    ----------
    resampling : ResamplingEngine
        Engine and dtype settings for all factor/channel combinations.
    batch_size : int
        Number of utterances in the batch.
    sequence_length : int
        Physical number of frames per utterance.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    input_overrides : dict[str, np.typing.NDArray] or None
        Named host-array replacements for hand-computed regression cases.

    Returns
    -------
    EngineRun
        Completed, reference-checked run with owned host/device buffers for replay.
    """

    engine, dtype_case = resampling.engine, resampling.dtype_case

    rng = np.random.default_rng(1000 + batch_size * 100 + sequence_length)
    host_inputs = {
        f"weights_{factor}": rng.dirichlet(np.ones(factor)).astype(np.float32)[:, None]
        for factor in FACTORS
    }
    for case_name, channels in CHANNEL_CASES.items():
        host_inputs[f"input_{case_name}"] = rng.normal(
            size=(batch_size, sequence_length, channels)
        ).astype(np.float32)
        host_inputs[f"scale_{case_name}"] = np.resize(
            np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float32), channels
        )
        for factor in FACTORS:
            later_length = (sequence_length + factor - 1) // factor
            host_inputs[f"later_{case_name}_{factor}"] = rng.normal(
                size=(batch_size, later_length, channels)
            ).astype(np.float32)

    if input_overrides is not None:
        unknown_names = input_overrides.keys() - host_inputs.keys()
        assert not unknown_names
        for name, values in input_overrides.items():
            assert values.shape == host_inputs[name].shape
            host_inputs[name] = np.array(values, dtype=np.float32, copy=True)

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    for case_name in CHANNEL_CASES:
        assert context.set_input_shape(
            f"input_{case_name}", host_inputs[f"input_{case_name}"].shape
        )
        for factor in FACTORS:
            name = f"later_{case_name}_{factor}"
            assert context.set_input_shape(name, host_inputs[name].shape)
    assert context.infer_shapes() == []

    for case_name, channels in CHANNEL_CASES.items():
        for factor in FACTORS:
            assert tuple(context.get_tensor_shape(f"down_{case_name}_{factor}")) == (
                batch_size,
                (sequence_length + factor - 1) // factor,
                channels,
            )
            assert tuple(context.get_tensor_shape(f"up_{case_name}_{factor}")) == (
                batch_size,
                sequence_length,
                channels,
            )

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    device_inputs = {}
    device_outputs = {}
    with stream:
        for name, values in host_inputs.items():
            device = cp.array(values, dtype=dtype_case.cupy_dtype)
            assert context.set_tensor_address(name, device.data.ptr)
            device_inputs[name] = device
        for name in resampling.output_names:
            output = cp.full(
                tuple(context.get_tensor_shape(name)),
                np.nan,
                dtype=dtype_case.cupy_dtype,
            )
            assert context.set_tensor_address(name, output.data.ptr)
            device_outputs[name] = output
        assert context.execute_async_v3(stream.ptr)
    stream.synchronize()

    run = EngineRun(
        context,
        stream,
        device_inputs,
        device_outputs,
        {name: quantize(values, dtype_case) for name, values in host_inputs.items()},
    )
    assert_resampling_outputs_match_reference(run, dtype_case)
    return run


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_resampling_plugins_match_reference(
    resampling_engine: ResamplingEngine,
    batch_size: int,
    sequence_length: int,
) -> None:
    run = run_engine(resampling_engine, batch_size, sequence_length)

    for case_name in CHANNEL_CASES:
        early = run.inputs[f"input_{case_name}"]
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_outputs[f"down_{case_name}_1"]).astype(np.float32),
            early,
        )
        zero_scale = run.inputs[f"scale_{case_name}"] == 0.0
        for factor in FACTORS:
            np.testing.assert_array_equal(
                cp.asnumpy(run.device_outputs[f"up_{case_name}_{factor}"]).astype(
                    np.float32
                )[:, :, zero_scale],
                early[:, :, zero_scale],
            )


def test_resampling_plugins_repeat_boundary_frames(
    resampling_engine: ResamplingEngine,
) -> None:
    factor = 4
    sequence_length = 5
    input_overrides = {
        f"weights_{factor}": np.array(
            ((1.0,), (2.0,), (4.0,), (8.0,)),
            dtype=np.float32,
        )
    }
    for case_name, channels in CHANNEL_CASES.items():
        early = np.zeros((1, sequence_length, channels), dtype=np.float32)
        early[0, :, 0] = (1.0, 2.0, 4.0, 8.0, 16.0)
        later = np.zeros((1, 2, channels), dtype=np.float32)
        later[0, :, 1] = (3.0, 7.0)
        scale = np.zeros(channels, dtype=np.float32)
        scale[1] = 1.0
        input_overrides[f"input_{case_name}"] = early
        input_overrides[f"later_{case_name}_{factor}"] = later
        input_overrides[f"scale_{case_name}"] = scale

    run = run_engine(
        resampling_engine,
        batch_size=1,
        sequence_length=sequence_length,
        input_overrides=input_overrides,
    )

    for case_name in CHANNEL_CASES:
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_outputs[f"down_{case_name}_{factor}"]).astype(
                np.float32
            )[0, :, 0],
            np.array((85.0, 240.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_outputs[f"up_{case_name}_{factor}"]).astype(
                np.float32
            )[0, :, 1],
            np.array((3.0, 3.0, 3.0, 3.0, 7.0), dtype=np.float32),
        )


def test_resampling_plugins_use_fp32_intermediates(
    resampling_engine: ResamplingEngine,
) -> None:
    factor = 8
    sequence_length = 8
    # FP16/BF16 accumulation rounds this alternating sum to 1; FP32 yields 4.
    cancellation_values = np.array(
        (2048.0, 1.0, -2048.0, 1.0, 2048.0, 1.0, -2048.0, 1.0),
        dtype=np.float32,
    )
    input_overrides = {f"weights_{factor}": np.ones((factor, 1), dtype=np.float32)}
    for case_name, channels in CHANNEL_CASES.items():
        early = np.zeros((1, sequence_length, channels), dtype=np.float32)
        early[0, :, 0] = cancellation_values
        # Low-precision interpolation rounds -0.5 out of the subtraction and
        # returns 0 at scale 1; FP32 intermediates preserve the later value.
        early[0, :, 1] = -2048.0
        later = np.zeros((1, 1, channels), dtype=np.float32)
        later[0, 0, 1] = -0.5
        input_overrides[f"input_{case_name}"] = early
        input_overrides[f"later_{case_name}_{factor}"] = later

    run = run_engine(
        resampling_engine,
        batch_size=1,
        sequence_length=sequence_length,
        input_overrides=input_overrides,
    )

    for case_name in CHANNEL_CASES:
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_outputs[f"down_{case_name}_{factor}"]).astype(
                np.float32
            )[0, :, 0],
            np.array([4.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            cp.asnumpy(run.device_outputs[f"up_{case_name}_{factor}"]).astype(
                np.float32
            )[0, :, 1],
            np.full(sequence_length, -0.5, dtype=np.float32),
        )


def test_resampling_plugins_support_cuda_graphs(
    resampling_engine: ResamplingEngine,
) -> None:
    run = run_engine(resampling_engine, 2, 17)

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)
    for replay in (1, 2):
        with run.stream:
            for name, device_input in run.device_inputs.items():
                updated = run.inputs[name] * np.float32(
                    1.0 + 0.125 * replay
                ) + np.float32(0.01 * replay)
                run.inputs[name] = quantize(updated, resampling_engine.dtype_case)
                cp.copyto(
                    device_input, cp.asarray(run.inputs[name], dtype=device_input.dtype)
                )
            for output in run.device_outputs.values():
                output.fill(np.nan)
            graph.launch(run.stream)
        run.stream.synchronize()
        assert_resampling_outputs_match_reference(run, resampling_engine.dtype_case)


def test_resampling_plugins_reuse_context_across_dynamic_shapes(
    resampling_engine: ResamplingEngine,
) -> None:
    context = resampling_engine.engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)

    for batch_size, sequence_length in ((1, 1), (2, 17), (1, 5), (3, 65), (2, 6)):
        run_engine(
            resampling_engine,
            batch_size,
            sequence_length,
            context=context,
            stream=stream,
        )


def test_resampling_plugins_support_concurrent_contexts(
    resampling_engine: ResamplingEngine,
) -> None:
    runs = (
        run_engine(resampling_engine, 3, 65),
        run_engine(resampling_engine, 2, 17),
    )
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr

    transforms = ((0.5, -0.125), (-0.75, 0.25))
    for run, (multiplier, offset) in zip(runs, transforms, strict=True):
        with run.stream:
            for name, device_input in run.device_inputs.items():
                updated = run.inputs[name] * np.float32(multiplier) + np.float32(offset)
                run.inputs[name] = quantize(updated, resampling_engine.dtype_case)
                cp.copyto(
                    device_input, cp.asarray(run.inputs[name], dtype=device_input.dtype)
                )
            for output in run.device_outputs.values():
                output.fill(np.nan)
            assert run.context.execute_async_v3(run.stream.ptr)

    for run in runs:
        run.stream.synchronize()
        assert_resampling_outputs_match_reference(run, resampling_engine.dtype_case)


@pytest.mark.parametrize("invalid_relationship", ("batch", "time"))
def test_upsample_plugin_rejects_runtime_shape_mismatch(
    resampling_engine: ResamplingEngine,
    invalid_relationship: str,
) -> None:
    run = run_engine(resampling_engine, 2, 17)
    case_name = "odd_multiblock"
    factor = 2
    later_name = f"later_{case_name}_{factor}"
    output_name = f"up_{case_name}_{factor}"
    invalid_shape = (
        (1, 9, CHANNEL_CASES[case_name])
        if invalid_relationship == "batch"
        else (2, 8, CHANNEL_CASES[case_name])
    )
    assert run.context.set_input_shape(
        later_name,
        invalid_shape,
    )

    with run.stream:
        for output in run.device_outputs.values():
            output.fill(np.nan)
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    assert not executed
    assert bool(cp.isnan(run.device_outputs[output_name]).all())


@pytest.mark.parametrize("plugin_name", (DOWNSAMPLE_NAME, UPSAMPLE_NAME))
@pytest.mark.parametrize(
    ("values", "field_type"),
    (
        pytest.param((), trt.PluginFieldType.INT32, id="missing"),
        pytest.param(([0],), trt.PluginFieldType.INT32, id="zero"),
        pytest.param(([-1],), trt.PluginFieldType.INT32, id="negative"),
        pytest.param(([],), trt.PluginFieldType.INT32, id="empty"),
        pytest.param(([2.0],), trt.PluginFieldType.FLOAT32, id="wrong-type"),
        pytest.param(([2, 4],), trt.PluginFieldType.INT32, id="multiple"),
        pytest.param(([2], [4]), trt.PluginFieldType.INT32, id="duplicate"),
    ),
)
def test_resampling_creators_reject_invalid_factor_fields(
    plugin_creators: PluginCreatorsFixture, plugin_name: str, values, field_type
) -> None:
    _, creators = plugin_creators
    dtype = np.float32 if field_type == trt.PluginFieldType.FLOAT32 else np.int32
    values = [np.array(value, dtype=dtype) for value in values]
    fields = trt.PluginFieldCollection(
        [trt.PluginField("factor", value, field_type) for value in values]
    )

    assert (
        creators[plugin_name].create_plugin(
            plugin_name, fields, trt.TensorRTPhase.BUILD
        )
        is None
    )


@pytest.mark.parametrize("plugin_name", (DOWNSAMPLE_NAME, UPSAMPLE_NAME))
@pytest.mark.parametrize("include_factor", (False, True))
def test_resampling_creators_ignore_unknown_fields(
    plugin_creators: PluginCreatorsFixture, plugin_name: str, include_factor: bool
) -> None:
    _, creators = plugin_creators
    metadata = np.array([17], dtype=np.int32)
    factor = np.array([2], dtype=np.int32)
    fields = [
        trt.PluginField("implementation_metadata", metadata, trt.PluginFieldType.INT32)
    ]
    if include_factor:
        fields.append(trt.PluginField("factor", factor, trt.PluginFieldType.INT32))

    plugin = creators[plugin_name].create_plugin(
        plugin_name, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
    )
    assert (plugin is not None) is include_factor


def build_static_contract(
    creator: trt.IPluginCreatorV3One,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[trt.DataType, ...] | None = None,
) -> trt.IHostMemory | None:
    """Attempt to build a static factor-two plugin, defaulting to FP16 inputs.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    shapes : tuple[tuple[int, ...], ...]
        Static input shapes in plugin binding order.
    dtypes : tuple[trt.DataType, ...] or None
        Input dtypes in binding order; None selects this helper's default contract.

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
    if dtypes is None:
        dtypes = (trt.float16,) * len(shapes)
    inputs = [
        network.add_input(f"input_{index}", dtype, shape)
        for index, (dtype, shape) in enumerate(zip(dtypes, shapes, strict=True))
    ]
    assert all(value is not None for value in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator, factor=2))
    if layer is None:
        return None

    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    return builder.build_serialized_network(network, config)


@pytest.mark.parametrize(
    ("shapes", "valid"),
    (
        pytest.param(((2, 5), (2, 1)), False, id="input-rank"),
        pytest.param(((2, 0, 7), (2, 1)), False, id="empty-sequence"),
        pytest.param(((0, 5, 7), (2, 1)), False, id="empty-batch"),
        pytest.param(((2, 5, 0), (2, 1)), False, id="empty-channels"),
        pytest.param(((2, 5, 7), (2,)), False, id="weights-rank"),
        pytest.param(((2, 5, 7), (3, 1)), False, id="factor-dimension"),
        pytest.param(((2, 5, 7), (2, 2)), False, id="weight-width"),
        pytest.param(((65_536, 1, 1), (2, 1)), False, id="batch-grid-overflow"),
        pytest.param(((1, 131_071, 1), (2, 1)), False, id="output-grid-overflow"),
        pytest.param(((2, 5, 7),), False, id="missing-input"),
        pytest.param(((2, 5, 7), (2, 1), (1,)), False, id="extra-input"),
        pytest.param(((2, 5, 7), (2, 1)), True, id="downsample"),
        pytest.param(((65_535, 1, 1), (2, 1)), True, id="batch-grid-limit"),
        pytest.param(((1, 131_070, 1), (2, 1)), True, id="output-grid-limit"),
    ),
)
def test_downsample_static_contracts(
    plugin_creators: PluginCreatorsFixture,
    shapes: tuple[tuple[int, ...], ...],
    valid: bool,
) -> None:
    _, creators = plugin_creators
    serialized = build_static_contract(creators[DOWNSAMPLE_NAME], shapes)
    assert (serialized is not None) is valid


@pytest.mark.parametrize(
    ("shapes", "valid"),
    (
        pytest.param(((2, 5), (2, 3, 7), (7,)), False, id="early-rank"),
        pytest.param(((2, 0, 7), (2, 0, 7), (7,)), False, id="empty-sequence"),
        pytest.param(((0, 5, 7), (0, 3, 7), (7,)), False, id="empty-batch"),
        pytest.param(((2, 5, 0), (2, 3, 0), (0,)), False, id="empty-channels"),
        pytest.param(((2, 5, 7), (2, 3), (7,)), False, id="later-rank"),
        pytest.param(((2, 5, 7), (2, 3, 7), (1, 7)), False, id="scale-rank"),
        pytest.param(((2, 5, 7), (1, 3, 7), (7,)), False, id="batch-dimension"),
        pytest.param(((2, 5, 7), (2, 2, 7), (7,)), False, id="time-dimension"),
        pytest.param(((2, 5, 7), (2, 3, 8), (7,)), False, id="channel-dimension"),
        pytest.param(((2, 5, 7), (2, 3, 7), (8,)), False, id="scale-dimension"),
        pytest.param(
            ((65_536, 1, 1), (65_536, 1, 1), (1,)), False, id="batch-grid-overflow"
        ),
        pytest.param(
            ((1, 65_536, 1), (1, 32_768, 1), (1,)), False, id="output-grid-overflow"
        ),
        pytest.param(((2, 5, 7), (2, 3, 7)), False, id="missing-input"),
        pytest.param(((2, 5, 7), (2, 3, 7), (7,), (1,)), False, id="extra-input"),
        pytest.param(((2, 5, 7), (2, 3, 7), (7,)), True, id="upsample"),
        pytest.param(
            ((65_535, 1, 1), (65_535, 1, 1), (1,)), True, id="batch-grid-limit"
        ),
        pytest.param(
            ((1, 65_535, 1), (1, 32_768, 1), (1,)), True, id="output-grid-limit"
        ),
    ),
)
def test_upsample_static_contracts(
    plugin_creators: PluginCreatorsFixture,
    shapes: tuple[tuple[int, ...], ...],
    valid: bool,
) -> None:
    _, creators = plugin_creators
    serialized = build_static_contract(creators[UPSAMPLE_NAME], shapes)
    assert (serialized is not None) is valid


@pytest.mark.parametrize(
    ("plugin_name", "dtypes"),
    (
        pytest.param(
            DOWNSAMPLE_NAME, (trt.int32, trt.int32), id="downsample-unsupported-dtype"
        ),
        pytest.param(
            DOWNSAMPLE_NAME, (trt.float16, trt.float32), id="downsample-mixed-dtypes"
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (trt.int32, trt.int32, trt.int32),
            id="upsample-unsupported-dtype",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (trt.float16, trt.float32, trt.float16),
            id="upsample-mixed-later-dtype",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (trt.float16, trt.float16, trt.float32),
            id="upsample-mixed-scale-dtype",
        ),
    ),
)
def test_resampling_plugins_reject_invalid_dtypes(
    plugin_creators: PluginCreatorsFixture,
    plugin_name: str,
    dtypes: tuple[trt.DataType, ...],
) -> None:
    _, creators = plugin_creators
    shapes = (
        ((2, 5, 7), (2, 1))
        if plugin_name == DOWNSAMPLE_NAME
        else ((2, 5, 7), (2, 3, 7), (7,))
    )
    assert build_static_contract(creators[plugin_name], shapes, dtypes) is None


@pytest.mark.parametrize("invalid_endpoint", (0, 1, 2), ids=("min", "opt", "max"))
@pytest.mark.parametrize("invalid_relationship", ("batch", "time"))
def test_upsample_plugin_rejects_invalid_profile_endpoints(
    plugin_creators: PluginCreatorsFixture,
    invalid_endpoint: int,
    invalid_relationship: str,
) -> None:
    _, creators = plugin_creators
    channels = 7
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    early = network.add_input("early", trt.float16, (-1, -1, channels))
    later = network.add_input("later", trt.float16, (-1, -1, channels))
    scale = network.add_input("scale", trt.float16, (channels,))
    assert all(tensor is not None for tensor in (early, later, scale))
    layer = network.add_plugin_v3(
        [early, later, scale],
        [],
        make_plugin(creators[UPSAMPLE_NAME], factor=2),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    later_shapes = [[1, 1, channels], [2, 9, channels], [3, 33, channels]]
    axis = 0 if invalid_relationship == "batch" else 1
    later_shapes[invalid_endpoint][axis] += 1 if invalid_endpoint == 0 else -1
    profile = builder.create_optimization_profile()
    set_profile_shape(
        profile,
        "early",
        (1, 1, channels),
        (2, 17, channels),
        (3, 65, channels),
    )
    set_profile_shape(profile, "later", *later_shapes)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


@pytest.mark.parametrize("plugin_name", (DOWNSAMPLE_NAME, UPSAMPLE_NAME))
def test_resampling_creators_expose_complete_contract(
    plugin_creators: PluginCreatorsFixture,
    plugin_name: str,
) -> None:
    _, creators = plugin_creators
    creator = creators[plugin_name]
    fields = tuple(creator.field_names)

    assert creator.name == plugin_name
    assert creator.plugin_version == PLUGIN_VERSION
    assert creator.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert len(fields) == 1
    assert fields[0].name == "factor"
    assert fields[0].type == trt.PluginFieldType.INT32
    assert fields[0].size == 1

    plugin = make_plugin(creator, factor=3)
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None
    assert core.plugin_name == plugin_name
    assert core.plugin_version == PLUGIN_VERSION
    assert core.plugin_namespace == TENSORRT_PLUGIN_NAMESPACE
    assert build is not None
    assert build.num_outputs == 1
    assert runtime is not None


@pytest.mark.parametrize("dtype_case", DTYPE_CASES, ids=lambda case: case.name)
def test_resampling_plugins_match_reference_with_folded_constants(
    plugin_creators: PluginCreatorsFixture,
    dtype_case: DTypeCase,
) -> None:
    _, creators = plugin_creators
    dtype = dtype_case.trt_dtype
    factor = 3
    channels = 8
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    early = network.add_input("constant_early", dtype, (-1, -1, channels))
    later = network.add_input("constant_later", dtype, (-1, -1, channels))
    assert early is not None and later is not None

    weight_source = np.array((0.25, -0.5, 1.0), dtype=np.float32)[:, None]
    scale_source = np.array(
        (-0.25, 0.0, 0.25, 0.5, 1.0, 1.25, -1.0, 0.75),
        dtype=np.float32,
    )
    weight_values = cp.asnumpy(cp.asarray(weight_source, dtype=dtype_case.cupy_dtype))
    scale_values = cp.asnumpy(cp.asarray(scale_source, dtype=dtype_case.cupy_dtype))
    weight_storage = trt.Weights(
        dtype,
        weight_values.ctypes.data,
        weight_values.size,
    )
    scale_storage = trt.Weights(
        dtype,
        scale_values.ctypes.data,
        scale_values.size,
    )
    weight_layer = network.add_constant(weight_values.shape, weight_storage)
    scale_layer = network.add_constant(scale_values.shape, scale_storage)
    assert weight_layer is not None and scale_layer is not None

    downsample = network.add_plugin_v3(
        [early, weight_layer.get_output(0)],
        [],
        make_plugin(creators[DOWNSAMPLE_NAME], factor),
    )
    upsample = network.add_plugin_v3(
        [early, later, scale_layer.get_output(0)],
        [],
        make_plugin(creators[UPSAMPLE_NAME], factor),
    )
    assert downsample is not None and upsample is not None
    downsample_output = downsample.get_output(0)
    downsample_output.name = "constant_down"
    network.mark_output(downsample_output)
    upsample_output = upsample.get_output(0)
    upsample_output.name = "constant_up"
    network.mark_output(upsample_output)

    profile = builder.create_optimization_profile()
    set_profile_shape(
        profile,
        "constant_early",
        (1, 1, channels),
        (2, 17, channels),
        (3, 65, channels),
    )
    set_profile_shape(
        profile,
        "constant_later",
        (1, 1, channels),
        (2, 6, channels),
        (3, 22, channels),
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    expected_io = {
        "constant_early": (trt.TensorIOMode.INPUT, dtype),
        "constant_later": (trt.TensorIOMode.INPUT, dtype),
        "constant_down": (trt.TensorIOMode.OUTPUT, dtype),
        "constant_up": (trt.TensorIOMode.OUTPUT, dtype),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, tensor_dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == tensor_dtype

    rng = np.random.default_rng(20260901)
    early_source = rng.normal(size=(2, 17, channels)).astype(np.float32)
    later_source = rng.normal(size=(2, 6, channels)).astype(np.float32)
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("constant_early", early_source.shape)
    assert context.set_input_shape("constant_later", later_source.shape)
    assert context.infer_shapes() == []
    assert tuple(context.get_tensor_shape("constant_down")) == (2, 6, channels)
    assert tuple(context.get_tensor_shape("constant_up")) == early_source.shape

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        early_device = cp.asarray(early_source, dtype=dtype_case.cupy_dtype)
        later_device = cp.asarray(later_source, dtype=dtype_case.cupy_dtype)
        down_device = cp.full((2, 6, channels), np.nan, dtype=dtype_case.cupy_dtype)
        up_device = cp.full(early_source.shape, np.nan, dtype=dtype_case.cupy_dtype)
        for name, value in (
            ("constant_early", early_device),
            ("constant_later", later_device),
            ("constant_down", down_device),
            ("constant_up", up_device),
        ):
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.execute_async_v3(stream.ptr)
    stream.synchronize()

    expected_early = quantize(early_source, dtype_case)
    expected_later = quantize(later_source, dtype_case)
    expected_weight = quantize(weight_source, dtype_case)
    expected_scale = quantize(scale_source, dtype_case)
    np.testing.assert_array_equal(
        cp.asnumpy(early_device).astype(np.float32),
        expected_early,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(later_device).astype(np.float32),
        expected_later,
    )
    np.testing.assert_array_equal(
        weight_values.astype(np.float32),
        expected_weight,
    )
    np.testing.assert_array_equal(
        scale_values.astype(np.float32),
        expected_scale,
    )
    np.testing.assert_allclose(
        cp.asnumpy(down_device).astype(np.float32),
        downsample_reference(
            expected_early,
            expected_weight,
            factor,
            dtype_case,
        ),
        rtol=dtype_case.tolerance,
        atol=dtype_case.tolerance,
    )
    np.testing.assert_allclose(
        cp.asnumpy(up_device).astype(np.float32),
        upsample_reference(
            expected_early,
            expected_later,
            expected_scale,
            factor,
            dtype_case,
        ),
        rtol=dtype_case.tolerance,
        atol=dtype_case.tolerance,
    )
