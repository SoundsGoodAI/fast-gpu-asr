#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer temporal-resampling plugins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.cuda


DOWNSAMPLE_NAME = "zipformer_downsample"
UPSAMPLE_NAME = "zipformer_upsample_bypass"
PLUGIN_VERSION = "1"
FACTORS = (1, 2, 4, 8)
CHANNEL_CASES = {"even_multiblock": 514, "odd_multiblock": 513}
SHAPE_CASES = ((1, 1), (2, 3), (2, 6), (2, 8), (2, 17), (3, 65))
DTYPE_CASES = (
    ("fp32", trt.float32, cp.float32, 1e-5),
    ("fp16", trt.float16, cp.float16, 3e-3),
    pytest.param(
        ("bf16", trt.bfloat16, cp.dtype("bfloat16"), 3e-2),
        marks=pytest.mark.sm80,
    ),
)


@pytest.fixture(scope="module")
def plugin_creators(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register the current Zipformer resampling plugin source."""

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


def make_plugin(creator, name: str, factor: int):
    """Create one resampling plugin with a fixed temporal factor."""

    factor_field = np.array([factor], dtype=np.int32)
    fields = trt.PluginFieldCollection(
        [trt.PluginField("factor", factor_field, trt.PluginFieldType.INT32)]
    )
    plugin = creator.create_plugin(name, fields, trt.TensorRTPhase.BUILD)
    assert plugin is not None
    return plugin


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=lambda case: case[0])
def resampling_engine(request, plugin_creators):
    """Build a dynamic engine covering scalar and vector resampling paths."""

    _, creators = plugin_creators
    _, dtype, cupy_dtype, tolerance = request.param
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
                make_plugin(creators[DOWNSAMPLE_NAME], DOWNSAMPLE_NAME, factor),
            )
            assert downsample is not None
            downsample_output = downsample.get_output(0)
            downsample_output.name = f"down_{case_name}_{factor}"
            network.mark_output(downsample_output)
            output_names.append(downsample_output.name)

            upsample = network.add_plugin_v3(
                [inputs[input_name], inputs[later_name], inputs[scale_name]],
                [],
                make_plugin(creators[UPSAMPLE_NAME], UPSAMPLE_NAME, factor),
            )
            assert upsample is not None
            upsample_output = upsample.get_output(0)
            upsample_output.name = f"up_{case_name}_{factor}"
            network.mark_output(upsample_output)
            output_names.append(upsample_output.name)

    profile = builder.create_optimization_profile()
    for case_name, channels in CHANNEL_CASES.items():
        profile.set_shape(
            f"input_{case_name}",
            (1, 1, channels),
            (2, 17, channels),
            (3, 65, channels),
        )
        for factor in FACTORS:
            profile.set_shape(
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
    return runtime, engine, output_names, cupy_dtype, tolerance


@dataclass
class EngineRun:
    """Device and host state retained from one plugin execution."""

    context: object
    stream: cp.cuda.Stream
    device_inputs: dict[str, cp.ndarray]
    device_outputs: dict[str, cp.ndarray]
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]


def quantize(values: np.ndarray, cupy_dtype) -> np.ndarray:
    """Round values through a device dtype and return them as FP32."""

    return cp.asnumpy(cp.array(values, dtype=cupy_dtype)).astype(np.float32)


def downsample_reference(
    values: np.ndarray, weights: np.ndarray, factor: int, cupy_dtype
) -> np.ndarray:
    """Apply last-frame-padded weighted downsampling with FP32 accumulation."""

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
    return quantize(output, cupy_dtype)


def upsample_reference(
    early: np.ndarray,
    later: np.ndarray,
    scale: np.ndarray,
    factor: int,
    cupy_dtype,
) -> np.ndarray:
    """Repeat lower-rate frames and apply the per-channel bypass scale."""

    repeated = np.repeat(later, factor, axis=1)[:, : early.shape[1]]
    return quantize(early + (repeated - early) * scale, cupy_dtype)


def assert_resampling_outputs_match_reference(
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    cupy_dtype,
    tolerance: float,
) -> None:
    """Compare every operation, channel layout, and factor with its reference."""

    for case_name in CHANNEL_CASES:
        early = inputs[f"input_{case_name}"]
        scale = inputs[f"scale_{case_name}"]
        for factor in FACTORS:
            expected_down = downsample_reference(
                early, inputs[f"weights_{factor}"], factor, cupy_dtype
            )
            expected_up = upsample_reference(
                early, inputs[f"later_{case_name}_{factor}"], scale, factor, cupy_dtype
            )
            np.testing.assert_allclose(
                outputs[f"down_{case_name}_{factor}"],
                expected_down,
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                outputs[f"up_{case_name}_{factor}"],
                expected_up,
                rtol=tolerance,
                atol=tolerance,
            )


def run_engine(
    engine,
    output_names,
    cupy_dtype,
    batch_size: int,
    sequence_length: int,
    *,
    context: object | None = None,
    stream: cp.cuda.Stream | None = None,
) -> EngineRun:
    """Execute all resampling paths and retain inputs needed by the reference."""

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

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    device_inputs = {}
    device_outputs = {}
    with stream:
        for name, values in host_inputs.items():
            device = cp.array(values, dtype=cupy_dtype)
            assert context.set_tensor_address(name, device.data.ptr)
            device_inputs[name] = device
        for name in output_names:
            output = cp.full(
                tuple(context.get_tensor_shape(name)), np.nan, dtype=cupy_dtype
            )
            assert context.set_tensor_address(name, output.data.ptr)
            device_outputs[name] = output
        assert context.execute_async_v3(stream.ptr)
    stream.synchronize()

    quantized_inputs = {
        name: cp.asnumpy(values).astype(np.float32)
        for name, values in device_inputs.items()
    }
    outputs = {
        name: cp.asnumpy(values).astype(np.float32)
        for name, values in device_outputs.items()
    }
    return EngineRun(
        context,
        stream,
        device_inputs,
        device_outputs,
        quantized_inputs,
        outputs,
    )


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_resampling_plugins_match_reference(
    resampling_engine, batch_size: int, sequence_length: int
) -> None:
    """Compare dynamic downsample and bypass-upsample outputs across dtypes."""

    _, engine, output_names, cupy_dtype, tolerance = resampling_engine
    run = run_engine(engine, output_names, cupy_dtype, batch_size, sequence_length)

    assert_resampling_outputs_match_reference(
        run.inputs, run.outputs, cupy_dtype, tolerance
    )


def test_resampling_plugins_support_cuda_graphs(resampling_engine) -> None:
    """Replay changed inputs through every path on a non-default stream."""

    _, engine, output_names, cupy_dtype, tolerance = resampling_engine
    run = run_engine(engine, output_names, cupy_dtype, 2, 17)

    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)
    for replay in (1, 2):
        with run.stream:
            for name, device_input in run.device_inputs.items():
                updated = run.inputs[name] * (1.0 + 0.125 * replay) + 0.01 * replay
                cp.copyto(device_input, cp.array(updated, dtype=device_input.dtype))
            for output in run.device_outputs.values():
                output.fill(np.nan)
            graph.launch(run.stream)
        run.stream.synchronize()

        replay_inputs = {
            name: cp.asnumpy(values).astype(np.float32)
            for name, values in run.device_inputs.items()
        }
        replay_outputs = {
            name: cp.asnumpy(values).astype(np.float32)
            for name, values in run.device_outputs.items()
        }
        assert_resampling_outputs_match_reference(
            replay_inputs, replay_outputs, cupy_dtype, tolerance
        )


def test_resampling_plugins_reuse_context_across_dynamic_shapes(
    resampling_engine,
) -> None:
    """Reuse one execution context while shapes grow, shrink, and grow again."""

    _, engine, output_names, cupy_dtype, tolerance = resampling_engine
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)

    for batch_size, sequence_length in ((1, 1), (2, 17), (1, 5), (3, 65), (2, 6)):
        run = run_engine(
            engine,
            output_names,
            cupy_dtype,
            batch_size,
            sequence_length,
            context=context,
            stream=stream,
        )
        assert_resampling_outputs_match_reference(
            run.inputs, run.outputs, cupy_dtype, tolerance
        )


def invalid_factor_fields(
    case: str,
) -> tuple[trt.PluginFieldCollection, tuple[np.ndarray, ...]]:
    """Build one malformed factor collection and retain its backing arrays."""

    if case == "missing":
        return trt.PluginFieldCollection([]), ()
    if case == "wrong-type":
        values = (np.array([2.0], dtype=np.float32),)
        field_type = trt.PluginFieldType.FLOAT32
    elif case == "empty":
        values = (np.array([], dtype=np.int32),)
        field_type = trt.PluginFieldType.INT32
    elif case == "multiple":
        values = (np.array([2, 4], dtype=np.int32),)
        field_type = trt.PluginFieldType.INT32
    elif case == "duplicate":
        values = (np.array([2], dtype=np.int32), np.array([4], dtype=np.int32))
        field_type = trt.PluginFieldType.INT32
    elif case in ("zero", "negative"):
        values = (np.array([0 if case == "zero" else -1], dtype=np.int32),)
        field_type = trt.PluginFieldType.INT32
    else:
        raise AssertionError(f"Unhandled invalid factor case {case!r}.")

    fields = trt.PluginFieldCollection(
        [trt.PluginField("factor", value, field_type) for value in values]
    )
    return fields, values


@pytest.mark.parametrize("plugin_name", (DOWNSAMPLE_NAME, UPSAMPLE_NAME))
@pytest.mark.parametrize(
    "field_case",
    ("missing", "zero", "negative", "empty", "wrong-type", "multiple", "duplicate"),
)
def test_resampling_creators_reject_invalid_factor_fields(
    plugin_creators, plugin_name: str, field_case: str
) -> None:
    """Apply every factor-field validation to both registered creators."""

    _, creators = plugin_creators
    fields, _backing_values = invalid_factor_fields(field_case)
    plugin = creators[plugin_name].create_plugin(
        plugin_name,
        fields,
        trt.TensorRTPhase.BUILD,
    )

    assert plugin is None


@pytest.mark.parametrize(
    ("plugin_name", "input_specs"),
    (
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (2, 5)), (trt.float16, (2, 1))),
            id="downsample-input-rank",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (2, 5, 7)), (trt.float16, (2,))),
            id="downsample-weights-rank",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (2, 5, 7)), (trt.float16, (3, 1))),
            id="downsample-factor-dimension",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (2, 5, 7)), (trt.float16, (2, 2))),
            id="downsample-weight-width",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (65_536, 1, 1)), (trt.float16, (2, 1))),
            id="downsample-batch-grid-overflow",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (1, 131_071, 1)), (trt.float16, (2, 1))),
            id="downsample-output-grid-overflow",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.int32, (2, 5, 7)), (trt.int32, (2, 1))),
            id="downsample-unsupported-dtype",
        ),
        pytest.param(
            DOWNSAMPLE_NAME,
            ((trt.float16, (2, 5, 7)), (trt.float32, (2, 1))),
            id="downsample-mixed-dtypes",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 3)),
                (trt.float16, (7,)),
            ),
            id="upsample-later-rank",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 3, 7)),
                (trt.float16, (1, 7)),
            ),
            id="upsample-scale-rank",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (1, 3, 7)),
                (trt.float16, (7,)),
            ),
            id="upsample-batch-dimension",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 2, 7)),
                (trt.float16, (7,)),
            ),
            id="upsample-time-dimension",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 3, 8)),
                (trt.float16, (7,)),
            ),
            id="upsample-channel-dimension",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 3, 7)),
                (trt.float16, (8,)),
            ),
            id="upsample-scale-dimension",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (1, 65_536, 1)),
                (trt.float16, (1, 32_768, 1)),
                (trt.float16, (1,)),
            ),
            id="upsample-output-grid-overflow",
        ),
        pytest.param(
            UPSAMPLE_NAME,
            (
                (trt.float16, (2, 5, 7)),
                (trt.float16, (2, 3, 7)),
                (trt.float32, (7,)),
            ),
            id="upsample-mixed-dtypes",
        ),
    ),
)
def test_resampling_plugins_reject_invalid_contracts(
    plugin_creators,
    plugin_name: str,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid ranks, dimensions, grid bounds, and numeric types."""

    _, creators = plugin_creators
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = [
        network.add_input(f"input_{index}", dtype, shape)
        for index, (dtype, shape) in enumerate(input_specs)
    ]
    assert all(value is not None for value in inputs)
    layer = network.add_plugin_v3(
        inputs,
        [],
        make_plugin(creators[plugin_name], plugin_name, factor=2),
    )
    if layer is None:
        return

    network.mark_output(layer.get_output(0))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    assert builder.build_serialized_network(network, config) is None
