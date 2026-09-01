#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet flash-attention plugin."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.sm80


PLUGIN_NAME = "parakeet_flash_attention"
PLUGIN_VERSION = "1"
INPUT_NAMES = (
    "qkv",
    "position",
    "content_bias",
    "position_bias",
    "valid_lengths",
)
DEFAULT_NUM_HEADS = 8
DEFAULT_HEAD_DIM = 64
DEFAULT_SCALE = DEFAULT_HEAD_DIM**-0.5
MAX_SEQUENCE_LENGTH = 512
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
SOFTMAX_DISPATCH_CASES = tuple(
    pytest.param(
        (1, 32 * (slots - 1) + 1),
        (32 * (slots - 1),),
        id=f"softmax-slots-{slots}",
    )
    for slots in range(4, 16)
)
SHAPE_CASES = (
    pytest.param((1, 1), (1,), id="minimum"),
    pytest.param((2, 3), (0, 3), id="empty-utterance"),
    pytest.param((2, 17), (17, 5), id="mixed-lengths"),
    pytest.param((1, 32), (31,), id="one-warp-boundary"),
    pytest.param((1, 33), (33,), id="second-warp-slot"),
    pytest.param(
        (3, 65),
        (INT32_MIN, 34, INT32_MAX),
        id="length-extremes",
    ),
    *SOFTMAX_DISPATCH_CASES,
    pytest.param((1, MAX_SEQUENCE_LENGTH), (511,), id="maximum"),
)


@dataclass(frozen=True)
class EngineCase:
    """Numeric dtype, attention layout, scale, and comparison tolerance."""

    name: str
    trt_dtype: object
    cupy_dtype: object
    tolerance: float
    num_heads: int = DEFAULT_NUM_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    scale: float = DEFAULT_SCALE

    @property
    def channels(self) -> int:
        return self.num_heads * self.head_dim


ENGINE_CASES = (
    # TensorRT may select TF32 or another reduced-mantissa FP32 cuBLAS tactic.
    EngineCase("fp32", trt.float32, cp.float32, 3e-4),
    EngineCase("fp16", trt.float16, cp.float16, 3e-2),
    EngineCase("bf16", trt.bfloat16, cp.dtype("bfloat16"), 1e-1),
)
ALTERNATE_ENGINE_CASE = EngineCase(
    "fp32-h3-d5-scale075",
    trt.float32,
    cp.float32,
    3e-4,
    num_heads=3,
    head_dim=5,
    scale=0.75,
)


@dataclass(frozen=True)
class AttentionEngine:
    """One deserialized engine and the contract used to build it."""

    runtime: object
    engine: object
    case: EngineCase


@dataclass(frozen=True)
class AttentionInputs:
    """Host inputs for one attention invocation."""

    qkv: np.ndarray
    position: np.ndarray
    content_bias: np.ndarray
    position_bias: np.ndarray
    valid_lengths: np.ndarray


@dataclass
class AttentionRun:
    """Execution state and device buffers retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    qkv: cp.ndarray
    position: cp.ndarray
    content_bias: cp.ndarray
    position_bias: cp.ndarray
    valid_lengths: cp.ndarray
    output: cp.ndarray


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile, register, and return the Parakeet attention creator."""

    library = compile_and_load_plugin(
        tmp_path_factory,
        "parakeet_flash_attention_plugin.cu",
        "initFastGpuAsrParakeetFlashAttentionPlugin",
        ("cublas", "cudart"),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return library, creator


def make_plugin(creator, scale: float):
    """Create a Parakeet attention plugin with one scalar scale field."""

    scale_value = np.array([scale], dtype=np.float32)
    field = trt.PluginField("scale", scale_value, trt.PluginFieldType.FLOAT32)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def add_dynamic_attention_layer(network, creator, case: EngineCase) -> None:
    """Add one dynamic five-input attention layer and mark its output."""

    channels = case.channels
    qkv = network.add_input("qkv", case.trt_dtype, (-1, -1, 3 * channels))
    position = network.add_input("position", case.trt_dtype, (1, -1, channels))
    content_bias = network.add_input(
        "content_bias", case.trt_dtype, (case.num_heads, case.head_dim)
    )
    position_bias = network.add_input(
        "position_bias", case.trt_dtype, (case.num_heads, case.head_dim)
    )
    valid_lengths = network.add_input("valid_lengths", trt.int32, (-1,))
    inputs = (qkv, position, content_bias, position_bias, valid_lengths)
    assert all(tensor is not None for tensor in inputs)

    layer = network.add_plugin_v3(
        list(inputs),
        [],
        make_plugin(creator, case.scale),
    )
    assert layer is not None
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)


def build_attention_engine(
    creator,
    case: EngineCase,
    *,
    max_batch_size: int = 3,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> AttentionEngine:
    """Build one dynamic engine for a concrete dtype and attention layout."""

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    add_dynamic_attention_layer(network, creator, case)

    opt_batch_size = min(2, max_batch_size)
    opt_sequence_length = min(17, max_sequence_length)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "qkv",
        (1, 1, 3 * case.channels),
        (opt_batch_size, opt_sequence_length, 3 * case.channels),
        (max_batch_size, max_sequence_length, 3 * case.channels),
    )
    profile.set_shape(
        "position",
        (1, 1, case.channels),
        (1, 2 * opt_sequence_length - 1, case.channels),
        (1, 2 * max_sequence_length - 1, case.channels),
    )
    profile.set_shape(
        "valid_lengths",
        (1,),
        (opt_batch_size,),
        (max_batch_size,),
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
        "qkv": (trt.TensorIOMode.INPUT, case.trt_dtype),
        "position": (trt.TensorIOMode.INPUT, case.trt_dtype),
        "content_bias": (trt.TensorIOMode.INPUT, case.trt_dtype),
        "position_bias": (trt.TensorIOMode.INPUT, case.trt_dtype),
        "valid_lengths": (trt.TensorIOMode.INPUT, trt.int32),
        "output": (trt.TensorIOMode.OUTPUT, case.trt_dtype),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return AttentionEngine(runtime, engine, case)


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def attention_engine(request, plugin_creator) -> AttentionEngine:
    """Build the production attention layout for every supported dtype."""

    _, creator = plugin_creator
    return build_attention_engine(creator, request.param)


def input_items(inputs: AttentionInputs) -> tuple[tuple[str, np.ndarray], ...]:
    """Return attention inputs in the plugin's binding order."""

    return (
        ("qkv", inputs.qkv),
        ("position", inputs.position),
        ("content_bias", inputs.content_bias),
        ("position_bias", inputs.position_bias),
        ("valid_lengths", inputs.valid_lengths),
    )


def make_inputs(
    case: EngineCase,
    batch_size: int,
    sequence_length: int,
    valid_lengths: tuple[int, ...],
    *,
    seed: int | None = None,
) -> AttentionInputs:
    """Create deterministic projected attention inputs."""

    assert len(valid_lengths) == batch_size
    if seed is None:
        seed = 20260819 + batch_size * 100 + sequence_length
    rng = np.random.default_rng(seed)
    relative_length = 2 * sequence_length - 1
    return AttentionInputs(
        rng.normal(
            0.0,
            0.2,
            (batch_size, sequence_length, 3 * case.channels),
        ).astype(np.float32),
        rng.normal(0.0, 0.2, (1, relative_length, case.channels)).astype(np.float32),
        rng.normal(0.0, 0.2, (case.num_heads, case.head_dim)).astype(np.float32),
        rng.normal(0.0, 0.2, (case.num_heads, case.head_dim)).astype(np.float32),
        np.array(valid_lengths, dtype=np.int32),
    )


def prepare_engine_run(
    attention_engine: AttentionEngine,
    inputs: AttentionInputs,
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
) -> AttentionRun:
    """Resolve shapes and bind sentinel-backed buffers without executing."""

    assert inputs.valid_lengths.dtype == np.int32
    if context is None:
        context = attention_engine.engine.create_execution_context()
    assert context is not None
    for name, value in input_items(inputs):
        assert context.set_input_shape(name, value.shape)
    expected_output_shape = (
        inputs.qkv.shape[0],
        inputs.qkv.shape[1],
        attention_engine.case.channels,
    )
    assert tuple(context.get_tensor_shape("output")) == expected_output_shape

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        qkv = cp.asarray(inputs.qkv, dtype=attention_engine.case.cupy_dtype)
        position = cp.asarray(
            inputs.position,
            dtype=attention_engine.case.cupy_dtype,
        )
        content_bias = cp.asarray(
            inputs.content_bias,
            dtype=attention_engine.case.cupy_dtype,
        )
        position_bias = cp.asarray(
            inputs.position_bias,
            dtype=attention_engine.case.cupy_dtype,
        )
        valid_lengths = cp.asarray(inputs.valid_lengths, dtype=cp.int32)
        output = cp.full(
            expected_output_shape,
            cp.nan,
            dtype=attention_engine.case.cupy_dtype,
        )
        device_inputs = (
            qkv,
            position,
            content_bias,
            position_bias,
            valid_lengths,
        )
        for name, value in zip(INPUT_NAMES, device_inputs, strict=True):
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
    return AttentionRun(
        context,
        stream,
        qkv,
        position,
        content_bias,
        position_bias,
        valid_lengths,
        output,
    )


def run_engine(
    attention_engine: AttentionEngine,
    inputs: AttentionInputs,
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
    synchronize: bool = True,
) -> AttentionRun:
    """Execute one attention shape and retain all device and context state."""

    run = prepare_engine_run(
        attention_engine,
        inputs,
        context=context,
        stream=stream,
    )
    with run.stream:
        assert run.context.execute_async_v3(run.stream.ptr)
    if synchronize:
        run.stream.synchronize()
    return run


def collect_run(run: AttentionRun) -> tuple[np.ndarray, AttentionInputs]:
    """Synchronize and copy one output plus its quantized inputs to the host."""

    run.stream.synchronize()
    numeric_inputs = tuple(
        cp.asnumpy(value).astype(np.float32)
        for value in (
            run.qkv,
            run.position,
            run.content_bias,
            run.position_bias,
        )
    )
    inputs = AttentionInputs(
        *numeric_inputs,
        cp.asnumpy(run.valid_lengths).astype(np.int32),
    )
    return cp.asnumpy(run.output).astype(np.float32), inputs


def reference_attention(
    inputs: AttentionInputs,
    case: EngineCase,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Evaluate relative-position attention independently in FP32 with PyTorch."""

    qkv = torch.from_numpy(inputs.qkv)
    position = torch.from_numpy(inputs.position)
    content_bias = torch.from_numpy(inputs.content_bias)
    position_bias = torch.from_numpy(inputs.position_bias)
    valid_lengths = torch.from_numpy(inputs.valid_lengths)
    batch_size, sequence_length, _ = qkv.shape
    query, key, value = qkv.chunk(3, dim=2)
    query = query.reshape(
        batch_size, sequence_length, case.num_heads, case.head_dim
    ).permute(0, 2, 1, 3)
    key = key.reshape(
        batch_size, sequence_length, case.num_heads, case.head_dim
    ).permute(0, 2, 1, 3)
    value = value.reshape(batch_size, sequence_length, case.num_heads, case.head_dim)
    position = position.reshape(1, -1, case.num_heads, case.head_dim).permute(
        0, 2, 1, 3
    )
    content_query = query + content_bias.unsqueeze(0).unsqueeze(2)
    position_query = query + position_bias.unsqueeze(0).unsqueeze(2)
    content_scores = torch.matmul(content_query, key.permute(0, 1, 3, 2))
    position_scores = torch.matmul(
        position_query,
        position.permute(0, 1, 3, 2),
    )
    relative_indexes = (
        sequence_length
        - 1
        - torch.arange(sequence_length).unsqueeze(1)
        + torch.arange(sequence_length).unsqueeze(0)
    )
    position_scores = position_scores.gather(
        3,
        relative_indexes.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, case.num_heads, -1, -1),
    )
    invalid_keys = torch.arange(sequence_length).unsqueeze(
        0
    ) >= valid_lengths.unsqueeze(1)
    attention_scale = case.scale if scale is None else scale
    weights = torch.softmax(
        ((content_scores + position_scores) * attention_scale).masked_fill(
            invalid_keys[:, None, None], float("-inf")
        ),
        dim=3,
    )
    weights = torch.where(
        (valid_lengths > 0).reshape(batch_size, 1, 1, 1),
        weights,
        torch.zeros((), dtype=weights.dtype),
    )
    return (
        torch.matmul(weights, value.permute(0, 2, 1, 3))
        .permute(0, 2, 1, 3)
        .reshape(batch_size, sequence_length, case.channels)
        .numpy()
    )


def assert_run_matches_reference(
    run: AttentionRun,
    case: EngineCase,
) -> tuple[np.ndarray, AttentionInputs]:
    """Compare one completed engine run with the independent reference."""

    actual, inputs = collect_run(run)
    expected = reference_attention(inputs, case)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=case.tolerance,
        atol=case.tolerance,
    )
    empty_rows = inputs.valid_lengths <= 0
    if empty_rows.any():
        np.testing.assert_array_equal(
            actual[empty_rows],
            np.zeros_like(actual[empty_rows]),
        )
    return actual, inputs


@pytest.mark.parametrize(("shape", "valid_lengths"), SHAPE_CASES)
def test_parakeet_flash_attention_plugin_matches_reference(
    attention_engine: AttentionEngine,
    shape: tuple[int, int],
    valid_lengths: tuple[int, ...],
) -> None:
    """Compare dynamic relative attention across all supported dtypes."""

    inputs = make_inputs(attention_engine.case, *shape, valid_lengths)
    assert_run_matches_reference(
        run_engine(attention_engine, inputs),
        attention_engine.case,
    )


def test_parakeet_flash_attention_supports_alternate_layout_and_scale(
    plugin_creator,
) -> None:
    """Preserve odd head dimensions and a nondefault serialized scale."""

    _, creator = plugin_creator
    attention_engine = build_attention_engine(
        creator,
        ALTERNATE_ENGINE_CASE,
        max_sequence_length=65,
    )
    inputs = make_inputs(
        ALTERNATE_ENGINE_CASE,
        2,
        33,
        (33, 17),
        seed=7001,
    )
    inputs.qkv[..., : 2 * ALTERNATE_ENGINE_CASE.channels] *= 3.0
    inputs.position[...] *= 2.0
    inputs.content_bias[...] *= 2.0
    inputs.position_bias[...] *= 2.0
    run = run_engine(attention_engine, inputs)
    actual, quantized_inputs = assert_run_matches_reference(
        run,
        ALTERNATE_ENGINE_CASE,
    )

    wrong_scale = reference_attention(
        quantized_inputs,
        ALTERNATE_ENGINE_CASE,
        scale=DEFAULT_SCALE,
    )
    assert not np.allclose(
        actual,
        wrong_scale,
        rtol=ALTERNATE_ENGINE_CASE.tolerance,
        atol=ALTERNATE_ENGINE_CASE.tolerance,
    )


def test_parakeet_flash_attention_masks_extreme_valid_logits(
    attention_engine: AttentionEngine,
) -> None:
    """Keep padding below a valid logit beyond the historical mask floor."""

    case = attention_engine.case
    qkv = np.zeros((1, 2, 3 * case.channels), dtype=np.float32)
    for head in range(case.num_heads):
        channel = head * case.head_dim
        qkv[0, :, channel] = 1.0
        qkv[0, 0, case.channels + channel] = -1200.0 / case.scale
    qkv[0, 0, 2 * case.channels :] = 1.0
    inputs = AttentionInputs(
        qkv,
        np.zeros((1, 3, case.channels), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.array((1,), dtype=np.int32),
    )

    actual, _ = assert_run_matches_reference(
        run_engine(attention_engine, inputs),
        case,
    )
    np.testing.assert_allclose(
        actual,
        np.ones_like(actual),
        rtol=case.tolerance,
        atol=case.tolerance,
    )


def test_parakeet_flash_attention_applies_content_attention_and_bias(
    attention_engine: AttentionEngine,
) -> None:
    """Exercise Q/K scoring and make the content bias select one query's key."""

    case = attention_engine.case
    sequence_length = 3
    qkv = np.zeros((1, sequence_length, 3 * case.channels), dtype=np.float32)
    qkv[0, :, 2 * case.channels :] = np.arange(
        1,
        sequence_length + 1,
        dtype=np.float32,
    )[:, None]
    for head in range(case.num_heads):
        head_start = head * case.head_dim
        for frame in range(sequence_length):
            if frame < 2:
                qkv[0, frame, head_start + frame] = 10.0
            qkv[0, frame, case.channels + head_start + frame] = 20.0
    content_bias = np.zeros((case.num_heads, case.head_dim), dtype=np.float32)
    content_bias[:, 2] = 5.0
    inputs = AttentionInputs(
        qkv,
        np.zeros((1, 5, case.channels), dtype=np.float32),
        content_bias,
        np.zeros_like(content_bias),
        np.array((sequence_length,), dtype=np.int32),
    )

    actual, _ = assert_run_matches_reference(
        run_engine(attention_engine, inputs),
        case,
    )
    expected = np.broadcast_to(
        np.arange(1, sequence_length + 1, dtype=np.float32)[None, :, None],
        actual.shape,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=case.tolerance,
        atol=case.tolerance,
    )


def test_parakeet_flash_attention_preserves_head_and_value_layout(
    attention_engine: AttentionEngine,
) -> None:
    """Expose head permutation and value-stride defects with uniform attention."""

    case = attention_engine.case
    batch_size, sequence_length, valid_length = 1, 7, 5
    qkv = np.zeros(
        (batch_size, sequence_length, 3 * case.channels),
        dtype=np.float32,
    )
    for key in range(sequence_length):
        for head in range(case.num_heads):
            start = 2 * case.channels + head * case.head_dim
            qkv[0, key, start : start + case.head_dim] = head * 10 + key
    inputs = AttentionInputs(
        qkv,
        np.zeros((1, 2 * sequence_length - 1, case.channels), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.array((valid_length,), dtype=np.int32),
    )

    actual, _ = assert_run_matches_reference(
        run_engine(attention_engine, inputs),
        case,
    )
    expected = np.empty_like(actual)
    for head in range(case.num_heads):
        start = head * case.head_dim
        expected[:, :, start : start + case.head_dim] = head * 10 + 2
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=case.tolerance,
        atol=case.tolerance,
    )


def test_parakeet_flash_attention_applies_relative_position_shift(
    attention_engine: AttentionEngine,
) -> None:
    """Align the center relative position with each query's matching key."""

    case = attention_engine.case
    sequence_length = 3
    qkv = np.zeros((1, sequence_length, 3 * case.channels), dtype=np.float32)
    for key in range(sequence_length):
        qkv[0, key, 2 * case.channels :] = key + 1
    position = np.full((1, 5, case.channels), -50.0, dtype=np.float32)
    for head in range(case.num_heads):
        position[0, sequence_length - 1, head * case.head_dim] = 50.0
    position_bias = np.zeros((case.num_heads, case.head_dim), dtype=np.float32)
    position_bias[:, 0] = 1.0
    inputs = AttentionInputs(
        qkv,
        position,
        np.zeros_like(position_bias),
        position_bias,
        np.array((sequence_length,), dtype=np.int32),
    )

    actual, _ = assert_run_matches_reference(
        run_engine(attention_engine, inputs),
        case,
    )
    expected = np.broadcast_to(
        np.arange(1, sequence_length + 1, dtype=np.float32)[None, :, None],
        actual.shape,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=case.tolerance,
        atol=case.tolerance,
    )


def test_parakeet_flash_attention_supports_cuda_graph_replay(
    attention_engine: AttentionEngine,
) -> None:
    """Capture, mutate every input and output, then replay twice."""

    case = attention_engine.case
    inputs = make_inputs(case, 2, 17, (17, 5), seed=8001)
    run = run_engine(attention_engine, inputs)

    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)

    for replay, valid_lengths in enumerate(((16, 7), (0, 17))):
        replay_inputs = make_inputs(
            case,
            2,
            17,
            valid_lengths,
            seed=8100 + replay,
        )
        with run.stream:
            for destination, (_, source) in zip(
                (
                    run.qkv,
                    run.position,
                    run.content_bias,
                    run.position_bias,
                    run.valid_lengths,
                ),
                input_items(replay_inputs),
                strict=True,
            ):
                cp.copyto(destination, cp.asarray(source, dtype=destination.dtype))
            run.output.fill(cp.nan)
            graph.launch(run.stream)

        assert_run_matches_reference(run, case)


def test_parakeet_flash_attention_reuses_context_across_shapes_and_streams(
    attention_engine: AttentionEngine,
) -> None:
    """Reuse one context while shape-dependent workspace and streams change."""

    context = attention_engine.engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    assert streams[0].ptr != streams[1].ptr
    shape_cases = (
        (1, 1, (1,)),
        (3, 65, (INT32_MIN, 34, INT32_MAX)),
        (1, 33, (32,)),
    )

    for index, (batch_size, sequence_length, valid_lengths) in enumerate(shape_cases):
        stream = streams[index % len(streams)]
        run = run_engine(
            attention_engine,
            make_inputs(
                attention_engine.case,
                batch_size,
                sequence_length,
                valid_lengths,
                seed=9000 + index,
            ),
            context=context,
            stream=stream,
        )
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_reference(run, attention_engine.case)


def test_parakeet_flash_attention_supports_concurrent_contexts(
    attention_engine: AttentionEngine,
) -> None:
    """Keep independent cuBLAS state for overlapping execution contexts."""

    first = run_engine(
        attention_engine,
        make_inputs(attention_engine.case, 1, 33, (31,), seed=10001),
        synchronize=False,
    )
    second = run_engine(
        attention_engine,
        make_inputs(
            attention_engine.case,
            3,
            65,
            (65, 34, 1),
            seed=10002,
        ),
        synchronize=False,
    )

    assert first.context is not second.context
    assert first.stream.ptr != second.stream.ptr
    assert_run_matches_reference(first, attention_engine.case)
    assert_run_matches_reference(second, attention_engine.case)


@pytest.mark.parametrize(
    "invalid_relationship",
    ("position-length", "length-batch"),
)
def test_parakeet_flash_attention_rejects_runtime_shape_mismatch(
    attention_engine: AttentionEngine,
    invalid_relationship: str,
) -> None:
    """Reject invalid concrete shapes that each fit the dynamic profile."""

    inputs = make_inputs(attention_engine.case, 2, 17, (17, 5), seed=11000)
    if invalid_relationship == "position-length":
        inputs = replace(inputs, position=inputs.position[:, :-1])
    else:
        assert invalid_relationship == "length-batch"
        inputs = replace(inputs, valid_lengths=inputs.valid_lengths[:1])
    run = prepare_engine_run(attention_engine, inputs)
    with run.stream:
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()

    assert not executed
    assert bool(cp.isnan(run.output).all())


def attention_input_specs(
    *,
    qkv_shape: tuple[int, ...] = (2, 7, 3 * DEFAULT_NUM_HEADS * DEFAULT_HEAD_DIM),
    position_shape: tuple[int, ...] = (
        1,
        13,
        DEFAULT_NUM_HEADS * DEFAULT_HEAD_DIM,
    ),
    content_bias_shape: tuple[int, ...] = (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM),
    position_bias_shape: tuple[int, ...] = (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM),
    length_shape: tuple[int, ...] = (2,),
    dtypes: tuple[object, ...] | None = None,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Create one static five-input TensorRT plugin contract."""

    if dtypes is None:
        dtypes = (trt.float16, trt.float16, trt.float16, trt.float16, trt.int32)
    return tuple(
        zip(
            dtypes,
            (
                qkv_shape,
                position_shape,
                content_bias_shape,
                position_bias_shape,
                length_shape,
            ),
            strict=True,
        )
    )


DEFAULT_CHANNELS = DEFAULT_NUM_HEADS * DEFAULT_HEAD_DIM
INVALID_CONTRACT_CASES = (
    pytest.param(attention_input_specs(qkv_shape=(2, 7)), id="qkv-rank"),
    pytest.param(
        attention_input_specs(position_shape=(1, 13)),
        id="position-rank",
    ),
    pytest.param(
        attention_input_specs(content_bias_shape=(DEFAULT_CHANNELS,)),
        id="content-bias-rank",
    ),
    pytest.param(
        attention_input_specs(
            position_bias_shape=(1, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM)
        ),
        id="position-bias-rank",
    ),
    pytest.param(
        attention_input_specs(length_shape=(2, 1)),
        id="length-rank",
    ),
    pytest.param(
        attention_input_specs(dtypes=(trt.int32,) * 5),
        id="unsupported-numeric-dtype",
    ),
    pytest.param(
        attention_input_specs(
            dtypes=(trt.float16, trt.float32, trt.float16, trt.float16, trt.int32)
        ),
        id="position-dtype",
    ),
    pytest.param(
        attention_input_specs(
            dtypes=(trt.float16, trt.float16, trt.float32, trt.float16, trt.int32)
        ),
        id="content-bias-dtype",
    ),
    pytest.param(
        attention_input_specs(
            dtypes=(trt.float16, trt.float16, trt.float16, trt.float32, trt.int32)
        ),
        id="position-bias-dtype",
    ),
    pytest.param(
        attention_input_specs(dtypes=(trt.float16,) * 4 + (trt.int64,)),
        id="length-dtype",
    ),
    pytest.param(
        attention_input_specs(qkv_shape=(2, 7, 3 * DEFAULT_CHANNELS - 1)),
        id="qkv-channels",
    ),
    pytest.param(
        attention_input_specs(position_shape=(2, 13, DEFAULT_CHANNELS)),
        id="position-batch",
    ),
    pytest.param(
        attention_input_specs(position_shape=(1, 12, DEFAULT_CHANNELS)),
        id="position-length",
    ),
    pytest.param(
        attention_input_specs(position_shape=(1, 13, DEFAULT_CHANNELS - 1)),
        id="position-channels",
    ),
    pytest.param(
        attention_input_specs(
            position_bias_shape=(DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM - 1)
        ),
        id="position-bias-shape-mismatch",
    ),
    pytest.param(
        attention_input_specs(
            content_bias_shape=(
                DEFAULT_NUM_HEADS * 2,
                DEFAULT_HEAD_DIM // 2,
            )
        ),
        id="content-bias-layout-mismatch",
    ),
    pytest.param(
        attention_input_specs(
            content_bias_shape=(DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM // 2),
            position_bias_shape=(DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM // 2),
        ),
        id="head-product-mismatch",
    ),
    pytest.param(
        attention_input_specs(length_shape=(1,)),
        id="length-batch",
    ),
    pytest.param(
        attention_input_specs(
            qkv_shape=(1, 513, 3 * DEFAULT_CHANNELS),
            position_shape=(1, 1025, DEFAULT_CHANNELS),
            length_shape=(1,),
        ),
        id="sequence-capacity",
    ),
    pytest.param(attention_input_specs()[:-1], id="missing-input"),
    pytest.param(
        attention_input_specs() + ((trt.float16, (1,)),),
        id="extra-input",
    ),
)


def cupy_dtype_for_tensorrt(dtype):
    """Return the CuPy storage dtype corresponding to one TensorRT dtype."""

    mapping = {
        trt.float32: cp.float32,
        trt.float16: cp.float16,
        trt.bfloat16: cp.dtype("bfloat16"),
        trt.int32: cp.int32,
        trt.int64: cp.int64,
    }
    return mapping[dtype]


@pytest.mark.parametrize("input_specs", INVALID_CONTRACT_CASES)
def test_parakeet_flash_attention_rejects_invalid_contracts(
    plugin_creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid counts, ranks, dtypes, and shape relationships."""

    _, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    names = tuple(
        INPUT_NAMES[index] if index < len(INPUT_NAMES) else f"extra_{index}"
        for index in range(len(input_specs))
    )
    inputs = [
        network.add_input(name, dtype, shape)
        for name, (dtype, shape) in zip(names, input_specs, strict=True)
    ]
    assert all(tensor is not None for tensor in inputs)
    layer = network.add_plugin_v3(
        inputs,
        [],
        make_plugin(creator, DEFAULT_SCALE),
    )
    if layer is None:
        return
    output = layer.get_output(0)
    output.name = "output"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        return
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    buffers = []
    with stream:
        for name, (dtype, shape) in zip(names, input_specs, strict=True):
            buffer = cp.zeros(shape, dtype=cupy_dtype_for_tensorrt(dtype))
            assert context.set_tensor_address(name, buffer.data.ptr)
            buffers.append(buffer)
        output_shape = tuple(context.get_tensor_shape("output"))
        assert all(dimension > 0 for dimension in output_shape)
        output_dtype = engine.get_tensor_dtype("output")
        output_cupy_dtype = cupy_dtype_for_tensorrt(output_dtype)
        if output_dtype in (trt.float32, trt.float16, trt.bfloat16):
            output = cp.full(output_shape, cp.nan, dtype=output_cupy_dtype)
        else:
            output = cp.full(output_shape, INT32_MIN, dtype=output_cupy_dtype)
        assert context.set_tensor_address("output", output.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    if output_dtype in (trt.float32, trt.float16, trt.bfloat16):
        assert bool(cp.isnan(output).all())
    else:
        assert bool((output == INT32_MIN).all())


def test_parakeet_flash_attention_rejects_profile_above_capacity(
    plugin_creator,
) -> None:
    """Reject a dynamic profile whose maximum exceeds 512 frames."""

    _, creator = plugin_creator
    case = ENGINE_CASES[1]
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    add_dynamic_attention_layer(network, creator, case)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "qkv",
        (1, 1, 3 * case.channels),
        (2, 17, 3 * case.channels),
        (3, 513, 3 * case.channels),
    )
    profile.set_shape(
        "position",
        (1, 1, case.channels),
        (1, 33, case.channels),
        (1, 1025, case.channels),
    )
    profile.set_shape("valid_lengths", (1,), (2,), (3,))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    assert config.add_optimization_profile(profile) == 0
    assert builder.build_serialized_network(network, config) is None


def invalid_scale_fields(
    case: str,
) -> tuple[trt.PluginFieldCollection, tuple[np.ndarray, ...]]:
    """Build one malformed scale collection and retain its backing arrays."""

    if case == "missing":
        return trt.PluginFieldCollection([]), ()
    if case == "wrong-name":
        values = (np.array([DEFAULT_SCALE], dtype=np.float32),)
        fields = [trt.PluginField("wrong", values[0], trt.PluginFieldType.FLOAT32)]
    elif case == "wrong-type":
        values = (np.array([1], dtype=np.int32),)
        fields = [trt.PluginField("scale", values[0], trt.PluginFieldType.INT32)]
    elif case == "empty":
        values = (np.array([], dtype=np.float32),)
        fields = [trt.PluginField("scale", values[0], trt.PluginFieldType.FLOAT32)]
    elif case == "multiple":
        values = (np.array([0.5, 1.0], dtype=np.float32),)
        fields = [trt.PluginField("scale", values[0], trt.PluginFieldType.FLOAT32)]
    elif case == "duplicate":
        values = (
            np.array([0.5], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        fields = [
            trt.PluginField("scale", value, trt.PluginFieldType.FLOAT32)
            for value in values
        ]
    elif case == "unknown-extra":
        values = (
            np.array([DEFAULT_SCALE], dtype=np.float32),
            np.array([1], dtype=np.int32),
        )
        fields = [
            trt.PluginField("scale", values[0], trt.PluginFieldType.FLOAT32),
            trt.PluginField("metadata", values[1], trt.PluginFieldType.INT32),
        ]
    else:
        invalid_values = {
            "zero": 0.0,
            "negative": -1.0,
            "nan": float("nan"),
            "infinity": float("inf"),
            "scaled-overflow": np.finfo(np.float32).max,
        }
        values = (np.array([invalid_values[case]], dtype=np.float32),)
        fields = [trt.PluginField("scale", values[0], trt.PluginFieldType.FLOAT32)]
    return trt.PluginFieldCollection(fields), values


@pytest.mark.parametrize(
    "field_case",
    (
        "missing",
        "wrong-name",
        "wrong-type",
        "empty",
        "multiple",
        "duplicate",
        "unknown-extra",
        "zero",
        "negative",
        "nan",
        "infinity",
        "scaled-overflow",
    ),
)
def test_parakeet_flash_attention_creator_rejects_invalid_scale_fields(
    plugin_creator,
    field_case: str,
) -> None:
    """Reject missing, malformed, nonfinite, and nonpositive scale fields."""

    _, creator = plugin_creator
    fields, _backing_values = invalid_scale_fields(field_case)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        fields,
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


@pytest.mark.parametrize(
    "scale",
    (
        pytest.param(np.finfo(np.float32).tiny, id="smallest-normal"),
        pytest.param(np.float32(1e38), id="large-finite"),
    ),
)
def test_parakeet_flash_attention_creator_accepts_positive_scale_boundaries(
    plugin_creator,
    scale: float,
) -> None:
    """Accept positive scales while their base-2 conversion remains finite."""

    _, creator = plugin_creator
    plugin = make_plugin(creator, scale)
    assert plugin is not None
