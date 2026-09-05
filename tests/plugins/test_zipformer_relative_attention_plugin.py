#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer relative-attention plugin."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

pytestmark = pytest.mark.cuda

PLUGIN_NAME = "zipformer_relative_attention"
PLUGIN_VERSION = "1"
PADDED_QUERY_HALO = 7
DEFAULT_NUM_HEADS = 4
DEFAULT_QUERY_DIM = 32
POSITION_DIM = 4
DEFAULT_PROJECTION_DIM = DEFAULT_NUM_HEADS * (2 * DEFAULT_QUERY_DIM + POSITION_DIM)
KERNEL_BOUNDARIES = (384, 385, 512, 513, 1024, 1025, 2048, 2049)
SHAPE_CASES = ((1, 1), (1, 7), (2, 65)) + tuple(
    (1, length) for length in KERNEL_BOUNDARIES
)
MASKING_KERNEL_CASES = (65, 1025, 2049)
ARCHITECTURE_CASES = ((1, 1), (3, 5), (8, 32))


@dataclass(frozen=True)
class DTypeCase:
    """TensorRT, CuPy, and reference settings for one numeric dtype."""

    name: str
    trt_dtype: trt.DataType
    cupy_dtype: type[np.generic] | np.dtype[np.generic]
    torch_dtype: torch.dtype
    element_tolerance: float
    max_row_l1_error: float


DTYPE_CASES = (
    DTypeCase("fp32", trt.float32, cp.float32, torch.float32, 2e-3, 2e-2),
    DTypeCase("fp16", trt.float16, cp.float16, torch.float16, 5e-4, 5e-3),
    pytest.param(
        DTypeCase(
            "bf16", trt.bfloat16, cp.dtype("bfloat16"), torch.bfloat16, 2e-3, 2e-2
        ),
        marks=pytest.mark.sm80,
    ),
)


type PluginCreatorFixture = tuple[ctypes.CDLL, trt.IPluginCreatorV3One]
type RelativeAttentionEngine = tuple[trt.Runtime, trt.ICudaEngine, DTypeCase]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory) -> PluginCreatorFixture:
    """Compile and register the current relative-attention plugin source.

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
        "zipformer_relative_attention_plugin.cu",
        "initFastGpuAsrZipformerRelativeAttentionPlugin",
        ("cublas", "cudart"),
    )

    registry = trt.get_builder_plugin_registry(trt.EngineCapability.STANDARD)
    creator = registry.get_creator(
        PLUGIN_NAME, PLUGIN_VERSION, TENSORRT_PLUGIN_NAMESPACE
    )
    assert creator is not None
    return library, creator


def make_plugin(creator: trt.IPluginCreatorV3One) -> trt.IPluginV3:
    """Create the parameter-free relative-attention plugin.

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


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=lambda case: case.name)
def relative_attention_engine(
    request: pytest.FixtureRequest, plugin_creator: PluginCreatorFixture
) -> RelativeAttentionEngine:
    """Build one dynamic relative-attention engine for a supported dtype.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    RelativeAttentionEngine
        Deserialized engine with its owning runtime.
    """

    _, creator = plugin_creator
    dtype_case: DTypeCase = request.param
    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    projection = network.add_input("projection", dtype_case.trt_dtype, (-1, -1, -1))
    position = network.add_input(
        "position", dtype_case.trt_dtype, (1, -1, -1, POSITION_DIM)
    )
    mask = network.add_input("mask", trt.bool, (-1, -1))
    assert projection is not None and position is not None and mask is not None

    layer = network.add_plugin_v3(
        [projection, position, mask], [], make_plugin(creator)
    )
    assert layer is not None
    scores = layer.get_output(0)
    scores.name = "scores"
    network.mark_output(scores)

    # Independent profile bounds need not satisfy the exact runtime relationships.
    profile_shapes = {
        "projection": (
            (1, 1, 2 * 1 + POSITION_DIM),
            (2, 65, DEFAULT_PROJECTION_DIM),
            (2, 2049, 8 * (2 * 32 + POSITION_DIM)),
        ),
        "position": (
            (1, 1, 1, POSITION_DIM),
            (1, 129, DEFAULT_NUM_HEADS, POSITION_DIM),
            (1, 4098, 8, POSITION_DIM),
        ),
        "mask": ((1, 1), (2, 65), (3, 2050)),
    }
    profile = builder.create_optimization_profile()
    for name, shapes in profile_shapes.items():
        profile.set_shape(name, *shapes)
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
        "projection": (trt.TensorIOMode.INPUT, dtype_case.trt_dtype),
        "position": (trt.TensorIOMode.INPUT, dtype_case.trt_dtype),
        "mask": (trt.TensorIOMode.INPUT, trt.bool),
        "scores": (trt.TensorIOMode.OUTPUT, dtype_case.trt_dtype),
    }
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return runtime, engine, dtype_case


def make_inputs(
    batch_size: int,
    sequence_length: int,
    num_heads: int = DEFAULT_NUM_HEADS,
    query_dim: int = DEFAULT_QUERY_DIM,
) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    """Create deterministic projections, positions, and key-padding masks.

    Parameters
    ----------
    batch_size : int
        Number of utterances in the batch.
    sequence_length : int
        Physical number of frames per utterance.
    num_heads : int
        Number of independently indexed attention heads.
    query_dim : int
        Number of query/key channels per head.

    Returns
    -------
    tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]
        FP32 packed projections, FP32 relative positions, and a Boolean padding
        mask.
    """

    rng = np.random.default_rng(1000 + batch_size * 100 + sequence_length)
    projection_dim = num_heads * (2 * query_dim + POSITION_DIM)
    projection = rng.normal(
        0.0, 0.25, (batch_size, sequence_length, projection_dim)
    ).astype(np.float32)
    position = rng.normal(
        0.0, 0.25, (1, 2 * sequence_length - 1, num_heads, POSITION_DIM)
    ).astype(np.float32)
    mask = np.zeros((batch_size, sequence_length), dtype=np.bool_)
    for batch in range(batch_size):
        valid_length = max(1, sequence_length - 9 - 3 * batch)
        mask[batch, valid_length:] = True
    return projection, position, mask


def reference_attention(
    projection: np.typing.NDArray,
    position: np.typing.NDArray,
    mask: np.typing.NDArray,
    torch_dtype: torch.dtype,
) -> np.typing.NDArray:
    """Evaluate dtype-rounded inputs using explicit query-key relative offsets.

    Parameters
    ----------
    projection : np.typing.NDArray
        Packed query, key, and positional-query projections in NTC layout.
    position : np.typing.NDArray
        Relative embeddings with shape (1, 2 * time - 1, heads, 4).
    mask : np.typing.NDArray
        Boolean key-padding mask of shape (batch, time), with True marking padding.
    torch_dtype : torch.dtype
        Input/GEMM storage precision used by the independent CPU reference.

    Returns
    -------
    np.typing.NDArray
        Dtype-rounded attention probabilities converted to FP32 in NHTT layout.
    """

    projection_t = torch.from_numpy(projection).to(torch_dtype)
    position_t = torch.from_numpy(position).to(torch_dtype)
    mask_t = torch.from_numpy(mask)
    batch_size, sequence_length, projection_dim = projection.shape
    num_heads = position.shape[2]
    query_dim = (projection_dim // num_heads - POSITION_DIM) // 2
    content_dim = num_heads * query_dim
    query, key, position_query = projection_t.split(
        (content_dim, content_dim, num_heads * POSITION_DIM), dim=2
    )
    query = query.reshape(batch_size, sequence_length, num_heads, query_dim).permute(
        0, 2, 1, 3
    )
    key = key.reshape(batch_size, sequence_length, num_heads, query_dim).permute(
        0, 2, 3, 1
    )
    position_query = position_query.reshape(
        batch_size, sequence_length, num_heads, POSITION_DIM
    ).permute(0, 2, 1, 3)
    position_scores = position_query @ position_t.permute(0, 2, 3, 1)
    queries = torch.arange(sequence_length)[:, None]
    keys = torch.arange(sequence_length)[None, :]
    relative_indices = sequence_length - 1 - queries + keys
    position_scores = torch.gather(
        position_scores,
        3,
        relative_indices[None, None].expand(
            batch_size, num_heads, sequence_length, sequence_length
        ),
    )
    scores = query @ key + position_scores
    expanded_mask = mask_t[:, None, None, :]
    scores = scores.masked_fill(expanded_mask, float("-inf")).softmax(dim=3)
    scores = scores.masked_fill(expanded_mask, 0.0)
    if sequence_length > PADDED_QUERY_HALO:
        query_mask = torch.zeros_like(mask_t)
        query_mask[:, PADDED_QUERY_HALO:] = mask_t[
            :, : sequence_length - PADDED_QUERY_HALO
        ]
        scores = scores.masked_fill(query_mask[:, None, :, None], 0.0)
    return scores.float().numpy()


@dataclass
class EngineRun:
    """Device buffers and execution state retained after one inference."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    projection: cp.ndarray
    position: cp.ndarray
    mask: cp.ndarray
    scores: cp.ndarray

    def host_scores(self) -> np.typing.NDArray:
        """Synchronize and return scores as FP32 for numerical assertions.

        Returns
        -------
        np.typing.NDArray
            Completed attention probabilities in NHTT layout, converted to host FP32.
        """

        self.stream.synchronize()
        return cp.asnumpy(self.scores).astype(np.float32)


def run_engine(
    engine: trt.ICudaEngine,
    dtype_case: DTypeCase,
    inputs: tuple[np.typing.NDArray, ...],
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    execute: bool = True,
) -> EngineRun:
    """Bind owned buffers and optionally enqueue; host_scores waits for completion.

    Parameters
    ----------
    engine : trt.ICudaEngine
        Deserialized engine whose runtime must remain alive during execution.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.
    inputs : tuple[np.typing.NDArray, ...]
        Host projection, relative embeddings, and Boolean key-padding mask.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    execute : bool
        Whether to enqueue inference after binding; False permits rejection tests.

    Returns
    -------
    EngineRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    for name, value in zip(("projection", "position", "mask"), inputs, strict=True):
        assert context.set_input_shape(name, value.shape)
    assert context.infer_shapes() == []
    projection, position, _ = inputs
    output_shape = (
        projection.shape[0],
        position.shape[2],
        projection.shape[1],
        projection.shape[1],
    )
    assert tuple(context.get_tensor_shape("scores")) == output_shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        buffers = [
            cp.array(value, dtype=dtype)
            for value, dtype in zip(
                inputs,
                (dtype_case.cupy_dtype, dtype_case.cupy_dtype, cp.bool_),
                strict=True,
            )
        ]
        scores = cp.full(output_shape, cp.nan, dtype=dtype_case.cupy_dtype)
        for name, buffer in zip(
            ("projection", "position", "mask", "scores"),
            (*buffers, scores),
            strict=True,
        ):
            assert context.set_tensor_address(name, buffer.data.ptr)
        if execute:
            assert context.execute_async_v3(stream.ptr)
    return EngineRun(context, stream, *buffers, scores)


def assert_run_preserves_inputs(
    run: EngineRun, dtype_case: DTypeCase, inputs: tuple[np.typing.NDArray, ...]
) -> None:
    """Compare completed device inputs with their original dtype-rounded values.

    Parameters
    ----------
    run : EngineRun
        Bound device buffers and the context/stream that own their pending work.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.
    inputs : tuple[np.typing.NDArray, ...]
        Host projection, relative embeddings, and Boolean key-padding mask.

    Notes
    -----
    The caller must synchronize the run's stream before invoking this helper.
    """

    for actual, expected in zip(
        (run.projection, run.position), inputs[:2], strict=True
    ):
        expected = torch.from_numpy(expected).to(dtype_case.torch_dtype).float().numpy()
        np.testing.assert_array_equal(cp.asnumpy(actual).astype(np.float32), expected)
    np.testing.assert_array_equal(cp.asnumpy(run.mask), inputs[2])


def assert_run_matches_reference(
    run: EngineRun, dtype_case: DTypeCase, inputs: tuple[np.typing.NDArray, ...]
) -> np.typing.NDArray:
    """Check immutable inputs, attention values, row sums, and exact masking.

    Parameters
    ----------
    run : EngineRun
        Bound device buffers and the context/stream that own their pending work.
    dtype_case : DTypeCase
        Device storage dtype, CPU rounding dtype, and numerical tolerance.
    inputs : tuple[np.typing.NDArray, ...]
        Host projection, relative embeddings, and Boolean key-padding mask.

    Returns
    -------
    np.typing.NDArray
        Completed FP32 attention probabilities with shape (batch, heads, time,
        time).
    """

    actual = run.host_scores()
    assert_run_preserves_inputs(run, dtype_case, inputs)
    expected = reference_attention(*inputs, dtype_case.torch_dtype)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=dtype_case.element_tolerance,
        atol=dtype_case.element_tolerance,
    )
    row_l1_error = np.abs(actual - expected).sum(axis=3).max()
    assert row_l1_error <= dtype_case.max_row_l1_error, row_l1_error
    for batch, mask in enumerate(inputs[2]):
        valid_length = np.count_nonzero(~mask)
        assert not mask[:valid_length].any() and mask[valid_length:].all()
        live_queries = min(actual.shape[2], valid_length + PADDED_QUERY_HALO)
        assert np.count_nonzero(actual[batch][..., mask]) == 0
        assert np.count_nonzero(actual[batch, :, live_queries:]) == 0
        if valid_length:
            np.testing.assert_allclose(
                actual[batch, :, :live_queries].sum(axis=2), 1, rtol=5e-3, atol=5e-3
            )
    return actual


def test_relative_attention_plugin_excludes_scores_below_old_sentinel(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    projection = np.zeros((1, 2, DEFAULT_PROJECTION_DIM), dtype=np.float32)
    content_dim = DEFAULT_NUM_HEADS * DEFAULT_QUERY_DIM
    projection[:, :, :content_dim] = 50.0
    projection[:, 0, content_dim : 2 * content_dim] = -40.0
    position = np.zeros((1, 3, DEFAULT_NUM_HEADS, POSITION_DIM), dtype=np.float32)
    mask = np.array(((False, True),), dtype=np.bool_)

    run = run_engine(engine, dtype_case, (projection, position, mask))
    actual = assert_run_matches_reference(run, dtype_case, (projection, position, mask))
    np.testing.assert_array_equal(actual[..., 0], 1)
    np.testing.assert_array_equal(actual[..., 1], 0)


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_relative_attention_plugin_matches_pytorch(
    relative_attention_engine: RelativeAttentionEngine,
    batch_size: int,
    sequence_length: int,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    inputs = make_inputs(batch_size, sequence_length)
    run = run_engine(engine, dtype_case, inputs)
    assert_run_matches_reference(run, dtype_case, inputs)


def test_relative_attention_plugin_applies_relative_offset(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    sequence_length = 17
    relative_offset = 2
    content_dim = DEFAULT_NUM_HEADS * DEFAULT_QUERY_DIM
    projection = np.zeros(
        (1, sequence_length, DEFAULT_PROJECTION_DIM), dtype=np.float32
    )
    for head in range(DEFAULT_NUM_HEADS):
        projection[:, :, 2 * content_dim + head * POSITION_DIM] = 20.0
    position = np.zeros(
        (1, 2 * sequence_length - 1, DEFAULT_NUM_HEADS, POSITION_DIM), dtype=np.float32
    )
    position[:, sequence_length - 1 + relative_offset, :, 0] = 20.0
    mask = np.zeros((1, sequence_length), dtype=np.bool_)

    run = run_engine(engine, dtype_case, (projection, position, mask))
    actual = assert_run_matches_reference(run, dtype_case, (projection, position, mask))
    expected = np.zeros(
        (1, DEFAULT_NUM_HEADS, sequence_length - relative_offset, sequence_length),
        dtype=np.float32,
    )
    query_indices = np.arange(sequence_length - relative_offset)
    expected[:, :, query_indices, query_indices + relative_offset] = 1.0
    np.testing.assert_array_equal(
        actual[:, :, : sequence_length - relative_offset], expected
    )


@pytest.mark.parametrize("sequence_length", KERNEL_BOUNDARIES)
def test_relative_attention_plugin_exercises_every_kernel_boundary_key(
    relative_attention_engine: RelativeAttentionEngine, sequence_length: int
) -> None:
    _, engine, dtype_case = relative_attention_engine
    content_dim = DEFAULT_NUM_HEADS * DEFAULT_QUERY_DIM
    projection = np.zeros(
        (1, sequence_length, DEFAULT_PROJECTION_DIM), dtype=np.float32
    )
    for head in range(DEFAULT_NUM_HEADS):
        channel = head * DEFAULT_QUERY_DIM
        projection[0, :, channel] = 20.0
        projection[0, sequence_length - 1, content_dim + channel] = 20.0
    position = np.zeros(
        (1, 2 * sequence_length - 1, DEFAULT_NUM_HEADS, POSITION_DIM), dtype=np.float32
    )
    mask = np.zeros((1, sequence_length), dtype=np.bool_)

    run = run_engine(engine, dtype_case, (projection, position, mask))
    actual = run.host_scores()

    assert_run_preserves_inputs(run, dtype_case, (projection, position, mask))
    np.testing.assert_array_equal(actual[..., sequence_length - 1], 1)
    assert np.count_nonzero(actual[..., : sequence_length - 1]) == 0


@pytest.mark.parametrize("sequence_length", MASKING_KERNEL_CASES)
def test_relative_attention_plugin_contains_nonfinite_padding(
    relative_attention_engine: RelativeAttentionEngine, sequence_length: int
) -> None:
    _, engine, dtype_case = relative_attention_engine
    projection, position, mask = make_inputs(1, sequence_length)
    content_dim = DEFAULT_NUM_HEADS * DEFAULT_QUERY_DIM
    projection[0, mask[0], content_dim : 2 * content_dim] = np.nan
    query_padding_mask = np.zeros(sequence_length, dtype=np.bool_)
    query_padding_mask[PADDED_QUERY_HALO:] = mask[
        0, : sequence_length - PADDED_QUERY_HALO
    ]
    projection[0, query_padding_mask] = np.nan

    run = run_engine(engine, dtype_case, (projection, position, mask))

    assert_run_matches_reference(run, dtype_case, (projection, position, mask))


@pytest.mark.parametrize(
    ("num_heads", "query_dim"),
    ARCHITECTURE_CASES,
    ids=(
        "minimum-1-head-query-1",
        "generic-3-head-query-5",
        "production-8-head-query-32",
    ),
)
def test_relative_attention_plugin_supports_head_layouts(
    relative_attention_engine: RelativeAttentionEngine, num_heads: int, query_dim: int
) -> None:
    _, engine, dtype_case = relative_attention_engine
    inputs = make_inputs(2, 17, num_heads, query_dim)
    run = run_engine(engine, dtype_case, inputs)
    assert_run_matches_reference(run, dtype_case, inputs)


def test_relative_attention_plugin_matches_pytorch_without_padding(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    projection, position, _ = make_inputs(1, 385)
    mask = np.zeros((1, 385), dtype=np.bool_)
    run = run_engine(engine, dtype_case, (projection, position, mask))

    assert_run_matches_reference(run, dtype_case, (projection, position, mask))


@pytest.mark.parametrize("sequence_length", MASKING_KERNEL_CASES)
def test_relative_attention_plugin_handles_fully_padded_sequence(
    relative_attention_engine: RelativeAttentionEngine, sequence_length: int
) -> None:
    _, engine, dtype_case = relative_attention_engine
    projection, position, _ = make_inputs(1, sequence_length)
    mask = np.ones((1, sequence_length), dtype=np.bool_)
    run = run_engine(engine, dtype_case, (projection, position, mask))

    assert_run_matches_reference(run, dtype_case, (projection, position, mask))


def test_relative_attention_plugin_supports_cuda_graphs(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    inputs = make_inputs(2, 65)
    run = run_engine(engine, dtype_case, inputs)
    run.stream.synchronize()
    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)
    projection, position, mask = inputs
    for replay in (1, 2):
        replay_mask = np.zeros_like(mask)
        replay_mask[0, 50 - replay :] = True
        replay_mask[1, 40 - replay :] = True
        replay_inputs = (
            projection * (1 + 0.25 * replay) + 0.01 * replay,
            position * (1 - 0.125 * replay),
            replay_mask,
        )
        with run.stream:
            for buffer, value in zip(
                (run.projection, run.position, run.mask), replay_inputs, strict=True
            ):
                cp.copyto(buffer, cp.array(value, dtype=buffer.dtype))
            run.scores.fill(cp.nan)
            graph.launch(run.stream)
        assert_run_matches_reference(run, dtype_case, replay_inputs)


def test_relative_attention_plugin_reuses_context_across_dynamic_shapes(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    for batch_size, sequence_length in ((1, 384), (2, 513), (1, 384)):
        inputs = make_inputs(batch_size, sequence_length)
        run = run_engine(engine, dtype_case, inputs, context, stream)
        assert run.context is context
        assert_run_matches_reference(run, dtype_case, inputs)


def test_relative_attention_plugin_handles_stream_changes(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    context = engine.create_execution_context()
    assert context is not None
    inputs = make_inputs(2, 65)
    first = run_engine(engine, dtype_case, inputs, context)
    assert_run_matches_reference(first, dtype_case, inputs)

    projection, position, mask = inputs
    replay_mask = np.zeros_like(mask)
    replay_mask[0, 51:] = True
    replay_mask[1, 39:] = True
    replay_inputs = (projection * 1.25 + 0.125, position * -0.75, replay_mask)
    second = run_engine(engine, dtype_case, replay_inputs, context)
    assert first.stream.ptr != second.stream.ptr
    assert second.context is context
    assert_run_matches_reference(second, dtype_case, replay_inputs)


def test_relative_attention_plugin_supports_concurrent_contexts(
    relative_attention_engine: RelativeAttentionEngine,
) -> None:
    _, engine, dtype_case = relative_attention_engine
    inputs = (make_inputs(1, 65), make_inputs(2, 65))
    runs = [run_engine(engine, dtype_case, values, execute=False) for values in inputs]
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr
    for run in runs:
        with run.stream:
            assert run.context.execute_async_v3(run.stream.ptr)
    for run, values in zip(runs, inputs, strict=True):
        assert_run_matches_reference(run, dtype_case, values)


@pytest.mark.parametrize(
    "input_index,shape",
    (
        pytest.param(1, (1, 128, 4, 4), id="position-length-shorter"),
        pytest.param(1, (1, 130, 4, 4), id="position-length-longer"),
        pytest.param(2, (1, 65), id="mask-batch-shorter"),
        pytest.param(2, (3, 65), id="mask-batch-longer"),
        pytest.param(2, (2, 64), id="mask-length-shorter"),
        pytest.param(2, (2, 66), id="mask-length-longer"),
        pytest.param(0, (2, 65, 271), id="projection-not-divisible-shorter"),
        pytest.param(0, (2, 65, 273), id="projection-not-divisible-longer"),
        pytest.param(0, (2, 65, 16), id="projection-without-query-dimensions"),
        pytest.param(0, (2, 65, 268), id="fractional-query-dimension-shorter"),
        pytest.param(0, (2, 65, 276), id="fractional-query-dimension-longer"),
    ),
)
def test_relative_attention_plugin_rejects_runtime_shape_mismatch(
    relative_attention_engine: RelativeAttentionEngine,
    input_index: int,
    shape: tuple[int, ...],
) -> None:
    _, engine, dtype_case = relative_attention_engine
    inputs = list(make_inputs(2, 65))
    inputs[input_index] = np.zeros(shape, dtype=inputs[input_index].dtype)
    run = run_engine(engine, dtype_case, tuple(inputs), execute=False)
    with run.stream:
        executed = run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert not executed
    assert bool(cp.isnan(run.scores).all())


def test_relative_attention_creator_exposes_parameter_free_contract(
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
    assert build.timing_cache_id

    second_plugin = make_plugin(creator)
    second_build = second_plugin.get_capability_interface(
        trt.PluginCapabilityType.BUILD
    )
    assert second_build is not None
    assert second_build.timing_cache_id == build.timing_cache_id


def test_relative_attention_creator_rejects_fields(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    field = trt.PluginField(
        "unexpected", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME, trt.PluginFieldCollection([field]), trt.TensorRTPhase.BUILD
    )
    assert plugin is None


VALID_INPUT_SPECS = (
    (trt.float16, (2, 7, DEFAULT_PROJECTION_DIM)),
    (trt.float16, (1, 13, DEFAULT_NUM_HEADS, POSITION_DIM)),
    (trt.bool, (2, 7)),
)


def static_contract_executes(
    creator: trt.IPluginCreatorV3One, input_specs: tuple[InputSpec, ...]
) -> bool:
    """Try a static contract and require untouched NaN output on execution failure.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.

    Returns
    -------
    bool
        True only if building and execution succeed; False on contract rejection.
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
    assert all(value is not None for value in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    if layer is None:
        return False
    scores = layer.get_output(0)
    scores.name = "scores"
    network.mark_output(scores)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return False

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        buffers = [
            cp.zeros(shape, dtype=trt.nptype(dtype)) for dtype, shape in input_specs
        ]
        for index, buffer in enumerate(buffers):
            assert context.set_tensor_address(f"input_{index}", buffer.data.ptr)
        output = cp.full(
            tuple(context.get_tensor_shape("scores")),
            cp.nan,
            dtype=trt.nptype(engine.get_tensor_dtype("scores")),
        )
        assert context.set_tensor_address("scores", output.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()
    assert bool(cp.isfinite(output).all() if executed else cp.isnan(output).all())
    return executed


@pytest.mark.parametrize(
    "input_specs,accepted",
    (
        pytest.param(VALID_INPUT_SPECS, True, id="valid"),
        pytest.param(VALID_INPUT_SPECS[:2], False, id="missing-input"),
        pytest.param(
            VALID_INPUT_SPECS + ((trt.float16, (1,)),), False, id="extra-input"
        ),
    ),
)
def test_relative_attention_static_input_count(
    plugin_creator: PluginCreatorFixture,
    input_specs: tuple[InputSpec, ...],
    accepted: bool,
) -> None:
    _, creator = plugin_creator
    assert static_contract_executes(creator, input_specs) is accepted


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({0: (trt.float16, (2, 7))}, id="projection-rank"),
        pytest.param({1: (trt.float16, (1, 13, 16))}, id="position-rank"),
        pytest.param({2: (trt.bool, (2, 1, 7))}, id="mask-rank"),
        pytest.param(
            {0: (trt.float16, (0, 7, 272)), 2: (trt.bool, (0, 7))}, id="empty-batch"
        ),
        pytest.param(
            {
                0: (trt.float16, (2, 0, 272)),
                1: (trt.float16, (1, 1, 4, 4)),
                2: (trt.bool, (2, 0)),
            },
            id="empty-sequence",
        ),
        pytest.param({0: (trt.float16, (2, 7, 0))}, id="empty-projection-channels"),
        pytest.param({1: (trt.float16, (1, 0, 4, 4))}, id="empty-position-length"),
        pytest.param({1: (trt.float16, (1, 13, 0, 4))}, id="empty-position-heads"),
        pytest.param({2: (trt.bool, (0, 7))}, id="empty-mask-batch"),
        pytest.param({2: (trt.bool, (2, 0))}, id="empty-mask-length"),
        pytest.param({1: (trt.float16, (2, 13, 4, 4))}, id="position-batch"),
        pytest.param({1: (trt.float16, (1, 12, 4, 4))}, id="position-length-shorter"),
        pytest.param({1: (trt.float16, (1, 14, 4, 4))}, id="position-length-longer"),
        pytest.param({1: (trt.float16, (1, 13, 4, 3))}, id="position-head-dim-shorter"),
        pytest.param({1: (trt.float16, (1, 13, 4, 5))}, id="position-head-dim-longer"),
        pytest.param({2: (trt.bool, (1, 7))}, id="mask-batch-shorter"),
        pytest.param({2: (trt.bool, (3, 7))}, id="mask-batch-longer"),
        pytest.param({2: (trt.bool, (2, 6))}, id="mask-length-shorter"),
        pytest.param({2: (trt.bool, (2, 8))}, id="mask-length-longer"),
        pytest.param(
            {0: (trt.float16, (2, 7, 271))}, id="projection-not-divisible-shorter"
        ),
        pytest.param(
            {0: (trt.float16, (2, 7, 273))}, id="projection-not-divisible-longer"
        ),
        pytest.param(
            {0: (trt.float16, (2, 7, 16))}, id="projection-without-query-dimensions"
        ),
        pytest.param(
            {0: (trt.float16, (2, 7, 268))}, id="fractional-query-dimension-shorter"
        ),
        pytest.param(
            {0: (trt.float16, (2, 7, 276))}, id="fractional-query-dimension-longer"
        ),
        pytest.param(
            {0: (trt.int32, (2, 7, 272)), 1: (trt.int32, (1, 13, 4, 4))},
            id="unsupported-numeric-dtype",
        ),
        pytest.param({1: (trt.float32, (1, 13, 4, 4))}, id="mixed-numeric-dtypes"),
        pytest.param({2: (trt.int32, (2, 7))}, id="non-boolean-mask"),
    ),
)
def test_relative_attention_plugin_rejects_invalid_contracts(
    plugin_creator: PluginCreatorFixture, overrides: dict[int, InputSpec]
) -> None:
    _, creator = plugin_creator
    input_specs = list(VALID_INPUT_SPECS)
    for index, spec in overrides.items():
        input_specs[index] = spec
    assert not static_contract_executes(creator, tuple(input_specs))
