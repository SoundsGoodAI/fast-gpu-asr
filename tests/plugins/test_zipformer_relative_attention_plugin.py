#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Zipformer relative-attention plugin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

cp = pytest.importorskip("cupy")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.cuda

PLUGIN_NAME = "zipformer_relative_attention"
PLUGIN_VERSION = "1"
PADDED_QUERY_HALO = 7
DEFAULT_NUM_HEADS = 4
DEFAULT_QUERY_DIM = 32
POSITION_DIM = 4
DEFAULT_PROJECTION_DIM = DEFAULT_NUM_HEADS * (2 * DEFAULT_QUERY_DIM + POSITION_DIM)
SHAPE_CASES = (
    (1, 7),
    (2, 65),
    (1, 384),
    (1, 385),
    (1, 512),
    (1, 513),
    (1, 1024),
    (1, 1025),
    (1, 2048),
    (1, 2049),
)
ARCHITECTURE_CASES = ((3, 5), (8, 32))


@dataclass(frozen=True)
class DTypeCase:
    """TensorRT, CuPy, and reference settings for one numeric dtype."""

    name: str
    trt_dtype: object
    cupy_dtype: object
    torch_dtype: torch.dtype
    element_tolerance: float
    max_row_l1_error: float


DTYPE_CASES = (
    DTypeCase("fp32", trt.float32, cp.float32, torch.float32, 2e-3, 2e-2),
    DTypeCase("fp16", trt.float16, cp.float16, torch.float16, 5e-4, 5e-3),
    pytest.param(
        DTypeCase(
            "bf16",
            trt.bfloat16,
            cp.dtype("bfloat16"),
            torch.bfloat16,
            2e-3,
            2e-2,
        ),
        marks=pytest.mark.sm80,
    ),
)


@pytest.fixture(scope="module")
def plugin_creator(tmp_path_factory: pytest.TempPathFactory):
    """Compile and register the current relative-attention plugin source."""

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


def make_plugin(creator):
    """Create the parameter-free relative-attention plugin."""

    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=lambda case: case.name)
def relative_attention_engine(request, plugin_creator):
    """Build one dynamic relative-attention engine for a supported dtype."""

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

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "projection",
        (1, 1, 3 * (2 * 5 + POSITION_DIM)),
        (2, 65, DEFAULT_PROJECTION_DIM),
        (2, 2049, 8 * (2 * 32 + POSITION_DIM)),
    )
    profile.set_shape(
        "position",
        (1, 1, 3, POSITION_DIM),
        (1, 129, DEFAULT_NUM_HEADS, POSITION_DIM),
        (1, 4097, 8, POSITION_DIM),
    )
    profile.set_shape("mask", (1, 1), (2, 65), (2, 2049))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = 3
    assert config.add_optimization_profile(profile) == 0
    serialized_engine = builder.build_serialized_network(network, config)
    assert serialized_engine is not None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    assert engine is not None
    assert engine.get_tensor_dtype("scores") == dtype_case.trt_dtype
    return runtime, engine, dtype_case


def make_inputs(
    batch_size: int,
    sequence_length: int,
    num_heads: int = DEFAULT_NUM_HEADS,
    query_dim: int = DEFAULT_QUERY_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic projections, positions, and key-padding masks."""

    rng = np.random.default_rng(1000 + batch_size * 100 + sequence_length)
    projection_dim = num_heads * (2 * query_dim + POSITION_DIM)
    projection = rng.normal(
        0.0, 0.25, (batch_size, sequence_length, projection_dim)
    ).astype(np.float32)
    position = rng.normal(
        0.0,
        0.25,
        (1, 2 * sequence_length - 1, num_heads, POSITION_DIM),
    ).astype(np.float32)
    mask = np.zeros((batch_size, sequence_length), dtype=np.bool_)
    for batch in range(batch_size):
        valid_length = max(1, sequence_length - 9 - 3 * batch)
        mask[batch, valid_length:] = True
    return projection, position, mask


def reference_attention(
    projection: cp.ndarray,
    position: cp.ndarray,
    mask: cp.ndarray,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    """Evaluate relative attention by directly indexing every query-key offset."""

    projection_t = torch.from_numpy(cp.asnumpy(projection).astype(np.float32)).to(
        dtype=torch_dtype
    )
    position_t = torch.from_numpy(cp.asnumpy(position).astype(np.float32)).to(
        dtype=torch_dtype
    )
    mask_t = torch.from_numpy(cp.asnumpy(mask))
    batch_size, sequence_length, projection_dim = projection_t.shape
    num_heads = position_t.shape[2]
    assert position_t.shape == (1, 2 * sequence_length - 1, num_heads, POSITION_DIM)
    assert projection_dim % num_heads == 0
    dimensions_per_head = projection_dim // num_heads
    assert dimensions_per_head > POSITION_DIM
    assert (dimensions_per_head - POSITION_DIM) % 2 == 0
    query_dim = (dimensions_per_head - POSITION_DIM) // 2
    content_dim = num_heads * query_dim

    query = (
        projection_t[:, :, :content_dim]
        .reshape(batch_size, sequence_length, num_heads, query_dim)
        .permute(0, 2, 1, 3)
    )
    key = (
        projection_t[:, :, content_dim : 2 * content_dim]
        .reshape(batch_size, sequence_length, num_heads, query_dim)
        .permute(0, 2, 3, 1)
    )
    position_query = (
        projection_t[:, :, 2 * content_dim :]
        .reshape(batch_size, sequence_length, num_heads, POSITION_DIM)
        .permute(0, 2, 1, 3)
    )
    position_scores = torch.matmul(position_query, position_t.permute(0, 2, 3, 1))
    query_indices = torch.arange(sequence_length)[:, None]
    key_indices = torch.arange(sequence_length)[None, :]
    relative_indices = sequence_length - 1 - query_indices + key_indices
    position_scores = torch.gather(
        position_scores,
        dim=3,
        index=relative_indices[None, None].expand(
            batch_size, num_heads, sequence_length, sequence_length
        ),
    )
    scores = torch.matmul(query, key) + position_scores
    expanded_mask = mask_t[:, None, None, :]
    scores = torch.softmax(
        scores.masked_fill(expanded_mask, float("-inf")), dim=3
    ).masked_fill(expanded_mask, 0.0)
    if sequence_length > PADDED_QUERY_HALO:
        query_padding_mask = torch.cat(
            (
                torch.zeros_like(mask_t[:, :PADDED_QUERY_HALO]),
                mask_t[:, :-PADDED_QUERY_HALO],
            ),
            dim=1,
        )
        scores = scores.masked_fill(query_padding_mask[:, None, :, None], 0.0)

    return scores.float().cpu().numpy()


def assert_matches_reference(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    tolerance: float,
    max_row_l1_error: float,
) -> None:
    """Check values, distribution error, source masking, and query pruning."""

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.all(actual >= 0.0)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    row_l1_error = np.abs(actual - expected).sum(axis=-1)
    assert float(row_l1_error.max(initial=0.0)) <= max_row_l1_error, (
        f"maximum row L1 error {row_l1_error.max()} exceeds {max_row_l1_error}"
    )

    sequence_length = actual.shape[2]
    for batch, batch_mask in enumerate(mask):
        padded_queries = np.flatnonzero(batch_mask)
        if padded_queries.size > 0:
            assert batch_mask[padded_queries[0] :].all(), (
                "test inputs must follow the plugin's contiguous-suffix mask contract"
            )

        masked_sources = actual[batch][..., batch_mask]
        np.testing.assert_array_equal(
            masked_sources,
            np.zeros_like(masked_sources),
        )

        query_padding_mask = np.zeros(sequence_length, dtype=np.bool_)
        if sequence_length > PADDED_QUERY_HALO:
            query_padding_mask[PADDED_QUERY_HALO:] = batch_mask[:-PADDED_QUERY_HALO]
        pruned_queries = actual[batch][:, query_padding_mask]
        np.testing.assert_array_equal(
            pruned_queries,
            np.zeros_like(pruned_queries),
        )

        if batch_mask.all():
            np.testing.assert_array_equal(
                actual[batch],
                np.zeros_like(actual[batch]),
            )
        else:
            evaluated = actual[batch][:, ~query_padding_mask]
            np.testing.assert_allclose(
                evaluated.sum(axis=-1),
                np.ones(evaluated.shape[:2], dtype=np.float32),
                rtol=5e-3,
                atol=5e-3,
            )


@dataclass
class EngineRun:
    """Device buffers and execution state retained after one inference."""

    context: object
    stream: cp.cuda.Stream
    projection: cp.ndarray
    position: cp.ndarray
    mask: cp.ndarray
    scores: cp.ndarray

    def host_scores(self) -> np.ndarray:
        """Synchronize and return scores as FP32 for numerical assertions."""

        self.stream.synchronize()
        return cp.asnumpy(self.scores).astype(np.float32)


def run_engine(
    engine,
    projection: np.ndarray,
    position: np.ndarray,
    mask: np.ndarray,
    dtype_case: DTypeCase,
    *,
    context=None,
    stream: cp.cuda.Stream | None = None,
    synchronize: bool = True,
) -> EngineRun:
    """Execute one dynamic shape and retain all device and context state."""

    if context is None:
        context = engine.create_execution_context()
    assert context is not None
    assert context.set_input_shape("projection", projection.shape)
    assert context.set_input_shape("position", position.shape)
    assert context.set_input_shape("mask", mask.shape)
    expected_output_shape = (
        projection.shape[0],
        position.shape[2],
        projection.shape[1],
        projection.shape[1],
    )
    assert tuple(context.get_tensor_shape("scores")) == expected_output_shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        projection_device = cp.asarray(projection, dtype=dtype_case.cupy_dtype)
        position_device = cp.asarray(position, dtype=dtype_case.cupy_dtype)
        mask_device = cp.asarray(mask, dtype=cp.bool_)
        scores_device = cp.full(
            expected_output_shape,
            cp.nan,
            dtype=dtype_case.cupy_dtype,
        )
        assert context.set_tensor_address("projection", projection_device.data.ptr)
        assert context.set_tensor_address("position", position_device.data.ptr)
        assert context.set_tensor_address("mask", mask_device.data.ptr)
        assert context.set_tensor_address("scores", scores_device.data.ptr)
        assert context.execute_async_v3(stream.ptr)
    if synchronize:
        stream.synchronize()
    return EngineRun(
        context,
        stream,
        projection_device,
        position_device,
        mask_device,
        scores_device,
    )


def assert_run_matches_reference(run: EngineRun, dtype_case: DTypeCase) -> None:
    """Compare one completed engine run with the independent reference."""

    actual = run.host_scores()
    expected = reference_attention(
        run.projection,
        run.position,
        run.mask,
        dtype_case.torch_dtype,
    )
    assert_matches_reference(
        actual,
        expected,
        cp.asnumpy(run.mask),
        dtype_case.element_tolerance,
        dtype_case.max_row_l1_error,
    )


def test_relative_attention_plugin_excludes_scores_below_old_sentinel(
    relative_attention_engine,
) -> None:
    """Keep a masked key excluded when valid logits are below -1000."""

    _, engine, dtype_case = relative_attention_engine
    projection = np.zeros((1, 2, DEFAULT_PROJECTION_DIM), dtype=np.float32)
    content_dim = DEFAULT_NUM_HEADS * DEFAULT_QUERY_DIM
    projection[:, :, :content_dim] = 50.0
    projection[:, 0, content_dim : 2 * content_dim] = -40.0
    position = np.zeros((1, 3, DEFAULT_NUM_HEADS, POSITION_DIM), dtype=np.float32)
    mask = np.array(((False, True),), dtype=np.bool_)

    actual = run_engine(engine, projection, position, mask, dtype_case).host_scores()

    np.testing.assert_allclose(actual[..., 0], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[..., 1], 0.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("batch_size,sequence_length", SHAPE_CASES)
def test_relative_attention_plugin_matches_pytorch(
    relative_attention_engine, batch_size: int, sequence_length: int
) -> None:
    """Compare every production kernel branch across supported dtypes."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, mask = make_inputs(batch_size, sequence_length)
    run = run_engine(
        engine,
        projection,
        position,
        mask,
        dtype_case,
    )

    assert_run_matches_reference(run, dtype_case)


@pytest.mark.parametrize(
    ("num_heads", "query_dim"),
    ARCHITECTURE_CASES,
    ids=("generic-3-head-query-5", "production-8-head-query-32"),
)
def test_relative_attention_plugin_supports_head_layouts(
    relative_attention_engine,
    num_heads: int,
    query_dim: int,
) -> None:
    """Support generic exported dimensions and both production head counts."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, mask = make_inputs(
        2,
        17,
        num_heads=num_heads,
        query_dim=query_dim,
    )

    run = run_engine(engine, projection, position, mask, dtype_case)

    assert_run_matches_reference(run, dtype_case)


def test_relative_attention_plugin_matches_pytorch_without_padding(
    relative_attention_engine,
) -> None:
    """Compare a fully valid sequence with the PyTorch reference."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, _ = make_inputs(1, 385)
    mask = np.zeros((1, 385), dtype=np.bool_)
    run = run_engine(engine, projection, position, mask, dtype_case)

    assert_run_matches_reference(run, dtype_case)


def test_relative_attention_plugin_handles_fully_padded_sequence(
    relative_attention_engine,
) -> None:
    """Return finite zeros when every source key is padded."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, _ = make_inputs(1, 65)
    mask = np.ones((1, 65), dtype=np.bool_)
    run = run_engine(engine, projection, position, mask, dtype_case)

    assert_run_matches_reference(run, dtype_case)


def test_relative_attention_plugin_supports_cuda_graphs(
    relative_attention_engine,
) -> None:
    """Replay changed attention inputs on a non-default CUDA stream."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, mask = make_inputs(2, 65)
    run = run_engine(engine, projection, position, mask, dtype_case)

    run.stream.begin_capture()
    assert run.context.execute_async_v3(run.stream.ptr)
    graph = run.stream.end_capture()
    graph.upload(run.stream)
    for replay in (1, 2):
        replay_projection = projection * (1.0 + 0.25 * replay) + 0.01 * replay
        replay_position = position * (1.0 - 0.125 * replay)
        replay_mask = np.zeros_like(mask)
        replay_mask[0, 50 - replay :] = True
        replay_mask[1, 40 - replay :] = True
        with run.stream:
            cp.copyto(
                run.projection,
                cp.asarray(replay_projection, dtype=dtype_case.cupy_dtype),
            )
            cp.copyto(
                run.position,
                cp.asarray(replay_position, dtype=dtype_case.cupy_dtype),
            )
            run.mask.set(replay_mask, stream=run.stream)
            run.scores.fill(cp.nan)
            graph.launch(run.stream)

        assert_run_matches_reference(run, dtype_case)


def test_relative_attention_plugin_reuses_context_across_dynamic_shapes(
    relative_attention_engine,
) -> None:
    """Reuse one execution context while batch and kernel-path shapes change."""

    _, engine, dtype_case = relative_attention_engine
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)

    for batch_size, sequence_length in ((1, 384), (2, 513), (1, 384)):
        projection, position, mask = make_inputs(batch_size, sequence_length)
        run = run_engine(
            engine,
            projection,
            position,
            mask,
            dtype_case,
            context=context,
            stream=stream,
        )

        assert run.context is context
        assert_run_matches_reference(run, dtype_case)


def test_relative_attention_plugin_handles_stream_changes(
    relative_attention_engine,
) -> None:
    """Reset the cuBLAS stream and workspace when one context changes streams."""

    _, engine, dtype_case = relative_attention_engine
    projection, position, mask = make_inputs(2, 65)
    context = engine.create_execution_context()
    assert context is not None
    first = run_engine(
        engine,
        projection,
        position,
        mask,
        dtype_case,
        context=context,
        stream=cp.cuda.Stream(non_blocking=True),
    )
    assert_run_matches_reference(first, dtype_case)

    replay_projection = projection * 1.25 + 0.125
    replay_position = position * -0.75
    replay_mask = np.zeros_like(mask)
    replay_mask[0, 51:] = True
    replay_mask[1, 39:] = True
    second = run_engine(
        engine,
        replay_projection,
        replay_position,
        replay_mask,
        dtype_case,
        context=context,
        stream=cp.cuda.Stream(non_blocking=True),
    )

    assert first.stream.ptr != second.stream.ptr
    assert second.context is context
    assert_run_matches_reference(second, dtype_case)


def test_relative_attention_plugin_supports_concurrent_contexts(
    relative_attention_engine,
) -> None:
    """Keep independent cuBLAS state for overlapping execution contexts."""

    _, engine, dtype_case = relative_attention_engine
    first_projection, first_position, first_mask = make_inputs(1, 65)
    second_projection, second_position, second_mask = make_inputs(2, 65)
    first = run_engine(
        engine,
        first_projection,
        first_position,
        first_mask,
        dtype_case,
        synchronize=False,
    )
    second = run_engine(
        engine,
        second_projection,
        second_position,
        second_mask,
        dtype_case,
        synchronize=False,
    )

    assert first.context is not second.context
    assert first.stream.ptr != second.stream.ptr
    assert_run_matches_reference(first, dtype_case)
    assert_run_matches_reference(second, dtype_case)


def test_relative_attention_creator_rejects_fields(plugin_creator) -> None:
    """Reject unexpected fields for the parameter-free plugin."""

    _, creator = plugin_creator
    field = trt.PluginField(
        "unexpected", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32
    )
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is None


def relative_attention_input_specs(
    *,
    projection_shape: tuple[int, ...] = (2, 7, DEFAULT_PROJECTION_DIM),
    position_shape: tuple[int, ...] = (1, 13, DEFAULT_NUM_HEADS, POSITION_DIM),
    mask_shape: tuple[int, ...] = (2, 7),
    projection_dtype=trt.float16,
    position_dtype=trt.float16,
    mask_dtype=trt.bool,
) -> tuple[tuple[object, tuple[int, ...]], ...]:
    """Build one static TensorRT input contract for rejection tests."""

    return (
        (projection_dtype, projection_shape),
        (position_dtype, position_shape),
        (mask_dtype, mask_shape),
    )


def cupy_dtype_for_tensorrt(dtype):
    """Return the CuPy storage dtype corresponding to one TensorRT dtype."""

    mapping = {
        trt.float32: cp.float32,
        trt.float16: cp.float16,
        trt.bfloat16: cp.dtype("bfloat16"),
        trt.int32: cp.int32,
        trt.bool: cp.bool_,
    }
    return mapping[dtype]


@pytest.mark.parametrize(
    "input_specs",
    (
        pytest.param(
            relative_attention_input_specs(projection_shape=(2, 7)),
            id="projection-rank",
        ),
        pytest.param(
            relative_attention_input_specs(position_shape=(1, 13, 16)),
            id="position-rank",
        ),
        pytest.param(
            relative_attention_input_specs(mask_shape=(2, 1, 7)),
            id="mask-rank",
        ),
        pytest.param(
            relative_attention_input_specs(
                position_shape=(2, 13, DEFAULT_NUM_HEADS, POSITION_DIM)
            ),
            id="position-batch",
        ),
        pytest.param(
            relative_attention_input_specs(
                position_shape=(1, 12, DEFAULT_NUM_HEADS, POSITION_DIM)
            ),
            id="position-length",
        ),
        pytest.param(
            relative_attention_input_specs(
                position_shape=(1, 13, DEFAULT_NUM_HEADS, 3)
            ),
            id="position-head-dimension",
        ),
        pytest.param(
            relative_attention_input_specs(mask_shape=(1, 7)),
            id="mask-batch",
        ),
        pytest.param(
            relative_attention_input_specs(mask_shape=(2, 6)),
            id="mask-length",
        ),
        pytest.param(
            relative_attention_input_specs(projection_shape=(2, 7, 271)),
            id="projection-not-divisible-by-heads",
        ),
        pytest.param(
            relative_attention_input_specs(projection_shape=(2, 7, 16)),
            id="projection-without-query-dimensions",
        ),
        pytest.param(
            relative_attention_input_specs(projection_shape=(2, 7, 276)),
            id="projection-with-fractional-query-dimension",
        ),
        pytest.param(
            relative_attention_input_specs(
                projection_dtype=trt.int32,
                position_dtype=trt.int32,
            ),
            id="unsupported-numeric-dtype",
        ),
        pytest.param(
            relative_attention_input_specs(position_dtype=trt.float32),
            id="mixed-numeric-dtypes",
        ),
        pytest.param(
            relative_attention_input_specs(mask_dtype=trt.int32),
            id="non-boolean-mask",
        ),
    ),
)
def test_relative_attention_plugin_rejects_invalid_contracts(
    plugin_creator,
    input_specs: tuple[tuple[object, tuple[int, ...]], ...],
) -> None:
    """Reject invalid contracts during layer, build, or runtime setup."""

    _, creator = plugin_creator
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = [
        network.add_input(name, dtype, shape)
        for name, (dtype, shape) in zip(
            ("projection", "position", "mask"), input_specs, strict=True
        )
    ]
    assert all(value is not None for value in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator))
    if layer is None:
        return

    scores = layer.get_output(0)
    scores.name = "scores"
    network.mark_output(scores)
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
    input_buffers = []
    with stream:
        for name, (dtype, shape) in zip(
            ("projection", "position", "mask"), input_specs, strict=True
        ):
            buffer = cp.zeros(shape, dtype=cupy_dtype_for_tensorrt(dtype))
            assert context.set_tensor_address(name, buffer.data.ptr)
            input_buffers.append(buffer)
        output = cp.full(
            tuple(context.get_tensor_shape("scores")),
            cp.nan,
            dtype=cupy_dtype_for_tensorrt(engine.get_tensor_dtype("scores")),
        )
        assert context.set_tensor_address("scores", output.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()

    assert not executed
    assert bool(cp.isnan(output).all())
