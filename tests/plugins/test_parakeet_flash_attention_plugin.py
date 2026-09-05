#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""TensorRT integration tests for the Parakeet flash-attention plugin."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import NamedTuple

import cupy as cp
import numpy as np
import pytest
import tensorrt as trt
import torch
from tensorrt_plugin_utils import compile_and_load_plugin

from fast_gpu_asr.constants import TENSORRT_PLUGIN_NAMESPACE

pytestmark = pytest.mark.cuda


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
LOG2_E = np.float32(1.4426950408889634)
MINIMUM_POSITIVE_SCALE = np.nextafter(np.float32(0), np.float32(1))
MAXIMUM_VALID_SCALE = np.float32(
    np.float64(np.finfo(np.float32).max) / np.float64(LOG2_E)
)
MAX_SEQUENCE_LENGTH = 512
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max
SOFTMAX_DISPATCH_CASES = tuple(
    pytest.param(
        32 * (slots - 1) + 1,
        id=f"softmax-slots-{slots}",
    )
    for slots in range(1, 17)
) + (pytest.param(MAX_SEQUENCE_LENGTH, id="softmax-maximum"),)
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
    pytest.param(
        (1, MAX_SEQUENCE_LENGTH),
        (MAX_SEQUENCE_LENGTH - 1,),
        id="maximum-partial-mask",
    ),
)

type CuPyDType = type[np.generic] | np.dtype[np.generic]
type InputSpec = tuple[trt.DataType, tuple[int, ...]]


@dataclass(frozen=True)
class EngineCase:
    """Numeric dtype, attention layout, scale, and comparison tolerance."""

    name: str
    trt_dtype: trt.DataType
    cupy_dtype: CuPyDType
    torch_dtype: torch.dtype
    tolerance: float
    num_heads: int = DEFAULT_NUM_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    scale: float = DEFAULT_SCALE

    @property
    def channels(self) -> int:
        """Return the concatenated value width across all attention heads.

        Returns
        -------
        int
            Number of attention heads multiplied by the channels per head.
        """

        return self.num_heads * self.head_dim


ENGINE_CASES = (
    # TensorRT may select TF32 or another reduced-mantissa FP32 cuBLAS tactic.
    EngineCase("fp32", trt.float32, cp.float32, torch.float32, 3e-4),
    EngineCase("fp16", trt.float16, cp.float16, torch.float16, 5e-3),
    pytest.param(
        EngineCase("bf16", trt.bfloat16, cp.dtype("bfloat16"), torch.bfloat16, 3e-2),
        marks=pytest.mark.sm80,
        id="bf16",
    ),
)
ALTERNATE_ENGINE_CASE = EngineCase(
    "fp32-h3-d5-scale075",
    trt.float32,
    cp.float32,
    torch.float32,
    3e-4,
    num_heads=3,
    head_dim=5,
    scale=0.75,
)
MINIMUM_ENGINE_CASE = EngineCase(
    "fp32-h1-d1-scale05",
    trt.float32,
    cp.float32,
    torch.float32,
    3e-4,
    num_heads=1,
    head_dim=1,
    scale=0.5,
)
SCORE_PROMOTION_CASES = (
    pytest.param(
        EngineCase(
            "fp16-score-promotion",
            trt.float16,
            cp.float16,
            torch.float16,
            0.0,
            num_heads=1,
            head_dim=2,
            scale=8.0,
        ),
        np.float32(4e-4),
        id="fp16",
    ),
    pytest.param(
        EngineCase(
            "bf16-score-promotion",
            trt.bfloat16,
            cp.dtype("bfloat16"),
            torch.bfloat16,
            0.0,
            num_heads=1,
            head_dim=2,
            scale=8.0,
        ),
        np.float32(3e-3),
        marks=pytest.mark.sm80,
        id="bf16",
    ),
)


@dataclass(frozen=True)
class AttentionEngine:
    """One deserialized engine and the contract used to build it."""

    runtime: trt.Runtime
    engine: trt.ICudaEngine
    case: EngineCase


class AttentionInputs(NamedTuple):
    """Named host arrays in plugin binding order."""

    qkv: np.typing.NDArray[np.float32]
    position: np.typing.NDArray[np.float32]
    content_bias: np.typing.NDArray[np.float32]
    position_bias: np.typing.NDArray[np.float32]
    valid_lengths: np.typing.NDArray[np.int32]


@dataclass
class AttentionRun:
    """Own the buffers, context, and stream until asynchronous inference finishes."""

    context: trt.IExecutionContext
    stream: cp.cuda.Stream
    inputs: tuple[cp.ndarray, ...]
    output: cp.ndarray


type PluginCreatorFixture = tuple[ctypes.CDLL, trt.IPluginCreatorV3One]


@pytest.fixture(scope="module")
def plugin_creator(
    tmp_path_factory: pytest.TempPathFactory,
) -> PluginCreatorFixture:
    """Compile, register, and return the Parakeet attention creator.

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


def make_plugin(
    creator: trt.IPluginCreatorV3One,
    scale: float,
) -> trt.IPluginV3:
    """Create a Parakeet attention plugin with one scalar scale field.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    scale : float
        Attention-logit multiplier serialized into the plugin.

    Returns
    -------
    trt.IPluginV3
        New plugin configured for the build phase.
    """

    scale_value = np.array([scale], dtype=np.float32)
    field = trt.PluginField("scale", scale_value, trt.PluginFieldType.FLOAT32)
    plugin = creator.create_plugin(
        PLUGIN_NAME,
        trt.PluginFieldCollection([field]),
        trt.TensorRTPhase.BUILD,
    )
    assert plugin is not None
    return plugin


def build_serialized_engine(
    creator: trt.IPluginCreatorV3One,
    input_specs: tuple[InputSpec, ...],
    scale: float = DEFAULT_SCALE,
    profiles: dict[str, tuple[tuple[int, ...], ...]] | None = None,
) -> trt.IHostMemory | None:
    """Build a static or dynamic contract, returning None on build rejection.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    input_specs : tuple[InputSpec, ...]
        Ordered TensorRT input dtypes and shapes, including intentionally invalid
        cases.
    scale : float
        Attention-logit multiplier serialized into the plugin.
    profiles : dict[str, tuple[tuple[int, ...], ...]] | None
        Mapping of input names to min/opt/max shapes; empty means static.

    Returns
    -------
    trt.IHostMemory | None
        Serialized engine bytes, or None when TensorRT rejects the contract.
    """

    logger = trt.Logger(trt.Logger.ERROR)
    assert trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    inputs = [
        network.add_input(
            INPUT_NAMES[index] if index < len(INPUT_NAMES) else f"extra_{index}",
            dtype,
            shape,
        )
        for index, (dtype, shape) in enumerate(input_specs)
    ]
    assert all(tensor is not None for tensor in inputs)
    layer = network.add_plugin_v3(inputs, [], make_plugin(creator, scale))
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
    return builder.build_serialized_network(network, config)


def build_attention_engine(
    creator: trt.IPluginCreatorV3One,
    case: EngineCase,
    max_batch_size: int = 3,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> AttentionEngine | None:
    """Build and deserialize a dynamic attention engine for one dtype and layout.

    Parameters
    ----------
    creator : trt.IPluginCreatorV3One
        Registered creator used to construct the plugin under test.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    max_batch_size : int
        Maximum supported batch size in the test profile.
    max_sequence_length : int
        Maximum supported sequence length in the test profile.

    Returns
    -------
    AttentionEngine | None
        Deserialized engine with its owning runtime, or None on build rejection.
    """

    input_specs = (
        (case.trt_dtype, (-1, -1, 3 * case.channels)),
        (case.trt_dtype, (1, -1, case.channels)),
        (case.trt_dtype, (case.num_heads, case.head_dim)),
        (case.trt_dtype, (case.num_heads, case.head_dim)),
        (trt.int32, (-1,)),
    )
    opt_batch, opt_length = min(2, max_batch_size), min(17, max_sequence_length)
    # Conservative profile bounds need not satisfy the exact runtime relationships.
    profiles = {
        "qkv": (
            (1, 1, 3 * case.channels),
            (opt_batch, opt_length, 3 * case.channels),
            (max_batch_size, max_sequence_length, 3 * case.channels),
        ),
        "position": (
            (1, 1, case.channels),
            (1, 2 * opt_length - 1, case.channels),
            (1, 2 * max_sequence_length, case.channels),
        ),
        "valid_lengths": ((1,), (opt_batch,), (max_batch_size + 1,)),
    }
    serialized = build_serialized_engine(creator, input_specs, case.scale, profiles)
    if serialized is None:
        return None
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    expected_io = {
        name: (trt.TensorIOMode.INPUT, dtype)
        for name, (dtype, _) in zip(INPUT_NAMES, input_specs, strict=True)
    }
    expected_io["output"] = (trt.TensorIOMode.OUTPUT, case.trt_dtype)
    assert engine.num_io_tensors == len(expected_io)
    for name, (mode, dtype) in expected_io.items():
        assert engine.get_tensor_mode(name) == mode
        assert engine.get_tensor_dtype(name) == dtype
    return AttentionEngine(runtime, engine, case)


@pytest.fixture(scope="module", params=ENGINE_CASES, ids=lambda case: case.name)
def attention_engine(
    request: pytest.FixtureRequest, plugin_creator: PluginCreatorFixture
) -> AttentionEngine:
    """Build the production attention layout for every supported dtype.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized dtype or layout selected for this module-scoped engine.
    plugin_creator : tuple
        Compiled library handles and the registered creator; retained for engine
        lifetime.

    Returns
    -------
    AttentionEngine
        Deserialized engine with its owning runtime.
    """

    _, creator = plugin_creator
    result = build_attention_engine(creator, request.param)
    assert result is not None
    return result


def make_inputs(
    case: EngineCase,
    batch_size: int,
    sequence_length: int,
    valid_lengths: tuple[int, ...],
    seed: int | None = None,
) -> AttentionInputs:
    """Create deterministic projected attention inputs.

    Parameters
    ----------
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    batch_size : int
        Number of utterances in the batch.
    sequence_length : int
        Physical number of frames per utterance.
    valid_lengths : tuple[int, ...]
        Declared valid frame counts; negative and oversized values exercise
        clamping.
    seed : int or None
        Local random seed; None derives a reproducible seed from the requested
        shape.

    Returns
    -------
    AttentionInputs
        Named host inputs in plugin binding order.
    """

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


def zero_inputs(
    case: EngineCase, sequence_length: int, valid_length: int
) -> AttentionInputs:
    """Create one zero-filled utterance for hand-computed attention cases.

    Parameters
    ----------
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    sequence_length : int
        Physical number of frames per utterance.
    valid_length : int
        Declared valid frame count for the single test utterance.

    Returns
    -------
    AttentionInputs
        Zero-valued numeric inputs and the supplied single-utterance valid length.
    """

    return AttentionInputs(
        np.zeros((1, sequence_length, 3 * case.channels), dtype=np.float32),
        np.zeros((1, 2 * sequence_length - 1, case.channels), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.zeros((case.num_heads, case.head_dim), dtype=np.float32),
        np.array((valid_length,), dtype=np.int32),
    )


def run_engine(
    attention_engine: AttentionEngine,
    inputs: AttentionInputs,
    context: trt.IExecutionContext | None = None,
    stream: cp.cuda.Stream | None = None,
    execute: bool = True,
) -> AttentionRun:
    """Bind sentinel-backed buffers and optionally enqueue on an ordered stream.

    Parameters
    ----------
    attention_engine : AttentionEngine
        Engine, owning runtime, and numeric/layout settings.
    inputs : AttentionInputs
        Host QKV projections, positions, two head biases, and INT32 valid lengths.
    context : trt.IExecutionContext or None
        Context to reuse after prior work completes; None creates a fresh context.
    stream : cp.cuda.Stream or None
        Stream ordering uploads and inference; None creates a nonblocking stream.
    execute : bool
        Whether to enqueue inference after binding; False permits rejection tests.

    Returns
    -------
    AttentionRun
        Run state retaining context, stream, and buffers until pending work
        completes.
    """

    if context is None:
        context = attention_engine.engine.create_execution_context()
    assert context is not None
    for name, value in zip(INPUT_NAMES, inputs, strict=True):
        assert context.set_input_shape(name, value.shape)
    assert context.infer_shapes() == []
    output_shape = (*inputs.qkv.shape[:2], attention_engine.case.channels)
    assert tuple(context.get_tensor_shape("output")) == output_shape
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        buffers = tuple(
            cp.array(
                value,
                dtype=cp.int32
                if name == "valid_lengths"
                else attention_engine.case.cupy_dtype,
            )
            for name, value in zip(INPUT_NAMES, inputs, strict=True)
        )
        output = cp.full(output_shape, cp.nan, dtype=attention_engine.case.cupy_dtype)
        for name, value in zip(INPUT_NAMES, buffers, strict=True):
            assert context.set_tensor_address(name, value.data.ptr)
        assert context.set_tensor_address("output", output.data.ptr)
        if execute:
            assert context.execute_async_v3(stream.ptr)
    return AttentionRun(context, stream, buffers, output)


def quantize_host_inputs(inputs: AttentionInputs, case: EngineCase) -> AttentionInputs:
    """Round numeric host inputs independently of CuPy's device conversion.

    Parameters
    ----------
    inputs : AttentionInputs
        Host QKV projections, positions, two head biases, and INT32 valid lengths.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.

    Returns
    -------
    AttentionInputs
        Independent copies rounded to the case dtype, stored as FP32; lengths remain
        INT32.
    """

    return AttentionInputs(
        *(
            torch.from_numpy(value).to(case.torch_dtype).float().numpy().copy()
            for value in inputs[:4]
        ),
        inputs.valid_lengths.copy(),
    )


def reference_attention(
    inputs: AttentionInputs,
    case: EngineCase,
    scale: float | None = None,
) -> np.typing.NDArray:
    """Evaluate Parakeet's mixed-precision attention path with PyTorch.

    Parameters
    ----------
    inputs : AttentionInputs
        Host QKV projections, positions, two head biases, and INT32 valid lengths.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    scale : float or None
        Optional override of the case's attention scale for sensitivity checks.

    Returns
    -------
    np.typing.NDArray
        Weighted values in NTC layout, rounded to storage dtype and returned as
        FP32.
    """

    qkv = torch.from_numpy(inputs.qkv).to(case.torch_dtype)
    position = torch.from_numpy(inputs.position).to(case.torch_dtype)
    content_bias = torch.from_numpy(inputs.content_bias).to(case.torch_dtype)
    position_bias = torch.from_numpy(inputs.position_bias).to(case.torch_dtype)
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
    content_scores = torch.matmul(
        content_query,
        key.permute(0, 1, 3, 2),
    ).float()
    position_scores = torch.matmul(
        position_query,
        position.permute(0, 1, 3, 2),
    ).float()
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
    ).to(case.torch_dtype)
    return (
        torch.matmul(weights, value.permute(0, 2, 1, 3))
        .permute(0, 2, 1, 3)
        .reshape(batch_size, sequence_length, case.channels)
        .float()
        .numpy()
    )


def assert_run_matches_reference(
    run: AttentionRun, case: EngineCase, host_inputs: AttentionInputs
) -> np.typing.NDArray:
    """Check unchanged inputs, finite output, numerical parity, and exact empty rows.

    Parameters
    ----------
    run : AttentionRun
        Bound device buffers and the context/stream that own their pending work.
    case : EngineCase
        Storage dtype, layout, and numerical tolerance for this engine.
    host_inputs : AttentionInputs
        Host QKV projections, positions, two head biases, and INT32 valid lengths.

    Returns
    -------
    np.typing.NDArray
        Completed output converted to FP32 for additional host assertions.
    """

    run.stream.synchronize()
    inputs = quantize_host_inputs(host_inputs, case)
    for expected, device_input in zip(inputs, run.inputs, strict=True):
        np.testing.assert_array_equal(
            cp.asnumpy(device_input).astype(expected.dtype), expected
        )
    actual = cp.asnumpy(run.output).astype(np.float32)
    expected = reference_attention(inputs, case)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual, expected, rtol=case.tolerance, atol=case.tolerance
    )
    np.testing.assert_array_equal(actual[inputs.valid_lengths <= 0], 0.0)
    return actual


@pytest.mark.parametrize(("shape", "valid_lengths"), SHAPE_CASES)
def test_parakeet_flash_attention_plugin_matches_reference(
    attention_engine: AttentionEngine,
    shape: tuple[int, int],
    valid_lengths: tuple[int, ...],
) -> None:
    inputs = make_inputs(attention_engine.case, *shape, valid_lengths)
    assert_run_matches_reference(
        run_engine(attention_engine, inputs), attention_engine.case, inputs
    )


@pytest.mark.parametrize("sequence_length", SOFTMAX_DISPATCH_CASES)
def test_parakeet_flash_attention_exercises_every_softmax_dispatch_slot(
    attention_engine: AttentionEngine, sequence_length: int
) -> None:
    case = attention_engine.case
    inputs = zero_inputs(case, sequence_length, sequence_length)
    for head in range(case.num_heads):
        channel = head * case.head_dim
        inputs.qkv[0, :, channel] = 20.0
        inputs.qkv[0, sequence_length - 1, case.channels + channel] = 20.0
    inputs.qkv[0, sequence_length - 1, 2 * case.channels :] = 1.0
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    np.testing.assert_allclose(actual, 1.0, rtol=0.0, atol=case.tolerance)


@pytest.mark.parametrize(
    "case", (ALTERNATE_ENGINE_CASE, MINIMUM_ENGINE_CASE), ids=lambda case: case.name
)
def test_parakeet_flash_attention_supports_head_layouts_and_scales(
    plugin_creator: PluginCreatorFixture, case: EngineCase
) -> None:
    _, creator = plugin_creator
    engine = build_attention_engine(creator, case, max_sequence_length=65)
    assert engine is not None
    inputs = make_inputs(case, 2, 33, (33, 17), seed=7001)
    inputs.qkv[..., : 2 * case.channels] *= 3.0
    inputs.position[...] *= 2.0
    inputs.content_bias[...] *= 2.0
    inputs.position_bias[...] *= 2.0
    actual = assert_run_matches_reference(run_engine(engine, inputs), case, inputs)
    wrong_scale = reference_attention(inputs, case, scale=DEFAULT_SCALE)
    assert not np.allclose(
        actual, wrong_scale, rtol=case.tolerance, atol=case.tolerance
    )


@pytest.mark.parametrize(("case", "position_delta"), SCORE_PROMOTION_CASES)
def test_parakeet_flash_attention_promotes_score_sum_before_softmax(
    plugin_creator: PluginCreatorFixture, case: EngineCase, position_delta: np.float32
) -> None:
    _, creator = plugin_creator
    engine = build_attention_engine(
        creator, case, max_batch_size=1, max_sequence_length=2
    )
    assert engine is not None
    inputs = zero_inputs(case, 2, 2)
    inputs.qkv[0, :, case.channels + 1] = 1.0
    inputs.qkv[0, 1, 2 * case.channels] = 1.0
    inputs.position[0, 1, 0] = position_delta
    inputs.position[0, 2, 0] = -position_delta
    inputs.content_bias[0] = (0.0, 1.5)
    inputs.position_bias[0] = (1.0, 0.0)
    actual = assert_run_matches_reference(run_engine(engine, inputs), case, inputs)
    positions = torch.from_numpy(inputs.position[0, 1:, 0]).to(case.torch_dtype)
    scores = torch.full((2,), 1.5, dtype=case.torch_dtype)
    promoted = torch.softmax(
        (scores.float() + positions.float()) * case.scale,
        dim=0,
    )[1].to(case.torch_dtype)
    unpromoted = torch.softmax((scores + positions).float() * case.scale, dim=0)[1].to(
        case.torch_dtype
    )
    assert not torch.equal(promoted, unpromoted)
    np.testing.assert_array_equal(actual[0, 0], (promoted.float().item(), 0.0))


def test_parakeet_flash_attention_masks_extreme_valid_logits(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = zero_inputs(case, 2, 1)
    for head in range(case.num_heads):
        channel = head * case.head_dim
        inputs.qkv[0, :, channel] = 1.0
        inputs.qkv[0, 0, case.channels + channel] = -1200.0 / case.scale
    inputs.qkv[0, 0, 2 * case.channels :] = 1.0
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    np.testing.assert_allclose(actual, 1.0, rtol=0.0, atol=case.tolerance)


def test_parakeet_flash_attention_applies_content_attention_and_bias(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = zero_inputs(case, 3, 3)
    inputs.qkv[0, :, 2 * case.channels :] = np.arange(1, 4, dtype=np.float32)[:, None]
    for head in range(case.num_heads):
        channel = head * case.head_dim
        for frame in range(3):
            if frame < 2:
                inputs.qkv[0, frame, channel + frame] = 10.0
            inputs.qkv[0, frame, case.channels + channel + frame] = 20.0
    inputs.content_bias[:, 2] = 5.0
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    expected = np.broadcast_to(
        np.arange(1, 4, dtype=np.float32)[None, :, None], actual.shape
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=case.tolerance)


def test_parakeet_flash_attention_preserves_head_and_value_layout(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = zero_inputs(case, 7, 4)
    for key in range(7):
        for head in range(case.num_heads):
            start = 2 * case.channels + head * case.head_dim
            inputs.qkv[0, key, start : start + case.head_dim] = head * 10 + key
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    expected = np.repeat(np.arange(case.num_heads) * 10 + 1.5, case.head_dim)
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected, actual.shape), rtol=0.0, atol=case.tolerance
    )


def test_parakeet_flash_attention_ignores_finite_padded_keys_and_values(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = make_inputs(case, 1, 17, (5,), seed=7501)
    changed_qkv = inputs.qkv.copy()
    changed_qkv[:, 5:, case.channels :] = np.random.default_rng(7502).normal(
        0.0, 8.0, changed_qkv[:, 5:, case.channels :].shape
    )
    changed = inputs._replace(qkv=changed_qkv)
    assert not np.array_equal(inputs.qkv, changed.qkv)
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    changed_actual = assert_run_matches_reference(
        run_engine(attention_engine, changed), case, changed
    )
    np.testing.assert_array_equal(actual, changed_actual)


def test_parakeet_flash_attention_applies_relative_position_shift(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = zero_inputs(case, 3, 3)
    inputs.qkv[0, :, 2 * case.channels :] = np.arange(1, 4, dtype=np.float32)[:, None]
    inputs.position.fill(-50.0)
    for head in range(case.num_heads):
        inputs.position[0, 1, head * case.head_dim] = 50.0
        inputs.position[0, 4, head * case.head_dim] = 50.0
    inputs.position_bias[:, 0] = 1.0
    actual = assert_run_matches_reference(
        run_engine(attention_engine, inputs), case, inputs
    )
    expected = np.broadcast_to(np.array((3.0, 1.0, 2.0))[None, :, None], actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=case.tolerance)


def test_parakeet_flash_attention_supports_cuda_graph_replay(
    attention_engine: AttentionEngine,
) -> None:
    case = attention_engine.case
    inputs = make_inputs(case, 2, 17, (17, 5), seed=8001)
    run = run_engine(attention_engine, inputs)
    run.stream.synchronize()
    with run.stream:
        run.stream.begin_capture()
        assert run.context.execute_async_v3(run.stream.ptr)
        graph = run.stream.end_capture()
        graph.upload(run.stream)
    for replay, lengths in enumerate(((16, 7), (0, 17))):
        inputs = make_inputs(case, 2, 17, lengths, seed=8100 + replay)
        with run.stream:
            for destination, source in zip(run.inputs, inputs, strict=True):
                cp.copyto(destination, cp.array(source, dtype=destination.dtype))
            run.output.fill(cp.nan)
            graph.launch(run.stream)
        assert_run_matches_reference(run, case, inputs)


def test_parakeet_flash_attention_reuses_context_across_shapes_and_streams(
    attention_engine: AttentionEngine,
) -> None:
    context = attention_engine.engine.create_execution_context()
    assert context is not None
    streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream.null)
    shape_cases = (
        (1, 1, (1,)),
        (3, 65, (INT32_MIN, 34, INT32_MAX)),
        (1, 33, (32,)),
    )
    for index, (batch, length, valid_lengths) in enumerate(shape_cases):
        stream = streams[index % len(streams)]
        inputs = make_inputs(
            attention_engine.case, batch, length, valid_lengths, seed=9000 + index
        )
        run = run_engine(attention_engine, inputs, context, stream)
        assert run.context is context
        assert run.stream is stream
        assert_run_matches_reference(run, attention_engine.case, inputs)


def test_parakeet_flash_attention_supports_concurrent_contexts(
    attention_engine: AttentionEngine,
) -> None:
    inputs = (
        make_inputs(attention_engine.case, 1, 33, (31,), seed=10001),
        make_inputs(attention_engine.case, 3, 65, (65, 34, 1), seed=10002),
    )
    runs = [run_engine(attention_engine, values, execute=False) for values in inputs]
    assert runs[0].context is not runs[1].context
    assert runs[0].stream.ptr != runs[1].stream.ptr
    for run in runs:
        assert run.context.execute_async_v3(run.stream.ptr)
    for run, values in zip(runs, inputs, strict=True):
        assert_run_matches_reference(run, attention_engine.case, values)


@pytest.mark.parametrize(
    "name,size",
    (
        pytest.param("position", 32, id="position-shorter"),
        pytest.param("position", 34, id="position-longer"),
        pytest.param("valid_lengths", 1, id="length-shorter"),
        pytest.param("valid_lengths", 3, id="length-longer"),
    ),
)
def test_parakeet_flash_attention_rejects_runtime_shape_mismatch(
    attention_engine: AttentionEngine, name: str, size: int
) -> None:
    inputs = make_inputs(attention_engine.case, 2, 17, (17, 5), seed=11000)
    shape = (1, size, attention_engine.case.channels) if name == "position" else (size,)
    inputs = inputs._replace(
        **{name: np.zeros(shape, dtype=getattr(inputs, name).dtype)}
    )
    run = run_engine(attention_engine, inputs, execute=False)
    assert not run.context.execute_async_v3(run.stream.ptr)
    run.stream.synchronize()
    assert bool(cp.isnan(run.output).all())


DEFAULT_CHANNELS = DEFAULT_NUM_HEADS * DEFAULT_HEAD_DIM
VALID_INPUT_SPECS = (
    (trt.float16, (2, 7, 3 * DEFAULT_CHANNELS)),
    (trt.float16, (1, 13, DEFAULT_CHANNELS)),
    (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM)),
    (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM)),
    (trt.int32, (2,)),
)


def execute_static_contract(
    creator: trt.IPluginCreatorV3One, input_specs: tuple[InputSpec, ...]
) -> bool:
    """Build and execute a contract, checking rejected calls leave output untouched.

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

    serialized = build_serialized_engine(creator, input_specs)
    if serialized is None:
        return False
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    stream = cp.cuda.Stream(non_blocking=True)
    buffers = []
    with stream:
        for index, (dtype, shape) in enumerate(input_specs):
            name = INPUT_NAMES[index] if index < len(INPUT_NAMES) else f"extra_{index}"
            buffer = cp.zeros(shape, dtype=trt.nptype(dtype))
            assert context.set_tensor_address(name, buffer.data.ptr)
            buffers.append(buffer)
        output_shape = tuple(context.get_tensor_shape("output"))
        assert all(dimension > 0 for dimension in output_shape)
        assert engine.get_tensor_dtype("output") == input_specs[0][0]
        output = cp.full(output_shape, cp.nan, dtype=trt.nptype(input_specs[0][0]))
        assert context.set_tensor_address("output", output.data.ptr)
        executed = context.execute_async_v3(stream.ptr)
    stream.synchronize()
    if executed:
        assert bool((output == 0).all())
    else:
        assert bool(cp.isnan(output).all())
    return executed


def test_parakeet_flash_attention_accepts_valid_static_contract(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert execute_static_contract(creator, VALID_INPUT_SPECS)


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({0: (trt.float16, (2, 7))}, id="qkv-rank"),
        pytest.param({1: (trt.float16, (1, 13))}, id="position-rank"),
        pytest.param({2: (trt.float16, (DEFAULT_CHANNELS,))}, id="content-bias-rank"),
        pytest.param(
            {3: (trt.float16, (1, DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM))},
            id="position-bias-rank",
        ),
        pytest.param({4: (trt.int32, (2, 1))}, id="length-rank"),
        pytest.param(
            {0: (trt.float16, (0, 7, 3 * DEFAULT_CHANNELS)), 4: (trt.int32, (0,))},
            id="empty-batch",
        ),
        pytest.param(
            {
                0: (trt.float16, (1, 0, 3 * DEFAULT_CHANNELS)),
                1: (trt.float16, (1, 1, DEFAULT_CHANNELS)),
                4: (trt.int32, (1,)),
            },
            id="empty-sequence",
        ),
        pytest.param({1: (trt.float16, (1, 0, DEFAULT_CHANNELS))}, id="empty-position"),
        pytest.param(
            {2: (trt.float16, (0, DEFAULT_HEAD_DIM))}, id="empty-content-bias"
        ),
        pytest.param(
            {3: (trt.float16, (DEFAULT_NUM_HEADS, 0))}, id="empty-position-bias"
        ),
        pytest.param({4: (trt.int32, (0,))}, id="empty-valid-lengths"),
        pytest.param(
            {index: (trt.int32, VALID_INPUT_SPECS[index][1]) for index in range(4)},
            id="unsupported-numeric-dtype",
        ),
        pytest.param(
            {1: (trt.float32, (1, 13, DEFAULT_CHANNELS))}, id="position-dtype"
        ),
        pytest.param(
            {2: (trt.float32, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM))},
            id="content-bias-dtype",
        ),
        pytest.param(
            {3: (trt.float32, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM))},
            id="position-bias-dtype",
        ),
        pytest.param({4: (trt.int64, (2,))}, id="length-dtype"),
        pytest.param(
            {0: (trt.float16, (2, 7, 3 * DEFAULT_CHANNELS - 1))},
            id="qkv-channels-shorter",
        ),
        pytest.param(
            {0: (trt.float16, (2, 7, 3 * DEFAULT_CHANNELS + 1))},
            id="qkv-channels-longer",
        ),
        pytest.param(
            {1: (trt.float16, (2, 13, DEFAULT_CHANNELS))}, id="position-batch"
        ),
        pytest.param(
            {1: (trt.float16, (1, 12, DEFAULT_CHANNELS))}, id="position-length-shorter"
        ),
        pytest.param(
            {1: (trt.float16, (1, 14, DEFAULT_CHANNELS))}, id="position-length-longer"
        ),
        pytest.param(
            {1: (trt.float16, (1, 13, DEFAULT_CHANNELS - 1))},
            id="position-channels-shorter",
        ),
        pytest.param(
            {1: (trt.float16, (1, 13, DEFAULT_CHANNELS + 1))},
            id="position-channels-longer",
        ),
        pytest.param(
            {3: (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM - 1))},
            id="position-bias-shape-mismatch",
        ),
        pytest.param(
            {2: (trt.float16, (DEFAULT_NUM_HEADS * 2, DEFAULT_HEAD_DIM // 2))},
            id="content-bias-layout-mismatch",
        ),
        pytest.param(
            {
                2: (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM // 2)),
                3: (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM // 2)),
            },
            id="head-product-smaller",
        ),
        pytest.param(
            {
                2: (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM + 1)),
                3: (trt.float16, (DEFAULT_NUM_HEADS, DEFAULT_HEAD_DIM + 1)),
            },
            id="head-product-larger",
        ),
        pytest.param({4: (trt.int32, (1,))}, id="length-batch-shorter"),
        pytest.param({4: (trt.int32, (3,))}, id="length-batch-longer"),
        pytest.param(
            {
                0: (trt.float16, (1, 513, 3 * DEFAULT_CHANNELS)),
                1: (trt.float16, (1, 1025, DEFAULT_CHANNELS)),
                4: (trt.int32, (1,)),
            },
            id="sequence-capacity",
        ),
    ),
)
def test_parakeet_flash_attention_rejects_invalid_contracts(
    plugin_creator: PluginCreatorFixture, overrides: dict[int, InputSpec]
) -> None:
    _, creator = plugin_creator
    specs = list(VALID_INPUT_SPECS)
    for index, spec in overrides.items():
        specs[index] = spec
    assert not execute_static_contract(creator, tuple(specs))


@pytest.mark.parametrize("count", (4, 6), ids=("missing-input", "extra-input"))
def test_parakeet_flash_attention_rejects_invalid_input_count(
    plugin_creator: PluginCreatorFixture, count: int
) -> None:
    _, creator = plugin_creator
    specs = (VALID_INPUT_SPECS + ((trt.float16, (1,)),))[:count]
    assert not execute_static_contract(creator, specs)


def test_parakeet_flash_attention_rejects_profile_above_capacity(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert (
        build_attention_engine(creator, ENGINE_CASES[1], max_sequence_length=513)
        is None
    )


@pytest.mark.parametrize(
    "specs",
    (
        pytest.param((), id="missing"),
        pytest.param(
            (("wrong", [DEFAULT_SCALE], trt.PluginFieldType.FLOAT32),), id="wrong-name"
        ),
        pytest.param((("scale", [1], trt.PluginFieldType.INT32),), id="wrong-type"),
        pytest.param((("scale", [], trt.PluginFieldType.FLOAT32),), id="empty"),
        pytest.param(
            (("scale", [0.5, 1.0], trt.PluginFieldType.FLOAT32),), id="multiple"
        ),
        pytest.param(
            (
                ("scale", [0.5], trt.PluginFieldType.FLOAT32),
                ("scale", [1.0], trt.PluginFieldType.FLOAT32),
            ),
            id="duplicate",
        ),
        pytest.param(
            (
                ("scale", [DEFAULT_SCALE], trt.PluginFieldType.FLOAT32),
                ("metadata", [1], trt.PluginFieldType.INT32),
            ),
            id="unknown-extra",
        ),
    ),
)
def test_parakeet_flash_attention_creator_rejects_invalid_scale_fields(
    plugin_creator: PluginCreatorFixture,
    specs: tuple[tuple[str, list[float], trt.PluginFieldType], ...],
) -> None:
    _, creator = plugin_creator
    values = [
        np.array(
            data, dtype=np.int32 if dtype == trt.PluginFieldType.INT32 else np.float32
        )
        for _, data, dtype in specs
    ]
    fields = [
        trt.PluginField(name, value, dtype)
        for (name, _, dtype), value in zip(specs, values, strict=True)
    ]
    assert (
        creator.create_plugin(
            PLUGIN_NAME, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD
        )
        is None
    )


@pytest.mark.parametrize(
    "scale",
    (0.0, -1.0, np.nan, np.inf, np.nextafter(MAXIMUM_VALID_SCALE, np.float32(np.inf))),
    ids=("zero", "negative", "nan", "infinity", "scaled-overflow"),
)
def test_parakeet_flash_attention_creator_rejects_invalid_scale_values(
    plugin_creator: PluginCreatorFixture, scale: float
) -> None:
    _, creator = plugin_creator
    value = np.array([scale], dtype=np.float32)
    fields = trt.PluginFieldCollection(
        [trt.PluginField("scale", value, trt.PluginFieldType.FLOAT32)]
    )
    assert creator.create_plugin(PLUGIN_NAME, fields, trt.TensorRTPhase.BUILD) is None


def test_parakeet_flash_attention_creator_exposes_complete_contract(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    assert (creator.name, creator.plugin_version, creator.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    (field,) = creator.field_names
    assert (field.name, field.type, field.size) == (
        "scale",
        trt.PluginFieldType.FLOAT32,
        1,
    )

    plugin = make_plugin(creator, DEFAULT_SCALE)
    core = plugin.get_capability_interface(trt.PluginCapabilityType.CORE)
    build = plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
    runtime = plugin.get_capability_interface(trt.PluginCapabilityType.RUNTIME)
    assert core is not None and build is not None and runtime is not None
    assert (core.plugin_name, core.plugin_version, core.plugin_namespace) == (
        PLUGIN_NAME,
        PLUGIN_VERSION,
        TENSORRT_PLUGIN_NAMESPACE,
    )
    assert build.num_outputs == 1


def test_parakeet_flash_attention_timing_cache_keys_include_scale(
    plugin_creator: PluginCreatorFixture,
) -> None:
    _, creator = plugin_creator
    plugins = [
        make_plugin(creator, scale) for scale in (DEFAULT_SCALE, DEFAULT_SCALE, 0.75)
    ]
    builds = [
        plugin.get_capability_interface(trt.PluginCapabilityType.BUILD)
        for plugin in plugins
    ]
    assert all(build is not None for build in builds)
    ids = [build.timing_cache_id for build in builds]
    assert all(ids)
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


@pytest.mark.parametrize(
    "scale",
    (
        pytest.param(MINIMUM_POSITIVE_SCALE, id="smallest-positive"),
        pytest.param(MAXIMUM_VALID_SCALE, id="largest-scaled-finite"),
    ),
)
def test_parakeet_flash_attention_creator_accepts_positive_scale_boundaries(
    plugin_creator: PluginCreatorFixture, scale: float
) -> None:
    _, creator = plugin_creator
    make_plugin(creator, scale)
