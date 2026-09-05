#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU numerical and ONNX export tests for Zipformer attention modules."""

import math
from pathlib import Path

import onnx
import pytest
import torch

from fast_gpu_asr.constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
    ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
)
from fast_gpu_asr.export.model.zipformer.attention import (
    CompactRelPositionalEncoding,
    NonlinAttention,
    RelPositionMultiheadAttentionWeights,
    SelfAttention,
)

NUM_HEADS = 3
QUERY_HEAD_DIM = 5
VALUE_HEAD_DIM = 6
POSITION_HEAD_DIM = 4
RELATIVE_ATTENTION_INPUT_DIM = 17
RELATIVE_ATTENTION_DIM = NUM_HEADS * (2 * QUERY_HEAD_DIM + POSITION_HEAD_DIM)
POSITION_INPUT_DIM = 9
POSITION_DIM = NUM_HEADS * POSITION_HEAD_DIM
RELATIVE_CONTENT_DIM = NUM_HEADS * QUERY_HEAD_DIM
SELF_ATTENTION_INPUT_DIM = 17
ALTERNATE_SELF_NUM_HEADS = 2
ALTERNATE_SELF_VALUE_HEAD_DIM = 5
PADDED_QUERY_HALO = 7
SUPPORTED_DTYPES = (
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
)
ONNX_DTYPES = {
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
}
RELATIVE_DTYPE_CASES = (
    pytest.param(torch.float32, 1e-6, 1e-5, id="fp32"),
    pytest.param(torch.float16, 5e-4, 2e-3, id="fp16"),
    pytest.param(torch.bfloat16, 2e-3, 2e-2, id="bf16"),
)
ATTENTION_VALUE_DTYPE_CASES = (
    pytest.param(torch.float32, 1e-6, 1e-5, id="fp32"),
    pytest.param(torch.float16, 5e-4, 2e-3, id="fp16"),
    pytest.param(torch.bfloat16, 3e-3, 2e-2, id="bf16"),
)


def make_relative_attention() -> RelPositionMultiheadAttentionWeights:
    """Create relative attention whose projections preserve test inputs.

    Returns
    -------
    RelPositionMultiheadAttentionWeights
        FP32 CPU module with identity projections and zero projection bias.
        Construction leaves the caller's RNG state unchanged.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(29)
        module = RelPositionMultiheadAttentionWeights(
            RELATIVE_ATTENTION_DIM,
            POSITION_DIM,
            NUM_HEADS,
            QUERY_HEAD_DIM,
            POSITION_HEAD_DIM,
        )
    with torch.no_grad():
        module.in_proj.weight.copy_(torch.eye(RELATIVE_ATTENTION_DIM))
        module.in_proj.bias.zero_()
        module.linear_pos.weight.copy_(torch.eye(POSITION_DIM))
    return module


def make_projected_relative_attention(
    dtype: torch.dtype = torch.float32,
) -> RelPositionMultiheadAttentionWeights:
    """Create deterministic relative attention with nonidentity projections.

    Parameters
    ----------
    dtype : torch.dtype
        Parameter precision.

    Returns
    -------
    RelPositionMultiheadAttentionWeights
        Seeded CPU module without changing global RNG state.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(31)
        return RelPositionMultiheadAttentionWeights(
            RELATIVE_ATTENTION_INPUT_DIM,
            POSITION_INPUT_DIM,
            NUM_HEADS,
            QUERY_HEAD_DIM,
            POSITION_HEAD_DIM,
        ).to(dtype)


def make_self_attention(
    dtype: torch.dtype = torch.float32,
    num_heads: int = NUM_HEADS,
    value_head_dim: int = VALUE_HEAD_DIM,
) -> SelfAttention:
    """Create deterministic self-attention projections.

    Parameters
    ----------
    dtype : torch.dtype
        Parameter precision.
    num_heads : int
        Number of independently weighted value heads.
    value_head_dim : int
        Projected channels per value head.

    Returns
    -------
    SelfAttention
        Seeded CPU module without changing global RNG state.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(37)
        return SelfAttention(SELF_ATTENTION_INPUT_DIM, num_heads, value_head_dim).to(
            dtype
        )


def make_nonlinear_attention(dtype: torch.dtype = torch.float32) -> NonlinAttention:
    """Create deterministic nonlinear-attention projections.

    Parameters
    ----------
    dtype : torch.dtype
        Parameter precision.

    Returns
    -------
    NonlinAttention
        Seeded CPU module with input width 12 and hidden width 11, leaving
        the caller's RNG state unchanged.
    """

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(43)
        return NonlinAttention(12, 11).to(dtype)


def reference_relative_attention(
    module: RelPositionMultiheadAttentionWeights,
    x: torch.Tensor,
    pos_emb: torch.Tensor,
    key_padding_mask: torch.Tensor,
    num_heads: int,
    query_head_dim: int,
    position_head_dim: int,
) -> torch.Tensor:
    """Evaluate projected relative attention by explicit query-key indexing.

    Parameters
    ----------
    module : RelPositionMultiheadAttentionWeights
        Source of content and positional projection parameters.
    x : torch.Tensor
        CPU features of shape ``(batch, frames, input_dim)``.
    pos_emb : torch.Tensor
        Embeddings of shape ``(1, 2 * frames - 1, position_input_dim)``.
    key_padding_mask : torch.Tensor
        Boolean mask of shape ``(batch, frames)`` with true right-padding slots.
    num_heads : int
        Number of content and positional heads.
    query_head_dim : int
        Query and key channels per head.
    position_head_dim : int
        Positional channels per head.

    Returns
    -------
    torch.Tensor
        Weights of shape ``(batch, heads, frames, frames)`` in ``x.dtype``,
        with masked keys and queries beyond the seven-frame halo zeroed.
    """

    batch_size, sequence_length, _ = x.shape
    content_dim = num_heads * query_head_dim
    position_dim = num_heads * position_head_dim
    projection = torch.nn.functional.linear(
        x, module.in_proj.weight, module.in_proj.bias
    )
    position = torch.nn.functional.linear(pos_emb, module.linear_pos.weight)
    query, key, position_query = torch.split(
        projection, (content_dim, content_dim, position_dim), dim=2
    )
    query = query.reshape(batch_size, sequence_length, num_heads, query_head_dim)
    key = key.reshape(batch_size, sequence_length, num_heads, query_head_dim)
    position_query = position_query.reshape(
        batch_size, sequence_length, num_heads, position_head_dim
    )
    position = position.reshape(
        1, 2 * sequence_length - 1, num_heads, position_head_dim
    )
    scores = torch.empty(
        batch_size, num_heads, sequence_length, sequence_length, dtype=x.dtype
    )
    for query_index in range(sequence_length):
        for key_index in range(sequence_length):
            relative_index = sequence_length - 1 - query_index + key_index
            scores[:, :, query_index, key_index] = (
                query[:, query_index] * key[:, key_index]
            ).sum(dim=2) + (
                position_query[:, query_index] * position[:, relative_index]
            ).sum(dim=2)

    expanded_key_mask = key_padding_mask[:, None, None]
    weights = torch.softmax(
        scores.masked_fill(expanded_key_mask, float("-inf")), dim=3
    ).masked_fill(expanded_key_mask, 0.0)
    for batch_index, valid_length in enumerate((~key_padding_mask).sum(dim=1)):
        weights[batch_index, :, valid_length + PADDED_QUERY_HALO :] = 0.0

    return weights


def reference_self_attention(
    module: SelfAttention,
    x: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply attention weights after independently arranging projected values.

    Parameters
    ----------
    module : SelfAttention
        Source of input and output projections and the head count.
    x : torch.Tensor
        Features of shape ``(batch, frames, input_dim)``.
    attention_weights : torch.Tensor
        Weights of shape ``(batch, heads, frames, frames)``.

    Returns
    -------
    torch.Tensor
        Output-projected context in the value projection's dtype.
    """

    batch_size, sequence_length, _ = x.shape
    values = module.in_proj(x).reshape(
        batch_size, sequence_length, module.num_heads, -1
    )
    weighted_values = torch.einsum(
        "bhqk,bkhd->bqhd", attention_weights.to(values.dtype), values
    )
    return module.out_proj(weighted_values.flatten(2))


def get_onnx_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return fixed ONNX dimensions as integers and symbolic ones as strings.

    Parameters
    ----------
    value : onnx.ValueInfoProto
        Tensor value information from an exported graph.

    Returns
    -------
    tuple[int | str, ...]
        Dimensions in their original axis order.
    """

    return tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


def check_onnx_model_with_custom_plugin(
    model: onnx.ModelProto, plugin_name: str
) -> onnx.NodeProto:
    """Check the opset and graph structure and return the sole matching plugin.

    Parameters
    ----------
    model : onnx.ModelProto
        Exported graph, left unchanged by this check.
    plugin_name : str
        Expected custom operator name.

    Returns
    -------
    onnx.NodeProto
        Matching node from the original graph.

    Notes
    -----
    The ONNX checker runs on a copy with custom plugin domains, validating
    graph wiring without requiring a standard schema for TensorRT operators.
    """

    assert [opset.version for opset in model.opset_import if opset.domain == ""] == [
        ONNX_OPSET_VERSION
    ]
    (plugin,) = [node for node in model.graph.node if node.op_type == plugin_name]
    assert plugin.domain == ""
    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(model)
    # TensorRT nodes have no standard ONNX schema; still check their graph wiring.
    for node in checker_model.graph.node:
        if node.op_type == plugin_name:
            node.domain = TENSORRT_PLUGIN_NAMESPACE
    checker_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(checker_model)
    return plugin


@pytest.mark.parametrize(("dtype", "atol", "rtol"), RELATIVE_DTYPE_CASES)
@pytest.mark.parametrize(
    ("sequence_length", "valid_lengths"),
    (
        pytest.param(1, (1, 1), id="single-frame"),
        pytest.param(7, (5, 7), id="halo-boundary"),
        pytest.param(12, (2, 4), id="padded-query-halo"),
    ),
)
def test_relative_attention_matches_indexed_reference(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    sequence_length: int,
    valid_lengths: tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(3)
    module = make_projected_relative_attention(dtype)
    x = torch.randn(
        (2, sequence_length, RELATIVE_ATTENTION_INPUT_DIM),
        dtype=dtype,
        generator=generator,
    )
    pos_emb = torch.randn(
        (1, 2 * sequence_length - 1, POSITION_INPUT_DIM),
        dtype=dtype,
        generator=generator,
    )
    key_padding_mask = (
        torch.arange(sequence_length)[None] >= torch.tensor(valid_lengths)[:, None]
    )
    expected = reference_relative_attention(
        module,
        x,
        pos_emb,
        key_padding_mask,
        NUM_HEADS,
        QUERY_HEAD_DIM,
        POSITION_HEAD_DIM,
    )

    actual = module(x, pos_emb, key_padding_mask)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for batch_index, valid_length in enumerate(valid_lengths):
        active_queries = min(sequence_length, valid_length + PADDED_QUERY_HALO)
        row_sums = actual[batch_index, :, :active_queries].sum(dim=2)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=atol, rtol=rtol
        )
        assert torch.count_nonzero(actual[batch_index, :, :, valid_length:]) == 0
        assert torch.count_nonzero(actual[batch_index, :, active_queries:]) == 0


def test_relative_attention_supports_alternate_head_layout() -> None:
    num_heads = 2
    query_head_dim = 3
    position_head_dim = 4
    sequence_length = 9
    generator = torch.Generator().manual_seed(41)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(47)
        module = RelPositionMultiheadAttentionWeights(
            11, 7, num_heads, query_head_dim, position_head_dim
        )
    x = torch.randn(1, sequence_length, 11, generator=generator)
    pos_emb = torch.randn(1, 2 * sequence_length - 1, 7, generator=generator)
    key_padding_mask = torch.arange(sequence_length).unsqueeze(0) >= 5

    expected = reference_relative_attention(
        module,
        x,
        pos_emb,
        key_padding_mask,
        num_heads,
        query_head_dim,
        position_head_dim,
    )
    actual = module(x, pos_emb, key_padding_mask)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_relative_attention_excludes_masked_scores_below_old_sentinel(
    dtype: torch.dtype,
) -> None:
    module = make_relative_attention().to(dtype)
    projection = torch.zeros(1, 2, RELATIVE_ATTENTION_DIM, dtype=dtype)
    projection[:, :, :RELATIVE_CONTENT_DIM] = 50.0
    projection[:, 0, RELATIVE_CONTENT_DIM : 2 * RELATIVE_CONTENT_DIM] = -40.0
    position = torch.zeros(1, 3, POSITION_DIM, dtype=dtype)
    mask = torch.tensor([[False, True]])

    actual = module(projection, position, mask)

    expected = torch.zeros(1, NUM_HEADS, 2, 2, dtype=dtype)
    expected[..., 0] = 1.0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_relative_attention_defines_all_masked_rows_as_zero(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(23)
    module = make_relative_attention().to(dtype)
    projection = torch.randn(
        1, 2, RELATIVE_ATTENTION_DIM, dtype=dtype, generator=generator
    )
    position = torch.randn(1, 3, POSITION_DIM, dtype=dtype, generator=generator)

    actual = module(projection, position, torch.ones(1, 2, dtype=torch.bool))

    expected = torch.zeros(1, NUM_HEADS, 2, 2, dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_relative_attention_exports_as_tensorrt_plugin(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(59)
    inputs = (
        torch.randn(
            2, 7, RELATIVE_ATTENTION_INPUT_DIM, dtype=dtype, generator=generator
        ),
        torch.randn(1, 13, POSITION_INPUT_DIM, dtype=dtype, generator=generator),
        torch.zeros(2, 7, dtype=torch.bool),
    )
    onnx_path = tmp_path / "relative_attention.onnx"
    dynamic_sequence_length = torch.export.Dim("sequence_length", min=1, max=32)

    # Reusing the required shared dimension triggers PyTorch's duplicate-name warning.
    with pytest.warns(
        UserWarning,
        match=r"The axis name: sequence_length .*another axis: sequence_length\.",
    ):
        torch.onnx.export(
            make_projected_relative_attention(dtype).eval(),
            inputs,
            onnx_path,
            input_names=("x", "pos_emb", "key_padding_mask"),
            output_names=("attention_weights",),
            dynamic_shapes=(
                {1: dynamic_sequence_length},
                {1: 2 * dynamic_sequence_length - 1},
                {1: dynamic_sequence_length},
            ),
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_node = check_onnx_model_with_custom_plugin(
        model, ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME
    )
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    }
    assert attributes == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode()}
    assert len(custom_node.input) == 3
    assert custom_node.input[0] != "x"
    assert custom_node.input[1] != "pos_emb"
    assert custom_node.input[2] == "key_padding_mask"
    assert list(custom_node.output) == ["attention_weights"]
    assert [value.name for value in model.graph.input] == [
        "x",
        "pos_emb",
        "key_padding_mask",
    ]
    assert [value.name for value in model.graph.output] == ["attention_weights"]

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    sequence_dimension = get_onnx_shape(value_info["x"])[1]
    assert isinstance(sequence_dimension, str)
    position_dimension = f"2*{sequence_dimension} - 1"
    expected_shapes = (
        ("x", (2, sequence_dimension, RELATIVE_ATTENTION_INPUT_DIM)),
        ("pos_emb", (1, position_dimension, POSITION_INPUT_DIM)),
        (custom_node.input[0], (2, sequence_dimension, RELATIVE_ATTENTION_DIM)),
        (custom_node.input[1], (1, position_dimension, NUM_HEADS, POSITION_HEAD_DIM)),
        ("attention_weights", (2, NUM_HEADS, sequence_dimension, sequence_dimension)),
    )
    for name, shape in expected_shapes:
        assert get_onnx_shape(value_info[name]) == shape, name
        assert value_info[name].type.tensor_type.elem_type == ONNX_DTYPES[dtype], name
    assert get_onnx_shape(value_info["key_padding_mask"]) == (2, sequence_dimension)
    assert (
        value_info["key_padding_mask"].type.tensor_type.elem_type
        == onnx.TensorProto.BOOL
    )
    assert not any(node.op_type == "Transpose" for node in model.graph.node)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_VALUE_DTYPE_CASES)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_self_attention_matches_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    generator = torch.Generator().manual_seed(4)
    weights = torch.softmax(
        torch.randn(
            2, NUM_HEADS, sequence_length, sequence_length, generator=generator
        ),
        dim=3,
    )
    module = make_self_attention(dtype)
    x = torch.randn(
        (2, sequence_length, SELF_ATTENTION_INPUT_DIM),
        dtype=dtype,
        generator=generator,
    )
    expected = reference_self_attention(module, x, weights)

    actual = module(x, weights)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_self_attention_supports_alternate_head_layout() -> None:
    generator = torch.Generator().manual_seed(67)
    module = make_self_attention(
        num_heads=ALTERNATE_SELF_NUM_HEADS,
        value_head_dim=ALTERNATE_SELF_VALUE_HEAD_DIM,
    )
    x = torch.randn(2, 5, SELF_ATTENTION_INPUT_DIM, generator=generator)
    weights = torch.softmax(
        torch.randn(2, ALTERNATE_SELF_NUM_HEADS, 5, 5, generator=generator),
        dim=3,
    )

    actual = module(x, weights)
    expected = reference_self_attention(module, x, weights)

    assert module.in_proj.out_features == (
        ALTERNATE_SELF_NUM_HEADS * ALTERNATE_SELF_VALUE_HEAD_DIM
    )
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_VALUE_DTYPE_CASES)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_nonlinear_attention_matches_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    generator = torch.Generator().manual_seed(5)
    weights = torch.softmax(
        torch.randn(
            2, NUM_HEADS, sequence_length, sequence_length, generator=generator
        ),
        dim=3,
    )
    module = make_nonlinear_attention(dtype)
    x = torch.randn(2, sequence_length, 12, dtype=dtype, generator=generator)
    gate, value, multiplier = module.in_proj(x).chunk(3, dim=2)
    expected = torch.einsum(
        "bqk,bkd->bqd", weights[:, 0].to(dtype), value * torch.tanh(gate)
    )
    expected = module.out_proj(expected * multiplier)

    actual = module(x, weights)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_nonlinear_attention_ignores_heads_after_first() -> None:
    generator = torch.Generator().manual_seed(53)
    module = make_nonlinear_attention()
    x = torch.randn(2, 5, 12, generator=generator)
    weights = torch.softmax(torch.randn(2, 3, 5, 5, generator=generator), dim=3)
    changed_weights = weights.clone()
    changed_weights[:, 1:] = torch.softmax(
        torch.randn(2, 2, 5, 5, generator=generator), dim=3
    )

    expected = module(x, weights)
    actual = module(x, changed_weights)

    assert not torch.equal(weights[:, 1:], changed_weights[:, 1:])
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("module_name", ("self", "nonlinear"))
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_attention_value_products_export_as_tensorrt_plugin(
    tmp_path: Path, module_name: str, dtype: torch.dtype
) -> None:
    generator = torch.Generator().manual_seed(61)
    if module_name == "self":
        module = make_self_attention(
            dtype,
            num_heads=ALTERNATE_SELF_NUM_HEADS,
            value_head_dim=ALTERNATE_SELF_VALUE_HEAD_DIM,
        ).eval()
        input_channels = SELF_ATTENTION_INPUT_DIM
        attention_heads = ALTERNATE_SELF_NUM_HEADS
        value_channels = ALTERNATE_SELF_NUM_HEADS * ALTERNATE_SELF_VALUE_HEAD_DIM
        value_heads = ALTERNATE_SELF_NUM_HEADS
    else:
        module = make_nonlinear_attention(dtype).eval()
        input_channels = 12
        attention_heads = NUM_HEADS
        value_channels = 11
        value_heads = 1
    x = torch.randn(2, 7, input_channels, dtype=dtype, generator=generator)
    attention_weights = torch.softmax(
        torch.randn(2, attention_heads, 7, 7, generator=generator), dim=3
    )
    onnx_path = tmp_path / "attention_value.onnx"
    dynamic_sequence_length = torch.export.Dim("sequence_length", min=1, max=32)

    # Keep query/key lengths tied to x; automatic dimensions lose that contract.
    with pytest.warns(
        UserWarning,
        match=r"The axis name: sequence_length .*another axis: sequence_length\.",
    ):
        torch.onnx.export(
            module,
            (x, attention_weights),
            onnx_path,
            input_names=("x", "attention_weights"),
            output_names=("output",),
            dynamic_shapes=(
                {1: dynamic_sequence_length},
                {2: dynamic_sequence_length, 3: dynamic_sequence_length},
            ),
            opset_version=ONNX_OPSET_VERSION,
        )

    model = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    custom_node = check_onnx_model_with_custom_plugin(
        model, ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME
    )
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    } == {
        "num_heads": value_heads,
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
    }
    assert len(custom_node.input) == 2
    assert len(custom_node.output) == 1
    assert custom_node.input[1] != "x"
    assert custom_node.output[0] != "output"
    assert not any(
        node.op_type in ("Reshape", "Transpose") for node in model.graph.node
    )
    assert [value.name for value in model.graph.input] == ["x", "attention_weights"]
    assert [value.name for value in model.graph.output] == ["output"]
    reachable_values = {custom_node.output[0]}
    for node in model.graph.node:
        if any(input_name in reachable_values for input_name in node.input):
            reachable_values.update(node.output)
    assert "output" in reachable_values

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    sequence_dimension = get_onnx_shape(value_info["x"])[1]
    assert isinstance(sequence_dimension, str)
    weights_shape = (2, attention_heads, sequence_dimension, sequence_dimension)
    expected_shapes = (
        ("x", (2, sequence_dimension, input_channels)),
        (custom_node.input[0], weights_shape),
        (custom_node.input[1], (2, sequence_dimension, value_channels)),
        (custom_node.output[0], (2, sequence_dimension, value_channels)),
        ("output", (2, sequence_dimension, input_channels)),
    )
    for name, shape in expected_shapes:
        assert get_onnx_shape(value_info[name]) == shape, name
        assert value_info[name].type.tensor_type.elem_type == ONNX_DTYPES[dtype], name
    assert get_onnx_shape(value_info["attention_weights"]) == weights_shape
    assert (
        value_info["attention_weights"].type.tensor_type.elem_type
        == onnx.TensorProto.FLOAT
    )


@pytest.mark.parametrize(
    ("embed_dim", "max_length", "sequence_length"),
    (
        pytest.param(4, 1, 1, id="minimum-capacity"),
        pytest.param(6, 8, 2, id="alternate-width"),
        pytest.param(8, 8, 8, id="maximum-capacity"),
        pytest.param(48, 8, 3, id="production-width"),
    ),
)
def test_compact_relative_positional_encoding_matches_scalar_formula(
    embed_dim: int, max_length: int, sequence_length: int
) -> None:
    encoding = CompactRelPositionalEncoding(embed_dim, max_length)
    inputs = torch.full((2, sequence_length, embed_dim), 3.0)
    actual = encoding(inputs)
    changed = encoding(torch.full_like(inputs, -5.0))
    expected = torch.empty(2 * sequence_length - 1, embed_dim)
    compression_length = math.sqrt(embed_dim)
    for row, offset in enumerate(range(-sequence_length + 1, sequence_length)):
        compressed = math.copysign(compression_length, offset) * math.log1p(
            abs(offset) / compression_length
        )
        angle = math.atan(2.0 * math.pi * compressed / embed_dim)
        for frequency in range(1, embed_dim // 2 + 1):
            expected[row, 2 * (frequency - 1)] = math.cos(angle * frequency)
            expected[row, 2 * frequency - 1] = math.sin(angle * frequency)
        expected[row, -1] = 1.0

    assert encoding.pos_emb.shape == (2 * max_length - 1, embed_dim)
    torch.testing.assert_close(actual, expected.unsqueeze(0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(changed, actual, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compact_relative_positional_encoding_uses_nonpersistent_typed_buffer(
    dtype: torch.dtype,
) -> None:
    encoding = CompactRelPositionalEncoding(8, 8)
    x = torch.zeros(2, 3, 8)
    expected = encoding(x).to(dtype)

    encoding.to(dtype)
    actual = encoding(x)

    assert [name for name, _ in encoding.named_buffers()] == ["pos_emb"]
    assert not encoding.pos_emb.requires_grad
    assert encoding.state_dict() == {}
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
