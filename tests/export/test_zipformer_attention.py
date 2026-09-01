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
POSITION_HEAD_DIM = 4
RELATIVE_ATTENTION_DIM = NUM_HEADS * (2 * QUERY_HEAD_DIM + POSITION_HEAD_DIM)
POSITION_DIM = NUM_HEADS * POSITION_HEAD_DIM
SELF_ATTENTION_DIM = NUM_HEADS * QUERY_HEAD_DIM
PADDED_QUERY_HALO = 7


def make_relative_attention() -> RelPositionMultiheadAttentionWeights:
    """Create relative attention whose projections preserve test inputs."""

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


def get_onnx_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    """Return fixed ONNX dimensions as integers and symbolic ones as strings."""

    return tuple(
        dimension.dim_param or dimension.dim_value
        for dimension in value.type.tensor_type.shape.dim
    )


def check_onnx_model_with_custom_plugin(
    model: onnx.ModelProto, plugin_name: str
) -> None:
    """Validate an ONNX graph after assigning its TensorRT node a custom domain."""

    checker_model = onnx.ModelProto()
    checker_model.CopyFrom(model)
    custom_nodes = [
        node for node in checker_model.graph.node if node.op_type == plugin_name
    ]
    assert len(custom_nodes) == 1
    custom_nodes[0].domain = TENSORRT_PLUGIN_NAMESPACE
    checker_model.opset_import.append(
        onnx.helper.make_opsetid(TENSORRT_PLUGIN_NAMESPACE, 1)
    )
    onnx.checker.check_model(checker_model)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 5e-4, 2e-3),
        (torch.bfloat16, 2e-3, 2e-2),
    ),
)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_relative_attention_matches_indexed_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    """Match relative attention against explicit query-key indexing."""

    generator = torch.Generator().manual_seed(3)
    batch_size = 2
    query = torch.randn(
        batch_size,
        sequence_length,
        NUM_HEADS,
        QUERY_HEAD_DIM,
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        batch_size,
        sequence_length,
        NUM_HEADS,
        QUERY_HEAD_DIM,
        dtype=dtype,
        generator=generator,
    )
    position_query = torch.randn(
        batch_size,
        sequence_length,
        NUM_HEADS,
        POSITION_HEAD_DIM,
        dtype=dtype,
        generator=generator,
    )
    position = torch.randn(
        1,
        2 * sequence_length - 1,
        NUM_HEADS,
        POSITION_HEAD_DIM,
        dtype=dtype,
        generator=generator,
    )
    projection = torch.cat(
        (
            query.reshape(batch_size, sequence_length, -1),
            key.reshape(batch_size, sequence_length, -1),
            position_query.reshape(batch_size, sequence_length, -1),
        ),
        dim=2,
    )
    key_padding_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    if sequence_length > 2:
        key_padding_mask[0, -2:] = True
    scores = torch.empty(
        batch_size,
        NUM_HEADS,
        sequence_length,
        sequence_length,
        dtype=dtype,
    )
    for query_index in range(sequence_length):
        for key_index in range(sequence_length):
            relative_index = sequence_length - 1 - query_index + key_index
            scores[:, :, query_index, key_index] = (
                query[:, query_index] * key[:, key_index]
            ).sum(dim=2) + (
                position_query[:, query_index] * position[:, relative_index]
            ).sum(dim=2)
    expanded_mask = key_padding_mask[:, None, None]
    expected = torch.softmax(
        scores.masked_fill(expanded_mask, float("-inf")), dim=3
    ).masked_fill(expanded_mask, 0.0)

    actual = make_relative_attention().to(dtype)(
        projection,
        position.reshape(1, 2 * sequence_length - 1, -1),
        key_padding_mask,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual.sum(dim=3),
        torch.ones_like(actual.sum(dim=3)),
        atol=atol,
        rtol=rtol,
    )
    assert torch.count_nonzero(actual[0, :, :, key_padding_mask[0]]) == 0


def test_relative_attention_excludes_masked_scores_below_old_sentinel() -> None:
    """Keep a masked key excluded even when every valid score is below -1000."""

    module = make_relative_attention()
    projection = torch.zeros(1, 2, RELATIVE_ATTENTION_DIM)
    projection[:, :, :SELF_ATTENTION_DIM] = 50.0
    projection[:, 0, SELF_ATTENTION_DIM : 2 * SELF_ATTENTION_DIM] = -40.0
    position = torch.zeros(1, 3, POSITION_DIM)
    mask = torch.tensor([[False, True]])

    actual = module(projection, position, mask)

    torch.testing.assert_close(actual[..., 0], torch.ones_like(actual[..., 0]))
    torch.testing.assert_close(actual[..., 1], torch.zeros_like(actual[..., 1]))


def test_relative_attention_defines_all_masked_rows_as_zero() -> None:
    """Avoid NaNs when no source key is valid."""

    generator = torch.Generator().manual_seed(23)
    module = make_relative_attention()
    projection = torch.randn(1, 2, RELATIVE_ATTENTION_DIM, generator=generator)
    position = torch.randn(1, 3, POSITION_DIM, generator=generator)

    actual = module(projection, position, torch.ones(1, 2, dtype=torch.bool))

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_relative_attention_zeros_queries_inside_padded_suffix() -> None:
    """Match the TensorRT plugin's seven-frame convolution halo."""

    generator = torch.Generator().manual_seed(17)
    module = make_relative_attention()
    sequence_length = 12
    projection = torch.randn(
        1, sequence_length, RELATIVE_ATTENTION_DIM, generator=generator
    )
    position = torch.randn(
        1, 2 * sequence_length - 1, POSITION_DIM, generator=generator
    )
    mask = torch.zeros(1, sequence_length, dtype=torch.bool)
    mask[:, 2:] = True

    actual = module(projection, position, mask)

    active_queries = 2 + PADDED_QUERY_HALO
    torch.testing.assert_close(
        actual[:, :, :active_queries].sum(dim=3),
        torch.ones_like(actual[:, :, :active_queries].sum(dim=3)),
    )
    torch.testing.assert_close(
        actual[:, :, active_queries:],
        torch.zeros_like(actual[:, :, active_queries:]),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_relative_attention_exports_as_tensorrt_plugin(
    tmp_path: Path,
    dtype: torch.dtype,
) -> None:
    """Check that ONNX retains one typed custom node for TensorRT parsing."""

    sequence_length = 7
    inputs = (
        torch.randn(2, sequence_length, RELATIVE_ATTENTION_DIM, dtype=dtype),
        torch.randn(1, 2 * sequence_length - 1, POSITION_DIM, dtype=dtype),
        torch.zeros(2, sequence_length, dtype=torch.bool),
    )
    onnx_path = tmp_path / f"relative_attention_{dtype}.onnx"
    dynamic_sequence_length = torch.export.Dim("sequence_length", min=1, max=32)

    torch.onnx.export(
        make_relative_attention().eval().to(dtype),
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
    check_onnx_model_with_custom_plugin(model, ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME)
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert (custom_node.domain, custom_node.op_type) == (
        "",
        ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
    )
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    }
    assert attributes == {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode("ascii")}
    assert len(custom_node.input) == 3
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
    expected_element_type = {
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
    }[dtype]
    x_shape = get_onnx_shape(value_info["x"])
    position_shape = get_onnx_shape(value_info["pos_emb"])
    sequence_dimension = x_shape[1]
    position_dimension = position_shape[1]
    assert isinstance(sequence_dimension, str)
    assert isinstance(position_dimension, str)
    assert position_dimension == f"2*{sequence_dimension} - 1"
    assert x_shape == (2, sequence_dimension, RELATIVE_ATTENTION_DIM)
    assert position_shape == (1, position_dimension, POSITION_DIM)
    assert get_onnx_shape(value_info["key_padding_mask"]) == (
        2,
        sequence_dimension,
    )
    assert get_onnx_shape(value_info[custom_node.input[0]]) == x_shape
    assert get_onnx_shape(value_info[custom_node.input[1]]) == (
        1,
        position_dimension,
        NUM_HEADS,
        POSITION_HEAD_DIM,
    )
    assert get_onnx_shape(value_info["attention_weights"]) == (
        2,
        NUM_HEADS,
        sequence_dimension,
        sequence_dimension,
    )
    for name in ("x", "pos_emb", *custom_node.input[:2], "attention_weights"):
        assert value_info[name].type.tensor_type.elem_type == expected_element_type
    assert (
        value_info["key_padding_mask"].type.tensor_type.elem_type
        == onnx.TensorProto.BOOL
    )
    assert not any(node.op_type == "Transpose" for node in model.graph.node)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 5e-4, 2e-3),
        (torch.bfloat16, 3e-3, 2e-2),
    ),
)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_self_attention_matches_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    """Match an independent multi-head contraction with reusable FP32 weights."""

    generator = torch.Generator().manual_seed(4)
    weights = torch.softmax(
        torch.randn(
            2,
            NUM_HEADS,
            sequence_length,
            sequence_length,
            generator=generator,
        ),
        dim=3,
    )
    self_attention = SelfAttention(SELF_ATTENTION_DIM, NUM_HEADS, QUERY_HEAD_DIM).to(
        dtype
    )
    self_input = torch.randn(
        2,
        sequence_length,
        SELF_ATTENTION_DIM,
        dtype=dtype,
        generator=generator,
    )
    self_value = self_attention.in_proj(self_input).reshape(
        2, sequence_length, NUM_HEADS, QUERY_HEAD_DIM
    )
    self_expected = torch.einsum("bhqk,bkhd->bqhd", weights.to(dtype), self_value)
    self_expected = self_attention.out_proj(
        self_expected.reshape(2, sequence_length, SELF_ATTENTION_DIM)
    )

    self_actual = self_attention(self_input, weights)

    assert self_actual.dtype == dtype
    torch.testing.assert_close(self_actual, self_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 5e-4, 2e-3),
        (torch.bfloat16, 3e-3, 2e-2),
    ),
)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_nonlinear_attention_matches_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    """Match gated first-head attention with reusable FP32 weights."""

    generator = torch.Generator().manual_seed(5)
    weights = torch.softmax(
        torch.randn(
            2,
            NUM_HEADS,
            sequence_length,
            sequence_length,
            generator=generator,
        ),
        dim=3,
    )
    nonlin_attention = NonlinAttention(12, 11).to(dtype)
    nonlin_input = torch.randn(2, sequence_length, 12, dtype=dtype, generator=generator)
    gate, value, multiplier = nonlin_attention.in_proj(nonlin_input).chunk(3, dim=2)
    nonlin_expected = torch.einsum(
        "bqk,bkd->bqd", weights[:, 0].to(dtype), value * torch.tanh(gate)
    )
    nonlin_expected = nonlin_attention.out_proj(nonlin_expected * multiplier)

    nonlin_actual = nonlin_attention(nonlin_input, weights)
    assert nonlin_actual.dtype == dtype
    torch.testing.assert_close(nonlin_actual, nonlin_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("module_name", ("self", "nonlinear"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_attention_value_products_export_as_tensorrt_plugin(
    tmp_path: Path, module_name: str, dtype: torch.dtype
) -> None:
    """Preserve the typed NTC attention-value plugin contract without transposes."""

    if module_name == "self":
        module = (
            SelfAttention(SELF_ATTENTION_DIM, NUM_HEADS, QUERY_HEAD_DIM)
            .eval()
            .to(dtype)
        )
        x = torch.randn(2, 7, SELF_ATTENTION_DIM, dtype=dtype)
        value_channels = SELF_ATTENTION_DIM
        value_heads = NUM_HEADS
    else:
        module = NonlinAttention(12, 11).eval().to(dtype)
        x = torch.randn(2, 7, 12, dtype=dtype)
        value_channels = 11
        value_heads = 1
    attention_weights = torch.softmax(torch.randn(2, NUM_HEADS, 7, 7), dim=3)
    onnx_path = tmp_path / f"{module_name}_attention_{dtype}.onnx"
    dynamic_sequence_length = torch.export.Dim("sequence_length", min=1, max=32)

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
    check_onnx_model_with_custom_plugin(model, ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME)
    custom_nodes = [
        node
        for node in model.graph.node
        if node.op_type == ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    assert (custom_node.domain, custom_node.op_type) == (
        "",
        ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
    )
    assert {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in custom_node.attribute
    } == {
        "num_heads": value_heads,
        "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE.encode(),
    }
    assert len(custom_node.input) == 2
    assert not any(
        node.op_type in ("Reshape", "Transpose") for node in model.graph.node
    )

    value_info = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    expected_element_type = {
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
    }[dtype]
    x_shape = get_onnx_shape(value_info["x"])
    sequence_dimension = x_shape[1]
    assert isinstance(sequence_dimension, str)
    assert x_shape == (2, sequence_dimension, x.size(2))
    assert get_onnx_shape(value_info["attention_weights"]) == (
        2,
        NUM_HEADS,
        sequence_dimension,
        sequence_dimension,
    )
    assert get_onnx_shape(value_info[custom_node.input[0]]) == (
        2,
        NUM_HEADS,
        sequence_dimension,
        sequence_dimension,
    )
    assert get_onnx_shape(value_info[custom_node.input[1]]) == (
        2,
        sequence_dimension,
        value_channels,
    )
    assert get_onnx_shape(value_info[custom_node.output[0]]) == (
        2,
        sequence_dimension,
        value_channels,
    )
    assert get_onnx_shape(value_info["output"]) == x_shape
    assert (
        value_info["attention_weights"].type.tensor_type.elem_type
        == onnx.TensorProto.FLOAT
    )
    for name in (*custom_node.input, custom_node.output[0], "output"):
        assert value_info[name].type.tensor_type.elem_type == expected_element_type


@pytest.mark.parametrize("sequence_length", (1, 2, 8))
def test_compact_relative_positional_encoding_matches_scalar_formula(
    sequence_length: int,
) -> None:
    """Match the precomputed position table against its scalar definition."""

    embed_dim = 8
    encoding = CompactRelPositionalEncoding(embed_dim=embed_dim, max_length=8)

    actual = encoding(torch.zeros(2, sequence_length, embed_dim))
    expected = torch.empty(2 * sequence_length - 1, embed_dim)
    compression_length = math.sqrt(embed_dim)
    for row, offset in enumerate(range(-sequence_length + 1, sequence_length)):
        compressed = (
            compression_length
            * (-1.0 if offset < 0 else 1.0 if offset > 0 else 0.0)
            * (
                math.log(abs(offset) + compression_length)
                - math.log(compression_length)
            )
        )
        angle = math.atan(2.0 * math.pi * compressed / embed_dim)
        for frequency in range(1, embed_dim // 2 + 1):
            expected[row, 2 * (frequency - 1)] = math.cos(angle * frequency)
            expected[row, 2 * frequency - 1] = math.sin(angle * frequency)
        expected[row, -1] = 1.0

    assert actual.shape == (1, 2 * sequence_length - 1, embed_dim)
    assert actual.dtype == torch.float32
    assert encoding.state_dict() == {}
    torch.testing.assert_close(actual[0], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_compact_relative_positional_encoding_follows_module_dtype(
    dtype: torch.dtype,
) -> None:
    """Preserve positional values when converting the module's dtype."""

    expected = CompactRelPositionalEncoding(embed_dim=8, max_length=8)(
        torch.zeros(2, 3, 8)
    ).to(dtype)
    encoding = CompactRelPositionalEncoding(embed_dim=8, max_length=8).to(dtype)

    actual = encoding(torch.zeros(2, 3, 8, dtype=dtype))

    assert actual.shape == (1, 5, 8)
    assert actual.dtype == dtype
    assert actual.device == encoding.pos_emb.device
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
