#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""CPU numerical tests for Parakeet attention modules."""

import math

import pytest
import torch

from fast_gpu_asr.export.model.parakeet.attention import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)

NUM_HEADS = 3
HEAD_DIM = 6
FEATURE_DIM = NUM_HEADS * HEAD_DIM
ATTENTION_DTYPE_CASES = (
    pytest.param(torch.float32, 1e-6, 1e-5, id="fp32"),
    pytest.param(torch.float16, 5e-4, 3e-3, id="fp16"),
    pytest.param(torch.bfloat16, 5e-3, 2e-2, id="bf16"),
)
FLOAT_DTYPES = (
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
)


def make_attention(
    dtype: torch.dtype = torch.float32,
    num_heads: int = NUM_HEADS,
    feature_dim: int = FEATURE_DIM,
) -> RelPositionMultiHeadAttention:
    """Create attention with deterministic, nontrivial parameters.

    Parameters
    ----------
    dtype : torch.dtype
        Parameter precision.
    num_heads : int
        Number of attention heads.
    feature_dim : int
        Feature width, divisible by ``num_heads``.

    Returns
    -------
    RelPositionMultiHeadAttention
        CPU attention module with seeded weights and unchanged global RNG.
    """

    generator = torch.Generator().manual_seed(8)
    with torch.random.fork_rng(devices=[]):
        attention = RelPositionMultiHeadAttention(num_heads, feature_dim).to(dtype)
    with torch.no_grad():
        for parameter in attention.parameters():
            values = 0.2 * torch.randn(parameter.shape, generator=generator)
            parameter.copy_(values.to(dtype))
    return attention


def reference_attention(
    attention: RelPositionMultiHeadAttention,
    x: torch.Tensor,
    pos_emb: torch.Tensor,
    output_lengths: torch.Tensor,
) -> torch.Tensor:
    """Evaluate relative attention by explicitly indexing every score.

    Parameters
    ----------
    attention : RelPositionMultiHeadAttention
        Source of projection parameters and positional biases.
    x : torch.Tensor
        Features of shape ``(batch, frames, feature_dim)``.
    pos_emb : torch.Tensor
        Relative embeddings of shape ``(1, 2 * frames - 1, feature_dim)``.
    output_lengths : torch.Tensor
        Valid key counts of shape ``(batch,)``; zero disables all keys.

    Returns
    -------
    torch.Tensor
        Output-projected context with the same shape and dtype as ``x``.
    """

    batch_size, sequence_length, feature_dim = x.shape
    num_heads, head_dim = attention.pos_bias_u.shape
    assert feature_dim == num_heads * head_dim
    query, key, value = attention.linear_qkv(x).chunk(3, dim=2)
    query = query.reshape(batch_size, sequence_length, num_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, num_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, num_heads, head_dim)
    position = attention.linear_pos(pos_emb.to(x.dtype)).reshape(
        pos_emb.size(0), pos_emb.size(1), num_heads, head_dim
    )

    scores = torch.empty(
        (batch_size, num_heads, sequence_length, sequence_length),
        dtype=torch.float32,
        device=x.device,
    )
    for query_index in range(sequence_length):
        for key_index in range(sequence_length):
            relative_index = sequence_length - 1 - query_index + key_index
            content_score = (
                (query[:, query_index] + attention.pos_bias_u) * key[:, key_index]
            ).sum(dim=2)
            position_score = (
                (query[:, query_index] + attention.pos_bias_v)
                * position[:, relative_index]
            ).sum(dim=2)
            scores[:, :, query_index, key_index] = (
                content_score.float() + position_score.float()
            ) / math.sqrt(head_dim)

    key_padding_mask = (
        torch.arange(sequence_length, device=x.device)[None] >= output_lengths[:, None]
    )
    weights = torch.softmax(
        scores.masked_fill(key_padding_mask[:, None, None], float("-inf")), dim=3
    )
    weights = weights.masked_fill((output_lengths <= 0)[:, None, None, None], 0.0).to(
        value.dtype
    )
    output = torch.einsum("bhqk,bkhd->bqhd", weights, value)
    return attention.linear_out(output.flatten(2))


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_DTYPE_CASES)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_parakeet_attention_matches_indexed_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    generator = torch.Generator().manual_seed(17 + sequence_length)
    attention = make_attention(dtype)
    x = torch.randn(2, sequence_length, FEATURE_DIM, dtype=dtype, generator=generator)
    pos_emb = torch.randn(
        (1, 2 * sequence_length - 1, FEATURE_DIM), dtype=dtype, generator=generator
    )
    output_lengths = torch.tensor(
        (sequence_length, max(1, sequence_length - 2)), dtype=torch.int32
    )

    expected = reference_attention(attention, x, pos_emb, output_lengths)

    actual = attention(x, pos_emb, output_lengths)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_parakeet_attention_supports_nondefault_head_layout() -> None:
    generator = torch.Generator().manual_seed(29)
    attention = make_attention(num_heads=4, feature_dim=12)
    inputs = torch.randn(2, 5, 12, generator=generator)
    positions = torch.randn(1, 9, 12, generator=generator)
    output_lengths = torch.tensor((5, 2), dtype=torch.int32)
    expected = reference_attention(attention, inputs, positions, output_lengths)

    actual = attention(inputs, positions, output_lengths)

    assert actual.shape == inputs.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_DTYPE_CASES[1:])
def test_parakeet_attention_casts_positions_to_input_dtype(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    generator = torch.Generator().manual_seed(31)
    attention = make_attention(dtype)
    inputs = torch.randn(2, 4, FEATURE_DIM, dtype=dtype, generator=generator)
    positions = torch.randn(1, 7, FEATURE_DIM, dtype=torch.float32, generator=generator)
    output_lengths = torch.tensor((4, 2), dtype=torch.int32)
    expected = reference_attention(attention, inputs, positions, output_lengths)

    actual = attention(inputs, positions, output_lengths)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("dtype", "position_delta"),
    (
        pytest.param(torch.float16, 3.5e-4, id="fp16"),
        pytest.param(torch.bfloat16, 2.77e-3, id="bf16"),
    ),
)
def test_parakeet_attention_promotes_score_sum_before_softmax(
    dtype: torch.dtype, position_delta: float
) -> None:
    with torch.random.fork_rng(devices=[]):
        attention = RelPositionMultiHeadAttention(n_head=1, n_feat=2).to(dtype)
    with torch.no_grad():
        # Produce content logits (1, 1), positional logits (+delta, -delta),
        # and scalar values (0, 1) in the first output channel.
        attention.linear_qkv.weight.zero_()
        attention.linear_qkv.weight[3, 1] = 1.0
        attention.linear_qkv.weight[4, 0] = 1.0
        attention.linear_pos.weight.copy_(torch.eye(2, dtype=dtype))
        attention.linear_out.weight.copy_(torch.eye(2, dtype=dtype))
        attention.pos_bias_u.copy_(torch.tensor(((0.0, 1.0),), dtype=dtype))
        attention.pos_bias_v.copy_(torch.tensor(((1.0, 0.0),), dtype=dtype))

    inputs = torch.tensor(((0.0, 1.0), (1.0, 1.0)), dtype=dtype).unsqueeze(0)
    positions = torch.zeros(1, 3, 2, dtype=dtype)
    positions[0, 1, 0] = position_delta
    positions[0, 2, 0] = -position_delta

    output = attention(inputs, positions, torch.tensor((2,), dtype=torch.int32))
    position_scores = positions[0, 1:, 0]
    promoted_logits = (
        torch.ones(2, dtype=torch.float32) + position_scores.float()
    ) / math.sqrt(2.0)
    promoted_weight = torch.softmax(promoted_logits, dim=0)[1].to(dtype)
    low_precision_logits = (torch.ones(2, dtype=dtype) + position_scores) / math.sqrt(
        2.0
    )
    low_precision_weight = torch.softmax(low_precision_logits, dim=0)[1]

    assert not torch.equal(promoted_weight, low_precision_weight)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output[0, 0],
        torch.stack((promoted_weight, torch.zeros((), dtype=dtype))),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_parakeet_attention_excludes_padding_below_old_mask_floor(
    dtype: torch.dtype,
) -> None:
    with torch.random.fork_rng(devices=[]):
        attention = RelPositionMultiHeadAttention(n_head=1, n_feat=2).to(dtype)
    with torch.no_grad():
        attention.linear_qkv.weight.zero_()
        attention.linear_qkv.weight[0, 1] = 1.0
        attention.linear_qkv.weight[2, 0] = -1200.0 * 2**0.5
        attention.linear_qkv.weight[4:] = torch.eye(2, dtype=dtype)
        attention.linear_pos.weight.zero_()
        attention.linear_out.weight.copy_(torch.eye(2, dtype=dtype))

    output = attention(
        torch.tensor(((1.0, 1.0), (0.0, 0.0)), dtype=dtype).unsqueeze(0),
        torch.zeros(1, 3, 2, dtype=dtype),
        torch.tensor((1,), dtype=torch.int32),
    )

    expected = torch.ones(1, 2, 2, dtype=dtype)
    torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_DTYPE_CASES)
def test_parakeet_attention_zeros_only_utterances_without_valid_frames(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    generator = torch.Generator().manual_seed(19)
    attention = make_attention(dtype)
    inputs = torch.randn(2, 3, FEATURE_DIM, dtype=dtype, generator=generator)
    positions = torch.randn(1, 5, FEATURE_DIM, dtype=dtype, generator=generator)
    output_lengths = torch.tensor((3, 0), dtype=torch.int32)
    expected = reference_attention(attention, inputs, positions, output_lengths)

    output = attention(inputs, positions, output_lengths)

    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    assert torch.count_nonzero(output[0]) > 0
    torch.testing.assert_close(
        output[1], torch.zeros_like(output[1]), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize(("dtype", "atol", "rtol"), ATTENTION_DTYPE_CASES)
def test_parakeet_attention_ignores_padded_keys_for_valid_queries(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    sequence_length = 6
    valid_length = 3
    generator = torch.Generator().manual_seed(43)
    attention = make_attention(dtype)
    inputs = torch.randn(
        1, sequence_length, FEATURE_DIM, dtype=dtype, generator=generator
    )
    changed_inputs = inputs.clone()
    changed_inputs[:, valid_length:] *= -8
    positions = torch.randn(
        (1, 2 * sequence_length - 1, FEATURE_DIM), dtype=dtype, generator=generator
    )
    output_lengths = torch.tensor((valid_length,), dtype=torch.int32)

    expected = reference_attention(attention, inputs, positions, output_lengths)
    changed_expected = reference_attention(
        attention, changed_inputs, positions, output_lengths
    )

    output = attention(inputs, positions, output_lengths)
    changed_output = attention(changed_inputs, positions, output_lengths)

    assert not torch.equal(inputs[:, valid_length:], changed_inputs[:, valid_length:])
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(changed_output, changed_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        output[:, :valid_length], changed_output[:, :valid_length], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    ("max_len", "sequence_length"), ((1, 1), (8, 1), (8, 2), (8, 8))
)
def test_parakeet_relative_positional_encoding_matches_formula(
    dtype: torch.dtype, max_len: int, sequence_length: int
) -> None:
    model_dim = 8
    encoding = RelPositionalEncoding(model_dim, max_len).to(dtype)
    inputs = torch.full((2, sequence_length, model_dim), 3.0)
    expected = torch.empty(2 * sequence_length - 1, model_dim)
    for row, position in enumerate(range(sequence_length - 1, -sequence_length, -1)):
        for channel in range(0, model_dim, 2):
            frequency = math.exp(-math.log(10000.0) * channel / model_dim)
            expected[row, channel] = math.sin(position * frequency)
            expected[row, channel + 1] = math.cos(position * frequency)

    actual = encoding(inputs)

    torch.testing.assert_close(
        actual, expected.unsqueeze(0).to(dtype), atol=1e-6, rtol=0.0
    )


def test_parakeet_relative_positional_encoding_uses_nonpersistent_buffer() -> None:
    encoding = RelPositionalEncoding(8, 8)

    assert [name for name, _ in encoding.named_buffers()] == ["pos_emb"]
    assert not encoding.pos_emb.requires_grad
    assert encoding.state_dict() == {}
