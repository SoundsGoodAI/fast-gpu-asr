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


def make_attention(dtype: torch.dtype = torch.float32) -> RelPositionMultiHeadAttention:
    """Create attention with deterministic, nontrivial parameters."""

    generator = torch.Generator().manual_seed(8)
    with torch.random.fork_rng(devices=[]):
        attention = RelPositionMultiHeadAttention(NUM_HEADS, FEATURE_DIM).to(dtype)
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
    """Evaluate relative attention by explicitly indexing every score."""

    batch_size, sequence_length, feature_dim = x.shape
    query, key, value = attention.linear_qkv(x).chunk(3, dim=2)
    query = query.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    key = key.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    value = value.reshape(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
    position = (
        attention.linear_pos(pos_emb)
        .reshape(1, 2 * sequence_length - 1, NUM_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
    )

    scores = torch.empty(
        batch_size,
        NUM_HEADS,
        sequence_length,
        sequence_length,
        dtype=torch.float32,
    )
    for query_index in range(sequence_length):
        for key_index in range(sequence_length):
            relative_index = sequence_length - 1 - query_index + key_index
            content_score = (
                (query[:, :, query_index] + attention.pos_bias_u) * key[:, :, key_index]
            ).sum(dim=2)
            position_score = (
                (query[:, :, query_index] + attention.pos_bias_v)
                * position[:, :, relative_index]
            ).sum(dim=2)
            scores[:, :, query_index, key_index] = (
                content_score.to(torch.float32) + position_score.to(torch.float32)
            ) / HEAD_DIM**0.5

    key_padding_mask = torch.arange(sequence_length).unsqueeze(
        0
    ) >= output_lengths.unsqueeze(1)
    weights = torch.softmax(
        scores.masked_fill(key_padding_mask[:, None, None], float("-inf")), dim=3
    )
    weights = weights.masked_fill(
        (output_lengths <= 0).reshape(batch_size, 1, 1, 1), 0.0
    ).to(value.dtype)
    output = torch.matmul(weights, value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return attention.linear_out(
        output.reshape(batch_size, sequence_length, feature_dim)
    )


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    (
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 5e-4, 3e-3),
        (torch.bfloat16, 5e-3, 2e-2),
    ),
)
@pytest.mark.parametrize("sequence_length", (1, 7))
def test_parakeet_attention_matches_indexed_reference(
    dtype: torch.dtype, atol: float, rtol: float, sequence_length: int
) -> None:
    """Match relative indexing, masking, projection, and mixed-precision behavior."""

    generator = torch.Generator().manual_seed(17 + sequence_length)
    attention = make_attention(dtype)
    x = torch.randn(2, sequence_length, FEATURE_DIM, dtype=dtype, generator=generator)
    pos_emb = torch.randn(
        1,
        2 * sequence_length - 1,
        FEATURE_DIM,
        dtype=dtype,
        generator=generator,
    )
    output_lengths = torch.tensor(
        (sequence_length, max(1, sequence_length - 2)), dtype=torch.int32
    )

    actual = attention(x, pos_emb, output_lengths)
    expected = reference_attention(attention, x, pos_emb, output_lengths)

    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_parakeet_attention_excludes_padding_below_old_mask_floor() -> None:
    """Keep a masked key excluded when every valid score is below -1000."""

    with torch.random.fork_rng(devices=[]):
        attention = RelPositionMultiHeadAttention(n_head=1, n_feat=2)
    with torch.no_grad():
        attention.linear_qkv.weight.zero_()
        attention.linear_qkv.weight[0, 1] = 1.0
        attention.linear_qkv.weight[2, 0] = -1200.0 * 2**0.5
        attention.linear_qkv.weight[4:] = torch.eye(2)
        attention.linear_pos.weight.zero_()
        attention.linear_out.weight.copy_(torch.eye(2))

    output = attention(
        torch.tensor(((1.0, 1.0), (0.0, 0.0))).unsqueeze(0),
        torch.zeros(1, 3, 2),
        torch.tensor((1,), dtype=torch.int32),
    )

    torch.testing.assert_close(output, torch.ones_like(output))


def test_parakeet_attention_zeros_only_utterances_without_valid_frames() -> None:
    """Return finite zeros per utterance when all of its keys are padded."""

    generator = torch.Generator().manual_seed(19)
    attention = make_attention()
    inputs = torch.randn(2, 3, FEATURE_DIM, generator=generator)
    positions = torch.randn(1, 5, FEATURE_DIM, generator=generator)

    output = attention(inputs, positions, torch.tensor((3, 0), dtype=torch.int32))

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) > 0
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("sequence_length", (1, 2, 8))
def test_parakeet_relative_positional_encoding_matches_formula(
    dtype: torch.dtype, sequence_length: int
) -> None:
    """Match every relative sinusoidal value and preserve buffer semantics."""

    model_dim = 8
    encoding = RelPositionalEncoding(model_dim=model_dim, max_len=8).to(dtype)
    inputs = torch.zeros(2, sequence_length, model_dim, dtype=dtype)
    expected = torch.empty(2 * sequence_length - 1, model_dim)
    for row, position in enumerate(range(sequence_length - 1, -sequence_length, -1)):
        for channel in range(0, model_dim, 2):
            frequency = math.exp(-math.log(10000.0) * channel / model_dim)
            expected[row, channel] = math.sin(position * frequency)
            expected[row, channel + 1] = math.cos(position * frequency)

    actual = encoding(inputs)

    assert actual.shape == (1, 2 * sequence_length - 1, model_dim)
    assert actual.dtype == dtype
    assert actual.device == encoding.pos_emb.device
    assert encoding.state_dict() == {}
    torch.testing.assert_close(actual[0], expected.to(dtype), atol=1e-6, rtol=0.0)
