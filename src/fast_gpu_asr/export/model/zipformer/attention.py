#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright 2022-2023 Xiaomi Corp. (Daniel Povey, Zengwei Yao)
# Modified from Icefall for batched TensorRT export; see NOTICE and LICENSE.
"""Offline Zipformer attention modules adapted for export-friendly inference."""

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
    ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
)


class SelfAttention(torch.nn.Module):
    """Apply precomputed multi-head attention weights to projected values."""

    def __init__(self, embed_dim: int, num_heads: int, value_head_dim: int) -> None:
        """Initialize the value and output projections.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for input and output of this module.
        num_heads : int
            The number of attention heads.
        value_head_dim : int
            The dimension of the value per head.
        """

        super().__init__()

        self.num_heads = num_heads
        self.in_proj = torch.nn.Linear(embed_dim, num_heads * value_head_dim)
        self.out_proj = torch.nn.Linear(num_heads * value_head_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply precomputed attention weights to projected input values.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        attn_weights : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Weights with shape ``(batch_size, num_heads, seq_len, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Attention weighted output of shape ``(batch_size, seq_len, embed_dim)``.
            The tensor has the same shape and dtype as ``x``. ONNX export passes
            projected values to the native Zipformer TensorRT plugin directly in
            NTC layout, avoiding external head-layout transposes.
        """

        x = self.in_proj(x)  # (batch_size, seq_len, num_heads * value_head_dim)
        attn_weights = attn_weights.to(x.dtype)

        if torch.onnx.is_in_onnx_export():
            x = torch.onnx.ops.symbolic(
                ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
                (attn_weights, x),
                {
                    "num_heads": self.num_heads,
                    "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE,
                },
                dtype=x.dtype,
                shape=x.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            batch_size = x.size(0)
            x = x.reshape(
                batch_size, x.size(1), self.num_heads, x.size(2) // self.num_heads
            ).permute(0, 2, 1, 3)
            x = torch.matmul(attn_weights, x).permute(0, 2, 1, 3)
            x = x.reshape(batch_size, x.size(1), self.num_heads * x.size(3))

        # The returned value has the same (N, T, C) shape as the input.
        return self.out_proj(x)


class NonlinAttention(torch.nn.Module):
    """Apply gated nonlinear attention using one precomputed attention head."""

    def __init__(self, embed_dim: int, att_dim: int) -> None:
        """Initialize the gated input and output projections.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for the input and output.
        att_dim : int
            The attention output dimension of this module.
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, att_dim * 3)
        self.out_proj = torch.nn.Linear(att_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply gated nonlinear attention to one input sequence.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        attn_weights : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Weights with shape ``(batch_size, num_heads, seq_len, seq_len)``.
            Nonlinear attention consumes the first attention head.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Attention weighted output of shape ``(batch_size, seq_len, embed_dim)``.
            The tensor has the same shape and dtype as ``x``. ONNX export passes
            projected values to the native Zipformer TensorRT plugin directly in
            NTC layout, avoiding external head-layout transposes.
        """

        s, x, y = self.in_proj(x).chunk(3, dim=2)
        x = x * torch.tanh(s)
        attn_weights = attn_weights.to(x.dtype)

        if torch.onnx.is_in_onnx_export():
            x = torch.onnx.ops.symbolic(
                ZIPFORMER_ATTENTION_VALUE_PLUGIN_NAME,
                (attn_weights, x),
                {"num_heads": 1, "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=x.dtype,
                shape=x.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            x = torch.matmul(attn_weights[:, 0], x)

        return self.out_proj(x * y)


class RelPositionMultiheadAttentionWeights(torch.nn.Module):
    """Compute reusable Transformer-XL-style relative attention weights."""

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
    ) -> None:
        """Initialize content and relative-position projections.

        Parameters
        ----------
        embed_dim : int
            The embedding dimension. The number of channels at the input to this module.
        pos_dim : int
            A dimension of the positional embeddings.
        num_heads : int
            The number of attention heads to compute weights.
        query_head_dim : int
            The dimension of the query and key per head.
        pos_head_dim : int
            The dimension of the projected positional encoding per head. The
            native TensorRT relative-attention plugin requires four channels.
        """

        super().__init__()

        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.content_dim = num_heads * query_head_dim

        in_proj_dim = (2 * query_head_dim + pos_head_dim) * num_heads
        self.in_proj = torch.nn.Linear(embed_dim, in_proj_dim)

        # Linear transformation for positional encoding.
        self.linear_pos = torch.nn.Linear(pos_dim, num_heads * pos_head_dim, bias=False)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized relative-position attention weights.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        pos_emb : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Positional embeddings with shape ``(1, 2 * seq_len - 1, pos_dim)``.
            The singleton batch dimension is broadcast across the input batch.
        key_padding_mask : torch.Tensor[torch.bool]
            Padding mask with shape ``(batch_size, seq_len)``. True positions
            must form one contiguous suffix and are excluded as attention
            sources.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Attention weights with shape ``(batch_size, num_heads, seq_len, seq_len)``.
            ONNX export represents the complete calculation with the native
            Zipformer relative-attention TensorRT plugin. Query rows at least
            seven positions inside the padded suffix are zero because they
            cannot contribute to a valid downstream convolution output.
        """

        projection = self.in_proj(x)
        position = self.linear_pos(pos_emb).reshape(
            pos_emb.size(0), pos_emb.size(1), self.num_heads, self.pos_head_dim
        )

        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                ZIPFORMER_RELATIVE_ATTENTION_PLUGIN_NAME,
                (projection, position, key_padding_mask),
                {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=projection.dtype,
                shape=(
                    projection.shape[0],
                    self.num_heads,
                    projection.shape[1],
                    projection.shape[1],
                ),
                version=ONNX_OPSET_VERSION,
            )

        batch_size = projection.size(0)
        sequence_length = projection.size(1)

        query = (
            projection[:, :, : self.content_dim]
            .reshape(batch_size, sequence_length, self.num_heads, self.query_head_dim)
            .permute(0, 2, 1, 3)
        )
        key = (
            projection[:, :, self.content_dim : 2 * self.content_dim]
            .reshape(batch_size, sequence_length, self.num_heads, self.query_head_dim)
            .permute(0, 2, 3, 1)
        )
        position_query = (
            projection[:, :, 2 * self.content_dim :]
            .reshape(batch_size, sequence_length, self.num_heads, self.pos_head_dim)
            .permute(0, 2, 1, 3)
        )
        position = position.permute(0, 2, 3, 1)

        position_scores = torch.matmul(position_query, position)
        position_scores = torch.cat(
            (
                torch.zeros(
                    batch_size,
                    self.num_heads,
                    sequence_length,
                    1,
                    dtype=projection.dtype,
                    device=projection.device,
                ),
                position_scores,
            ),
            dim=3,
        )
        position_scores = position_scores.reshape(
            batch_size, self.num_heads, position_scores.size(3), sequence_length
        )
        position_scores = position_scores[:, :, 1:].reshape(
            batch_size, self.num_heads, sequence_length, position.size(3)
        )

        scores = torch.matmul(query, key) + position_scores[:, :, :, :sequence_length]
        expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scores = torch.softmax(scores.masked_fill(expanded_mask, float("-inf")), dim=3)
        # Define an all-masked row as zero and make source exclusion exact.
        scores = scores.masked_fill(expanded_mask, 0.0)
        if sequence_length > 7:
            query_padding_mask = torch.cat(
                (torch.zeros_like(key_padding_mask[:, :7]), key_padding_mask[:, :-7]),
                dim=1,
            )
            scores = scores.masked_fill(
                query_padding_mask.unsqueeze(1).unsqueeze(3), 0.0
            )

        return scores


class CompactRelPositionalEncoding(torch.nn.Module):
    """Precompute compact relative positional encodings for one export profile.

    This version is "compact" meaning it is able to
    encode the important information about the relative positions in a relatively small
    number of dimensions.
    This implementation works by projecting the interval [-infinity, infinity] to a
    finite interval using the torch.atan() function before computing the Fourier
    transform of that fixed interval. The torch.atan() function would compress the
    "long tails" too small, making it hard to distinguish between different magnitudes
    of large offsets. To mitigate this a logarithmic function is used to compress large
    offsets to a smaller range before applying torch.atan().
    """

    def __init__(self, embed_dim: int, max_length: int) -> None:
        """Initialize the compressed sinusoidal position table.

        Parameters
        ----------
        embed_dim : int
            The positional embedding dimension.
        max_length : int
            The maximum input length supported by the exported inference profile.
        """

        super().__init__()

        # if max_length == 4, the x would contain [-3, -2, -1, 0, 1, 2, 3]
        x = torch.arange(-max_length + 1, max_length, dtype=torch.float32)

        # Compression length is an arbitrary heuristic, if it is larger we have more
        # resolution for small time offsets but less resolution for large time
        # offsets.
        compress_len = torch.tensor(embed_dim**0.5, dtype=torch.float32)

        # Compressing x within the next line of code, similarly to uncompressed x, it
        # goes from -infinity to infinity as the sequence length goes from -infinity
        # to infinity, but it does so more slowly than sequence length for the large
        # absolute values of sequence length.
        # The formula is chosen so that d(x_compressed) / dx is equal to 1 around
        # x == 0, which is important.
        x = (
            compress_len
            * torch.sign(x)
            * (torch.log(torch.abs(x) + compress_len) - torch.log(compress_len))
        )

        # results between -pi and pi
        x = torch.atan(2.0 * torch.pi * x / embed_dim)

        freqs = torch.arange(1, embed_dim // 2 + 1, dtype=torch.float32)
        x = x.unsqueeze(1) * freqs

        pos_emb = torch.zeros(x.size(0), embed_dim, dtype=torch.float32)
        pos_emb[:, 0::2] = torch.cos(x)
        pos_emb[:, 1::2] = torch.sin(x)
        pos_emb[:, embed_dim - 1] = 1.0  # for bias.
        self.register_buffer("pos_emb", pos_emb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select relative positional embeddings for the length of ``x``.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
            Its sequence length determines the returned relative positional embeddings.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Relative positional embeddings with shape
            ``(1, 2 * seq_len - 1, embed_dim)``. The singleton batch dimension
            avoids repeating the same positional table for every utterance. The
            dtype and device follow this module's registered positional buffer;
            callers must place the module and input on compatible devices.
        """

        # (2 * seq_len - 1, embed_dim), i.e. (pos_len, embed_dim).
        return self.pos_emb[
            self.pos_emb.size(0) // 2 - x.size(1) + 1 : self.pos_emb.size(0) // 2
            + x.size(1)
        ].unsqueeze(0)
