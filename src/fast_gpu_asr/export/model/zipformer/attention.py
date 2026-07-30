#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""Offline Zipformer attention modules adapted for export-friendly inference."""

import torch


class SelfAttention(torch.nn.Module):
    """
    The simplest possible attention module. This one works with precomputed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, value_head_dim: int, device: torch.device
    ) -> None:
        """
        SelfAttention initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for input and output of this module.
        num_heads : int
            The number of attention heads.
        value_head_dim : int
            The dimension of the value per head.
        device : torch.device
            The device used to store the layer positional embeddings.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(
            embed_dim, num_heads * value_head_dim, device=device
        )
        self.out_proj = torch.nn.Linear(
            num_heads * value_head_dim, embed_dim, device=device
        )

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the SelfAttention module. Returns attention weighted
        input features.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        attn_weights : torch.Tensor[torch.float32]
            Weights with shape ``(batch_size, num_heads, seq_len, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Attention weighted output of shape ``(batch_size, seq_len, embed_dim)``.
            The tensor has the same shape as ``x``.
        """

        batch_size = x.size(0)
        num_heads = attn_weights.size(1)

        x = self.in_proj(x)  # (batch_size, seq_len, num_heads * value_head_dim)

        x = x.reshape(batch_size, x.size(1), num_heads, x.size(2) // num_heads).permute(
            0, 2, 1, 3
        )

        # (batch_size, num_heads, seq_len, seq_len) x
        # (batch_size, num_heads, seq_len, value_head_dim) ->
        # (batch_size, num_heads, seq_len, value_head_dim)
        x = torch.matmul(attn_weights, x)

        # (batch_size, num_heads, seq_len, value_head_dim) ->
        # (batch_size, seq_len, num_heads, value_head_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, x.size(1), num_heads * x.size(3))

        # The returned value has the same (N, T, C) shape as the input.
        x = self.out_proj(x)

        return x


class NonlinAttention(torch.nn.Module):
    """
    This is like the ConvolutionModule, but refactored so that we use multiplication by
    attention weights from RelPositionMultiheadAttentionWeights instead of convolution.
    The second nonlinearity after the attention mechanism is omitted.
    """

    def __init__(self, embed_dim: int, att_dim: int, device: torch.device) -> None:
        """
        NonlinAttention initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for the input and output.
        att_dim : int
            The attention output dimension of this module.
        device : torch.device
            The device used to store the positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, att_dim * 3, device=device)
        self.out_proj = torch.nn.Linear(att_dim, embed_dim, device=device)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the NonlinAttention module. Returns attention weighted
        input tensor.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        attn_weights : torch.Tensor[torch.float32]
            A tensor of shape ``(batch_size, seq_len, seq_len)`` that corresponds
            to a single attention head with (seq_len, seq_len) being interpreted
            as (tgt_seq_len, src_seq_len). Expected attn_weights.sum(dim=2) == 1.0.
            Note: the first dimension here corresponds to a batch size.

        Returns
        -------
        torch.Tensor[torch.float32]
            Attention weighted output of shape ``(batch_size, seq_len, embed_dim)``.
            The tensor has the same shape as ``x``.
        """

        x = self.in_proj(x)

        s, x, y = x.chunk(3, dim=2)
        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, att_dim) ->
        # (batch_size, seq_len, att_dim)
        x = torch.matmul(attn_weights, x * torch.tanh(s)) * y

        x = self.out_proj(x)

        return x


class RelPositionMultiheadAttentionWeights(torch.nn.Module):
    """
    Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example,
    the SelfAttention module which allows you to compute conventional self-attention.

    This is heavily modified from:
    "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        device: torch.device,
    ) -> None:
        """
        RelPositionMultiheadAttentionWeights initialization.

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
            The dimension of the projected positional encoding per head.
        device : torch.device
            The device used to store the layer positional embeddings. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim

        in_proj_dim = (2 * query_head_dim + pos_head_dim) * num_heads
        self.in_proj = torch.nn.Linear(embed_dim, in_proj_dim, device=device)

        # Linear transformation for positional encoding.
        self.linear_pos = torch.nn.Linear(
            pos_dim, num_heads * pos_head_dim, bias=False, device=device
        )

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Does a forward pass of the RelPositionMultiheadAttentionWeights module.
        Returns attention weights.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
        pos_emb : torch.Tensor[torch.float32]
            Positional embeddings with shape
            ``(batch_size, 2 * seq_len - 1, pos_dim)``.
        key_padding_mask : torch.Tensor[torch.bool]
            Padding mask with shape ``(batch_size, seq_len)``. True positions
            are excluded as attention sources.

        Returns
        -------
        torch.Tensor[torch.float32]
            Attention weights with shape
            ``(batch_size, num_heads, seq_len, seq_len)``.
        """

        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.in_proj(x)

        query_dim = self.query_head_dim * self.num_heads

        # Self-attention.
        q = x[:, :, :query_dim]
        k = x[:, :, query_dim : 2 * query_dim]
        # p is the position-encoding query.
        p = x[:, :, 2 * query_dim :]

        q = q.reshape(batch_size, seq_len, self.num_heads, self.query_head_dim)
        p = p.reshape(batch_size, seq_len, self.num_heads, self.pos_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.query_head_dim)

        q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, query_head_dim)
        p = p.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, pos_head_dim)
        k = k.permute(0, 2, 3, 1)  # (batch_size, num_heads, key_head_dim, seq_len)

        attn_scores = torch.matmul(q, k)  # (batch_size, num_heads, seq_len, seq_len)

        pos_len = pos_emb.size(1)  # 2 * seq_len - 1
        # (batch_size, pos_len, num_heads * pos_head_dim)
        pos_emb = self.linear_pos(pos_emb)
        pos_emb = pos_emb.reshape(
            pos_emb.size(0), pos_len, self.num_heads, self.pos_head_dim
        ).permute(0, 2, 3, 1)  # (batch_size, num_heads, pos_head_dim, pos_len)

        # (batch_size, num_heads, seq_len, pos_head_dim) x
        # (batch_size, num_heads, pos_head_dim, pos_len) ->
        # (batch_size, num_heads, seq_len, pos_len)
        # where pos_len represents relative position.
        pos_scores = torch.matmul(p, pos_emb)

        # Now we need to perform the relative shift of the pos_scores, to do that we
        # need to add a column of zeros to the left side of the last dimension and
        # perform the relative shift.
        pos_scores_pad = torch.zeros(
            pos_scores.size(0),
            pos_scores.size(1),
            pos_scores.size(2),
            1,
            dtype=pos_scores.dtype,
            device=pos_scores.device,
        )
        # (batch_size, num_heads, seq_len, pos_len + 1)
        pos_scores = torch.cat((pos_scores_pad, pos_scores), dim=3)
        pos_scores = pos_scores.reshape(
            batch_size, self.num_heads, pos_len + 1, seq_len
        )  # (batch_size, num_heads, pos_len + 1, seq_len)
        # Now drop the extra row that had been added over padding and reshape.
        pos_scores = pos_scores[:, :, 1:].reshape(
            batch_size, self.num_heads, seq_len, pos_len
        )  # (batch_size, num_heads, seq_len, pos_len)

        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores + pos_scores[:, :, :, : attn_scores.size(3)]
        attn_scores = attn_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), -1000.0
        )
        attn_weights = torch.softmax(attn_scores, dim=3)

        return attn_weights


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module. This version is "compact" meaning it is able to
    encode the important information about the relative positions in a relatively small
    number of dimensions.
    This implementation works by projecting the interval [-infinity, infinity] to a
    finite interval using the torch.atan() function before computing the Fourier
    transform of that fixed interval. The torch.atan() function would compress the
    "long tails" too small, making it hard to distinguish between different magnitudes
    of large offsets. To mitigate this a logarithmic function is used to compress large
    offsets to a smaller range before applying torch.atan().
    """

    def __init__(self, embed_dim: int, max_length: int, device: torch.device) -> None:
        """
        CompactRelPositionalEncoding initialization.

        Parameters
        ----------
        embed_dim : int
            The positional embedding dimension.
        max_length : int
            The maximum length of the input that this module will be able to handle
            without regenerating the positional embeddings. Longer inputs cause the
            positional table to be regenerated.
        device : torch.device
            The device used to store the layer positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                "Embedding dimension for CompactRelPositionalEncoding "
                f"should be an even number, but got {embed_dim}.",
            )

        self.embed_dim = embed_dim
        self.pos_emb = self.create_pos_emb(max_length, device)

    def create_pos_emb(self, max_length: int, device: torch.device) -> torch.Tensor:
        """
        Create relative positional embeddings for a maximum sequence length.

        Parameters
        ----------
        max_length : int
            The maximum length of the input that can be handled by this layer.
            Increasing it allows longer inputs but consumes more memory.
        device : torch.device
            The device used to store the positional embeddings.
            Should be either torch.device("cpu") or torch.device("cuda").

        Returns
        -------
        torch.Tensor[torch.float32]
            Relative positional embeddings with shape
            ``(2 * max_length - 1, embed_dim)``.
        """

        # if max_length == 4, the x would contain [-3, -2, -1, 0, 1, 2, 3]
        x = torch.arange(
            -max_length + 1, max_length, dtype=torch.float32, device=device
        )

        # Compression length is an arbitrary heuristic, if it is larger we have more
        # resolution for small time offsets but less resolution for large time
        # offsets.
        compress_len = torch.tensor(
            self.embed_dim**0.5, dtype=torch.float32, device=device
        )

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
        x = torch.atan(2.0 * torch.pi * x / self.embed_dim)

        freqs = torch.arange(
            1, self.embed_dim // 2 + 1, dtype=torch.float32, device=device
        )
        x = x.unsqueeze(1) * freqs

        pos_emb = torch.zeros(
            x.size(0), self.embed_dim, dtype=torch.float32, device=device
        )
        pos_emb[:, 0::2] = torch.cos(x)
        pos_emb[:, 1::2] = torch.sin(x)
        pos_emb[:, self.embed_dim - 1] = 1.0  # for bias.

        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the CompactRelPositionalEncoding module.
        Returns relative positional embeddings for the temporal dimension of ``x``.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input with shape ``(batch_size, seq_len, embed_dim)``.
            Its sequence length determines the returned relative positional embeddings.

        Returns
        -------
        torch.Tensor[torch.float32]
            Relative positional embeddings with shape
            ``(batch_size, 2 * seq_len - 1, embed_dim)``.
        """

        if self.pos_emb.size(0) < 2 * x.size(1) - 1:
            self.pos_emb = self.create_pos_emb(x.size(1), x.device)

        # (batch_size, 2 * seq_len - 1, embed_dim), i.e.
        # (batch_size, pos_len, embed_dim).
        pos_emb = self.pos_emb[
            self.pos_emb.size(0) // 2 - x.size(1) + 1 : self.pos_emb.size(0) // 2
            + x.size(1)
        ]

        return pos_emb.unsqueeze(0).expand(x.size(0), pos_emb.size(0), pos_emb.size(1))
