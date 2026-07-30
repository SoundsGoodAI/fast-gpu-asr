#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Relative-position attention modules used by NVIDIA Parakeet TDT models."""

import torch


class RelPositionMultiHeadAttention(torch.nn.Module):
    """
    Parakeet relative-position multi-head self-attention for offline inference.
    """

    def __init__(self, n_head: int, n_feat: int, device: torch.device) -> None:
        """
        RelPositionMultiHeadAttention initialization.

        Parameters
        ----------
        n_head : int
            The number of attention heads.
        n_feat : int
            The input and output hidden dimension of the attention module.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if n_feat % n_head != 0:
            raise ValueError(f"n_feat={n_feat} must be divisible by n_head={n_head}.")

        self.h = n_head
        self.d_k = n_feat // n_head
        self.s_d_k = self.d_k**0.5

        self.linear_q = torch.nn.Linear(n_feat, n_feat, bias=False, device=device)
        self.linear_k = torch.nn.Linear(n_feat, n_feat, bias=False, device=device)
        self.linear_v = torch.nn.Linear(n_feat, n_feat, bias=False, device=device)

        self.linear_out = torch.nn.Linear(n_feat, n_feat, bias=False, device=device)
        self.linear_pos = torch.nn.Linear(n_feat, n_feat, bias=False, device=device)

        self.pos_bias_u = torch.nn.Parameter(
            torch.zeros(n_head, self.d_k, device=device)
        )
        self.pos_bias_v = torch.nn.Parameter(
            torch.zeros(n_head, self.d_k, device=device)
        )

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Does a forward pass of the RelPositionMultiHeadAttention module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, n_feat)``.
        pos_emb : torch.Tensor[torch.float32]
            The relative positional embeddings of shape
            ``(1, 2 * num_frames - 1, n_feat)``.
        key_padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, num_frames)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output features of shape ``(batch_size, num_frames, n_feat)``.
        """

        batch_size, num_frames, _ = x.size()

        q = self.linear_q(x).reshape(batch_size, num_frames, self.h, self.d_k)
        k = self.linear_k(x).reshape(batch_size, num_frames, self.h, self.d_k)
        v = self.linear_v(x).reshape(batch_size, num_frames, self.h, self.d_k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        p = self.linear_pos(pos_emb).reshape(
            pos_emb.size(0), pos_emb.size(1), self.h, self.d_k
        )
        p = p.permute(0, 2, 3, 1)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        attn_scores = torch.matmul(q_with_bias_u, k)
        pos_scores = torch.matmul(q_with_bias_v, p)
        pos_scores = torch.nn.functional.pad(pos_scores, pad=(1, 0))
        pos_scores = pos_scores.reshape(
            batch_size, pos_scores.size(1), pos_emb.size(1) + 1, num_frames
        )
        pos_scores = pos_scores[:, :, 1:].reshape(
            batch_size, pos_scores.size(1), num_frames, pos_emb.size(1)
        )
        pos_scores = pos_scores[:, :, :, : attn_scores.size(3)]
        attn_scores = (attn_scores + pos_scores) / self.s_d_k
        attn_scores = attn_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), -1000.0
        )
        attn_weights = torch.softmax(attn_scores, dim=3)

        x = torch.matmul(attn_weights, v)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, num_frames, self.h * self.d_k)
        x = self.linear_out(x)

        return x


class RelPositionalEncoding(torch.nn.Module):
    """
    Transformer-XL relative sinusoidal positional encoding used by Parakeet.
    """

    def __init__(self, model_dim: int, max_len: int, device: torch.device) -> None:
        """
        RelPositionalEncoding initialization.

        Parameters
        ----------
        model_dim : int
            The encoder hidden dimension.
        max_len : int
            The initial maximum post-subsampling sequence length.
        device : torch.device
            The device used to store the positional buffer. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.model_dim = model_dim
        self.max_len = max_len
        self.div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, dtype=torch.float32, device=device)
            * -(
                torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device))
                / self.model_dim
            ),
        )
        self.pos_emb = self.create_pos_emb(max_len, device)

    def create_pos_emb(self, max_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a relative sinusoidal positional table.

        Parameters
        ----------
        max_len : int
            The maximum query length supported by the generated table.
        device : torch.device
            The device used to store the positional table. Should be
            either torch.device("cpu") or torch.device("cuda").

        Returns
        -------
        torch.Tensor[torch.float32]
            The positional table of shape ``(2 * max_len - 1, model_dim)``.
        """

        positions = torch.arange(
            max_len - 1, -max_len, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        pos_emb = torch.zeros(
            2 * max_len - 1, self.model_dim, dtype=torch.float32, device=device
        )
        pos_emb[:, 0::2] = torch.sin(positions * self.div_term)
        pos_emb[:, 1::2] = torch.cos(positions * self.div_term)

        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the RelPositionalEncoding module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, model_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            The relative positional embeddings of shape
            ``(1, 2 * num_frames - 1, model_dim)``.
        """

        input_len = x.size(1)
        if 2 * input_len - 1 > self.pos_emb.size(0):
            self.pos_emb = self.create_pos_emb(input_len, x.device)

        center_pos = self.pos_emb.size(0) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pos_emb[start_pos:end_pos].unsqueeze(0)

        return pos_emb
