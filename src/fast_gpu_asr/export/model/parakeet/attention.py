#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Modified from NeMo for batched TensorRT export; see NOTICE and LICENSE.
"""Relative-position attention modules used by NVIDIA Parakeet TDT models."""

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
)


class RelPositionMultiHeadAttention(torch.nn.Module):
    """Parakeet relative-position multi-head attention for offline inference."""

    def __init__(self, n_head: int, n_feat: int) -> None:
        """Initialize attention projections and per-head positional biases.

        Parameters
        ----------
        n_head : int
            Number of attention heads.
        n_feat : int
            Input and output hidden dimension. It must be divisible by ``n_head``.
        """

        super().__init__()

        self.h = n_head
        self.d_k = n_feat // n_head
        self.s_d_k = self.d_k**0.5

        self.linear_qkv = torch.nn.Linear(n_feat, 3 * n_feat, bias=False)

        self.linear_out = torch.nn.Linear(n_feat, n_feat, bias=False)
        self.linear_pos = torch.nn.Linear(n_feat, n_feat, bias=False)

        self.pos_bias_u = torch.nn.Parameter(torch.zeros(n_head, self.d_k))
        self.pos_bias_v = torch.nn.Parameter(torch.zeros(n_head, self.d_k))

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, output_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Apply full-context relative-position self-attention.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, n_feat)``.
        pos_emb : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Relative positional embeddings with shape
            ``(1, 2 * num_frames - 1, n_feat)``.
        output_lengths : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output features with shape ``(batch_size, num_frames, n_feat)`` and
            the same dtype as ``x``. Eager execution converts the content and
            position scores to ``torch.float32`` before combining and normalizing
            them. ONNX export replaces relative scoring, masking, softmax, and
            value aggregation with one native TensorRT FlashAttention plugin node.
        """

        batch_size, num_frames, _ = x.size()

        qkv = self.linear_qkv(x)
        p = self.linear_pos(pos_emb.to(x.dtype))

        if torch.onnx.is_in_onnx_export():
            x = torch.onnx.ops.symbolic(
                PARAKEET_FLASH_ATTENTION_PLUGIN_NAME,
                (qkv, p, self.pos_bias_u, self.pos_bias_v, output_lengths),
                {
                    "scale": 1.0 / self.s_d_k,
                    "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE,
                },
                dtype=x.dtype,
                shape=x.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            q, k, v = qkv.chunk(3, dim=2)
            q = q.reshape(batch_size, num_frames, self.h, self.d_k).permute(0, 2, 1, 3)
            k = k.reshape(batch_size, num_frames, self.h, self.d_k).permute(0, 2, 1, 3)
            v = v.reshape(batch_size, num_frames, self.h, self.d_k)
            p = p.reshape(pos_emb.size(0), pos_emb.size(1), self.h, self.d_k).permute(
                0, 2, 1, 3
            )
            content_query = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
            position_query = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
            content_scores = torch.matmul(content_query, k.permute(0, 1, 3, 2)).to(
                torch.float32
            )
            position_scores = torch.matmul(position_query, p.permute(0, 1, 3, 2)).to(
                torch.float32
            )
            position_scores = torch.nn.functional.pad(position_scores, (1, 0))
            position_scores = position_scores.reshape(
                batch_size, self.h, position_scores.size(3), num_frames
            )
            position_scores = position_scores[:, :, 1:].reshape(
                batch_size, self.h, num_frames, p.size(2)
            )
            scores = (
                content_scores + position_scores[:, :, :, :num_frames]
            ) / self.s_d_k
            key_padding_mask = torch.arange(
                num_frames, dtype=output_lengths.dtype, device=output_lengths.device
            ).unsqueeze(0) >= output_lengths.unsqueeze(1)
            weights = torch.softmax(
                scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                ),
                dim=3,
            )
            weights = weights.masked_fill(
                (output_lengths <= 0).reshape(batch_size, 1, 1, 1), 0.0
            ).to(v.dtype)
            x = (
                torch.matmul(weights, v.permute(0, 2, 1, 3))
                .permute(0, 2, 1, 3)
                .reshape(batch_size, num_frames, self.h * self.d_k)
            )

        x = self.linear_out(x)

        return x


class RelPositionalEncoding(torch.nn.Module):
    """Precompute Transformer-XL relative sinusoidal positions for Parakeet."""

    def __init__(self, model_dim: int, max_len: int) -> None:
        """Initialize the relative sinusoidal position table.

        Parameters
        ----------
        model_dim : int
            Encoder hidden dimension.
        max_len : int
            Maximum post-subsampling sequence length supported by the export profile.

        Notes
        -----
        The positional table is a non-persistent buffer: module dtype and device
        conversions include it, while source checkpoints do not need to contain it.
        """

        super().__init__()

        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / model_dim),
        )
        positions = torch.arange(
            max_len - 1, -max_len, -1, dtype=torch.float32
        ).unsqueeze(1)
        pos_emb = torch.zeros(2 * max_len - 1, model_dim, dtype=torch.float32)
        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pos_emb", pos_emb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select relative positions for the current sequence length.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, model_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            The relative positional embeddings of shape
            ``(1, 2 * num_frames - 1, model_dim)``. The dtype and device match
            the precomputed positional buffer.
        """

        center_pos = self.pos_emb.size(0) // 2 + 1
        start_pos = center_pos - x.size(1)
        end_pos = center_pos + x.size(1) - 1
        pos_emb = self.pos_emb[start_pos:end_pos].unsqueeze(0)

        return pos_emb
