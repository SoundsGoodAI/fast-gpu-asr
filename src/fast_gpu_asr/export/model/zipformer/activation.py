#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright 2022-2023 Xiaomi Corp. (Daniel Povey)
# Modified from Icefall for TensorRT export; see NOTICE and LICENSE.
"""Swoosh activation modules used by the offline Zipformer encoder."""

import torch


class SwooshL(torch.nn.Module):
    """Apply the Swoosh-L activation used by Zipformer feedforward blocks.

    Swoosh-L is defined as ``softplus(x - 4.0) - 0.08 * x - 0.035``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Swoosh-L elementwise.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Floating-point input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        return torch.nn.functional.softplus(x - 4.0) - 0.08 * x - 0.035


class SwooshR(torch.nn.Module):
    """Apply the Swoosh-R activation used by Zipformer convolution blocks.

    Swoosh-R is defined as ``softplus(x - 1.0) - 0.08 * x - 0.313261687``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Swoosh-R elementwise.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Floating-point input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        return torch.nn.functional.softplus(x - 1.0) - 0.08 * x - 0.313261687
