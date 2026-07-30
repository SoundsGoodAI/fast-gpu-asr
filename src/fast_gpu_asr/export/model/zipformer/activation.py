#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""ONNX-friendly activation functions used by Zipformer inference modules."""

import torch


class SwooshL(torch.nn.Module):
    """ONNX-friendly Swoosh-L activation used by Zipformer feedforward blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Does a forward pass and returns Swoosh-L activation.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input of arbitrary shape ``(*)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Swoosh-L output with the same shape as ``x``.
        """
        logaddexp = torch.clamp(x - 4.0, min=0.0) + torch.log1p(
            torch.exp(-torch.abs(x - 4.0))
        )

        return logaddexp - 0.08 * x - 0.035


class SwooshR(torch.nn.Module):
    """ONNX-friendly Swoosh-R activation used by convolutional Zipformer blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Does a forward pass and returns Swoosh-R activation.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input of arbitrary shape ``(*)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Swoosh-R output with the same shape as ``x``.
        """
        logaddexp = torch.clamp(x - 1.0, min=0.0) + torch.log1p(
            torch.exp(-torch.abs(x - 1.0))
        )

        return logaddexp - 0.08 * x - 0.313261687
