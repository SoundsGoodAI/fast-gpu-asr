#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""Offline Zipformer subsampling and normalization modules."""

import torch

from .activation import SwooshL, SwooshR


class Conv2dSubsampling(torch.nn.Module):
    """
    Convolutional frontend that subsamples log-mel features by two in time and four
    in frequency before the Zipformer encoder.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer1_channels: int,
        layer2_channels: int,
        layer3_channels: int,
        device: torch.device,
    ) -> None:
        """
        Conv2dSubsampling initialization.

        Parameters
        ----------
        input_dim : int
            The number of input channels. Corresponds to the
            number of features in the input feature tensor.
        output_dim : int
            The number of output channels.
        layer1_channels : int
            The number of output channels in the first Conv2d layer.
        layer2_channels : int
            The number of output channels in the second Conv2d layer.
        layer3_channels : int
            The number of output channels in the third Conv2d layer.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if input_dim < 7:
            raise ValueError(
                "The input feature dimension of the Conv2dSubsampling layer cannot be "
                "less than seven; otherwise, frequency subsampling produces an empty "
                f"output. Expected input_dim to be at least 7 but got {input_dim}.",
            )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, layer1_channels, 3, padding=(0, 1), device=device),
            SwooshR(),
            torch.nn.Conv2d(
                layer1_channels, layer2_channels, 3, stride=2, device=device
            ),
            SwooshR(),
            torch.nn.Conv2d(
                layer2_channels, layer3_channels, 3, stride=(1, 2), device=device
            ),
            SwooshR(),
        )

        self.convnext = ConvNeXt(layer3_channels, device=device)

        out_width = (((input_dim - 1) // 2) - 1) // 2
        self.out = torch.nn.Linear(
            out_width * layer3_channels, output_dim, device=device
        )
        self.out_norm = BiasNorm(output_dim, device=device)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the Conv2dSubsampling module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Padded input features with shape ``(batch_size, num_frames, input_dim)``.
            Frames beyond ``x_lens`` must be filled with ``ZERO_LOG``.
        x_lens : torch.Tensor[torch.int32]
            Valid input lengths with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Subsampled features and valid lengths. The feature tensor has shape
            ``(batch_size, subsampled_num_frames, output_dim)``.
        """
        # (batch_size, seq_len, input_dim) -> (batch_size, 1, seq_len, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.convnext(x)

        # (batch_size, output_dim, seq_len', ((input_dim - 1) // 2 - 1) // 2).
        batch_size, output_dim, seq_len, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, output_dim * f)
        # (batch_size, seq_len', output_dim * layer3_channels))
        x = self.out(x)
        # (batch_size, seq_len', output_dim)
        x = self.out_norm(x)

        return x, (x_lens - 7) // 2


class ConvNeXt(torch.nn.Module):
    """
    Simplified ConvNeXt block based on https://arxiv.org/abs/2206.14747.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        ConvNeXt initialization.

        Parameters
        ----------
        num_channels : int
            The number of input and output channels for ConvNeXt module.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.padding = 3

        self.depthwise_conv = torch.nn.Conv2d(
            num_channels,
            num_channels,
            7,
            groups=num_channels,
            padding=self.padding,
            device=device,
        )

        self.activation = SwooshL()
        self.pointwise_conv1 = torch.nn.Conv2d(
            num_channels, num_channels * 3, 1, device=device
        )
        self.pointwise_conv2 = torch.nn.Conv2d(
            num_channels * 3, num_channels, 1, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the ConvNeXt module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input with shape
            ``(batch_size, num_channels, num_input_frames, num_freqs)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output with the same shape as ``x``.
        """

        bypass = x

        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x

        return x


class BiasNorm(torch.nn.Module):
    """
    This is a simpler replacement for LayerNorm.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        BiasNorm initialization.

        Parameters
        ----------
        num_channels : int
            The number of input channels.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.scale = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32, device=device),
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_channels, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the BiasNorm module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            A tensor with shape ``(batch_size, seq_len, num_channels)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            A normalized tensor with the same shape as ``x``.
        """

        return (
            x
            * self.scale
            / torch.mean((x - self.bias) ** 2, dim=2, keepdim=True) ** 0.5
        )
