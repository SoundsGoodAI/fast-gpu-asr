#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Offline Zipformer subsampling and normalization modules."""

import torch

from .activation import SwooshL, SwooshR


class Conv2dSubsampling(torch.nn.Module):
    """Subsample log-mel features by two in time and four in frequency."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer1_channels: int,
        layer2_channels: int,
        layer3_channels: int,
        batch_partitions: int,
    ) -> None:
        """Initialize convolutional subsampling and output normalization.

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
        batch_partitions : int
            Number of batch partitions used by the convolutional frontend.

        """

        super().__init__()

        self.batch_partitions = batch_partitions
        self.padding = 3

        self.conv_activation = SwooshR()
        self.conv1 = torch.nn.Conv2d(1, layer1_channels, 3, padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(layer1_channels, layer2_channels, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(layer2_channels, layer3_channels, 3, stride=(1, 2))
        self.depthwise_conv = torch.nn.Conv2d(
            layer3_channels,
            layer3_channels,
            7,
            groups=layer3_channels,
            padding=self.padding,
        )

        self.convnext_activation = SwooshL()
        self.pointwise_conv1 = torch.nn.Linear(layer3_channels, layer3_channels * 3)
        self.pointwise_conv2 = torch.nn.Linear(layer3_channels * 3, layer3_channels)

        out_width = (((input_dim - 1) // 2) - 1) // 2
        self.out = torch.nn.Linear(out_width * layer3_channels, output_dim)
        self.out_norm = BiasNorm(output_dim)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample padded log-mel features and update valid lengths.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Padded input features with shape ``(batch_size, num_frames, input_dim)``.
            The top-level encoder casts ``torch.float32`` log-mel features to its
            configured dtype before invoking this module. Frames beyond ``x_lens``
            must be filled with ``ZERO_LOG``.
        x_lens : torch.Tensor[torch.int32]
            Valid input lengths with shape ``(batch_size,)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
            torch.Tensor[torch.int32],
        ]
            Subsampled features with shape
            ``(batch_size, subsampled_num_frames, output_dim)`` and
            ``torch.int32`` valid lengths. Features retain the configured encoder
            dtype even when the first convolution uses ``torch.float16`` for
            ``torch.bfloat16`` export.
        """
        x = self.conv_activation(
            self.conv1(x.unsqueeze(1).to(self.conv1.weight.dtype))
        ).to(self.conv2.weight.dtype)
        x = self.conv_activation(self.conv2(x))
        output_lens = torch.clamp((x_lens - 7) // 2, min=0, max=x.size(2) - 2)
        valid_frames = torch.arange(
            x.size(2) - 2,
            dtype=output_lens.dtype,
            device=output_lens.device,
        ).unsqueeze(0) < output_lens.unsqueeze(1)

        if self.batch_partitions > 1:
            bypass = torch.empty(
                (
                    x.size(0),
                    self.conv3.out_channels,
                    x.size(2) - 2,
                    (x.size(3) - 1) // 2,
                ),
                dtype=x.dtype,
                device=x.device,
            )
            outputs = torch.empty_like(bypass)
            for partition in range(self.batch_partitions):
                start = partition * x.size(0) // self.batch_partitions
                end = (partition + 1) * x.size(0) // self.batch_partitions
                partition_output = self.conv_activation(self.conv3(x[start:end]))
                partition_output = partition_output * valid_frames[start:end].unsqueeze(
                    1
                ).unsqueeze(3)

                partition_bypass = partition_output
                partition_output = self.depthwise_conv(partition_output)

                bypass[start:end] = partition_bypass
                outputs[start:end] = partition_output

            x = outputs
        else:
            x = self.conv_activation(self.conv3(x))
            x = x * valid_frames.unsqueeze(1).unsqueeze(3)
            bypass = x
            x = self.depthwise_conv(x)

        x = self.pointwise_conv1(x.permute(0, 2, 3, 1))
        x = self.convnext_activation(x)
        x = self.pointwise_conv2(x).permute(0, 3, 1, 2)
        x = bypass + x

        batch_size, output_dim, seq_len, frequency_dim = x.size()
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, output_dim * frequency_dim
        )
        x = self.out_norm(self.out(x))

        return x, output_lens


class BiasNorm(torch.nn.Module):
    """Apply Zipformer's channel-biased root-mean-square normalization."""

    def __init__(self, num_channels: int) -> None:
        """Initialize checkpointed bias and scale buffers.

        Parameters
        ----------
        num_channels : int
            The number of input channels.
        """

        super().__init__()

        self.register_buffer("scale", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(num_channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one channel-last tensor.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            A tensor with shape ``(batch_size, seq_len, num_channels)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            A normalized tensor with the same shape and dtype as ``x``.
        """

        output_dtype = x.dtype
        x = x.to(torch.float32)
        centered = x - self.bias.to(torch.float32)
        rms = torch.sqrt(torch.mean(centered * centered, dim=2, keepdim=True))
        rms = torch.clamp(rms, min=torch.finfo(torch.float32).tiny)
        x = (x * self.scale.to(torch.float32) / rms).to(output_dtype)

        return x
