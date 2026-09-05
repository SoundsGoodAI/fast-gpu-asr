#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Modified from NeMo for batched TensorRT export; see NOTICE and LICENSE.
"""Fast Conformer encoder used by NVIDIA Parakeet TDT models."""

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
    TENSORRT_PLUGIN_NAMESPACE,
)
from .attention import RelPositionalEncoding, RelPositionMultiHeadAttention
from .features import FeatureExtractor


class ParakeetTDTEncoder(torch.nn.Module):
    """Audio-to-encoder module for Parakeet TDT export."""

    def __init__(
        self,
        samp_freq: int,
        frame_shift_ms: int,
        frame_length_ms: int,
        feature_dim: int,
        preemph: float,
        low_freq: int,
        high_freq: int,
        n_layers: int,
        model_dim: int,
        subsampling_conv_channels: int,
        feed_forward_expansion_factor: int,
        n_heads: int,
        pos_emb_max_len: int,
        conv_kernel_size: int,
        subsampling_batch_partitions: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the waveform frontend and Fast Conformer encoder.

        Parameters
        ----------
        samp_freq : int
            The model sampling rate in Hertz.
        frame_shift_ms : int
            The hop size between neighboring frames in milliseconds.
        frame_length_ms : int
            The analysis window size in milliseconds.
        feature_dim : int
            The number of mel filterbank channels.
        preemph : float
            The waveform pre-emphasis coefficient.
        low_freq : int
            The lower mel filterbank frequency in Hertz.
        high_freq : int
            The upper mel filterbank frequency in Hertz.
        n_layers : int
            The number of Conformer layers.
        model_dim : int
            The encoder hidden dimension.
        subsampling_conv_channels : int
            The intermediate convolution channels in the subsampling module.
        feed_forward_expansion_factor : int
            The feed-forward expansion factor.
        n_heads : int
            The number of attention heads.
        pos_emb_max_len : int
            The maximum post-subsampling sequence length for relative position buffers.
        conv_kernel_size : int
            The depthwise convolution kernel size inside each Conformer layer.
        subsampling_batch_partitions : int
            Number of batch partitions used through the third subsampling convolution
            to keep TensorRT tensors within the CASK ``int32`` element limit.
        dtype : torch.dtype
            Floating-point dtype used by convolutional subsampling and the
            Conformer layers. Supported values are ``torch.float32``,
            ``torch.float16``, and ``torch.bfloat16``. Feature extraction remains
            ``torch.float32``.
        """

        super().__init__()

        self.feature_extractor = FeatureExtractor(
            samp_freq,
            frame_shift_ms,
            frame_length_ms,
            feature_dim,
            preemph,
            low_freq,
            high_freq,
        )
        self.encoder = FastConformer(
            feature_dim,
            n_layers,
            model_dim,
            subsampling_conv_channels,
            feed_forward_expansion_factor,
            n_heads,
            pos_emb_max_len,
            conv_kernel_size,
            subsampling_batch_partitions,
        )

        for module in (self.encoder.pre_encode, *self.encoder.layers):
            module.to(dtype=dtype)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert padded waveforms into encoder embeddings.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded waveforms of shape ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int64]
            Valid sample counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
            torch.Tensor[torch.int32],
        ]
            Encoder output with the configured floating-point dtype and shape
            ``(batch_size, subsampled_num_frames, model_dim)`` and
            ``torch.int32`` valid lengths.
        """

        features, feature_lengths = self.feature_extractor(audio, audio_lengths)
        features = features.to(self.encoder.pre_encode.conv1.weight.dtype)
        encoder_out, encoder_out_lengths = self.encoder(features, feature_lengths)

        return encoder_out, encoder_out_lengths


class FastConformer(torch.nn.Module):
    """Fast Conformer encoder matching the Parakeet TDT encoder layout."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        model_dim: int,
        subsampling_conv_channels: int,
        feed_forward_expansion_factor: int,
        n_heads: int,
        pos_emb_max_len: int,
        conv_kernel_size: int,
        subsampling_batch_partitions: int,
    ) -> None:
        """Initialize subsampling, positional encoding, and Conformer layers.

        Parameters
        ----------
        input_dim : int
            The number of input feature channels.
        n_layers : int
            The number of Conformer layers.
        model_dim : int
            The encoder hidden dimension.
        subsampling_conv_channels : int
            The intermediate convolution channels in the subsampling module.
        feed_forward_expansion_factor : int
            The feed-forward expansion factor.
        n_heads : int
            The number of attention heads.
        pos_emb_max_len : int
            The maximum post-subsampling sequence length for relative position buffers.
        conv_kernel_size : int
            The depthwise convolution kernel size inside each Conformer layer.
        subsampling_batch_partitions : int
            Number of batch partitions used through the third subsampling
            convolution.
        """

        super().__init__()

        self.pre_encode = ConvSubsampling(
            input_dim,
            model_dim,
            subsampling_conv_channels,
            subsampling_batch_partitions,
        )
        self.pos_enc = RelPositionalEncoding(model_dim, pos_emb_max_len)
        self.layers = torch.nn.ModuleList(
            ConformerLayer(
                model_dim,
                model_dim * feed_forward_expansion_factor,
                n_heads,
                conv_kernel_size,
            )
            for _ in range(n_layers)
        )

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of padded log-mel feature sequences.

        Parameters
        ----------
        features : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, input_dim)``.
        feature_lengths : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
            torch.Tensor[torch.int32],
        ]
            Encoder output with the same floating-point dtype as ``features`` and shape
            ``(batch_size, subsampled_num_frames, model_dim)`` and
            ``torch.int32`` valid lengths.
        """

        x, output_lengths = self.pre_encode(features, feature_lengths)
        pos_emb = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, pos_emb, output_lengths)

        return x, output_lengths


class ConformerLayer(torch.nn.Module):
    """Single Conformer encoder layer."""

    def __init__(
        self, model_dim: int, feed_forward_dim: int, n_heads: int, conv_kernel_size: int
    ) -> None:
        """Initialize one Macaron Conformer layer.

        Parameters
        ----------
        model_dim : int
            The input and output hidden dimension.
        feed_forward_dim : int
            The intermediate feed-forward dimension.
        n_heads : int
            The number of attention heads.
        conv_kernel_size : int
            The kernel size of the depthwise convolution module.
        """

        super().__init__()

        self.norm_feed_forward1 = torch.nn.LayerNorm(model_dim)
        self.feed_forward1 = ConformerFeedForward(model_dim, feed_forward_dim)

        self.norm_conv = torch.nn.LayerNorm(model_dim)
        self.conv = ConformerConvolution(model_dim, conv_kernel_size)

        self.norm_self_att = torch.nn.LayerNorm(model_dim)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, model_dim)

        self.norm_feed_forward2 = torch.nn.LayerNorm(model_dim)
        self.feed_forward2 = ConformerFeedForward(model_dim, feed_forward_dim)

        self.norm_out = torch.nn.LayerNorm(model_dim)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, output_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Apply the two feed-forward, attention, and convolution branches.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, model_dim)``.
        pos_emb : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            The relative positional embeddings of shape
            ``(1, 2 * num_frames - 1, model_dim)``.
        output_lengths : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output features with the same dtype as ``x`` and shape
            ``(batch_size, num_frames, model_dim)``.
        """

        # Checkpoint loading folds the Macaron 0.5 scale into both FFN outputs.
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, output_lengths)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, output_lengths)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        x = residual + x

        return self.norm_out(x)


class ConvSubsampling(torch.nn.Module):
    """Parakeet ``dw_striding`` 2D convolutional subsampling."""

    def __init__(
        self, feat_dim: int, feat_out: int, conv_channels: int, batch_partitions: int
    ) -> None:
        """Initialize three-stage depthwise convolutional subsampling.

        Parameters
        ----------
        feat_dim : int
            The number of input mel bins.
        feat_out : int
            The output feature dimension after the projection layer.
        conv_channels : int
            The intermediate convolution channel count.
        batch_partitions : int
            Number of batch partitions used through the third convolution.
        """

        super().__init__()

        self.batch_partitions = batch_partitions

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, conv_channels, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(
            conv_channels,
            conv_channels,
            3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )
        self.conv3 = torch.nn.Conv2d(
            conv_channels,
            conv_channels,
            3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )

        self.pointwise_conv1 = torch.nn.Conv2d(conv_channels, conv_channels, 1)
        self.pointwise_conv2 = torch.nn.Conv2d(conv_channels, conv_channels, 1)

        out_length = (((feat_dim + 1) // 2 + 1) // 2 + 1) // 2
        self.out = torch.nn.Linear(conv_channels * out_length, feat_out)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample padded log-mel sequences by a factor of eight.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, feat_dim)``.
        x_lens : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
            torch.Tensor[torch.int32],
        ]
            Subsampled features with the same floating-point dtype as ``x`` and shape
            ``(batch_size, subsampled_num_frames, feat_out)`` and their
            ``torch.int32`` valid lengths.
        """

        x_lens = (x_lens + 1) // 2
        padding_mask = torch.arange(
            (x.size(1) + 1) // 2, dtype=x_lens.dtype, device=x_lens.device
        ).unsqueeze(0) >= x_lens.unsqueeze(1)
        conv2_mask = padding_mask[:, ::2]
        conv3_mask = conv2_mask[:, ::2]

        if self.batch_partitions > 1:
            conv2_lengths = (x_lens + 1) // 2
            partition_outputs: list[torch.Tensor] = []
            for partition in range(self.batch_partitions):
                start = partition * x.size(0) // self.batch_partitions
                end = (partition + 1) * x.size(0) // self.batch_partitions
                partition_output = self.conv1(x[start:end].unsqueeze(1))
                partition_output = self.relu(
                    partition_output.masked_fill(
                        padding_mask[start:end].unsqueeze(1).unsqueeze(3), 0.0
                    )
                )
                partition_output = self.conv2(partition_output)
                partition_mask = conv2_mask[start:end].unsqueeze(1).unsqueeze(3)
                partition_output = torch.nn.functional.linear(
                    partition_output.permute(0, 2, 3, 1),
                    self.pointwise_conv1.weight.squeeze(3).squeeze(2),
                    self.pointwise_conv1.bias,
                ).permute(0, 3, 1, 2)
                partition_output = self.relu(
                    partition_output.masked_fill(partition_mask, 0.0)
                )
                partition_outputs.append(self.conv3(partition_output))

            x = torch.cat(partition_outputs, dim=0)
            x_lens = conv2_lengths
        else:
            x = self.conv1(x.unsqueeze(1))
            x = self.relu(x.masked_fill(padding_mask.unsqueeze(1).unsqueeze(3), 0.0))
            x = self.conv2(x)

            x_lens = (x_lens + 1) // 2
            padding_mask = conv2_mask.unsqueeze(1).unsqueeze(3)

            x = torch.nn.functional.linear(
                x.permute(0, 2, 3, 1),
                self.pointwise_conv1.weight.squeeze(3).squeeze(2),
                self.pointwise_conv1.bias,
            ).permute(0, 3, 1, 2)
            x = self.relu(x.masked_fill(padding_mask, 0.0))
            x = self.conv3(x)

        x_lens = (x_lens + 1) // 2

        x = torch.nn.functional.linear(
            x.permute(0, 2, 3, 1),
            self.pointwise_conv2.weight.squeeze(3).squeeze(2),
            self.pointwise_conv2.bias,
        )
        padding_mask = conv3_mask.unsqueeze(2).unsqueeze(3)
        x = self.relu(x.masked_fill(padding_mask, 0.0))

        b, t, f, c = x.size()
        x = self.out(x.permute(0, 1, 3, 2).reshape(b, t, c * f))

        return x, x_lens


class ConformerFeedForward(torch.nn.Module):
    """Macaron feed-forward branch used in a Conformer layer."""

    def __init__(self, model_dim: int, feed_forward_dim: int) -> None:
        """Initialize the bias-free Conformer feed-forward projections.

        Parameters
        ----------
        model_dim : int
            The input and output hidden dimension.
        feed_forward_dim : int
            The intermediate feed-forward dimension.
        """

        super().__init__()

        self.linear1 = torch.nn.Linear(model_dim, feed_forward_dim, bias=False)
        self.activation = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(feed_forward_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Conformer feed-forward branch.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, model_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output features with the same dtype as ``x`` and shape
            ``(batch_size, num_frames, model_dim)``.
        """

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class ConformerConvolution(torch.nn.Module):
    """Conformer convolution module."""

    def __init__(self, model_dim: int, kernel_size: int) -> None:
        """Initialize the pointwise and depthwise convolution branches.

        Parameters
        ----------
        model_dim : int
            The hidden dimension, also the number of channels of the convolution module.
        kernel_size : int
            Odd kernel size of the depthwise convolution.

        """

        super().__init__()

        self.pointwise_conv1 = torch.nn.Linear(model_dim, 2 * model_dim, bias=False)
        self.depthwise_conv = torch.nn.Conv1d(
            model_dim, model_dim, kernel_size, groups=model_dim, bias=False
        )
        self.batch_norm = torch.nn.BatchNorm1d(model_dim)
        self.activation = torch.nn.SiLU()
        self.pointwise_conv2 = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor, output_lengths: torch.Tensor) -> torch.Tensor:
        """Apply the gated Conformer convolution branch.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, num_frames, model_dim)``.
        output_lengths : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output features with the same dtype as ``x`` and shape
            ``(batch_size, num_frames, model_dim)``.

        Notes
        -----
        Eager execution applies the original depthwise convolution and BatchNorm
        modules. ONNX export folds the evaluation-mode BatchNorm parameters into
        the depthwise weights and emits one native TensorRT plugin node.
        """

        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=2)

        if torch.onnx.is_in_onnx_export():
            batch_norm_scale = self.batch_norm.weight * torch.rsqrt(
                self.batch_norm.running_var + self.batch_norm.eps
            )
            depthwise_weight = (
                (self.depthwise_conv.weight * batch_norm_scale.reshape(-1, 1, 1))
                .squeeze(1)
                .permute(1, 0)
                .contiguous()
            )
            depthwise_bias = self.batch_norm.bias - (
                self.batch_norm.running_mean * batch_norm_scale
            )

            x = torch.onnx.ops.symbolic(
                PARAKEET_CONFORMER_CONVOLUTION_PLUGIN_NAME,
                (x, output_lengths, depthwise_weight, depthwise_bias),
                {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=x.dtype,
                shape=x.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            padding_mask = torch.arange(
                x.size(1), dtype=output_lengths.dtype, device=output_lengths.device
            ).unsqueeze(0) >= output_lengths.unsqueeze(1)
            x = x.masked_fill(padding_mask.unsqueeze(2), 0.0).permute(0, 2, 1)
            x = torch.nn.functional.pad(x, (self.padding, self.padding))
            x = self.depthwise_conv(x)
            x = self.batch_norm(x)
            x = self.activation(x).permute(0, 2, 1)

        x = self.pointwise_conv2(x)

        return x
