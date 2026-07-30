#!/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Fast Conformer encoder used by NVIDIA Parakeet TDT models."""

import torch

from .attention import RelPositionalEncoding, RelPositionMultiHeadAttention
from .features import FeatureExtractor


class ParakeetTDTEncoder(torch.nn.Module):
    """
    Audio-to-encoder module for Parakeet TDT export.
    """

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
    ) -> None:
        """
        ParakeetTDTEncoder initialization.

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
        """

        super().__init__()

        device = torch.device("cpu")

        self.feature_extractor = FeatureExtractor(
            samp_freq,
            frame_shift_ms,
            frame_length_ms,
            feature_dim,
            preemph,
            low_freq,
            high_freq,
            device,
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
            device,
        )

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass from waveform to encoder embeddings.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded waveforms of shape ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int32]
            Valid sample counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Encoder output of shape
            ``(batch_size, subsampled_num_frames, model_dim)`` and
            ``torch.int32`` valid lengths.
        """

        features, feature_lengths = self.feature_extractor(audio, audio_lengths)
        encoder_out, encoder_out_lengths = self.encoder(features, feature_lengths)

        return encoder_out, encoder_out_lengths


class FastConformer(torch.nn.Module):
    """
    Fast Conformer encoder matching the Parakeet TDT encoder layout.
    """

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
        device: torch.device,
    ) -> None:
        """
        FastConformer initialization.

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
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.pre_encode = ConvSubsampling(
            input_dim, model_dim, subsampling_conv_channels, device
        )
        self.pos_enc = RelPositionalEncoding(model_dim, pos_emb_max_len, device)
        self.layers = torch.nn.ModuleList(
            ConformerLayer(
                model_dim,
                model_dim * feed_forward_expansion_factor,
                n_heads,
                conv_kernel_size,
                device,
            )
            for _ in range(n_layers)
        )

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the FastConformer module.

        Parameters
        ----------
        features : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, input_dim)``.
        feature_lengths : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Encoder output of shape
            ``(batch_size, subsampled_num_frames, model_dim)`` and
            ``torch.int32`` valid lengths.
        """

        x, output_lengths = self.pre_encode(features, feature_lengths)
        padding_mask = torch.arange(x.size(1), device=output_lengths.device).unsqueeze(
            0
        ) >= output_lengths.unsqueeze(1)
        pos_emb = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, pos_emb, padding_mask)

        return x, output_lengths


class ConformerLayer(torch.nn.Module):
    """
    Single Conformer encoder layer.
    """

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        n_heads: int,
        conv_kernel_size: int,
        device: torch.device,
    ) -> None:
        """
        ConformerLayer initialization.

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
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.fc_factor = 0.5

        self.norm_feed_forward1 = torch.nn.LayerNorm(model_dim, device=device)
        self.feed_forward1 = ConformerFeedForward(model_dim, feed_forward_dim, device)

        self.norm_conv = torch.nn.LayerNorm(model_dim, device=device)
        self.conv = ConformerConvolution(model_dim, conv_kernel_size, device)

        self.norm_self_att = torch.nn.LayerNorm(model_dim, device=device)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, model_dim, device)

        self.norm_feed_forward2 = torch.nn.LayerNorm(model_dim, device=device)
        self.feed_forward2 = ConformerFeedForward(model_dim, feed_forward_dim, device)

        self.norm_out = torch.nn.LayerNorm(model_dim, device=device)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Does a forward pass of the ConformerLayer module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, model_dim)``.
        pos_emb : torch.Tensor[torch.float32]
            The relative positional embeddings of shape
            ``(1, 2 * num_frames - 1, model_dim)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, num_frames)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output features of shape ``(batch_size, num_frames, model_dim)``.
        """

        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, padding_mask)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x, padding_mask)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        x = residual + x * self.fc_factor

        return self.norm_out(x)


class ConvSubsampling(torch.nn.Module):
    """
    Parakeet ``dw_striding`` 2D convolutional subsampling.
    """

    def __init__(
        self, feat_dim: int, feat_out: int, conv_channels: int, device: torch.device
    ) -> None:
        """
        ConvSubsampling initialization.

        Parameters
        ----------
        feat_dim : int
            The number of input mel bins.
        feat_out : int
            The output feature dimension after the projection layer.
        conv_channels : int
            The intermediate convolution channel count.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(
            1, conv_channels, 3, stride=2, padding=1, device=device
        )
        self.conv2 = torch.nn.Conv2d(
            conv_channels,
            conv_channels,
            3,
            stride=2,
            padding=1,
            groups=conv_channels,
            device=device,
        )
        self.conv3 = torch.nn.Conv2d(
            conv_channels,
            conv_channels,
            3,
            stride=2,
            padding=1,
            groups=conv_channels,
            device=device,
        )

        self.pointwise_conv1 = torch.nn.Conv2d(
            conv_channels, conv_channels, 1, device=device
        )
        self.pointwise_conv2 = torch.nn.Conv2d(
            conv_channels, conv_channels, 1, device=device
        )

        out_length = (((feat_dim + 1) // 2 + 1) // 2 + 1) // 2
        self.out = torch.nn.Linear(conv_channels * out_length, feat_out, device=device)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the ConvSubsampling module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, feat_dim)``.
        x_lens : torch.Tensor[torch.int32]
            Valid frame counts of shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Subsampled features of shape
            ``(batch_size, subsampled_num_frames, feat_out)`` and their
            ``torch.int32`` valid lengths.
        """

        x = self.conv1(x.unsqueeze(1))

        x_lens = (x_lens + 1) // 2
        padding_mask = torch.arange(
            x.size(2), dtype=x_lens.dtype, device=x_lens.device
        ).unsqueeze(0) >= x_lens.unsqueeze(1)
        x = self.relu(x.masked_fill(padding_mask.unsqueeze(1).unsqueeze(3), 0.0))
        x = self.conv2(x)

        x_lens = (x_lens + 1) // 2
        padding_mask = torch.arange(
            x.size(2), dtype=x_lens.dtype, device=x_lens.device
        ).unsqueeze(0) >= x_lens.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(3)
        x = self.pointwise_conv1(x.masked_fill(padding_mask, 0.0))
        x = self.relu(x.masked_fill(padding_mask, 0.0))
        x = self.conv3(x.masked_fill(padding_mask, 0.0))

        x_lens = (x_lens + 1) // 2
        padding_mask = torch.arange(
            x.size(2), dtype=x_lens.dtype, device=x_lens.device
        ).unsqueeze(0) >= x_lens.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(3)
        x = self.pointwise_conv2(x.masked_fill(padding_mask, 0.0))
        x = self.relu(x.masked_fill(padding_mask, 0.0))
        x = x.masked_fill(padding_mask, 0.0)

        b, c, t, f = x.size()
        x = self.out(x.permute(0, 2, 1, 3).reshape(b, t, c * f))

        return x, x_lens


class ConformerFeedForward(torch.nn.Module):
    """
    Macaron feed-forward branch used in a Conformer layer.
    """

    def __init__(
        self, model_dim: int, feed_forward_dim: int, device: torch.device
    ) -> None:
        """
        ConformerFeedForward initialization.

        Parameters
        ----------
        model_dim : int
            The input and output hidden dimension.
        feed_forward_dim : int
            The intermediate feed-forward dimension.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.linear1 = torch.nn.Linear(
            model_dim, feed_forward_dim, bias=False, device=device
        )
        self.activation = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(
            feed_forward_dim, model_dim, bias=False, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the ConformerFeedForward module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, model_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output features of shape ``(batch_size, num_frames, model_dim)``.
        """

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class ConformerConvolution(torch.nn.Module):
    """
    Conformer convolution module.
    """

    def __init__(self, model_dim: int, kernel_size: int, device: torch.device) -> None:
        """
        ConformerConvolution initialization.

        Parameters
        ----------
        model_dim : int
            The hidden dimension, also the number of channels of the convolution module.
        kernel_size : int
            The kernel size of the depthwise convolution module. Should be an odd
            number.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if (kernel_size - 1) % 2 != 0:
            raise ValueError("Conformer convolution kernel size must be odd.")

        self.pointwise_conv1 = torch.nn.Linear(
            model_dim, 2 * model_dim, bias=False, device=device
        )
        self.depthwise_conv = torch.nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=0,
            groups=model_dim,
            bias=False,
            device=device,
        )
        self.batch_norm = torch.nn.BatchNorm1d(model_dim, device=device)
        self.activation = torch.nn.SiLU()
        self.pointwise_conv2 = torch.nn.Linear(
            model_dim, model_dim, bias=False, device=device
        )
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the ConformerConvolution module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, num_frames, model_dim)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, num_frames)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output features of shape ``(batch_size, num_frames, model_dim)``.
        """

        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=2)

        x = x.permute(0, 2, 1)
        x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)
        x = torch.nn.functional.pad(x, (self.padding, self.padding))
        x = self.activation(self.batch_norm(self.depthwise_conv(x)))
        x = x.permute(0, 2, 1)

        x = self.pointwise_conv2(x)

        return x
