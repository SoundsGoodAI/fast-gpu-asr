#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""Offline Zipformer encoder assembled from export-friendly inference blocks."""

import copy

import torch

from .activation import SwooshL, SwooshR
from .attention import (
    CompactRelPositionalEncoding,
    NonlinAttention,
    RelPositionMultiheadAttentionWeights,
    SelfAttention,
)
from .features import FeatureExtractor
from .subsampling import BiasNorm, Conv2dSubsampling


class Zipformer2(torch.nn.Module):
    """
    Audio-to-encoder module for offline Zipformer2 export.
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
        min_frames: int,
        subsample_output_dim: int,
        subsample_layer1_channels: int,
        subsample_layer2_channels: int,
        subsample_layer3_channels: int,
        encoder_dims: list[int],
        num_encoder_layers: list[int],
        downsampling_factors: list[int],
        num_heads: list[int],
        feedforward_dims: list[int],
        cnn_module_kernels: list[int],
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        pos_dim: int,
        pos_max_len: int,
        output_dim: int,
        use_ctc: bool,
    ) -> None:
        """
        Zipformer2 initialization.

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
            The lower mel-filterbank frequency in Hertz.
        high_freq : int
            The upper mel-filterbank frequency in Hertz.
        min_frames : int
            The minimum feature length accepted by the encoder.
        subsample_output_dim : int
            The output dimension of the subsampling module represented by
            Conv2dSubsampling.
        subsample_layer1_channels : int
            The number of output channels in the first Conv2d layer of the
            Conv2dSubsampling module.
        subsample_layer2_channels : int
            The number of output channels in the second Conv2d layer of the
            Conv2dSubsampling module.
        subsample_layer3_channels : int
            The number of output channels in the third Conv2d layer of the
            Conv2dSubsampling module.
        encoder_dims : list[int]
            A list of 6 integers, the embedding dimension of Zipformer2EncoderLayer
            module in each Zipformer2Encoder stack. Dimensions must be nondecreasing
            through the fourth stack and nonincreasing afterward.
        num_encoder_layers : list[int]
            A list of 6 integers, the number of Zipformer2EncoderLayer modules in each
            Zipformer2Encoder stack.
        downsampling_factors : list[int]
            A list of 6 positive integers, the downsampling factor of each
            Zipformer2Encoder stack.
            Note: this is in addition to the downsampling factor of 2 that is applied in
            the Conv2dSubsampling module.
        num_heads : list[int]
            A list of 6 integers, the number of heads for attention weights and
            self-attention of the Zipformer2EncoderLayer module in each
            Zipformer2Encoder stack.
        feedforward_dims : list[int]
            A list of 6 integers, the hidden dimension of the feedforward module of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        cnn_module_kernels : list[int]
            A list of 6 integers, the kernel size of the convolution module of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        query_head_dim : int
            The dimension of the query and key per attention head in attention weights
            of the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        pos_head_dim : int
            The dimension of the projected positional encoding per attention head in
            attention weights of the Zipformer2EncoderLayer module in each
            Zipformer2Encoder stack.
        value_head_dim : int
            The dimension of the value per attention head in self-attention of
            the Zipformer2EncoderLayer module in each Zipformer2Encoder stack.
        pos_dim : int
            The dimension of the relative positional embeddings in each
            Zipformer2Encoder stack.
        pos_max_len : int
            The initial maximum sequence length of the relative positional embeddings
            in each Zipformer2Encoder stack. Longer inputs require regenerating the
            positional table and may degrade inference speed.
        output_dim : int
            The output dimension after final output projection.
        use_ctc : bool
            Whether the output projection contains a CTC head. If true, log-softmax is
            applied to the final output.
        """

        super().__init__()

        device = torch.device("cpu")

        self.encoder_dims = tuple(encoder_dims)
        self.downsampling_factors = tuple(downsampling_factors)
        projection_dim = max(encoder_dims)
        self.projection_dim = projection_dim
        self.ctc = use_ctc

        self.feature_extractor = FeatureExtractor(
            samp_freq,
            frame_shift_ms,
            frame_length_ms,
            feature_dim,
            preemph,
            low_freq,
            high_freq,
            min_frames,
            device,
        )

        self.subsampling = Conv2dSubsampling(
            feature_dim,
            subsample_output_dim,
            subsample_layer1_channels,
            subsample_layer2_channels,
            subsample_layer3_channels,
            device,
        )

        encoders = []
        for i, num_layers in enumerate(num_encoder_layers):
            encoder_layer = Zipformer2EncoderLayer(
                encoder_dims[i],
                pos_dim,
                num_heads[i],
                query_head_dim,
                pos_head_dim,
                value_head_dim,
                feedforward_dims[i],
                cnn_module_kernels[i],
                device,
            )

            encoder = Zipformer2Encoder(
                encoder_layer,
                num_layers,
                encoder_dims[i],
                pos_dim,
                pos_max_len,
                downsampling_factors[i],
                device,
            )

            encoders.append(encoder)

        (
            self.encoder_1,
            self.encoder_2,
            self.encoder_3,
            self.encoder_4,
            self.encoder_5,
            self.encoder_6,
        ) = encoders

        self.downsample_output = SimpleDownsample(2, device)
        self.projection_output = torch.nn.Linear(
            projection_dim, output_dim, device=device
        )

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Does a forward pass from waveform to encoder embeddings.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded waveforms with shape ``(batch_size, num_samples)``.
        audio_lengths : torch.Tensor[torch.int32]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Encoder embeddings with shape
            ``(batch_size, num_encoder_frames, output_dim)`` and
            ``torch.int32`` valid frame counts.
        """

        x, x_lens = self.feature_extractor(audio, audio_lengths)
        x, x_lens = self.subsampling(x, x_lens)

        batch_size, seq_len, _ = x.size()
        padding_mask = torch.arange(seq_len, device=x_lens.device).unsqueeze(
            0
        ) >= x_lens.unsqueeze(1)

        output = torch.empty(
            batch_size, seq_len, self.projection_dim, dtype=x.dtype, device=x.device
        )

        # Encoder 1
        x = self.encoder_1(x, padding_mask)
        output[:, :, : x.size(2)] = x

        # Encoder 2
        pad = torch.zeros(
            x.size(0),
            x.size(1),
            self.encoder_dims[1] - x.size(2),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)
        x = self.encoder_2(x, padding_mask)
        output[:, :, : x.size(2)] = x

        # Encoder 3
        pad = torch.zeros(
            x.size(0),
            x.size(1),
            self.encoder_dims[2] - x.size(2),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)
        x = self.encoder_3(x, padding_mask)
        output[:, :, : x.size(2)] = x

        # Encoder 4
        pad = torch.zeros(
            x.size(0),
            x.size(1),
            self.encoder_dims[3] - x.size(2),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat((x, pad), dim=2)
        x = self.encoder_4(x, padding_mask)
        output[:, :, : x.size(2)] = x

        # Encoder 5
        x = x[:, :, : self.encoder_dims[4]]
        x = self.encoder_5(x, padding_mask)
        output[:, :, : x.size(2)] = x

        # Encoder 6
        x = x[:, :, : self.encoder_dims[5]]
        x = self.encoder_6(x, padding_mask)
        output[:, :, : x.size(2)] = x

        output = self.downsample_output(output)
        output_lens = (x_lens + 1) // 2
        output = self.projection_output(output)
        if self.ctc:
            output = torch.nn.functional.log_softmax(output, dim=2)

        return output, output_lens


class Zipformer2Encoder(torch.nn.Module):
    """
    Zipformer2Encoder is a stack of Zipformer2EncoderLayer modules.
    """

    def __init__(
        self,
        encoder_layer: torch.nn.Module,
        num_layers: int,
        embed_dim: int,
        pos_dim: int,
        pos_max_len: int,
        downsample: int,
        device: torch.device,
    ) -> None:
        """
        Zipformer2Encoder initialization.

        Parameters
        ----------
        encoder_layer : torch.nn.Module
            An instance of the Zipformer2EncoderLayer class.
        num_layers : int
            The number of encoder Zipformer2EncoderLayer modules in the stack.
        embed_dim : int
            The input and output embedding dimension. The embedding dimension is the
            same for input and output of this module.
        pos_dim : int
            The dimension for the relative positional embedding.
        pos_max_len : int
            The initial maximum sequence length of the relative positional table.
        downsample : int
            The downsampling factor of the module, the input will be downsampled in the
            beginning and upsampled back at the end.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.downsample = SimpleDownsample(downsample, device)
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, pos_max_len, device)

        self.layers = torch.nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)],
        )
        self.upsample = SimpleUpsample(downsample)
        self.out_combiner = BypassModule(embed_dim, device)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the Zipformer2Encoder module and returns an output
        tensor with the same shape as its input.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output with the same shape as ``x``.
        """

        x_orig = x

        x = self.downsample(x)
        downsampled_padding_mask = padding_mask[:, :: self.downsample.weights.size(0)]
        pos_emb = self.encoder_pos(x)
        for mod in self.layers:
            x = mod(x, pos_emb, downsampled_padding_mask)

        x = self.upsample(x)
        # Remove any extra frames that are not a multiple of downsample_factor
        x = x[:, : x_orig.size(1)]
        x = self.out_combiner(x_orig, x)

        return x


class Zipformer2EncoderLayer(torch.nn.Module):
    """
    Zipformer2EncoderLayer module, the basic block of Zipformer2Encoder encoder stack.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        cnn_module_kernel: int,
        device: torch.device,
    ) -> None:
        """
        Zipformer2EncoderLayer initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for input and output of this module.
        pos_dim : int
            The dimension of the relative positional embedding.
        num_heads : int
            The number of heads for attention weights and self-attention.
        query_head_dim : int
            The dimension of the query and key per attention head in attention weights.
        pos_head_dim : int
            The dimension of the projected positional encoding per attention head in
            attention weights.
        value_head_dim : int
            The dimension of the value per attention head in self-attention.
        feedforward_dim : int
            The hidden dimension of the feedforward modules.
        cnn_module_kernel : int
            The kernel size of the convolution modules.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.bypass = BypassModule(embed_dim, device)
        self.bypass_mid = BypassModule(embed_dim, device)

        self.feed_forward1 = FeedforwardModule(
            embed_dim, (feedforward_dim * 3) // 4, device
        )
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, device)
        self.feed_forward3 = FeedforwardModule(
            embed_dim, (feedforward_dim * 5) // 4, device
        )

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim, num_heads, query_head_dim, pos_head_dim, device
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim, device)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim, device)
        self.nonlin_attention = NonlinAttention(embed_dim, 3 * embed_dim // 4, device)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel, device)
        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel, device)

        self.norm = BiasNorm(embed_dim, device)

    def forward(
        self, x: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Does a forward pass of the Zipformer2EncoderLayer module. Returns an output
        tensor with the same shape as the input.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        pos_emb : torch.Tensor[torch.float32]
            Positional embeddings with shape
            ``(batch_size, 2 * seq_len - 1, pos_dim)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output with the same shape as ``x``.
        """

        x_orig = x

        # (batch_size, num_heads, seq_len, seq_len_2)
        attn_weights = self.self_attn_weights(x, pos_emb, padding_mask)

        x = x + self.feed_forward1(x)
        x = x + self.nonlin_attention(x, attn_weights[:, 0])
        x = x + self.self_attn1(x, attn_weights)
        x = x + self.conv_module1(x, padding_mask)

        x = x + self.feed_forward2(x)
        x = self.bypass_mid(x_orig, x)
        x = x + self.self_attn2(x, attn_weights)
        x = x + self.conv_module2(x, padding_mask)

        x = x + self.feed_forward3(x)
        x = self.norm(x)
        x = self.bypass(x_orig, x)

        return x


class ConvolutionModule(torch.nn.Module):
    """
    ConvolutionModule in Zipformer2 encoder.
    """

    def __init__(self, embed_dim: int, kernel_size: int, device: torch.device) -> None:
        """
        ConvolutionModule initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension, also the number of channels of
            convolution modules. The embedding dimension is the same for input and
            output of this module.
        kernel_size : int
            The kernel size of the depthwise convolution module.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                "ConvolutionModule kernel size should be "
                f"an odd number but got {kernel_size} instead.",
            )

        self.in_proj = torch.nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.depthwise_conv = torch.nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size,
            groups=embed_dim,
            padding=kernel_size // 2,
            device=device,
        )

        self.activation = SwooshR()
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, device=device)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the ConvolutionModule. Returns a processed tensor with
        the same shape as the input.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output with the same shape as ``x``.
        """

        x = self.in_proj(x)  # (batch_size, seq_len, 2 * embed_dim)

        x, s = x.chunk(2, dim=2)
        x = x * torch.sigmoid(s)  # (batch_size, seq_len, embed_dim)

        # exchange the temporal dimension and the feature dimension for depthwise
        # convolution.
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len).
        x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)

        x = self.activation(x)
        x = self.out_proj(x)  # (batch_size, seq_len, embed_dim)

        return x


class FeedforwardModule(torch.nn.Module):
    """
    Feedforward module in Zipformer2 encoder.
    """

    def __init__(
        self, embed_dim: int, feedforward_dim: int, device: torch.device
    ) -> None:
        """
        FeedforwardModule initialization.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for the input and output.
        feedforward_dim : int
            The module hidden dimension.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, feedforward_dim, device=device)
        self.activation = SwooshL()
        self.out_proj = torch.nn.Linear(feedforward_dim, embed_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the FeedforwardModule.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output with the same shape as ``x``.
        """

        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)

        return x


class BypassModule(torch.nn.Module):
    """
    A bypass module that implements a learnable bypass scale for each input channel.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        BypassModule initialization.

        Parameters
        ----------
        num_channels : int
            The number of input channels, corresponds to the number of learnable bypass
            scales.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()
        self.bypass_scale = torch.nn.Parameter(
            torch.ones(num_channels, dtype=torch.float32, device=device),
        )

    def forward(self, x_early: torch.Tensor, x_later: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the BypassModule.

        Parameters
        ----------
        x_early : torch.Tensor[torch.float32]
            Input of shape ``(batch_size, seq_len, num_channels)``.
            The module input that will be propagated with (1 - self.bypass_scale)
            weight.
        x_later : torch.Tensor[torch.float32]
            Input of shape ``(batch_size, seq_len, num_channels)``.
            The module input that will be propagated with self.bypass_scale weight.

        Returns
        -------
        torch.Tensor[torch.float32]
            Combined output with the same shape as both inputs.
        """

        # This is just a slightly more efficient implementation of
        # (1.0 - self.bypass_scale) * x_early + self.bypass_scale * x_later
        return x_early + (x_later - x_early) * self.bypass_scale


class SimpleDownsample(torch.nn.Module):
    """
    A downsample layer, does downsampling by weighted sum aggregation.
    """

    def __init__(self, downsample: int, device: torch.device) -> None:
        """
        SimpleDownsample initialization.

        Parameters
        ----------
        downsample : int
            The module downsampling factor.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.zeros(downsample, 1, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the SimpleDownsample module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input of shape ``(batch_size, seq_len, num_channels)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Downsampled output of shape
            ``(batch_size, ceil(seq_len / downsample), num_channels)``.
        """

        downsample = self.weights.size(0)
        if downsample == 1:
            return x

        batch_size, seq_len, in_channels = x.size()
        downsampled_seq_len = (seq_len + downsample - 1) // downsample

        # Always append enough copies of the final frame, then select the exact
        # multiple required by the reshape.
        pad = x[:, seq_len - 1 : seq_len, :].expand(
            batch_size, downsample - 1, in_channels
        )
        x = torch.cat((x, pad), dim=1)
        x = x[:, : downsampled_seq_len * downsample, :]

        # (batch_size, seq_len, in_channels)
        # -> (batch_size, seq_len // downsample, downsample, in_channels)
        x = x.reshape(batch_size, downsampled_seq_len, downsample, in_channels)
        x = torch.sum(x * self.weights, dim=2)

        return x


class SimpleUpsample(torch.nn.Module):
    """
    An upsample layer, does upsampling by repeating the input frames.
    """

    def __init__(self, upsample: int) -> None:
        """
        SimpleUpsample initialization.

        Parameters
        ----------
        upsample : int
            The module upsampling factor.
        """

        super().__init__()
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the SimpleUpsample module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            Input of shape ``(batch_size, seq_len, num_channels)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Output of shape ``(batch_size, seq_len * upsample, num_channels)``.
        """

        if self.upsample == 1:
            return x

        x = torch.repeat_interleave(x, self.upsample, dim=1)

        return x
