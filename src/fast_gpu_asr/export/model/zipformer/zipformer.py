#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Offline Zipformer encoder assembled from export-friendly inference blocks."""

import copy

import torch

from ....constants import (
    ONNX_OPSET_VERSION,
    TENSORRT_PLUGIN_NAMESPACE,
    ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
    ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
    ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
    ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
)
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
    """Audio-to-encoder module for offline Zipformer2 export.

    Waveform feature extraction remains in ``torch.float32``. The extracted
    features cross one explicit boundary into the configured encoder dtype
    before convolutional subsampling. BF16 export keeps the first subsampling
    convolution in ``torch.float16`` because TensorRT executes that convolution
    substantially faster in FP16, then converts back to ``torch.bfloat16`` for
    the rest of subsampling and all six encoder stacks.
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
        subsampling_batch_partitions: int,
        encoder_dims: list[int],
        num_encoder_layers: list[int],
        downsampling_factors: list[int],
        bypass_scales: list[torch.Tensor],
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
        dtype: torch.dtype,
    ) -> None:
        """Initialize the waveform frontend and six Zipformer encoder stacks.

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
        subsampling_batch_partitions : int
            Number of batch partitions used by the convolutional subsampling frontend.
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
        bypass_scales : list[torch.Tensor]
            Learned per-channel output bypass scales for the 6 encoder stacks.
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
            Maximum sequence length supported by the fixed relative positional table
            in each ``Zipformer2Encoder`` stack.
        output_dim : int
            The output dimension after final output projection.
        use_ctc : bool
            Whether the output projection contains a CTC head. CTC heads return
            normalized log probabilities.
        dtype : torch.dtype
            Floating-point dtype used by subsampling and the encoder stacks.
            The final output projection remains ``torch.float32`` for both
            transducer and CTC models.
        """

        super().__init__()

        self.encoder_dims = tuple(encoder_dims)
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
        )

        self.subsampling = Conv2dSubsampling(
            feature_dim,
            subsample_output_dim,
            subsample_layer1_channels,
            subsample_layer2_channels,
            subsample_layer3_channels,
            subsampling_batch_partitions,
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
            )

            encoder = Zipformer2Encoder(
                encoder_layer,
                num_layers,
                pos_dim,
                pos_max_len,
                downsampling_factors[i],
                bypass_scales[i],
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

        self.downsample_output = SimpleDownsample(2)
        self.projection_output = torch.nn.Linear(projection_dim, output_dim)

        for module in (
            self.subsampling,
            *encoders,
            self.downsample_output,
        ):
            module.to(dtype=dtype)

        if dtype == torch.bfloat16:
            self.subsampling.conv1.to(dtype=torch.float16)

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert padded waveforms into encoder embeddings.

        Parameters
        ----------
        audio : torch.Tensor[torch.float32]
            Padded waveforms with shape
            ``(batch_size, num_samples + right_padding)``. The runtime appends
            the reflected right context required by the feature extractor.
        audio_lengths : torch.Tensor[torch.int64]
            Valid sample counts with shape ``(batch_size,)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32],
            torch.Tensor[torch.int32],
        ]
            Encoder embeddings with shape
            ``(batch_size, num_encoder_frames, output_dim)`` and
            ``torch.int32`` valid frame counts. The final projection returns
            ``torch.float32`` embeddings or CTC log probabilities.
        """

        x, x_lens = self.feature_extractor(audio, audio_lengths)
        x, x_lens = self.subsampling(x, x_lens)

        padding_mask = torch.arange(
            x.size(1),
            dtype=x_lens.dtype,
            device=x_lens.device,
        ).unsqueeze(0) >= x_lens.unsqueeze(1)

        # Encoder 1
        encoder_1_output = self.encoder_1(x, x_lens, padding_mask)

        # Encoder 2
        x = torch.nn.functional.pad(
            encoder_1_output, (0, self.encoder_dims[1] - self.encoder_dims[0])
        )
        encoder_2_output = self.encoder_2(x, x_lens, padding_mask)

        # Encoder 3
        x = torch.nn.functional.pad(
            encoder_2_output, (0, self.encoder_dims[2] - self.encoder_dims[1])
        )
        encoder_3_output = self.encoder_3(x, x_lens, padding_mask)

        # Encoder 4
        x = torch.nn.functional.pad(
            encoder_3_output, (0, self.encoder_dims[3] - self.encoder_dims[2])
        )
        encoder_4_output = self.encoder_4(x, x_lens, padding_mask)

        # Encoder 5
        x = encoder_4_output[:, :, : self.encoder_dims[4]]
        encoder_5_output = self.encoder_5(x, x_lens, padding_mask)

        # Encoder 6
        x = encoder_5_output[:, :, : self.encoder_dims[5]]
        encoder_6_output = self.encoder_6(x, x_lens, padding_mask)

        if torch.onnx.is_in_onnx_export():
            # All six inputs are intentional: besides assembling the three surviving
            # channel bands, the opaque plugin prevents unsafe cross-stack Myelin
            # fusion observed when earlier stack outputs disappear from the graph.
            output = torch.onnx.ops.symbolic(
                ZIPFORMER_OUTPUT_ASSEMBLY_PLUGIN_NAME,
                (
                    encoder_1_output,
                    encoder_2_output,
                    encoder_3_output,
                    encoder_4_output,
                    encoder_5_output,
                    encoder_6_output,
                ),
                {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=encoder_4_output.dtype,
                shape=encoder_4_output.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            output = torch.cat(
                (
                    encoder_6_output,
                    encoder_5_output[:, :, self.encoder_dims[5] :],
                    encoder_4_output[:, :, self.encoder_dims[4] :],
                ),
                dim=2,
            )

        output = self.downsample_output(output)
        output_lens = (x_lens + 1) // 2
        output = self.projection_output(output.to(self.projection_output.weight.dtype))

        if self.ctc:
            output = torch.nn.functional.log_softmax(output, dim=2)

        return output, output_lens


class Zipformer2Encoder(torch.nn.Module):
    """Apply one temporally resampled stack of Zipformer encoder layers."""

    def __init__(
        self,
        encoder_layer: torch.nn.Module,
        num_layers: int,
        pos_dim: int,
        pos_max_len: int,
        downsample: int,
        bypass_scale: torch.Tensor,
    ) -> None:
        """Initialize temporal resampling, positional encoding, and encoder layers.

        Parameters
        ----------
        encoder_layer : torch.nn.Module
            An instance of the Zipformer2EncoderLayer class.
        num_layers : int
            The number of encoder Zipformer2EncoderLayer modules in the stack.
        pos_dim : int
            The dimension for the relative positional embedding.
        pos_max_len : int
            Maximum sequence length supported by the fixed relative positional table.
        downsample : int
            The downsampling factor of the module, the input will be downsampled in the
            beginning and upsampled back at the end.
        bypass_scale : torch.Tensor[torch.float32]
            Learned per-channel bypass scales with shape ``(embed_dim,)``.
        """

        super().__init__()

        self.downsample_value = downsample
        self.register_buffer("bypass_scale", bypass_scale)

        self.downsample = SimpleDownsample(downsample)
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, pos_max_len)
        self.upsample = SimpleUpsample(downsample)

        self.layers = torch.nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)],
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one stack and restore its original temporal resolution.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        lengths : torch.Tensor[torch.int32]
            Valid frame counts with shape ``(batch_size,)``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, seq_len)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        x_orig = x

        x = self.downsample(x)
        downsampled_lengths = (
            lengths + self.downsample_value - 1
        ) // self.downsample_value
        downsampled_padding_mask = padding_mask[:, :: self.downsample_value]
        pos_emb = self.encoder_pos(x)
        for mod in self.layers:
            x = mod(x, pos_emb, downsampled_padding_mask, downsampled_lengths)

        x = self.upsample(x_orig, x, self.bypass_scale)

        return x


class Zipformer2EncoderLayer(torch.nn.Module):
    """Combine Zipformer feed-forward, attention, convolution, and bypass branches."""

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
    ) -> None:
        """Initialize one Zipformer encoder layer.

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
        """

        super().__init__()

        self.bypass = BypassModule(embed_dim)
        self.bypass_mid = BypassModule(embed_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4)
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim)
        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim, num_heads, query_head_dim, pos_head_dim
        )
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.nonlin_attention = NonlinAttention(embed_dim, 3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel)
        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel)

        self.norm = BiasNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        padding_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one Zipformer encoder layer.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        pos_emb : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Positional embeddings with shape ``(1, 2 * seq_len - 1, pos_dim)``.
            The singleton batch dimension is broadcast over ``x``.
        padding_mask : torch.Tensor[torch.bool]
            Padding mask of shape ``(batch_size, seq_len)``.
        valid_lengths : torch.Tensor[torch.int32]
            Valid frame counts with shape ``(batch_size,)`` shared by both
            convolution modules in every layer of an encoder stack.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        x_orig = x

        attn_weights = self.self_attn_weights(x, pos_emb, padding_mask)

        x = x + self.feed_forward1(x)
        x = x + self.nonlin_attention(x, attn_weights)
        x = x + self.self_attn1(x, attn_weights)
        x = x + self.conv_module1(x, valid_lengths)

        x = x + self.feed_forward2(x)
        x = self.bypass_mid(x_orig, x)
        x = x + self.self_attn2(x, attn_weights)
        x = x + self.conv_module2(x, valid_lengths)

        x = x + self.feed_forward3(x)
        x = self.norm(x)
        x = self.bypass(x_orig, x)

        return x


class ConvolutionModule(torch.nn.Module):
    """Apply Zipformer's gated depthwise temporal convolution branch."""

    def __init__(self, embed_dim: int, kernel_size: int) -> None:
        """Initialize pointwise projections and depthwise convolution.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension, also the number of channels of
            convolution modules. The embedding dimension is the same for input and
            output of this module.
        kernel_size : int
            The kernel size of the depthwise convolution module.

        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, 2 * embed_dim)
        self.depthwise_conv = torch.nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size,
            groups=embed_dim,
            padding=kernel_size // 2,
        )

        self.activation = SwooshR()
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        """Apply gated convolution while suppressing padded frames.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.
        valid_lengths : torch.Tensor[torch.int32]
            Valid frame counts with shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        x = self.in_proj(x)
        x = torch.nn.functional.glu(x, dim=2)

        if torch.onnx.is_in_onnx_export():
            depthwise_weight = (
                self.depthwise_conv.weight.squeeze(1).permute(1, 0).contiguous()
            )
            x = torch.onnx.ops.symbolic(
                ZIPFORMER_CONVOLUTION_PLUGIN_NAME,
                (x, valid_lengths, depthwise_weight, self.depthwise_conv.bias),
                {"plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=x.dtype,
                shape=x.shape,
                version=ONNX_OPSET_VERSION,
            )
        else:
            padding_mask = torch.arange(
                x.size(1), dtype=valid_lengths.dtype, device=valid_lengths.device
            ).unsqueeze(0) >= valid_lengths.unsqueeze(1)
            x = x.masked_fill(padding_mask.unsqueeze(2), 0.0).permute(0, 2, 1)
            x = self.depthwise_conv(x)
            x = self.activation(x).permute(0, 2, 1)

        x = self.out_proj(x)  # (batch_size, seq_len, embed_dim)

        return x


class FeedforwardModule(torch.nn.Module):
    """Apply one Swoosh-L Zipformer feed-forward branch."""

    def __init__(self, embed_dim: int, feedforward_dim: int) -> None:
        """Initialize input and output projections.

        Parameters
        ----------
        embed_dim : int
            The input and output embedding dimension. The number of channels is the same
            for the input and output.
        feedforward_dim : int
            The module hidden dimension.
        """

        super().__init__()

        self.in_proj = torch.nn.Linear(embed_dim, feedforward_dim)
        self.activation = SwooshL()
        self.out_proj = torch.nn.Linear(feedforward_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward branch.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input features of shape ``(batch_size, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Output with the same shape and dtype as ``x``.
        """

        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)

        return x


class BypassModule(torch.nn.Module):
    """Apply a checkpointed bypass scale to each input channel."""

    def __init__(self, num_channels: int) -> None:
        """Initialize one learned scale per channel.

        Parameters
        ----------
        num_channels : int
            The number of input channels and corresponding bypass scales.
        """

        super().__init__()
        self.register_buffer(
            "bypass_scale", torch.ones(num_channels, dtype=torch.float32)
        )

    def forward(self, x_early: torch.Tensor, x_later: torch.Tensor) -> torch.Tensor:
        """Interpolate between early and later representations.

        Parameters
        ----------
        x_early : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input of shape ``(batch_size, seq_len, num_channels)``.
            The module input that will be propagated with (1 - self.bypass_scale)
            weight.
        x_later : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input of shape ``(batch_size, seq_len, num_channels)``.
            The module input that will be propagated with self.bypass_scale weight.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Combined output with the same shape and dtype as both inputs.
        """

        # This is just a slightly more efficient implementation of
        # (1.0 - self.bypass_scale) * x_early + self.bypass_scale * x_later
        return x_early + (x_later - x_early) * self.bypass_scale


class SimpleDownsample(torch.nn.Module):
    """Downsample time with checkpointed normalized aggregation weights."""

    def __init__(self, downsample: int) -> None:
        """Initialize the checkpointed aggregation logits.

        Parameters
        ----------
        downsample : int
            The module downsampling factor.
        """

        super().__init__()
        self.register_buffer("weights", torch.zeros(downsample, 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate neighboring frames at the configured temporal factor.

        Parameters
        ----------
        x : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Input of shape ``(batch_size, seq_len, num_channels)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Downsampled output of shape
            ``(batch_size, ceil(seq_len / downsample), num_channels)`` with the same
            dtype as ``x``.
        """

        downsample = self.weights.size(0)
        if downsample == 1:
            return x

        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                ZIPFORMER_DOWNSAMPLE_PLUGIN_NAME,
                (x, self.weights),
                {"factor": downsample, "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE},
                dtype=x.dtype,
                shape=(
                    x.shape[0],
                    (x.shape[1] + downsample - 1) // downsample,
                    x.shape[2],
                ),
                version=ONNX_OPSET_VERSION,
            )

        batch_size, sequence_length, channels = x.shape
        output_length = (sequence_length + downsample - 1) // downsample
        x = torch.nn.functional.pad(x, (0, 0, 0, downsample - 1), mode="replicate")
        x = x[:, : output_length * downsample]
        x = x.reshape(batch_size, output_length, downsample, channels)
        x = torch.sum(x * self.weights, dim=2)

        return x


class SimpleUpsample(torch.nn.Module):
    """Repeat downsampled frames and combine them with a bypass input."""

    def __init__(self, upsample: int) -> None:
        """Initialize the temporal upsampling factor.

        Parameters
        ----------
        upsample : int
            The module upsampling factor.
        """

        super().__init__()
        self.upsample = upsample

    def forward(
        self, x_early: torch.Tensor, x_later: torch.Tensor, bypass_scale: torch.Tensor
    ) -> torch.Tensor:
        """Upsample the later representation and apply the bypass scale.

        Parameters
        ----------
        x_early : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Bypass input with shape ``(batch_size, seq_len, num_channels)``.
        x_later : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Downsampled input with shape
            ``(batch_size, ceil(seq_len / upsample), num_channels)``.
        bypass_scale : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Per-channel interpolation scale with shape ``(num_channels,)``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Upsampled and combined output with the same shape and dtype as ``x_early``.
        """

        if torch.onnx.is_in_onnx_export():
            return torch.onnx.ops.symbolic(
                ZIPFORMER_UPSAMPLE_BYPASS_PLUGIN_NAME,
                (x_early, x_later, bypass_scale),
                {
                    "factor": self.upsample,
                    "plugin_namespace": TENSORRT_PLUGIN_NAMESPACE,
                },
                dtype=x_early.dtype,
                shape=x_early.shape,
                version=ONNX_OPSET_VERSION,
            )

        if self.upsample > 1:
            x_later = torch.repeat_interleave(x_later, self.upsample, dim=1)
            x_later = x_later[:, : x_early.size(1)]

        return x_early + (x_later - x_early) * bypass_scale
