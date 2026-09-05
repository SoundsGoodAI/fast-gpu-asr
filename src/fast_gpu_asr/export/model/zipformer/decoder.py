#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
"""Stateless Zipformer transducer predictor and joiner modules."""

import torch


class Decoder(torch.nn.Module):
    """Project stateless token contexts into the Zipformer joiner space."""

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        joiner_dim: int,
        context_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the stateless prediction network.

        Parameters
        ----------
        vocab_size : int
            Number of token IDs represented by the embedding table, including blank.
        decoder_dim : int
            Token embedding and convolution channel dimension.
        joiner_dim : int
            Output dimension of the predictor projection.
        context_size : int
            Number of previous token IDs used to predict the next token. A value of
            one represents a bigram predictor, two represents a trigram predictor,
            and ``n`` represents an ``(n + 1)``-gram predictor.
        dtype : torch.dtype
            Floating-point dtype used by the prediction network. Supported values
            are ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.
        """

        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(vocab_size, decoder_dim, dtype=dtype)
        self.conv = (
            torch.nn.Conv1d(
                decoder_dim,
                decoder_dim,
                context_size,
                groups=decoder_dim // 4,
                bias=False,
                dtype=dtype,
            )
            if context_size > 1
            else torch.nn.Identity()
        )
        self.decoder_proj = torch.nn.Linear(decoder_dim, joiner_dim, dtype=dtype)

    @torch.inference_mode()
    def make_context_lookup(self, chunk_size: int) -> torch.Tensor:
        """Precompute predictor vectors for every possible token context.

        Context values span ``-1`` through ``vocab_size - 1``. Negative one is
        the missing left context used at the beginning of an utterance. Rows are
        ordered as base-``vocab_size + 1`` numbers after shifting each context
        value by one, matching the runtime CUDA lookup.

        Parameters
        ----------
        chunk_size : int
            Number of contexts evaluated together while constructing the table.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Predictor table with shape
            ``((vocab_size + 1) ** context_size, joiner_dim)``.

        Raises
        ------
        ValueError
            Raised when ``chunk_size`` is not a positive integer.
        """

        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size}."
            )

        num_values = self.vocab_size + 1
        num_contexts = num_values**self.context_size
        lookup = torch.empty(
            num_contexts,
            self.decoder_proj.out_features,
            dtype=self.decoder_proj.weight.dtype,
            device=self.decoder_proj.weight.device,
        )
        for start in range(0, num_contexts, chunk_size):
            end = min(start + chunk_size, num_contexts)
            indexes = torch.arange(
                start, end, dtype=torch.int64, device=self.decoder_proj.weight.device
            )
            contexts = torch.empty(
                end - start,
                self.context_size,
                dtype=torch.int64,
                device=self.decoder_proj.weight.device,
            )
            for position in range(self.context_size - 1, -1, -1):
                contexts[:, position] = torch.remainder(indexes, num_values) - 1
                indexes = torch.floor_divide(indexes, num_values)
            lookup[start:end] = self(contexts)

        return lookup

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """Project token contexts into the joiner space.

        Parameters
        ----------
        decoder_input : torch.Tensor[torch.int32 | torch.int64]
            Token context with shape ``(num_hyps, context_size)`` containing the
            most recently decoded token IDs. TensorRT export uses ``torch.int32``;
            context lookup construction uses ``torch.int64``.

        Returns
        -------
        torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Predictor vectors with shape ``(num_hyps, joiner_dim)``.
        """

        embeddings = self.embedding(torch.clamp(decoder_input, min=0)) * (
            decoder_input.unsqueeze(2) >= 0
        )

        if self.context_size > 1:
            embeddings = embeddings.permute(0, 2, 1)
            embeddings = self.conv(embeddings)
            embeddings = embeddings.permute(0, 2, 1)

        features = embeddings[:, 0, :]
        if self.context_size > 1:
            features = torch.nn.functional.relu(features)

        return self.decoder_proj(features)


class Joiner(torch.nn.Module):
    """Convert predictor and encoder projections into token log probabilities."""

    def __init__(self, joiner_dim: int, vocab_size: int, dtype: torch.dtype) -> None:
        """Initialize the output projection.

        Parameters
        ----------
        joiner_dim : int
            Dimension shared by predictor and encoder projections.
        vocab_size : int
            Number of output tokens, including blank.
        dtype : torch.dtype
            Floating-point dtype used by the output projection. Supported values
            are ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.
        """

        super().__init__()

        self.output_proj = torch.nn.Linear(joiner_dim, vocab_size, dtype=dtype)

    def forward(
        self, decoder_out: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        """Combine predictor and encoder projections.

        Parameters
        ----------
        decoder_out : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Precomputed predictor vectors with shape ``(num_hyps, joiner_dim)``.
        encoder_out : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Encoder projections with shape ``(num_hyps, joiner_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Token log probabilities with shape ``(num_hyps, vocab_size)``.
        """

        logits = self.output_proj(torch.tanh(encoder_out + decoder_out))
        probs = torch.log_softmax(logits.to(torch.float32), dim=1)

        return probs
