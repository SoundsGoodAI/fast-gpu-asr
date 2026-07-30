#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""Stateless transducer combined decoder and joiner modules used by Zipformer."""

import torch


class Decoder(torch.nn.Module):
    """
    A helper module to combine decoder, decoder projection, and joiner inference
    together.
    """

    def __init__(
        self, vocab_size: int, decoder_dim: int, joiner_dim: int, context_size: int
    ) -> None:
        """
        Decoder initialization.

        Parameters
        ----------
        vocab_size : int
            The number of tokens or modeling units, including blank.
        decoder_dim : int
            A dimension of the decoder embeddings, and the decoder output.
        joiner_dim : int
            Input joiner dimension.
        context_size : int
            A number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """

        super().__init__()

        device = torch.device("cpu")

        self.embedding = torch.nn.Embedding(vocab_size, decoder_dim, device=device)
        if context_size < 1:
            raise ValueError(
                "RNN-T decoder context size should be an integer greater "
                f"than or equal to 1, but got {context_size}.",
            )
        self.context_size = context_size

        self.conv = torch.nn.Conv1d(
            decoder_dim,
            decoder_dim,
            context_size,
            groups=decoder_dim // 4,
            bias=False,
            device=device,
        )

        self.decoder_proj = torch.nn.Linear(decoder_dim, joiner_dim, device=device)
        self.output_proj = torch.nn.Linear(joiner_dim, vocab_size, device=device)

    def forward(
        self, decoder_input: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the stateless decoder and joiner for all active hypotheses.

        The output contains token log probabilities for every active
        hypothesis. Runtime decoders apply blank penalties, add hypothesis
        scores, and sort token candidates.

        Parameters
        ----------
        decoder_input : torch.Tensor[torch.int32]
            Token context with shape ``(num_hyps, context_size)`` containing the
            most recently decoded token IDs.
        encoder_out : torch.Tensor[torch.float32]
            Encoder projections with shape ``(num_hyps, joiner_dim)``.

        Returns
        -------
        torch.Tensor[torch.float32]
            Token log probabilities with shape ``(num_hyps, vocab_size)``.
        """

        # Negative IDs represent missing left context at the start of an utterance.
        embeddings = self.embedding(torch.clamp(decoder_input, min=0)) * (
            decoder_input >= 0
        ).unsqueeze(2)

        if self.context_size > 1:
            embeddings = embeddings.permute(0, 2, 1)
            embeddings = self.conv(embeddings)
            embeddings = embeddings.permute(0, 2, 1)
            embeddings = torch.nn.functional.relu(embeddings)

        decoder_out = self.decoder_proj(embeddings)

        logits = self.output_proj(torch.tanh(encoder_out + decoder_out[:, 0, :]))
        probs = torch.log_softmax(logits, dim=1)

        return probs
