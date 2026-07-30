#!/bin/env python3
# Copyright SoundsGoodAI 2026
"""TDT decoder and joiner modules used by Parakeet."""

import torch


class Decoder(torch.nn.Module):
    """Combined TDT prediction network and joiner."""

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        pred_rnn_layers: int,
        num_extra_outputs: int,
    ) -> None:
        """
        Decoder initialization.

        Parameters
        ----------
        vocab_size : int
            The tokenizer vocabulary size, excluding the RNNT blank token.
        encoder_dim : int
            The encoder hidden dimension.
        decoder_dim : int
            The prediction network hidden dimension.
        joiner_dim : int
            The joiner network hidden dimension.
        pred_rnn_layers : int
            The number of prediction network LSTM layers.
        num_extra_outputs : int
            The number of TDT duration outputs after the vocabulary and blank outputs.
        """

        super().__init__()

        device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(
            vocab_size + 1, decoder_dim, padding_idx=vocab_size, device=device
        )
        self.lstm = torch.nn.LSTM(
            decoder_dim, decoder_dim, num_layers=pred_rnn_layers, device=device
        )
        self.decoder_proj = torch.nn.Linear(decoder_dim, joiner_dim, device=device)
        self.encoder_proj = torch.nn.Linear(encoder_dim, joiner_dim, device=device)
        self.output_proj = torch.nn.Linear(
            joiner_dim, vocab_size + 1 + num_extra_outputs, device=device
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the TDT prediction network and joiner for all active hypotheses.

        Parameters
        ----------
        encoder_output : torch.Tensor[torch.float32]
            Encoder output with shape ``(num_hyps, encoder_dim)``.
        targets : torch.Tensor[torch.int32]
            Token input with shape ``(num_hyps, 1)``.
        input_states_1 : torch.Tensor[torch.float32]
            LSTM hidden state with shape ``(pred_rnn_layers, num_hyps, decoder_dim)``.
        input_states_2 : torch.Tensor[torch.float32]
            LSTM cell state with shape ``(pred_rnn_layers, num_hyps, decoder_dim)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
        ]
            A tuple of four float tensors:
            - token and blank log probabilities of shape ``(num_hyps, vocab_size + 1)``.
            - duration log probabilities of shape ``(num_hyps, num_extra_outputs)``.
            - output LSTM hidden state with shape
              ``(pred_rnn_layers, num_hyps, decoder_dim)``.
            - output LSTM cell state with shape
              ``(pred_rnn_layers, num_hyps, decoder_dim)``.
        """

        decoder_out = self.embedding(targets).permute(1, 0, 2)
        decoder_out, (output_states_1, output_states_2) = self.lstm(
            decoder_out, (input_states_1, input_states_2)
        )
        decoder_out = self.decoder_proj(decoder_out[0])
        encoder_out = self.encoder_proj(encoder_output)
        joiner_out = self.output_proj(
            torch.nn.functional.relu(encoder_out + decoder_out)
        )
        token_log_probs = torch.log_softmax(joiner_out[:, : self.vocab_size + 1], dim=1)
        duration_log_probs = torch.log_softmax(
            joiner_out[:, self.vocab_size + 1 :], dim=1
        )

        return token_log_probs, duration_log_probs, output_states_1, output_states_2
