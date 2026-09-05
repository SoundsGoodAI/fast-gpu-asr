#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Modified from NeMo for batched TensorRT export; see NOTICE and LICENSE.
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
        dtype: torch.dtype,
    ) -> None:
        """Initialize the TDT prediction network and joiner.

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
        dtype : torch.dtype
            Floating-point dtype used by the prediction network and joiner.
            Supported values are ``torch.float32``, ``torch.float16``, and
            ``torch.bfloat16``.
        """

        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(
            vocab_size + 1, decoder_dim, padding_idx=vocab_size, dtype=dtype
        )
        self.lstm = torch.nn.LSTM(
            decoder_dim, decoder_dim, num_layers=pred_rnn_layers, dtype=dtype
        )
        self.decoder_proj = torch.nn.Linear(decoder_dim, joiner_dim, dtype=dtype)
        self.encoder_proj = torch.nn.Linear(encoder_dim, joiner_dim, dtype=dtype)
        self.output_proj = torch.nn.Linear(
            joiner_dim, vocab_size + 1 + num_extra_outputs, dtype=dtype
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the TDT prediction network and joiner for all active hypotheses.

        Parameters
        ----------
        encoder_output : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            Encoder output with shape ``(num_hyps, encoder_dim)``.
        targets : torch.Tensor[torch.int32]
            Token input with shape ``(num_hyps, 1)``.
        input_states_1 : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            LSTM hidden state with shape ``(pred_rnn_layers, num_hyps, decoder_dim)``.
        input_states_2 : torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16]
            LSTM cell state with shape ``(pred_rnn_layers, num_hyps, decoder_dim)``.

        Returns
        -------
        tuple[
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32],
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
            torch.Tensor[torch.float32 | torch.float16 | torch.bfloat16],
        ]
            A tuple containing:
            - ``torch.float32`` token and blank log probabilities with shape
              ``(num_hyps, vocab_size + 1)``. The final column is the blank.
            - ``torch.float32`` duration log probabilities with shape
              ``(num_hyps, num_extra_outputs)``.
            - Output LSTM hidden state with the same floating-point dtype as
              ``input_states_1`` and shape
              ``(pred_rnn_layers, num_hyps, decoder_dim)``.
            - Output LSTM cell state with the same floating-point dtype as
              ``input_states_2`` and shape
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
        token_log_probs = torch.log_softmax(
            joiner_out[:, : self.vocab_size + 1].to(torch.float32), dim=1
        )
        duration_log_probs = torch.log_softmax(
            joiner_out[:, self.vocab_size + 1 :].to(torch.float32), dim=1
        )

        return token_log_probs, duration_log_probs, output_states_1, output_states_2
