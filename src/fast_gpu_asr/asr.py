#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Public batched offline TensorRT ASR pipeline."""

from pathlib import Path
from threading import Lock

import cupy as cp
import numpy as np
from omegaconf import OmegaConf

from .constants import (
    MODEL_CONFIG_FILE,
    MODEL_TYPE_ZIPFORMER,
    PARAKEET_DECODER_TENSORRT_FILE,
    PARAKEET_TENSORRT_FILE,
    TOKENIZER_FILE,
    ZIPFORMER_DECODER_TENSORRT_FILE,
    ZIPFORMER_TENSORRT_FILE,
)
from .decoder.parakeet_decoder import ParakeetModifiedBeamSearchDecoder
from .decoder.postprocessor import PostProcessor
from .decoder.zipformer_decoder import (
    CTCGreedyDecoder,
    ZipformerModifiedBeamSearchDecoder,
)
from .encoder.encoder import Encoder
from .utils import validate_model


class ASR:
    """Transcribe normalized mono audio with a packaged TensorRT model."""

    def __init__(
        self, model_dir: str | Path, device_id: int = 0, validate: bool = True
    ) -> None:
        """Initialize a Zipformer or Parakeet TensorRT pipeline.

        Parameters
        ----------
        model_dir : str | Path
            Directory produced by a Fast GPU ASR exporter.
        device_id : int, default=0
            CUDA device ordinal used by the encoder and decoder.
        validate : bool, default=True
            Whether to validate model metadata and runtime artifacts before
            initialization. Disable only for a bundle that has already been
            validated.
        """

        model_dir_path = Path(model_dir)
        model_config = OmegaConf.load(model_dir_path / MODEL_CONFIG_FILE)

        with cp.cuda.Device(device_id):
            if validate:
                validate_model(model_dir_path, model_config)

            encoder_params = model_config.audio_encoder_params
            decoder_params = model_config.decoder_params
            encoder_frame_shift_sec = (
                encoder_params.frame_shift_ms / 1000
            ) * encoder_params.subsampling_factor

            if model_config.model_type == MODEL_TYPE_ZIPFORMER:
                encoder_file = ZIPFORMER_TENSORRT_FILE
                right_padding_samples = encoder_params.right_padding_samples
                if model_config.decoder_type != "ctc_greedy_search":
                    context_size = decoder_params.context_size
                    vocab_size = model_config.vocab_size
            else:
                encoder_file = PARAKEET_TENSORRT_FILE
                right_padding_samples = 0
                tdt_durations = tuple(decoder_params.tdt_durations)
                max_symbols_per_timestep = decoder_params.max_symbols_per_timestep

            self.stream = cp.cuda.Stream(null=False, non_blocking=True, ptds=False)

            self.encoder = Encoder(
                model_dir_path / encoder_file,
                model_config.model_samplerate,
                device_id,
                self.stream,
                right_padding_samples,
            )

            if model_config.decoder_type == "ctc_greedy_search":
                self.decoder = CTCGreedyDecoder(
                    model_config.blank_id,
                    encoder_frame_shift_sec,
                    decoder_params.blank_penalty,
                    device_id,
                    self.stream,
                )
            elif model_config.model_type == MODEL_TYPE_ZIPFORMER:
                self.decoder = ZipformerModifiedBeamSearchDecoder(
                    model_dir_path / ZIPFORMER_DECODER_TENSORRT_FILE,
                    self.encoder.batch_size,
                    context_size,
                    vocab_size,
                    model_config.blank_id,
                    encoder_frame_shift_sec,
                    decoder_params.blank_penalty,
                    device_id,
                    self.stream,
                )
            else:
                self.decoder = ParakeetModifiedBeamSearchDecoder(
                    model_dir_path / PARAKEET_DECODER_TENSORRT_FILE,
                    self.encoder.batch_size,
                    model_config.blank_id,
                    tdt_durations,
                    max_symbols_per_timestep,
                    encoder_frame_shift_sec,
                    decoder_params.blank_penalty,
                    device_id,
                    self.stream,
                )

        self.postprocessor = PostProcessor(
            model_dir_path / TOKENIZER_FILE, model_config.model_samplerate
        )

        self.call_lock = Lock()

    def __call__(
        self, audios: list[np.typing.NDArray[np.float32]]
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Transcribe one batch.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            One-dimensional waveforms normalized to ``[-1.0, 1.0]`` and
            sampled at ``encoder.sample_rate``.

        Returns
        -------
        tuple[list[str], list[list[tuple[str, float, float]]]]
            Decoded texts and ``(word, start, end)`` tuples.
        """

        with self.call_lock:
            encoder_output, encoder_output_lengths = self.encoder(audios)
            token_ids, timestamps = self.decoder(encoder_output, encoder_output_lengths)
            texts, word_timestamps = self.postprocessor(audios, token_ids, timestamps)

        return texts, word_timestamps
