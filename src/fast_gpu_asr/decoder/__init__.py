#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""GPU-resident decoding for Zipformer and Parakeet encoder outputs.

This internal package implements shared CTC greedy decoding, Zipformer and
Parakeet modified transducer beam search, SentencePiece postprocessing,
and the CUDA kernels used by both decoder families. Decoder instances execute
on the encoder's CuPy stream and transfer only final token IDs and timestamps
to host memory.
"""
