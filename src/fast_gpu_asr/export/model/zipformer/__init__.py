#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Export-oriented PyTorch components for offline Zipformer models.

The package contains the Kaldi-compatible waveform frontend, convolutional
subsampling, Zipformer encoder stacks, attention and activation modules, and
the transducer predictor and joiner. The condensed graph accepts converted
Icefall checkpoints and exposes operators that lower to the native Zipformer
TensorRT plugins during engine construction.
"""
