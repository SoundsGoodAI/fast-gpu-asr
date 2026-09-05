#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for Zipformer bundle export, metadata conversion, and validation."""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import cast
from unittest.mock import Mock, call

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import fast_gpu_asr.export.export_zipformer as zipformer_exporter
from fast_gpu_asr.constants import INT32_MAX
from fast_gpu_asr.export.export_utils import validate_zipformer
from fast_gpu_asr.export.export_zipformer import (
    adjust_state_dict,
    export_model_to_onnx,
    export_zipformer,
    get_subsampling_batch_partitions,
    make_model,
    make_runtime_config,
    parse_args,
)


class FakeTokenizer:
    """Expose a Zipformer tokenizer with a trailing unknown token."""

    def __init__(self, model_file: str) -> None:
        """Record the tokenizer path supplied by the exporter.

        Parameters
        ----------
        model_file : str
            Placeholder tokenizer path; its contents are not parsed.
        """

        self.model_file = model_file

    def vocab_size(self) -> int:
        """Return the vocabulary size including the trailing unknown token.

        Returns
        -------
        int
            Size five, of which four tokens remain after export trimming.
        """

        return 5

    def unk_id(self) -> int:
        """Return the trailing unknown-token ID excluded from decoding.

        Returns
        -------
        int
            Unknown-token ID four.
        """

        return 4

    def piece_to_id(self, piece: str) -> int:
        """Resolve the blank piece used during export validation.

        Parameters
        ----------
        piece : str
            Requested piece, expected to be ``<blk>``.

        Returns
        -------
        int
            Blank-token ID zero.
        """

        assert piece == "<blk>"
        return 0

    def id_to_piece(self, token_id: int) -> str:
        """Return the blank piece or a distinct ordinary token string.

        Parameters
        ----------
        token_id : int
            ID to render as a placeholder piece.

        Returns
        -------
        str
            ``<blk>`` for zero, or an ID-specific ordinary piece.
        """

        return "<blk>" if token_id == 0 else str(token_id)


def make_model_config() -> DictConfig:
    """Return a compact valid source configuration for Zipformer export tests.

    Returns
    -------
    DictConfig
        Fresh six-stack configuration that each test can mutate independently.
    """

    return OmegaConf.create(
        {
            "model_type": "zipformer-transducer",
            "library_name": "k2",
            "min_encoder_input_frames": 9,
            "decoding": {"beam_size": 6},
            "feature_opts": {
                "frame_opts": {
                    "samp_freq": 16000,
                    "frame_shift_ms": 10,
                    "frame_length_ms": 25,
                    "dither": 0.0,
                    "preemph_coeff": 0.97,
                    "window_type": "povey",
                    "blackman_coeff": 0.42,
                    "snip_edges": False,
                },
                "mel_opts": {
                    "num_bins": 80,
                    "low_freq": 20,
                    "high_freq": 7600,
                },
            },
            "model_params": {
                "feature_dim": 80,
                "subsampling_factor": 4,
                "num_encoder_layers": "2,2,4,5,4,2",
                "downsampling_factor": "1,2,4,8,4,2",
                "feedforward_dim": "512,1024,2048,3072,2048,1024",
                "num_heads": "4,4,4,8,4,4",
                "encoder_dim": "192,384,768,1024,768,384",
                "query_head_dim": "32",
                "value_head_dim": "12",
                "pos_head_dim": "4",
                "pos_dim": 48,
                "cnn_module_kernel": "31,31,15,15,15,31",
                "decoder_dim": 512,
                "joiner_dim": 512,
                "context_size": 2,
                "causal": False,
                "use_transducer": True,
                "use_ctc": True,
                "use_attention_decoder": False,
            },
        },
    )


def make_state_dict(
    decoder_type: str = "transducer_modified_beam_search",
    output_dim: int = 512,
    input_dim: int = 1024,
) -> OrderedDict[str, torch.Tensor]:
    """Return the selected minimal decoder head for source validation tests.

    Parameters
    ----------
    decoder_type : str
        Decoder route determining the projection checkpoint prefix.
    output_dim : int
        Number of projection rows, allowing intentional mismatch cases.
    input_dim : int
        Number of projection columns.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Independent zero-filled FP32 projection weights and bias.
    """

    projection_prefix = (
        "ctc_output.1" if decoder_type == "ctc_greedy_search" else "projection_output"
    )
    return OrderedDict(
        {
            f"{projection_prefix}.weight": torch.zeros(output_dim, input_dim),
            f"{projection_prefix}.bias": torch.zeros(output_dim),
        }
    )


def make_export_args(
    decoder_type: str = "transducer_modified_beam_search",
) -> argparse.Namespace:
    """Return valid Zipformer exporter arguments suitable for local mutation.

    Parameters
    ----------
    decoder_type : str
        Requested decoder route.

    Returns
    -------
    argparse.Namespace
        Fresh batch-one FP32 arguments with placeholder paths. Greedy callers
        set the effective beam themselves when bypassing the export entry point.
    """

    return argparse.Namespace(
        batch_size=1,
        beam=6,
        debug=False,
        decoder_type=decoder_type,
        decoder_precision="fp32",
        encoder_precision="fp32",
        min_audio_seconds=0.5,
        opt_audio_seconds=15.0,
        max_audio_seconds=120.0,
        model_path=Path("model.pt"),
        optimization_level=5,
        output_dir=Path("output"),
    )


def write_zipformer_sources(model_dir: Path) -> Path:
    """Write a valid config and placeholder checkpoint and tokenizer files.

    Parameters
    ----------
    model_dir : Path
        Source directory to create, including missing parents.

    Returns
    -------
    Path
        Checkpoint path. Only the YAML config is parseable; loading the other
        two files requires the test's checkpoint and tokenizer replacements.
    """

    model_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(make_model_config(), model_dir / "config.yaml")
    (model_dir / "bpe.model").write_bytes(b"tokenizer")
    model_path = model_dir / "model.pt"
    model_path.write_bytes(b"checkpoint")
    return model_path


@pytest.mark.parametrize("missing_name", ("config.yaml", "model.pt", "bpe.model"))
def test_export_zipformer_rejects_missing_source_before_replacing_output(
    tmp_path: Path,
    missing_name: str,
) -> None:
    source_dir = tmp_path / "source"
    model_path = write_zipformer_sources(source_dir)
    missing_path = source_dir / missing_name
    missing_path.unlink()
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel_path = output_dir / "existing.trt"
    sentinel_path.write_bytes(b"existing")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_path))):
        export_zipformer(args)

    assert sentinel_path.read_bytes() == b"existing"


@pytest.mark.parametrize("source_location", ("same-directory", "nested", "symlink"))
def test_export_zipformer_rejects_destination_containing_sources(
    tmp_path: Path, source_location: str
) -> None:
    args = make_export_args()
    args.output_dir = tmp_path / "bundle"
    source_dir = args.output_dir
    if source_location != "same-directory":
        source_dir /= "nested/source"
    args.model_path = write_zipformer_sources(source_dir)
    if source_location == "symlink":
        alias = tmp_path / "source-alias"
        alias.symlink_to(source_dir, target_is_directory=True)
        args.model_path = alias / "model.pt"

    with pytest.raises(ValueError, match="contains required source file"):
        export_zipformer(args)

    assert args.model_path.read_bytes() == b"checkpoint"
    assert (source_dir / "config.yaml").is_file()
    assert (source_dir / "bpe.model").read_bytes() == b"tokenizer"


def test_export_zipformer_replaces_output_before_loading_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = write_zipformer_sources(tmp_path / "source")
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel_path = output_dir / "old.trt"
    sentinel_path.write_bytes(b"known-good")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    monkeypatch.setattr(
        OmegaConf, "load", Mock(side_effect=RuntimeError("config load failed"))
    )

    with pytest.raises(RuntimeError, match="config load failed"):
        export_zipformer(args)

    assert list(output_dir.iterdir()) == []
    assert model_path.read_bytes() == b"checkpoint"


@pytest.mark.parametrize(
    "decoder_type",
    ("ctc_greedy_search", "transducer_greedy_search"),
)
@pytest.mark.parametrize("initial_beam", (1, 6))
def test_export_zipformer_forces_greedy_beam(
    decoder_type: str,
    initial_beam: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model_path = write_zipformer_sources(tmp_path / "source")
    args = make_export_args(decoder_type)
    args.model_path = model_path
    args.output_dir = tmp_path / "bundle"
    args.beam = initial_beam

    monkeypatch.setattr(
        OmegaConf, "load", Mock(side_effect=RuntimeError("stop export"))
    )

    with (
        caplog.at_level("WARNING", logger=zipformer_exporter.__name__),
        pytest.raises(RuntimeError, match="stop export"),
    ):
        export_zipformer(args)

    assert args.beam == 1
    assert ("Overriding beam" in caplog.text) == (initial_beam != 1)


def test_export_zipformer_retains_nontrailing_unknown_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = make_export_args()
    args.model_path = write_zipformer_sources(tmp_path / "source")
    args.output_dir = tmp_path / "bundle"
    monkeypatch.setattr(FakeTokenizer, "unk_id", lambda self: 1)
    monkeypatch.setattr(zipformer_exporter.spm, "SentencePieceProcessor", FakeTokenizer)
    monkeypatch.setattr(torch, "load", Mock(return_value={"model": {}}))
    monkeypatch.setattr(zipformer_exporter, "adjust_state_dict", lambda state: state)
    validate = Mock(side_effect=RuntimeError("stop after validation"))
    monkeypatch.setattr(zipformer_exporter, "validate_zipformer", validate)

    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="stop after"):
        export_zipformer(args)

    validate.assert_called_once()
    assert validate.call_args.args[2] == 5
    assert "retaining all entries" in caplog.text


@pytest.mark.parametrize("debug", (False, True))
@pytest.mark.parametrize(
    ("decoder_type", "subsampling_partitions"),
    (
        ("ctc_greedy_search", 1),
        ("transducer_greedy_search", 2),
        ("transducer_modified_beam_search", 1),
    ),
)
def test_export_zipformer_validates_exact_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
    decoder_type: str,
    subsampling_partitions: int,
) -> None:
    args = make_export_args(decoder_type)
    args.model_path = write_zipformer_sources(tmp_path / "source")
    args.output_dir = tmp_path / "nested/bundle"
    args.batch_size = 3
    args.beam = 4
    args.debug = debug
    args.encoder_precision = "fp16"
    args.decoder_precision = "bf16"
    args.max_audio_seconds = 120.006
    source_config = make_model_config()
    use_ctc = decoder_type == "ctc_greedy_search"
    state = make_state_dict(decoder_type, output_dim=4 if use_ctc else 512)
    state["subsampling.conv3.weight"] = torch.zeros(128, 1, 3, 3)
    checkpoint = {"source": torch.ones(1)}
    torch.save({"model": checkpoint}, args.model_path)
    checkpoint_bytes = args.model_path.read_bytes()
    encoder = torch.nn.Identity()
    decoder = None if use_ctc else torch.nn.Identity()
    joiner = None if use_ctc else torch.nn.Identity()

    tokenizer = Mock(side_effect=FakeTokenizer)
    load_checkpoint = Mock(wraps=torch.load)
    adjust = Mock(return_value=state)
    validate_source = Mock(wraps=validate_zipformer)
    partitions = Mock(return_value=subsampling_partitions)

    def make_export_model(*args):
        """Require source validation before returning placeholder model modules.

        Parameters
        ----------
        *args : tuple
            Model-construction arguments recorded by the surrounding Mock.

        Returns
        -------
        tuple
            Encoder, predictor, and joiner placeholders for the selected route.
        """

        validate_source.assert_called_once()
        return encoder, decoder, joiner

    make = Mock(side_effect=make_export_model)
    monkeypatch.setattr(zipformer_exporter.spm, "SentencePieceProcessor", tokenizer)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(zipformer_exporter, "adjust_state_dict", adjust)
    monkeypatch.setattr(zipformer_exporter, "validate_zipformer", validate_source)
    monkeypatch.setattr(
        zipformer_exporter, "get_subsampling_batch_partitions", partitions
    )
    monkeypatch.setattr(zipformer_exporter, "make_model", make)

    loaded_configs = []
    real_load = OmegaConf.load

    def load_config(path):
        """Record loaded configs to verify validation uses the published copy.

        Parameters
        ----------
        path : Path
            YAML file delegated to the original OmegaConf loader.

        Returns
        -------
        DictConfig
            Loaded configuration, also appended to the test's call history.
        """

        config = real_load(path)
        loaded_configs.append((Path(path), config))
        return config

    monkeypatch.setattr(OmegaConf, "load", load_config)

    def export_onnx(encoder, decoder, joiner, config, export_args):
        """Write placeholder graph and cache artifacts for the selected decoder.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder placeholder recorded by the surrounding Mock.
        decoder : torch.nn.Module | None
            Predictor placeholder, absent for CTC.
        joiner : torch.nn.Module | None
            Joiner placeholder, absent for CTC.
        config : DictConfig
            Source configuration recorded by the surrounding Mock.
        export_args : argparse.Namespace
            Export arguments recorded by the surrounding Mock.

        Returns
        -------
        tuple[Path, Path | None]
            Encoder and optional decoder paths. Contents are sentinel bytes,
            not valid graphs, so this test also substitutes graph cleanup.
        """

        encoder_path = args.output_dir / "zipformer.onnx"
        encoder_path.write_bytes(b"encoder")
        decoder_path = None
        if not use_ctc:
            decoder_path = args.output_dir / "decoder.onnx"
            decoder_path.write_bytes(b"decoder")
            (args.output_dir / "decoder_contexts.pt").write_bytes(b"contexts")
        return encoder_path, decoder_path

    def build_engine(onnx_path, engine_path, profiles, optimization_level):
        """Require an exported graph before writing a placeholder engine.

        Parameters
        ----------
        onnx_path : Path
            Existing placeholder graph.
        engine_path : Path
            Destination for engine sentinel bytes.
        profiles : dict
            Optimization profiles recorded by the surrounding Mock.
        optimization_level : int
            Builder level recorded by the surrounding Mock.
        """

        assert onnx_path.is_file()
        engine_path.write_bytes(b"engine")

    expected_artifacts = {"bpe.model", "model_config.yaml", "zipformer.trt"}
    if not use_ctc:
        expected_artifacts.update({"decoder_contexts.pt", "decoder.trt"})
    if debug:
        expected_artifacts.add("zipformer.onnx")
        if not use_ctc:
            expected_artifacts.add("decoder.onnx")

    def validate_bundle(model_dir, config):
        """Check the published config and complete decoder-specific artifact set.

        Parameters
        ----------
        model_dir : Path
            Published directory to inspect.
        config : DictConfig
            Configuration expected to be the last copy loaded from disk.
        """

        assert model_dir == args.output_dir
        assert config is loaded_configs[-1][1]
        assert {path.name for path in model_dir.iterdir()} == expected_artifacts
        assert all(path.is_file() for path in model_dir.iterdir())
        assert (model_dir / "bpe.model").read_bytes() == b"tokenizer"

    build = Mock(side_effect=build_engine)
    export = Mock(side_effect=export_onnx)
    validate = Mock(side_effect=validate_bundle)
    monkeypatch.setattr(zipformer_exporter, "export_model_to_onnx", export)
    monkeypatch.setattr(zipformer_exporter, "build_tensorrt_engine", build)
    monkeypatch.setattr(zipformer_exporter, "remove_onnx_artifacts", Path.unlink)
    monkeypatch.setattr(zipformer_exporter, "validate_model", validate)

    export_zipformer(args)

    tokenizer.assert_called_once_with(
        model_file=str(args.model_path.parent / "bpe.model")
    )
    load_checkpoint.assert_called_once_with(
        args.model_path, map_location=torch.device("cpu"), weights_only=True
    )
    adjust.assert_called_once()
    torch.testing.assert_close(adjust.call_args.args[0], checkpoint)
    validate_source.assert_called_once_with(source_config, state, 4, args)
    partitions.assert_called_once_with(source_config, 128, args)
    export.assert_called_once_with(encoder, decoder, joiner, source_config, args)
    make.assert_called_once_with(
        source_config,
        state,
        4,
        6001,
        decoder_type,
        subsampling_partitions,
        torch.float16,
        torch.bfloat16,
    )
    assert [path for path, _ in loaded_configs] == [
        args.model_path.parent / "config.yaml",
        args.output_dir / "model_config.yaml",
    ]
    published = loaded_configs[-1][1]
    assert published == make_runtime_config(source_config, 4, 0, 6001, args)
    assert args.beam == (4 if decoder_type == "transducer_modified_beam_search" else 1)
    validate.assert_called_once_with(args.output_dir, published)
    expected_builds = [
        call(
            args.output_dir / "zipformer.onnx",
            args.output_dir / "zipformer.trt",
            {"audio": ((3, 8200), (3, 240200), (3, 1920296))},
            args.optimization_level,
        )
    ]
    if not use_ctc:
        expected_builds.append(
            call(
                args.output_dir / "decoder.onnx",
                args.output_dir / "decoder.trt",
                {},
                args.optimization_level,
            )
        )
    assert build.call_args_list == expected_builds
    assert args.model_path.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize(
    ("decoder_type", "use_ctc"),
    (
        ("transducer_modified_beam_search", False),
        ("ctc_greedy_search", True),
    ),
)
def test_make_model_wires_configuration_and_state_dicts(
    monkeypatch: pytest.MonkeyPatch,
    decoder_type: str,
    use_ctc: bool,
) -> None:
    constructors = {
        name: Mock(return_value=Mock(spec=("load_state_dict", "eval")))
        for name in ("Zipformer2", "Decoder", "Joiner")
    }
    for name, constructor in constructors.items():
        monkeypatch.setattr(zipformer_exporter, name, constructor)

    bypass_scales = [torch.full((2,), float(index)) for index in range(1, 7)]
    tensors = {
        "subsampling.out.weight": torch.zeros(96, 4),
        "subsampling.conv1.weight": torch.zeros(8, 1, 3, 3),
        "subsampling.conv2.weight": torch.zeros(16, 8, 3, 3),
        "subsampling.conv3.weight": torch.zeros(32, 16, 3, 3),
        "encoder_1.layers.0.weight": torch.ones(2),
        "downsample_output.weights": torch.ones(2),
        "projection_output.weight": torch.zeros(512, 1024),
        "projection_output.bias": torch.zeros(512),
        "ctc_output.1.weight": torch.zeros(7, 1024),
        "ctc_output.1.bias": torch.zeros(7),
        "decoder.embedding.weight": torch.ones(7, 512),
        "decoder.conv.weight": torch.ones(512, 4, 2),
        "joiner.decoder_proj.weight": torch.ones(512, 512),
        "joiner.output_linear.weight": torch.ones(7, 512),
        "joiner.output_linear.bias": torch.ones(7),
        "ignored.weight": torch.ones(1),
    }
    state_dict = OrderedDict(
        (f"encoder_{index}.bypass_scale", bypass_scales[index - 1])
        for index in range(1, 7)
    )
    state_dict.update(tensors)

    encoder, decoder, joiner = make_model(
        make_model_config(),
        state_dict,
        vocab_size=7,
        pos_emb_max_len=6000,
        decoder_type=decoder_type,
        subsampling_batch_partitions=3,
        encoder_dtype=torch.float16,
        decoder_dtype=torch.bfloat16,
    )

    expected_encoder_kwargs = {
        "bypass_scales": bypass_scales,
        "samp_freq": 16_000,
        "frame_shift_ms": 10,
        "frame_length_ms": 25,
        "feature_dim": 80,
        "preemph": 0.97,
        "low_freq": 20,
        "high_freq": 7600,
        "min_frames": 9,
        "subsample_output_dim": 96,
        "subsample_layer1_channels": 8,
        "subsample_layer2_channels": 16,
        "subsample_layer3_channels": 32,
        "subsampling_batch_partitions": 3,
        "encoder_dims": [192, 384, 768, 1024, 768, 384],
        "num_encoder_layers": [2, 2, 4, 5, 4, 2],
        "downsampling_factors": [1, 2, 4, 8, 4, 2],
        "num_heads": [4, 4, 4, 8, 4, 4],
        "feedforward_dims": [512, 1024, 2048, 3072, 2048, 1024],
        "cnn_module_kernels": [31, 31, 15, 15, 15, 31],
        "query_head_dim": 32,
        "pos_head_dim": 4,
        "value_head_dim": 12,
        "pos_dim": 48,
        "pos_max_len": 6000,
        "output_dim": 7 if use_ctc else 512,
        "use_ctc": use_ctc,
        "dtype": torch.float16,
    }
    constructors["Zipformer2"].assert_called_once_with(**expected_encoder_kwargs)
    assert encoder is constructors["Zipformer2"].return_value
    encoder_keys = {
        *(f"encoder_{index}.bypass_scale" for index in range(1, 7)),
        "encoder_1.layers.0.weight",
        "subsampling.out.weight",
        "subsampling.conv1.weight",
        "subsampling.conv2.weight",
        "subsampling.conv3.weight",
        "downsample_output.weights",
    }
    expected_encoder_state = {key: state_dict[key] for key in encoder_keys}
    projection_prefix = "ctc_output.1" if use_ctc else "projection_output"
    expected_encoder_state.update(
        {
            f"projection_output.{suffix}": state_dict[f"{projection_prefix}.{suffix}"]
            for suffix in ("weight", "bias")
        }
    )
    encoder.load_state_dict.assert_called_once_with(expected_encoder_state, strict=True)
    encoder.eval.assert_called_once_with()

    if use_ctc:
        assert decoder is joiner is None
        constructors["Decoder"].assert_not_called()
        constructors["Joiner"].assert_not_called()
    else:
        constructors["Decoder"].assert_called_once_with(
            vocab_size=7,
            decoder_dim=512,
            joiner_dim=512,
            context_size=2,
            dtype=torch.bfloat16,
        )
        constructors["Joiner"].assert_called_once_with(512, 7, dtype=torch.bfloat16)
        assert decoder is constructors["Decoder"].return_value
        assert joiner is constructors["Joiner"].return_value
        decoder.load_state_dict.assert_called_once_with(
            {
                "embedding.weight": tensors["decoder.embedding.weight"],
                "conv.weight": tensors["decoder.conv.weight"],
                "decoder_proj.weight": tensors["joiner.decoder_proj.weight"],
            },
            strict=True,
        )
        joiner.load_state_dict.assert_called_once_with(
            {
                "output_proj.weight": tensors["joiner.output_linear.weight"],
                "output_proj.bias": tensors["joiner.output_linear.bias"],
            },
            strict=True,
        )
        decoder.eval.assert_called_once_with()
        joiner.eval.assert_called_once_with()


@pytest.mark.parametrize("overrides", (False, True))
def test_parse_args(monkeypatch: pytest.MonkeyPatch, overrides: bool) -> None:
    argv = [
        "fast-gpu-asr-export-zipformer",
        "--model-path",
        "model.pt",
        "--output-dir",
        "output",
        "--batch-size",
        "1",
        "--decoder-type",
        "ctc_greedy_search",
        "--beam",
        "1",
        "--min-audio-seconds",
        "0.1",
        "--opt-audio-seconds",
        "15",
        "--max-audio-seconds",
        "40",
    ]
    expected = vars(make_export_args("ctc_greedy_search"))
    expected.update(beam=1, min_audio_seconds=0.1, max_audio_seconds=40.0)
    if overrides:
        argv += [
            "--batch-size",
            "256",
            "--decoder-type",
            "transducer_modified_beam_search",
            "--beam",
            "6",
            "--encoder-precision",
            "bf16",
            "--decoder-precision",
            "fp16",
            "--opt-audio-seconds",
            "8",
            "--optimization-level",
            "3",
            "--debug",
        ]
        expected.update(
            batch_size=256,
            decoder_type="transducer_modified_beam_search",
            beam=6,
            encoder_precision="bf16",
            decoder_precision="fp16",
            opt_audio_seconds=8.0,
            optimization_level=3,
            debug=True,
        )
    monkeypatch.setattr(sys, "argv", argv)

    assert vars(parse_args()) == expected


def test_main_configures_logging_and_runs_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_export_args()
    parse = Mock(return_value=args)
    export = Mock()
    configure_logging = Mock()
    monkeypatch.setattr(zipformer_exporter, "parse_args", parse)
    monkeypatch.setattr(zipformer_exporter, "export_zipformer", export)
    monkeypatch.setattr(zipformer_exporter.logging, "basicConfig", configure_logging)

    zipformer_exporter.main()

    parse.assert_called_once_with()
    export.assert_called_once_with(args)
    configure_logging.assert_called_once_with(
        format="%(asctime)s %(levelname)s %(message)s",
        level=zipformer_exporter.logging.INFO,
    )


@pytest.mark.parametrize(
    "decoder_type",
    (
        "ctc_greedy_search",
        "transducer_greedy_search",
        "transducer_modified_beam_search",
    ),
)
def test_validate_zipformer_accepts_supported_decoder(decoder_type: str) -> None:
    validate_zipformer(
        make_model_config(),
        make_state_dict(decoder_type),
        512,
        make_export_args(decoder_type),
    )


def test_validate_zipformer_rejects_unsupported_decoder() -> None:
    args = make_export_args()
    args.decoder_type = "unsupported_search"

    with pytest.raises(ValueError, match="Zipformer export supports only"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("decoder_type", "config_field", "message"),
    (
        ("ctc_greedy_search", "use_ctc", "does not enable the CTC head"),
        (
            "transducer_modified_beam_search",
            "use_transducer",
            "does not enable the transducer head",
        ),
    ),
)
def test_validate_zipformer_rejects_disabled_decoder_head(
    decoder_type: str,
    config_field: str,
    message: str,
) -> None:
    model_config = make_model_config()
    model_config.model_params[config_field] = False

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_zipformer(
            model_config,
            make_state_dict(decoder_type),
            512,
            make_export_args(decoder_type),
        )


@pytest.mark.parametrize("config_field", ("use_ctc", "use_transducer"))
def test_validate_zipformer_rejects_nonboolean_decoder_flags(
    config_field: str,
) -> None:
    model_config = make_model_config()
    model_config.model_params[config_field] = 1

    with pytest.raises(ValueError, match=rf"model_params.{config_field}.*boolean"):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_params.subsampling_factor", 4.0),
        ("model_params.causal", 0),
        ("model_params.use_attention_decoder", 0),
        ("feature_opts.frame_opts.dither", False),
        ("feature_opts.frame_opts.snip_edges", 0),
    ),
)
def test_validate_zipformer_accepts_equivalent_fixed_values(
    field: str,
    value: bool | float | int,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("batch_size", 0, "batch_size must be positive"),
        ("batch_size", 1.5, "batch_size must be positive"),
        ("beam", 0, "beam must be a positive integer between 1 and 512"),
        ("beam", 1.5, "beam must be a positive integer between 1 and 512"),
        ("beam", 513, "beam must be a positive integer between 1 and 512"),
        (
            "min_audio_seconds",
            0.0,
            "0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds",
        ),
        (
            "opt_audio_seconds",
            0.25,
            "0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds",
        ),
        (
            "max_audio_seconds",
            10.0,
            "0 < min_audio_seconds <= opt_audio_seconds <= max_audio_seconds",
        ),
    ),
)
def test_validate_zipformer_rejects_invalid_export_profile(
    field: str,
    value: float | int,
    message: str,
) -> None:
    args = make_export_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("encoder_precision", "fp64"),
        ("encoder_precision", 16),
        ("decoder_precision", "fp64"),
        ("decoder_precision", None),
    ),
)
def test_validate_zipformer_rejects_unsupported_precision(
    field: str,
    value: str | int | None,
) -> None:
    args = make_export_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=rf"{field} must be one of"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


def test_validate_zipformer_rejects_transducer_projection_mismatch() -> None:
    with pytest.raises(RuntimeError, match="transducer projection contains 256"):
        validate_zipformer(
            make_model_config(),
            make_state_dict(output_dim=256),
            512,
            make_export_args(),
        )


@pytest.mark.parametrize(
    "decoder_type", ("ctc_greedy_search", "transducer_modified_beam_search")
)
@pytest.mark.parametrize("input_dim", (1023, 1025))
def test_validate_zipformer_rejects_projection_input_mismatch(
    decoder_type: str,
    input_dim: int,
) -> None:
    with pytest.raises(RuntimeError, match=rf"accepts {input_dim} input features"):
        validate_zipformer(
            make_model_config(),
            make_state_dict(decoder_type, input_dim=input_dim),
            512,
            make_export_args(decoder_type),
        )


@pytest.mark.parametrize("decoder_dim", (3, 9))
def test_validate_zipformer_rejects_incompatible_predictor_dimension(
    decoder_dim: int,
) -> None:
    model_config = make_model_config()
    model_config.model_params.decoder_dim = decoder_dim

    with pytest.raises(ValueError, match="grouped-convolution count"):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


def test_validate_zipformer_ctc_ignores_transducer_predictor_limits() -> None:
    model_config = make_model_config()
    model_config.model_params.decoder_dim = 3
    model_config.model_params.context_size = 3

    validate_zipformer(
        model_config,
        make_state_dict("ctc_greedy_search"),
        512,
        make_export_args("ctc_greedy_search"),
    )


@pytest.mark.parametrize("vocab_size", (0, -1, 1.5))
def test_validate_zipformer_rejects_invalid_vocabulary(
    vocab_size: int | float,
) -> None:
    with pytest.raises(ValueError, match="vocab_size.*positive integer"):
        validate_zipformer(
            make_model_config(),
            make_state_dict(),
            vocab_size,  # type: ignore[arg-type]
            make_export_args(),
        )


@pytest.mark.parametrize(
    "field",
    (
        "min_encoder_input_frames",
        "feature_opts.mel_opts.num_bins",
        "model_params.feature_dim",
        "model_params.pos_dim",
        "model_params.decoder_dim",
        "model_params.joiner_dim",
        "model_params.context_size",
        "decoding.beam_size",
    ),
)
def test_validate_zipformer_rejects_nonpositive_integer_config(
    field: str,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, 0)

    with pytest.raises(
        ValueError,
        match=rf"Expected {re.escape(field)} to be a positive integer",
    ):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("query_head_dim", 32.5),
        ("query_head_dim", "32.5"),
        ("value_head_dim", "32.5"),
        ("pos_head_dim", "32.5"),
    ),
)
def test_validate_zipformer_rejects_nonintegral_head_dimensions(
    field: str,
    value: float | str,
) -> None:
    model_config = make_model_config()
    model_config.model_params[field] = value

    with pytest.raises(
        ValueError, match=rf"model_params.{field} to contain one integer"
    ):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("query_head_dim", 0),
        ("query_head_dim", -1),
        ("query_head_dim", "0"),
        ("query_head_dim", "-1"),
        ("value_head_dim", 0),
        ("pos_head_dim", 0),
    ),
)
def test_validate_zipformer_rejects_nonpositive_head_dimensions(
    field: str,
    value: int | str,
) -> None:
    model_config = make_model_config()
    model_config.model_params[field] = value

    with pytest.raises(
        ValueError,
        match=rf"model_params.{field} to contain one positive integer",
    ):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_encoder_layers", "32.5"),
        ("downsampling_factor", "32.5"),
        ("feedforward_dim", "32.5"),
        ("num_heads", 32),
        ("num_heads", "32.5"),
        ("encoder_dim", "32.5"),
        ("cnn_module_kernel", "32.5"),
    ),
)
def test_validate_zipformer_rejects_malformed_stack_sequence(
    field: str,
    value: int | str,
) -> None:
    model_config = make_model_config()
    model_config.model_params[field] = value

    with pytest.raises(ValueError, match=rf"{field} to contain only integers"):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize(
    ("precision", "encoder_dims", "alignment"),
    (
        ("fp16", "192,384,768,1020,768,384", 8),
        ("bf16", "192,384,768,1024,764,384", 8),
    ),
)
def test_validate_zipformer_rejects_unaligned_output_assembly_channels(
    precision: str,
    encoder_dims: str,
    alignment: int,
) -> None:
    model_config = make_model_config()
    model_config.model_params.encoder_dim = encoder_dims
    args = make_export_args()
    args.encoder_precision = precision

    with pytest.raises(ValueError, match=rf"final three.*divisible by {alignment}"):
        validate_zipformer(model_config, make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("precision", "encoder_dims", "alignment"),
    (
        ("fp32", "190,384,768,1024,768,384", 4),
        ("fp32", "192,384,768,1022,768,384", 4),
        ("fp16", "193,384,768,1024,768,384", 2),
        ("bf16", "193,384,768,1024,768,384", 2),
    ),
)
def test_validate_zipformer_rejects_unaligned_convolution_channels(
    precision: str,
    encoder_dims: str,
    alignment: int,
) -> None:
    model_config = make_model_config()
    model_config.model_params.encoder_dim = encoder_dims
    args = make_export_args()
    args.encoder_precision = precision

    with pytest.raises(
        ValueError,
        match=rf"Every model_params.encoder_dim value must be divisible by {alignment}",
    ):
        validate_zipformer(model_config, make_state_dict(), 512, args)


def test_validate_zipformer_rejects_audio_sample_dimension_overflow() -> None:
    args = make_export_args()
    args.max_audio_seconds = 1e308

    with pytest.raises(ValueError, match="signed 32-bit TensorRT audio-sample"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


def test_validate_zipformer_rejects_resampling_batch_grid_overflow() -> None:
    args = make_export_args()
    args.batch_size = 65_536
    args.beam = 1

    with pytest.raises(ValueError, match=r"CUDA grid\.z limit of 65535"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


def test_validate_zipformer_resampling_time_grid_boundary() -> None:
    args = make_export_args()
    args.max_audio_seconds = 20_972_400 / 16_000
    validate_zipformer(make_model_config(), make_state_dict(), 512, args)

    args.max_audio_seconds = 20_972_560 / 16_000
    with pytest.raises(ValueError, match=r"CUDA grid\.y limit of 65535"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


def test_validate_zipformer_rejects_decoder_capacity_overflow() -> None:
    args = make_export_args()
    args.batch_size = INT32_MAX // args.beam + 1

    with pytest.raises(ValueError, match="decoder capacity exceeds signed 32-bit"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("joiner_dim", "vocab_size", "tensor_name"),
    ((512, 512, "decoder_input"), (1, 1024, "tokens_log_prob")),
)
def test_validate_zipformer_rejects_decoder_tensor_overflow(
    joiner_dim: int,
    vocab_size: int,
    tensor_name: str,
) -> None:
    model_config = make_model_config()
    model_config.model_params.joiner_dim = joiner_dim
    args = make_export_args()
    args.beam = 1
    elements_per_hypothesis = (
        joiner_dim if tensor_name == "decoder_input" else vocab_size
    )
    args.batch_size = INT32_MAX // elements_per_hypothesis + 1

    with pytest.raises(ValueError, match=rf"tensor {tensor_name} exceeds"):
        validate_zipformer(
            model_config,
            make_state_dict(output_dim=joiner_dim),
            vocab_size,
            args,
        )


def test_validate_zipformer_rejects_too_few_mel_bins() -> None:
    model_config = make_model_config()
    model_config.feature_opts.mel_opts.num_bins = 6
    model_config.model_params.feature_dim = 6

    with pytest.raises(ValueError, match="at least seven mel bins"):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


def test_validate_zipformer_minimum_audio_frame_boundary() -> None:
    args = make_export_args()
    args.min_audio_seconds = 1360 / 16_000
    validate_zipformer(make_model_config(), make_state_dict(), 512, args)

    args.min_audio_seconds = 1359 / 16_000
    with pytest.raises(ValueError, match="requires at least 9"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


def test_validate_zipformer_feature_workspace_boundary() -> None:
    args = make_export_args()
    args.max_audio_seconds = 40.0
    args.batch_size = 259
    validate_zipformer(make_model_config(), make_state_dict(), 512, args)

    args.batch_size = 260
    with pytest.raises(ValueError, match="signed 32-bit TensorRT workspace"):
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("batch_size", "partitions"),
    ((128, 1), (256, 1), (384, 1), (512, 2), (1327, 4)),
)
def test_zipformer_subsampling_batch_partitions(
    batch_size: int,
    partitions: int,
) -> None:
    args = make_export_args()
    args.batch_size = batch_size
    args.max_audio_seconds = 40.0

    assert (
        get_subsampling_batch_partitions(make_model_config(), 128, args) == partitions
    )


@pytest.mark.parametrize("layer3_channels", (0, -1, 1.0))
def test_zipformer_subsampling_rejects_invalid_channels(
    layer3_channels: int | float,
) -> None:
    with pytest.raises(ValueError, match="layer3_channels must be a positive integer"):
        get_subsampling_batch_partitions(
            make_model_config(),
            cast(int, layer3_channels),
            make_export_args(),
        )


def test_zipformer_subsampling_single_item_cask_boundary() -> None:
    model_config = make_model_config()
    args = make_export_args()
    args.batch_size = 1
    sample_rate = model_config.feature_opts.frame_opts.samp_freq
    frame_shift = (
        model_config.feature_opts.frame_opts.frame_shift_ms * sample_rate // 1000
    )
    layer3_channels = 128
    conv_features = ((model_config.model_params.feature_dim - 3) // 2 - 2) // 2 + 1
    cask_element_limit = 1 << 31
    elements_per_frame = layer3_channels * conv_features
    maximum_conv_frames = cask_element_limit // elements_per_frame

    accepted_feature_frames = 2 * maximum_conv_frames + 8
    accepted_samples = accepted_feature_frames * frame_shift - frame_shift // 2
    args.max_audio_seconds = accepted_samples / sample_rate
    assert get_subsampling_batch_partitions(model_config, layer3_channels, args) == 1

    rejected_feature_frames = accepted_feature_frames + 1
    rejected_samples = rejected_feature_frames * frame_shift - frame_shift // 2
    args.max_audio_seconds = rejected_samples / sample_rate
    with pytest.raises(ValueError, match="One Zipformer subsampling item exceeds"):
        get_subsampling_batch_partitions(model_config, layer3_channels, args)


@pytest.mark.parametrize(
    ("decoder_type", "beam", "use_ctc"),
    (
        ("transducer_modified_beam_search", 6, False),
        ("ctc_greedy_search", 1, True),
    ),
)
def test_make_zipformer_runtime_config(
    decoder_type: str,
    beam: int,
    use_ctc: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_export_args(decoder_type)
    args.beam = beam
    validate = Mock(wraps=zipformer_exporter.validate_model_config)
    monkeypatch.setattr(zipformer_exporter, "validate_model_config", validate)
    runtime_config = make_runtime_config(
        make_model_config(),
        257,
        0,
        6000,
        args,
    )
    expected_decoder_params = {"beam": beam, "blank_penalty": 0.0}
    if not use_ctc:
        expected_decoder_params.update(
            {"context_size": 2, "decoder_dim": 512, "joiner_dim": 512}
        )

    validate.assert_called_once_with(runtime_config)
    assert OmegaConf.to_container(runtime_config, resolve=True) == {
        "model_type": "zipformer_asr",
        "decoder_type": decoder_type,
        "model_samplerate": 16_000,
        "vocab_size": 257,
        "blank_id": 0,
        "audio_encoder_params": {
            "feature_dim": 80,
            "encoder_dims": [192, 384, 768, 1024, 768, 384],
            "num_encoder_layers": [2, 2, 4, 5, 4, 2],
            "downsampling_factors": [1, 2, 4, 8, 4, 2],
            "feedforward_dims": [512, 1024, 2048, 3072, 2048, 1024],
            "pos_emb_max_len": 6000,
            "output_dim": 257 if use_ctc else 512,
            "frame_shift_ms": 10,
            "right_padding_samples": 200,
            "subsampling_factor": 4,
            "use_ctc": use_ctc,
            "min_audio_seconds": 0.5,
            "opt_audio_seconds": 15.0,
            "max_audio_seconds": 120.0,
        },
        "decoder_params": expected_decoder_params,
    }


@pytest.mark.parametrize(
    ("decoder_type", "projection_prefix"),
    (
        ("ctc_greedy_search", "ctc_output.1"),
        ("transducer_modified_beam_search", "projection_output"),
    ),
)
@pytest.mark.parametrize("missing_tensor", ("weight", "bias"))
def test_validate_zipformer_rejects_missing_decoder_head_tensor(
    decoder_type: str,
    projection_prefix: str,
    missing_tensor: str,
) -> None:
    state_dict = make_state_dict(decoder_type)
    del state_dict[f"{projection_prefix}.{missing_tensor}"]

    with pytest.raises(
        RuntimeError,
        match=rf"does not contain the {re.escape(projection_prefix)} head",
    ):
        validate_zipformer(
            make_model_config(),
            state_dict,
            512,
            make_export_args(decoder_type),
        )


@pytest.mark.parametrize(
    ("decoder_type", "tensor_name", "tensor", "message"),
    (
        (
            "ctc_greedy_search",
            "ctc_output.1.weight",
            torch.zeros(512),
            "weight must have rank 2",
        ),
        (
            "transducer_modified_beam_search",
            "projection_output.bias",
            torch.zeros(512, 1),
            "bias must have shape",
        ),
        (
            "transducer_modified_beam_search",
            "projection_output.bias",
            torch.zeros(511),
            "bias must have shape",
        ),
    ),
)
def test_validate_zipformer_rejects_malformed_decoder_head(
    decoder_type: str,
    tensor_name: str,
    tensor: torch.Tensor,
    message: str,
) -> None:
    state_dict = make_state_dict(decoder_type)
    state_dict[tensor_name] = tensor

    with pytest.raises(RuntimeError, match=message):
        validate_zipformer(
            make_model_config(),
            state_dict,
            512,
            make_export_args(decoder_type),
        )


def test_validate_zipformer_rejects_ctc_vocab_mismatch() -> None:
    with pytest.raises(RuntimeError, match="513 outputs"):
        validate_zipformer(
            make_model_config(),
            make_state_dict("ctc_greedy_search", 513),
            512,
            make_export_args("ctc_greedy_search"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_params.pos_head_dim", "2", "requires model_params.pos_head_dim=4"),
        ("model_params.pos_dim", 47, "model_params.pos_dim to be even"),
        ("model_params.context_size", 3, "context_size at most 2"),
        (
            "model_params.cnn_module_kernel",
            "31,31,14,15,15,31",
            "Every model_params.cnn_module_kernel value must be odd",
        ),
        ("min_encoder_input_frames", 8, "at least nine input frames"),
        (
            "model_params.subsampling_factor",
            2,
            "Expected model_params.subsampling_factor=4",
        ),
        ("model_params.causal", True, "Expected model_params.causal=False"),
        (
            "model_params.use_attention_decoder",
            True,
            "Expected model_params.use_attention_decoder=False",
        ),
        (
            "model_params.encoder_dim",
            "192,384",
            "model_params.encoder_dim to contain six positive integers",
        ),
        (
            "model_params.encoder_dim",
            "192,384,300,1024,768,384",
            "model_params.encoder_dim must be nondecreasing",
        ),
        (
            "model_params.encoder_dim",
            "192,384,768,1024,800,900",
            "model_params.encoder_dim must be nondecreasing",
        ),
        (
            "feature_opts.mel_opts.num_bins",
            128,
            "feature_opts.mel_opts.num_bins and model_params.feature_dim must match",
        ),
        ("feature_opts.mel_opts.low_freq", -1, "integer Zipformer mel bounds"),
        ("feature_opts.mel_opts.high_freq", 8001, "integer Zipformer mel bounds"),
        (
            "feature_opts.frame_opts.samp_freq",
            8000,
            "Expected feature_opts.frame_opts.samp_freq=16000",
        ),
        (
            "feature_opts.frame_opts.frame_length_ms",
            20,
            "Expected feature_opts.frame_opts.frame_length_ms=25",
        ),
        (
            "feature_opts.frame_opts.dither",
            0.1,
            "Expected feature_opts.frame_opts.dither=0.0",
        ),
        (
            "feature_opts.frame_opts.preemph_coeff",
            1.0,
            "Expected feature_opts.frame_opts.preemph_coeff=0.97",
        ),
        (
            "feature_opts.frame_opts.window_type",
            "hamming",
            "Expected feature_opts.frame_opts.window_type=povey",
        ),
        (
            "feature_opts.frame_opts.snip_edges",
            True,
            "Expected feature_opts.frame_opts.snip_edges=False",
        ),
        (
            "feature_opts.frame_opts.frame_shift_ms",
            30,
            "Expected feature_opts.frame_opts.frame_shift_ms=10",
        ),
    ),
)
def test_validate_zipformer_rejects_incompatible_values(
    field: str,
    value: bool | float | int | str,
    message: str,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


@pytest.mark.parametrize("value", (None, 2.0, "2"))
def test_validate_zipformer_rejects_noninteger_context_size(
    value: float | str | None,
) -> None:
    model_config = make_model_config()
    model_config.model_params.context_size = value

    with pytest.raises(ValueError, match="context_size.*positive integer"):
        validate_zipformer(model_config, make_state_dict(), 512, make_export_args())


def test_adjust_zipformer_state_dict() -> None:
    generator = torch.Generator().manual_seed(0)
    downsample_bias = torch.tensor([0.0, 1.0])
    output_downsample_bias = torch.tensor([1.0, 2.0, 3.0])
    log_scale = torch.tensor(0.5)
    pointwise_weight = torch.randn(12, 4, 1, 1, generator=generator)
    second_pointwise_weight = torch.randn(4, 12, 1, 1, generator=generator)
    conv1_weight = torch.randn(8, 1, 3, 3, generator=generator)
    conv1_bias = torch.randn(8, generator=generator)
    conv2_weight = torch.randn(16, 8, 3, 3, generator=generator)
    conv2_bias = torch.randn(16, generator=generator)
    conv3_weight = torch.randn(32, 16, 3, 3, generator=generator)
    conv3_bias = torch.randn(32, generator=generator)
    out_combiner_scale = torch.rand(6, generator=generator)
    state_dict = OrderedDict(
        (
            ("encoder_embed.out_norm.log_scale", log_scale),
            ("encoder_embed.conv.0.weight", conv1_weight),
            ("encoder_embed.conv.0.bias", conv1_bias),
            ("encoder_embed.conv.4.weight", conv2_weight),
            ("encoder_embed.conv.4.bias", conv2_bias),
            ("encoder_embed.conv.7.weight", conv3_weight),
            ("encoder_embed.conv.7.bias", conv3_bias),
            ("encoder_embed.convnext.pointwise_conv1.weight", pointwise_weight),
            (
                "encoder_embed.convnext.pointwise_conv2.weight",
                second_pointwise_weight,
            ),
            ("encoder.encoders.0.downsample.bias", downsample_bias),
            ("encoder.encoders.1.downsample.bias", downsample_bias),
            ("encoder.downsample_output.bias", output_downsample_bias),
            (
                "encoder.encoders.0.layers.0.self_attn_weights.in_proj.weight",
                torch.randn(8, 4, generator=generator),
            ),
            (
                "encoder.encoders.0.layers.0.bypass_scale",
                torch.ones(4),
            ),
            (
                "encoder.encoders.1.out_combiner.bypass_scale",
                out_combiner_scale,
            ),
            ("joiner.encoder_proj.weight", torch.randn(4, 8, generator=generator)),
        ),
    )
    original_values = {key: value.clone() for key, value in state_dict.items()}
    expected = {
        "subsampling.out_norm.scale": torch.exp(log_scale),
        "subsampling.conv1.weight": conv1_weight,
        "subsampling.conv1.bias": conv1_bias,
        "subsampling.conv2.weight": conv2_weight,
        "subsampling.conv2.bias": conv2_bias,
        "subsampling.conv3.weight": conv3_weight,
        "subsampling.conv3.bias": conv3_bias,
        "subsampling.pointwise_conv1.weight": pointwise_weight[:, :, 0, 0],
        "subsampling.pointwise_conv2.weight": second_pointwise_weight[:, :, 0, 0],
        "encoder_1.downsample.weights": torch.zeros(1, 1, dtype=torch.float32),
        "encoder_2.downsample.weights": torch.softmax(downsample_bias, dim=0).unsqueeze(
            1
        ),
        "downsample_output.weights": torch.softmax(
            output_downsample_bias, dim=0
        ).unsqueeze(1),
        "encoder_1.layers.0.self_attn_weights.in_proj.weight": state_dict[
            "encoder.encoders.0.layers.0.self_attn_weights.in_proj.weight"
        ],
        "encoder_1.bypass_scale": torch.ones(4, dtype=torch.float32),
        "encoder_2.bypass_scale": out_combiner_scale,
        "projection_output.weight": state_dict["joiner.encoder_proj.weight"],
    }

    torch.testing.assert_close(adjust_state_dict(state_dict), expected)
    torch.testing.assert_close(state_dict, original_values)


@pytest.mark.parametrize(
    "weight_shape",
    (
        pytest.param((8, 4), id="rank-two"),
        pytest.param((8, 4, 1, 2), id="non-pointwise-kernel"),
    ),
)
@pytest.mark.parametrize("pointwise_layer", ("pointwise_conv1", "pointwise_conv2"))
def test_adjust_zipformer_state_dict_rejects_invalid_pointwise_kernel(
    pointwise_layer: str,
    weight_shape: tuple[int, ...],
) -> None:
    state_dict = OrderedDict(
        {f"encoder_embed.convnext.{pointwise_layer}.weight": torch.zeros(weight_shape)}
    )

    with pytest.raises(ValueError, match=r"shape \(out_channels, in_channels, 1, 1\)"):
        adjust_state_dict(state_dict)


def test_adjust_zipformer_state_dict_requires_first_attention_projection() -> None:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    with pytest.raises(KeyError) as error:
        adjust_state_dict(state_dict)

    assert error.value.args == ("encoder_1.layers.0.self_attn_weights.in_proj.weight",)


def test_adjust_zipformer_state_dict_rejects_alias_collision() -> None:
    state_dict = OrderedDict(
        (
            ("projection_output.weight", torch.ones(2, 2)),
            ("joiner.encoder_proj.weight", torch.zeros(2, 2)),
        )
    )

    with pytest.raises(ValueError, match="both map to projection_output.weight"):
        adjust_state_dict(state_dict)


@pytest.mark.parametrize("missing_component", ("decoder", "joiner"))
def test_export_zipformer_rejects_incomplete_transducer_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_component: str
) -> None:
    args = make_export_args()
    args.output_dir = tmp_path
    export = Mock()
    monkeypatch.setattr(torch.onnx, "export", export)

    with pytest.raises(RuntimeError, match=rf"{missing_component} was not initialized"):
        export_model_to_onnx(
            torch.nn.Identity(),
            None if missing_component == "decoder" else torch.nn.Identity(),
            None if missing_component == "joiner" else torch.nn.Identity(),
            make_model_config(),
            args,
        )

    export.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("decoder_type", "decoder_batch"),
    (
        ("ctc_greedy_search", None),
        ("transducer_greedy_search", 3),
        ("transducer_modified_beam_search", 12),
    ),
)
def test_export_zipformer_onnx_inputs_and_context_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_type: str,
    decoder_batch: int | None,
) -> None:
    args = make_export_args(decoder_type)
    args.output_dir = tmp_path
    args.batch_size = 3
    args.beam = 4
    args.opt_audio_seconds = 2.5
    encoder = torch.nn.Identity()
    context_lookup = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    decoder = joiner = None
    if decoder_batch is not None:
        decoder = Mock(spec=zipformer_exporter.Decoder)
        decoder.make_context_lookup.return_value = context_lookup
        with torch.random.fork_rng(devices=[]):
            joiner = zipformer_exporter.Joiner(512, 4, dtype=torch.float16)

    def record_export(*args, **kwargs):
        """Require inference mode when the exporter prepares each ONNX graph.

        Parameters
        ----------
        *args : tuple
            Positional ONNX export arguments recorded by the surrounding Mock.
        **kwargs : dict
            ONNX export options recorded by the surrounding Mock.
        """

        assert torch.is_inference_mode_enabled()

    export = Mock(side_effect=record_export)
    monkeypatch.setattr(torch.onnx, "export", export)

    encoder_path, decoder_path = export_model_to_onnx(
        encoder, decoder, joiner, make_model_config(), args
    )

    assert encoder_path == tmp_path / "zipformer.onnx"
    encoder_call = export.call_args_list[0]
    module, (audio, lengths), path = encoder_call.args
    assert module is encoder
    assert path == encoder_path
    torch.testing.assert_close(
        audio, torch.zeros(3, 40_200, dtype=torch.float32), rtol=0, atol=0
    )
    torch.testing.assert_close(lengths, torch.full((3,), 40_000, dtype=torch.int64))
    assert encoder_call.kwargs == {
        "input_names": ("audio", "audio_lengths"),
        "output_names": ("encoder_output", "encoder_output_lengths"),
        "opset_version": zipformer_exporter.ONNX_OPSET_VERSION,
        "dynamic_shapes": {
            "audio": {1: torch.export.Dim.DYNAMIC},
            "audio_lengths": {},
        },
    }
    if decoder_batch is None:
        assert decoder_path is None
        assert export.call_count == 1
        assert list(tmp_path.iterdir()) == []
    else:
        assert decoder_path == tmp_path / "decoder.onnx"
        assert export.call_count == 2
        decoder.make_context_lookup.assert_called_once_with(chunk_size=8192)
        torch.testing.assert_close(
            torch.load(tmp_path / "decoder_contexts.pt", weights_only=True),
            context_lookup,
        )
        decoder_call = export.call_args_list[1]
        module, inputs, path = decoder_call.args
        assert module is joiner
        assert path == decoder_path
        assert len(inputs) == 2
        for tensor in inputs:
            torch.testing.assert_close(
                tensor,
                torch.zeros(decoder_batch, 512, dtype=torch.float16),
                rtol=0,
                atol=0,
            )
        assert decoder_call.kwargs == {
            "input_names": ("decoder_input", "encoder_output"),
            "output_names": ("tokens_log_prob",),
            "opset_version": zipformer_exporter.ONNX_OPSET_VERSION,
        }


@pytest.mark.parametrize(
    ("blank_id", "blank_piece"),
    (
        pytest.param(4, "<unk>", id="outside-decoder-vocabulary"),
        pytest.param(1, "not-blank", id="wrong-piece-at-valid-id"),
    ),
)
def test_export_zipformer_rejects_invalid_blank_before_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blank_id: int,
    blank_piece: str,
) -> None:
    class InvalidBlankTokenizer(FakeTokenizer):
        """Inject the selected invalid blank mapping into the fake vocabulary."""

        def piece_to_id(self, piece: str) -> int:
            """Return the invalid blank ID selected by this test case.

            Parameters
            ----------
            piece : str
                Requested piece, expected to be ``<blk>``.

            Returns
            -------
            int
                Deliberately invalid blank ID from the test parameters.
            """

            assert piece == "<blk>"
            return blank_id

        def id_to_piece(self, token_id: int) -> str:
            """Substitute the test piece only at the configured blank ID.

            Parameters
            ----------
            token_id : int
                ID requested during blank-token validation.

            Returns
            -------
            str
                Test-configured blank piece or the ordinary fake-token spelling.
            """

            return (
                blank_piece if token_id == blank_id else super().id_to_piece(token_id)
            )

    args = make_export_args()
    args.output_dir = tmp_path / "bundle"
    args.model_path = write_zipformer_sources(tmp_path / "source")
    monkeypatch.setattr(
        zipformer_exporter.spm,
        "SentencePieceProcessor",
        InvalidBlankTokenizer,
    )

    load_checkpoint = Mock()
    monkeypatch.setattr(torch, "load", load_checkpoint)

    with pytest.raises(ValueError, match="exact <blk> piece"):
        zipformer_exporter.export_zipformer(args)

    load_checkpoint.assert_not_called()
