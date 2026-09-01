#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for Zipformer bundle export, metadata conversion, and validation."""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import tensorrt as trt
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
    make_runtime_config,
    parse_args,
)
from fast_gpu_asr.utils import (
    ASRInitializationError,
    validate_decoder_engine,
    validate_encoder_engine,
    validate_model,
    validate_model_config,
    validate_tokenizer,
    validate_zipformer_context_lookup,
)


class FakeEngine:
    """Expose the TensorRT metadata methods used by runtime validation."""

    def __init__(
        self,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, trt.DataType],
        profiles: dict[str, tuple[tuple[int, ...], ...]] | None = None,
    ) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.names = input_names + output_names
        self.shapes = shapes
        self.dtypes = dtypes
        self.profiles = profiles or {}
        self.locations = {name: trt.TensorLocation.DEVICE for name in self.names}
        self.formats = {name: trt.TensorFormat.LINEAR for name in self.names}
        self.num_io_tensors = len(self.names)

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_mode(self, name: str) -> trt.TensorIOMode:
        return (
            trt.TensorIOMode.INPUT
            if name in self.input_names
            else trt.TensorIOMode.OUTPUT
        )

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], ...]:
        assert profile_index == 0
        return self.profiles[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> trt.TensorFormat:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> trt.TensorLocation:
        return self.locations[name]


class FakeTokenizer:
    """Expose a Zipformer tokenizer with a trailing unknown token."""

    def __init__(self, model_file: str) -> None:
        self.model_file = model_file

    def vocab_size(self) -> int:
        return 5

    def unk_id(self) -> int:
        return 4

    def piece_to_id(self, piece: str) -> int:
        assert piece == "<blk>"
        return 0

    def id_to_piece(self, token_id: int) -> str:
        return "<blk>" if token_id == 0 else str(token_id)


def make_model_config() -> DictConfig:
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
    projection_prefix = (
        "ctc_output.1" if decoder_type == "ctc_greedy_search" else "projection_output"
    )
    return OrderedDict(
        (
            (f"{projection_prefix}.weight", torch.zeros(output_dim, input_dim)),
            (f"{projection_prefix}.bias", torch.zeros(output_dim)),
        ),
    )


def make_export_args(
    decoder_type: str = "transducer_modified_beam_search",
) -> argparse.Namespace:
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
    """Write minimal source placeholders needed before export validation."""

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.yaml").write_text("model: config\n")
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


def test_export_zipformer_rejects_destination_containing_sources(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "model"
    model_path = write_zipformer_sources(output_dir)
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    with pytest.raises(ValueError, match="contains required source file"):
        export_zipformer(args)

    assert model_path.read_bytes() == b"checkpoint"
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "bpe.model").is_file()


def test_export_zipformer_replaces_existing_output_before_export(
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

    def fail_export(_config_path: Path) -> None:
        (output_dir / zipformer_exporter.ZIPFORMER_TENSORRT_FILE).write_bytes(
            b"partial"
        )
        raise RuntimeError("engine build failed")

    monkeypatch.setattr(zipformer_exporter.OmegaConf, "load", fail_export)

    with pytest.raises(RuntimeError, match="engine build failed"):
        export_zipformer(args)

    assert not sentinel_path.exists()
    assert (
        output_dir / zipformer_exporter.ZIPFORMER_TENSORRT_FILE
    ).read_bytes() == b"partial"


@pytest.mark.parametrize(
    "decoder_type",
    ("ctc_greedy_search", "transducer_greedy_search"),
)
def test_export_zipformer_forces_greedy_beam(
    decoder_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = write_zipformer_sources(tmp_path / "source")
    args = make_export_args(decoder_type)
    args.model_path = model_path
    args.output_dir = tmp_path / "bundle"
    args.beam = 6

    def stop_export(_config_path: Path) -> None:
        raise RuntimeError("stop export")

    monkeypatch.setattr(zipformer_exporter.OmegaConf, "load", stop_export)

    with pytest.raises(RuntimeError, match="stop export"):
        export_zipformer(args)

    assert args.beam == 1


@pytest.mark.parametrize("debug", (False, True))
@pytest.mark.parametrize(
    "decoder_type",
    ("ctc_greedy_search", "transducer_modified_beam_search"),
)
def test_export_zipformer_validates_exact_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
    decoder_type: str,
) -> None:
    source_dir = tmp_path / "source"
    model_path = write_zipformer_sources(source_dir)
    source_config = make_model_config()
    OmegaConf.save(source_config, source_dir / "config.yaml")
    output_dir = tmp_path / "bundle"
    args = make_export_args(decoder_type)
    args.batch_size = 3
    args.beam = 4
    args.debug = debug
    args.model_path = model_path
    args.output_dir = output_dir
    adjusted_state_dict = OrderedDict(
        (("subsampling.conv3.weight", torch.zeros(128, 1, 3, 3)),)
    )
    encoder = object()
    decoder = None if decoder_type == "ctc_greedy_search" else object()
    joiner = None if decoder_type == "ctc_greedy_search" else object()
    runtime_configs: list[DictConfig] = []
    build_calls: list[tuple[Path, Path, dict[str, object], int]] = []
    events: list[str] = []

    def load_tokenizer(*, model_file: str) -> FakeTokenizer:
        assert Path(model_file) == source_dir / zipformer_exporter.TOKENIZER_FILE
        return FakeTokenizer(model_file)

    checkpoint_state_dict = OrderedDict((("source", torch.zeros(1)),))

    def load_checkpoint(
        checkpoint_path: Path,
        *,
        map_location: torch.device,
        weights_only: bool,
    ) -> dict[str, OrderedDict[str, torch.Tensor]]:
        assert checkpoint_path == model_path
        assert map_location == torch.device("cpu")
        assert weights_only is True
        return {"model": checkpoint_state_dict}

    def adjust_checkpoint(
        state_dict: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        assert tuple(state_dict) == ("source",)
        assert state_dict["source"] is checkpoint_state_dict["source"]
        return adjusted_state_dict

    monkeypatch.setattr(
        zipformer_exporter.spm,
        "SentencePieceProcessor",
        load_tokenizer,
    )
    monkeypatch.setattr(zipformer_exporter.torch, "load", load_checkpoint)
    monkeypatch.setattr(
        zipformer_exporter,
        "adjust_state_dict",
        adjust_checkpoint,
    )

    def validate_source(
        model_config: DictConfig,
        state_dict: OrderedDict[str, torch.Tensor],
        vocab_size: int,
        export_args: argparse.Namespace,
    ) -> None:
        events.append("source-validate")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert state_dict is adjusted_state_dict
        assert vocab_size == 4
        assert export_args is args

    monkeypatch.setattr(zipformer_exporter, "validate_zipformer", validate_source)

    def get_partitions(
        model_config: DictConfig,
        channels: int,
        export_args: argparse.Namespace,
    ) -> int:
        events.append("partition")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert channels == 128
        assert export_args is args
        return 1

    monkeypatch.setattr(
        zipformer_exporter,
        "get_subsampling_batch_partitions",
        get_partitions,
    )

    def make_export_model(
        model_config: DictConfig,
        state_dict: OrderedDict[str, torch.Tensor],
        vocab_size: int,
        _pos_emb_max_len: int,
        selected_decoder_type: str,
        partitions: int,
        encoder_dtype: torch.dtype,
        decoder_dtype: torch.dtype,
    ) -> tuple[object, object | None, object | None]:
        events.append("model")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert state_dict is adjusted_state_dict
        assert vocab_size == 4
        assert selected_decoder_type == decoder_type
        assert partitions == 1
        assert encoder_dtype == torch.float32
        assert decoder_dtype == torch.float32
        return encoder, decoder, joiner

    monkeypatch.setattr(zipformer_exporter, "make_model", make_export_model)
    real_make_runtime_config = zipformer_exporter.make_runtime_config

    def record_runtime_config(
        model_config: DictConfig,
        vocab_size: int,
        blank_id: int,
        pos_emb_max_len: int,
        export_args: argparse.Namespace,
    ) -> DictConfig:
        events.append("runtime-config")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert vocab_size == 4
        assert blank_id == 0
        assert export_args is args
        runtime_config = real_make_runtime_config(
            model_config,
            vocab_size,
            blank_id,
            pos_emb_max_len,
            export_args,
        )
        runtime_configs.append(runtime_config)
        return runtime_config

    def export_onnx(
        export_encoder: object,
        export_decoder: object | None,
        export_joiner: object | None,
        model_config: DictConfig,
        export_args: argparse.Namespace,
    ) -> tuple[Path, Path | None]:
        assert export_encoder is encoder
        assert export_decoder is decoder
        assert export_joiner is joiner
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert export_args is args
        encoder_path = output_dir / zipformer_exporter.ZIPFORMER_ONNX_FILE
        encoder_path.write_bytes(b"encoder")
        decoder_path = None
        if decoder is not None:
            decoder_path = output_dir / zipformer_exporter.ZIPFORMER_DECODER_ONNX_FILE
            decoder_path.write_bytes(b"decoder")
            (
                output_dir / zipformer_exporter.ZIPFORMER_DECODER_CONTEXTS_FILE
            ).write_bytes(b"contexts")
        events.append("onnx")
        return encoder_path, decoder_path

    def build_engine(
        onnx_path: Path,
        engine_path: Path,
        profiles: dict[str, object],
        optimization_level: int,
    ) -> None:
        engine_path.write_bytes(b"engine")
        build_calls.append((onnx_path, engine_path, profiles, optimization_level))
        events.append(f"build:{engine_path.name}")

    def remove_graph(path: Path) -> None:
        path.unlink()
        events.append(f"remove:{path.name}")

    def validate_bundle(model_dir: Path, model_config: DictConfig) -> None:
        events.append("validate")
        assert len(runtime_configs) == 1
        assert model_config is not runtime_configs[0]
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            runtime_configs[0]
        )
        assert (
            model_dir / zipformer_exporter.TOKENIZER_FILE
        ).read_bytes() == b"tokenizer"
        assert (model_dir / zipformer_exporter.MODEL_CONFIG_FILE).is_file()
        assert (model_dir / zipformer_exporter.ZIPFORMER_TENSORRT_FILE).is_file()
        assert (model_dir / zipformer_exporter.ZIPFORMER_ONNX_FILE).exists() is debug
        has_decoder = decoder_type != "ctc_greedy_search"
        assert (
            model_dir / zipformer_exporter.ZIPFORMER_DECODER_TENSORRT_FILE
        ).exists() is has_decoder
        assert (
            model_dir / zipformer_exporter.ZIPFORMER_DECODER_CONTEXTS_FILE
        ).exists() is has_decoder
        assert (
            model_dir / zipformer_exporter.ZIPFORMER_DECODER_ONNX_FILE
        ).exists() is (debug and has_decoder)

    monkeypatch.setattr(
        zipformer_exporter,
        "make_runtime_config",
        record_runtime_config,
    )
    monkeypatch.setattr(zipformer_exporter, "export_model_to_onnx", export_onnx)
    monkeypatch.setattr(zipformer_exporter, "build_tensorrt_engine", build_engine)
    monkeypatch.setattr(zipformer_exporter, "remove_onnx_artifacts", remove_graph)
    monkeypatch.setattr(zipformer_exporter, "validate_model", validate_bundle)

    export_zipformer(args)

    sample_rate = source_config.feature_opts.frame_opts.samp_freq
    right_padding = (
        source_config.feature_opts.frame_opts.frame_length_ms * sample_rate // 2000
    )
    expected_profile = tuple(
        (args.batch_size, round(seconds * sample_rate) + right_padding)
        for seconds in (
            args.min_audio_seconds,
            args.opt_audio_seconds,
            args.max_audio_seconds,
        )
    )
    assert build_calls[0] == (
        output_dir / zipformer_exporter.ZIPFORMER_ONNX_FILE,
        output_dir / zipformer_exporter.ZIPFORMER_TENSORRT_FILE,
        {"audio": expected_profile},
        args.optimization_level,
    )
    if decoder is None:
        assert len(build_calls) == 1
    else:
        assert build_calls[1] == (
            output_dir / zipformer_exporter.ZIPFORMER_DECODER_ONNX_FILE,
            output_dir / zipformer_exporter.ZIPFORMER_DECODER_TENSORRT_FILE,
            {},
            args.optimization_level,
        )
    assert events[-1] == "validate"
    assert events[:5] == [
        "source-validate",
        "partition",
        "model",
        "runtime-config",
        "onnx",
    ]
    assert events.count("validate") == 1
    assert len([event for event in events if event.startswith("remove:")]) == (
        0 if debug else 1 + int(decoder is not None)
    )


def make_encoder_engine(
    runtime_config: DictConfig,
    output_dim: int | None = None,
) -> FakeEngine:
    """Build encoder metadata matching one Zipformer runtime configuration."""

    batch_size = 8
    audio_profile = tuple(
        (
            batch_size,
            round(seconds * runtime_config.model_samplerate)
            + runtime_config.audio_encoder_params.right_padding_samples,
        )
        for seconds in (
            runtime_config.audio_encoder_params.min_audio_seconds,
            runtime_config.audio_encoder_params.opt_audio_seconds,
            runtime_config.audio_encoder_params.max_audio_seconds,
        )
    )
    if output_dim is None:
        output_dim = runtime_config.audio_encoder_params.output_dim
    return FakeEngine(
        ("audio", "audio_lengths"),
        ("encoder_output", "encoder_output_lengths"),
        {
            "audio": (batch_size, -1),
            "audio_lengths": (batch_size,),
            "encoder_output": (batch_size, -1, output_dim),
            "encoder_output_lengths": (batch_size,),
        },
        {
            "audio": trt.float32,
            "audio_lengths": trt.int64,
            "encoder_output": trt.float16,
            "encoder_output_lengths": trt.int32,
        },
        {
            "audio": audio_profile,
            "audio_lengths": ((batch_size,),) * 3,
        },
    )


def test_zipformer_precision_defaults_to_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
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
        ],
    )

    args = parse_args()

    assert args.encoder_precision == "fp32"
    assert args.decoder_precision == "fp32"


@pytest.mark.parametrize(
    "decoder_type",
    ("ctc_greedy_search", "transducer_modified_beam_search"),
)
def test_validate_zipformer_accepts_supported_decoder(decoder_type: str) -> None:
    validate_zipformer(
        make_model_config(),
        make_state_dict(decoder_type),
        512,
        make_export_args(decoder_type),
    )


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
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("batch_size", 0, "batch_size must be positive"),
        ("batch_size", 1.5, "batch_size must be positive"),
        ("beam", 0, "beam must be positive"),
        ("beam", 1.5, "beam must be positive"),
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


def test_validate_zipformer_rejects_unsupported_position_head_dimension() -> None:
    model_config = make_model_config()
    model_config.model_params.pos_head_dim = "2"

    with pytest.raises(ValueError, match="requires model_params.pos_head_dim=4"):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


def test_validate_zipformer_rejects_odd_position_dimension() -> None:
    model_config = make_model_config()
    model_config.model_params.pos_dim = 47

    with pytest.raises(ValueError, match="model_params.pos_dim to be even"):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


def test_validate_zipformer_rejects_incompatible_predictor_dimension() -> None:
    model_config = make_model_config()
    model_config.model_params.decoder_dim = 9

    with pytest.raises(ValueError, match="grouped-convolution count"):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


def test_validate_zipformer_rejects_nonpositive_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocab_size.*positive integer"):
        validate_zipformer(
            make_model_config(),
            make_state_dict(),
            0,
            make_export_args(),
        )


@pytest.mark.parametrize("value", (32.5, "32.5", True))
def test_validate_zipformer_rejects_nonintegral_head_dimensions(
    value: float | str | bool,
) -> None:
    model_config = make_model_config()
    model_config.model_params.query_head_dim = value

    with pytest.raises(
        ValueError, match="model_params.query_head_dim to contain one integer"
    ):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


@pytest.mark.parametrize(
    ("precision", "encoder_dims", "alignment"),
    (
        ("fp32", "192,384,768,1022,768,384", 4),
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

    with pytest.raises(ValueError, match=rf"divisible by {alignment}"):
        validate_zipformer(model_config, make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("precision", "encoder_dims", "alignment"),
    (
        ("fp32", "190,384,768,1024,768,384", 4),
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


def test_validate_zipformer_rejects_even_convolution_kernel() -> None:
    model_config = make_model_config()
    model_config.model_params.cnn_module_kernel = "31,31,14,15,15,31"

    with pytest.raises(
        ValueError, match="Every model_params.cnn_module_kernel value must be odd"
    ):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


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


@pytest.mark.parametrize(
    ("max_audio_samples", "accepted"),
    ((20_972_400, True), (20_972_560, False)),
)
def test_validate_zipformer_resampling_time_grid_boundary(
    max_audio_samples: int,
    accepted: bool,
) -> None:
    args = make_export_args()
    args.opt_audio_seconds = max_audio_samples / 16_000
    args.max_audio_seconds = max_audio_samples / 16_000

    if accepted:
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)
    else:
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
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


def test_validate_zipformer_rejects_too_few_subsampling_frames() -> None:
    model_config = make_model_config()
    model_config.min_encoder_input_frames = 8

    with pytest.raises(ValueError, match="at least nine input frames"):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


@pytest.mark.parametrize(("batch_size", "accepted"), ((259, True), (260, False)))
def test_validate_zipformer_feature_workspace_boundary(
    batch_size: int,
    accepted: bool,
) -> None:
    args = make_export_args()
    args.batch_size = batch_size
    args.max_audio_seconds = 40.0

    if accepted:
        validate_zipformer(make_model_config(), make_state_dict(), 512, args)
    else:
        with pytest.raises(ValueError, match="signed 32-bit TensorRT workspace"):
            validate_zipformer(make_model_config(), make_state_dict(), 512, args)


@pytest.mark.parametrize(
    ("batch_size", "partitions"),
    ((128, 1), (256, 1), (384, 1), (512, 2)),
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
    layer3_channels: int | float | bool,
) -> None:
    with pytest.raises(ValueError, match="layer3_channels must be a positive integer"):
        get_subsampling_batch_partitions(
            make_model_config(),
            layer3_channels,  # type: ignore[arg-type]
            make_export_args(),
        )


def test_zipformer_subsampling_rejects_single_item_over_cask_limit() -> None:
    args = make_export_args()
    args.batch_size = 1
    args.max_audio_seconds = 20_000.0

    with pytest.raises(ValueError, match="One Zipformer subsampling item exceeds"):
        get_subsampling_batch_partitions(make_model_config(), 128, args)


def test_make_zipformer_runtime_config() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )

    validate_model_config(runtime_config)
    assert runtime_config.model_type == "zipformer_asr"
    assert runtime_config.model_samplerate == 16000
    assert runtime_config.vocab_size == 512
    assert runtime_config.blank_id == 0
    assert list(runtime_config.audio_encoder_params.encoder_dims) == [
        192,
        384,
        768,
        1024,
        768,
        384,
    ]
    assert runtime_config.audio_encoder_params.pos_emb_max_len == 6000
    assert runtime_config.audio_encoder_params.frame_shift_ms == 10
    assert runtime_config.audio_encoder_params.right_padding_samples == 200
    assert runtime_config.audio_encoder_params.subsampling_factor == 4
    assert runtime_config.audio_encoder_params.min_audio_seconds == 0.5
    assert runtime_config.audio_encoder_params.opt_audio_seconds == 15.0
    assert runtime_config.audio_encoder_params.max_audio_seconds == 120.0
    for field in (
        "cnn_module_kernels",
        "num_heads",
        "query_head_dim",
        "pos_head_dim",
        "value_head_dim",
        "pos_dim",
    ):
        assert field not in runtime_config.audio_encoder_params
    assert runtime_config.decoder_params.beam == 6
    assert runtime_config.decoder_params.blank_penalty == 0.0
    assert "context_cache" not in runtime_config.decoder_params
    assert runtime_config.decoder_params.context_size == 2


def test_make_zipformer_ctc_runtime_config_omits_transducer_metadata() -> None:
    args = make_export_args("ctc_greedy_search")
    args.beam = 1

    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        args,
    )

    validate_model_config(runtime_config)
    assert runtime_config.decoder_type == "ctc_greedy_search"
    assert runtime_config.audio_encoder_params.output_dim == 512
    assert runtime_config.audio_encoder_params.use_ctc is True
    assert set(runtime_config.decoder_params) == {"beam", "blank_penalty"}


def test_validate_model_config_rejects_zipformer_output_dimension_mismatch() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.output_dim = 256

    with pytest.raises(
        ASRInitializationError, match="audio_encoder_params.output_dim=512"
    ):
        validate_model_config(runtime_config)


@pytest.mark.parametrize(
    "field",
    (
        "encoder_dims",
        "num_encoder_layers",
        "downsampling_factors",
        "feedforward_dims",
    ),
)
def test_validate_model_config_rejects_nonpositive_zipformer_sequence_value(
    field: str,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params[field][2] = 0

    with pytest.raises(
        ASRInitializationError,
        match=rf"audio_encoder_params\.{field}\[2\].*positive integer",
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_invalid_zipformer_sequence() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.encoder_dims = [192] * 5

    with pytest.raises(
        ASRInitializationError,
        match="contain six positive integers",
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_invalid_zipformer_dimension_order() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.encoder_dims = [
        192,
        384,
        768,
        512,
        768,
        384,
    ]

    with pytest.raises(ASRInitializationError, match="nondecreasing"):
        validate_model_config(runtime_config)


@pytest.mark.parametrize("blank_id", (-1, 512, 0.0))
def test_validate_model_config_rejects_invalid_zipformer_blank_id(
    blank_id: int | float | bool,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.blank_id = blank_id

    with pytest.raises(ASRInitializationError, match="Expected Zipformer blank_id"):
        validate_model_config(runtime_config)


def test_validate_zipformer_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_export_args()
    args.beam = 4
    runtime_config = make_runtime_config(
        make_model_config(),
        4,
        0,
        6000,
        args,
    )

    with pytest.raises(ASRInitializationError, match="Missing SentencePiece tokenizer"):
        validate_tokenizer(tmp_path, runtime_config)

    tokenizer_path = tmp_path / "bpe.model"
    tokenizer_path.touch()
    monkeypatch.setattr(
        "fast_gpu_asr.utils.spm.SentencePieceProcessor",
        FakeTokenizer,
    )
    validate_tokenizer(tmp_path, runtime_config)

    runtime_config.blank_id = 1
    with pytest.raises(ASRInitializationError, match="Zipformer blank_id 0"):
        validate_tokenizer(tmp_path, runtime_config)
    runtime_config.blank_id = 0

    runtime_config.vocab_size = 5
    with pytest.raises(ASRInitializationError, match="tokenizer vocabulary size 5"):
        validate_tokenizer(tmp_path, runtime_config)


@pytest.mark.parametrize(
    "decoder_type",
    ("ctc_greedy_search", "transducer_greedy_search"),
)
def test_validate_model_config_rejects_nonunit_greedy_beam(
    decoder_type: str,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.decoder_type = decoder_type
    runtime_config.decoder_params.beam = 2

    with pytest.raises(ASRInitializationError, match=f"beam=1 for {decoder_type}"):
        validate_model_config(runtime_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "audio_encoder_params.max_audio_seconds",
            float("inf"),
            "exceeds signed 32-bit sample indexing",
        ),
        (
            "audio_encoder_params.max_audio_seconds",
            1e300,
            "exceeds signed 32-bit sample indexing",
        ),
        ("decoder_params.blank_penalty", float("nan"), "finite float"),
    ),
)
def test_validate_model_config_rejects_nonfinite_values(
    field: str,
    value: float,
    message: str,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    OmegaConf.update(runtime_config, field, value)

    with pytest.raises(ASRInitializationError, match=re.escape(message)):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_unsupported_context_size() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.decoder_params.context_size = 3

    with pytest.raises(ASRInitializationError, match="context_size at most 2"):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_zipformer_search_candidate_overflow() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.vocab_size = 50_000
    runtime_config.decoder_params.beam = 50_000

    with pytest.raises(
        ASRInitializationError, match="Zipformer per-utterance search table"
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_inconsistent_ctc_metadata() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.use_ctc = True

    with pytest.raises(
        ASRInitializationError, match="audio_encoder_params.use_ctc=False"
    ):
        validate_model_config(runtime_config)


def test_validate_encoder_engine_rejects_output_dimension_mismatch() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    engine = make_encoder_engine(runtime_config, output_dim=256)

    with pytest.raises(
        ASRInitializationError,
        match="encoder_output shape .*512.*256",
    ):
        validate_encoder_engine(engine, runtime_config)


def test_validate_model_reports_missing_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    monkeypatch.setattr("fast_gpu_asr.utils.validate_tokenizer", lambda *_: None)

    with pytest.raises(ASRInitializationError, match="Missing TensorRT engine"):
        validate_model(tmp_path, runtime_config)


def test_validate_decoder_engine_rejects_vocabulary_dimension_mismatch() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    batch_size = 8
    decoder_capacity = batch_size * runtime_config.decoder_params.beam
    joiner_dim = runtime_config.decoder_params.joiner_dim
    engine = FakeEngine(
        ("decoder_input", "encoder_output"),
        ("tokens_log_prob",),
        {
            "decoder_input": (decoder_capacity, joiner_dim),
            "encoder_output": (decoder_capacity, joiner_dim),
            "tokens_log_prob": (decoder_capacity, runtime_config.vocab_size + 1),
        },
        {
            "decoder_input": trt.float16,
            "encoder_output": trt.float16,
            "tokens_log_prob": trt.float32,
        },
    )

    with pytest.raises(
        ASRInitializationError,
        match="tokens_log_prob shape .*512.*513",
    ):
        validate_decoder_engine(engine, runtime_config, batch_size)


def test_validate_zipformer_context_lookup(tmp_path: Path) -> None:
    args = make_export_args()
    args.beam = 4
    runtime_config = make_runtime_config(
        make_model_config(),
        4,
        0,
        6000,
        args,
    )
    runtime_config.decoder_params.joiner_dim = 3
    context_lookup_path = tmp_path / "decoder_contexts.pt"

    with pytest.raises(ASRInitializationError, match="Missing predictor context cache"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    context_lookup_path.write_bytes(b"")
    with pytest.raises(ASRInitializationError, match="Failed to load predictor"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    torch.save({"context_lookup": torch.empty(25, 3)}, context_lookup_path)
    with pytest.raises(ASRInitializationError, match="contain one tensor"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    torch.save(torch.empty(24, 3), context_lookup_path)
    with pytest.raises(ASRInitializationError, match="context lookup shape .*25.*24"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    torch.save(torch.empty(25, 3, dtype=torch.int32), context_lookup_path)
    with pytest.raises(ASRInitializationError, match="FP16, FP32, or BF16"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    sparse_lookup = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64),
        torch.empty(0),
        size=(25, 3),
        check_invariants=True,
    )
    torch.save(sparse_lookup, context_lookup_path)
    with pytest.raises(ASRInitializationError, match="contiguous dense CPU tensor"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    noncontiguous_lookup = torch.empty(3, 25).T
    assert not noncontiguous_lookup.is_contiguous()
    torch.save(noncontiguous_lookup, context_lookup_path)
    with pytest.raises(ASRInitializationError, match="contiguous dense CPU tensor"):
        validate_zipformer_context_lookup(tmp_path, runtime_config)

    torch.save(torch.zeros(25, 3), context_lookup_path)
    validate_zipformer_context_lookup(tmp_path, runtime_config)

    for nonfinite_value in (float("nan"), float("inf"), float("-inf")):
        nonfinite_lookup = torch.zeros(25, 3)
        nonfinite_lookup[7, 1] = nonfinite_value
        torch.save(nonfinite_lookup, context_lookup_path)
        with pytest.raises(ASRInitializationError, match="finite"):
            validate_zipformer_context_lookup(tmp_path, runtime_config)


def test_validate_model_config_rejects_negative_right_padding() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.right_padding_samples = -1

    with pytest.raises(
        ASRInitializationError, match="non-negative signed-32-bit integer"
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_padding_only_minimum_profile() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        512,
        0,
        6000,
        make_export_args(),
    )
    runtime_config.audio_encoder_params.min_audio_seconds = 1e-8
    runtime_config.audio_encoder_params.right_padding_samples = 1

    with pytest.raises(
        ASRInitializationError, match="fit inside the minimum audio profile"
    ):
        validate_model_config(runtime_config)


def test_validate_zipformer_rejects_missing_ctc_head() -> None:
    state_dict = make_state_dict("ctc_greedy_search")
    del state_dict["ctc_output.1.weight"]

    with pytest.raises(RuntimeError, match="does not contain the ctc_output.1 head"):
        validate_zipformer(
            make_model_config(),
            state_dict,
            512,
            make_export_args("ctc_greedy_search"),
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
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


@pytest.mark.parametrize("value", (None, 2.0, "2"))
def test_validate_zipformer_rejects_noninteger_context_size(
    value: float | str | None,
) -> None:
    model_config = make_model_config()
    model_config.model_params.context_size = value

    with pytest.raises(ValueError, match="context_size.*positive integer"):
        validate_zipformer(
            model_config,
            make_state_dict(),
            512,
            make_export_args(),
        )


def test_adjust_zipformer_state_dict() -> None:
    downsample_bias = torch.tensor([0.0, 1.0])
    log_scale = torch.tensor(0.5)
    pointwise_weight = torch.randn(12, 4, 1, 1)
    conv1_weight = torch.randn(8, 1, 3, 3)
    conv2_weight = torch.randn(16, 8, 3, 3)
    conv3_weight = torch.randn(32, 16, 3, 3)
    out_combiner_scale = torch.rand(6)
    state_dict = OrderedDict(
        (
            ("encoder_embed.out_norm.log_scale", log_scale),
            ("encoder_embed.conv.0.weight", conv1_weight),
            ("encoder_embed.conv.4.weight", conv2_weight),
            ("encoder_embed.conv.7.weight", conv3_weight),
            ("encoder_embed.convnext.pointwise_conv1.weight", pointwise_weight),
            ("encoder.encoders.0.downsample.bias", downsample_bias),
            ("encoder.encoders.1.downsample.bias", downsample_bias),
            (
                "encoder.encoders.0.layers.0.self_attn_weights.in_proj.weight",
                torch.randn(8, 4),
            ),
            (
                "encoder.encoders.0.layers.0.bypass_scale",
                torch.ones(4),
            ),
            (
                "encoder.encoders.1.out_combiner.bypass_scale",
                out_combiner_scale,
            ),
            ("joiner.encoder_proj.weight", torch.randn(4, 8)),
        ),
    )

    modified_state_dict = adjust_state_dict(state_dict)

    assert set(state_dict).isdisjoint(modified_state_dict)
    torch.testing.assert_close(
        modified_state_dict["subsampling.out_norm.scale"],
        torch.exp(log_scale),
    )
    torch.testing.assert_close(
        modified_state_dict["encoder_1.downsample.weights"],
        torch.zeros(1, 1),
    )
    torch.testing.assert_close(
        modified_state_dict["encoder_2.downsample.weights"],
        torch.softmax(downsample_bias, dim=0).unsqueeze(1),
    )
    torch.testing.assert_close(
        modified_state_dict["projection_output.weight"],
        state_dict["joiner.encoder_proj.weight"],
    )
    torch.testing.assert_close(
        modified_state_dict["subsampling.pointwise_conv1.weight"],
        pointwise_weight[:, :, 0, 0],
    )
    torch.testing.assert_close(
        modified_state_dict["subsampling.conv1.weight"], conv1_weight
    )
    torch.testing.assert_close(
        modified_state_dict["subsampling.conv2.weight"], conv2_weight
    )
    torch.testing.assert_close(
        modified_state_dict["subsampling.conv3.weight"], conv3_weight
    )
    assert "encoder_1.layers.0.bypass_scale" not in modified_state_dict
    assert modified_state_dict["encoder_1.bypass_scale"].shape == (4,)
    assert modified_state_dict["encoder_1.bypass_scale"].dtype == torch.float32
    torch.testing.assert_close(
        modified_state_dict["encoder_2.bypass_scale"], out_combiner_scale
    )


def test_adjust_zipformer_state_dict_rejects_invalid_pointwise_kernel() -> None:
    state_dict = OrderedDict(
        (
            (
                "encoder_embed.convnext.pointwise_conv1.weight",
                torch.zeros(8, 4, 1, 2),
            ),
        )
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


def test_export_zipformer_rejects_missing_transducer_joiner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_export_args()
    args.output_dir = tmp_path
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="joiner was not initialized"):
        export_model_to_onnx(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            None,
            make_model_config(),
            args,
        )


@pytest.mark.parametrize(
    ("decoder_type", "expected_decoder_batch"),
    (
        ("transducer_greedy_search", 3),
        ("transducer_modified_beam_search", 12),
    ),
)
def test_export_zipformer_onnx_uses_context_cache_and_fixed_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_type: str,
    expected_decoder_batch: int,
) -> None:
    class FakeDecoder:
        def make_context_lookup(self, chunk_size: int) -> torch.Tensor:
            assert chunk_size == 8192
            return torch.arange(12, dtype=torch.float16).reshape(3, 4)

    class FakeJoiner:
        output_proj = torch.nn.Linear(1, 1, dtype=torch.float16)

    calls: list[tuple[object, tuple[torch.Tensor, ...], Path, dict[str, object]]] = []
    saved: list[tuple[torch.Tensor, Path]] = []

    def record_export(
        module: object,
        inputs: tuple[torch.Tensor, ...],
        path: Path,
        **kwargs: object,
    ) -> None:
        calls.append((module, inputs, path, kwargs))

    def record_save(value: torch.Tensor, path: Path) -> None:
        saved.append((value, path))

    monkeypatch.setattr(zipformer_exporter.torch.onnx, "export", record_export)
    monkeypatch.setattr(zipformer_exporter.torch, "save", record_save)
    args = make_export_args(decoder_type)
    args.output_dir = tmp_path
    args.batch_size = 3
    args.beam = 4
    args.opt_audio_seconds = 2.5
    encoder = object()
    decoder = FakeDecoder()
    joiner = FakeJoiner()

    encoder_path, decoder_path = export_model_to_onnx(
        encoder,  # type: ignore[arg-type]
        decoder,  # type: ignore[arg-type]
        joiner,  # type: ignore[arg-type]
        make_model_config(),
        args,
    )

    assert encoder_path == tmp_path / "zipformer.onnx"
    assert decoder_path == tmp_path / "decoder.onnx"
    assert len(calls) == 2
    assert len(saved) == 1
    torch.testing.assert_close(
        saved[0][0],
        torch.arange(12, dtype=torch.float16).reshape(3, 4),
    )
    assert saved[0][1] == tmp_path / "decoder_contexts.pt"

    encoder_module, encoder_inputs, encoder_export_path, encoder_kwargs = calls[0]
    assert encoder_module is encoder
    assert encoder_export_path == encoder_path
    assert tuple(encoder_inputs[0].shape) == (3, 40_200)
    assert tuple(encoder_inputs[1].shape) == (3,)
    assert encoder_inputs[0].dtype == torch.float32
    assert encoder_inputs[1].dtype == torch.int64
    assert encoder_kwargs["input_names"] == ("audio", "audio_lengths")
    assert encoder_kwargs["output_names"] == (
        "encoder_output",
        "encoder_output_lengths",
    )
    assert encoder_kwargs["opset_version"] == zipformer_exporter.ONNX_OPSET_VERSION
    dynamic_shapes = encoder_kwargs["dynamic_shapes"]
    assert isinstance(dynamic_shapes, dict)
    assert set(dynamic_shapes) == {"audio", "audio_lengths"}
    assert set(dynamic_shapes["audio"]) == {1}
    assert dynamic_shapes["audio_lengths"] == {}

    decoder_module, decoder_inputs, decoder_export_path, decoder_kwargs = calls[1]
    assert decoder_module is joiner
    assert decoder_export_path == decoder_path
    assert tuple(decoder_inputs[0].shape) == (expected_decoder_batch, 512)
    assert tuple(decoder_inputs[1].shape) == (expected_decoder_batch, 512)
    assert decoder_inputs[0].dtype == torch.float16
    assert decoder_inputs[1].dtype == torch.float16
    assert decoder_kwargs["input_names"] == ("decoder_input", "encoder_output")
    assert decoder_kwargs["output_names"] == ("tokens_log_prob",)
    assert decoder_kwargs["opset_version"] == zipformer_exporter.ONNX_OPSET_VERSION
    assert "dynamic_shapes" not in decoder_kwargs


def test_export_zipformer_ctc_omits_decoder_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def record_export(
        _module: object,
        _inputs: tuple[torch.Tensor, ...],
        path: Path,
        **_kwargs: object,
    ) -> None:
        calls.append(path)

    monkeypatch.setattr(zipformer_exporter.torch.onnx, "export", record_export)
    args = make_export_args("ctc_greedy_search")
    args.output_dir = tmp_path

    encoder_path, decoder_path = export_model_to_onnx(
        object(),  # type: ignore[arg-type]
        None,
        None,
        make_model_config(),
        args,
    )

    assert encoder_path == tmp_path / "zipformer.onnx"
    assert decoder_path is None
    assert calls == [encoder_path]


def test_export_zipformer_rejects_missing_blank_before_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingBlankTokenizer(FakeTokenizer):
        def piece_to_id(self, piece: str) -> int:
            assert piece == "<blk>"
            return self.unk_id()

        def id_to_piece(self, token_id: int) -> str:
            return "<unk>" if token_id == self.unk_id() else str(token_id)

    args = make_export_args()
    args.output_dir = tmp_path / "bundle"
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    args.model_path = model_dir / "model.pt"
    args.model_path.write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("model: config\n")
    (model_dir / "bpe.model").write_bytes(b"tokenizer")
    monkeypatch.setattr(zipformer_exporter.OmegaConf, "load", lambda _path: object())
    monkeypatch.setattr(
        zipformer_exporter.spm,
        "SentencePieceProcessor",
        MissingBlankTokenizer,
    )

    def reject_checkpoint_load(*_args: object, **_kwargs: object) -> None:
        pytest.fail("checkpoint loading must happen after tokenizer validation")

    monkeypatch.setattr(zipformer_exporter.torch, "load", reject_checkpoint_load)

    with pytest.raises(ValueError, match="exact <blk> piece"):
        zipformer_exporter.export_zipformer(args)
