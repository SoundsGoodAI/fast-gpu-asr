#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for Parakeet bundle export, metadata conversion, and validation."""

import argparse
import io
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import fast_gpu_asr.export.export_parakeet as parakeet_exporter
from fast_gpu_asr.export.export_parakeet import (
    adjust_state_dict,
    export_model_to_onnx,
    export_parakeet,
    extract_member,
    extract_parakeet_archive,
    get_subsampling_batch_partitions,
    make_model,
    make_runtime_config,
    parse_args,
)
from fast_gpu_asr.export.export_utils import validate_parakeet
from fast_gpu_asr.utils import ASRInitializationError, validate_model_config


def make_model_config() -> DictConfig:
    """Return a compact valid source configuration for Parakeet export tests."""

    return OmegaConf.create(
        {
            "sample_rate": 16000,
            "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
            "preprocessor": {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window": "hann",
                "frame_splicing": 1,
                "window_stride": 0.01,
                "window_size": 0.025,
                "features": 128,
                "n_fft": 512,
                "log": True,
                "pad_to": 0,
                "pad_value": 0.0,
            },
            "encoder": {
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "self_attention_model": "rel_pos",
                "att_context_style": "regular",
                "xscaling": False,
                "untie_biases": True,
                "use_bias": False,
                "conv_norm_type": "batch_norm",
                "att_context_size": [-1, -1],
                "n_layers": 24,
                "d_model": 1024,
                "subsampling_conv_channels": 256,
                "ff_expansion_factor": 4,
                "n_heads": 8,
                "pos_emb_max_len": 5000,
                "conv_kernel_size": 9,
            },
            "decoder": {
                "blank_as_pad": True,
                "vocab_size": 1024,
                "prednet": {
                    "pred_hidden": 640,
                    "pred_rnn_layers": 2,
                },
            },
            "joint": {
                "jointnet": {
                    "activation": "relu",
                    "encoder_hidden": 1024,
                    "joint_hidden": 640,
                },
                "num_extra_outputs": 5,
            },
            "decoding": {"greedy": {"max_symbols": 10}},
        },
    )


def make_export_args() -> argparse.Namespace:
    """Return valid Parakeet exporter arguments suitable for local mutation."""

    return argparse.Namespace(
        batch_size=1,
        beam=6,
        debug=False,
        decoder_precision="fp32",
        decoder_type="transducer_modified_beam_search",
        encoder_precision="fp32",
        min_audio_seconds=0.5,
        opt_audio_seconds=15.0,
        max_audio_seconds=40.0,
        model_path=Path("model.nemo"),
        optimization_level=5,
        output_dir=Path("output"),
    )


def test_export_parakeet_rejects_destination_containing_source(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    model_path = output_dir / "model.nemo"
    model_path.write_bytes(b"source")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    with pytest.raises(ValueError, match="contains required source file"):
        export_parakeet(args)

    assert model_path.read_bytes() == b"source"


def test_export_parakeet_rejects_missing_source_before_replacing_output(
    tmp_path: Path,
) -> None:
    """Preserve an existing bundle when the source archive does not exist."""

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel_path = output_dir / "existing.trt"
    sentinel_path.write_bytes(b"existing")
    args = make_export_args()
    args.model_path = tmp_path / "missing.nemo"
    args.output_dir = output_dir

    with pytest.raises(FileNotFoundError, match=str(args.model_path)):
        export_parakeet(args)

    assert sentinel_path.read_bytes() == b"existing"


def test_export_parakeet_writes_directly_to_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.nemo"
    model_path.write_bytes(b"source")
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel_path = output_dir / "old.trt"
    sentinel_path.write_bytes(b"known-good")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    def fail_export(*_args: object) -> None:
        (output_dir / "model.trt").write_bytes(b"partial")
        raise RuntimeError("engine build failed")

    monkeypatch.setattr(parakeet_exporter, "extract_parakeet_archive", fail_export)

    with pytest.raises(RuntimeError, match="engine build failed"):
        export_parakeet(args)

    assert not sentinel_path.exists()
    assert (output_dir / "model.trt").read_bytes() == b"partial"


def test_export_parakeet_forces_greedy_beam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.nemo"
    model_path.write_bytes(b"source")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = tmp_path / "bundle"
    args.decoder_type = "transducer_greedy_search"
    args.beam = 6

    def stop_export(*_args: object) -> None:
        raise RuntimeError("stop export")

    monkeypatch.setattr(
        parakeet_exporter,
        "extract_parakeet_archive",
        stop_export,
    )

    with pytest.raises(RuntimeError, match="stop export"):
        export_parakeet(args)

    assert args.beam == 1


def test_parse_args_reads_required_values_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fast-gpu-asr-export-parakeet",
            "--model-path",
            "model.nemo",
            "--output-dir",
            "output",
            "--batch-size",
            "1",
            "--decoder-type",
            "transducer_greedy_search",
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

    assert args.model_path == Path("model.nemo")
    assert args.output_dir == Path("output")
    assert args.batch_size == 1
    assert args.decoder_type == "transducer_greedy_search"
    assert args.beam == 1
    assert args.encoder_precision == "fp32"
    assert args.decoder_precision == "fp32"
    assert args.min_audio_seconds == 0.1
    assert args.opt_audio_seconds == 15.0
    assert args.max_audio_seconds == 40.0
    assert args.optimization_level == 5
    assert args.debug is False


def test_export_parakeet_onnx_uses_fixed_decoder_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDecoder:
        output_proj = torch.nn.Linear(1, 1, dtype=torch.float16)

    calls: list[tuple[object, tuple[torch.Tensor, ...], Path, dict[str, object]]] = []

    def record_export(
        module: object,
        inputs: tuple[torch.Tensor, ...],
        path: Path,
        **kwargs: object,
    ) -> None:
        calls.append((module, inputs, path, kwargs))

    monkeypatch.setattr(parakeet_exporter.torch.onnx, "export", record_export)
    args = make_export_args()
    args.output_dir = tmp_path
    args.batch_size = 3
    args.beam = 4
    args.opt_audio_seconds = 2.5
    encoder = object()
    decoder = FakeDecoder()

    encoder_path, decoder_path = export_model_to_onnx(
        encoder,  # type: ignore[arg-type]
        decoder,  # type: ignore[arg-type]
        make_model_config(),
        args,
    )

    assert encoder_path == tmp_path / "parakeet.onnx"
    assert decoder_path == tmp_path / "tdt_decoder.onnx"
    assert len(calls) == 2

    encoder_module, encoder_inputs, encoder_export_path, encoder_kwargs = calls[0]
    assert encoder_module is encoder
    assert encoder_export_path == encoder_path
    assert tuple(encoder_inputs[0].shape) == (3, 40_000)
    assert encoder_inputs[0].dtype == torch.float32
    assert torch.count_nonzero(encoder_inputs[0]).item() == 0
    assert tuple(encoder_inputs[1].shape) == (3,)
    assert encoder_inputs[1].dtype == torch.int64
    assert torch.equal(
        encoder_inputs[1],
        torch.full((3,), 40_000, dtype=torch.int64),
    )
    assert encoder_kwargs["input_names"] == ("audio", "audio_lengths")
    assert encoder_kwargs["output_names"] == (
        "encoder_output",
        "encoder_output_lengths",
    )
    dynamic_shapes = encoder_kwargs["dynamic_shapes"]
    assert dynamic_shapes == {
        "audio": {1: torch.export.Dim.DYNAMIC},
        "audio_lengths": {},
    }
    assert encoder_kwargs["opset_version"] == parakeet_exporter.ONNX_OPSET_VERSION

    decoder_module, decoder_inputs, decoder_export_path, decoder_kwargs = calls[1]
    assert decoder_module is decoder
    assert decoder_export_path == decoder_path
    assert tuple(decoder_inputs[0].shape) == (12, 1024)
    assert tuple(decoder_inputs[1].shape) == (12, 1)
    assert tuple(decoder_inputs[2].shape) == (2, 12, 640)
    assert tuple(decoder_inputs[3].shape) == (2, 12, 640)
    assert decoder_inputs[0].dtype == torch.float16
    assert decoder_inputs[1].dtype == torch.int32
    assert decoder_inputs[2].dtype == torch.float16
    assert decoder_inputs[3].dtype == torch.float16
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in decoder_inputs)
    assert decoder_kwargs["input_names"] == (
        "encoder_output",
        "targets",
        "input_states_1",
        "input_states_2",
    )
    assert decoder_kwargs["output_names"] == (
        "token_log_probs",
        "duration_log_probs",
        "output_states_1",
        "output_states_2",
    )
    assert "dynamic_shapes" not in decoder_kwargs
    assert decoder_kwargs["opset_version"] == parakeet_exporter.ONNX_OPSET_VERSION


@pytest.mark.parametrize("debug", (False, True))
def test_export_parakeet_validates_exact_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
) -> None:
    """Wire every export stage correctly and validate only published artifacts."""

    output_dir = tmp_path / "bundle"
    source_config_path = tmp_path / "source.yaml"
    checkpoint_path = tmp_path / "weights.ckpt"
    tokenizer_path = tmp_path / "source.model"
    source_config = make_model_config()
    OmegaConf.save(source_config, source_config_path)
    checkpoint_path.write_bytes(b"checkpoint")
    tokenizer_path.write_bytes(b"tokenizer")
    args = make_export_args()
    args.batch_size = 3
    args.beam = 4
    args.debug = debug
    args.encoder_precision = "fp16"
    args.decoder_precision = "bf16"
    args.model_path = tmp_path / "model.nemo"
    args.model_path.write_bytes(b"archive")
    args.output_dir = output_dir
    adjusted_state_dict = OrderedDict((("adjusted", torch.zeros(1)),))
    checkpoint_state_dict = OrderedDict((("source", torch.ones(1)),))
    encoder = object()
    decoder = object()
    runtime_configs: list[DictConfig] = []
    build_calls: list[
        tuple[
            Path,
            Path,
            dict[
                str,
                tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            ],
            int,
        ]
    ] = []
    events: list[str] = []

    def extract_archive(
        model_path: Path, temporary_dir: Path
    ) -> tuple[Path, Path, Path]:
        events.append("extract")
        assert model_path == args.model_path
        assert temporary_dir.is_dir()
        assert temporary_dir != output_dir
        return source_config_path, checkpoint_path, tokenizer_path

    def validate_source(
        model_config: DictConfig,
        export_args: argparse.Namespace,
    ) -> None:
        events.append("source-validate")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert export_args is args

    def load_checkpoint(
        path: Path,
        *,
        map_location: torch.device,
        weights_only: bool,
    ) -> OrderedDict[str, torch.Tensor]:
        events.append("checkpoint-load")
        assert path == checkpoint_path
        assert map_location == torch.device("cpu")
        assert weights_only is True
        return checkpoint_state_dict

    def adjust_checkpoint(
        state_dict: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        events.append("checkpoint-adjust")
        assert state_dict is checkpoint_state_dict
        return adjusted_state_dict

    def get_partitions(
        model_config: DictConfig,
        export_args: argparse.Namespace,
    ) -> int:
        events.append("partition")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert export_args is args
        return 2

    def make_export_model(
        model_config: DictConfig,
        state_dict: OrderedDict[str, torch.Tensor],
        partitions: int,
        encoder_dtype: torch.dtype,
        decoder_dtype: torch.dtype,
    ) -> tuple[object, object]:
        events.append("model")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert state_dict is adjusted_state_dict
        assert partitions == 2
        assert encoder_dtype == torch.float16
        assert decoder_dtype == torch.bfloat16
        return encoder, decoder

    real_make_runtime_config = parakeet_exporter.make_runtime_config

    def record_runtime_config(
        model_config: DictConfig,
        export_args: argparse.Namespace,
    ) -> DictConfig:
        events.append("runtime-config")
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert export_args is args
        runtime_config = real_make_runtime_config(model_config, export_args)
        runtime_configs.append(runtime_config)
        return runtime_config

    def export_onnx(
        export_encoder: object,
        export_decoder: object,
        model_config: DictConfig,
        export_args: argparse.Namespace,
    ) -> tuple[Path, Path]:
        assert export_encoder is encoder
        assert export_decoder is decoder
        assert OmegaConf.to_container(model_config) == OmegaConf.to_container(
            source_config
        )
        assert export_args is args
        encoder_path = output_dir / parakeet_exporter.PARAKEET_ONNX_FILE
        decoder_path = output_dir / parakeet_exporter.PARAKEET_DECODER_ONNX_FILE
        encoder_path.write_bytes(b"encoder")
        decoder_path.write_bytes(b"decoder")
        events.append("onnx")
        return encoder_path, decoder_path

    def build_engine(
        onnx_path: Path,
        engine_path: Path,
        profiles: dict[
            str,
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        ],
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
            model_dir / parakeet_exporter.TOKENIZER_FILE
        ).read_bytes() == b"tokenizer"
        assert (model_dir / parakeet_exporter.MODEL_CONFIG_FILE).is_file()
        assert (model_dir / parakeet_exporter.PARAKEET_TENSORRT_FILE).is_file()
        assert (model_dir / parakeet_exporter.PARAKEET_DECODER_TENSORRT_FILE).is_file()
        assert (model_dir / parakeet_exporter.PARAKEET_ONNX_FILE).exists() is debug
        assert (
            model_dir / parakeet_exporter.PARAKEET_DECODER_ONNX_FILE
        ).exists() is debug

    monkeypatch.setattr(parakeet_exporter, "extract_parakeet_archive", extract_archive)
    monkeypatch.setattr(parakeet_exporter, "validate_parakeet", validate_source)
    monkeypatch.setattr(parakeet_exporter.torch, "load", load_checkpoint)
    monkeypatch.setattr(parakeet_exporter, "adjust_state_dict", adjust_checkpoint)
    monkeypatch.setattr(
        parakeet_exporter,
        "get_subsampling_batch_partitions",
        get_partitions,
    )
    monkeypatch.setattr(parakeet_exporter, "make_model", make_export_model)
    monkeypatch.setattr(
        parakeet_exporter,
        "make_runtime_config",
        record_runtime_config,
    )
    monkeypatch.setattr(parakeet_exporter, "export_model_to_onnx", export_onnx)
    monkeypatch.setattr(parakeet_exporter, "build_tensorrt_engine", build_engine)
    monkeypatch.setattr(parakeet_exporter, "remove_onnx_artifacts", remove_graph)
    monkeypatch.setattr(parakeet_exporter, "validate_model", validate_bundle)

    parakeet_exporter.export_parakeet(args)

    sample_rate = source_config.sample_rate
    expected_profile = tuple(
        (args.batch_size, round(seconds * sample_rate))
        for seconds in (
            args.min_audio_seconds,
            args.opt_audio_seconds,
            args.max_audio_seconds,
        )
    )
    assert build_calls == [
        (
            output_dir / parakeet_exporter.PARAKEET_ONNX_FILE,
            output_dir / parakeet_exporter.PARAKEET_TENSORRT_FILE,
            {"audio": expected_profile},
            args.optimization_level,
        ),
        (
            output_dir / parakeet_exporter.PARAKEET_DECODER_ONNX_FILE,
            output_dir / parakeet_exporter.PARAKEET_DECODER_TENSORRT_FILE,
            {},
            args.optimization_level,
        ),
    ]
    expected_events = [
        "extract",
        "source-validate",
        "checkpoint-load",
        "checkpoint-adjust",
        "partition",
        "model",
        "runtime-config",
        "onnx",
        f"build:{parakeet_exporter.PARAKEET_TENSORRT_FILE}",
        f"build:{parakeet_exporter.PARAKEET_DECODER_TENSORRT_FILE}",
    ]
    if not debug:
        expected_events.extend(
            (
                f"remove:{parakeet_exporter.PARAKEET_ONNX_FILE}",
                f"remove:{parakeet_exporter.PARAKEET_DECODER_ONNX_FILE}",
            )
        )
    expected_events.append("validate")
    assert events == expected_events
    assert events[-1] == "validate"


def test_make_model_wires_configuration_and_state_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct both modules with exact source metadata and strict state loading."""

    class FakeModule:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.loaded_state_dict: OrderedDict[str, torch.Tensor] | None = None
            self.strict: bool | None = None
            self.eval_called = False

        def load_state_dict(
            self,
            state_dict: OrderedDict[str, torch.Tensor],
            *,
            strict: bool,
        ) -> None:
            self.loaded_state_dict = state_dict
            self.strict = strict

        def eval(self) -> None:
            self.eval_called = True

    modules: list[FakeModule] = []

    def make_fake_module(**kwargs: object) -> FakeModule:
        module = FakeModule(**kwargs)
        modules.append(module)
        return module

    monkeypatch.setattr(parakeet_exporter, "ParakeetTDTEncoder", make_fake_module)
    monkeypatch.setattr(parakeet_exporter, "Decoder", make_fake_module)
    encoder_weight = torch.ones(1)
    embedding_weight = torch.ones(2)
    ignored_weight = torch.ones(3)
    state_dict = OrderedDict(
        (
            ("encoder.weight", encoder_weight),
            ("embedding.weight", embedding_weight),
            ("ignored.weight", ignored_weight),
        )
    )

    encoder, decoder = make_model(
        make_model_config(),
        state_dict,
        3,
        torch.float16,
        torch.bfloat16,
    )

    assert modules == [encoder, decoder]
    assert encoder.kwargs == {
        "samp_freq": 16_000,
        "frame_shift_ms": 10,
        "frame_length_ms": 25,
        "feature_dim": 128,
        "preemph": 0.97,
        "low_freq": 0,
        "high_freq": 8_000,
        "n_layers": 24,
        "model_dim": 1_024,
        "subsampling_conv_channels": 256,
        "feed_forward_expansion_factor": 4,
        "n_heads": 8,
        "pos_emb_max_len": 5_000,
        "conv_kernel_size": 9,
        "subsampling_batch_partitions": 3,
        "dtype": torch.float16,
    }
    assert decoder.kwargs == {
        "vocab_size": 1_024,
        "encoder_dim": 1_024,
        "decoder_dim": 640,
        "joiner_dim": 640,
        "pred_rnn_layers": 2,
        "num_extra_outputs": 5,
        "dtype": torch.bfloat16,
    }
    assert encoder.loaded_state_dict is not None
    assert tuple(encoder.loaded_state_dict) == ("encoder.weight",)
    assert encoder.loaded_state_dict["encoder.weight"] is encoder_weight
    assert decoder.loaded_state_dict is not None
    assert tuple(decoder.loaded_state_dict) == ("embedding.weight",)
    assert decoder.loaded_state_dict["embedding.weight"] is embedding_weight
    assert encoder.strict is True
    assert decoder.strict is True
    assert encoder.eval_called is True
    assert decoder.eval_called is True


def test_make_parakeet_runtime_config() -> None:
    """Publish every runtime field and omit source-only architecture metadata."""

    runtime_config = make_runtime_config(
        make_model_config(),
        make_export_args(),
    )

    validate_model_config(runtime_config)
    assert OmegaConf.to_container(runtime_config, resolve=True) == {
        "model_type": "parakeet_asr",
        "decoder_type": "transducer_modified_beam_search",
        "model_samplerate": 16_000,
        "vocab_size": 1_024,
        "blank_id": 1_024,
        "audio_encoder_params": {
            "feature_dim": 128,
            "frame_shift_ms": 10,
            "n_layers": 24,
            "model_dim": 1_024,
            "pos_emb_max_len": 5_000,
            "subsampling_factor": 8,
            "min_audio_seconds": 0.5,
            "opt_audio_seconds": 15.0,
            "max_audio_seconds": 40.0,
        },
        "decoder_params": {
            "encoder_dim": 1_024,
            "decoder_dim": 640,
            "joiner_dim": 640,
            "pred_rnn_layers": 2,
            "num_extra_outputs": 5,
            "beam": 6,
            "blank_penalty": 0.0,
            "max_symbols_per_timestep": 10,
            "tdt_durations": [0, 1, 2, 3, 4],
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "audio_encoder_params.model_dim",
            512,
            "model_dim and decoder_params.encoder_dim",
        ),
        (
            "decoder_params.num_extra_outputs",
            4,
            "number of decoder_params.tdt_durations",
        ),
        (
            "decoder_params.tdt_durations",
            [1, 2, 3, 4, 5],
            "must contain zero",
        ),
        (
            "decoder_params.tdt_durations",
            [0, 1, 1, 3, 4],
            "must contain unique values",
        ),
        (
            "decoder_params.tdt_durations",
            [0, 1, 2, 3, 1 << 31],
            "signed 32-bit integers",
        ),
        (
            "audio_encoder_params.pos_emb_max_len",
            0,
            "positive integer",
        ),
        ("blank_id", 1024.0, "Expected blank_id=1024"),
        ("decoder_params.beam", 1025, "beam <= vocab_size"),
    ),
)
def test_validate_model_config_rejects_inconsistent_parakeet_values(
    field: str,
    value: float | int | list[int],
    message: str,
) -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        make_export_args(),
    )
    OmegaConf.update(runtime_config, field, value)

    with pytest.raises(ASRInitializationError, match=message):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_search_candidate_overflow() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        make_export_args(),
    )
    runtime_config.vocab_size = 50_000
    runtime_config.blank_id = 50_000
    runtime_config.decoder_params.beam = 50_000

    with pytest.raises(
        ASRInitializationError, match="Parakeet per-utterance search table"
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_rejects_invalid_audio_profile() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        make_export_args(),
    )
    runtime_config.audio_encoder_params.opt_audio_seconds = 0.25

    with pytest.raises(
        ASRInitializationError,
        match="min_audio_seconds <= opt_audio_seconds",
    ):
        validate_model_config(runtime_config)


def test_validate_model_config_accepts_parakeet_greedy_mode() -> None:
    runtime_config = make_runtime_config(
        make_model_config(),
        make_export_args(),
    )
    runtime_config.decoder_type = "transducer_greedy_search"
    runtime_config.decoder_params.beam = 1

    validate_model_config(runtime_config)


def test_validate_parakeet_accepts_beam_one_modified_beam() -> None:
    args = make_export_args()
    args.beam = 1

    validate_parakeet(make_model_config(), args)
    runtime_config = make_runtime_config(make_model_config(), args)

    assert runtime_config.decoder_type == "transducer_modified_beam_search"
    assert runtime_config.decoder_params.beam == 1


def test_validate_parakeet_accepts_greedy_export() -> None:
    args = make_export_args()
    args.decoder_type = "transducer_greedy_search"
    args.beam = 1

    validate_parakeet(make_model_config(), args)
    runtime_config = make_runtime_config(make_model_config(), args)

    assert runtime_config.decoder_type == "transducer_greedy_search"
    assert runtime_config.decoder_params.beam == 1


def test_validate_parakeet_rejects_beam_larger_than_vocabulary() -> None:
    model_config = make_model_config()
    args = make_export_args()
    args.beam = model_config.decoder.vocab_size + 1

    with pytest.raises(ValueError, match="beam must not exceed decoder.vocab_size"):
        validate_parakeet(model_config, args)


def test_validate_parakeet_rejects_decoder_capacity_overflow() -> None:
    args = make_export_args()
    args.batch_size = (1 << 31) // args.beam + 1

    with pytest.raises(ValueError, match=r"batch_size \* beam"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("field", "value", "tensor_name"),
    (
        ("joint.jointnet.encoder_hidden", 357_913_944, "encoder_output"),
        ("decoder.prednet.pred_hidden", (1 << 31) // 12 + 1, "input_states_1"),
        ("decoder.vocab_size", (1 << 31) // 6, "token_log_probs"),
    ),
)
def test_validate_parakeet_rejects_decoder_tensor_volume_overflow(
    field: str, value: int, tensor_name: str
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)
    if field == "joint.jointnet.encoder_hidden":
        model_config.encoder.d_model = value

    with pytest.raises(ValueError, match=rf"tensor {tensor_name} exceeds"):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize(
    ("field", "value", "parameter_name"),
    (
        ("encoder.d_model", 50_000, "encoder feed-forward weight"),
        ("decoder.vocab_size", 4_000_000, "decoder embedding weight"),
        ("decoder.prednet.pred_hidden", 25_000, "decoder recurrent weight"),
        ("joint.jointnet.joint_hidden", 3_000_000, "encoder projection weight"),
        (
            "joint.jointnet.joint_hidden",
            2_086_000,
            "output projection weight",
        ),
    ),
)
def test_validate_parakeet_rejects_parameter_tensor_overflow(
    field: str,
    value: int,
    parameter_name: str,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)
    if field == "encoder.d_model":
        model_config.joint.jointnet.encoder_hidden = value

    with pytest.raises(ValueError, match=parameter_name):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_encoder_decoder_interface_mismatch() -> None:
    model_config = make_model_config()
    model_config.joint.jointnet.encoder_hidden = 512

    with pytest.raises(ValueError, match="encoder.d_model.*encoder_hidden must match"):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize(
    ("durations", "message"),
    (
        ([], "non-negative signed 32-bit integers"),
        ([0, 1, 2, 3, -1], "non-negative signed 32-bit integers"),
        ([0, 1, 1, 3, 4], "unique values"),
        ([1, 2, 3, 4, 5], "contain zero"),
        ([0, 1, 2, 3], "must match joint.num_extra_outputs"),
        ([0, 1, 2, 3, 1 << 31], "signed 32-bit integers"),
    ),
)
def test_validate_parakeet_rejects_invalid_tdt_durations(
    durations: list[int],
    message: str,
) -> None:
    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = durations

    with pytest.raises(ValueError, match=message):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_only_zero_duration() -> None:
    """Require at least one advancing TDT duration even for one duration output."""

    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = [0]
    model_config.joint.num_extra_outputs = 1

    with pytest.raises(ValueError, match="zero and at least one positive duration"):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_audio_sample_dimension_overflow() -> None:
    args = make_export_args()
    args.max_audio_seconds = 1e308

    with pytest.raises(ValueError, match="signed 32-bit TensorRT audio-sample"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("batch_size", 0, "batch_size must be positive"),
        ("beam", 0, "beam must be positive"),
        ("min_audio_seconds", 0.0, "Expected 0 < min_audio_seconds"),
        ("opt_audio_seconds", 0.25, "min_audio_seconds <= opt_audio_seconds"),
        ("max_audio_seconds", 10.0, "opt_audio_seconds <= max_audio_seconds"),
    ),
)
def test_validate_parakeet_rejects_invalid_export_arguments(
    field: str,
    value: float | int,
    message: str,
) -> None:
    """Reject invalid fixed-batch and duration-profile arguments."""

    args = make_export_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        validate_parakeet(make_model_config(), args)


def test_validate_parakeet_normalization_length_boundary() -> None:
    """Require two feature frames for unbiased per-feature variance."""

    args = make_export_args()
    args.min_audio_seconds = 320 / 16_000
    validate_parakeet(make_model_config(), args)

    args.min_audio_seconds = 319 / 16_000
    with pytest.raises(ValueError, match="at least 320 samples"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("encoder.d_model", 1023, "d_model must be even"),
        ("encoder.n_heads", 7, "d_model must be divisible"),
        ("encoder.conv_kernel_size", 8, "conv_kernel_size must be odd"),
    ),
)
def test_validate_parakeet_rejects_incompatible_encoder_dimensions(
    field: str, value: int, message: str
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError, match=message):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_unaligned_fp32_convolution_channels() -> None:
    model_config = make_model_config()
    model_config.encoder.d_model = 1026
    model_config.encoder.n_heads = 2
    model_config.joint.jointnet.encoder_hidden = 1026

    with pytest.raises(ValueError, match="divisible by 4 for fp32"):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize("precision", ("fp16", "bf16"))
def test_validate_parakeet_accepts_paired_convolution_channels(
    precision: str,
) -> None:
    model_config = make_model_config()
    model_config.encoder.d_model = 1026
    model_config.encoder.n_heads = 2
    model_config.joint.jointnet.encoder_hidden = 1026
    args = make_export_args()
    args.encoder_precision = precision

    validate_parakeet(model_config, args)


def test_validate_parakeet_positional_capacity_boundary() -> None:
    model_config = make_model_config()
    args = make_export_args()
    max_samples = round(args.max_audio_seconds * model_config.sample_rate)
    feature_frames = (
        max_samples
        // round(model_config.preprocessor.window_stride * model_config.sample_rate)
        + 1
    )
    encoder_frames = (((feature_frames + 1) // 2 + 1) // 2 + 1) // 2
    model_config.encoder.pos_emb_max_len = encoder_frames

    validate_parakeet(model_config, args)

    model_config.encoder.pos_emb_max_len -= 1
    with pytest.raises(ValueError, match="maximum profile produces"):
        validate_parakeet(model_config, args)


def test_validate_parakeet_flash_attention_capacity_boundary() -> None:
    args = make_export_args()
    args.max_audio_seconds = 655_359 / 16_000

    validate_parakeet(make_model_config(), args)

    args.max_audio_seconds = 655_360 / 16_000
    with pytest.raises(ValueError, match="supports at most 512"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_rate", 8_000),
        ("preprocessor.sample_rate", 8_000),
        ("preprocessor.normalize", "all_features"),
        ("preprocessor.window", "hamming"),
        ("preprocessor.frame_splicing", 2),
        ("preprocessor.n_fft", 1024),
        ("preprocessor.log", False),
        ("preprocessor.mag_power", 1.0),
        ("preprocessor.mel_norm", None),
        ("preprocessor.pad_to", 16),
        ("preprocessor.pad_value", 1.0),
        ("encoder.subsampling", "striding"),
        ("encoder.subsampling_factor", 4),
        ("encoder.self_attention_model", "abs_pos"),
        ("encoder.att_context_style", "chunked_limited"),
        ("encoder.xscaling", True),
        ("encoder.untie_biases", False),
        ("encoder.use_bias", True),
        ("encoder.conv_norm_type", "layer_norm"),
        ("encoder.att_context_size", [-1, 0]),
        ("decoder.blank_as_pad", False),
        ("joint.jointnet.activation", "swish"),
    ),
)
def test_validate_parakeet_rejects_unsupported_fixed_values(
    field: str,
    value: bool | float | int | str | list[int] | None,
) -> None:
    """Reject every fixed source-model semantic required by the exporter."""

    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_unsupported_decoder_type() -> None:
    """Reject decoder modes that the Parakeet TDT bundle cannot provide."""

    args = make_export_args()
    args.decoder_type = "ctc_greedy_search"

    with pytest.raises(ValueError, match="supports only"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(("window_size", "n_fft"), ((0.016, 256), (0.033, 1024)))
def test_validate_parakeet_rejects_fft_size_mismatch(
    window_size: float,
    n_fft: int,
) -> None:
    model_config = make_model_config()
    model_config.preprocessor.window_size = window_size

    with pytest.raises(ValueError, match=rf"requires n_fft={n_fft}"):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_feature_workspace_overflow_boundary() -> None:
    args = make_export_args()
    args.batch_size = 259

    validate_parakeet(make_model_config(), args)

    args.batch_size = 260
    with pytest.raises(ValueError, match="workspace limit"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("batch_size", "expected_partitions"),
    ((64, 1), (128, 2), (131, 3), (256, 4)),
)
def test_subsampling_batch_uses_minimum_cask_safe_partitions(
    batch_size: int,
    expected_partitions: int,
) -> None:
    """Choose the smallest safe partition count at production batch sizes."""

    args = make_export_args()
    args.batch_size = batch_size
    args.max_audio_seconds = 40.0

    assert (
        get_subsampling_batch_partitions(make_model_config(), args)
        == expected_partitions
    )


def test_subsampling_batch_rejects_single_item_over_cask_limit() -> None:
    args = make_export_args()
    args.max_audio_seconds = 20_000.0

    with pytest.raises(ValueError, match="One Parakeet subsampling item exceeds"):
        get_subsampling_batch_partitions(make_model_config(), args)


def add_archive_member(
    archive: tarfile.TarFile,
    name: str,
    data: bytes = b"data",
) -> None:
    """Add one regular in-memory file to a test tar archive."""

    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize(
    "tokenizer_reference",
    ("nemo:nested/tokenizer.model", "nested/tokenizer.model"),
    ids=("nemo-uri", "archive-path"),
)
def test_extract_parakeet_archive_resolves_tokenizer_reference(
    tmp_path: Path,
    tokenizer_reference: str,
) -> None:
    """Extract required files and resolve both supported tokenizer path forms."""

    archive_path = tmp_path / "model.nemo"
    config = OmegaConf.create({"tokenizer": {"model_path": tokenizer_reference}})
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(
            archive,
            "model/model_config.yaml",
            OmegaConf.to_yaml(config).encode(),
        )
        add_archive_member(archive, "model/model_weights.ckpt", b"checkpoint")
        add_archive_member(archive, "artifacts/tokenizer.model", b"tokenizer")
    output_dir = tmp_path / "extracted"
    output_dir.mkdir()

    config_path, checkpoint_path, tokenizer_path = extract_parakeet_archive(
        archive_path,
        output_dir,
    )

    assert config_path == output_dir / "model_config.yaml"
    assert checkpoint_path == output_dir / "model_weights.ckpt"
    assert tokenizer_path == output_dir / "tokenizer.model"
    assert OmegaConf.to_container(OmegaConf.load(config_path)) == {
        "tokenizer": {"model_path": tokenizer_reference}
    }
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert tokenizer_path.read_bytes() == b"tokenizer"


def test_extract_member_matches_exact_basename(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(archive, "attacker_model_config.yaml", b"wrong")
        add_archive_member(archive, "nested/model_config.yaml", b"right")

    with tarfile.open(archive_path) as archive:
        output_path = extract_member(archive, "model_config.yaml", tmp_path)

    assert output_path.read_bytes() == b"right"


def test_extract_member_rejects_ambiguous_basename(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(archive, "first/model_config.yaml")
        add_archive_member(archive, "second/model_config.yaml")

    with (
        tarfile.open(archive_path) as archive,
        pytest.raises(ValueError, match="Expected one model_config.yaml"),
    ):
        extract_member(archive, "model_config.yaml", tmp_path)


def test_extract_member_rejects_nonregular_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("model_config.yaml")
        member.type = tarfile.SYMTYPE
        member.linkname = "elsewhere"
        archive.addfile(member)

    with (
        tarfile.open(archive_path) as archive,
        pytest.raises(FileNotFoundError, match="Missing model_config.yaml"),
    ):
        extract_member(archive, "model_config.yaml", tmp_path)


def test_extract_member_rejects_unreadable_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(archive, "model_config.yaml")

    with tarfile.open(archive_path) as archive:
        monkeypatch.setattr(archive, "extractfile", lambda _member: None)
        with pytest.raises(FileNotFoundError, match="Unable to read"):
            extract_member(archive, "model_config.yaml", tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("preprocessor.window_stride", "0.01"),
        ("preprocessor.window_size", "0.025"),
        ("preprocessor.preemph", "0.97"),
        ("preprocessor.lowfreq", "0"),
        ("preprocessor.highfreq", "8000"),
        ("preprocessor.features", "128"),
        ("encoder.n_layers", "24"),
        ("encoder.d_model", "1024"),
        ("encoder.subsampling_conv_channels", "256"),
        ("encoder.ff_expansion_factor", "4"),
        ("encoder.n_heads", "8"),
        ("encoder.pos_emb_max_len", "5000"),
        ("encoder.conv_kernel_size", "9"),
        ("decoder.vocab_size", "1024"),
        ("decoder.prednet.pred_hidden", "640"),
        ("decoder.prednet.pred_rnn_layers", "2"),
        ("joint.jointnet.encoder_hidden", "1024"),
        ("joint.jointnet.joint_hidden", "640"),
        ("joint.num_extra_outputs", "5"),
        ("decoding.greedy.max_symbols", "10"),
    ),
)
def test_validate_parakeet_rejects_wrong_scalar_types(field: str, value: str) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("preprocessor.preemph", 1.0, "float in"),
        ("preprocessor.lowfreq", 8000, "mel bounds"),
        ("preprocessor.highfreq", 8001, "mel bounds"),
        ("preprocessor.features", 0, "positive integer"),
        ("preprocessor.window_size", 0.00001, "at least 2"),
        ("preprocessor.window_stride", 0.00001, "at least one sample"),
        ("preprocessor.window_stride", 0.03, "must not exceed"),
    ),
)
def test_validate_parakeet_rejects_invalid_feature_values(
    field: str,
    value: float | int,
    message: str,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError, match=message):
        validate_parakeet(model_config, make_export_args())


def test_rename_parakeet_state_dict() -> None:
    pointwise_weight = torch.randn(8, 4, 1)
    second_pointwise_weight = torch.randn(4, 8, 1)
    feed_forward_weight = torch.randn(4, 8)
    second_feed_forward_weight = torch.randn(8, 4)
    query_weight = torch.randn(4, 4)
    key_weight = torch.randn(4, 4)
    value_weight = torch.randn(4, 4)
    subsampling_conv1_weight = torch.randn(8, 1, 3, 3)
    subsampling_conv2_weight = torch.randn(8, 8, 3, 3)
    subsampling_pointwise1_weight = torch.randn(8, 8, 1, 1)
    subsampling_conv3_weight = torch.randn(8, 8, 3, 3)
    subsampling_pointwise2_weight = torch.randn(8, 8, 1, 1)
    embedding_weight = torch.randn(16, 4)
    lstm_weight = torch.randn(16, 4)
    output_projection_weight = torch.randn(16, 4)
    decoder_projection_weight = torch.randn(4, 4)
    encoder_projection_weight = torch.randn(4, 4)
    untouched_weight = torch.randn(4)
    state_dict = OrderedDict(
        (
            (
                "encoder.pre_encode.conv.0.weight",
                subsampling_conv1_weight,
            ),
            (
                "encoder.pre_encode.conv.2.weight",
                subsampling_conv2_weight,
            ),
            (
                "encoder.pre_encode.conv.3.weight",
                subsampling_pointwise1_weight,
            ),
            (
                "encoder.pre_encode.conv.5.weight",
                subsampling_conv3_weight,
            ),
            (
                "encoder.pre_encode.conv.6.weight",
                subsampling_pointwise2_weight,
            ),
            (
                "encoder.layers.0.conv.pointwise_conv1.weight",
                pointwise_weight,
            ),
            (
                "encoder.layers.0.conv.pointwise_conv2.weight",
                second_pointwise_weight,
            ),
            (
                "encoder.layers.0.feed_forward1.linear2.weight",
                feed_forward_weight,
            ),
            (
                "encoder.layers.0.feed_forward2.linear2.weight",
                second_feed_forward_weight,
            ),
            ("encoder.layers.0.self_attn.linear_q.weight", query_weight),
            ("encoder.layers.0.self_attn.linear_k.weight", key_weight),
            ("encoder.layers.0.self_attn.linear_v.weight", value_weight),
            (
                "decoder.prediction.embed.weight",
                embedding_weight,
            ),
            (
                "decoder.prediction.dec_rnn.lstm.weight_ih_l0",
                lstm_weight,
            ),
            (
                "joint.joint_net.2.weight",
                output_projection_weight,
            ),
            (
                "joint.pred.weight",
                decoder_projection_weight,
            ),
            (
                "joint.enc.weight",
                encoder_projection_weight,
            ),
            ("encoder.layers.0.norm_out.weight", untouched_weight),
        ),
    )

    modified_state_dict = adjust_state_dict(state_dict)

    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.conv.pointwise_conv1.weight"],
        pointwise_weight.squeeze(2),
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.conv.pointwise_conv2.weight"],
        second_pointwise_weight.squeeze(2),
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.feed_forward1.linear2.weight"],
        feed_forward_weight * 0.5,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.feed_forward2.linear2.weight"],
        second_feed_forward_weight * 0.5,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.self_attn.linear_qkv.weight"],
        torch.cat((query_weight, key_weight, value_weight)),
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.pre_encode.conv1.weight"],
        subsampling_conv1_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.pre_encode.conv2.weight"],
        subsampling_conv2_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.pre_encode.pointwise_conv1.weight"],
        subsampling_pointwise1_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.pre_encode.conv3.weight"],
        subsampling_conv3_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.pre_encode.pointwise_conv2.weight"],
        subsampling_pointwise2_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["embedding.weight"],
        embedding_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["lstm.weight_ih_l0"],
        lstm_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["output_proj.weight"],
        output_projection_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["decoder_proj.weight"],
        decoder_projection_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder_proj.weight"],
        encoder_projection_weight,
    )
    torch.testing.assert_close(
        modified_state_dict["encoder.layers.0.norm_out.weight"],
        untouched_weight,
    )
    assert not any(
        key in modified_state_dict
        for key in (
            "encoder.layers.0.self_attn.linear_q.weight",
            "encoder.layers.0.self_attn.linear_k.weight",
            "encoder.layers.0.self_attn.linear_v.weight",
        )
    )


@pytest.mark.parametrize("shape", ((4, 4), (4, 4, 2)))
def test_adjust_state_dict_rejects_invalid_pointwise_weight_shape(
    shape: tuple[int, ...],
) -> None:
    state_dict = OrderedDict(
        (("encoder.layers.0.conv.pointwise_conv1.weight", torch.ones(shape)),)
    )

    with pytest.raises(ValueError, match="Expected pointwise Conv1d weight"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_alias_collision() -> None:
    state_dict = OrderedDict(
        (
            ("embedding.weight", torch.ones(2, 2)),
            ("decoder.prediction.embed.weight", torch.zeros(2, 2)),
        )
    )

    with pytest.raises(ValueError, match="both map to embedding.weight"):
        adjust_state_dict(state_dict)


@pytest.mark.parametrize("missing_projection", ("q", "k", "v"))
def test_adjust_state_dict_rejects_missing_attention_projection(
    missing_projection: str,
) -> None:
    state_dict = OrderedDict(
        (
            f"encoder.layers.0.self_attn.linear_{projection}.weight",
            torch.ones(2, 2),
        )
        for projection in ("q", "k", "v")
        if projection != missing_projection
    )

    with pytest.raises(ValueError, match="Missing attention projection companions"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_split_and_fused_attention_projections() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.layers.0.self_attn.linear_q.weight", torch.ones(2, 2)),
            ("encoder.layers.0.self_attn.linear_k.weight", torch.ones(2, 2)),
            ("encoder.layers.0.self_attn.linear_v.weight", torch.ones(2, 2)),
            ("encoder.layers.0.self_attn.linear_qkv.weight", torch.ones(6, 2)),
        )
    )

    with pytest.raises(ValueError, match="both split attention projections"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_incompatible_attention_projections() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.layers.0.self_attn.linear_q.weight", torch.ones(2, 2)),
            ("encoder.layers.0.self_attn.linear_k.weight", torch.ones(2)),
            ("encoder.layers.0.self_attn.linear_v.weight", torch.ones(2, 2)),
        )
    )

    with pytest.raises(ValueError, match="matching rank-2"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_unequal_attention_projection_rows() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.layers.0.self_attn.linear_q.weight", torch.ones(1, 4)),
            ("encoder.layers.0.self_attn.linear_k.weight", torch.ones(2, 4)),
            ("encoder.layers.0.self_attn.linear_v.weight", torch.ones(9, 4)),
        )
    )

    with pytest.raises(ValueError, match="matching rank-2"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_mixed_attention_projection_dtypes() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.layers.0.self_attn.linear_q.weight", torch.ones(2, 4)),
            (
                "encoder.layers.0.self_attn.linear_k.weight",
                torch.ones(2, 4, dtype=torch.float64),
            ),
            ("encoder.layers.0.self_attn.linear_v.weight", torch.ones(2, 4)),
        )
    )

    with pytest.raises(ValueError, match="matching rank-2"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_rejects_mixed_attention_projection_devices() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.layers.0.self_attn.linear_q.weight", torch.ones(2, 4)),
            (
                "encoder.layers.0.self_attn.linear_k.weight",
                torch.ones(2, 4, device="meta"),
            ),
            ("encoder.layers.0.self_attn.linear_v.weight", torch.ones(2, 4)),
        )
    )

    with pytest.raises(ValueError, match="matching rank-2"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_removes_attention_projection_suffix_literally() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.stage.self_attn.linear_q.weight", torch.ones(2, 2)),
            ("encoder.stage.self_attn.linear_k.weight", torch.ones(2, 2)),
            ("encoder.stage.self_attn.linear_v.weight", torch.ones(2, 2)),
        )
    )

    adjusted_state_dict = adjust_state_dict(state_dict)

    assert tuple(adjusted_state_dict) == ("encoder.stage.self_attn.linear_qkv.weight",)
