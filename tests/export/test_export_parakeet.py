#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for Parakeet bundle export, metadata conversion, and validation."""

import argparse
import io
import re
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, call

import onnx
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


def make_model_config() -> DictConfig:
    """Return a compact valid source configuration for Parakeet export tests.

    Returns
    -------
    DictConfig
        Fresh NeMo-style configuration that each test can mutate independently.
    """

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
    """Return valid Parakeet exporter arguments suitable for local mutation.

    Returns
    -------
    argparse.Namespace
        Fresh batch-one FP32 modified-beam-search arguments with placeholder
        paths; callers exercising file access must replace those paths.
    """

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


@pytest.mark.parametrize("use_symlink", (False, True), ids=("direct", "symlink"))
def test_export_parakeet_rejects_destination_containing_source(
    tmp_path: Path,
    use_symlink: bool,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    contained_model_path = output_dir / "model.nemo"
    contained_model_path.write_bytes(b"source")
    model_path = contained_model_path
    if use_symlink:
        model_path = tmp_path / "model-link.nemo"
        model_path.symlink_to(contained_model_path)
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = output_dir

    with pytest.raises(ValueError, match="contains required source file"):
        export_parakeet(args)

    assert contained_model_path.read_bytes() == b"source"
    if use_symlink:
        assert model_path.is_symlink()


def test_export_parakeet_rejects_missing_source_before_replacing_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    sentinel_path = output_dir / "existing.trt"
    sentinel_path.write_bytes(b"existing")
    args = make_export_args()
    args.model_path = tmp_path / "missing.nemo"
    args.output_dir = output_dir

    with pytest.raises(FileNotFoundError, match=re.escape(str(args.model_path))):
        export_parakeet(args)

    assert sentinel_path.read_bytes() == b"existing"


def test_export_parakeet_replaces_output_before_extraction(
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

    temporary_dirs: list[Path] = []

    def fail_export(extracted_model_path: Path, temporary_dir: Path) -> None:
        """Fail after checking that export cleared the previous bundle.

        Parameters
        ----------
        extracted_model_path : Path
            Source archive path supplied by the exporter.
        temporary_dir : Path
            Extraction directory recorded for the later cleanup assertion.

        Raises
        ------
        RuntimeError
            Always, to simulate archive extraction failure.
        """

        temporary_dirs.append(temporary_dir)
        assert extracted_model_path == model_path
        assert temporary_dir.is_dir()
        assert not any(output_dir.iterdir())
        raise RuntimeError("archive extraction failed")

    monkeypatch.setattr(parakeet_exporter, "extract_parakeet_archive", fail_export)

    with pytest.raises(RuntimeError, match="archive extraction failed"):
        export_parakeet(args)

    assert len(temporary_dirs) == 1
    assert not temporary_dirs[0].exists()
    assert not any(output_dir.iterdir())
    assert model_path.read_bytes() == b"source"


@pytest.mark.parametrize(
    ("decoder_type", "initial_beam", "expected_beam", "warning_expected"),
    (
        pytest.param("transducer_greedy_search", 1, 1, False, id="greedy-already-one"),
        pytest.param("transducer_greedy_search", 6, 1, True, id="greedy-override"),
        pytest.param(
            "transducer_modified_beam_search",
            6,
            6,
            False,
            id="modified-beam-preserved",
        ),
    ),
)
def test_export_parakeet_applies_decoder_beam_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    decoder_type: str,
    initial_beam: int,
    expected_beam: int,
    warning_expected: bool,
) -> None:
    model_path = tmp_path / "model.nemo"
    model_path.write_bytes(b"source")
    args = make_export_args()
    args.model_path = model_path
    args.output_dir = tmp_path / "bundle"
    args.decoder_type = decoder_type
    args.beam = initial_beam
    observed_beams: list[int] = []

    def stop_export(_model_path: Path, _temporary_dir: Path) -> None:
        """Capture the effective beam before any source processing.

        Parameters
        ----------
        _model_path : Path
            Unused archive path from the extraction callback.
        _temporary_dir : Path
            Unused extraction directory from the callback.

        Raises
        ------
        RuntimeError
            Always, to stop export immediately after recording the beam.
        """

        observed_beams.append(args.beam)
        raise RuntimeError("stop export")

    monkeypatch.setattr(
        parakeet_exporter,
        "extract_parakeet_archive",
        stop_export,
    )

    with (
        caplog.at_level("WARNING", logger=parakeet_exporter.__name__),
        pytest.raises(RuntimeError, match="stop export"),
    ):
        export_parakeet(args)

    assert observed_beams == [expected_beam]
    assert ("Overriding beam" in caplog.text) is warning_expected


@pytest.mark.parametrize(
    ("command", "overrides"),
    (
        pytest.param(
            "export --model-path model.nemo --output-dir output --batch-size 1 "
            "--decoder-type transducer_greedy_search --beam 1 "
            "--min-audio-seconds 0.1 --opt-audio-seconds 15 --max-audio-seconds 40",
            {
                "decoder_type": "transducer_greedy_search",
                "beam": 1,
                "min_audio_seconds": 0.1,
            },
            id="defaults",
        ),
        pytest.param(
            "export --model-path model.nemo --output-dir output --batch-size 256 "
            "--decoder-type transducer_modified_beam_search --beam 6 "
            "--encoder-precision bf16 --decoder-precision fp16 "
            "--min-audio-seconds 0.1 --opt-audio-seconds 8 --max-audio-seconds 40 "
            "--optimization-level 3 --debug",
            {
                "batch_size": 256,
                "encoder_precision": "bf16",
                "decoder_precision": "fp16",
                "min_audio_seconds": 0.1,
                "opt_audio_seconds": 8.0,
                "optimization_level": 3,
                "debug": True,
            },
            id="overrides",
        ),
    ),
)
def test_parse_args(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    overrides: dict[str, bool | float | int | str],
) -> None:
    monkeypatch.setattr(sys, "argv", command.split())
    expected = make_export_args()
    vars(expected).update(overrides)

    assert parse_args() == expected


def test_main_configures_logging_and_runs_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_export_args()
    parse = Mock(return_value=args)
    export = Mock()
    configure_logging = Mock()
    monkeypatch.setattr(parakeet_exporter, "parse_args", parse)
    monkeypatch.setattr(parakeet_exporter, "export_parakeet", export)
    monkeypatch.setattr(parakeet_exporter.logging, "basicConfig", configure_logging)

    parakeet_exporter.main()

    parse.assert_called_once_with()
    export.assert_called_once_with(args)
    configure_logging.assert_called_once_with(
        format="%(asctime)s %(levelname)s %(message)s",
        level=parakeet_exporter.logging.INFO,
    )


def test_export_parakeet_onnx_uses_fixed_decoder_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def record_export(module, inputs, path, **kwargs) -> None:
        """Check the inference context while Mock records the ONNX arguments.

        Parameters
        ----------
        module : torch.nn.Module
            Model passed to the mocked ONNX exporter.
        inputs : tuple[torch.Tensor, ...]
            Example inputs, inspected through the Mock's call record.
        path : Path
            Requested graph destination; no file is written.
        **kwargs : dict
            ONNX options captured by the surrounding Mock.
        """

        assert torch.is_inference_mode_enabled()

    export = Mock(side_effect=record_export)
    monkeypatch.setattr(parakeet_exporter.torch.onnx, "export", export)
    args = make_export_args()
    args.output_dir = tmp_path
    args.batch_size = 3
    args.beam = 4
    args.opt_audio_seconds = 2.75
    model_config = make_model_config()
    model_config.encoder.d_model = 384
    model_config.decoder.prednet.pred_hidden = 192
    model_config.decoder.prednet.pred_rnn_layers = 3
    model_config.joint.jointnet.encoder_hidden = 384
    encoder = torch.nn.Identity()
    decoder = torch.nn.Module()
    decoder.output_proj = torch.nn.Linear(1, 1, dtype=torch.float16, device="meta")

    paths = export_model_to_onnx(encoder, decoder, model_config, args)

    assert paths == (tmp_path / "parakeet.onnx", tmp_path / "tdt_decoder.onnx")
    assert export.call_count == 2
    encoder_call, decoder_call = export.call_args_list
    for export_call, module, path in zip(
        export.call_args_list,
        (encoder, decoder),
        paths,
        strict=True,
    ):
        assert export_call.args[0] is module
        assert export_call.args[2] == path

    torch.testing.assert_close(
        encoder_call.args[1],
        (torch.zeros(3, 44_000), torch.full((3,), 44_000, dtype=torch.int64)),
        rtol=0,
        atol=0,
    )
    assert encoder_call.kwargs == {
        "input_names": ("audio", "audio_lengths"),
        "output_names": ("encoder_output", "encoder_output_lengths"),
        "dynamic_shapes": {
            "audio": {1: torch.export.Dim.DYNAMIC},
            "audio_lengths": {},
        },
        "opset_version": parakeet_exporter.ONNX_OPSET_VERSION,
    }
    torch.testing.assert_close(
        decoder_call.args[1],
        (
            torch.zeros(12, 384, dtype=torch.float16),
            torch.zeros(12, 1, dtype=torch.int32),
            torch.zeros(3, 12, 192, dtype=torch.float16),
            torch.zeros(3, 12, 192, dtype=torch.float16),
        ),
        rtol=0,
        atol=0,
    )
    assert decoder_call.kwargs == {
        "input_names": (
            "encoder_output",
            "targets",
            "input_states_1",
            "input_states_2",
        ),
        "output_names": (
            "token_log_probs",
            "duration_log_probs",
            "output_states_1",
            "output_states_2",
        ),
        "opset_version": parakeet_exporter.ONNX_OPSET_VERSION,
    }


@pytest.mark.parametrize("debug", (False, True))
@pytest.mark.parametrize(("batch_size", "partitions"), ((3, 1), (128, 2)))
def test_export_parakeet_validates_exact_published_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
    batch_size: int,
    partitions: int,
) -> None:
    args = make_export_args()
    args.batch_size = batch_size
    args.beam = 4
    args.debug = debug
    args.encoder_precision = "fp16"
    args.decoder_precision = "bf16"
    args.optimization_level = 3
    args.model_path = tmp_path / "model.nemo"
    args.output_dir = tmp_path / "nested" / "bundle"
    source_config = make_model_config()
    source_config.tokenizer = {"model_path": "nemo:artifacts/source.model"}
    weight = torch.arange(4, dtype=torch.float32)
    checkpoint = io.BytesIO()
    torch.save(OrderedDict({"decoder.prediction.embed.weight": weight}), checkpoint)
    with tarfile.open(args.model_path, "w") as archive:
        add_archive_member(
            archive,
            "model_config.yaml",
            OmegaConf.to_yaml(source_config).encode(),
        )
        add_archive_member(archive, "model_weights.ckpt", checkpoint.getvalue())
        add_archive_member(archive, "artifacts/source.model", b"tokenizer")
    source_bytes = args.model_path.read_bytes()
    encoder, decoder = torch.nn.Identity(), torch.nn.Identity()
    construct_model = Mock(return_value=(encoder, decoder))
    validate_source = Mock(wraps=validate_parakeet)
    load_checkpoint = Mock(wraps=torch.load)
    validate_config = Mock(wraps=parakeet_exporter.validate_model_config)

    def export_onnx(encoder, decoder, model_config, export_args) -> tuple[Path, Path]:
        """Write small ONNX graphs so production cleanup can run unchanged.

        Parameters
        ----------
        encoder : torch.nn.Module
            Placeholder encoder recorded by the surrounding Mock.
        decoder : torch.nn.Module
            Placeholder decoder recorded by the surrounding Mock.
        model_config : DictConfig
            Source configuration passed by the exporter.
        export_args : argparse.Namespace
            Arguments providing the output directory.

        Returns
        -------
        tuple[Path, Path]
            Newly written encoder and decoder graph paths.
        """

        paths = tuple(
            export_args.output_dir / name
            for name in ("parakeet.onnx", "tdt_decoder.onnx")
        )
        for path in paths:
            onnx.save(
                onnx.helper.make_model(
                    onnx.helper.make_graph([], path.stem, [], []),
                ),
                path,
            )
        return paths

    def build_engine(onnx_path, engine_path, profiles, optimization_level) -> None:
        """Stand in for TensorRT only after extraction resources are released.

        Parameters
        ----------
        onnx_path : Path
            Existing graph whose basename becomes the engine's sentinel bytes.
        engine_path : Path
            Destination engine path.
        profiles : dict
            Optimization profiles recorded by the surrounding Mock.
        optimization_level : int
            Builder level recorded by the surrounding Mock.
        """

        assert onnx_path.is_file()
        assert not load_checkpoint.call_args.args[0].parent.exists()
        engine_path.write_bytes(onnx_path.name.encode())

    def validate_bundle(model_dir: Path, model_config: DictConfig) -> None:
        """Check the reloaded config and the complete published bundle.

        Parameters
        ----------
        model_dir : Path
            Published directory to check for the exact expected artifact set.
        model_config : DictConfig
            Reloaded runtime metadata, compared with its saved source.
        """

        assert model_dir == args.output_dir
        assert model_config is not validate_config.call_args.args[0]
        assert model_config == validate_config.call_args.args[0]
        assert model_config == OmegaConf.load(model_dir / "model_config.yaml")
        expected_files = {
            "model_config.yaml",
            "bpe.model",
            "parakeet.trt",
            "tdt_decoder.trt",
        }
        if debug:
            expected_files.update(("parakeet.onnx", "tdt_decoder.onnx"))
        assert {path.name for path in model_dir.iterdir()} == expected_files
        assert all(path.is_file() for path in model_dir.iterdir())
        assert (model_dir / "bpe.model").read_bytes() == b"tokenizer"
        assert (model_dir / "parakeet.trt").read_bytes() == b"parakeet.onnx"
        assert (model_dir / "tdt_decoder.trt").read_bytes() == b"tdt_decoder.onnx"

    export_onnx_mock = Mock(side_effect=export_onnx)
    build = Mock(side_effect=build_engine)
    validate = Mock(side_effect=validate_bundle)
    monkeypatch.setattr(parakeet_exporter, "make_model", construct_model)
    monkeypatch.setattr(parakeet_exporter, "validate_parakeet", validate_source)
    monkeypatch.setattr(parakeet_exporter.torch, "load", load_checkpoint)
    monkeypatch.setattr(parakeet_exporter, "validate_model_config", validate_config)
    monkeypatch.setattr(parakeet_exporter, "export_model_to_onnx", export_onnx_mock)
    monkeypatch.setattr(parakeet_exporter, "build_tensorrt_engine", build)
    monkeypatch.setattr(parakeet_exporter, "validate_model", validate)

    export_parakeet(args)

    validate_source.assert_called_once_with(source_config, args)
    load_checkpoint.assert_called_once_with(
        load_checkpoint.call_args.args[0],
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert load_checkpoint.call_args.args[0].name == "model_weights.ckpt"
    construct_model.assert_called_once()
    config, state_dict, *settings = construct_model.call_args.args
    assert config == source_config
    torch.testing.assert_close(
        state_dict,
        OrderedDict({"embedding.weight": weight}),
        rtol=0,
        atol=0,
    )
    assert settings == [partitions, torch.float16, torch.bfloat16]
    export_onnx_mock.assert_called_once_with(encoder, decoder, source_config, args)
    validate_config.assert_called_once()
    validate.assert_called_once_with(args.output_dir, validate_config.call_args.args[0])
    assert build.call_args_list == [
        call(
            args.output_dir / "parakeet.onnx",
            args.output_dir / "parakeet.trt",
            {
                "audio": (
                    (batch_size, 8_000),
                    (batch_size, 240_000),
                    (batch_size, 640_000),
                )
            },
            3,
        ),
        call(
            args.output_dir / "tdt_decoder.onnx",
            args.output_dir / "tdt_decoder.trt",
            {},
            3,
        ),
    ]
    assert args.model_path.read_bytes() == source_bytes


@pytest.mark.parametrize(
    ("feature_values", "expected_feature_values"),
    (
        pytest.param({}, (0.97, 0, 8_000), id="defaults"),
        pytest.param(
            {"preemph": 0.5, "lowfreq": 80, "highfreq": 7_600},
            (0.5, 80, 7_600),
            id="configured",
        ),
    ),
)
def test_make_model_wires_configuration_and_state_dicts(
    monkeypatch: pytest.MonkeyPatch,
    feature_values: dict[str, float | int],
    expected_feature_values: tuple[float, int, int],
) -> None:
    make_encoder = Mock(return_value=Mock(spec=torch.nn.Module))
    make_decoder = Mock(return_value=Mock(spec=torch.nn.Module))
    monkeypatch.setattr(parakeet_exporter, "ParakeetTDTEncoder", make_encoder)
    monkeypatch.setattr(parakeet_exporter, "Decoder", make_decoder)
    encoder_weight = torch.ones(1)
    decoder_weights = OrderedDict(
        (
            ("embedding.weight", torch.ones(2)),
            ("lstm.weight_ih_l0", torch.ones(3)),
            ("lstm.weight_hh_l2", torch.ones(4)),
            ("lstm.bias_ih_l1", torch.ones(5)),
            ("decoder_proj.weight", torch.ones(6)),
            ("encoder_proj.bias", torch.ones(7)),
            ("output_proj.weight", torch.ones(8)),
        )
    )
    state_dict = OrderedDict(
        (
            ("encoder.pre_encode.conv1.weight", encoder_weight),
            *decoder_weights.items(),
            ("feature_extractor.window", torch.ones(9)),
        )
    )
    model_config = make_model_config()
    model_config.preprocessor.update(feature_values)
    model_config.preprocessor.window_stride = 0.012
    model_config.preprocessor.window_size = 0.031
    model_config.preprocessor.features = 96
    model_config.encoder.n_layers = 7
    model_config.encoder.d_model = 384
    model_config.encoder.subsampling_conv_channels = 80
    model_config.encoder.ff_expansion_factor = 3
    model_config.encoder.n_heads = 6
    model_config.encoder.pos_emb_max_len = 777
    model_config.encoder.conv_kernel_size = 15
    model_config.decoder.vocab_size = 321
    model_config.decoder.prednet.pred_hidden = 192
    model_config.decoder.prednet.pred_rnn_layers = 3
    model_config.joint.jointnet.encoder_hidden = 384
    model_config.joint.jointnet.joint_hidden = 224
    model_config.joint.num_extra_outputs = 7

    encoder, decoder = make_model(
        model_config,
        state_dict,
        3,
        torch.float16,
        torch.bfloat16,
    )

    assert encoder is make_encoder.return_value
    assert decoder is make_decoder.return_value
    make_encoder.assert_called_once_with(
        samp_freq=16_000,
        frame_shift_ms=12,
        frame_length_ms=31,
        feature_dim=96,
        preemph=expected_feature_values[0],
        low_freq=expected_feature_values[1],
        high_freq=expected_feature_values[2],
        n_layers=7,
        model_dim=384,
        subsampling_conv_channels=80,
        feed_forward_expansion_factor=3,
        n_heads=6,
        pos_emb_max_len=777,
        conv_kernel_size=15,
        subsampling_batch_partitions=3,
        dtype=torch.float16,
    )
    make_decoder.assert_called_once_with(
        vocab_size=321,
        encoder_dim=384,
        decoder_dim=192,
        joiner_dim=224,
        pred_rnn_layers=3,
        num_extra_outputs=7,
        dtype=torch.bfloat16,
    )
    for module, expected_weights in (
        (encoder, OrderedDict({"encoder.pre_encode.conv1.weight": encoder_weight})),
        (decoder, decoder_weights),
    ):
        module.load_state_dict.assert_called_once_with(expected_weights, strict=True)
        module.eval.assert_called_once_with()


def test_make_parakeet_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = [0, 2, 4]
    model_config.preprocessor.window_stride = 0.012
    model_config.preprocessor.features = 96
    model_config.encoder.n_layers = 7
    model_config.encoder.d_model = 384
    model_config.encoder.pos_emb_max_len = 777
    model_config.decoder.vocab_size = 321
    model_config.decoder.prednet.pred_hidden = 192
    model_config.decoder.prednet.pred_rnn_layers = 3
    model_config.joint.jointnet.encoder_hidden = 384
    model_config.joint.jointnet.joint_hidden = 224
    model_config.joint.num_extra_outputs = 3
    model_config.decoding.greedy.max_symbols = 7
    args = make_export_args()
    args.beam = 5
    args.min_audio_seconds = 0.25
    args.opt_audio_seconds = 7.5
    args.max_audio_seconds = 30.0
    validate = Mock(wraps=parakeet_exporter.validate_model_config)
    monkeypatch.setattr(parakeet_exporter, "validate_model_config", validate)
    runtime_config = make_runtime_config(model_config, args)

    validate.assert_called_once_with(runtime_config)
    assert OmegaConf.to_container(runtime_config, resolve=True) == {
        "model_type": "parakeet_asr",
        "decoder_type": "transducer_modified_beam_search",
        "model_samplerate": 16_000,
        "vocab_size": 321,
        "blank_id": 321,
        "audio_encoder_params": {
            "feature_dim": 96,
            "frame_shift_ms": 12,
            "n_layers": 7,
            "model_dim": 384,
            "pos_emb_max_len": 777,
            "subsampling_factor": 8,
            "min_audio_seconds": 0.25,
            "opt_audio_seconds": 7.5,
            "max_audio_seconds": 30.0,
        },
        "decoder_params": {
            "encoder_dim": 384,
            "decoder_dim": 192,
            "joiner_dim": 224,
            "pred_rnn_layers": 3,
            "num_extra_outputs": 3,
            "beam": 5,
            "blank_penalty": 0.0,
            "max_symbols_per_timestep": 7,
            "tdt_durations": [0, 2, 4],
        },
    }


@pytest.mark.parametrize(
    "decoder_type",
    ("transducer_modified_beam_search", "transducer_greedy_search"),
)
def test_validate_parakeet_accepts_beam_one(decoder_type: str) -> None:
    model_config = make_model_config()
    args = make_export_args()
    args.decoder_type = decoder_type
    args.beam = 1

    validate_parakeet(model_config, args)
    runtime_config = make_runtime_config(model_config, args)

    assert runtime_config.decoder_type == decoder_type
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
        ([0, 1, 2.0, 3, 4], "non-negative signed 32-bit integers"),
        ([0, 1, 1, 3, 4], "unique values"),
        ([1, 2, 3, 4, 5], "contain zero"),
        ([0, 1, 2, 3], "must match joint.num_extra_outputs"),
        ([0, 1, 2, 3, 1 << 31], "signed 32-bit integers"),
    ),
)
def test_validate_parakeet_rejects_invalid_tdt_durations(
    durations: list[float | int],
    message: str,
) -> None:
    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = durations

    with pytest.raises(ValueError, match=message):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_only_zero_duration() -> None:
    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = [0]
    model_config.joint.num_extra_outputs = 1

    with pytest.raises(ValueError, match="zero and at least one positive duration"):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize("durations", (None, 1))
def test_validate_parakeet_rejects_nonsequence_tdt_durations(
    durations: int | None,
) -> None:
    model_config = make_model_config()
    model_config.model_defaults.tdt_durations = durations

    with pytest.raises(TypeError, match="not iterable"):
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
        ("batch_size", 1.5, "batch_size must be positive"),
        ("beam", 0, "beam must be positive"),
        ("beam", 1.5, "beam must be positive"),
        ("min_audio_seconds", 0.0, "Expected 0 < min_audio_seconds"),
        ("min_audio_seconds", "0.5", "Expected 0 < min_audio_seconds"),
        ("opt_audio_seconds", 0.25, "min_audio_seconds <= opt_audio_seconds"),
        ("max_audio_seconds", 10.0, "opt_audio_seconds <= max_audio_seconds"),
    ),
)
def test_validate_parakeet_rejects_invalid_export_arguments(
    field: str,
    value: float | int | str,
    message: str,
) -> None:
    args = make_export_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("preprocessor.frame_splicing", True),
        ("preprocessor.n_fft", 512.0),
        ("preprocessor.log", 1),
        ("preprocessor.mag_power", 2),
        ("preprocessor.pad_to", False),
        ("preprocessor.pad_value", 0),
        ("encoder.subsampling_factor", 8.0),
        ("encoder.xscaling", 0),
        ("encoder.untie_biases", 1),
        ("encoder.use_bias", 0),
        ("decoder.blank_as_pad", 1),
    ),
)
def test_validate_parakeet_accepts_equivalent_fixed_values(
    field: str,
    value: bool | float | int,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_accepts_omitted_optional_preprocessor_defaults() -> None:
    model_config = make_model_config()
    for field in ("n_fft", "log", "pad_to", "pad_value"):
        del model_config.preprocessor[field]

    validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize("field", ("encoder_precision", "decoder_precision"))
@pytest.mark.parametrize("value", ("fp8", None, True, ["fp32"]))
def test_validate_parakeet_rejects_unsupported_precision(
    field: str,
    value: bool | list[str] | str | None,
) -> None:
    args = make_export_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=rf"{field} must be one of"):
        validate_parakeet(make_model_config(), args)


def test_validate_parakeet_normalization_length_boundary() -> None:
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
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError, match=re.escape(field)):
        validate_parakeet(model_config, make_export_args())


def test_validate_parakeet_rejects_unsupported_decoder_type() -> None:
    args = make_export_args()
    args.decoder_type = "ctc_greedy_search"

    with pytest.raises(ValueError, match="supports only"):
        validate_parakeet(make_model_config(), args)


@pytest.mark.parametrize("attention_context", (None, -1))
def test_validate_parakeet_rejects_nonsequence_attention_context(
    attention_context: int | None,
) -> None:
    model_config = make_model_config()
    model_config.encoder.att_context_size = attention_context

    with pytest.raises(ValueError, match="full-context offline attention"):
        validate_parakeet(model_config, make_export_args())


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
    args = make_export_args()
    args.batch_size = batch_size
    args.max_audio_seconds = 40.0

    assert (
        get_subsampling_batch_partitions(make_model_config(), args)
        == expected_partitions
    )


def test_subsampling_batch_partitions_use_configured_frontend_dimensions() -> None:
    model_config = make_model_config()
    model_config.preprocessor.window_stride = 0.02
    model_config.preprocessor.features = 80
    model_config.encoder.subsampling_conv_channels = 128
    args = make_export_args()
    args.batch_size = 256

    assert get_subsampling_batch_partitions(model_config, args) == 1


def test_subsampling_batch_cask_single_item_boundary() -> None:
    model_config = make_model_config()
    args = make_export_args()
    args.batch_size = 1
    sample_rate = model_config.preprocessor.sample_rate
    hop_length = round(model_config.preprocessor.window_stride * sample_rate)
    elements_per_frame = model_config.encoder.subsampling_conv_channels * (
        (model_config.preprocessor.features + 1) // 2
    )
    cask_element_limit = 1 << 31
    assert cask_element_limit % elements_per_frame == 0
    maximum_conv1_frames = cask_element_limit // elements_per_frame

    accepted_feature_frames = 2 * maximum_conv1_frames - 1
    accepted_samples = (accepted_feature_frames - 1) * hop_length
    args.max_audio_seconds = accepted_samples / sample_rate
    assert get_subsampling_batch_partitions(model_config, args) == 1

    rejected_feature_frames = 2 * maximum_conv1_frames + 1
    rejected_samples = (rejected_feature_frames - 1) * hop_length
    args.max_audio_seconds = rejected_samples / sample_rate
    with pytest.raises(ValueError, match="One Parakeet subsampling item exceeds"):
        get_subsampling_batch_partitions(model_config, args)


def add_archive_member(
    archive: tarfile.TarFile,
    name: str,
    data: bytes = b"data",
) -> None:
    """Add one regular in-memory file to a test tar archive.

    Parameters
    ----------
    archive : tarfile.TarFile
        Archive already open for writing.
    name : str
        Member name stored verbatim, including intentionally unsafe test paths.
    data : bytes
        Contents of the new member.
    """

    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize(
    ("tokenizer_reference", "tokenizer_name"),
    (
        ("nemo:nested/nemo_tokenizer.model", "nemo_tokenizer.model"),
        ("nested/archive_tokenizer.model", "archive_tokenizer.model"),
    ),
    ids=("nemo-uri", "archive-path"),
)
def test_extract_parakeet_archive_resolves_tokenizer_reference(
    tmp_path: Path,
    tokenizer_reference: str,
    tokenizer_name: str,
) -> None:
    archive_path = tmp_path / "model.nemo"
    config = OmegaConf.create({"tokenizer": {"model_path": tokenizer_reference}})
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(
            archive,
            "model/model_config.yaml",
            OmegaConf.to_yaml(config).encode(),
        )
        add_archive_member(archive, "model/model_weights.ckpt", b"checkpoint")
        add_archive_member(archive, "artifacts/tokenizer.model", b"wrong")
        add_archive_member(archive, f"artifacts/{tokenizer_name}", b"tokenizer")
    output_dir = tmp_path / "extracted"
    output_dir.mkdir()

    config_path, checkpoint_path, tokenizer_path = extract_parakeet_archive(
        archive_path,
        output_dir,
    )

    assert config_path == output_dir / "model_config.yaml"
    assert checkpoint_path == output_dir / "model_weights.ckpt"
    assert tokenizer_path == output_dir / tokenizer_name
    assert OmegaConf.load(config_path) == config
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert tokenizer_path.read_bytes() == b"tokenizer"


@pytest.mark.parametrize(
    "missing_name",
    ("model_config.yaml", "model_weights.ckpt", "custom_tokenizer.model"),
)
def test_extract_parakeet_archive_rejects_missing_required_member(
    tmp_path: Path,
    missing_name: str,
) -> None:
    archive_path = tmp_path / "model.nemo"
    config = OmegaConf.create(
        {"tokenizer": {"model_path": "nemo:artifacts/custom_tokenizer.model"}}
    )
    members = {
        "model_config.yaml": OmegaConf.to_yaml(config).encode(),
        "model_weights.ckpt": b"checkpoint",
        "custom_tokenizer.model": b"tokenizer",
    }
    with tarfile.open(archive_path, "w") as archive:
        for name, data in members.items():
            if name != missing_name:
                add_archive_member(archive, f"model/{name}", data)

    output_dir = tmp_path / "extracted"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=missing_name):
        extract_parakeet_archive(archive_path, output_dir)


def test_extract_member_matches_exact_basename(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(archive, "attacker_model_config.yaml", b"wrong")
        add_archive_member(archive, "nested/model_config.yaml", b"right")

    with tarfile.open(archive_path) as archive:
        output_path = extract_member(archive, "model_config.yaml", tmp_path)

    assert output_path.read_bytes() == b"right"


def test_extract_member_flattens_untrusted_archive_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.nemo"
    output_dir = tmp_path / "extracted"
    output_dir.mkdir()
    with tarfile.open(archive_path, "w") as archive:
        add_archive_member(archive, "../model_config.yaml", b"config")

    with tarfile.open(archive_path) as archive:
        output_path = extract_member(archive, "model_config.yaml", output_dir)

    assert output_path == output_dir / "model_config.yaml"
    assert output_path.read_bytes() == b"config"
    assert not (tmp_path / "model_config.yaml").exists()


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

    with pytest.raises(ValueError, match=re.escape(field.rsplit(".", 1)[-1])):
        validate_parakeet(model_config, make_export_args())


@pytest.mark.parametrize(
    "field",
    ("preprocessor.window_stride", "preprocessor.window_size"),
)
@pytest.mark.parametrize(
    "value",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ),
)
def test_validate_parakeet_rejects_nonfinite_feature_timing(
    field: str,
    value: float,
) -> None:
    model_config = make_model_config()
    OmegaConf.update(model_config, field, value)

    with pytest.raises(ValueError, match="positive finite float"):
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


def test_adjust_state_dict_converts_complete_parakeet_layout() -> None:
    rng = torch.Generator().manual_seed(0)
    pointwise_weight = torch.randn(8, 4, 1, generator=rng)
    second_pointwise_weight = torch.randn(4, 8, 1, generator=rng)
    feed_forward_weight = torch.randn(4, 8, generator=rng)
    second_feed_forward_weight = torch.randn(8, 4, generator=rng)
    query_weight = torch.randn(4, 4, generator=rng)
    key_weight = torch.randn(4, 4, generator=rng)
    value_weight = torch.randn(4, 4, generator=rng)
    subsampling_conv1_weight = torch.randn(8, 1, 3, 3, generator=rng)
    subsampling_conv2_weight = torch.randn(8, 8, 3, 3, generator=rng)
    subsampling_pointwise1_weight = torch.randn(8, 8, 1, 1, generator=rng)
    subsampling_conv3_weight = torch.randn(8, 8, 3, 3, generator=rng)
    subsampling_pointwise2_weight = torch.randn(8, 8, 1, 1, generator=rng)
    embedding_weight = torch.randn(16, 4, generator=rng)
    lstm_weight = torch.randn(16, 4, generator=rng)
    output_projection_weight = torch.randn(16, 4, generator=rng)
    decoder_projection_weight = torch.randn(4, 4, generator=rng)
    encoder_projection_weight = torch.randn(4, 4, generator=rng)
    untouched_weight = torch.randn(4, generator=rng)
    state_dict = OrderedDict(
        {
            "encoder.pre_encode.conv.0.weight": subsampling_conv1_weight,
            "encoder.pre_encode.conv.2.weight": subsampling_conv2_weight,
            "encoder.pre_encode.conv.3.weight": subsampling_pointwise1_weight,
            "encoder.pre_encode.conv.5.weight": subsampling_conv3_weight,
            "encoder.pre_encode.conv.6.weight": subsampling_pointwise2_weight,
            "encoder.layers.0.conv.pointwise_conv1.weight": pointwise_weight,
            "encoder.layers.0.conv.pointwise_conv2.weight": second_pointwise_weight,
            "encoder.layers.0.feed_forward1.linear2.weight": feed_forward_weight,
            "encoder.layers.0.feed_forward2.linear2.weight": second_feed_forward_weight,
            "encoder.layers.0.self_attn.linear_q.weight": query_weight,
            "encoder.layers.0.self_attn.linear_k.weight": key_weight,
            "encoder.layers.0.self_attn.linear_v.weight": value_weight,
            "decoder.prediction.embed.weight": embedding_weight,
            "decoder.prediction.dec_rnn.lstm.weight_ih_l0": lstm_weight,
            "joint.joint_net.2.weight": output_projection_weight,
            "joint.pred.weight": decoder_projection_weight,
            "joint.enc.weight": encoder_projection_weight,
            "encoder.layers.0.norm_out.weight": untouched_weight,
        }
    )
    original_items = tuple(state_dict.items())
    original_values = {key: value.clone() for key, value in original_items}
    expected = {
        "encoder.pre_encode.conv1.weight": subsampling_conv1_weight,
        "encoder.pre_encode.conv2.weight": subsampling_conv2_weight,
        "encoder.pre_encode.pointwise_conv1.weight": subsampling_pointwise1_weight,
        "encoder.pre_encode.conv3.weight": subsampling_conv3_weight,
        "encoder.pre_encode.pointwise_conv2.weight": subsampling_pointwise2_weight,
        "encoder.layers.0.conv.pointwise_conv1.weight": pointwise_weight.squeeze(2),
        "encoder.layers.0.conv.pointwise_conv2.weight": second_pointwise_weight.squeeze(
            2
        ),
        "encoder.layers.0.feed_forward1.linear2.weight": feed_forward_weight * 0.5,
        "encoder.layers.0.feed_forward2.linear2.weight": second_feed_forward_weight
        * 0.5,
        "encoder.layers.0.self_attn.linear_qkv.weight": torch.cat(
            (query_weight, key_weight, value_weight),
        ),
        "embedding.weight": embedding_weight,
        "lstm.weight_ih_l0": lstm_weight,
        "output_proj.weight": output_projection_weight,
        "decoder_proj.weight": decoder_projection_weight,
        "encoder_proj.weight": encoder_projection_weight,
        "encoder.layers.0.norm_out.weight": untouched_weight,
    }

    adjusted = adjust_state_dict(state_dict)

    assert isinstance(adjusted, OrderedDict)
    assert tuple(state_dict) == tuple(original_values)
    assert all(state_dict[key] is value for key, value in original_items)
    torch.testing.assert_close(state_dict, original_values, rtol=0, atol=0)
    torch.testing.assert_close(adjusted, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("source_key", "expected_key"),
    (
        (
            "decoder.prediction.dec_rnn.lstm.weight_hh_l1",
            "lstm.weight_hh_l1",
        ),
        (
            "decoder.prediction.dec_rnn.lstm.bias_ih_l1",
            "lstm.bias_ih_l1",
        ),
        ("joint.joint_net.2.bias", "output_proj.bias"),
        ("joint.pred.bias", "decoder_proj.bias"),
        ("joint.enc.bias", "encoder_proj.bias"),
    ),
)
def test_adjust_state_dict_preserves_parameter_suffixes_when_renaming(
    source_key: str,
    expected_key: str,
) -> None:
    parameter = torch.arange(4, dtype=torch.float32)

    adjusted_state_dict = adjust_state_dict(OrderedDict(((source_key, parameter),)))

    assert tuple(adjusted_state_dict) == (expected_key,)
    assert adjusted_state_dict[expected_key] is parameter


def test_adjust_state_dict_fuses_every_attention_layer() -> None:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for projection, projection_index in (("v", 3), ("q", 1), ("k", 2)):
        for layer in range(2):
            state_dict[
                f"encoder.layers.{layer}.self_attn.linear_{projection}.weight"
            ] = torch.full((2, 2), layer * 10 + projection_index, dtype=torch.float32)

    adjusted_state_dict = adjust_state_dict(state_dict)

    assert set(adjusted_state_dict) == {
        "encoder.layers.0.self_attn.linear_qkv.weight",
        "encoder.layers.1.self_attn.linear_qkv.weight",
    }
    for layer in range(2):
        expected = torch.cat(
            tuple(
                torch.full((2, 2), layer * 10 + projection_index, dtype=torch.float32)
                for projection_index in range(1, 4)
            )
        )
        torch.testing.assert_close(
            adjusted_state_dict[f"encoder.layers.{layer}.self_attn.linear_qkv.weight"],
            expected,
        )


@pytest.mark.parametrize("pointwise_layer", ("pointwise_conv1", "pointwise_conv2"))
@pytest.mark.parametrize("shape", ((4, 4), (4, 4, 2)))
def test_adjust_state_dict_rejects_invalid_pointwise_weight_shape(
    pointwise_layer: str,
    shape: tuple[int, ...],
) -> None:
    state_dict = OrderedDict(
        {
            f"encoder.layers.0.conv.{pointwise_layer}.weight": torch.ones(shape),
        }
    )

    with pytest.raises(ValueError, match="Expected pointwise Conv1d weight"):
        adjust_state_dict(state_dict)


@pytest.mark.parametrize(
    "reverse_order",
    (False, True),
    ids=("target-first", "alias-first"),
)
def test_adjust_state_dict_rejects_alias_collision(reverse_order: bool) -> None:
    items = (
        ("embedding.weight", torch.ones(2, 2)),
        ("decoder.prediction.embed.weight", torch.zeros(2, 2)),
    )
    state_dict = OrderedDict(reversed(items) if reverse_order else items)

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


@pytest.mark.parametrize(
    ("shape", "dtype", "device"),
    (
        pytest.param((2,), torch.float32, "cpu", id="rank"),
        pytest.param((1, 4), torch.float32, "cpu", id="rows"),
        pytest.param((2, 4), torch.float64, "cpu", id="dtype"),
        pytest.param((2, 4), torch.float32, "meta", id="device"),
    ),
)
def test_adjust_state_dict_rejects_incompatible_attention_projections(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
) -> None:
    state_dict = OrderedDict(
        {
            "encoder.layers.0.self_attn.linear_q.weight": torch.ones(2, 4),
            "encoder.layers.0.self_attn.linear_k.weight": torch.ones(
                shape,
                dtype=dtype,
                device=device,
            ),
            "encoder.layers.0.self_attn.linear_v.weight": torch.ones(2, 4),
        }
    )

    with pytest.raises(ValueError, match="matching rank-2"):
        adjust_state_dict(state_dict)


def test_adjust_state_dict_preserves_attention_prefix_exactly() -> None:
    state_dict = OrderedDict(
        (
            ("encoder.stage.self_attn.linear_q.weight", torch.ones(2, 2)),
            ("encoder.stage.self_attn.linear_k.weight", torch.ones(2, 2)),
            ("encoder.stage.self_attn.linear_v.weight", torch.ones(2, 2)),
        )
    )

    adjusted_state_dict = adjust_state_dict(state_dict)

    assert tuple(adjusted_state_dict) == ("encoder.stage.self_attn.linear_qkv.weight",)
