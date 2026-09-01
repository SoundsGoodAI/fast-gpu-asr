#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for shared ONNX cleanup and TensorRT export utilities."""

from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from google.protobuf.message import DecodeError
from onnx import TensorProto, helper

import fast_gpu_asr.export.export_utils as export_utils
from fast_gpu_asr.export.export_utils import (
    build_tensorrt_engine,
    remove_onnx_artifacts,
)

PROFILE_SHAPES = ((1, 1), (1, 2), (1, 3))
SERIALIZED_ENGINE = b"serialized engine"


class FakeTensorRTInput:
    """One fake TensorRT network input."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class FakeTensorRTProfile:
    """Record optimization-profile shape requests."""

    def __init__(self) -> None:
        self.shapes: list[
            tuple[
                str,
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ]
        ] = []
        self.accept_shapes = True

    def set_shape(
        self,
        name: str,
        min_shape: tuple[int, ...],
        opt_shape: tuple[int, ...],
        max_shape: tuple[int, ...],
    ) -> bool:
        self.shapes.append((name, min_shape, opt_shape, max_shape))
        return self.accept_shapes


class FakeTensorRTConfig:
    """Record builder configuration without loading TensorRT."""

    def __init__(self) -> None:
        self.engine_capability: int | None = None
        self.tactic_sources: int | None = None
        self.accept_tactic_sources = True
        self.flags: list[int] = []
        self.preview_features: list[tuple[int, bool]] = []
        self.builder_optimization_level: int | None = None
        self.profiles: list[FakeTensorRTProfile] = []

    def set_tactic_sources(self, sources: int) -> bool:
        self.tactic_sources = sources
        return self.accept_tactic_sources

    def set_flag(self, flag: int) -> None:
        self.flags.append(flag)

    def set_preview_feature(self, feature: int, enabled: bool) -> None:
        self.preview_features.append((feature, enabled))

    def add_optimization_profile(self, profile: FakeTensorRTProfile) -> int:
        self.profiles.append(profile)
        return len(self.profiles) - 1


@pytest.fixture
def fake_tensorrt_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Replace TensorRT's parser and builder with observable CPU fakes."""

    state = SimpleNamespace(
        inputs=[],
        parse_result=True,
        parse_errors=["parser error"],
        plugin_result=True,
        serialized_engine=SERIALIZED_ENGINE,
        network_flags=None,
        loaded_plugins=0,
        events=[],
    )
    state.network = SimpleNamespace(
        num_inputs=0,
        get_input=lambda index: state.inputs[index],
    )
    state.profile = FakeTensorRTProfile()
    state.config = FakeTensorRTConfig()

    class FakeLogger:
        INFO = 1

        def __init__(self, severity: int) -> None:
            self.severity = severity

    class FakeParser:
        def __init__(self, network: object, logger: FakeLogger) -> None:
            state.parser_args = (network, logger)

        @property
        def num_errors(self) -> int:
            return len(state.parse_errors)

        def parse_from_file(self, path: str) -> bool:
            state.events.append("parse")
            state.parsed_path = path
            return state.parse_result

        def get_error(self, index: int) -> str:
            return state.parse_errors[index]

    class FakeBuilder:
        def __init__(self, logger: FakeLogger) -> None:
            state.logger = logger

        def create_network(self, flags: int) -> object:
            state.network_flags = flags
            state.network.num_inputs = len(state.inputs)
            return state.network

        def create_builder_config(self) -> FakeTensorRTConfig:
            return state.config

        def create_optimization_profile(self) -> FakeTensorRTProfile:
            return state.profile

        def build_serialized_network(
            self,
            network: object,
            config: FakeTensorRTConfig,
        ) -> bytes | None:
            state.events.append("build")
            state.built = (network, config)
            return state.serialized_engine

    def initialize_standard_plugins(logger: FakeLogger, namespace: str) -> bool:
        state.events.append("standard_plugins")
        state.standard_plugin_args = (logger, namespace)
        return state.plugin_result

    fake_trt = SimpleNamespace(
        Builder=FakeBuilder,
        BuilderFlag=SimpleNamespace(TF32=10, SPARSE_WEIGHTS=11, FP16=12, BF16=13),
        EngineCapability=SimpleNamespace(STANDARD=20),
        Logger=FakeLogger,
        NetworkDefinitionCreationFlag=SimpleNamespace(PREFER_AOT_PYTHON_PLUGINS=2),
        OnnxParser=FakeParser,
        PreviewFeature=SimpleNamespace(ALIASED_PLUGIN_IO_10_03=30),
        TacticSource=SimpleNamespace(__members__={"A": 0, "B": 2}),
        init_libnvinfer_plugins=initialize_standard_plugins,
    )
    monkeypatch.setattr(export_utils, "trt", fake_trt)

    def load_plugins() -> None:
        state.loaded_plugins += 1
        state.events.append("custom_plugins")

    monkeypatch.setattr(export_utils, "load_tensorrt_plugins", load_plugins)
    return state


def write_external_onnx(path: Path, *locations: str) -> None:
    """Write a minimal ONNX graph referencing the requested external tensors."""

    tensors = []
    for index, location in enumerate(locations):
        tensor = TensorProto(
            name=f"weight_{index}",
            data_type=TensorProto.FLOAT,
            dims=[1],
        )
        tensor.data_location = TensorProto.EXTERNAL
        external_entry = tensor.external_data.add()
        external_entry.key = "location"
        external_entry.value = location
        tensors.append(tensor)
    graph = helper.make_graph((), "external", (), (), initializer=tensors)
    onnx.save_model(helper.make_model(graph), path)


def test_remove_onnx_artifacts_removes_graph_and_nested_external_data(
    tmp_path: Path,
) -> None:
    """Delete the graph and every nested external-data file it references."""

    onnx_path = tmp_path / "model.onnx"
    external_paths = (
        tmp_path / "weights" / "model.data",
        tmp_path / "weights" / "scales.data",
    )
    external_paths[0].parent.mkdir()
    for external_path in external_paths:
        external_path.write_bytes(b"weights")
    write_external_onnx(onnx_path, "weights/model.data", "weights/scales.data")

    remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()
    assert all(not external_path.exists() for external_path in external_paths)


def test_remove_onnx_artifacts_tolerates_missing_external_data(tmp_path: Path) -> None:
    """Remove a valid graph when an external-data artifact is already absent."""

    onnx_path = tmp_path / "model.onnx"
    write_external_onnx(onnx_path, "missing.data")

    remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()


def test_remove_onnx_artifacts_removes_malformed_graph(tmp_path: Path) -> None:
    """Remove the graph in a finally block while preserving parse diagnostics."""

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not an ONNX graph")

    with pytest.raises(DecodeError):
        remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()


def test_remove_onnx_artifacts_reports_missing_graph(tmp_path: Path) -> None:
    """Do not silently treat an absent ONNX graph as a successful cleanup."""

    with pytest.raises(FileNotFoundError):
        remove_onnx_artifacts(tmp_path / "missing.onnx")


def test_build_tensorrt_engine_configures_dynamic_network(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
) -> None:
    """Apply every builder policy and serialize one accepted dynamic profile."""

    fake_tensorrt_builder.inputs[:] = [
        FakeTensorRTInput("audio", (-1, -1)),
        FakeTensorRTInput("lengths", (8,)),
    ]
    profiles = {"audio": ((1, 1600), (8, 16_000), (8, 640_000))}
    onnx_path = tmp_path / "encoder.onnx"
    engine_path = tmp_path / "encoder.trt"

    build_tensorrt_engine(onnx_path, engine_path, profiles, 5)

    config = fake_tensorrt_builder.config
    assert fake_tensorrt_builder.loaded_plugins == 1
    assert fake_tensorrt_builder.events == [
        "standard_plugins",
        "custom_plugins",
        "parse",
        "build",
    ]
    assert fake_tensorrt_builder.standard_plugin_args == (
        fake_tensorrt_builder.logger,
        "",
    )
    assert fake_tensorrt_builder.parser_args == (
        fake_tensorrt_builder.network,
        fake_tensorrt_builder.logger,
    )
    assert fake_tensorrt_builder.parsed_path == str(onnx_path)
    assert fake_tensorrt_builder.built == (
        fake_tensorrt_builder.network,
        config,
    )
    assert fake_tensorrt_builder.network_flags == 1 << 2
    assert config.engine_capability == 20
    assert config.tactic_sources == (1 << 0) | (1 << 2)
    assert config.flags == [10, 11, 12, 13]
    assert config.preview_features == [(30, True)]
    assert config.builder_optimization_level == 5
    assert config.profiles == [fake_tensorrt_builder.profile]
    assert fake_tensorrt_builder.profile.shapes == [
        ("audio", (1, 1600), (8, 16_000), (8, 640_000))
    ]
    assert engine_path.read_bytes() == SERIALIZED_ENGINE


def test_build_tensorrt_engine_accepts_fully_static_network(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
) -> None:
    """Avoid adding a meaningless optimization profile to a static graph."""

    fake_tensorrt_builder.inputs[:] = [FakeTensorRTInput("tokens", (6, 2))]
    engine_path = tmp_path / "decoder.trt"
    engine_path.write_bytes(b"stale engine")

    build_tensorrt_engine(
        tmp_path / "decoder.onnx",
        engine_path,
        {},
        3,
    )

    assert fake_tensorrt_builder.config.profiles == []
    assert fake_tensorrt_builder.profile.shapes == []
    assert engine_path.read_bytes() == SERIALIZED_ENGINE


@pytest.mark.parametrize("missing_flag", ("FP16", "BF16"), ids=("fp16", "bf16"))
def test_build_tensorrt_engine_uses_available_precision_flags(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
    missing_flag: str,
) -> None:
    """Remain compatible with TensorRT releases lacking one precision flag."""

    delattr(export_utils.trt.BuilderFlag, missing_flag)

    build_tensorrt_engine(
        tmp_path / "decoder.onnx",
        tmp_path / "decoder.trt",
        {},
        3,
    )

    expected_flags = {
        name: value
        for name, value in {
            "TF32": 10,
            "SPARSE_WEIGHTS": 11,
            "FP16": 12,
            "BF16": 13,
        }.items()
        if name != missing_flag
    }
    assert fake_tensorrt_builder.config.flags == list(expected_flags.values())


def test_build_tensorrt_engine_requires_plugin_initialization(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
) -> None:
    """Fail before parsing when TensorRT cannot initialize plugin creators."""

    fake_tensorrt_builder.plugin_result = False

    with pytest.raises(RuntimeError, match="initialize TensorRT plugins"):
        build_tensorrt_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.trt",
            {},
            5,
        )

    assert fake_tensorrt_builder.loaded_plugins == 0
    assert fake_tensorrt_builder.events == ["standard_plugins"]
    assert not (tmp_path / "model.trt").exists()


def test_build_tensorrt_engine_reports_every_parser_error(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
) -> None:
    """Preserve all TensorRT ONNX diagnostics when parsing fails."""

    fake_tensorrt_builder.parse_result = False
    fake_tensorrt_builder.parse_errors[:] = ["first", "second"]

    with pytest.raises(RuntimeError, match="first\nsecond"):
        build_tensorrt_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.trt",
            {},
            5,
        )

    assert fake_tensorrt_builder.events == [
        "standard_plugins",
        "custom_plugins",
        "parse",
    ]
    assert not (tmp_path / "model.trt").exists()


@pytest.mark.parametrize(
    ("inputs", "profiles"),
    (
        ((("audio", (-1, -1)),), {}),
        (
            (("audio", (-1, -1)),),
            {"audio": PROFILE_SHAPES, "extra": PROFILE_SHAPES},
        ),
        ((("tokens", (6, 2)),), {"tokens": PROFILE_SHAPES}),
    ),
    ids=("missing", "extra", "static"),
)
def test_build_tensorrt_engine_requires_exact_dynamic_profile_names(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
    inputs: tuple[tuple[str, tuple[int, ...]], ...],
    profiles: dict[
        str,
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    ],
) -> None:
    """Reject missing, extra, and static-input profile entries."""

    fake_tensorrt_builder.inputs[:] = [
        FakeTensorRTInput(name, shape) for name, shape in inputs
    ]

    with pytest.raises(ValueError, match="Expected TensorRT profiles"):
        build_tensorrt_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.trt",
            profiles,
            5,
        )

    assert fake_tensorrt_builder.config.profiles == []
    assert not (tmp_path / "model.trt").exists()


@pytest.mark.parametrize(
    (
        "accept_tactic_sources",
        "accept_profile_shapes",
        "serialized_engine",
        "exception_type",
        "message",
    ),
    (
        (False, True, SERIALIZED_ENGINE, RuntimeError, "rejected tactic source mask"),
        (True, False, SERIALIZED_ENGINE, ValueError, "Invalid TensorRT profile"),
        (True, True, None, RuntimeError, "Failed to build TensorRT engine"),
    ),
    ids=("tactic-sources", "profile-shape", "serialization"),
)
def test_build_tensorrt_engine_rejects_builder_failures(
    tmp_path: Path,
    fake_tensorrt_builder: SimpleNamespace,
    accept_tactic_sources: bool,
    accept_profile_shapes: bool,
    serialized_engine: bytes | None,
    exception_type: type[Exception],
    message: str,
) -> None:
    """Surface each failure reported by TensorRT's builder interfaces."""

    fake_tensorrt_builder.inputs[:] = [FakeTensorRTInput("audio", (-1, -1))]
    fake_tensorrt_builder.config.accept_tactic_sources = accept_tactic_sources
    fake_tensorrt_builder.profile.accept_shapes = accept_profile_shapes
    fake_tensorrt_builder.serialized_engine = serialized_engine

    with pytest.raises(exception_type, match=message):
        build_tensorrt_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.trt",
            {"audio": PROFILE_SHAPES},
            5,
        )

    assert not (tmp_path / "model.trt").exists()
