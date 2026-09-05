#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for shared ONNX cleanup and TensorRT export utilities."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import onnx
import pytest
from google.protobuf.message import DecodeError
from onnx import TensorProto, helper

import fast_gpu_asr.export.export_utils as export_utils
from fast_gpu_asr.export.export_utils import (
    build_tensorrt_engine,
    remove_onnx_artifacts,
)

PROFILE_SHAPES = ((8, 1), (8, 2), (8, 3))


@pytest.fixture
def fake_tensorrt(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Record builder calls without invoking TensorRT or requiring a GPU.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores patched TensorRT bindings and plugin loading after the test.

    Returns
    -------
    SimpleNamespace
        Configurable builder, parser, profile, and plugin-loader mocks, plus
        a shared call trace for asserting export operation order.
    """

    network = Mock(num_inputs=1)
    network.get_input.return_value = SimpleNamespace(name="audio", shape=(8, -1))
    profile = Mock()
    profile.set_shape.return_value = None
    config = Mock()
    config.set_tactic_sources.return_value = True
    config.add_optimization_profile.return_value = 0
    parser = Mock(num_errors=2)
    parser.parse_from_file.return_value = True
    parser.get_error.side_effect = ["first", "second"].__getitem__
    builder = Mock()
    builder.create_network.return_value = network
    builder.create_builder_config.return_value = config
    builder.create_optimization_profile.return_value = profile
    builder.build_serialized_network.return_value = b"serialized engine"
    load_plugins = Mock(return_value=None)
    trt = SimpleNamespace(
        Builder=Mock(return_value=builder),
        BuilderFlag=SimpleNamespace(TF32=10, SPARSE_WEIGHTS=11, FP16=12, BF16=13),
        EngineCapability=SimpleNamespace(STANDARD=20),
        Logger=Mock(INFO=1),
        NetworkDefinitionCreationFlag=SimpleNamespace(PREFER_AOT_PYTHON_PLUGINS=2),
        OnnxParser=Mock(return_value=parser),
        PreviewFeature=SimpleNamespace(ALIASED_PLUGIN_IO_10_03=30),
        TacticSource=SimpleNamespace(__members__={"A": 0, "B": 2}),
        init_libnvinfer_plugins=Mock(return_value=True),
    )
    pipeline = Mock()
    for name, operation in (
        ("initialize", trt.init_libnvinfer_plugins),
        ("load_plugins", load_plugins),
        ("parse", parser.parse_from_file),
        ("set_shape", profile.set_shape),
        ("add_profile", config.add_optimization_profile),
        ("build", builder.build_serialized_network),
    ):
        pipeline.attach_mock(operation, name)
    monkeypatch.setattr(export_utils, "trt", trt)
    monkeypatch.setattr(export_utils, "load_tensorrt_plugins", load_plugins)
    return SimpleNamespace(
        trt=trt,
        network=network,
        profile=profile,
        config=config,
        parser=parser,
        builder=builder,
        load_plugins=load_plugins,
        pipeline=pipeline,
    )


@pytest.fixture
def build_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create an ONNX source and a prior engine for artifact-lifecycle checks.

    Parameters
    ----------
    tmp_path : Path
        Per-test temporary directory.

    Returns
    -------
    tuple[Path, Path]
        ONNX source and existing engine paths with distinct sentinel contents.
    """

    onnx_path, engine_path = tmp_path / "model.onnx", tmp_path / "model.trt"
    onnx_path.write_bytes(b"onnx graph")
    engine_path.write_bytes(b"stale engine")
    return onnx_path, engine_path


def assert_build_artifacts_unchanged(paths: tuple[Path, Path]) -> None:
    """Check that a failed build preserved both its input and the prior engine.

    Parameters
    ----------
    paths : tuple[Path, Path]
        Source and engine paths created by ``build_paths``.
    """

    assert paths[0].read_bytes() == b"onnx graph"
    assert paths[1].read_bytes() == b"stale engine"


def write_onnx(path: Path, locations=(), metadata=()) -> None:
    """Write external initializers with location and optional offset/length metadata.

    Parameters
    ----------
    path : Path
        Destination ONNX graph path; parent directories must already exist.
    locations : iterable[str]
        External-data locations, one per initializer. No data files are created.
    metadata : iterable[tuple[str, str]]
        Extra external-data entries shared by all initializers, such as byte
        offsets and lengths. These are metadata, not additional file paths.
    """

    tensors = [
        TensorProto(
            name=f"weight_{index}",
            data_type=TensorProto.FLOAT,
            dims=[1],
            data_location=TensorProto.EXTERNAL,
            external_data=[
                onnx.StringStringEntryProto(key=key, value=value)
                for key, value in (("location", location), *metadata)
            ],
        )
        for index, location in enumerate(locations)
    ]
    graph = helper.make_graph((), "external", (), (), initializer=tensors)
    onnx.save_model(helper.make_model(graph), path)


def test_remove_onnx_artifacts_removes_only_referenced_data(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    weights = tmp_path / "weights"
    weights.mkdir()
    external_paths = [weights / "model.data", weights / "scales.data"]
    keep_paths = [weights / "keep.data", tmp_path / "0", tmp_path / "4"]
    for path in external_paths:
        path.write_bytes(b"weights")
    for path in keep_paths:
        path.write_bytes(b"keep")
    write_onnx(
        onnx_path,
        ("weights/model.data", "weights/scales.data", "weights/model.data"),
        (("offset", "0"), ("length", "4")),
    )

    remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()
    assert all(not path.exists() for path in external_paths)
    assert all(path.read_bytes() == b"keep" for path in keep_paths)


def test_remove_onnx_artifacts_removes_attribute_external_data(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    external_path = tmp_path / "constant.data"
    external_path.write_bytes(b"constant")
    tensor = TensorProto(
        name="constant",
        data_type=TensorProto.FLOAT,
        dims=[1],
        data_location=TensorProto.EXTERNAL,
        external_data=[
            onnx.StringStringEntryProto(key="location", value="constant.data")
        ],
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", (), ("output",), value=tensor)],
        "attribute-external",
        (),
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    onnx.save_model(helper.make_model(graph), onnx_path)

    remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()
    assert not external_path.exists()


@pytest.mark.parametrize(
    "locations", ((), ("missing.data",)), ids=("no-external-data", "missing")
)
def test_remove_onnx_artifacts_without_external_files(
    tmp_path: Path, locations
) -> None:
    onnx_path = tmp_path / "model.onnx"
    write_onnx(onnx_path, locations)

    remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()


def test_remove_onnx_artifacts_removes_malformed_graph(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not an ONNX graph")

    with pytest.raises(DecodeError):
        remove_onnx_artifacts(onnx_path)

    assert not onnx_path.exists()


def test_remove_onnx_artifacts_reports_missing_graph(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        remove_onnx_artifacts(tmp_path / "missing.onnx")


@pytest.mark.parametrize("location", ("", "../outside.data"), ids=("empty", "parent"))
def test_remove_onnx_artifacts_rejects_unsafe_location(
    tmp_path: Path,
    location: str,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside_path = tmp_path / "outside.data"
    outside_path.write_bytes(b"keep")
    onnx_path = model_dir / "model.onnx"
    write_onnx(onnx_path, (location,))

    with pytest.raises(ValueError, match="Unsafe ONNX external-data location"):
        remove_onnx_artifacts(onnx_path)

    assert outside_path.read_bytes() == b"keep"
    assert not onnx_path.exists()


@pytest.mark.parametrize("inside", (False, True), ids=("outside", "inside"))
def test_remove_onnx_artifacts_rejects_absolute_location(
    tmp_path: Path,
    inside: bool,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    external_path = (model_dir if inside else tmp_path) / "weights.data"
    external_path.write_bytes(b"keep")
    onnx_path = model_dir / "model.onnx"
    write_onnx(onnx_path, (str(external_path),))

    with pytest.raises(ValueError, match="Unsafe ONNX external-data location"):
        remove_onnx_artifacts(onnx_path)

    assert external_path.read_bytes() == b"keep"
    assert not onnx_path.exists()


@pytest.mark.parametrize("symlink_kind", ("parent", "file"))
def test_remove_onnx_artifacts_rejects_symlink_escape(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    model_dir, outside_dir = tmp_path / "model", tmp_path / "outside"
    model_dir.mkdir()
    outside_dir.mkdir()
    outside_path = outside_dir / "weights.data"
    outside_path.write_bytes(b"keep")
    if symlink_kind == "parent":
        (model_dir / "weights").symlink_to(outside_dir, target_is_directory=True)
        location = "weights/weights.data"
    else:
        (model_dir / "weights.data").symlink_to(outside_path)
        location = "weights.data"
    onnx_path = model_dir / "model.onnx"
    write_onnx(onnx_path, (location,))

    with pytest.raises(ValueError, match="Unsafe ONNX external-data location"):
        remove_onnx_artifacts(onnx_path)

    assert outside_path.read_bytes() == b"keep"
    assert not onnx_path.exists()


def test_remove_onnx_artifacts_removes_graph_when_external_cleanup_fails(
    tmp_path: Path,
) -> None:
    external_directory = tmp_path / "weights"
    external_directory.mkdir()
    onnx_path = tmp_path / "model.onnx"
    write_onnx(onnx_path, (external_directory.name,))

    with pytest.raises(IsADirectoryError):
        remove_onnx_artifacts(onnx_path)

    assert external_directory.is_dir()
    assert not onnx_path.exists()


def test_build_tensorrt_engine_configures_dynamic_network(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake = fake_tensorrt
    inputs = [
        SimpleNamespace(name="audio", shape=(8, -1)),
        SimpleNamespace(name="features", shape=(8, -1, 80)),
        SimpleNamespace(name="lengths", shape=(8,)),
    ]
    fake.network.num_inputs = len(inputs)
    fake.network.get_input.side_effect = inputs.__getitem__
    profiles = {
        "audio": PROFILE_SHAPES,
        "features": ((8, 1, 80), (8, 10, 80), (8, 100, 80)),
    }

    def serialize(network, config):
        """Check builder policies before serialization can observe them.

        Parameters
        ----------
        network : Mock
            Parsed network passed to the fake builder.
        config : Mock
            Configuration that must already contain all requested policies.

        Returns
        -------
        bytes
            Sentinel serialized engine contents.
        """

        assert network is fake.network
        assert config is fake.config
        assert config.engine_capability == 20
        config.set_tactic_sources.assert_called_once_with((1 << 0) | (1 << 2))
        config.set_flag.assert_has_calls(
            [call(flag) for flag in (10, 11, 12, 13)], any_order=True
        )
        assert config.set_flag.call_count == 4
        config.set_preview_feature.assert_called_once_with(30, True)
        assert config.builder_optimization_level == 5
        return b"serialized engine"

    fake.builder.build_serialized_network.side_effect = serialize

    build_tensorrt_engine(*build_paths, profiles, 5)

    fake.trt.Logger.assert_called_once_with(1)
    fake.trt.Builder.assert_called_once_with(fake.trt.Logger.return_value)
    fake.trt.OnnxParser.assert_called_once_with(
        fake.network, fake.trt.Logger.return_value
    )
    fake.builder.create_network.assert_called_once_with(1 << 2)
    fake.builder.create_optimization_profile.assert_called_once_with()
    assert fake.pipeline.mock_calls == [
        call.initialize(fake.trt.Logger.return_value, ""),
        call.load_plugins(),
        call.parse(str(build_paths[0])),
        *(call.set_shape(name, *shapes) for name, shapes in profiles.items()),
        call.add_profile(fake.profile),
        call.build(fake.network, fake.config),
    ]
    assert build_paths[0].read_bytes() == b"onnx graph"
    assert build_paths[1].read_bytes() == b"serialized engine"


@pytest.mark.parametrize("optimization_level", (0, 3))
def test_build_tensorrt_engine_accepts_fully_static_network(
    build_paths,
    fake_tensorrt: SimpleNamespace,
    optimization_level: int,
) -> None:
    fake = fake_tensorrt
    fake.network.get_input.return_value.shape = (8, 3)

    build_tensorrt_engine(*build_paths, {}, optimization_level)

    assert fake.config.builder_optimization_level == optimization_level
    fake.builder.create_optimization_profile.assert_not_called()
    fake.config.add_optimization_profile.assert_not_called()
    assert build_paths[0].read_bytes() == b"onnx graph"
    assert build_paths[1].read_bytes() == b"serialized engine"


@pytest.mark.parametrize("missing_flag", ("FP16", "BF16"))
def test_build_tensorrt_engine_uses_available_precision_flags(
    build_paths,
    fake_tensorrt: SimpleNamespace,
    missing_flag: str,
) -> None:
    fake = fake_tensorrt
    flags = vars(fake.trt.BuilderFlag).copy()
    flags.pop(missing_flag)
    delattr(fake.trt.BuilderFlag, missing_flag)

    build_tensorrt_engine(*build_paths, {"audio": PROFILE_SHAPES}, 3)

    fake.config.set_flag.assert_has_calls(
        [call(flag) for flag in flags.values()], any_order=True
    )
    assert fake.config.set_flag.call_count == len(flags)


@pytest.mark.parametrize("optimization_level", (-1, 6, 1.5))
def test_build_tensorrt_engine_rejects_invalid_optimization_level_before_setup(
    build_paths,
    fake_tensorrt: SimpleNamespace,
    optimization_level: float | int,
) -> None:
    with pytest.raises(ValueError, match="integer from 0 through 5"):
        build_tensorrt_engine(*build_paths, {}, optimization_level)

    fake_tensorrt.trt.Logger.assert_not_called()
    assert fake_tensorrt.pipeline.mock_calls == []
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_requires_plugin_initialization(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake_tensorrt.trt.init_libnvinfer_plugins.return_value = False

    with pytest.raises(RuntimeError, match="initialize TensorRT plugins"):
        build_tensorrt_engine(*build_paths, {}, 5)

    fake_tensorrt.load_plugins.assert_not_called()
    fake_tensorrt.parser.parse_from_file.assert_not_called()
    fake_tensorrt.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_propagates_custom_plugin_failure_before_parse(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake_tensorrt.load_plugins.side_effect = RuntimeError("custom plugin failed")

    with pytest.raises(RuntimeError, match="custom plugin failed"):
        build_tensorrt_engine(*build_paths, {}, 5)

    fake_tensorrt.parser.parse_from_file.assert_not_called()
    fake_tensorrt.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_reports_every_parser_error(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake_tensorrt.parser.parse_from_file.return_value = False

    with pytest.raises(RuntimeError) as error:
        build_tensorrt_engine(*build_paths, {}, 5)

    assert str(error.value) == f"Failed to parse {build_paths[0]}:\nfirst\nsecond"
    fake_tensorrt.builder.create_builder_config.assert_not_called()
    fake_tensorrt.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


@pytest.mark.parametrize(
    ("input_shape", "profiles"),
    (
        ((8, -1), {}),
        ((8, -1), {"audio": PROFILE_SHAPES, "extra": PROFILE_SHAPES}),
        ((8, -1), {"other": PROFILE_SHAPES}),
        ((8, 3), {"audio": PROFILE_SHAPES}),
    ),
    ids=("missing", "extra", "wrong-name", "static"),
)
def test_build_tensorrt_engine_requires_exact_dynamic_profile_names(
    build_paths,
    fake_tensorrt: SimpleNamespace,
    input_shape,
    profiles,
) -> None:
    fake = fake_tensorrt
    fake.network.get_input.return_value.shape = input_shape

    with pytest.raises(ValueError, match="Expected TensorRT profiles"):
        build_tensorrt_engine(*build_paths, profiles, 5)

    fake.builder.create_optimization_profile.assert_not_called()
    fake.config.add_optimization_profile.assert_not_called()
    fake.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_rejects_optimization_profile(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake = fake_tensorrt
    fake.config.add_optimization_profile.return_value = -1

    with pytest.raises(ValueError, match="rejected optimization profile"):
        build_tensorrt_engine(*build_paths, {"audio": PROFILE_SHAPES}, 5)

    fake.profile.set_shape.assert_called_once_with("audio", *PROFILE_SHAPES)
    fake.config.add_optimization_profile.assert_called_once_with(fake.profile)
    fake.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_rejects_tactic_source_mask(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake = fake_tensorrt
    fake.config.set_tactic_sources.return_value = False

    with pytest.raises(RuntimeError, match="rejected tactic source mask"):
        build_tensorrt_engine(*build_paths, {"audio": PROFILE_SHAPES}, 5)

    fake.builder.create_optimization_profile.assert_not_called()
    fake.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_reports_invalid_profile_shape(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake = fake_tensorrt
    profile_error = ValueError("inconsistent dimensions")
    fake.profile.set_shape.side_effect = profile_error

    with pytest.raises(ValueError) as error:
        build_tensorrt_engine(*build_paths, {"audio": PROFILE_SHAPES}, 5)

    assert str(error.value) == (
        "Invalid TensorRT profile for audio: "
        f"{PROFILE_SHAPES[0]}, {PROFILE_SHAPES[1]}, {PROFILE_SHAPES[2]}."
    )
    assert error.value.__cause__ is profile_error
    fake.profile.set_shape.assert_called_once_with("audio", *PROFILE_SHAPES)
    fake.config.add_optimization_profile.assert_not_called()
    fake.builder.build_serialized_network.assert_not_called()
    assert_build_artifacts_unchanged(build_paths)


def test_build_tensorrt_engine_preserves_existing_engine_when_build_fails(
    build_paths,
    fake_tensorrt: SimpleNamespace,
) -> None:
    fake = fake_tensorrt
    fake.builder.build_serialized_network.return_value = None

    with pytest.raises(RuntimeError, match="Failed to build TensorRT engine"):
        build_tensorrt_engine(*build_paths, {"audio": PROFILE_SHAPES}, 5)

    fake.builder.build_serialized_network.assert_called_once_with(
        fake.network, fake.config
    )
    assert_build_artifacts_unchanged(build_paths)


@pytest.mark.sm80
@pytest.mark.parametrize("dynamic", (False, True), ids=("static", "dynamic"))
@pytest.mark.parametrize(
    ("onnx_dtype", "dtype"),
    (
        (TensorProto.FLOAT, "float32"),
        (TensorProto.FLOAT16, "float16"),
        (TensorProto.BFLOAT16, "bfloat16"),
    ),
    ids=("fp32", "fp16", "bf16"),
)
def test_build_tensorrt_engine_executes_real_network(
    tmp_path: Path, dynamic: bool, onnx_dtype: int, dtype: str
) -> None:
    import cupy as cp
    import tensorrt as trt

    shape = (2, "samples" if dynamic else 3)
    graph = helper.make_graph(
        [helper.make_node("Add", ["audio", "audio"], ["output"])],
        "double-audio",
        [helper.make_tensor_value_info("audio", onnx_dtype, shape)],
        [helper.make_tensor_value_info("output", onnx_dtype, shape)],
    )
    onnx_path, engine_path = tmp_path / "model.onnx", tmp_path / "model.trt"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        onnx_path,
    )
    profiles = {"audio": ((2, 1), (2, 3), (2, 5))} if dynamic else {}

    with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True) as stream:
        build_tensorrt_engine(onnx_path, engine_path, profiles, 5)
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        assert engine is not None
        assert engine.num_optimization_profiles == 1
        context = engine.create_execution_context()
        assert context is not None
        if dynamic:
            assert (
                tuple(
                    tuple(dimensions)
                    for dimensions in engine.get_tensor_profile_shape("audio", 0)
                )
                == profiles["audio"]
            )

        for samples in (1, 3, 5, 4, 1) if dynamic else (3,):
            audio = cp.arange(2 * samples, dtype=cp.float32).reshape(2, samples)
            expected = (audio * 2).astype(dtype)
            audio = audio.astype(dtype)
            assert context.set_input_shape("audio", audio.shape)
            assert tuple(context.get_tensor_shape("output")) == audio.shape
            output = cp.empty_like(audio)
            assert context.set_tensor_address("audio", audio.data.ptr)
            assert context.set_tensor_address("output", output.data.ptr)
            assert context.execute_async_v3(stream.ptr)
            stream.synchronize()
            cp.testing.assert_array_equal(output, expected)
