#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for TensorRT encoder initialization, audio staging, and execution."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt

import fast_gpu_asr.encoder.encoder as encoder_module
from fast_gpu_asr.encoder.encoder import Encoder
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError


class FakeArray:
    """Minimal NumPy-backed stand-in for the CuPy arrays used by ``Encoder``."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.data = SimpleNamespace(ptr=id(values))

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def size(self) -> int:
        return self.values.size

    def __getitem__(self, key: object) -> "FakeArray":
        return FakeArray(self.values[key])

    def fill(self, value: float | int) -> None:
        self.values.fill(value)

    def reshape(self, shape: tuple[int, ...]) -> "FakeArray":
        return FakeArray(self.values.reshape(shape))

    def set(self, source: np.ndarray, *, stream: object) -> None:
        del stream
        np.copyto(self.values, source)


class FakeScope:
    """Context manager carrying the fields used by CUDA device and stream code."""

    def __init__(self, identifier: int = 0) -> None:
        self.id = identifier
        self.ptr = identifier + 100

    def __enter__(self) -> "FakeScope":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeGraph:
    """Record graph upload and replay calls."""

    def __init__(self) -> None:
        self.uploads = 0
        self.launches = 0

    def upload(self, stream: object) -> None:
        del stream
        self.uploads += 1

    def launch(self, stream: object) -> None:
        del stream
        self.launches += 1


class FakeStream(FakeScope):
    """Record synchronization and CUDA graph capture operations."""

    def __init__(self) -> None:
        super().__init__(17)
        self.synchronizations = 0
        self.capture_starts = 0
        self.captured_graphs: list[FakeGraph] = []

    def synchronize(self) -> None:
        self.synchronizations += 1

    def begin_capture(self) -> None:
        self.capture_starts += 1

    def end_capture(self) -> FakeGraph:
        graph = FakeGraph()
        self.captured_graphs.append(graph)
        return graph


class FakeCaptureError(RuntimeError):
    """CuPy-compatible graph capture error carrying a CUDA status code."""

    def __init__(self, status: int) -> None:
        super().__init__(f"capture failed with status {status}")
        self.status = status


class InvalidatingFakeStream(FakeStream):
    """Invalidate every attempted CUDA graph capture."""

    def end_capture(self) -> FakeGraph:
        raise FakeCaptureError(901)


class FakeEvent:
    """Record host-transfer completion fences."""

    def __init__(self) -> None:
        self.records = 0
        self.synchronizations = 0

    def record(self, stream: object) -> None:
        del stream
        self.records += 1

    def synchronize(self) -> None:
        self.synchronizations += 1


class FakeExecutionContext:
    """Provide the dynamic-shape TensorRT methods used by ``Encoder``."""

    def __init__(self) -> None:
        self.audio_shape = (2, 4)
        self.accept_profile = True
        self.accept_input_shape = True
        self.rejected_address: str | None = None
        self.execute_results: list[bool] = []
        self.execute_calls = 0
        self.profile_calls: list[tuple[int, int]] = []
        self.tensor_addresses: list[str] = []
        self.device_memory: tuple[int, int] | None = None
        self.aux_stream_ptrs: list[int] | None = None

    def set_optimization_profile_async(
        self, profile_index: int, stream_ptr: int
    ) -> bool:
        self.profile_calls.append((profile_index, stream_ptr))
        return self.accept_profile

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        assert name == "audio"
        if not self.accept_input_shape:
            return False
        self.audio_shape = shape
        return True

    def update_device_memory_size_for_shapes(self) -> int:
        return 64

    def set_device_memory(self, pointer: int, size: int) -> None:
        self.device_memory = (pointer, size)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        assert name == "encoder_output"
        return (self.audio_shape[0], max(1, self.audio_shape[1] // 2), 4)

    def set_tensor_address(self, name: str, pointer: int) -> bool:
        assert pointer > 0
        self.tensor_addresses.append(name)
        return name != self.rejected_address

    def set_aux_streams(self, stream_ptrs: list[int]) -> None:
        self.aux_stream_ptrs = stream_ptrs

    def execute_async_v3(self, stream_ptr: int) -> bool:
        assert stream_ptr == 117
        self.execute_calls += 1
        return self.execute_results.pop(0) if self.execute_results else True


class FakeEncoderEngine:
    """Provide the TensorRT encoder metadata used during initialization."""

    def __init__(
        self,
        context: FakeExecutionContext | None,
        output_dtype: trt.DataType = trt.float16,
        num_aux_streams: int = 0,
    ) -> None:
        self.context = context
        self.output_dtype = output_dtype
        self.num_aux_streams = num_aux_streams

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        assert (name, profile_index) == ("audio", 0)
        return (2, 6), (2, 8), (2, 10)

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        return {
            "audio": trt.float32,
            "audio_lengths": trt.int64,
            "encoder_output": self.output_dtype,
            "encoder_output_lengths": trt.int32,
        }[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        assert name in ("audio_lengths", "encoder_output_lengths")
        return (2,)

    def create_execution_context(self, strategy: object) -> FakeExecutionContext | None:
        assert strategy == trt.ExecutionContextAllocationStrategy.USER_MANAGED
        return self.context


def make_runtime_encoder(
    monkeypatch: pytest.MonkeyPatch,
    trim_devices: list[int] | None = None,
    trim_status: int = 0,
) -> Encoder:
    """Create a NumPy-backed encoder that exercises successful runtime flow."""

    encoder = make_uninitialized_encoder()
    encoder.right_padding_samples = 2
    encoder.device = FakeScope(3)  # type: ignore[assignment]
    encoder.stream = FakeStream()  # type: ignore[assignment]
    encoder.encoder = FakeExecutionContext()  # type: ignore[assignment]
    encoder.dtypes = {"audio": np.float32, "encoder_output": np.float32}
    encoder.lengths = FakeArray(np.empty(2, dtype=np.int64))  # type: ignore[assignment]
    encoder.lengths_host = np.empty(2, dtype=np.int64)
    encoder.output_lengths = FakeArray(np.array([2, 1], dtype=np.int32))  # type: ignore[assignment]
    encoder.audio = None
    encoder.audio_host = None
    encoder.encoder_output = None
    encoder.cuda_graph = None
    encoder.cuda_graph_shape = None
    encoder.cuda_graph_supported = True
    encoder.host_transfer_event = FakeEvent()  # type: ignore[assignment]
    encoder.host_transfer_pending = False
    encoder.context_memory = None
    encoder.context_memory_size = 0
    encoder.aux_streams = []
    encoder.audio_copy_pool = None  # type: ignore[assignment]

    monkeypatch.setattr(
        encoder_module.cp,
        "empty",
        lambda size, dtype: FakeArray(np.empty(size, dtype=dtype)),
    )
    monkeypatch.setattr(
        encoder_module.cpx,
        "empty_pinned",
        lambda size, dtype: np.empty(size, dtype=dtype),
    )
    monkeypatch.setattr(
        encoder_module.cp.cuda,
        "Memory",
        lambda size: SimpleNamespace(ptr=9000 + size),
    )
    memory_pool = SimpleNamespace(free_all_blocks=lambda: None)
    monkeypatch.setattr(
        encoder_module.cp, "get_default_memory_pool", lambda: memory_pool
    )
    monkeypatch.setattr(
        encoder_module.cp,
        "get_default_pinned_memory_pool",
        lambda: memory_pool,
    )
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 1)
    if trim_devices is None:
        trim_devices = []
    monkeypatch.setattr(
        encoder_module,
        "runtime",
        SimpleNamespace(
            cudaDeviceGraphMemTrim=lambda device_id: (
                trim_devices.append(device_id) or trim_status,
            ),
            cudaError_t=SimpleNamespace(cudaSuccess=0),
        ),
    )
    return encoder


def make_uninitialized_encoder() -> Encoder:
    """Construct the validation-only portion of an encoder without CUDA."""

    encoder = Encoder.__new__(Encoder)
    encoder.batch_size = 2
    encoder.min_samples = 4
    encoder.max_samples = 8
    encoder.sample_rate = 4
    encoder.right_padding_samples = 0
    return encoder


@pytest.mark.parametrize("encoder_output_dtype", (trt.float16, trt.bfloat16))
def test_encoder_initializes_engine_metadata_and_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_output_dtype: trt.DataType,
) -> None:
    context = FakeExecutionContext()
    aux_streams: list[FakeScope] = []
    loaded_paths: list[Path] = []

    def make_aux_stream(**kwargs: object) -> FakeScope:
        assert kwargs == {"null": False, "non_blocking": True, "ptds": False}
        stream = FakeScope(len(aux_streams) + 1)
        aux_streams.append(stream)
        return stream

    monkeypatch.setattr(encoder_module.cp.cuda, "Device", FakeScope)
    monkeypatch.setattr(encoder_module.cp.cuda, "Stream", make_aux_stream)
    monkeypatch.setattr(encoder_module.cp.cuda, "Event", lambda **_: FakeEvent())

    def load_engine(engine_path: Path) -> FakeEncoderEngine:
        loaded_paths.append(engine_path)
        return FakeEncoderEngine(context, encoder_output_dtype, num_aux_streams=2)

    monkeypatch.setattr(encoder_module, "get_engine", load_engine)
    monkeypatch.setattr(
        encoder_module,
        "get_names",
        lambda _: (
            ("audio", "audio_lengths"),
            ("encoder_output", "encoder_output_lengths"),
        ),
    )
    monkeypatch.setattr(
        encoder_module.cp,
        "empty",
        lambda shape, dtype: FakeArray(np.empty(shape, dtype=dtype)),
    )
    monkeypatch.setattr(
        encoder_module.cpx,
        "empty_pinned",
        lambda shape, dtype: np.empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 1)

    engine_path = tmp_path / "encoder.trt"
    encoder = Encoder(
        engine_path,
        sample_rate=16_000,
        device_id=3,
        stream=FakeStream(),  # type: ignore[arg-type]
        right_padding_samples=2,
    )
    try:
        assert encoder.batch_size == 2
        assert encoder.min_samples == 4
        assert encoder.max_samples == 8
        assert encoder.sample_rate == 16_000
        assert encoder.right_padding_samples == 2
        expected_output_dtype = (
            encoder_module.cp.dtype("bfloat16")
            if encoder_output_dtype == trt.bfloat16
            else np.dtype(np.float16)
        )
        assert encoder.dtypes == {
            "audio": np.dtype(np.float32),
            "audio_lengths": np.dtype(np.int64),
            "encoder_output": expected_output_dtype,
            "encoder_output_lengths": np.dtype(np.int32),
        }
        assert loaded_paths == [engine_path]
        assert encoder.device.id == 3
        assert context.profile_calls == [(0, 117)]
        assert encoder.lengths.shape == (2,)
        assert encoder.lengths_host.shape == (2,)
        assert encoder.output_lengths.shape == (2,)
        assert encoder.aux_streams == aux_streams
        assert len(aux_streams) == 2
    finally:
        encoder.audio_copy_pool.shutdown(wait=True)


def test_copy_audio_range_reflects_and_zero_pads() -> None:
    encoder = Encoder.__new__(Encoder)
    encoder.right_padding_samples = 3
    audios = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
    ]
    audio_host = np.empty((2, 8), dtype=np.float32)

    encoder.copy_audio_range(audios, audio_host, 0, len(audios))

    np.testing.assert_array_equal(
        audio_host,
        np.array(
            (
                (1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 0.0),
                (5.0, 6.0, 6.0, 5.0, 5.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )


def test_copy_audio_range_without_right_padding() -> None:
    encoder = make_uninitialized_encoder()
    audio_host = np.full((2, 6), -1.0, dtype=np.float32)

    encoder.copy_audio_range(
        [np.array([1.0, 2.0], dtype=np.float32)],
        audio_host,
        0,
        1,
    )

    np.testing.assert_array_equal(audio_host[0], [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(audio_host[1], np.full(6, -1.0, dtype=np.float32))


def test_encoder_successfully_prepares_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.aux_streams = [FakeScope(1), FakeScope(2)]  # type: ignore[list-item]

    encoder_output, output_lengths = encoder(
        [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    )

    assert isinstance(encoder.audio, FakeArray)
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 6),
        np.array(
            (
                (1.0, 2.0, 3.0, 3.0, 2.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 4])
    assert encoder_output.shape == (1, 3, 4)
    np.testing.assert_array_equal(output_lengths.values, [2])
    assert encoder.encoder.tensor_addresses == [
        "audio",
        "audio_lengths",
        "encoder_output",
        "encoder_output_lengths",
    ]
    assert encoder.encoder.device_memory == (9064, 64)
    assert encoder.encoder.aux_stream_ptrs == [101, 102]
    assert encoder.host_transfer_event.records == 1
    assert encoder.host_transfer_pending
    assert encoder.stream.synchronizations == 0


def test_encoder_converts_audio_and_accepts_exact_maximum_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)

    encoder_output, output_lengths = encoder(
        [np.arange(8, dtype=np.float64)]  # type: ignore[list-item]
    )

    assert isinstance(encoder.audio, FakeArray)
    assert encoder.audio.dtype == np.float32
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 10),
        np.array(
            (
                (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 6.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [8, 4])
    assert encoder_output.shape == (1, 5, 4)
    np.testing.assert_array_equal(output_lengths.values, [2])


def test_encoder_successfully_stages_audio_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.audio_copy_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 1)
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 2)

    try:
        encoder_output, output_lengths = encoder(
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0], dtype=np.float32),
            ]
        )
    finally:
        encoder.audio_copy_pool.shutdown(wait=True)

    assert isinstance(encoder.audio, FakeArray)
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 6),
        np.array(
            (
                (1.0, 2.0, 3.0, 3.0, 2.0, 0.0),
                (4.0, 5.0, 5.0, 4.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 2])
    assert encoder_output.shape == (2, 3, 4)
    np.testing.assert_array_equal(output_lengths.values, [2, 1])
    assert encoder.host_transfer_event.records == 1
    assert encoder.host_transfer_pending


def test_encoder_captures_replays_and_invalidates_dynamic_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    assert encoder.encoder.execute_calls == 1
    assert encoder.cuda_graph_shape == (2, 6)
    assert encoder.cuda_graph is None

    encoder([waveform])
    graph = encoder.cuda_graph
    assert isinstance(graph, FakeGraph)
    assert encoder.encoder.execute_calls == 2
    assert graph.uploads == 1
    assert graph.launches == 1

    encoder([waveform])
    assert encoder.encoder.execute_calls == 2
    assert graph.launches == 2

    encoder([np.ones(6, dtype=np.float32)])
    assert encoder.encoder.execute_calls == 3
    assert encoder.cuda_graph is None
    assert encoder.cuda_graph_shape == (2, 8)
    assert trim_devices == [3]


def test_encoder_synchronizes_before_destroying_graph_for_smaller_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    large_waveform = np.ones(6, dtype=np.float32)

    encoder([large_waveform])
    encoder([large_waveform])
    assert isinstance(encoder.cuda_graph, FakeGraph)
    synchronizations = encoder.stream.synchronizations

    encoder([np.ones(3, dtype=np.float32)])

    assert encoder.stream.synchronizations == synchronizations + 1
    assert encoder.cuda_graph is None
    assert trim_devices == []


def test_encoder_synchronizes_before_replacing_live_device_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder([np.ones(3, dtype=np.float32)])
    synchronizations = encoder.stream.synchronizations

    encoder([np.ones(6, dtype=np.float32)])

    # Input and output growth each release a potentially live allocation.
    assert encoder.stream.synchronizations == synchronizations + 2


def test_encoder_drains_parallel_workers_and_dma_after_copy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.audio_copy_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 1)
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 2)

    transfer_started = Event()
    failed_worker_finished = Event()
    failure = RuntimeError("host copy failed")
    original_set = FakeArray.set
    original_copy = encoder.copy_audio_range

    def recording_set(array: FakeArray, source: np.ndarray, *, stream: object) -> None:
        original_set(array, source, stream=stream)
        transfer_started.set()

    def copy_or_fail(
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        if start == 1:
            assert transfer_started.wait(timeout=5)
            failed_worker_finished.set()
            raise failure
        original_copy(audios, audio_host, start, end)

    monkeypatch.setattr(FakeArray, "set", recording_set)
    monkeypatch.setattr(encoder, "copy_audio_range", copy_or_fail)
    try:
        with pytest.raises(RuntimeError) as error:
            encoder(
                [
                    np.ones(3, dtype=np.float32),
                    np.ones(3, dtype=np.float32),
                ]
            )
    finally:
        encoder.audio_copy_pool.shutdown(wait=True)

    assert error.value is failure
    assert failed_worker_finished.is_set()
    assert encoder.stream.synchronizations == 1
    assert not encoder.host_transfer_pending


def test_encoder_falls_back_after_cuda_graph_capture_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    encoder.stream = InvalidatingFakeStream()  # type: ignore[assignment]
    monkeypatch.setattr(
        encoder_module.cp.cuda.runtime,
        "CUDARuntimeError",
        FakeCaptureError,
    )
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        encoder([waveform])

    assert encoder.encoder.execute_calls == 3
    assert not encoder.cuda_graph_supported
    assert encoder.cuda_graph is None
    assert encoder.cuda_graph_shape is None
    assert trim_devices == [3]

    encoder([waveform])
    assert encoder.encoder.execute_calls == 4
    assert encoder.stream.capture_starts == 1


def test_encoder_falls_back_when_captured_execution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    encoder.encoder.execute_results = [True, False, True]
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    with pytest.warns(RuntimeWarning, match="CUDA graph capture failed"):
        encoder([waveform])

    assert encoder.encoder.execute_calls == 3
    assert not encoder.cuda_graph_supported
    assert encoder.cuda_graph is None
    assert encoder.cuda_graph_shape is None
    assert trim_devices == [3]


def test_encoder_reports_cuda_graph_memory_trim_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices, trim_status=17)
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    encoder([waveform])
    assert isinstance(encoder.cuda_graph, FakeGraph)

    with pytest.raises(ASRInferenceError, match="Failed to trim CUDA graph memory"):
        encoder([np.ones(6, dtype=np.float32)])

    assert trim_devices == [3]


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("shape", "rejected encoder input shape"),
        ("address", "rejected encoder tensor address for encoder_output"),
        ("execution", "TensorRT encoder execution failed"),
    ),
)
def test_encoder_reports_tensorrt_runtime_rejection(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    if failure == "shape":
        encoder.encoder.accept_input_shape = False
    elif failure == "address":
        encoder.encoder.rejected_address = "encoder_output"
    else:
        encoder.encoder.execute_results = [False]

    with pytest.raises(ASRInferenceError, match=message):
        encoder([np.ones(3, dtype=np.float32)])


@pytest.mark.parametrize(
    ("audios", "message"),
    (
        ([], "Expected 1 to 2 audio waveforms"),
        (
            [
                np.ones(1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ],
            "Expected 1 to 2 audio waveforms",
        ),
        ([np.empty(0, dtype=np.float32)], "non-empty one-dimensional"),
        ([np.ones((1, 2), dtype=np.float32)], "non-empty one-dimensional"),
        ([np.ones(9, dtype=np.float32)], "2.000-second TensorRT profile"),
    ),
)
def test_encoder_rejects_invalid_audio_before_cuda(
    audios: list[np.typing.NDArray[np.float32]],
    message: str,
) -> None:
    encoder = make_uninitialized_encoder()

    with pytest.raises(ASRInferenceError, match=message):
        encoder(audios)


@pytest.mark.parametrize(
    ("context_available", "message"),
    (
        (False, "create the encoder execution context"),
        (True, "select encoder optimization profile 0"),
    ),
)
def test_encoder_rejects_invalid_execution_context_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_available: bool,
    message: str,
) -> None:
    class FakeDevice:
        def __init__(self, device_id: int) -> None:
            assert device_id == 3

        def __enter__(self) -> "FakeDevice":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeStream:
        ptr = 17

    fake_context = FakeExecutionContext() if context_available else None
    if fake_context is not None:
        fake_context.accept_profile = False
    monkeypatch.setattr(encoder_module.cp.cuda, "Device", FakeDevice)
    monkeypatch.setattr(
        encoder_module,
        "get_engine",
        lambda _: FakeEncoderEngine(fake_context),
    )
    monkeypatch.setattr(
        encoder_module,
        "get_names",
        lambda _: (
            ("audio", "audio_lengths"),
            ("encoder_output", "encoder_output_lengths"),
        ),
    )

    with pytest.raises(ASRInitializationError, match=message):
        Encoder(tmp_path / "encoder.trt", 16000, 3, FakeStream(), 0)  # type: ignore[arg-type]


def test_encoder_waits_before_reusing_pinned_host_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_uninitialized_encoder()
    events: list[str] = []

    class FakeContext:
        ptr = 11

        def __enter__(self) -> "FakeContext":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeEvent:
        def synchronize(self) -> None:
            events.append("synchronize")

    failure = RuntimeError("stop after staging fence")

    def fail_allocation(*args: object, **kwargs: object) -> None:
        events.append("allocate")
        raise failure

    encoder.device = FakeContext()  # type: ignore[assignment]
    encoder.stream = FakeContext()  # type: ignore[assignment]
    encoder.host_transfer_event = FakeEvent()  # type: ignore[assignment]
    encoder.host_transfer_pending = True
    encoder.cuda_graph = None
    encoder.cuda_graph_shape = None
    encoder.audio = None
    encoder.dtypes = {"audio": np.float32}
    monkeypatch.setattr(encoder_module.cp, "empty", fail_allocation)

    with pytest.raises(RuntimeError) as error:
        encoder([np.ones(4, dtype=np.float32)])

    assert error.value is failure
    assert events == ["synchronize", "allocate"]
    assert not encoder.host_transfer_pending
