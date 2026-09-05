#!/usr/bin/env python3
# Copyright SoundsGoodAI 2026 - Daniil Kulko

"""Tests for TensorRT encoder initialization, audio staging, and execution."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace, TracebackType
from typing import Any

import numpy as np
import pytest
import tensorrt as trt

import fast_gpu_asr.encoder.encoder as encoder_module
from fast_gpu_asr.encoder.encoder import Encoder
from fast_gpu_asr.utils import ASRInferenceError, ASRInitializationError


class FakeArray:
    """Minimal NumPy-backed stand-in for the CuPy arrays used by ``Encoder``."""

    def __init__(self, values: np.typing.NDArray[np.generic]) -> None:
        """Wrap a NumPy array and expose its address as a CUDA-like pointer.

        Parameters
        ----------
        values : np.typing.NDArray[np.generic]
            Host array that stores the fake device allocation.
        """

        self.values = values
        self.data = SimpleNamespace(ptr=values.ctypes.data)

    @property
    def dtype(self) -> np.dtype:
        """Return the wrapped array's dtype.

        Returns
        -------
        np.dtype
            NumPy dtype of the fake device allocation.
        """

        return self.values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the wrapped array's shape.

        Returns
        -------
        tuple[int, ...]
            Dimensions of the fake device allocation.
        """

        return self.values.shape

    @property
    def size(self) -> int:
        """Return the wrapped array's element count.

        Returns
        -------
        int
            Number of elements in the fake device allocation.
        """

        return self.values.size

    def __getitem__(self, key: Any) -> "FakeArray":
        """Wrap a view selected from the underlying NumPy array.

        Parameters
        ----------
        key : Any
            NumPy index or slice applied to the wrapped array.

        Returns
        -------
        FakeArray
            Fake array backed by the selected view.
        """

        return FakeArray(self.values[key])

    def fill(self, value: float | int) -> None:
        """Fill the wrapped array with one scalar value.

        Parameters
        ----------
        value : float | int
            Value assigned to every array element.
        """

        self.values.fill(value)

    def reshape(self, shape: tuple[int, ...]) -> "FakeArray":
        """Return a reshaped view wrapped as another fake array.

        Parameters
        ----------
        shape : tuple[int, ...]
            Requested dimensions for the view.

        Returns
        -------
        FakeArray
            Fake array backed by the reshaped view.
        """

        return FakeArray(self.values.reshape(shape))

    def set(self, source: np.typing.NDArray[np.generic], stream: "FakeScope") -> None:
        """Emulate an asynchronous host-to-device copy on an active stream.

        Parameters
        ----------
        source : np.typing.NDArray[np.generic]
            Host values copied into the fake device allocation.
        stream : FakeScope
            Active stream associated with the transfer.
        """

        assert stream.active
        np.copyto(self.values, source)


class FakeScope:
    """Context manager carrying the fields used by CUDA device and stream code."""

    def __init__(self, identifier: int = 0) -> None:
        """Initialize an inactive CUDA-like scope and synthetic pointer.

        Parameters
        ----------
        identifier : int
            Synthetic CUDA device or stream identifier.
        """

        self.id = identifier
        self.ptr = identifier + 100
        self.active = False

    def __enter__(self) -> "FakeScope":
        """Enter the scope while rejecting recursive activation.

        Returns
        -------
        FakeScope
            Active context manager instance.
        """

        assert not self.active
        self.active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave an active scope and ignore exception metadata.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Type of the exception leaving the scope, when present.
        exc_value : BaseException | None
            Exception instance leaving the scope, when present.
        traceback : TracebackType | None
            Exception traceback leaving the scope, when present.
        """

        del exc_type, exc_value, traceback
        assert self.active
        self.active = False


class FakeGraph:
    """Record graph upload and replay calls."""

    def __init__(self) -> None:
        """Initialize graph operation counters."""

        self.uploads = 0
        self.launches = 0

    def upload(self, stream: FakeScope) -> None:
        """Record upload to the expected active stream.

        Parameters
        ----------
        stream : FakeScope
            Active stream receiving the captured graph.
        """

        assert stream.active
        assert stream.ptr == 117
        self.uploads += 1

    def launch(self, stream: FakeScope) -> None:
        """Record replay on the expected active stream.

        Parameters
        ----------
        stream : FakeScope
            Active stream used to replay the graph.
        """

        assert stream.active
        assert stream.ptr == 117
        self.launches += 1


class FakeStream(FakeScope):
    """Record synchronization and CUDA graph capture operations."""

    def __init__(self, device: FakeScope | None = None) -> None:
        """Initialize stream counters and an optional owning device.

        Parameters
        ----------
        device : FakeScope | None
            Device scope that must be active whenever the stream is entered.
        """

        super().__init__(17)
        self.device = device
        self.synchronizations = 0
        self.capture_starts = 0

    def __enter__(self) -> "FakeStream":
        """Enter the stream while its owning device is active.

        Returns
        -------
        FakeStream
            Active fake CUDA stream.
        """

        if self.device is not None:
            assert self.device.active
        super().__enter__()
        return self

    def synchronize(self) -> None:
        """Record synchronization of an active stream."""

        assert self.active
        self.synchronizations += 1

    def begin_capture(self) -> None:
        """Record the start of CUDA graph capture."""

        assert self.active
        self.capture_starts += 1

    def end_capture(self) -> FakeGraph:
        """Finish CUDA graph capture and return a fresh fake graph.

        Returns
        -------
        FakeGraph
            Captured graph used to verify upload and replay behavior.
        """

        assert self.active
        return FakeGraph()


class FakeCaptureError(RuntimeError):
    """CuPy-compatible graph capture error carrying a CUDA status code."""

    def __init__(self, status: int) -> None:
        """Initialize the error with a CUDA runtime status.

        Parameters
        ----------
        status : int
            CUDA error code exposed through the CuPy-compatible ``status`` field.
        """

        super().__init__(f"capture failed with status {status}")
        self.status = status


class FailingCaptureStream(FakeStream):
    """Fail CUDA graph capture with a configured runtime status."""

    def __init__(self, device: FakeScope, status: int = 901) -> None:
        """Initialize a stream whose capture ends with the requested error.

        Parameters
        ----------
        device : FakeScope
            Device scope that owns the stream.
        status : int
            CUDA status raised when graph capture ends.
        """

        super().__init__(device)
        self.error = FakeCaptureError(status)

    def end_capture(self) -> FakeGraph:
        """Raise the configured CUDA graph capture error.

        Raises
        ------
        FakeCaptureError
            Always raised with the status supplied at initialization.
        """

        assert self.active
        raise self.error


class GraphClearTrackingEncoder(Encoder):
    """Record the stream state when a live CUDA graph reference is cleared."""

    graph_clear_synchronizations: int | None

    def __setattr__(self, name: str, value: Any) -> None:
        """Record synchronization count when a live graph is discarded.

        Parameters
        ----------
        name : str
            Attribute being assigned.
        value : Any
            New attribute value.
        """

        if (
            name == "cuda_graph"
            and value is None
            and getattr(self, "cuda_graph", None) is not None
        ):
            self.graph_clear_synchronizations = self.stream.synchronizations
        super().__setattr__(name, value)


class FakeEvent:
    """Record host-transfer completion fences."""

    def __init__(self) -> None:
        """Initialize event operation counters."""

        self.records = 0
        self.synchronizations = 0

    def record(self, stream: FakeScope) -> None:
        """Record the event on an active stream.

        Parameters
        ----------
        stream : FakeScope
            Active stream whose preceding work the event fences.
        """

        assert stream.active
        self.records += 1

    def synchronize(self) -> None:
        """Record a host wait for event completion."""

        self.synchronizations += 1


class FakeExecutionContext:
    """Provide the dynamic-shape TensorRT methods used by ``Encoder``."""

    def __init__(self) -> None:
        """Initialize configurable TensorRT responses and call records."""

        self.audio_shape = (2, 4)
        self.accept_profile = True
        self.accept_input_shape = True
        self.rejected_address: str | None = None
        self.required_device_memory_size = 64
        self.execute_results: list[bool] = []
        self.execute_calls = 0
        self.profile_calls: list[tuple[int, int]] = []
        self.tensor_addresses: list[tuple[str, int]] = []
        self.device_memory: tuple[int, int] | None = None
        self.aux_stream_ptrs: list[int] | None = None
        self.stream: FakeStream | None = None

    def set_optimization_profile_async(
        self, profile_index: int, stream_ptr: int
    ) -> bool:
        """Record profile selection and return its configured result.

        Parameters
        ----------
        profile_index : int
            TensorRT optimization-profile index.
        stream_ptr : int
            Synthetic CUDA stream pointer used for asynchronous selection.

        Returns
        -------
        bool
            Whether the fake context accepts the profile.
        """

        self.profile_calls.append((profile_index, stream_ptr))
        return self.accept_profile

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        """Accept and retain an audio shape unless rejection is configured.

        Parameters
        ----------
        name : str
            TensorRT input tensor name; only ``audio`` is accepted.
        shape : tuple[int, ...]
            Runtime audio tensor dimensions.

        Returns
        -------
        bool
            Whether shape assignment succeeds.
        """

        assert name == "audio"
        if not self.accept_input_shape:
            return False
        self.audio_shape = shape
        return True

    def update_device_memory_size_for_shapes(self) -> int:
        """Return the configured context-memory requirement.

        Returns
        -------
        int
            Number of bytes required for the current runtime shapes.
        """

        return self.required_device_memory_size

    def set_device_memory(self, pointer: int, size: int) -> None:
        """Record the user-managed TensorRT context-memory allocation.

        Parameters
        ----------
        pointer : int
            Synthetic device-memory pointer.
        size : int
            Allocation size in bytes.
        """

        self.device_memory = (pointer, size)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Derive a synthetic encoder-output shape from the audio shape.

        Parameters
        ----------
        name : str
            Tensor name; only ``encoder_output`` is accepted.

        Returns
        -------
        tuple[int, ...]
            Dynamic output shape for the current audio input.
        """

        assert name == "encoder_output"
        return (self.audio_shape[0], max(1, self.audio_shape[1] // 2), 4)

    def set_tensor_address(self, name: str, pointer: int) -> bool:
        """Record a binding and reject its name when configured to do so.

        Parameters
        ----------
        name : str
            TensorRT tensor name.
        pointer : int
            Synthetic device address bound to the tensor.

        Returns
        -------
        bool
            Whether TensorRT accepts the binding.
        """

        assert pointer > 0
        self.tensor_addresses.append((name, pointer))
        return name != self.rejected_address

    def set_aux_streams(self, stream_ptrs: list[int]) -> None:
        """Record auxiliary CUDA stream pointers assigned to TensorRT.

        Parameters
        ----------
        stream_ptrs : list[int]
            Synthetic pointers for TensorRT auxiliary streams.
        """

        self.aux_stream_ptrs = stream_ptrs

    def execute_async_v3(self, stream_ptr: int) -> bool:
        """Record asynchronous execution and return its configured result.

        Parameters
        ----------
        stream_ptr : int
            Synthetic CUDA stream pointer used for execution.

        Returns
        -------
        bool
            Next configured execution result, or ``True`` by default.
        """

        assert stream_ptr == 117
        if self.stream is not None:
            assert self.stream.active
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
        """Initialize configurable context, output dtype, and stream metadata.

        Parameters
        ----------
        context : FakeExecutionContext | None
            Execution context returned by the fake engine.
        output_dtype : trt.DataType
            TensorRT dtype reported for ``encoder_output``.
        num_aux_streams : int
            Number of auxiliary streams requested by the engine.
        """

        self.context = context
        self.output_dtype = output_dtype
        self.num_aux_streams = num_aux_streams

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Return the synthetic minimum, optimum, and maximum audio shapes.

        Parameters
        ----------
        name : str
            Profile tensor name; only ``audio`` is accepted.
        profile_index : int
            Optimization-profile index; only profile zero is accepted.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
            Minimum, optimum, and maximum audio shapes.
        """

        assert (name, profile_index) == ("audio", 0)
        return (2, 6), (2, 8), (2, 10)

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        """Return the configured TensorRT dtype for a named tensor.

        Parameters
        ----------
        name : str
            TensorRT input or output tensor name.

        Returns
        -------
        trt.DataType
            Dtype associated with the named tensor.
        """

        return {
            "audio": trt.float32,
            "audio_lengths": trt.int64,
            "encoder_output": self.output_dtype,
            "encoder_output_lengths": trt.int32,
        }[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return the fixed batch shape for encoder length tensors.

        Parameters
        ----------
        name : str
            Name of an input or output length tensor.

        Returns
        -------
        tuple[int, ...]
            Fixed batch shape of the requested tensor.
        """

        assert name in ("audio_lengths", "encoder_output_lengths")
        return (2,)

    def create_execution_context(
        self, strategy: trt.ExecutionContextAllocationStrategy
    ) -> FakeExecutionContext | None:
        """Return the configured user-managed execution context.

        Parameters
        ----------
        strategy : trt.ExecutionContextAllocationStrategy
            Allocation strategy requested by the runtime.

        Returns
        -------
        FakeExecutionContext | None
            Configured context, or ``None`` to emulate creation failure.
        """

        assert strategy == trt.ExecutionContextAllocationStrategy.USER_MANAGED
        return self.context


def make_runtime_encoder(
    monkeypatch: pytest.MonkeyPatch,
    trim_devices: list[int] | None = None,
    trim_status: int = 0,
    encoder_type: type[Encoder] = Encoder,
) -> Encoder:
    """Create a NumPy-backed encoder that exercises successful runtime flow.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace CuPy and CUDA operations with CPU-backed fakes.
    trim_devices : list[int] | None
        Optional list that records CUDA devices whose graph pools are trimmed.
    trim_status : int
        CUDA status returned by the fake graph-memory trim operation.
    encoder_type : type[Encoder]
        Encoder subclass instantiated without running its constructor.

    Returns
    -------
    Encoder
        Configured encoder suitable for deterministic runtime tests.
    """

    encoder = make_uninitialized_encoder(encoder_type)
    if isinstance(encoder, GraphClearTrackingEncoder):
        encoder.graph_clear_synchronizations = None
    encoder.right_padding_samples = 2
    encoder.device = FakeScope(3)  # type: ignore[assignment]
    encoder.stream = FakeStream(encoder.device)  # type: ignore[assignment]
    encoder.encoder = FakeExecutionContext()  # type: ignore[assignment]
    encoder.encoder.stream = encoder.stream
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
        encoder_module.cp.cuda, "Memory", lambda size: SimpleNamespace(ptr=9000 + size)
    )
    memory_pool = SimpleNamespace(free_all_blocks=lambda: None)
    monkeypatch.setattr(
        encoder_module.cp, "get_default_memory_pool", lambda: memory_pool
    )
    monkeypatch.setattr(
        encoder_module.cp, "get_default_pinned_memory_pool", lambda: memory_pool
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
    monkeypatch.setattr(
        encoder_module.cp.cuda.runtime, "CUDARuntimeError", FakeCaptureError
    )
    return encoder


def make_uninitialized_encoder(encoder_type: type[Encoder] = Encoder) -> Encoder:
    """Construct the validation-only portion of an encoder without CUDA.

    Parameters
    ----------
    encoder_type : type[Encoder]
        Encoder subclass instantiated without running its constructor.

    Returns
    -------
    Encoder
        Partially initialized encoder with fixed runtime limits.
    """

    encoder = encoder_type.__new__(encoder_type)
    encoder.batch_size = 2
    encoder.min_samples = 4
    encoder.max_samples = 8
    encoder.sample_rate = 4
    encoder.right_padding_samples = 0
    return encoder


@pytest.mark.parametrize(
    ("encoder_output_dtype", "reported_cpu_count", "expected_workers"),
    (
        pytest.param(trt.float32, 1, 1, id="fp32-single-cpu"),
        pytest.param(trt.float16, None, 1, id="fp16-unknown-cpu"),
        pytest.param(trt.bfloat16, 8, 2, id="bf16-batch-limited"),
    ),
)
def test_encoder_initializes_engine_metadata_and_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_output_dtype: trt.DataType,
    reported_cpu_count: int | None,
    expected_workers: int,
) -> None:
    context = FakeExecutionContext()
    device = FakeScope(3)
    aux_streams: list[FakeScope] = []
    loaded_paths: list[Path] = []
    pool_calls: list[tuple[int, str]] = []

    def make_device(device_id: int) -> FakeScope:
        """Return the expected fake CUDA device.

        Parameters
        ----------
        device_id : int
            CUDA ordinal requested by the encoder.

        Returns
        -------
        FakeScope
            Shared fake device scope used by the test.
        """

        assert device_id == 3
        return device

    def make_aux_stream(null: bool, non_blocking: bool, ptds: bool) -> FakeScope:
        """Create and record one nonblocking TensorRT auxiliary stream.

        Parameters
        ----------
        null : bool
            Whether CuPy requested the null stream.
        non_blocking : bool
            Whether the stream should avoid implicit synchronization.
        ptds : bool
            Whether CuPy requested per-thread default-stream behavior.

        Returns
        -------
        FakeScope
            Newly recorded auxiliary stream.
        """

        assert (null, non_blocking, ptds) == (False, True, False)
        assert device.active
        stream = FakeScope(len(aux_streams) + 1)
        aux_streams.append(stream)
        return stream

    monkeypatch.setattr(encoder_module.cp.cuda, "Device", make_device)
    monkeypatch.setattr(encoder_module.cp.cuda, "Stream", make_aux_stream)
    monkeypatch.setattr(encoder_module.cp.cuda, "Event", lambda **_: FakeEvent())

    def load_engine(engine_path: Path) -> FakeEncoderEngine:
        """Record the engine path and return fake TensorRT metadata.

        Parameters
        ----------
        engine_path : Path
            Serialized engine path requested during initialization.

        Returns
        -------
        FakeEncoderEngine
            Engine exposing the configured context and output dtype.
        """

        assert device.active
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

    def allocate_device(shape: int | tuple[int, ...], dtype: np.dtype) -> FakeArray:
        """Allocate a fake device array while the device scope is active.

        Parameters
        ----------
        shape : int | tuple[int, ...]
            Requested allocation shape.
        dtype : np.dtype
            Requested NumPy-compatible dtype.

        Returns
        -------
        FakeArray
            NumPy-backed fake device allocation.
        """

        assert device.active
        return FakeArray(np.empty(shape, dtype=dtype))

    monkeypatch.setattr(encoder_module.cp, "empty", allocate_device)
    monkeypatch.setattr(
        encoder_module.cpx,
        "empty_pinned",
        lambda shape, dtype: np.empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: reported_cpu_count)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 5)

    def make_audio_copy_pool(
        max_workers: int, thread_name_prefix: str
    ) -> SimpleNamespace:
        """Record thread-pool construction without creating worker threads.

        Parameters
        ----------
        max_workers : int
            Number of audio-copy workers requested by the encoder.
        thread_name_prefix : str
            Prefix assigned to worker thread names.

        Returns
        -------
        SimpleNamespace
            Minimal stand-in for the executor retained by the encoder.
        """

        pool_calls.append((max_workers, thread_name_prefix))
        return SimpleNamespace()

    monkeypatch.setattr(encoder_module, "ThreadPoolExecutor", make_audio_copy_pool)

    engine_path = tmp_path / "encoder.trt"
    stream = FakeStream(device)
    encoder = Encoder(
        engine_path,
        sample_rate=16_000,
        device_id=3,
        stream=stream,  # type: ignore[arg-type]
        right_padding_samples=2,
    )
    expected_output_dtype = {
        trt.float32: np.dtype(np.float32),
        trt.float16: np.dtype(np.float16),
        trt.bfloat16: encoder_module.cp.dtype("bfloat16"),
    }[encoder_output_dtype]

    assert (encoder.batch_size, encoder.min_samples, encoder.max_samples) == (2, 4, 8)
    assert (encoder.sample_rate, encoder.right_padding_samples) == (16_000, 2)
    assert encoder.dtypes == {
        "audio": np.dtype(np.float32),
        "audio_lengths": np.dtype(np.int64),
        "encoder_output": expected_output_dtype,
        "encoder_output_lengths": np.dtype(np.int32),
    }
    assert loaded_paths == [engine_path]
    assert encoder.device is device
    assert encoder.stream is stream
    assert encoder.encoder is context
    assert context.profile_calls == [(0, 117)]
    assert encoder.lengths.shape == encoder.lengths_host.shape == (2,)
    assert encoder.lengths.dtype == encoder.lengths_host.dtype == np.dtype(np.int64)
    assert encoder.output_lengths.shape == (2,)
    assert encoder.output_lengths.dtype == np.dtype(np.int32)
    assert encoder.aux_streams == aux_streams
    assert [aux_stream.ptr for aux_stream in aux_streams] == [101, 102]
    assert pool_calls == [(expected_workers, "fast-gpu-asr-audio")]
    assert encoder.context_memory is encoder.audio is encoder.audio_host is None
    assert encoder.encoder_output is encoder.cuda_graph is None
    assert encoder.cuda_graph_shape is None
    assert encoder.cuda_graph_supported
    assert not encoder.host_transfer_pending
    assert not device.active


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


def test_encoder_supports_zero_right_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.right_padding_samples = 0

    encoder_output, output_lengths = encoder(
        [np.array((1.0, 2.0, 3.0), dtype=np.float32)]
    )

    assert isinstance(encoder.audio, FakeArray)
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 4),
        np.array(((1.0, 2.0, 3.0, 0.0), (0.0, 0.0, 0.0, 0.0)), dtype=np.float32),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 4])
    assert encoder.encoder.audio_shape == (2, 4)
    assert encoder.cuda_graph_shape == (2, 4)
    assert encoder_output.shape == (1, 2, 4)
    np.testing.assert_array_equal(output_lengths.values, [2])


def test_encoder_prepares_partial_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.aux_streams = [FakeScope(1), FakeScope(2)]  # type: ignore[list-item]
    waveform = np.arange(6, dtype=np.float32)[::2]

    encoder_output, output_lengths = encoder([waveform])

    assert isinstance(encoder.audio, FakeArray)
    assert isinstance(encoder.encoder_output, FakeArray)
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 6),
        np.array(
            ((0.0, 2.0, 4.0, 4.0, 2.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 4])
    assert encoder_output.shape == (1, 3, 4)
    assert encoder_output.dtype == np.dtype(np.float32)
    assert encoder_output.data.ptr == encoder.encoder_output.data.ptr
    np.testing.assert_array_equal(output_lengths.values, [2])
    assert dict(encoder.encoder.tensor_addresses) == {
        "audio": encoder.audio.data.ptr,
        "audio_lengths": encoder.lengths.data.ptr,
        "encoder_output": encoder.encoder_output.data.ptr,
        "encoder_output_lengths": encoder.output_lengths.data.ptr,
    }
    assert encoder.encoder.device_memory == (9064, 64)
    assert encoder.encoder.aux_stream_ptrs == [101, 102]
    assert encoder.host_transfer_event.records == 1
    assert encoder.host_transfer_pending
    assert encoder.stream.synchronizations == 0


def test_encoder_clears_inactive_rows_when_reusing_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder(
        [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([7.0, 8.0], dtype=np.float32),
        ]
    )
    encoder([np.array([4.0, 5.0, 6.0], dtype=np.float32)])

    assert isinstance(encoder.audio, FakeArray)
    np.testing.assert_array_equal(
        encoder.audio.values.reshape(2, 6)[1], np.zeros(6, dtype=np.float32)
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 4])


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


def test_encoder_stages_audio_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.audio_copy_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 1)
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 2)
    copied_ranges: list[tuple[int, int]] = []
    copy_audio_range = encoder.copy_audio_range

    def record_copy(
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        """Record each worker range before staging its audio.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Prepared waveforms for the actual input batch.
        audio_host : np.typing.NDArray[np.float32]
            Pinned host staging buffer.
        start : int
            Inclusive first waveform index handled by the worker.
        end : int
            Exclusive last waveform index handled by the worker.
        """

        copied_ranges.append((start, end))
        copy_audio_range(audios, audio_host, start, end)

    monkeypatch.setattr(encoder, "copy_audio_range", record_copy)

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
            ((1.0, 2.0, 3.0, 3.0, 2.0, 0.0), (4.0, 5.0, 5.0, 4.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(encoder.lengths.values, [3, 2])
    assert encoder_output.shape == (2, 3, 4)
    np.testing.assert_array_equal(output_lengths.values, [2, 1])
    assert sorted(copied_ranges) == [(0, 1), (1, 2)]
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
    assert encoder.stream.synchronizations == 1

    encoder([waveform])
    assert encoder.encoder.execute_calls == 2
    assert graph.launches == 2
    assert encoder.stream.synchronizations == 1

    encoder([np.ones(6, dtype=np.float32)])
    assert encoder.encoder.execute_calls == 3
    assert encoder.cuda_graph is None
    assert encoder.cuda_graph_shape == (2, 8)
    assert graph.launches == 2
    assert trim_devices == [3]


def test_encoder_synchronizes_before_clearing_graph_for_smaller_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(
        monkeypatch, trim_devices, encoder_type=GraphClearTrackingEncoder
    )
    large_waveform = np.ones(6, dtype=np.float32)

    encoder([large_waveform])
    encoder([large_waveform])
    assert isinstance(encoder.cuda_graph, FakeGraph)
    synchronizations = encoder.stream.synchronizations

    encoder([np.ones(3, dtype=np.float32)])

    assert encoder.stream.synchronizations == synchronizations + 1
    assert encoder.graph_clear_synchronizations == synchronizations + 1
    assert encoder.cuda_graph is None
    assert trim_devices == []


def test_encoder_synchronizes_before_replacing_live_device_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder([np.ones(3, dtype=np.float32)])
    synchronizations = encoder.stream.synchronizations

    allocation_synchronizations: list[int] = []
    original_empty = encoder_module.cp.empty

    def record_allocation(size: int, dtype: np.dtype) -> FakeArray:
        """Record synchronization state before each device allocation.

        Parameters
        ----------
        size : int
            Number of elements requested by the encoder.
        dtype : np.dtype
            Dtype requested for the allocation.

        Returns
        -------
        FakeArray
            Fake device allocation returned by the original stub.
        """

        allocation_synchronizations.append(encoder.stream.synchronizations)
        return original_empty(size, dtype)

    monkeypatch.setattr(encoder_module.cp, "empty", record_allocation)

    encoder([np.ones(6, dtype=np.float32)])

    assert len(allocation_synchronizations) == 2
    assert synchronizations < allocation_synchronizations[0]
    assert allocation_synchronizations[0] < allocation_synchronizations[1]


def test_encoder_reuses_device_and_host_buffer_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder([np.ones(6, dtype=np.float32)])
    audio = encoder.audio
    audio_host = encoder.audio_host
    encoder_output = encoder.encoder_output
    context_memory = encoder.context_memory
    synchronizations = encoder.stream.synchronizations

    encoder([np.ones(3, dtype=np.float32)])

    assert encoder.audio is audio
    assert encoder.audio_host is audio_host
    assert encoder.encoder_output is encoder_output
    assert encoder.context_memory is context_memory
    assert encoder.stream.synchronizations == synchronizations


def test_encoder_synchronizes_before_growing_context_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    waveform = np.ones(3, dtype=np.float32)
    encoder([waveform])
    original_memory = encoder.context_memory
    synchronizations = encoder.stream.synchronizations
    memory_allocation_synchronizations: list[int] = []
    allocate_memory = encoder_module.cp.cuda.Memory

    def record_memory_allocation(size: int) -> SimpleNamespace:
        """Record synchronization state before context-memory allocation.

        Parameters
        ----------
        size : int
            Requested TensorRT context-memory size in bytes.

        Returns
        -------
        SimpleNamespace
            Fake CUDA allocation returned by the original stub.
        """

        memory_allocation_synchronizations.append(encoder.stream.synchronizations)
        return allocate_memory(size)

    encoder.cuda_graph_supported = False
    encoder.encoder.required_device_memory_size = 96
    monkeypatch.setattr(encoder_module.cp.cuda, "Memory", record_memory_allocation)
    encoder([waveform])

    assert original_memory is not None
    assert len(memory_allocation_synchronizations) == 1
    assert memory_allocation_synchronizations[0] > synchronizations
    assert encoder.context_memory is not original_memory
    assert encoder.context_memory_size == 96
    assert encoder.encoder.device_memory == (encoder.context_memory.ptr, 96)


def test_encoder_handles_zero_sized_context_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.encoder.required_device_memory_size = 0
    monkeypatch.setattr(
        encoder_module.cp.cuda,
        "Memory",
        lambda _: pytest.fail("zero-sized context memory must not be allocated"),
    )

    encoder([np.ones(3, dtype=np.float32)])

    assert encoder.context_memory is None
    assert encoder.context_memory_size == 0
    assert encoder.encoder.device_memory is None


def test_encoder_synchronizes_enqueued_dma_after_parallel_copy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.audio_copy_pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 1)
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 2)

    transfer_started = Event()
    failure = RuntimeError("host copy failed")
    original_set = FakeArray.set
    original_copy = encoder.copy_audio_range

    def recording_set(
        array: FakeArray, source: np.typing.NDArray[np.generic], stream: FakeScope
    ) -> None:
        """Signal when the emulated host-to-device transfer is enqueued.

        Parameters
        ----------
        array : FakeArray
            Destination fake device allocation.
        source : np.typing.NDArray[np.generic]
            Host values copied into the allocation.
        stream : FakeScope
            Active stream associated with the transfer.
        """

        original_set(array, source, stream=stream)
        transfer_started.set()

    def copy_or_fail(
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        """Fail the second worker after the first worker enqueues its transfer.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Prepared waveforms for the actual input batch.
        audio_host : np.typing.NDArray[np.float32]
            Pinned host staging buffer.
        start : int
            Inclusive first waveform index handled by the worker.
        end : int
            Exclusive last waveform index handled by the worker.

        Raises
        ------
        RuntimeError
            Raised for the worker beginning at index one.
        """

        if start == 1:
            assert transfer_started.wait(timeout=5)
            raise failure
        original_copy(audios, audio_host, start, end)

    monkeypatch.setattr(FakeArray, "set", recording_set)
    monkeypatch.setattr(encoder, "copy_audio_range", copy_or_fail)
    try:
        with pytest.raises(RuntimeError) as error:
            encoder([np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)])
    finally:
        encoder.audio_copy_pool.shutdown(wait=True)

    assert error.value is failure
    assert transfer_started.is_set()
    assert encoder.stream.synchronizations == 1
    assert not encoder.host_transfer_pending


@pytest.mark.parametrize(
    ("failed_ndim", "expected_synchronizations"),
    (pytest.param(2, 0, id="audio-transfer"), pytest.param(1, 1, id="length-transfer")),
)
def test_encoder_cleans_up_after_serial_transfer_failure(
    monkeypatch: pytest.MonkeyPatch, failed_ndim: int, expected_synchronizations: int
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    failure = RuntimeError("transfer failed")
    original_set = FakeArray.set

    def set_or_fail(
        array: FakeArray, source: np.typing.NDArray[np.generic], stream: FakeScope
    ) -> None:
        """Fail the transfer whose host source has the selected rank.

        Parameters
        ----------
        array : FakeArray
            Destination fake device allocation.
        source : np.typing.NDArray[np.generic]
            Host array whose rank selects whether the transfer fails.
        stream : FakeScope
            Active stream associated with the transfer.

        Raises
        ------
        RuntimeError
            Raised when ``source.ndim`` equals the parametrized failing rank.
        """

        if source.ndim == failed_ndim:
            raise failure
        original_set(array, source, stream=stream)

    monkeypatch.setattr(FakeArray, "set", set_or_fail)

    with pytest.raises(RuntimeError) as error:
        encoder([np.ones(3, dtype=np.float32)])

    assert error.value is failure
    assert encoder.stream.synchronizations == expected_synchronizations
    assert encoder.host_transfer_event.records == 0
    assert not encoder.host_transfer_pending


@pytest.mark.parametrize("second_worker_fails", (False, True))
def test_encoder_preserves_early_parallel_copy_failure_while_draining_workers(
    monkeypatch: pytest.MonkeyPatch, second_worker_fails: bool
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    encoder.audio_copy_pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(encoder_module, "AUDIO_SAMPLES_PER_WORKER", 1)
    monkeypatch.setattr(encoder_module, "cpu_count", lambda: 2)
    # Already-completed futures have no guaranteed as_completed() ordering.
    monkeypatch.setattr(encoder_module, "as_completed", iter)

    first_failure = RuntimeError("first host copy failed")
    worker_starts: list[int] = []
    original_copy = encoder.copy_audio_range

    def copy_or_fail(
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        """Raise worker failures while recording that every task was drained.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Prepared waveforms for the actual input batch.
        audio_host : np.typing.NDArray[np.float32]
            Pinned host staging buffer.
        start : int
            Inclusive first waveform index handled by the worker.
        end : int
            Exclusive last waveform index handled by the worker.

        Raises
        ------
        RuntimeError
            Raised for the first worker and optionally for the second worker.
        """

        worker_starts.append(start)
        if start == 0:
            raise first_failure
        if second_worker_fails:
            raise RuntimeError("second host copy failed")
        original_copy(audios, audio_host, start, end)

    monkeypatch.setattr(
        FakeArray,
        "set",
        lambda *_, **__: pytest.fail("DMA enqueued after a worker failed"),
    )
    monkeypatch.setattr(encoder, "copy_audio_range", copy_or_fail)
    try:
        with pytest.raises(RuntimeError) as error:
            encoder([np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)])
    finally:
        encoder.audio_copy_pool.shutdown(wait=True)

    assert error.value is first_failure
    assert worker_starts == [0, 1]
    assert encoder.stream.synchronizations == 0
    assert encoder.host_transfer_event.records == 0
    assert not encoder.host_transfer_pending


@pytest.mark.parametrize("failure", ("capture", "execution"))
def test_encoder_disables_cuda_graph_after_capture_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    if failure == "capture":
        encoder.stream = FailingCaptureStream(encoder.device)  # type: ignore[assignment]
        encoder.encoder.stream = encoder.stream
    else:
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

    encoder([waveform])
    assert encoder.encoder.execute_calls == 4
    assert encoder.stream.capture_starts == 1


def test_encoder_propagates_non_invalidation_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    stream = FailingCaptureStream(encoder.device, status=17)
    encoder.stream = stream  # type: ignore[assignment]
    encoder.encoder.stream = stream
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    with pytest.raises(FakeCaptureError) as error:
        encoder([waveform])

    assert error.value is stream.error
    assert encoder.encoder.execute_calls == 2
    assert encoder.cuda_graph_supported
    assert encoder.cuda_graph is None
    assert trim_devices == []


def test_encoder_reports_trim_failure_after_capture_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices, trim_status=17)
    encoder.stream = FailingCaptureStream(encoder.device)  # type: ignore[assignment]
    encoder.encoder.stream = encoder.stream
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    with (
        pytest.warns(RuntimeWarning, match="CUDA graph capture failed"),
        pytest.raises(ASRInferenceError, match="device 3: error 17"),
    ):
        encoder([waveform])

    assert encoder.encoder.execute_calls == 2
    assert not encoder.cuda_graph_supported
    assert trim_devices == [3]


def test_encoder_reports_fallback_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_devices: list[int] = []
    encoder = make_runtime_encoder(monkeypatch, trim_devices)
    encoder.encoder.execute_results = [True, False, False]
    waveform = np.ones(3, dtype=np.float32)

    encoder([waveform])
    with (
        pytest.warns(RuntimeWarning, match="CUDA graph capture failed"),
        pytest.raises(ASRInferenceError, match="TensorRT encoder execution failed"),
    ):
        encoder([waveform])

    assert encoder.encoder.execute_calls == 3
    assert not encoder.cuda_graph_supported
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
    synchronizations = encoder.stream.synchronizations

    with pytest.raises(ASRInferenceError, match="device 3: error 17"):
        encoder([np.ones(6, dtype=np.float32)])

    assert trim_devices == [3]
    assert encoder.stream.synchronizations == synchronizations + 1
    assert encoder.cuda_graph is None
    assert encoder.cuda_graph_shape == (2, 6)
    assert not encoder.host_transfer_pending


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("shape", "rejected encoder input shape"),
        ("address", "rejected encoder tensor address for encoder_output"),
        ("execution", "TensorRT encoder execution failed"),
    ),
)
def test_encoder_reports_tensorrt_runtime_rejection(
    monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    if failure == "shape":
        encoder.encoder.accept_input_shape = False
    elif failure == "execution":
        encoder.encoder.execute_results = [False]
    else:
        encoder.encoder.rejected_address = "encoder_output"

    with pytest.raises(ASRInferenceError, match=message):
        encoder([np.ones(3, dtype=np.float32)])


@pytest.mark.parametrize(
    ("audios", "message"),
    (
        ([], "Expected 1 to 2 audio waveforms"),
        ([np.ones(1, dtype=np.float32)] * 3, "Expected 1 to 2 audio waveforms"),
        ([np.empty(0, dtype=np.float32)], "non-empty one-dimensional"),
        ([np.ones((1, 2), dtype=np.float32)], "non-empty one-dimensional"),
        ([np.ones(9, dtype=np.float32)], "2.000-second TensorRT profile"),
    ),
)
def test_encoder_rejects_invalid_audio_before_cuda(
    audios: list[np.typing.NDArray[np.float32]], message: str
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
    device = FakeScope(3)
    fake_context = FakeExecutionContext() if context_available else None
    if fake_context is not None:
        fake_context.accept_profile = False
    monkeypatch.setattr(encoder_module.cp.cuda, "Device", lambda _: device)
    monkeypatch.setattr(
        encoder_module, "get_engine", lambda _: FakeEncoderEngine(fake_context)
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
        Encoder(
            tmp_path / "encoder.trt",
            16000,
            3,
            SimpleNamespace(ptr=17),  # type: ignore[arg-type]
            0,
        )

    assert not device.active


def test_encoder_waits_before_reusing_pinned_host_staging_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = make_runtime_encoder(monkeypatch)
    events: list[str] = []
    original_synchronize = encoder.host_transfer_event.synchronize
    original_copy = encoder.copy_audio_range

    def record_synchronize() -> None:
        """Record the event wait before delegating to the fake event."""

        events.append("synchronize")
        original_synchronize()

    def record_copy(
        audios: list[np.typing.NDArray[np.float32]],
        audio_host: np.typing.NDArray[np.float32],
        start: int,
        end: int,
    ) -> None:
        """Record host-buffer reuse before delegating to audio staging.

        Parameters
        ----------
        audios : list[np.typing.NDArray[np.float32]]
            Prepared waveforms for the actual input batch.
        audio_host : np.typing.NDArray[np.float32]
            Pinned host staging buffer being reused.
        start : int
            Inclusive first waveform index copied by this call.
        end : int
            Exclusive last waveform index copied by this call.
        """

        events.append("copy")
        original_copy(audios, audio_host, start, end)

    monkeypatch.setattr(encoder.host_transfer_event, "synchronize", record_synchronize)
    monkeypatch.setattr(encoder, "copy_audio_range", record_copy)
    encoder.encoder.accept_input_shape = False

    with pytest.raises(ASRInferenceError, match="rejected encoder input shape"):
        encoder([np.ones(3, dtype=np.float32)])

    assert encoder.host_transfer_pending
    events.clear()
    encoder.encoder.accept_input_shape = True

    encoder([np.ones(3, dtype=np.float32)])

    assert events == ["synchronize", "copy"]
