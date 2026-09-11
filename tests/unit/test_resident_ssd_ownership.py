from types import SimpleNamespace
import pytest
from test_native_reader_retirement import setup
from tessera.compiler.resident_ssd import ResidentSSDFrame
from tessera.compiler.native_stream_epoch import NativeStreamEpoch
from tessera.control import vjp


def frame_setup():
    generation, native = setup()
    frame = ResidentSSDFrame.__new__(ResidentSSDFrame)
    frame.__dict__.update(generation.frame.__dict__)
    frame._ready = lambda **kwargs: None
    frame.closed = False
    frame._retirement = None
    frame._capture = generation._submission
    frame._capture_inputs = (object(),)
    frame._retirement_poisoned = False
    frame.owner = SimpleNamespace(frames=[frame], _forward=SimpleNamespace(_bound=native))
    generation.frame = frame
    frame._forward_epoch = NativeStreamEpoch(frame, native, (), generation._submission)
    return frame, generation, native


def test_whole_frame_retires_after_forward_and_derivative_readers():
    frame, generation, native = frame_setup()
    with frame.read_forward(21):
        with pytest.raises(ValueError, match="scopes"):
            frame.retire_async(23)
    with generation.read(22):
        pass
    native.ready = False
    frame.retire_async(23)
    assert not frame.poll_close()
    assert not frame.closed
    assert not frame.buffers
    assert not any(c[0] in ("event_wait", "stream_sync") for c in native.calls)
    native.ready = True
    assert frame.poll_close()
    assert not frame.owner.frames
    assert not frame._capture_inputs


def test_eventless_reader_keeps_whole_frame_retryable():
    frame, generation, native = frame_setup()
    native.record_failure = True
    with pytest.raises(RuntimeError):
        with generation.read(22):
            pass
    with pytest.raises(RuntimeError, match="unproven"):
        frame.retire_async(23)
    assert frame._retirement is None and len(frame.buffers) == 2
    native.record_failure = False
    generation.wait()
    frame.retire_async(23)
    assert frame.poll_close()


def test_public_vjp_dispatch_has_no_numpy_conversion():
    class Program:
        def __tessera_vjp__(self, *inputs, stream):
            return inputs, stream

    value = object()
    assert vjp(Program(), value, stream=7) == ((value,), 7)
    with pytest.raises(ValueError, match="owned native"):
        vjp(lambda x: x, value, stream=7)
