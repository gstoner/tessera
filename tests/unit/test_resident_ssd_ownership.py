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


def program_setup():
    import threading
    from tessera.compiler.resident_ssd import ResidentSSDProgram
    program = object.__new__(ResidentSSDProgram)
    program._lock = threading.RLock()
    program.closed = program._closing = False
    program.frames = []
    calls = []
    def poll(**kwargs):
        assert kwargs == {'defer_unload': True}
        calls.append('module_poll')
        return len(calls) > 2
    program._forward = SimpleNamespace(close_if_complete=poll)
    program._reverse = SimpleNamespace(close_if_complete=poll)
    return program,calls


def test_program_retirement_defers_modules_until_frames_complete():
    program,calls = program_setup()
    frame,generation,native = frame_setup()
    program._forward._bound = native
    frame.owner = program
    program.frames = [frame]
    native.ready = False
    program.retire_async(23)
    with pytest.raises(ValueError, match='retiring'):
        program.capture(object())
    assert not program.poll_close() and not calls
    native.ready = True
    assert not program.poll_close()
    assert len(calls) == 2  # Both unloads start; neither blocks the caller.
    assert program.poll_close() and program.closed
    assert not any(c[0] in ('event_wait','stream_sync') for c in native.calls)


def test_program_preflights_all_active_readers_before_retiring_any_frame():
    program,calls = program_setup()
    first,generation,native = frame_setup()
    second,other,_ = frame_setup()
    program.frames = [first,second]
    with other.read(22):
        with pytest.raises(ValueError, match='scopes'):
            program.retire_async(23)
    assert not program._closing
    assert first._retirement is None and second._retirement is None


def test_program_retry_after_partial_retirement_keeps_capture_closed():
    program,calls = program_setup()
    starts = []
    first = SimpleNamespace(closed=False,_submissions=[],_forward_epoch=None,_retirement=None)
    second = SimpleNamespace(closed=False,_submissions=[],_forward_epoch=None,_retirement=None)
    def retire_first(stream):
        starts.append('first')
        first._retirement = object()
    def retire_second(stream):
        starts.append('second')
        if starts.count('second') == 1:
            raise RuntimeError('missing dependency event')
        second._retirement = object()
    first.retire_async,second.retire_async = retire_first,retire_second
    program.frames = [first,second]
    with pytest.raises(RuntimeError, match='dependency'):
        program.retire_async(23)
    assert program._closing
    program.retire_async(23)
    assert starts == ['first','second','second']


def test_program_failed_unload_retains_owner_and_refuses_sync_close():
    program,calls = program_setup()
    def failed(**kwargs):
        raise RuntimeError('driver unload failed; owner retained')
    program._reverse.close_if_complete = failed
    program.retire_async(23)
    with pytest.raises(RuntimeError, match='owner retained'):
        program.poll_close()
    assert program._closing and not program.closed
    with pytest.raises(RuntimeError, match='owner retained'):
        program.close()
