"""Completion is not success; checked generations retain independent status."""
import ctypes as ct
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler.native_persistent_tape import CheckedDerivativeSubmission


def fixture(status_value=0):
    state = SimpleNamespace(ready=False, checks=0, value=status_value, frees=[])
    output = SimpleNamespace(pointer=ct.c_void_p(10))
    status = SimpleNamespace(pointer=ct.c_void_p(20))
    def check_status(buffer):
        assert buffer is status
        state.checks += 1
        if state.value:
            raise RuntimeError('persistent GPU product guard failed; outputs are unavailable')
    frame = SimpleNamespace(_lock=threading.RLock(), _ready=lambda: None,
        _check_status=check_status, buffers=[output, status], _submissions=[],
        check=lambda code: None, sync=lambda: 0,
        free=lambda pointer: state.frees.append(pointer.value) or 0)
    ticket = SimpleNamespace(poll=lambda: state.ready)
    submission = SimpleNamespace(ticket=ticket, wait=lambda: setattr(state, 'ready', True))
    result = CheckedDerivativeSubmission(frame, submission, (output,), 21, status)
    frame._submissions.append(result)
    return state, result


def test_outputs_require_completion_and_checked_status():
    state, result = fixture()
    with pytest.raises(ValueError, match='successful wait or poll'):
        _ = result.outputs
    assert not result.poll() and state.checks == 0
    state.ready = True
    assert result.poll() and state.checks == 1
    assert len(result.outputs) == 1
    result.wait()
    assert state.checks == 1


def test_failed_generation_never_exposes_outputs_and_can_be_released():
    state, result = fixture(1)
    with pytest.raises(RuntimeError, match='guard failed'):
        result.wait()
    state.value = 0
    with pytest.raises(RuntimeError, match='guard failed'):
        _ = result.outputs
    with pytest.raises(RuntimeError, match='guard failed'):
        result.poll()
    result.release()
    assert state.frees == [10, 20] and not result.frame.buffers
    with pytest.raises(ValueError, match='released'):
        _ = result.outputs


def test_success_in_one_generation_does_not_clear_another_failure():
    _, failed = fixture(1)
    _, passed = fixture(0)
    passed.wait()
    with pytest.raises(RuntimeError, match='guard failed'):
        failed.wait()
    assert len(passed.outputs) == 1


def test_frame_close_drains_failed_checked_generation():
    from tessera.compiler.native_persistent_tape import PersistentTapeFrame
    state, result = fixture(1)
    with pytest.raises(RuntimeError, match='guard failed'):
        result.wait()
    frame = result.frame
    frame.closed = False
    frame._bindings = []
    frame._release = lambda start: frame.buffers.clear()
    PersistentTapeFrame.close(frame)
    assert frame.closed and not frame.buffers and not frame._submissions
    assert state.checks == 1
