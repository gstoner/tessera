"""Host-side ownership interleavings; no device execution evidence."""
from contextlib import contextmanager
import ctypes as ct
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from tessera.compiler.resident_rocm_attention import ResidentROCmAttentionTape


def _owner(monkeypatch, execute, closed):
    from tessera import runtime as rt
    class HIP:
        def __init__(self):
            self.fail_record = False
            self.hipStreamWaitEvent = lambda *args: 0
            self.hipStreamGetDevice = lambda stream, pointer: self.hipGetDevice(pointer)
        def hipEventCreate(self, pointer):
            ct.cast(pointer, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(41)
            return 0
        def hipEventRecord(self, *args):
            return 77 if self.fail_record else 0
        def hipEventDestroy(self, *args):
            return 0
        def hipGetDevice(self, pointer):
            ct.cast(pointer,ct.POINTER(ct.c_int))[0] = 0
            return 0
        def hipSetDevice(self, device):
            return 0
    execute.stream = ct.c_void_p(11)
    execute.device_outputs = {"dq": (1234, (2,2), "float32")}
    @contextmanager
    def bind(program, buffers):
        try:
            yield execute
        finally:
            closed.set()
    monkeypatch.setattr(rt,"_load_hip_for_launch",lambda: HIP())
    monkeypatch.setattr(rt,"_bind_rocm_attention_backward_program",bind)
    arg=lambda name: SimpleNamespace(name=name,direction="input",layout="row_major")
    program=SimpleNamespace(descriptors=[SimpleNamespace(buffers=[arg("q")]),
                                         SimpleNamespace(buffers=[arg("q"),arg("do")])])
    return ResidentROCmAttentionTape(program,{"q":np.ones((2,2),np.float16),"do":np.ones((2,2),np.float16)})


def test_retirement_waits_for_submission_and_preserves_snapshot(monkeypatch):
    entered, release, closed = threading.Event(),threading.Event(),threading.Event()
    def execute(*,cotangent):
        entered.set()
        assert release.wait(5)
        return cotangent.copy()
    tape=_owner(monkeypatch,execute,closed)
    value=np.ones((2,2),np.float16)
    future=tape.submit(value)
    try:
        assert entered.wait(5)
        value.fill(9)
        retirement=tape.retire()
        assert not closed.is_set() and not retirement.done()
        with pytest.raises(ValueError,match="retiring"):
            tape.submit(value)
    finally:
        release.set()
        tape.close()
    np.testing.assert_array_equal(future.result(),1)
    assert closed.is_set()


def test_failed_submission_poison_blocks_queued_work_but_allows_retirement(monkeypatch):
    entered, release, closed=threading.Event(),threading.Event(),threading.Event()
    calls=[]
    def execute(*,cotangent):
        calls.append(1); entered.set()
        assert release.wait(5)
        raise RuntimeError("injected copyback failure")
    tape=_owner(monkeypatch,execute,closed)
    first=tape.submit(np.ones((2,2),np.float16))
    assert entered.wait(5)
    second=tape.submit(np.ones((2,2),np.float16))
    release.set()
    try:
        with pytest.raises(RuntimeError,match="copyback"):
            first.result()
        with pytest.raises(RuntimeError,match="previous submission"):
            second.result()
        with pytest.raises(ValueError,match="failed"):
            tape.submit(np.ones((2,2),np.float16))
    finally:
        tape.close()
    assert calls == [1] and closed.is_set()


def test_invalid_cotangent_does_not_poison_owner(monkeypatch):
    closed=threading.Event()
    tape=_owner(monkeypatch,lambda **kw:kw["cotangent"],closed)
    with tape:
        with pytest.raises(ValueError,match="shape/storage"):
            tape.submit(np.ones((3,2),np.float16))
        with pytest.raises(ValueError,match="shape/storage"):
            tape.submit(np.ones((2,2),np.float32))
        np.testing.assert_array_equal(tape.submit(np.ones((2,2),np.float16)).result(),1)
    assert closed.is_set()


def test_worker_cannot_deadlock_itself_with_blocking_close(monkeypatch):
    closed=threading.Event()
    def execute(**kwargs):
        with pytest.raises(RuntimeError,match="owning worker"):
            tape.close()
        return kwargs["cotangent"]
    tape=_owner(monkeypatch,execute,closed)
    with tape:
        tape.submit(np.ones((2,2),np.float16)).result(timeout=5)
    assert closed.is_set()


def test_reader_failure_retains_owner_and_retirement_is_not_cancellable(monkeypatch):
    closed=threading.Event()
    tape=_owner(monkeypatch,lambda **kw:kw["cotangent"],closed)
    with pytest.raises(ValueError,match="completed backward"):
        tape.reader(22)
    tape.submit(np.ones((2,2),np.float16)).result()
    reader=tape.reader(22)
    retirement=tape.retire()
    assert not retirement.cancel() and not retirement.done()
    tape._hip.fail_record=True
    with pytest.raises(RuntimeError,match="ownership retained"):
        reader.close()
    assert not closed.is_set() and not retirement.done()
    tape._hip.fail_record=False
    reader.close()
    retirement.result(timeout=5)
    tape.close()
    assert closed.is_set()
    with pytest.raises(ValueError,match="released"):
        _=reader.outputs


def test_all_external_readers_must_release_before_retirement(monkeypatch):
    closed=threading.Event()
    tape=_owner(monkeypatch,lambda **kw:kw["cotangent"],closed)
    tape.submit(np.ones((2,2),np.float16)).result()
    first,second=tape.reader(22),tape.reader(33)
    retirement=tape.retire()
    first.close()
    assert not retirement.done() and not closed.is_set()
    second.close()
    retirement.result(timeout=5)
    assert closed.is_set()


def test_wrong_device_reader_does_not_poison_or_pin_owner(monkeypatch):
    closed=threading.Event()
    tape=_owner(monkeypatch,lambda **kw:kw["cotangent"],closed)
    with tape:
        tape.submit(np.ones((2,2),np.float16)).result()
        def wrong_device(stream,pointer):
            ct.cast(pointer,ct.POINTER(ct.c_int))[0]=1
            return 0
        tape._hip.hipStreamGetDevice=wrong_device
        with pytest.raises(ValueError,match="another device"):
            tape.reader(22)
        assert tape._readers == 0
        tape.submit(np.ones((2,2),np.float16)).result()
    assert closed.is_set()
