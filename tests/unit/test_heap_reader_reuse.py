"""The actual resident owner must order reuse, not merely the protocol model."""
import threading
from types import SimpleNamespace
import pytest
from test_native_reader_retirement import setup
from tessera.compiler.native_stream_epoch import NativeStreamEpoch
from tessera.compiler.resident_object_pool import ResidentObjectPool


def pool():
    owner, native = setup()
    p = object.__new__(ResidentObjectPool)
    p._lock = threading.RLock()
    p._ready = lambda **kw: None
    p._collection = None
    p._state = p._roots = p._edges = p._marks = p._status = None
    p.epoch = NativeStreamEpoch(p, native, owner._reader_buffers, owner._submission)
    def submit(*args):
        native.calls.append(('kernel', args[0]))
        return SimpleNamespace(ticket='submitted')
    p._retire = p._reclaim = SimpleNamespace(submit=submit)
    return p, native


def test_reuse_refuses_open_reader_and_orders_closed_reader_before_kernel():
    p, native = pool()
    with p.read(21):
        with pytest.raises(ValueError, match='scopes'):
            p.reclaim_retired(22)
        assert not any(c[0] == 'kernel' for c in native.calls)
    assert p.reclaim_retired(22) == 'submitted'
    assert native.calls.index(('stream_wait', 22)) < native.calls.index(('kernel', 22))
    assert not any(c[0] in ('stream_sync', 'event_wait') for c in native.calls)


def test_failed_reader_record_does_not_commit_reuse_and_explicit_completion_retries():
    p, native = pool()
    native.record_failure = True
    with pytest.raises(RuntimeError, match='injected'):
        with p.read(21):
            pass
    with pytest.raises(RuntimeError, match='unproven'):
        p.reclaim_retired(22)
    assert not p.epoch.retiring and not any(c[0] == 'kernel' for c in native.calls)
    native.record_failure = False
    p.epoch.wait()
    assert p.reclaim_retired(22) == 'submitted'


def test_failed_retire_enqueue_keeps_completion_dependency():
    p, native = pool()
    def fail(*args):
        raise RuntimeError('enqueue failed after partial submission')
    p._retire = SimpleNamespace(submit=fail)
    with pytest.raises(RuntimeError, match='enqueue'):
        p.retire_unreachable(21)
    p.reclaim_retired(22)
    assert ('stream_wait', 22) in native.calls
