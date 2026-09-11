import ctypes as ct
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from test_native_reader_retirement import setup
from tessera.compiler.native_stream_epoch import NativeStreamEpoch
from tessera.compiler.resident_incremental_pool import ResidentIncrementalPool
from tessera.compiler.resident_object_pool import _UNCERTAIN_POOLS


def pool():
    owner, native = setup()
    p = object.__new__(ResidentIncrementalPool)
    p._lock = threading.RLock()
    p._ready = lambda **kw: None
    p._object_readers = []
    p._object_uncertain = False
    p._mark_active = False
    p._dimensions = (4, 8, 1)
    p._state = p._roots = p._edges = p._marks = p._status = p._pins = None
    p._payload = SimpleNamespace(pointer=ct.c_void_p(100))
    p.native = native
    p.epoch = NativeStreamEpoch(p, native, owner._reader_buffers, owner._submission)
    calls = []
    def binding(mode):
        def submit(*args):
            calls.append(mode)
            return SimpleNamespace(ticket=owner._submission.ticket)
        return SimpleNamespace(submit=submit)
    p._binding = binding
    p._status_values = lambda ticket: np.array([0, 8, 1])
    return p, native, calls


def test_reader_borrows_only_its_object_while_metadata_can_retire():
    p, _, calls = pool()
    with p.read_object(21, 3, 1) as view:
        spec = view.__cuda_array_interface__
        assert spec['data'] == (124, True) and spec['shape'] == (8,)
        p._mark_active = True
        p.finish_mark(22)
        p.reclaim_retired(22)
        assert 'retire_marked' in calls and 'reclaim_pinned' in calls
        assert 'unpin' not in calls
    with pytest.raises(ValueError, match='lease is closed'):
        _ = view.__cuda_array_interface__


def test_pending_reader_does_not_block_metadata_but_cannot_unpin():
    p, native, calls = pool()
    with p.read_object(21, 0, 1):
        pass
    native.ready = False
    p.reclaim_retired(22)
    assert 'unpin' not in calls and len(p._object_readers) == 1
    native.ready = True
    p.reclaim_retired(22)
    assert calls[-2:] == ['unpin', 'reclaim_pinned'] and not p._object_readers


def test_failed_reader_record_remains_pinned_until_explicit_completion():
    p, native, calls = pool()
    reader = p.read_object(21, 0, 1)
    reader.__enter__()
    native.record_failure = True
    with pytest.raises(RuntimeError, match='injected'):
        reader.__exit__(None, None, None)
    native.record_failure = False
    p.reclaim_retired(22)
    assert 'unpin' not in calls
    reader.completion.wait()
    p.reclaim_retired(22)
    assert 'unpin' in calls


def test_uncertain_unpin_quarantines_without_a_second_decrement():
    p, _, _ = pool()
    with p.read_object(21, 0, 1):
        pass
    def fail(*args):
        raise RuntimeError('uncertain decrement')
    p._binding = lambda mode: SimpleNamespace(submit=fail)
    try:
        with pytest.raises(RuntimeError, match='uncertain'):
            p.reclaim_retired(22)
        assert p._object_uncertain and p in _UNCERTAIN_POOLS and p._object_readers
    finally:
        _UNCERTAIN_POOLS.remove(p)


def test_stale_admission_does_not_poison_the_pool():
    p, _, _ = pool()
    p._status_values = lambda ticket: np.array([2, 0, 0])
    with pytest.raises(ValueError, match='stale'):
        with p.read_object(21, 0, 1):
            pass
    assert not p._object_readers and not p._object_uncertain


def test_metadata_scope_refusal_before_unpin_submission_is_retryable():
    p, _, calls = pool()
    with p.read_object(21, 0, 1):
        pass
    with p.read(23):
        with pytest.raises(ValueError, match='scopes'):
            p.reclaim_retired(22)
        assert not p._object_uncertain and 'unpin' not in calls
    p.reclaim_retired(22)
    assert not p._object_readers
