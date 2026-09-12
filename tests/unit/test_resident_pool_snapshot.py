import ctypes as ct
from types import SimpleNamespace
import pytest
from test_native_reader_retirement import setup
from tessera.compiler.native_stream_epoch import NativeStreamEpoch
from tessera.compiler.resident_pool_snapshot import ResidentPoolSnapshot


def pool_setup(monkeypatch):
    owner, native = setup()
    parent = owner.frame
    parent.native, parent.closed, parent._snapshots = native, False, []
    ready = parent._ready
    parent._ready = lambda **kwargs: ready()
    parent.alloc = lambda *args: 0
    parent.free = lambda pointer: native.calls.append(('free', pointer.value)) or 0
    parent._copy_async = lambda *args: native.calls.append(('copy', args[-1].value)) or 0
    parent.epoch = NativeStreamEpoch(parent, native, owner._reader_buffers, owner._submission)
    parent._access = lambda: parent.epoch
    for source in owner._reader_buffers:
        source.shape, source.dtype, source.nbytes = (4,), 'fp32', 16
    class Buffer:
        def __init__(self, frame, shape, dtype):
            self.shape, self.dtype = shape, dtype
            self.pointer = ct.c_void_p(100 + len(frame.buffers))
            self.nbytes = 16
            frame.buffers.append(self)
    monkeypatch.setattr('tessera.compiler.resident_pool_snapshot._Buffer', Buffer)
    native._sync = lambda: native.calls.append(('sync',)) or 0
    return parent, native


def test_snapshot_reader_does_not_exclude_live_epoch_mutation(monkeypatch):
    pool, native = pool_setup(monkeypatch)
    snap = ResidentPoolSnapshot(pool, 21)
    with snap.read(22):
        with pool.epoch.write(23):
            native.calls.append(('sweep',))
        with pytest.raises(ValueError, match='active readers'):
            snap.close()
        assert not any(c[0] == 'free' for c in native.calls)
    snap.close()
    assert snap.closed and not pool._snapshots
    assert sum(c[0] == 'free' for c in native.calls) == 2


def test_uncertain_copy_keeps_snapshot_allocations_owned(monkeypatch):
    pool, native = pool_setup(monkeypatch)
    pool._copy_async = lambda *args: 1
    with pytest.raises(RuntimeError, match='injected'):
        ResidentPoolSnapshot(pool, 21)
    assert len(pool._snapshots) == 1
    assert not any(c[0] == 'free' for c in native.calls)
    pool._snapshots[0].close()
    assert ('sync',) in native.calls
    assert not pool._snapshots


def test_live_reader_refusal_allocates_no_snapshot_storage(monkeypatch):
    pool, _ = pool_setup(monkeypatch)
    with pool.epoch.read(21):
        with pytest.raises(ValueError, match='closed live reader'):
            ResidentPoolSnapshot(pool, 22)
    assert not pool._snapshots


@pytest.mark.parametrize('published', [True, False])
def test_snapshot_cleanup_can_recover_poisoned_parent(monkeypatch, published):
    pool, native = pool_setup(monkeypatch)
    snap = ResidentPoolSnapshot(pool, 21)
    if not published:
        snap.epoch = None
    def ready(*, recovery=False):
        if not recovery:
            raise RuntimeError('poisoned parent')
    pool._ready = ready
    with pytest.raises(RuntimeError, match='poisoned'):
        snap.read(22)
    snap.close()
    assert snap.closed and not snap.buffers and not snap._closing
    assert not pool._snapshots
    assert sum(c[0] == 'free' for c in native.calls) == 2


def test_recovery_close_still_refuses_active_snapshot_reader(monkeypatch):
    pool, native = pool_setup(monkeypatch)
    snap = ResidentPoolSnapshot(pool, 21)
    with snap.read(22):
        def ready(*, recovery=False):
            if not recovery:
                raise RuntimeError('poisoned parent')
        pool._ready = ready
        with pytest.raises(ValueError, match='active readers'):
            snap.close()
        assert not snap._closing and not snap.closed
        assert not any(c[0] == 'free' for c in native.calls)
    snap.close()
    assert snap.closed


def test_recovery_completion_failure_retains_snapshot(monkeypatch):
    pool, native = pool_setup(monkeypatch)
    snap = ResidentPoolSnapshot(pool, 21)
    snap.epoch = None
    native._sync = lambda: 1
    with pytest.raises(RuntimeError):
        snap.close()
    assert snap in pool._snapshots and snap.buffers and not snap._closing
    assert not any(c[0] == 'free' for c in native.calls)
