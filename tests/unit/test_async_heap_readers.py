import pytest
from test_incremental_object_readers import pool as make_pool
from tessera.compiler.native_reader_retirement import _record
from tessera.compiler.heap_writer_model import explore_writers


class Receipt:
    device = None
    def __init__(self, p, native):
        self.pool, self.native = p, native
        self.ticket = None
        self.values = (0, 8, 1)
        self.copies = 0
    def enqueue_copy(self, stream):
        self.copies += 1
        self.ticket = _record(self.native, stream, (self.pool, self), [])
        if self.copies == 2:
            self.native.ready = False
    def poll(self):
        return self.values if self.ticket.poll() else None


def setup():
    p, native, calls = make_pool()
    receipt = Receipt(p, native)
    p._heap_receipts = [receipt]
    p._free_receipts = [receipt]
    def forbidden(*args):
        raise AssertionError('asynchronous reader used synchronous status path')
    p._status_values = forbidden
    return p, native, calls, receipt


def test_private_receipt_admission_and_unpin_never_wait():
    p, native, calls, receipt = setup()
    native.ready = False
    reader = p.begin_read_object(21, 0, 1)
    assert not reader.poll()
    with pytest.raises(ValueError, match='pending'):
        reader.__enter__()
    native.ready = True
    assert reader.poll()
    with reader as view:
        assert view.__cuda_array_interface__['shape'] == (8,)
    assert not p.poll_object_readers(22)  # unpin copy is pending
    assert reader in p._object_readers and receipt not in p._free_receipts
    native.ready = True
    assert p.poll_object_readers(22)
    assert receipt in p._free_receipts and calls.count('unpin') == 1
    assert not any(c[0] in ('event_wait', 'stream_sync') for c in native.calls)


def test_failed_admission_recycles_receipt_without_poisoning():
    p, _, _, receipt = setup()
    reader = p.begin_read_object(21, 0, 1)
    receipt.values = (2, 0, 0)
    with pytest.raises(ValueError, match='stale'):
        reader.poll()
    assert not p._object_readers and p._free_receipts == [receipt] and not p._object_uncertain


def test_cancelled_pending_admission_still_unpins_once():
    p, native, calls, _ = setup()
    native.ready = False
    reader = p.begin_read_object(21, 0, 1)
    reader.cancel()
    assert 'unpin' not in calls
    native.ready = True
    assert not p.poll_object_readers(22)
    native.ready = True
    assert p.poll_object_readers(22)
    assert calls.count('unpin') == 1
    with pytest.raises(ValueError, match='cancelled'):
        reader.__enter__()


def test_racing_writer_model_requires_atomic_reservation():
    assert explore_writers()['counterexample'] is None
    assert explore_writers(split_reservation=True)['counterexample'] is not None


def test_async_unpin_presubmission_refusal_is_retryable():
    p, native, calls, _ = setup()
    reader = p.begin_read_object(21, 0, 1)
    with reader:
        pass
    with p.read(23):
        with pytest.raises(ValueError, match='scopes'):
            p.poll_object_readers(22)
        assert not reader._unpin_started and not p._object_uncertain
        assert 'unpin' not in calls
    assert not p.poll_object_readers(22)
    native.ready = True
    assert p.poll_object_readers(22)
    assert calls.count('unpin') == 1


def test_uncertain_async_unpin_copy_retains_receipt_and_pin_owner():
    from tessera.compiler.resident_object_pool import _UNCERTAIN_POOLS
    p, _, calls, receipt = setup()
    reader = p.begin_read_object(21, 0, 1)
    with reader:
        pass
    def fail(stream):
        raise RuntimeError('uncertain status copy')
    receipt.enqueue_copy = fail
    try:
        with pytest.raises(RuntimeError, match='uncertain status copy'):
            p.poll_object_readers(22)
        assert calls.count('unpin') == 1
        assert p._object_uncertain and p in _UNCERTAIN_POOLS
        assert reader in p._object_readers and receipt not in p._free_receipts
    finally:
        _UNCERTAIN_POOLS.remove(p)
