import threading
from types import SimpleNamespace
import pytest
from test_async_heap_readers import setup
from tessera.compiler.heap_finalization import MarkFinalization, PoolTeardown


def test_finalization_keeps_mark_active_until_private_status_completes():
    p, native, _, receipt = setup()
    p._mark_active = True
    ticket = p.finish_mark_async(21)
    native.ready = False
    assert not ticket.poll() and p._mark_active
    native.ready = True
    assert ticket.poll() and not p._mark_active
    assert p._pending_finalization is None and receipt in p._free_receipts


def test_incomplete_finalization_can_be_retried():
    p, _, _, receipt = setup()
    p._mark_active = True
    ticket = p.finish_mark_async(21)
    receipt.values = (2, 0, 0)
    with pytest.raises(ValueError, match='incomplete'):
        ticket.poll()
    assert p._mark_active and p._pending_finalization is None


def test_teardown_poll_does_not_wait_for_driver_and_retains_owner():
    entered, release = threading.Event(), threading.Event()
    calls = []
    p = SimpleNamespace(native=SimpleNamespace(_enter_unload_context=lambda: calls.append('enter'),
                                               _leave_unload_context=lambda: calls.append('leave')))
    def close():
        entered.set()
        assert release.wait(5)
        calls.append('close')
    p.close = close
    p._poison_objects = lambda: calls.append('poison')
    ticket = PoolTeardown.submit(p)
    assert entered.wait(5)
    try:
        assert not ticket.poll() and ticket.pool is p
    finally:
        release.set()
    assert ticket.done.wait(5) and ticket.poll()
    assert calls == ['enter', 'close', 'leave']


def test_same_gate_prevents_publication_retirement_race():
    from tessera.compiler.heap_writer_model import explore_retirement
    assert explore_retirement()['counterexample'] is None
    assert explore_retirement(omit_gate=True)['counterexample'] is not None


def test_failed_teardown_stays_retained_and_is_not_retried():
    from tessera.compiler import heap_finalization as module
    calls = []
    def fail():
        calls.append('close')
        raise RuntimeError('uncertain free')
    p = SimpleNamespace(native=SimpleNamespace(_enter_unload_context=lambda: None,
                                               _leave_unload_context=lambda: None),
                        close=fail, _poison_objects=lambda: calls.append('poison'))
    ticket = PoolTeardown.submit(p)
    assert ticket.done.wait(5)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='retained'):
                ticket.poll()
        assert ticket in module._LIVE and calls == ['close', 'poison']
    finally:
        # The fake driver has no retained resources; undo only this test's slot.
        module._LIVE.remove(ticket)
        module._SLOTS.release()
