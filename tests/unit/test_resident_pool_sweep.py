from contextlib import contextmanager
from types import SimpleNamespace
import threading
import pytest
from tessera.compiler.resident_object_pool import ResidentObjectPool


def pool():
    p = object.__new__(ResidentObjectPool)
    p._lock = threading.RLock()
    p._dimensions = (4, 8, 1)
    p._sweep_cursor = 0
    p._ready = lambda: None
    p._collection = SimpleNamespace(ticket=SimpleNamespace(wait_on=lambda s: None))
    p._snapshot = (None,) * 5
    p._state = p._roots = p._edges = p._marks = p._status = None
    calls = []
    def submit(*args):
        calls.append(args[-3:-1])
        return SimpleNamespace(ticket='ticket')
    p._seeded = SimpleNamespace(submit=submit)
    @contextmanager
    def write(stream):
        yield
    p._access = lambda: SimpleNamespace(write=write)
    return p, calls


def test_sweep_batches_advance_without_finishing_collection_early():
    p, calls = pool()
    assert p.finish_collection(21, sweep_budget=2) == 'ticket'
    assert p._collection is not None
    assert p._sweep_cursor == 2
    p.finish_collection(21, sweep_budget=2)
    assert p._collection is None
    assert calls == [(0, 2), (2, 4)]


def test_failed_batch_does_not_advance_cursor():
    p, _ = pool()
    def fail(*args):
        raise RuntimeError('failed')
    p._seeded.submit = fail
    with pytest.raises(RuntimeError):
        p.finish_collection(21, sweep_budget=1)
    assert p._sweep_cursor == 0 and p._collection is not None
    assert p._collection_uncertain


@pytest.mark.parametrize('budget', [True, 0, -1, 5])
def test_invalid_sweep_budget_refused(budget):
    p, _ = pool()
    with pytest.raises(ValueError, match='budget'):
        p.finish_collection(21, sweep_budget=budget)
