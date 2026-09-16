from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from tessera.compiler.resident_gated_pool import ResidentGatedPool, _GatedBinding
from tessera.compiler.native_isolated_heap import IsolatedHeapPool
from tessera.compiler.heap_barrier_contract import ATOMIC_MODES
from tessera.compiler.gpu_heap_collection import materialize_pool
from tessera.compiler.scheduled_matmul import find_tessera_opt


@pytest.mark.parametrize('mode', ATOMIC_MODES)
def test_every_admitted_metadata_operation_has_atomic_replay(mode):
    tool = find_tessera_opt()
    if tool is None or not Path('/usr/lib/llvm-23/bin/mlir-opt').exists():
        pytest.skip('native compiler required')
    program = materialize_pool(4, 8, 'atomic_' + mode, compiler=tool,
        llvm_bin='/usr/lib/llvm-23/bin', backend='nvidia', chip='sm_120', payload_dtype='int8', references=1)
    assert program.validate()[-2].name == 'gate'


def test_gated_binding_injects_owned_gate_not_a_caller_gate():
    calls = []
    p = SimpleNamespace(_gate=object())
    binding = _GatedBinding(p, SimpleNamespace(submit=lambda *args: calls.append(args)))
    binding.submit(123, 'state', 'status', 1)
    assert calls == [(123, 'state', 'status', p._gate, 1)]


def test_gated_owner_refuses_live_metadata_and_legacy_import():
    p = object.__new__(ResidentGatedPool)
    for fn in (lambda: p.read(1), lambda: p.snapshot(1), lambda: p.from_objects({})):
        with pytest.raises(ValueError):
            fn()


def parent():
    p = object.__new__(IsolatedHeapPool)
    p._lock = threading.RLock()
    p.closed = p.failed = False
    p._pending = p._recovery = None
    p.width, p.timeout = 8, 30
    p._dimensions, p._options = (4, 8, 1), dict(backend='nvidia', chip='sm_120')
    p.channel = SimpleNamespace(send=lambda value: pytest.fail('invalid request reached worker'))
    return p


@pytest.mark.parametrize('value', [np.zeros(7, np.int8), np.zeros(8, np.float32), np.zeros((2,4), np.int8)])
def test_invalid_heap_payload_never_poisons_worker(value):
    p = parent()
    with pytest.raises(ValueError, match='exact int8'):
        p.submit('allocate', value)
    assert not p.failed and p._pending is None


def test_recovery_does_not_close_channel_before_death():
    p = parent()
    calls = []
    p.channel = SimpleNamespace(close=lambda: calls.append('closed'))
    p._recovery = SimpleNamespace(poll=lambda: False)
    p.lease = SimpleNamespace(reusable=False)
    assert not p.poll_recovery() and not calls
    p._recovery.poll = lambda: True
    with pytest.raises(RuntimeError, match='unconfirmed'):
        p.poll_recovery()
    assert not calls and not p.closed
    p.lease.reusable = True
    assert p.poll_recovery() and p.closed and calls == ['closed']


def test_worker_slot_is_released_once_only_after_death():
    p = parent()
    releases = []
    p._slot_owned = True
    p._slot = SimpleNamespace(release=lambda: releases.append(True))
    p.channel = SimpleNamespace(close=lambda: None)
    p._recovery = SimpleNamespace(poll=lambda: False)
    p.lease = SimpleNamespace(reusable=False)
    assert not p.poll_recovery() and not releases
    p._recovery.poll = lambda: True
    p.lease.reusable = True
    assert p.poll_recovery() and p.poll_recovery()
    assert releases == [True]


def test_replacement_requires_confirmed_death_then_reprobes(monkeypatch):
    from tessera.compiler import native_isolated_heap as module
    p = parent()
    p.lease = SimpleNamespace(reusable=False)
    for closed, failed, reusable in [(False, False, False), (True, False, True), (False, True, True), (True, True, False)]:
        p.closed, p.failed, p.lease.reusable = closed, failed, reusable
        with pytest.raises(ValueError, match='confirmed'):
            p.replacement()
    p.closed = p.failed = p.lease.reusable = True
    admitted = []
    monkeypatch.setattr(module.IsolatedHeapPool, '__init__',
                        lambda self, *args, **kwargs: admitted.append((args, kwargs)))
    fresh = p.replacement()
    # Same dimensions, options and timeout; a brand-new worker, never this one.
    assert fresh is not p and admitted == [((4, 8, 1), dict(timeout_seconds=30, backend='nvidia', chip='sm_120'))]


def test_startup_without_verified_health_probe_never_admits_an_owner(monkeypatch):
    from tessera.compiler import native_isolated_heap as module
    from tessera.compiler import gpu_heap_collection
    events = []
    class Lease:
        def __init__(self, process, *, context_identity, timeout_seconds):
            events.append(('lease', context_identity))
        def mark_uncertain(self):
            events.append('uncertain')
        def recover(self):
            events.append('recover')
    class Process:
        pid = 4242
        def __init__(self, *, target, args, daemon):
            events.append(('spawn', args[1]))
        def start(self):
            events.append('start')
    class End:
        def __init__(self, ready):
            self.ready = ready
        def poll(self, timeout=None):
            return True
        def recv(self):
            return self.ready
        def close(self):
            events.append('closed')
    monkeypatch.setattr(gpu_heap_collection, 'emit_pool', lambda *a, **k: None)
    monkeypatch.setattr(module, 'DriverIsolationLease', Lease)
    monkeypatch.setattr(module, '_ProcessBoundary', lambda process: process)
    before = module._WORKER_SLOTS._value
    for ready in [('ready',), ('ready', 'heap-health-v0'), ('error', 'RuntimeError', 'health probe: readback')]:
        events.clear()
        ctx = SimpleNamespace(Pipe=lambda: (End(ready), End(ready)), Process=Process)
        monkeypatch.setattr(module.mp, 'get_context', lambda kind: ctx)
        with pytest.raises(RuntimeError, match='health probe not verified'):
            IsolatedHeapPool(4, 8, 1, backend='nvidia')
        assert 'uncertain' in events and 'recover' in events and events.count('closed') == 2
        assert module._WORKER_SLOTS._value == before and not module._UNCERTAIN


def _probe_pool(state_rows, *, readback=None):
    """A scripted pool for the in-worker probe: inspect returns the given rows."""
    calls = []
    import ctypes as ct
    class Ticket:
        def wait(self):
            calls.append('wait')
    class Request:
        def poll(self):
            return True
        def __enter__(self):
            return SimpleNamespace(__cuda_array_interface__={'data': (0, False)})
        def __exit__(self, *exc):
            return False
    rows = iter(state_rows)
    def download(host, device, nbytes):
        if readback is not None:
            ct.memmove(host, readback.ctypes.data, nbytes)
        return 0
    pool = SimpleNamespace(
        check=lambda rc: calls.append(('check', rc)), _upload=lambda *a: 0, _download=download,
        _status_values=lambda ticket: np.array([0, 0, 1]), allocate=lambda *a: Ticket(),
        inspect_metadata=lambda stream: (np.array(next(rows), np.int64),),
        prepare_readers=lambda n: calls.append(('readers', n)), begin_read_object=lambda *a: Request(),
        poll_object_readers=lambda stream: True, set_graph=lambda *a: Ticket(), begin_mark=lambda s: None,
        mark_step=lambda *a: Ticket(), finish_mark_async=lambda s: SimpleNamespace(poll=lambda: True),
        reclaim_retired=lambda s: Ticket())
    return pool, calls


def test_health_probe_refuses_unverified_metadata_readback_and_reclamation():
    from tessera.compiler.native_isolated_heap import _probe_health, HEALTH_PROBE
    buffers = tuple(SimpleNamespace(pointer=None) for _ in range(3))
    pattern = (np.arange(8, dtype=np.int64) % 7 - 3).astype(np.int8)
    live = [[1, 8, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    reclaimed = [[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    pool, calls = _probe_pool([live, reclaimed], readback=pattern)
    assert _probe_health(pool, 1, (4, 8, 1), buffers) == HEALTH_PROBE
    assert ('readers', 1) in calls
    # No live slot after allocation.
    pool, _ = _probe_pool([[[0, 0, 0]] * 4], readback=pattern)
    with pytest.raises(RuntimeError, match='metadata did not verify'):
        _probe_health(pool, 1, (4, 8, 1), buffers)
    # Wrong bytes come back through the pin.
    pool, _ = _probe_pool([live], readback=np.zeros(8, np.int8))
    with pytest.raises(RuntimeError, match='readback did not verify'):
        _probe_health(pool, 1, (4, 8, 1), buffers)
    # The slot survives an empty-graph mark and reclaim.
    pool, _ = _probe_pool([live, live], readback=pattern)
    with pytest.raises(RuntimeError, match='reclamation did not verify'):
        _probe_health(pool, 1, (4, 8, 1), buffers)

