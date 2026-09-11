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
