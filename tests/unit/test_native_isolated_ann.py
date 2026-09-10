"""Host IPC/lifetime proof; these tests make no GPU execution claim."""
from dataclasses import dataclass
from types import SimpleNamespace
import time
import re

import numpy as np
import pytest

from tessera.compiler.native_isolated_ann import IsolatedNativeANN, _UNCERTAIN_WORKERS
from benchmarks.record_native_ann_execution import source


def host_worker(connection, pair, input_bound, absolute_budget):
    from tessera.compiler.native_isolated_ann import _serve
    _serve(connection, pair, input_bound, absolute_budget)


@pytest.fixture(autouse=True)
def host_transport(monkeypatch):
    monkeypatch.setattr('tessera.compiler.native_isolated_ann._ann_worker', host_worker)


@dataclass
class Pair:
    stall: bool = False
    unhealthy: bool = False
    health_stall: bool = False

    @property
    def original(self):
        return SimpleNamespace(binding_digest='original')

    @property
    def transformed(self):
        return SimpleNamespace(binding_digest='transformed')

    @property
    def logical(self):
        return SimpleNamespace(original=re.sub(r'"tessera\.(matmul|add|mul)"\((%\w+),\s*(%\w+)\)', r'tessera.\1 \2, \3', source(activation="square").replace(">,tensor", ">, tensor")))

    def bind(self, **kwargs):
        return Runner(self.stall, self.unhealthy, self.health_stall)


class Runner:
    def __init__(self, stall, unhealthy=False, health_stall=False):
        self.stall = stall
        self.unhealthy = unhealthy
        self.health_stall = health_stall
        self.shape = (3, 2)

    def verify(self, probes):
        if self.health_stall:
            time.sleep(60)
        if self.unhealthy:
            raise ValueError("injected numerical health failure")
        return True

    def run(self, value, **kwargs):
        if self.stall:
            time.sleep(60)
        return value * 2

    def close(self):
        pass


def test_process_transport_returns_private_checked_result_and_joins():
    with IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1) as runner:
        x = np.ones((3, 2), np.float32)
        result = runner.run(x)
        np.testing.assert_array_equal(result, x * 2)
        assert not np.shares_memory(result, x)
    assert runner._process.exitcode == 0


def test_uncertain_request_requires_process_death_before_replacement():
    runner = IsolatedNativeANN(Pair(stall=True), input_bound=1, absolute_budget=.1)
    try:
        runner.submit(np.ones((3, 2), np.float32))
        with pytest.raises(ValueError, match='outstanding'):
            runner.submit(np.ones((3, 2), np.float32))
        runner._pending = (runner._pending[0], 0)
        with pytest.raises(TimeoutError, match='uncertain'):
            runner.poll()
        assert runner in _UNCERTAIN_WORKERS
        with pytest.raises(RuntimeError, match='uncertain'):
            runner.run(np.ones((3, 2), np.float32))
        runner.recover()
        assert runner.lease.reusable and runner._process.exitcode is not None
        assert runner not in _UNCERTAIN_WORKERS
        with IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1) as replacement:
            assert replacement._process.pid != runner._process.pid
            np.testing.assert_array_equal(replacement.run(np.ones((3, 2), np.float32)), 2)
    finally:
        if not runner.closed:
            runner._poison()
            runner.recover()


def test_nonfinite_timeout_refuses_before_spawning():
    for value in (float('nan'), float('inf'), True, 0):
        with pytest.raises(ValueError, match='finite'):
            IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1, timeout_seconds=value)


def test_async_recovery_finalizes_pending_host_ownership():
    runner = IsolatedNativeANN(Pair(stall=True), input_bound=1, absolute_budget=.1)
    try:
        runner.submit(np.ones((3, 2), np.float32))
        runner._pending = (runner._pending[0], 0)
        with pytest.raises(TimeoutError):
            runner.poll()
        ticket = runner.recover_async()
        assert runner.recover_async() is ticket
        assert ticket.done.wait(10)
        assert runner.poll_recovery()
        assert runner.poll_recovery()
        assert runner.closed and runner._pending is None
        assert runner not in _UNCERTAIN_WORKERS
        assert runner._process.exitcode is not None
    finally:
        if not runner.closed:
            runner.recover()


def test_failed_health_probe_never_admits_worker():
    with pytest.raises(RuntimeError, match='health failure'):
        IsolatedNativeANN(Pair(unhealthy=True), input_bound=1, absolute_budget=.1)


def test_replacement_requires_death_and_a_fresh_probe():
    with IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1) as healthy:
        with pytest.raises(ValueError, match='confirmed'):
            healthy.replacement()
    runner = IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1)
    runner._poison()
    with pytest.raises(ValueError, match='confirmed'):
        runner.replacement()
    runner.recover()
    with runner.replacement() as replacement:
        assert replacement._process.pid != runner._process.pid
        np.testing.assert_array_equal(replacement.run(np.ones((3, 2), np.float32)), 2)
    runner.pair.unhealthy = True
    with pytest.raises(RuntimeError, match='health failure'):
        runner.replacement()


def test_hung_health_probe_is_bounded_and_worker_reclaimed():
    before = set(_UNCERTAIN_WORKERS)
    with pytest.raises(TimeoutError, match='startup timed out'):
        IsolatedNativeANN(Pair(health_stall=True), input_bound=1,
                          absolute_budget=.1, timeout_seconds=3)
    assert _UNCERTAIN_WORKERS == before


def test_invalid_inputs_do_not_poison_healthy_worker():
    with IsolatedNativeANN(Pair(), input_bound=1, absolute_budget=.1) as runner:
        pid = runner._process.pid
        for value in (np.zeros((2, 3), np.float32), np.full((3, 2), np.nan, np.float32),
                      np.full((3, 2), np.inf, np.float32), np.full((3, 2), 2, np.float32)):
            with pytest.raises(ValueError, match='shape or domain'):
                runner.submit(value)
            assert not runner.failed and runner._pending is None and runner._next == 0
        np.testing.assert_array_equal(runner.run(np.ones((3, 2), np.float32)), 2)
        assert runner._process.pid == pid


@pytest.mark.parametrize('raise_in_body', [False, True])
def test_context_exit_reclaims_pending_worker(raise_in_body):
    error = LookupError('caller failure')
    runner = IsolatedNativeANN(Pair(stall=True), input_bound=1, absolute_budget=.1)
    try:
        with runner:
            runner.submit(np.ones((3, 2), np.float32))
            if raise_in_body:
                raise error
    except LookupError as caught:
        assert raise_in_body and caught is error
    assert runner.closed and runner._process.exitcode is not None
    assert runner._pending is None and runner not in _UNCERTAIN_WORKERS


def test_context_exit_preserves_body_error_when_recovery_is_uncertain(monkeypatch):
    runner = IsolatedNativeANN(Pair(stall=True), input_bound=1, absolute_budget=.1)
    recover = runner.recover
    error = LookupError('original caller error')
    def fail_recovery():
        raise OSError('unconfirmed process death')
    monkeypatch.setattr(runner, 'recover', fail_recovery)
    try:
        with pytest.raises(LookupError) as caught:
            with runner:
                runner.submit(np.ones((3, 2), np.float32))
                raise error
        assert caught.value is error
        assert runner in _UNCERTAIN_WORKERS
        assert 'cleanup incomplete' in error.__notes__[0]
    finally:
        recover()
