from tessera.compiler.native_driver_isolation import DriverIsolationLease


class Process:
    def __init__(self): self.code=None; self.calls=[]
    def poll(self): return self.code
    def terminate(self): self.calls.append('terminate'); self.code=-15
    def wait(self, timeout=None): self.calls.append('wait'); return self.code


def test_uncertain_driver_context_is_reclaimed_by_process_death_once():
    process=Process(); lease=DriverIsolationLease(process,context_identity='worker-1')
    lease.mark_uncertain(); lease.recover(); lease.recover()
    assert process.calls == ['terminate','wait'] and lease.reusable


def test_healthy_isolation_cannot_be_destroyed_as_recovery():
    lease=DriverIsolationLease(Process(),context_identity='worker-1')
    try: lease.recover()
    except ValueError as error: assert 'uncertain' in str(error)
    else: raise AssertionError('healthy isolation destroyed')


def test_stalled_isolation_is_killed_after_bounded_terminate():
    import subprocess
    class Stalled(Process):
        def wait(self, timeout=None):
            self.calls.append(('wait',timeout))
            if len(self.calls)==2: raise subprocess.TimeoutExpired('worker',timeout)
            return self.code
        def terminate(self): self.calls.append('terminate')
        def kill(self): self.calls.append('kill'); self.code=-9
    process=Stalled(); lease=DriverIsolationLease(process,context_identity='worker',timeout_seconds=.25)
    lease.mark_uncertain(); lease.recover()
    assert process.calls == ['terminate',('wait',.25),'kill',('wait',.25)]


def test_nonfinite_timeouts_are_not_bounded():
    import pytest
    for value in (float('nan'), float('inf'), True, 0):
        with pytest.raises(ValueError, match='finite'):
            DriverIsolationLease(Process(), context_identity='worker', timeout_seconds=value)


def test_async_recovery_retains_owner_and_bounds_admission(monkeypatch):
    import threading
    from tessera.compiler import native_driver_isolation as module
    entered, release = threading.Event(), threading.Event()
    class Waiting(Process):
        def terminate(self):
            entered.set()
            release.wait(5)
            super().terminate()
    monkeypatch.setattr(module, '_RECOVERY_SLOTS', threading.BoundedSemaphore(1))
    lease = DriverIsolationLease(Waiting(), context_identity='worker')
    lease.mark_uncertain()
    owner = object()
    ticket = module.IsolationRecovery.submit(lease, owner=owner)
    try:
        assert entered.wait(5)
        assert not ticket.poll() and ticket.owner is owner
        assert module.IsolationRecovery.submit(lease, owner=owner) is None
    finally:
        release.set()
    assert ticket.done.wait(5)
    assert ticket.poll() and ticket.owner is None and lease.reusable


def test_async_unconfirmed_death_quarantines_owner(monkeypatch):
    import threading
    import pytest
    from tessera.compiler import native_driver_isolation as module
    class Unkillable(Process):
        def terminate(self):
            raise OSError('termination outcome unknown')
    monkeypatch.setattr(module, '_RECOVERY_SLOTS', threading.BoundedSemaphore(1))
    monkeypatch.setattr(module, '_RECOVERIES', set())
    lease = DriverIsolationLease(Unkillable(), context_identity='worker')
    lease.mark_uncertain()
    owner = object()
    ticket = module.IsolationRecovery.submit(lease, owner=owner)
    assert ticket.done.wait(5)
    with pytest.raises(RuntimeError, match='owner retained'):
        ticket.poll()
    assert ticket.owner is owner and ticket in module._RECOVERIES
    assert not lease.reusable
    assert module.IsolationRecovery.submit(lease, owner=owner) is None
