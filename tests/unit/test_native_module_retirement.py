"""A blocked driver unload cannot block polling or admit unbounded workers."""
import threading
from types import SimpleNamespace
import pytest
from tessera.compiler import native_module_retirement as retirement
from tests.unit.test_native_gpu_streams import binding


def test_unload_is_off_thread_and_binding_cannot_relaunch(monkeypatch):
    native, _ = binding()
    entered, release = threading.Event(), threading.Event()
    native._enter_unload_context = lambda: None
    native._leave_unload_context = lambda: None
    native._directory = SimpleNamespace(cleanup=lambda: None)
    def unload(module):
        entered.set()
        assert release.wait(5)
        return 0
    native._unload = unload
    try:
        assert not native.close_if_complete(defer_unload=True)
        assert entered.wait(2)
        assert not native.close_if_complete(defer_unload=True)
        with pytest.raises(ValueError, match='retiring'):
            native.submit((1,), grid=(1,1,1), block=(1,1,1), stream=7)
    finally:
        release.set()
        assert native._module_retirement.done.wait(2)
    assert native.close_if_complete(defer_unload=True)
    assert not native._module.value


def test_full_admission_retains_module_without_spawning(monkeypatch):
    native, _ = binding()
    monkeypatch.setattr(retirement, '_SLOTS', threading.BoundedSemaphore(0))
    assert not native.close_if_complete(defer_unload=True)
    assert native._module.value == 1
    assert native._module_retirement is None


def test_unload_failure_retains_owner_and_does_not_retry(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(retirement, '_SLOTS', slots)
    native, calls = binding()
    native._enter_unload_context = lambda: None
    native._leave_unload_context = lambda: None
    native._unload = lambda module: calls.append('unload') or 1
    native._directory = SimpleNamespace(cleanup=lambda: pytest.fail('failed module cleaned'))
    assert not native.close_if_complete(defer_unload=True)
    assert native._module_retirement.done.wait(2)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='owner retained'):
            native.close_if_complete(defer_unload=True)
    assert calls == ['unload'] and native._module.value == 1
    assert not slots.acquire(blocking=False)
    retirement._LIVE.discard(native._module_retirement)


def test_uncertain_unload_can_recover_only_by_destroying_isolated_context(monkeypatch):
    from tessera.compiler.native_driver_isolation import DriverIsolationLease
    class Process:
        code=None
        def poll(self): return self.code
        def terminate(self): self.code=-15
        def wait(self, timeout=None): return self.code
    native,_=binding()
    native._isolation_lease=DriverIsolationLease(Process(),context_identity='worker')
    native._unload=lambda module: 1
    native._directory=SimpleNamespace(cleanup=lambda:None)
    with pytest.raises(RuntimeError):native.close_if_complete()
    ticket=native._module_retirement
    assert ticket.recover_isolation() and ticket.isolation.reusable
    assert not native._module.value and ticket not in retirement._LIVE


def test_post_unload_cleanup_retry_never_repeats_driver_action(monkeypatch):
    slots=threading.BoundedSemaphore(1)
    monkeypatch.setattr(retirement,'_SLOTS',slots)
    native,calls=binding()
    native._enter_unload_context=lambda:None
    native._leave_unload_context=lambda:None
    native._unload=lambda module:calls.append('unload') or 0
    attempts=[]
    def cleanup():
        attempts.append(1)
        if len(attempts)==1:raise OSError('temporary cleanup error')
    native._directory=SimpleNamespace(cleanup=cleanup)
    assert not native.close_if_complete(defer_unload=True)
    assert native._module_retirement.done.wait(2)
    with pytest.raises(RuntimeError):native.close_if_complete(defer_unload=True)
    native.retry_retirement_cleanup()
    assert native._module_retirement.done.wait(2)
    assert native.close_if_complete(defer_unload=True)
    assert calls==['unload'] and len(attempts)==2
    assert slots.acquire(blocking=False)
    slots.release()


def test_driver_failure_cannot_be_retried_as_cleanup(monkeypatch):
    monkeypatch.setattr(retirement,'_SLOTS',threading.BoundedSemaphore(1))
    native,calls=binding()
    native._enter_unload_context=lambda:None
    native._leave_unload_context=lambda:None
    native._unload=lambda module:calls.append('unload') or 1
    assert not native.close_if_complete(defer_unload=True)
    ticket=native._module_retirement
    assert ticket.done.wait(2)
    with pytest.raises(ValueError,match='post-unload'):native.retry_retirement_cleanup()
    assert calls==['unload']
    retirement._LIVE.discard(ticket)


def test_context_exit_failure_is_not_filesystem_recovery(monkeypatch):
    monkeypatch.setattr(retirement,'_SLOTS',threading.BoundedSemaphore(1))
    native,calls=binding()
    native._enter_unload_context=lambda:None
    def failed_exit():raise RuntimeError('context exit failed')
    native._leave_unload_context=failed_exit
    native._unload=lambda module:calls.append('unload') or 0
    native._directory=SimpleNamespace(cleanup=lambda:pytest.fail('cleanup before context exit'))
    assert not native.close_if_complete(defer_unload=True)
    ticket=native._module_retirement
    assert ticket.done.wait(2)
    with pytest.raises(ValueError,match='post-unload'):native.retry_retirement_cleanup()
    assert calls==['unload']
    retirement._LIVE.discard(ticket)


def test_unload_and_context_exit_failure_preserve_both_causes(monkeypatch):
    monkeypatch.setattr(retirement,'_SLOTS',threading.BoundedSemaphore(1))
    native,calls=binding()
    unload_error=RuntimeError('uncertain unload')
    exit_error=RuntimeError('uncertain context exit')
    native._enter_unload_context=lambda:None
    def unload(module):
        calls.append('unload')
        raise unload_error
    def leave():raise exit_error
    native._unload=unload
    native._leave_unload_context=leave
    native._directory=SimpleNamespace(cleanup=lambda:pytest.fail('uncertain owner cleaned'))
    assert not native.close_if_complete(defer_unload=True)
    ticket=native._module_retirement
    try:
        assert ticket.done.wait(2)
        with pytest.raises(RuntimeError,match='driver_unload; context exit also failed') as caught:ticket.poll()
        assert caught.value.__cause__ is unload_error
        assert ticket.context_exit_error is exit_error
        with pytest.raises(ValueError,match='post-unload'):ticket.retry_cleanup()
        assert calls==['unload'] and ticket in retirement._LIVE
    finally:retirement._LIVE.discard(ticket)


@pytest.mark.parametrize('synchronize',[False,True])
def test_synchronous_unknown_unload_never_retries_or_relaunches(synchronize):
    native,calls=binding()
    native._sync=lambda:0
    failure=RuntimeError('unknown unload outcome')
    def unload(module):
        calls.append('unload')
        raise failure
    native._unload=unload
    native._directory=SimpleNamespace(cleanup=lambda:pytest.fail('unknown module cleaned'))
    operation=native.close if synchronize else native.close_if_complete
    try:
        with pytest.raises(RuntimeError,match='unknown unload'):operation()
        for _ in range(2):
            with pytest.raises(RuntimeError,match='owner retained'):native.close()
        with pytest.raises(ValueError,match='retiring'):
            native.submit((1,),grid=(1,1,1),block=(1,1,1),stream=7)
        assert calls==['unload']
        assert native._module_retirement in retirement._LIVE
        assert native._module_retirement.error is failure
    finally:retirement._LIVE.discard(native._module_retirement)


def test_async_isolation_cleanup_waits_for_process_death():
    from tessera.compiler.native_driver_isolation import DriverIsolationLease
    entered, release = threading.Event(), threading.Event()
    class Process:
        code = None
        def poll(self): return self.code
        def terminate(self):
            entered.set()
            release.wait(5)
            self.code = -15
        def wait(self, timeout=None): return self.code
    calls = []
    owner = SimpleNamespace(_module=42,
        _directory=SimpleNamespace(cleanup=lambda: calls.append('cleanup')),
        _isolation_lease=DriverIsolationLease(Process(), context_identity='module-worker'))
    ticket = retirement.ModuleRetirement.retain_failure(owner, RuntimeError('driver'), 'driver_unload')
    recovery = ticket.recover_isolation_async()
    try:
        assert entered.wait(5)
        assert not ticket.poll_isolation_recovery()
        assert not calls and owner._module == 42
    finally:
        release.set()
    assert recovery.done.wait(5)
    assert ticket.poll_isolation_recovery()
    assert ticket.poll_isolation_recovery()
    assert calls == ['cleanup'] and owner._module == 0
    assert ticket not in retirement._LIVE
