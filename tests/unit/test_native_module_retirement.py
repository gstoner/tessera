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
