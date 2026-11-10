"""Context-scoped close of prepared NVIDIA owners, with the driver faked."""
import os
import warnings

import pytest

from tessera.compiler import prepared_nvidia_matmul as prepared


class _Lib:
    # Plain functions, like ctypes symbols, so callers may set argtypes.
    def __init__(self):
        self.closed, self.forgotten = [], []

        def close(handle):
            self.closed.append(handle)
            return 0

        def forget(handle):
            self.forgotten.append(handle)
            return 0

        self.tessera_nvidia_matmul_close = close
        self.tessera_nvidia_matmul_forget = forget


@pytest.fixture
def live(monkeypatch):
    contexts = set()
    monkeypatch.setattr(prepared, "_live_context_identities", lambda: set(contexts))
    monkeypatch.setattr(prepared, "_deferred", {})
    monkeypatch.setattr(prepared, "_forgotten_handles", [])
    monkeypatch.setattr(prepared, "_exiting", False)
    return contexts


def test_forked_child_never_probes_the_driver(monkeypatch):
    def probe():
        raise AssertionError("a forked child must not call the CUDA driver")
    monkeypatch.setattr(prepared, "_live_context_identities", probe)
    lib = _Lib()
    assert prepared.close_in_live_context(lib, 7, 3, os.getpid() + 1) == 0
    assert lib.closed == lib.forgotten == []


def test_live_context_closes(live):
    lib = _Lib()
    live.add(3)
    assert prepared.close_in_live_context(lib, 7, 3, os.getpid()) == 0
    assert lib.closed == [7] and not prepared.deferred_handles()


def test_noncurrent_context_is_deferred_until_it_is_live(live):
    lib = _Lib()
    with pytest.warns(ResourceWarning, match="not provably live"):
        prepared.close_in_live_context(lib, 7, 3, os.getpid())
    assert prepared.deferred_handles() == ((7, 3),)
    assert lib.closed == lib.forgotten == []
    prepared.close_deferred()
    assert prepared.deferred_handles() == ((7, 3),)
    live.add(3)
    prepared.close_deferred()
    assert lib.closed == [7] and not prepared.deferred_handles()
    assert lib.forgotten == [] and not prepared.forgotten_handles()


def test_a_live_close_settles_deferred_owners(live):
    lib = _Lib()
    with pytest.warns(ResourceWarning):
        prepared.close_in_live_context(lib, 7, 3, os.getpid())
    live.update({3, 4})
    prepared.close_in_live_context(lib, 8, 4, os.getpid())
    assert sorted(lib.closed) == [7, 8] and not prepared.deferred_handles()


def test_exit_forgets_what_is_still_deferred(live):
    lib = _Lib()
    with pytest.warns(ResourceWarning):
        prepared.close_in_live_context(lib, 7, 3, os.getpid())
    prepared._close_deferred_at_exit()
    assert lib.forgotten == [7] and lib.closed == []
    assert prepared.forgotten_handles() == ((7, 3),)
    # A finalizer that runs after the exit hook forgets directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        prepared.close_in_live_context(lib, 9, 5, os.getpid())
    assert lib.forgotten == [7, 9] and not prepared.deferred_handles()


def test_settle_skips_owners_of_another_process(live):
    lib = _Lib()
    prepared._deferred[7] = (lib, 3, os.getpid() + 1)
    live.add(3)
    prepared.close_deferred(final=True)
    assert lib.closed == lib.forgotten == [] and prepared.deferred_handles() == ((7, 3),)
