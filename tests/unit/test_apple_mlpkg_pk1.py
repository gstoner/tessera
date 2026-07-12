"""PK1 — `.mtlpackage` load + compile foundation.

Tests the C ABI surface ``tessera_apple_gpu_mlpkg_{compile,destroy,
is_compiled,last_error_kind}`` and the Python wrapper
``tessera.apple_mlpkg`` for:

* **Symbol resolution** — all 4 PK1 symbols bind cleanly.
* **Skip semantics** — non-existent path returns ``None`` with
  ``last_error_kind() == ERROR_LIBRARY_LOAD_FAILED``; non-Darwin /
  pre-macOS-26 hosts return ``None`` with
  ``ERROR_OS_UNAVAILABLE``.
* **Lifecycle** — destroy after successful load + compile works;
  destroy on NULL handle is safe; double-destroy is safe.
* **Real artifact** — when ``tests/fixtures/apple_gpu/*.mtlpackage``
  exists (developer dropped one in), compile succeeds + ``is_compiled``
  returns True. Otherwise tests skip with a clear reason.

PK1 deliberately does NOT exercise execution — `Pipeline.dispatch()`
raises ``NotImplementedError`` until PK4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.apple_mlpkg import (
    ERROR_LIBRARY_LOAD_FAILED,
    ERROR_NONE,
    ERROR_OS_UNAVAILABLE,
    Pipeline,
    compile_mlpackage,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
)
from tessera._apple_gpu_dispatch import apple_gpu_runtime


_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "apple_gpu"


def _find_mtlpackage() -> Path | None:
    """Locate a ``.mtlpackage`` test fixture if one is present.

    Developers can drop the Apple-sample ``matrix-multiplication.mtlpackage``
    (or any other device_verified_jit Metal package) into ``tests/fixtures/apple_gpu/``
    to exercise the real-artifact path. Empty / missing → tests skip
    that path cleanly.
    """
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


# ---- Symbol resolution -------------------------------------------------

def test_all_pk1_symbols_resolve():
    """The 4 C ABI entry points must bind cleanly when the runtime
    loads. A missing symbol indicates the runtime dylib didn't pick up
    PK1's additions (stale cache / build config drift)."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    from tessera.apple_mlpkg import (
        _bind_compile, _bind_destroy, _bind_is_compiled,
        _bind_last_error_kind,
    )
    assert _bind_compile() is not None, "tessera_apple_gpu_mlpkg_compile missing"
    assert _bind_destroy() is not None, "tessera_apple_gpu_mlpkg_destroy missing"
    assert _bind_is_compiled() is not None, "tessera_apple_gpu_mlpkg_is_compiled missing"
    assert _bind_last_error_kind() is not None, (
        "tessera_apple_gpu_mlpkg_last_error_kind missing")


# ---- Skip semantics ----------------------------------------------------

def test_compile_returns_none_for_nonexistent_path():
    """A path that doesn't resolve to a Metal package must NOT raise —
    it must return None with the precise ``LIBRARY_LOAD_FAILED`` enum.
    Critical for callers that probe whether a package is available.
    Skip when packaged ML isn't available — on macOS<26 the runtime
    returns ``ERROR_OS_UNAVAILABLE`` regardless of path (P1 fix)."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    result = compile_mlpackage("/nonexistent/path/no.mtlpackage")
    assert result is None
    assert last_error_kind() == ERROR_LIBRARY_LOAD_FAILED


def test_last_error_kind_clears_after_read():
    """The error code is one-shot — reading it returns the code AND
    resets to NONE. Catches a regression where stale errors persist
    across calls."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    compile_mlpackage("/nonexistent/path/no.mtlpackage")
    # First read: the error
    err = last_error_kind()
    assert err == ERROR_LIBRARY_LOAD_FAILED
    # Second read: cleared
    assert last_error_kind() == ERROR_NONE


def test_compile_returns_none_when_runtime_unavailable(monkeypatch):
    """If the runtime fails to build / load, ``compile_mlpackage``
    must return ``None`` without raising and without trying to call
    into the C ABI."""
    import tessera.apple_mlpkg as mod
    monkeypatch.setattr(mod, "apple_gpu_runtime", lambda: None)
    assert mod.compile_mlpackage("/anything") is None


# ---- Lifecycle ---------------------------------------------------------

def test_destroy_on_null_handle_is_safe():
    """``Pipeline(handle=0, ...).destroy()`` must be a no-op. Catches
    a regression where the destroy bridge passes ``CFBridgingRelease``
    a NULL pointer."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    assert pipe.is_compiled is False
    pipe.destroy()  # must not raise
    assert pipe.is_compiled is False


def test_double_destroy_is_safe():
    """Calling destroy twice on the same Pipeline must not crash or
    double-free."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    pipe.destroy()
    pipe.destroy()  # no-op the second time


def test_context_manager_destroys_on_exit():
    """The Pipeline is a context manager that releases the C handle
    on ``__exit__``. Verify by exercising the with-block lifecycle on
    a hand-constructed empty handle (handle=0)."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pipe as p:
        assert p is pipe
        assert p.is_compiled is False
    # After exit, internal handle cleared.
    assert pipe._handle == 0


def test_dispatch_on_destroyed_handle_raises():
    """PK4 wired the real dispatch path (this test was once asserting
    ``NotImplementedError`` pre-PK4). Now ``dispatch()`` on a
    destroyed Pipeline must raise the SAME lifecycle error as the
    other PK3+ methods — ``RuntimeError("...already destroyed")`` —
    not silently fail. Catches a regression where the destroyed-handle
    guard is missed in the new dispatch path."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    with pytest.raises(RuntimeError, match="already destroyed"):
        pipe.dispatch()


# ---- Real-artifact path (skipped when no fixture available) ------------

def test_compile_loads_real_metal_package():
    """When a ``.mtlpackage`` fixture is present under
    ``tests/fixtures/apple_gpu/``, compile must succeed and
    ``is_compiled`` must return True. The bundled Apple
    ``matrix-multiplication.mtlpackage`` has dynamic-shape inputs,
    so we pass concrete dims via PK1.5's ``input_dimensions=`` (Apple
    sample picks 4x4 for the demo)."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(
            f"No .mtlpackage fixture in {_FIXTURES_DIR} — drop one in to "
            f"exercise the real-artifact path. Apple's sample "
            f"`matrix-multiplication.mtlpackage` is a known-good source.")
    # Bundled package: inputA at buffer 0 = (K, M); inputB at buffer 1 = (N, K).
    M = N = K = 4
    pipe = compile_mlpackage(
        pkg, function_name="main",
        input_dimensions={0: (K, M), 1: (N, K)})
    if pipe is None:
        err = last_error_kind()
        pytest.fail(
            f"compile_mlpackage failed for {pkg}; "
            f"last_error_kind={err}. Common causes: macOS < 26, "
            f"the package was device_verified_jit against a different OS, or the "
            f"function 'main' isn't the entry point.")
    try:
        assert pipe.is_compiled is True
        # No error after successful compile.
        assert last_error_kind() == ERROR_NONE
        # Metadata round-trips on the Python side.
        assert pipe.package_path == str(pkg)
        assert pipe.function_name == "main"
        assert "device_verified_jit" in repr(pipe)
    finally:
        pipe.destroy()


def test_destroyed_pipeline_reports_destroyed_state():
    """After explicit destroy(), is_compiled flips to False and repr()
    reflects the state. Catches a regression where the Python side
    caches the device_verified_jit state."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip("no .mtlpackage fixture; see "
                    "test_compile_loads_real_metal_package")
    M = N = K = 4
    pipe = compile_mlpackage(
        pkg, function_name="main",
        input_dimensions={0: (K, M), 1: (N, K)})
    if pipe is None:
        pytest.skip(f"compile_mlpackage failed; last_error_kind="
                    f"{last_error_kind()}")
    assert pipe.is_compiled is True
    pipe.destroy()
    assert pipe.is_compiled is False
    assert "destroyed" in repr(pipe)


# ---- Stub parity (the off-Darwin path) ---------------------------------

def test_helpers_return_clear_signals_when_runtime_unavailable(monkeypatch):
    """Even when the runtime is unavailable, the module's public API
    must return deterministic answers — no exceptions, no segfaults.
    Verify by monkey-patching the runtime probe to return None."""
    import tessera.apple_mlpkg as mod
    monkeypatch.setattr(mod, "apple_gpu_runtime", lambda: None)
    assert mod.compile_mlpackage("anywhere") is None
    assert mod.last_error_kind() == ERROR_OS_UNAVAILABLE
