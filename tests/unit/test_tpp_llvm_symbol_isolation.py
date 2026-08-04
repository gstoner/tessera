"""The embedded TPP driver must not bind to another library's LLVM.

Root-caused 2026-08-04. `runtime.py` loads `libhiprtc.so` with `RTLD_GLOBAL`,
and HIPRTC statically links its own LLVM. That publishes a full set of LLVM
symbols into the global namespace, so `libtessera_tpp_capi.so` -- which
statically links a *different* LLVM/MLIR build -- resolved its LLVM references
against HIPRTC's copy. Two incompatible LLVMs sharing global state segfault the
moment a pass pipeline runs.

**This was mistaken for flakiness for a long time.** Three `test_solvers_tpp.py`
tests "failed" whenever the full unit suite ran and passed standalone, so they
were repeatedly written off as xdist ordering flakes. They were neither flaky
nor xdist-specific: the suite segfaulted single-process too, deterministically,
with exit 139 inside `tessera_tpp_run_pipeline`. A crash is not a flake, and
"passes alone, fails together" is a symptom of shared process state, not of
scheduling luck.

The fix is `RTLD_DEEPBIND` on the CAPI load, so it prefers its own symbol table.

This file is the behavioural gate: it reproduces the collision directly rather
than asserting that a particular flag is passed, because the property that
matters is "the driver survives a foreign LLVM in the global namespace", not
"the loader sets flag X".
"""

from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
import textwrap

import pytest

from tessera.solvers import tpp


def _hiprtc_path() -> str | None:
    """A shared library that statically links its own LLVM, if present.

    HIPRTC is the one that actually caused this, and it is what `runtime.py`
    loads RTLD_GLOBAL on any ROCm host.
    """
    for pattern in ("/opt/rocm/core/lib/libhiprtc.so*",
                    "/opt/rocm*/lib/libhiprtc.so*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


@pytest.mark.skipif(not tpp.embedded_driver_available(),
                    reason="tessera_tpp_capi not built in this checkout")
@pytest.mark.skipif(_hiprtc_path() is None,
                    reason="no libhiprtc (non-ROCm host): collision not reachable")
def test_tpp_survives_a_foreign_llvm_in_the_global_namespace():
    """The regression, run in a subprocess because the failure is a SEGFAULT.

    An in-process assertion cannot survive the thing being tested, which is
    exactly why this went undiagnosed: the crash takes the test runner with it
    and surfaces as "worker crashed", not as a failure with a traceback.
    """
    script = textwrap.dedent(
        f"""
        import ctypes
        # Publish a foreign LLVM into the global namespace, as runtime.py does.
        ctypes.CDLL({_hiprtc_path()!r}, mode=ctypes.RTLD_GLOBAL)
        from tessera.solvers import tpp
        out = tpp.solve("module {{}}\\n")
        assert "module" in out, out
        print("OK")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "python" + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"embedded TPP driver died with a foreign LLVM loaded "
        f"(returncode={result.returncode}; -11 is SIGSEGV, the original bug). "
        f"stderr: {result.stderr[-400:]}"
    )
    assert "OK" in result.stdout, result.stdout


@pytest.mark.skipif(not tpp.embedded_driver_available(),
                    reason="tessera_tpp_capi not built in this checkout")
def test_the_capi_is_loaded_with_deepbind_where_the_platform_has_it():
    """Ratchet on the mechanism, since the behavioural test above is skipped on
    hosts without HIPRTC — which includes most CI.

    Stated as "if the platform has DEEPBIND, use it" rather than pinning the
    exact mode value: macOS has no RTLD_DEEPBIND and degrades to RTLD_NOW,
    which is the pre-fix behaviour and is correct there.
    """
    import inspect

    src = inspect.getsource(tpp._load_capi)
    if hasattr(os, "RTLD_DEEPBIND"):
        assert "RTLD_DEEPBIND" in src, (
            "the CAPI load no longer requests DEEPBIND; a foreign LLVM in the "
            "global namespace will bind over this library's own and segfault"
        )
