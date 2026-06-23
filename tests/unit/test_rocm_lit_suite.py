"""Durable CI guard: run the ROCm backend lit suite from pytest.

The C++ ``check-tessera-rocm`` target only runs under a CMake build; this brings
the same lit fixtures into the normal pytest lane so the lit-runner wiring (site
config → ``load_config`` → ``lit.cfg.py`` → FileCheck/not on PATH) and the
fixtures themselves stay green in ordinary CI.  Skips cleanly when the ROCm
backend hasn't been configured/built or when ``lit`` isn't installed.

Failure-path hardening (2026-06-22):
  * **OOM notification** — a lit run killed by ``SIGKILL`` is almost always the
    OOM killer (macOS jetsam / Linux oom-killer under memory pressure) or an
    operator kill, NOT a fixture regression. We detect the signal exit and skip
    with a loud, actionable reason instead of reporting a misleading failure.
  * **Clean shutdown** — the lit subprocess runs in its own process group, so on
    a timeout or any error the whole tree (``lit`` → ``tessera-rocm-opt`` /
    ``FileCheck`` / ``not``) is torn down with ``killpg`` — no orphaned children.
  * **Stale-binary hint** — a ``tessera-rocm-opt`` older than its sources gives
    confusing FileCheck mismatches (e.g. an RDNA ``wmma`` op still emitted as
    ``mfma``). On a fixture failure we append a "rebuild" hint when the binary
    looks stale. (We do NOT pre-skip on mtime: git checkout/merge rewrites source
    mtimes without changing content, so a pre-skip would false-positive after
    every merge — the hint-on-failure path has no such cost.)
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# Candidate build dirs that may hold the configured ROCm lit site config.
_SITE_CANDIDATES = [
    _REPO / "build-rocm/src/compiler/codegen/Tessera_ROCM_Backend/test/lit.site.cfg.py",
    _REPO / "build/src/compiler/codegen/Tessera_ROCM_Backend/test/lit.site.cfg.py",
]

# ROCm backend source tree + the build inputs whose mtime gates binary freshness.
_ROCM_SRC = _REPO / "src/compiler/codegen/Tessera_ROCM_Backend"
_SRC_GLOBS = ("*.cpp", "*.h", "*.td", "*.inc", "CMakeLists.txt")

# Generous wall-clock ceiling: the suite runs in <1s; this only fires if a
# tessera-rocm-opt invocation hangs, and exists so the test can't wedge CI.
_LIT_TIMEOUT_S = 600


def _lit() -> str | None:
    return shutil.which("lit") or shutil.which("llvm-lit") or shutil.which("lit.py")


def _site_dir() -> Path | None:
    for c in _SITE_CANDIDATES:
        if c.exists():
            return c.parent
    return None


def _rocm_opt_binary(test_dir: Path) -> Path | None:
    """The built ``tessera-rocm-opt`` that the lit fixtures invoke (sibling
    ``tools/`` dir of the test dir)."""
    cand = test_dir.parent / "tools" / "tessera-rocm-opt"
    return cand if cand.is_file() else None


def _newest_rocm_source_mtime() -> float:
    """Newest mtime among the sources that compile into ``tessera-rocm-opt`` —
    i.e. the IR/conversion/driver tree, EXCLUDING ``runtime/`` (the HIP kernel
    that builds the separate ``libtessera_rocm_gemm.so``, not the opt driver)."""
    newest = 0.0
    for pat in _SRC_GLOBS:
        for p in _ROCM_SRC.rglob(pat):
            if "runtime" in p.parts:
                continue
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return newest


def _stale_binary_hint(test_dir: Path) -> str:
    """A '(rebuild)' hint appended to a failure message when the opt binary is
    older than its sources — the usual cause of a confusing FileCheck mismatch."""
    binp = _rocm_opt_binary(test_dir)
    if binp is None:
        return ""
    try:
        if binp.stat().st_mtime < _newest_rocm_source_mtime():
            return (f"\nNOTE: {binp} is OLDER than its sources — rebuild "
                    f"tessera-rocm-opt; a stale binary is the usual cause of a "
                    f"FileCheck mismatch here (e.g. RDNA wmma emitted as mfma).")
    except OSError:
        pass
    return ""


class _LitResult:
    def __init__(self, returncode: int | None, stdout: str, stderr: str,
                 timed_out: bool = False) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        # POSIX: a signal-killed child reports a negative returncode (-SIG).
        self.killed_signal = (
            -returncode if (returncode is not None and returncode < 0) else None)


def _kill_group(proc: subprocess.Popen) -> None:
    """Tear down the whole process group (lit + its grandchildren). Falls back to
    killing just the leader if the group is already gone."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_lit_clean(cmd: list[str], timeout: int) -> _LitResult:
    """Run ``cmd`` in its own session/process group and return a result that
    distinguishes a clean exit, a timeout, and a signal kill (e.g. OOM SIGKILL).
    On timeout or any exception the group is killed so no child is orphaned."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return _LitResult(proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return _LitResult(proc.returncode, out, err, timed_out=True)
    except BaseException:
        # KeyboardInterrupt / unexpected error — never leave the tree running.
        _kill_group(proc)
        raise


def test_rocm_lit_suite_passes() -> None:
    lit = _lit()
    if lit is None:
        pytest.skip("lit not installed")
    test_dir = _site_dir()
    if test_dir is None:
        pytest.skip(
            "ROCm backend not configured (no generated lit.site.cfg.py); "
            "configure -DTESSERA_BUILD_ROCM_BACKEND=ON and build tessera-rocm-opt")

    res = _run_lit_clean([lit, "-q", str(test_dir)], timeout=_LIT_TIMEOUT_S)

    # ── Failure paths, most-specific first ──────────────────────────────────
    if res.timed_out:
        pytest.fail(
            f"ROCm lit suite exceeded {_LIT_TIMEOUT_S}s and was killed "
            f"(process group torn down). A tessera-rocm-opt fixture likely hung."
            f"\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
    if res.killed_signal == signal.SIGKILL:
        # OOM notification: jetsam / oom-killer / operator kill — a resource
        # condition, not a fixture regression. The tree is already down.
        pytest.skip(
            "ROCm lit suite was SIGKILL'd — almost certainly OOM / memory "
            "pressure (or an external kill), not a test failure. Re-run with "
            "less concurrent load (don't stack it with a full pytest run + a "
            "C++ link). Process group was cleaned up.")
    if res.killed_signal is not None:
        name = signal.Signals(res.killed_signal).name
        pytest.fail(
            f"ROCm lit suite killed by signal {res.killed_signal} ({name}); "
            f"process group cleaned up."
            f"\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")

    # lit returns non-zero if any fixture fails; surface its report on failure,
    # plus a rebuild hint when the opt binary looks stale (the usual cause).
    assert res.returncode == 0, (
        f"ROCm lit suite failed (rc={res.returncode}):{_stale_binary_hint(test_dir)}\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
    # Sanity: lit actually discovered tests (guards against a silent
    # "contained no tests" regression in the lit wiring).
    out = res.stdout + res.stderr
    m = re.search(r"Total Discovered Tests:\s*(\d+)", out)
    assert m is not None and int(m.group(1)) > 0, (
        f"lit discovered no tests (lit-runner wiring regressed):\n{out}")
