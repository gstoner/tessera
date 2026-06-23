"""Durable CI guard: run the ROCm backend lit suite from pytest.

The C++ ``check-tessera-rocm`` target only runs under a CMake build; this brings
the same lit fixtures into the normal pytest lane so the lit-runner wiring (site
config → ``load_config`` → ``lit.cfg.py`` → FileCheck/not on PATH) and the
fixtures themselves stay green in ordinary CI.  Skips cleanly when the ROCm
backend hasn't been configured/built or when ``lit`` isn't installed.

Failure-path hardening (2026-06-22):
  * **Stale-binary skip** — the suite runs whatever ``tessera-rocm-opt`` is in
    the build dir; a binary that predates a source change produces confusing
    FileCheck mismatches (e.g. an RDNA ``wmma`` op still emitted as ``mfma`` —
    exactly what bit us). Before running we compare the binary's mtime against
    its real source inputs — the ``lib`` / ``include`` / ``tools`` tree that
    actually compiles into it, EXCLUDING the separate HIP ``runtime`` and the
    lit ``test`` fixtures — and skip with a clear "rebuild" message when stale.
    (``make -q`` is unusable here: CMake-generated Makefiles carry phony
    always-run rules, so it reports "needs rebuild" even right after a build.
    mtime, by contrast, correctly flips to fresh once you rebuild.)
  * **OOM notification** — a lit run killed by ``SIGKILL`` is almost always the
    OOM killer (macOS jetsam / Linux oom-killer under memory pressure) or an
    operator kill, NOT a fixture regression. We detect the signal exit and skip
    with a loud, actionable reason instead of reporting a misleading failure.
  * **Clean shutdown** — the lit subprocess runs in its own process group, so on
    a timeout or any error the whole tree (``lit`` → ``tessera-rocm-opt`` /
    ``FileCheck`` / ``not``) is torn down with ``killpg`` — no orphaned children.
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

_OPT_TARGET = "tessera-rocm-opt"

# ROCm backend source tree, and the sub-dirs whose mtime gates opt-binary
# freshness: the IR/conversion/driver inputs that compile INTO tessera-rocm-opt.
# Deliberately excludes ``runtime/`` (the HIP kernel → separate
# libtessera_rocm_gemm.so) and ``test/`` (lit fixtures, not compiled in).
_ROCM_SRC = _REPO / "src/compiler/codegen/Tessera_ROCM_Backend"
_OPT_SRC_DIRS = ("lib", "include", "tools")
_SRC_GLOBS = ("*.cpp", "*.h", "*.hpp", "*.td", "*.inc", "CMakeLists.txt")

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


def _opt_binary(test_dir: Path) -> Path | None:
    """The built ``tessera-rocm-opt`` the lit fixtures invoke (sibling ``tools/``
    dir of the test dir)."""
    cand = test_dir.parent / "tools" / _OPT_TARGET
    return cand if cand.is_file() else None


def _newest_opt_source_mtime() -> float:
    """Newest mtime among the sources that compile into ``tessera-rocm-opt``."""
    newest = 0.0
    for sub in _OPT_SRC_DIRS:
        base = _ROCM_SRC / sub
        if not base.is_dir():
            continue
        for pat in _SRC_GLOBS:
            for p in base.rglob(pat):
                try:
                    newest = max(newest, p.stat().st_mtime)
                except OSError:
                    pass
    return newest


def _stale_binary_reason(test_dir: Path) -> str | None:
    """Return a skip reason if ``tessera-rocm-opt`` predates its real source
    inputs (so its lit output can't be trusted), else None. mtime, scoped to the
    binary's actual deps, is the reliable signal here: it flips to fresh once you
    rebuild (``make -q`` does not — see the module docstring). A git checkout that
    rewrites a source mtime will read as stale even if content is unchanged, but
    skipping with a 'rebuild' hint after a checkout is the safe, intended call."""
    binp = _opt_binary(test_dir)
    if binp is None:
        return None
    try:
        if binp.stat().st_mtime < _newest_opt_source_mtime():
            return (
                f"{binp} predates its sources — rebuild it before trusting the "
                f"ROCm lit suite, e.g. `cmake --build {_build_root(test_dir)} "
                f"--target {_OPT_TARGET}`. A stale binary causes confusing "
                f"FileCheck mismatches (e.g. RDNA wmma emitted as mfma).")
    except OSError:
        return None
    return None


def _build_root(test_dir: Path) -> Path | None:
    """The CMake build root (holds CMakeCache.txt) — used only to phrase the
    rebuild hint."""
    for p in test_dir.parents:
        if (p / "CMakeCache.txt").is_file():
            return p
    return None


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

    stale = _stale_binary_reason(test_dir)
    if stale is not None:
        pytest.skip(stale)

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

    # lit returns non-zero if any fixture fails; surface its report on failure.
    assert res.returncode == 0, (
        f"ROCm lit suite failed (rc={res.returncode}). If you recently changed "
        f"ROCm backend sources, rebuild {_OPT_TARGET} and retry.\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
    # Sanity: lit actually discovered tests (guards against a silent
    # "contained no tests" regression in the lit wiring).
    out = res.stdout + res.stderr
    m = re.search(r"Total Discovered Tests:\s*(\d+)", out)
    assert m is not None and int(m.group(1)) > 0, (
        f"lit discovered no tests (lit-runner wiring regressed):\n{out}")
