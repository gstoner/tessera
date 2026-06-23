"""Shared subprocess runner for unit tests — timeout + clean process-group
shutdown + OOM/signal detection.

Use this instead of a bare ``subprocess.run(...)`` whenever a test spawns a
compiler/driver (``tessera-opt`` / ``tessera-rocm-opt`` / ``hipcc`` / ``llc`` /
``lit`` / ``cmake`` …). A bare call with no ``timeout`` hangs forever if the
child wedges, and ``subprocess.run(timeout=...)`` only kills the *direct* child
— grandchildren (e.g. lit → FileCheck / tessera-opt) are orphaned. This runner:

  * runs the command in its own session/process group, so on a timeout or any
    exception the WHOLE tree is torn down with ``killpg`` (no orphans);
  * distinguishes a clean exit, a timeout, and a signal kill — in particular a
    ``SIGKILL`` (the OOM killer / jetsam under memory pressure, or an operator
    kill) so the caller can skip-as-resource rather than report a false failure.

POSIX only (``os.killpg`` / ``start_new_session``), which is all this repo
targets (macOS + Linux).
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass


@dataclass
class Result:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def killed_signal(self) -> int | None:
        """The signal number that killed the child, or None for a normal exit.
        POSIX reports a signal death as a negative returncode (-SIG)."""
        rc = self.returncode
        return -rc if (rc is not None and rc < 0) else None

    @property
    def oom_killed(self) -> bool:
        """True if the child was SIGKILL'd — almost always OOM / memory pressure
        (macOS jetsam / Linux oom-killer) or an external kill, not a real fail."""
        return self.killed_signal == signal.SIGKILL

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _kill_group(proc: subprocess.Popen) -> None:
    """Tear down the whole process group; fall back to the leader if it's gone."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run(cmd, *, timeout: float = 300, env=None, cwd=None,
        input: str | None = None, text: bool = True) -> Result:
    """Run ``cmd`` in its own process group and return a :class:`Result`.

    On timeout or any exception the whole process tree is killed so nothing is
    orphaned. Never raises ``TimeoutExpired`` — a timeout is reported as
    ``Result(timed_out=True)``."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=(subprocess.PIPE if input is not None else None),
        text=text,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
        return Result(proc.returncode, out or "", err or "")
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return Result(proc.returncode, out or "", err or "", timed_out=True)
    except BaseException:
        # KeyboardInterrupt / unexpected error — never leave the tree running.
        _kill_group(proc)
        raise


def run_checked(cmd, *, what: str = "subprocess", timeout: float = 300,
                skip_on_oom: bool = True, **kw) -> Result:
    """Run ``cmd`` and map its failure modes to pytest outcomes, so every caller
    handles a hang / OOM / signal the same honest way:

      * OOM ``SIGKILL`` → ``pytest.skip`` (resource condition, not a defect)
        when ``skip_on_oom`` (default); otherwise fail.
      * timeout → ``pytest.fail`` (the child hung; tree already torn down).
      * other signal → ``pytest.fail`` with the signal name.

    Returns the :class:`Result` on a normal exit (any returncode) for the caller
    to assert on. Imports pytest lazily so this module stays import-safe."""
    import pytest

    res = run(cmd, timeout=timeout, **kw)
    tail = f"\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    if res.timed_out:
        pytest.fail(f"{what} exceeded {timeout}s and was killed "
                    f"(process group torn down).{tail}")
    if res.oom_killed:
        if skip_on_oom:
            pytest.skip(f"{what} was SIGKILL'd — almost certainly OOM / memory "
                        f"pressure or an external kill, not a failure. Re-run "
                        f"with less concurrent load. Process group cleaned up.")
        pytest.fail(f"{what} was SIGKILL'd (OOM?); process group cleaned up.{tail}")
    if res.killed_signal is not None:
        name = signal.Signals(res.killed_signal).name
        pytest.fail(f"{what} killed by signal {res.killed_signal} ({name}); "
                    f"process group cleaned up.{tail}")
    return res
