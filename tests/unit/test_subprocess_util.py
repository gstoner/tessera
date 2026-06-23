"""Guard the shared subprocess runner (_subprocess.run): timeout + clean
process-group shutdown + OOM/signal detection."""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from _subprocess import Result, run


def _gone_or_zombie(pid: int) -> bool:
    """True if ``pid`` is no longer a *running* process — either reaped (gone)
    or a zombie awaiting its (now-dead) parent's reaper. After the runner kills
    the process group, a grandchild is reparented to init and lingers briefly as
    a zombie; on Linux ``os.kill(zombie, 0)`` still succeeds until init reaps it,
    so a bare ``os.kill`` check races. A zombie is effectively dead (not an
    orphan consuming resources)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return True  # reaped / gone
    try:
        with open(f"/proc/{pid}/stat") as fh:
            line = fh.read()
        # `pid (comm) state ...` — comm may contain ')'; take the char after the
        # last ')'.
        return line[line.rfind(")") + 2] == "Z"
    except OSError:
        return True  # /proc entry vanished between the two checks (Linux)
    except IndexError:
        return False
    # No /proc (e.g. macOS): the os.kill above is authoritative; still running.
    return False


def test_normal_exit_captures_output():
    res = run([sys.executable, "-c", "print('hi'); import sys; sys.exit(0)"], timeout=30)
    assert res.ok and res.returncode == 0
    assert res.stdout.strip() == "hi"
    assert not res.timed_out and res.killed_signal is None and not res.oom_killed


def test_nonzero_exit_is_not_a_kill():
    res = run([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=30)
    assert res.returncode == 3 and not res.ok
    assert res.killed_signal is None and not res.oom_killed and not res.timed_out


def test_sigkill_detected_as_oom():
    # Simulates the OOM killer / jetsam sending SIGKILL.
    res = run([sys.executable, "-c",
               "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], timeout=30)
    assert res.returncode == -signal.SIGKILL
    assert res.killed_signal == signal.SIGKILL
    assert res.oom_killed is True


def test_timeout_kills_whole_process_tree():
    # Parent spawns a long-sleeping grandchild and writes its pid, then sleeps.
    # On timeout the runner must killpg the group so the grandchild dies too.
    prog = (
        "import subprocess, sys, time, os\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({__file__!r} + '.childpid', 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    )
    res = run([sys.executable, "-c", prog], timeout=2)
    assert res.timed_out is True
    pidfile = __file__ + ".childpid"
    try:
        child_pid = int(open(pidfile).read().strip())
    finally:
        try:
            os.remove(pidfile)
        except OSError:
            pass
    # The grandchild must be torn down (clean group teardown, no orphan). Poll
    # for it to stop running — it may briefly be a zombie before init reaps it.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _gone_or_zombie(child_pid):
            break
        time.sleep(0.1)
    else:
        pytest.fail(
            f"grandchild pid {child_pid} still running 10s after the timeout — "
            f"the process group was not torn down (orphan leak)")


def test_result_dataclass_signal_math():
    assert Result(-9, "", "").killed_signal == 9
    assert Result(-9, "", "").oom_killed is True
    assert Result(0, "", "").killed_signal is None
    assert Result(1, "", "").oom_killed is False
