"""Guard the shared subprocess runner (_subprocess.run): timeout + clean
process-group shutdown + OOM/signal detection."""

from __future__ import annotations

import os
import signal
import sys

import pytest

from _subprocess import Result, run


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
    # The grandchild must be dead (clean group teardown, no orphan).
    with pytest.raises(OSError):
        os.kill(child_pid, 0)


def test_result_dataclass_signal_math():
    assert Result(-9, "", "").killed_signal == 9
    assert Result(-9, "", "").oom_killed is True
    assert Result(0, "", "").killed_signal is None
    assert Result(1, "", "").oom_killed is False
