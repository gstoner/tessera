"""A remembered build failure must not outlive the environment that caused it.

`_build_rocm_family_hsaco` caches failures so a host that cannot serialize ROCm
stops forking `tessera-opt` once per launch (measured 70.6 ms per call, 677x the
work being done). But `_rocm_serializer_env` re-reads `os.environ` on every
call **deliberately**: whether a build can succeed is a function of the live
environment, not of the process.

So an unscoped negative cache turns a *recoverable* misconfiguration into a
permanent one. A notebook or long-lived service that starts without `ROCM_PATH`,
caches a refusal, and then exports a valid toolkit path would keep falling back
to CPU -- or keep raising under `TESSERA_STRICT_DISPATCH=1` -- until restart.
That is a worse failure than the cost the cache removes.
"""
from __future__ import annotations

import pytest

from tessera import runtime as rt


def test_a_failure_is_remembered_only_for_its_own_environment(monkeypatch):
    """Same env -> reuse the refusal. Changed env -> drop it and retry."""
    monkeypatch.setenv("ROCM_PATH", "/nonexistent/a")
    failure = rt._HsacoBuildFailure("no serializer")
    assert failure.env == rt._rocm_build_env_fingerprint()

    monkeypatch.setenv("ROCM_PATH", "/nonexistent/b")
    assert failure.env != rt._rocm_build_env_fingerprint(), (
        "a changed ROCM_PATH must invalidate the remembered refusal; otherwise "
        "exporting a valid toolkit cannot recover without a process restart")


@pytest.mark.parametrize("var", ["ROCM_PATH", "TESSERA_ROCM_CHIP", "PATH"])
def test_every_input_that_decides_the_build_is_in_the_fingerprint(monkeypatch, var):
    """Each of these reaches `_rocm_serializer_env` or the chip selection, so
    each must be able to invalidate a refusal on its own."""
    monkeypatch.setenv(var, "/first")
    before = rt._rocm_build_env_fingerprint()
    monkeypatch.setenv(var, "/second")
    assert rt._rocm_build_env_fingerprint() != before, var


def test_the_fingerprint_does_no_filesystem_work(monkeypatch):
    """It runs on every cache hit. Probing the filesystem here would hand back
    part of the cost the cache exists to remove."""
    import os
    called = []
    for name in ("stat", "listdir", "access"):
        real = getattr(os, name)
        monkeypatch.setattr(os, name, lambda *a, _n=name, _r=real, **k: (
            called.append(_n), _r(*a, **k))[1])
    rt._rocm_build_env_fingerprint()
    assert called == [], f"fingerprint touched the filesystem: {called}"


def test_a_cached_failure_reraises_rather_than_returning_none(monkeypatch):
    """The refusal must keep flowing through `_rocm_compiled_failed`, so the
    dispatch fallback is still recorded and TESSERA_STRICT_DISPATCH still
    raises. Only the subprocess is skipped."""
    cache: dict = {}
    key = ("family", "k")
    cache[key] = rt._HsacoBuildFailure("cached reason")
    seen = []
    monkeypatch.setattr(rt, "_rocm_compiled_failed",
                        lambda reason: seen.append(reason))
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: None)
    try:
        rt._build_rocm_family_hsaco("family", "directive", cache, key)
    except Exception:
        pass
    assert seen == ["cached reason"], (
        "a cache hit must route through the fallback funnel, not silently "
        f"return or raise a different error; saw {seen}")


# --------------------------------------------------------------------------
# The behaviour, not the helper.
#
# The three tests above exercise `_rocm_build_env_fingerprint` in isolation and
# ALL SIX PASSED with the environment check in `_build_rocm_family_hsaco`
# replaced by `if True:` -- i.e. with the exact defect this file exists to
# prevent still present. Testing the ingredient is not testing the rule.
# --------------------------------------------------------------------------

#: A family the ROCm pipeline actually registers. `ROCMExecutablePipeline`
#: rejects unknown names before reaching the subprocess, so a placeholder here
#: silently makes the test exercise nothing -- which is how the first draft of
#: these two tests reported "0 attempts".
_FAMILY = "normalization"


def _driver(monkeypatch, attempts):
    """Make `_build_rocm_family_hsaco` reach its subprocess and fail there,
    counting how many times it actually got that far."""
    monkeypatch.setattr(rt, "_tessera_opt_path", lambda: "/bin/true")
    monkeypatch.setattr(rt, "_rocm_chip", lambda: "gfx1151")
    monkeypatch.setattr(rt, "_rocm_serializer_env", lambda: None)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "no serializer"

    import subprocess
    def fake_run(*a, **k):
        attempts.append(1)
        return _Result()
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_an_unchanged_environment_does_not_re_fork(monkeypatch):
    attempts: list = []
    _driver(monkeypatch, attempts)
    monkeypatch.setenv("ROCM_PATH", "/nonexistent/a")
    cache: dict = {}
    for _ in range(5):
        with pytest.raises(Exception):
            rt._build_rocm_family_hsaco(_FAMILY, "directive", cache, ("k",))
    assert attempts == [1], (
        "the subprocess must run once and the refusal be reused; "
        f"it ran {len(attempts)} times -- this is the 70.6 ms per launch")


def test_a_changed_environment_retries_the_build(monkeypatch):
    """The P2 on #667: a cached refusal must not outlive its environment, or
    exporting a valid ROCM_PATH cannot recover without a process restart."""
    attempts: list = []
    _driver(monkeypatch, attempts)
    cache: dict = {}
    monkeypatch.setenv("ROCM_PATH", "/nonexistent/a")
    with pytest.raises(Exception):
        rt._build_rocm_family_hsaco(_FAMILY, "directive", cache, ("k",))
    assert len(attempts) == 1

    monkeypatch.setenv("ROCM_PATH", "/nonexistent/b")   # user fixes their env
    with pytest.raises(Exception):
        rt._build_rocm_family_hsaco(_FAMILY, "directive", cache, ("k",))
    assert len(attempts) == 2, (
        "a changed toolchain environment must invalidate the cached refusal "
        "and re-attempt the build; it did not, so a recoverable "
        "misconfiguration is permanent for the process")
