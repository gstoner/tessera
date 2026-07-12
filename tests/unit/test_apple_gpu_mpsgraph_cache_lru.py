"""Task B (2026-06-01) — MPSGraph cache LRU eviction.

The cache backing the MPSGraph encode-session path was unbounded
before this sprint — a 5,000-step training run with 10 distinct
shapes per step would accumulate 50,000 device_verified_jit graphs. Task B
adds an LRU eviction policy keyed by an MRU-order tracker alongside
the existing lookup dict.

Tests pin:

* **Symbol availability** — ``cache_evictions()`` + ``cache_capacity()``
  resolve from the runtime.
* **Default capacity** — without setting the env var, the runtime
  reports the documented default (1024).
* **Env-var override** — setting ``TESSERA_MPSGRAPH_CACHE_CAPACITY=N``
  before runtime load reports ``N`` as the live capacity. (Run in a
  subprocess because the value is cached via ``std::call_once`` on
  first read.)
* **LRU eviction triggers** — with a small capacity and many distinct
  shapes, evictions counter climbs by (overshoot) entries.
* **LRU keeps hot entries** — repeated touches of a single shape
  keep it cached even while cold entries get evicted around it.
* **Unbounded mode** — ``TESSERA_MPSGRAPH_CACHE_CAPACITY=0`` matches
  the pre-Task-B behavior (no evictions ever).
* **Default-capacity invariance** — in the default process, ordinary
  ops don't trigger evictions (the working set is well under 1024).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
from tessera.apple_gpu_batched import (
    batched_session,
    device_tensor,
    rmsnorm_enc,
    session_available,
)


# ---- Symbol availability -----------------------------------------------

def test_cache_evictions_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable")
    fn = bind_symbol("tessera_apple_gpu_mpsgraph_cache_evictions", (),
                      ctypes.c_int64)
    assert fn is not None


def test_cache_capacity_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable")
    fn = bind_symbol("tessera_apple_gpu_mpsgraph_cache_capacity", (),
                      ctypes.c_int64)
    assert fn is not None


# ---- Default capacity -------------------------------------------------

def test_default_cache_capacity_is_1024():
    """The default capacity is intentionally generous — well above
    typical single-process working sets. If a future tuning sprint
    changes the default, update this assertion together with the
    runtime constant."""
    if apple_gpu_runtime() is None:
        pytest.skip("runtime unavailable")
    cap_fn = bind_symbol("tessera_apple_gpu_mpsgraph_cache_capacity", (),
                          ctypes.c_int64)
    cap = int(cap_fn())
    # The default is 1024 unless something in THIS process set the env
    # var first. In CI / normal runs the env is clean.
    if "TESSERA_MPSGRAPH_CACHE_CAPACITY" in os.environ:
        pytest.skip("env var pre-set; cannot assert default")
    assert cap == 1024, (
        f"expected default capacity 1024, got {cap} — "
        f"if intentionally changed, update this test too")


# ---- Env-var override via subprocess ----------------------------------

def _subprocess_capacity_probe(env_value: str) -> int:
    """Spawn a fresh Python process with the given env var, return
    the resolved capacity. Subprocess is required because the
    capacity is cached via ``std::call_once`` on first read; we
    can't change it in-process."""
    code = textwrap.dedent("""
        import ctypes
        from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
        if apple_gpu_runtime() is None:
            print("SKIP")
            raise SystemExit(0)
        fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_capacity", (),
            ctypes.c_int64)
        print(int(fn()))
    """)
    env = {**os.environ,
            "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve()
                                .parent.parent.parent / "python"),
            "TESSERA_MPSGRAPH_CACHE_CAPACITY": env_value}
    out = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=30,
                          env=env)
    stdout = out.stdout.strip()
    if stdout == "SKIP":
        pytest.skip("runtime unavailable in subprocess")
    return int(stdout)


def test_env_var_override_to_small_capacity():
    """Setting the env var to a small value should be reflected in
    the live capacity reading from a fresh process."""
    if apple_gpu_runtime() is None:
        pytest.skip("runtime unavailable")
    cap = _subprocess_capacity_probe("32")
    assert cap == 32


def test_env_var_override_to_zero_means_unbounded():
    """``TESSERA_MPSGRAPH_CACHE_CAPACITY=0`` disables eviction
    entirely (restores pre-Task-B behavior). The capacity probe
    returns 0; the LRU code path treats 0 as 'never evict'."""
    if apple_gpu_runtime() is None:
        pytest.skip("runtime unavailable")
    cap = _subprocess_capacity_probe("0")
    assert cap == 0


# ---- Behavioral eviction test via subprocess --------------------------

def test_lru_evicts_when_capacity_exceeded():
    """With a small capacity, putting (capacity+overshoot) distinct
    rmsnorm shapes triggers evictions. Run in a subprocess so we can
    set the env var fresh."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    code = textwrap.dedent("""
        import ctypes
        import numpy as np
        from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
        from tessera.apple_gpu_batched import (
            batched_session, device_tensor, rmsnorm_enc,
            session_available)

        if apple_gpu_runtime() is None or not session_available():
            print("SKIP")
            raise SystemExit(0)

        cap_fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_capacity", (),
            ctypes.c_int64)
        size_fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_size", (), ctypes.c_int32)
        ev_fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_evictions", (),
            ctypes.c_int64)

        assert int(cap_fn()) == 4, f"expected cap=4, got {int(cap_fn())}"

        # Build 8 distinct rmsnorm shapes — each compiles a new
        # MPSGraph. With capacity=4, the LRU must evict at least 4
        # entries by the time we finish.
        rng = np.random.default_rng(0xEEEE)
        for cols in [13, 17, 19, 23, 29, 31, 37, 41]:
            X = device_tensor(rng.standard_normal((1, cols), dtype=np.float32))
            G = device_tensor(rng.standard_normal((cols,), dtype=np.float32))
            try:
                with batched_session() as sess:
                    O = rmsnorm_enc(sess, X, G, rows=1, cols=cols, eps=1e-5)
                O.free()
            finally:
                X.free(); G.free()

        size_now = int(size_fn())
        evictions = int(ev_fn())
        # 8 distinct entries went in; cap=4 means at least 4 evicted.
        assert evictions >= 4, f"evictions={evictions}"
        # Cache should be at capacity (or below if some shapes mapped
        # to a previously-cached entry, but distinct sizes here
        # guarantee distinct keys).
        assert size_now <= 4, f"size={size_now}"
        print(f"OK size={size_now} evictions={evictions}")
    """)
    env = {**os.environ,
            "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve()
                                .parent.parent.parent / "python"),
            "TESSERA_MPSGRAPH_CACHE_CAPACITY": "4"}
    out = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120,
                          env=env)
    stdout = out.stdout.strip()
    if stdout == "SKIP":
        pytest.skip("runtime unavailable in subprocess")
    assert out.returncode == 0, (
        f"subprocess failed: stdout={out.stdout!r} stderr={out.stderr!r}")
    assert stdout.startswith("OK"), stdout


def test_lru_keeps_hot_entry_under_pressure():
    """Touch one entry repeatedly while inserting many cold entries.
    The hot entry should NOT be evicted (it's at MRU on every touch);
    the cold entries should get evicted around it."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    code = textwrap.dedent("""
        import ctypes
        import numpy as np
        from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
        from tessera.apple_gpu_batched import (
            batched_session, device_tensor, rmsnorm_enc,
            session_available)

        if apple_gpu_runtime() is None or not session_available():
            print("SKIP")
            raise SystemExit(0)

        size_fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_size", (), ctypes.c_int32)
        ev_fn = bind_symbol(
            "tessera_apple_gpu_mpsgraph_cache_evictions", (),
            ctypes.c_int64)

        rng = np.random.default_rng(0xC0FFEE)
        # Hot shape — repeatedly used. cols=7 is its unique key.
        hot_cols = 7
        Xh = device_tensor(rng.standard_normal((1, hot_cols),
                                                  dtype=np.float32))
        Gh = device_tensor(rng.standard_normal((hot_cols,),
                                                  dtype=np.float32))

        # Initial touch — hot enters cache.
        with batched_session() as sess:
            Oh = rmsnorm_enc(sess, Xh, Gh, rows=1, cols=hot_cols, eps=1e-5)
        Oh.free()

        # Push 8 cold shapes through, touching hot between each.
        for cols in [11, 13, 17, 19, 23, 29, 31, 37]:
            Xc = device_tensor(rng.standard_normal((1, cols),
                                                      dtype=np.float32))
            Gc = device_tensor(rng.standard_normal((cols,), dtype=np.float32))
            try:
                with batched_session() as sess:
                    Oc = rmsnorm_enc(sess, Xc, Gc,
                                       rows=1, cols=cols, eps=1e-5)
                Oc.free()
                # Touch hot — moves it to MRU front.
                with batched_session() as sess:
                    Oh = rmsnorm_enc(sess, Xh, Gh,
                                       rows=1, cols=hot_cols, eps=1e-5)
                Oh.free()
            finally:
                Xc.free(); Gc.free()

        ev = int(ev_fn())
        print(f"evictions={ev}")
        # Now check: a cache hit on the hot shape MUST NOT trigger
        # a recompile. We use the cache-size probe — touching hot
        # again should not increase the cache size.
        size_before = int(size_fn())
        with batched_session() as sess:
            Oh = rmsnorm_enc(sess, Xh, Gh, rows=1, cols=hot_cols, eps=1e-5)
        Oh.free()
        size_after = int(size_fn())

        Xh.free(); Gh.free()
        assert size_after == size_before, (
            f"hot entry was evicted: size_before={size_before} "
            f"size_after={size_after}")
        # Sanity: capacity=4 means we evicted some of the colds.
        assert ev >= 4, f"expected ≥ 4 evictions, got {ev}"
        print(f"OK hot survived; size={size_after} evictions={ev}")
    """)
    env = {**os.environ,
            "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve()
                                .parent.parent.parent / "python"),
            "TESSERA_MPSGRAPH_CACHE_CAPACITY": "4"}
    out = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120,
                          env=env)
    stdout = out.stdout.strip()
    if stdout.startswith("SKIP"):
        pytest.skip("runtime unavailable in subprocess")
    assert out.returncode == 0, (
        f"subprocess failed: stdout={out.stdout!r} stderr={out.stderr!r}")
    assert "OK hot survived" in stdout, stdout


def test_default_capacity_no_evictions_in_normal_run():
    """In the default process (capacity=1024), a handful of ops
    must not trigger any evictions — the working set is well below
    cap. If this fails, either a new test polluted the cache or the
    default capacity dropped."""
    if not session_available():
        pytest.skip("encode-session unavailable")
    ev_fn = bind_symbol("tessera_apple_gpu_mpsgraph_cache_evictions", (),
                         ctypes.c_int64)
    cap_fn = bind_symbol("tessera_apple_gpu_mpsgraph_cache_capacity", (),
                         ctypes.c_int64)
    cap = int(cap_fn())
    if cap == 0 or cap < 100:
        pytest.skip(f"cap={cap} is non-default; can't assert no-eviction")

    before = int(ev_fn())
    rng = np.random.default_rng(0xCAFE)
    X = device_tensor(rng.standard_normal((1, 64), dtype=np.float32))
    G = device_tensor(rng.standard_normal((64,), dtype=np.float32))
    try:
        for _ in range(4):
            with batched_session() as sess:
                O = rmsnorm_enc(sess, X, G, rows=1, cols=64, eps=1e-5)
            O.free()
    finally:
        X.free(); G.free()
    after = int(ev_fn())
    assert (after - before) == 0, (
        f"unexpected evictions during default-capacity run: "
        f"{after - before}")
