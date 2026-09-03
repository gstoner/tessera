"""APPLE-DISPATCH-WEDGE-1 — a device that stops answering must stop being asked.

`commit_mpsgraph_and_wait_with_timeout` waits 30 s (60 s at one site) and, on
expiry, reports the timeout and returns. Correctly — but with no memory that it
happened, so every later dispatch paid the full timeout again. An Apple sweep
was observed stalled for **70 minutes** where the healthy run takes 4 minutes.

Nothing accumulated by design: the runtime's `g_last_gpu_error_kind` is a
thread-local *last*-error for reporting, and the wait helper clears the dispatch
telemetry **on entry**, erasing the previous timeout before the next attempt.

Host-free: these drive the real `_apple_gpu_run_checked` with the error channel
faked, so the breaker's own logic is under test rather than a reimplementation
of it. Whether the 70 minutes was one uninterruptible wait or ~140 sequential
timeouts is still unresolved (see the Apple plan) — the breaker addresses the
second and bounds the damage either way.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt

TIMEOUT = rt._APPLE_GPU_ERROR_KIND_TIMEOUT
LIMIT = rt._APPLE_GPU_DISPATCH_TIMEOUT_LIMIT


@pytest.fixture(autouse=True)
def _closed_breaker():
    rt.reset_apple_gpu_dispatch_breaker()
    yield
    rt.reset_apple_gpu_dispatch_breaker()


class _Channel:
    """Stands in for the C last-error channel, and counts real dispatches."""

    def __init__(self, kind: int, detail: str | None = "simulated"):
        self.kind, self.detail, self.dispatches = kind, detail, 0

    def install(self, monkeypatch):
        monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
        monkeypatch.setattr(rt, "_apple_gpu_peek_gpu_error_kind", lambda: self.kind)
        monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: self.detail)
        monkeypatch.setattr(rt, "_note_dispatch_fallback",
                            lambda *a, **k: None)
        return self

    def kernel(self):
        self.dispatches += 1
        return "gpu"


def _run(channel):
    return rt._apple_gpu_run_checked("tessera.add", channel.kernel, lambda: "host")


def test_repeated_timeouts_stop_reaching_the_device(monkeypatch):
    """The whole point: the (LIMIT+1)-th call must not dispatch at all."""
    channel = _Channel(TIMEOUT).install(monkeypatch)

    for _ in range(LIMIT):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    # Ten more calls, zero further dispatches -- ten timeouts not paid.
    for _ in range(10):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT, (
        "breaker open but the device was still being asked")


def test_the_breaker_needs_a_streak_not_a_single_timeout(monkeypatch):
    """One timeout can be a slow dispatch under load; three in a row is a
    device that stopped answering. Tripping on the first would turn an
    ordinary hiccup into a process-wide GPU shutdown."""
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for index in range(LIMIT - 1):
        _run(channel)
        assert rt.apple_gpu_dispatch_breaker_state()["open"] is False, index


def test_a_success_closes_the_breaker_and_forgets_the_streak(monkeypatch):
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT - 1):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == LIMIT - 1

    channel.detail = None                      # the device answers again
    assert _run(channel) == "gpu"
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0

    channel.detail = "simulated"
    for _ in range(LIMIT - 1):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False, (
        "the streak survived a successful dispatch")


@pytest.mark.parametrize("kind", [2, 3, 4])
def test_an_ordinary_op_failure_never_trips_the_breaker(kind, monkeypatch):
    """Kinds 2-4 are per-op failures -- a bad buffer, an unsupported shape.

    They say nothing about whether the device is answering, and counting them
    would open the breaker on a workload that merely uses an unsupported op a
    few times in a row.
    """
    channel = _Channel(kind).install(monkeypatch)
    for _ in range(LIMIT * 3):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT * 3
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False


def test_an_interleaved_op_failure_does_not_manufacture_a_streak(monkeypatch):
    """Timeout, op-failure, timeout is not two consecutive timeouts."""
    channel = _Channel(TIMEOUT).install(monkeypatch)
    _run(channel)
    channel.kind = 2
    _run(channel)
    channel.kind = TIMEOUT
    _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 1
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False


def test_reset_reopens_the_device_for_a_caller_that_knows_it_recovered(monkeypatch):
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        _run(channel)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    rt.reset_apple_gpu_dispatch_breaker()
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False
    channel.detail = None
    assert _run(channel) == "gpu"
    assert channel.dispatches == LIMIT + 1


def test_the_escape_hatch_restores_the_old_behaviour(monkeypatch):
    """A breaker nobody can turn off is a new way to lose a working device."""
    monkeypatch.setenv("TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER", "1")
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT * 2):
        assert _run(channel) == "host"
    assert channel.dispatches == LIMIT * 2, "the breaker fired despite the opt-out"
    assert rt.apple_gpu_dispatch_breaker_state()["disabled"] is True


def test_the_open_breaker_still_returns_the_correct_host_result(monkeypatch):
    """Cheapness is not the contract; correctness is.

    Skipping the dispatch must not skip the computation -- an open breaker
    returns the host fallback's value, which is the same answer the GPU lane
    would have produced, just slower.
    """
    import numpy as np

    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        _run(channel)

    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    out = rt._apple_gpu_run_checked(
        "tessera.add", lambda: pytest.fail("dispatched while open"),
        lambda: a + b)
    np.testing.assert_array_equal(out, a + b)


# ── Device-resident paths (APPLE-DISPATCH-WEDGE-1, second instance) ──────────
# The `DeviceTensor` paths called their C symbol directly and returned `None`
# on any rc != 1. They never touched the last-error channel, so an open breaker
# did not stop them asking the device (30 s each), and their timeouts neither
# counted toward the streak nor produced a diagnostic. These pin the routing.


class _Resident:
    """A stand-in `DeviceTensor`: only what the resident paths read."""

    def __init__(self, shape, dtype):
        import numpy as np

        self.shape, self.dtype, self.handle, self.freed = tuple(shape), np.dtype(dtype), object(), 0

    def free(self):
        self.freed += 1


class _Sym:
    """A fake C symbol: counts dispatches, returns a fixed rc."""

    def __init__(self, rc):
        self.rc, self.calls = rc, 0

    def __call__(self, *args):
        self.calls += 1
        return self.rc


RESIDENT_PATHS = [
    # (dev-sym accessor to fake, python path, argument factory)
    ("_apple_gpu_gather_blocks_dev_sym",
     lambda: rt._apple_gpu_gather_blocks_device(
         _Resident((4, 2, 8), "float32"), _Resident((2,), "int32"), 4, 2, 2, 8)),
    ("_apple_gpu_rowop_dev_sym",
     lambda: rt._apple_gpu_rowop_device(_Resident((3, 8), "float32"), 2)),
    ("_apple_gpu_paged_latent_attention_dev_sym",
     lambda: rt._apple_gpu_paged_latent_attention_device(
         _Resident((1, 8), "float32"), _Resident((4, 2, 8), "float32"),
         _Resident((4, 2, 4), "float32"), _Resident((2,), "int32"),
         num_blocks=4, n_blocks=2, block_size=2, logical_length=3,
         latent_dim=8, rope_dim=4, causal_offset=0, window=0, scale=1.0)),
    ("_apple_gpu_dense_latent_attention_dev_sym",
     lambda: rt._apple_gpu_dense_latent_attention_device(
         _Resident((1, 8), "float32"), _Resident((3, 8), "float32"),
         _Resident((3, 4), "float32"), logical_length=3, latent_dim=8,
         rope_dim=4, causal_offset=0, window=0, scale=1.0)),
]


@pytest.fixture
def _resident_output(monkeypatch):
    """`DeviceTensor.empty` hands out a stand-in so the test is host-free and
    can see whether the path freed the output it allocated."""
    made = []

    def _empty(shape, dtype):
        made.append(_Resident(shape, dtype))
        return made[-1]

    monkeypatch.setattr(rt.DeviceTensor, "empty", staticmethod(_empty))
    return made


@pytest.mark.parametrize("accessor,invoke", RESIDENT_PATHS, ids=[p[0] for p in RESIDENT_PATHS])
def test_resident_path_does_not_ask_an_open_breaker(accessor, invoke, monkeypatch, _resident_output):
    sym = _Sym(rc=1)
    monkeypatch.setattr(rt, accessor, lambda: sym)
    _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    assert invoke() is None, "an open breaker must hand the caller to its host path"
    assert sym.calls == 0, "breaker open but the resident path still dispatched"
    assert [t.freed for t in _resident_output] == [1], "the undispatched output leaked"


@pytest.mark.parametrize("accessor,invoke", RESIDENT_PATHS, ids=[p[0] for p in RESIDENT_PATHS])
def test_resident_path_timeouts_count_toward_the_streak(accessor, invoke, monkeypatch, _resident_output):
    """The C side reports kind 1 and returns rc 0 on expiry; that must be one
    streak entry per call, and the LIMIT-th call must open the breaker."""
    sym = _Sym(rc=0)
    monkeypatch.setattr(rt, accessor, lambda: sym)
    _Channel(TIMEOUT).install(monkeypatch)
    for index in range(LIMIT):
        assert invoke() is None
        assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == index + 1
    assert sym.calls == LIMIT
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True
    assert invoke() is None
    assert sym.calls == LIMIT, "the (LIMIT+1)-th resident call still paid a timeout"


@pytest.mark.parametrize("accessor,invoke", RESIDENT_PATHS, ids=[p[0] for p in RESIDENT_PATHS])
def test_resident_path_validation_decline_is_not_a_timeout(accessor, invoke, monkeypatch, _resident_output):
    """rc 0 with no error kind is the C side declining the shape; it must not
    count as evidence about the device, and must still free the output."""
    sym = _Sym(rc=0)
    monkeypatch.setattr(rt, accessor, lambda: sym)
    _Channel(0, detail=None).install(monkeypatch)
    for _ in range(LIMIT + 1):
        assert invoke() is None
    assert sym.calls == LIMIT + 1
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0
    assert all(t.freed == 1 for t in _resident_output)


@pytest.mark.parametrize("accessor,invoke", RESIDENT_PATHS, ids=[p[0] for p in RESIDENT_PATHS])
def test_resident_path_success_returns_the_output_and_closes_the_streak(accessor, invoke, monkeypatch, _resident_output):
    sym = _Sym(rc=1)
    monkeypatch.setattr(rt, accessor, lambda: sym)
    channel = _Channel(TIMEOUT).install(monkeypatch)
    rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 1
    channel.detail = None
    out = invoke()
    assert out is _resident_output[0] and out.freed == 0
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0


def test_resident_timeout_is_a_named_fallback_not_a_silent_none(monkeypatch, _resident_output):
    """Decision #21: a reported timeout on the resident path must land in the
    fallback log under the op's name, and raise under strict dispatch."""
    monkeypatch.setattr(rt, "_apple_gpu_gather_blocks_dev_sym", lambda: _Sym(rc=0))
    monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
    monkeypatch.setattr(rt, "_apple_gpu_peek_gpu_error_kind", lambda: TIMEOUT)
    monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: "did not signal")
    log = []
    monkeypatch.setattr(rt, "_note_dispatch_fallback", lambda op, reason, exc=None: log.append((op, reason)))
    assert RESIDENT_PATHS[0][1]() is None
    assert log and log[0][0] == "apple_gpu.gather_blocks_dev" and "did not signal" in log[0][1]


def test_every_resident_dev_symbol_dispatch_is_breaker_routed():
    """Drift gate, per DISPATCH rather than per function.

    An earlier form of this test asked only whether a function that dispatches
    on a resident symbol *also mentions* the helper somewhere. Review caught
    that a function with two dispatches, one wrapped and one not, satisfies it
    -- and so does adding a bare ``sym(...)`` beside an existing wrapped call,
    which is exactly how this regresses. So each call to a name bound from a
    ``*_dev_sym()`` / ``*_dev_f32()`` accessor, or from a ``getattr`` on a
    ``..._dev`` / ``ts_dev_cast`` symbol, is checked for a
    ``_apple_gpu_device_call_checked`` call among its own ANCESTORS.
    """
    import ast
    import inspect
    import re

    src = inspect.getsource(rt)
    tree = ast.parse(src)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    accessor = re.compile(r"_apple_gpu_\w+_dev_(?:sym|f32)$")
    symbol = re.compile(r"(?:_dev|_dev_f32|ts_dev_cast)$")

    def _routed(node):
        """Is this call lexically inside a _apple_gpu_device_call_checked call?"""
        cur = parents.get(node)
        while cur is not None:
            if (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name)
                    and cur.func.id == "_apple_gpu_device_call_checked"):
                return True
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                return False
            cur = parents.get(cur)
        return False

    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name == "_apple_gpu_device_call_checked":
            continue
        # Names bound to a resident symbol anywhere in this function ...
        bound = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            callee = node.value.func
            hit = False
            if isinstance(callee, ast.Name) and accessor.search(callee.id):
                hit = True
            elif (isinstance(callee, ast.Name) and callee.id == "getattr"
                  and len(node.value.args) >= 2):
                name = node.value.args[1]
                hit = (isinstance(name, ast.Constant) and isinstance(name.value, str)
                       and bool(symbol.search(name.value)))
            if hit:
                bound.update(x.id for x in node.targets if isinstance(x, ast.Name))
        if not bound:
            continue
        # ... and EVERY call of one of them must be routed. An accessor that
        # only binds argtypes never calls the symbol and so has nothing to fail.
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in bound and not _routed(node)):
                offenders.append(f"{fn.name}: {node.func.id}(...) at line {node.lineno}")
    assert offenders == [], (
        "resident dispatch(es) bypass the breaker:\n  " + "\n  ".join(offenders))


def test_the_drift_gate_rejects_a_second_unwrapped_dispatch(monkeypatch):
    """The gate must fail the case that motivated rewriting it: a function
    where ONE dispatch is wrapped and a second is not. Asserted against a
    synthetic module rather than by breaking the real one."""
    import ast
    import re
    import textwrap

    source = textwrap.dedent("""
        def _both_shapes():
            sym = _apple_gpu_thing_dev_sym()
            ok = _apple_gpu_device_call_checked("a", lambda: sym(1))
            rc = sym(2)                     # <- the one that must be caught
            return ok and rc == 1
    """)
    tree = ast.parse(source)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    accessor = re.compile(r"_apple_gpu_\w+_dev_(?:sym|f32)$")

    def routed(node):
        cur = parents.get(node)
        while cur is not None:
            if (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name)
                    and cur.func.id == "_apple_gpu_device_call_checked"):
                return True
            if isinstance(cur, (ast.FunctionDef, ast.Module)):
                return False
            cur = parents.get(cur)
        return False

    bound = {x.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and accessor.search(n.value.func.id)
             for x in n.targets if isinstance(x, ast.Name)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id in bound]
    assert len(calls) == 2, "fixture should hold two dispatches"
    assert [routed(c) for c in calls].count(False) == 1, (
        "a function-wide flag would call this clean; the per-dispatch check "
        "must see exactly one unrouted call")


# ── The MTL4 lanes: a bounded wait that reports nothing ─────────────────────
# `mtl4_encode_and_wait` waits 10 s and returns false WITHOUT touching the
# error channel, so by return value alone an MTL4 stall is identical to a shape
# decline: the streak reset and no diagnostic was emitted. Classified by
# duration instead, which is the only signal the C side leaves.

SILENT = rt._APPLE_GPU_SILENT_FAILURE_TIMEOUT_S


class _Clock:
    """A monotonic clock the test advances by hand, so no test sleeps."""

    def __init__(self):
        self.now = 1000.0

    def install(self, monkeypatch, advance_per_call):
        monkeypatch.setattr(rt.time, "monotonic", lambda: self.now)

        def _tick():
            self.now += advance_per_call
            return 0                       # rc 0 -- the C side reports failure

        return _tick


@pytest.fixture
def _silent_channel(monkeypatch):
    """The C error channel says NOTHING, as the MTL4 path leaves it."""
    monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
    monkeypatch.setattr(rt, "_apple_gpu_peek_gpu_error_kind", lambda: 0)
    monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: None)
    notes = []
    monkeypatch.setattr(rt, "_note_dispatch_fallback",
                        lambda op, reason, exc=None: notes.append((op, reason)))
    return notes


def test_a_silent_stall_counts_as_a_timeout(monkeypatch, _silent_channel):
    """The MTL4 defect: 10 s, false, nothing reported. It must still open."""
    tick = _Clock().install(monkeypatch, advance_per_call=10.0)
    for index in range(LIMIT):
        assert rt._apple_gpu_device_call_checked("apple_gpu.mtl4_matmul2d_dev", tick) is False
        assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == index + 1
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True


def test_a_silent_stall_is_a_named_fallback(monkeypatch, _silent_channel):
    """Decision #21: it may not be a silent False."""
    tick = _Clock().install(monkeypatch, advance_per_call=10.0)
    rt._apple_gpu_device_call_checked("apple_gpu.mtl4_matmul2d_dev", tick)
    assert _silent_channel, "a silent stall produced no diagnostic at all"
    op, reason = _silent_channel[0]
    assert op == "apple_gpu.mtl4_matmul2d_dev"
    assert "10.0s" in reason and "error channel" in reason


def test_a_fast_decline_is_still_not_a_timeout(monkeypatch, _silent_channel):
    """The other half: a shape decline returns in microseconds and must keep
    resetting the streak, or an unsupported-shape workload opens the breaker."""
    tick = _Clock().install(monkeypatch, advance_per_call=0.000_01)
    for _ in range(LIMIT + 2):
        assert rt._apple_gpu_device_call_checked("apple_gpu.mtl4_matmul2d_dev", tick) is False
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is False
    assert _silent_channel == [], "a validation decline must stay silent"


def test_the_threshold_separates_the_two_classes_by_a_wide_margin():
    """The constant is only defensible while it sits far from both classes:
    four orders of magnitude above a validation decline, and at most half the
    shortest real wait (`mtl4_encode_and_wait`'s 10 s)."""
    assert 0.001 < SILENT <= 5.0
    assert SILENT >= 1000 * 0.000_01
    assert SILENT <= 10.0 / 2


def test_the_numpy_lanes_are_not_duration_classified(monkeypatch, _silent_channel):
    """`silent_failure_timeout_s` defaults to None, so a slow numpy-lane
    dispatch that reports nothing is still a success, exactly as before."""
    clock = _Clock()
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock.now)

    def _slow():
        clock.now += 60.0
        return "gpu"

    assert rt._apple_gpu_run_checked("tessera.add", _slow, lambda: "host") == "gpu"
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0


def test_an_array_returning_lane_is_never_truth_tested(monkeypatch, _silent_channel):
    """Regression guard for the classifier's own operand order.

    The numpy lanes return ndarrays, and `not <multi-element array>` raises
    ValueError. The silent-failure test may therefore only be reached after
    the opt-in check has passed -- which only the bool-returning resident
    adapter triggers. Written because the first draft had the two conjuncts
    the other way round, which would have raised on every numpy dispatch that
    reported no error, i.e. on every successful one.
    """
    import numpy as np

    clock = _Clock()
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock.now)

    def _slow_array():
        clock.now += 60.0                 # comfortably past the threshold
        return np.zeros(4)                # falsy elementwise, ambiguous as a bool

    out = rt._apple_gpu_run_checked("tessera.add", _slow_array, lambda: "host")
    np.testing.assert_array_equal(out, np.zeros(4))
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0


# ── Direct C-symbol sites (APPLE-DISPATCH-WEDGE-1, third coverage pass) ──────
# After the resident paths were routed, a re-enumeration found the remaining
# class: functions that bind a symbol from `getattr(runtime, "tessera_apple_gpu_*")`
# or an `_apple_gpu_*_sym()` / `_apple_gpu_*_f32()` accessor and call it with
# `rc = sym(...)` -- linalg, random, GQA, batched attention, MPSGraph control
# flow, the Metal 4 lanes, `msl_spec_accept`, the MLP session's host-input
# `run`, and the value-lane executors with their `*_available` probes. These
# pin the routing for one representative per family, then a gate over the
# whole class -- classified against the .mm, so a probe that never commits a
# command buffer is exempt by evidence, not by a hand-kept list.

_RAISED = object()


class _Runtime:
    """A fake loaded dylib exposing exactly the symbols given."""

    def __init__(self, **symbols):
        self.__dict__.update(symbols)


def _install(monkeypatch, **accessors):
    for name, value in accessors.items():
        monkeypatch.setattr(rt, name, value)


def _mlp_session_run(sym):
    def _invoke():
        session = rt.AppleGPUMLPSession(np.ones((2, 3), np.float16), np, dtype="f16")
        assert session.ran_on_gpu, "the fake create symbol must hand back a handle"
        return session.run(np.ones((4, 2), np.float16))

    return _invoke


def _catching(fn):
    def _invoke():
        try:
            return fn()
        except ValueError:
            return _RAISED

    return _invoke


_GQA_Q = np.ones((2, 3, 4), np.float32)     # [q_heads=2, Sq, D]
_GQA_KV = np.ones((1, 3, 4), np.float32)    # [kv_heads=1, Sk, D]
_SCAN = dict(Wh=np.eye(2, dtype=np.float32) * 0.5, Wx=np.ones((3, 2), np.float32),
             xseq=np.ones((4, 3), np.float32), init=np.zeros(2, np.float32))


def _scan_reference():
    carry = _SCAN["init"].astype(np.float64)
    ys = []
    for t in range(4):
        carry = np.tanh(carry @ _SCAN["Wh"] + _SCAN["xseq"][t] @ _SCAN["Wx"])
        ys.append(carry)
    return np.asarray(ys, np.float32)


def _site(id_, install, invoke, *, declined, ok_rc=1, fail_rc=0, once=False):
    """One direct site: `install(mp, sym)` binds the fake symbol, `invoke(sym)`
    returns a zero-arg call of the runtime function, `declined(result)` says
    whether the result is the host/decline outcome. `fail_rc=None` marks a
    void symbol -- it cannot decline; failure reaches Python only through the
    error channel. `once` marks a probe that caches its first answer per
    process, so it can only be driven through the breaker one time."""
    return pytest.param(install, invoke, declined, ok_rc, fail_rc, once, id=id_)


DIRECT_SITES = [
    _site("linalg.cholesky",
          lambda mp, sym: _install(mp, _apple_gpu_linalg_sym=lambda name, argtypes: sym),
          lambda sym: lambda: rt._apple_gpu_chol_2d(np.eye(2, dtype=np.float32), np),
          declined=lambda r: r is None, ok_rc=0, fail_rc=1),
    _site("linalg.tri_solve_batched",
          lambda mp, sym: _install(mp, _apple_gpu_linalg_sym=lambda name, argtypes: sym),
          lambda sym: lambda: rt._apple_gpu_tri_solve_batched_msl(
              np.ones((2, 2, 2), np.float32), np.ones((2, 2, 1), np.float32), np,
              lower=True, trans=False, unit=False),
          declined=lambda r: r is None),
    _site("random.uniform",
          lambda mp, sym: _install(mp, _apple_gpu_random_sym=lambda name: sym),
          lambda sym: lambda: rt.apple_gpu_random_uniform((4,), np, seed=1),
          declined=lambda r: r[1] is False and r[0].shape == (4,)),
    _site("attention.gqa_f32",
          lambda mp, sym: _install(mp, _apple_gpu_flash_attn_variant_status=lambda suffix: sym),
          lambda sym: lambda: rt._apple_gpu_dispatch_gqa(_GQA_Q, _GQA_KV, _GQA_KV, 2, 1, np),
          declined=lambda r: r is None),
    _site("attention.bsmm_f32",
          lambda mp, sym: _install(mp, _apple_gpu_bsmm_f32=lambda: sym),
          lambda sym: lambda: rt._apple_gpu_dispatch_batched_attention(_GQA_Q, _GQA_Q, _GQA_Q, np),
          declined=lambda r: r is None),
    _site("control_flow.cf_scan",
          lambda mp, sym: _install(mp, _apple_gpu_cf_scan_f32=lambda: sym),
          lambda sym: lambda: rt.apple_gpu_cf_scan(np=np, **_SCAN),
          declined=lambda r: np.allclose(r, _scan_reference())),
    _site("control_flow.cf_while_generate",
          lambda mp, sym: _install(mp, _apple_gpu_cf_while_generate_f32=lambda: sym),
          lambda sym: lambda: rt.apple_gpu_cf_while_generate(
              np.eye(2, dtype=np.float32), np.ones((2, 3), np.float32), np.ones(2, np.float32),
              0, 1, 3, 2, 3, np),
          declined=lambda r: r == ([0, 0, 0], 3)),
    _site("metal4.mtl4_matmul2d_f16",
          lambda mp, sym: _install(mp, _apple_gpu_mtl4_matmul2d_f16_sym=lambda: sym),
          lambda sym: lambda: rt.apple_gpu_mtl4_matmul2d_f16(
              np.ones((2, 3), np.float16), np.ones((3, 2), np.float16), np),
          declined=lambda r: r[1] is False and np.allclose(r[0], 3.0)),
    _site("metal4.mtl4_conv2d_f16",
          lambda mp, sym: _install(mp, _apple_gpu_mtl4_conv2d_sym=lambda dtype: sym,
                                   _apple_gpu_mtl4_matmul2d_sym=lambda name, fused: None),
          lambda sym: lambda: rt.apple_gpu_conv2d(
              np.ones((1, 3, 3, 1), np.float16), np.ones((1, 1, 1, 2), np.float16), np),
          declined=lambda r: r[1] is False and r[0].shape == (1, 3, 3, 2)),
    _site("metal4.mlp_session.run",
          lambda mp, sym: _install(mp, _apple_gpu_mtl4_mlp_session_create_sym=lambda: (lambda *a: 1),
                                   _apple_gpu_mtl4_mlp_session_run_sym=lambda: sym,
                                   _apple_gpu_mtl4_mlp_session_destroy_sym=lambda: (lambda h: None)),
          _mlp_session_run,
          declined=lambda r: r.shape == (4, 3) and np.allclose(r, 2.0)),
    _site("msl_spec_accept",
          lambda mp, sym: _install(mp, _load_apple_gpu_runtime=lambda: _Runtime(tessera_apple_gpu_msl_spec_accept=sym)),
          lambda sym: lambda: rt.apple_gpu_msl_spec_accept(
              np.asarray([[1, 2]], np.int32), np.asarray([[1, 2, 3]], np.int32), np),
          declined=lambda r: r == (0, 2, 3, [1, 2])),
    _site("value_probe.ebm_energy_quadratic",
          lambda mp, sym: _install(mp, _apple_gpu_ebm_energy_quadratic_value_f32=lambda: sym,
                                   _APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE=None),
          lambda sym: lambda: rt._apple_gpu_ebm_energy_quadratic_value_available(),
          declined=lambda r: r is False, once=True),
    _site("value_lane.ppo_policy_loss",
          lambda mp, sym: _install(mp, _apple_gpu_ppo_policy_loss_f32=lambda: sym,
                                   _apple_gpu_ppo_policy_loss_available=lambda: True),
          lambda sym: _catching(lambda: rt._dispatch_gpu_ppo_policy_loss(
              [np.zeros(3, np.float32)] * 3,
              {"symbol": "tessera_apple_gpu_ppo_policy_loss_f32", "clip_epsilon": 0.2}, np)),
          declined=lambda r: r is _RAISED),
    _site("value_lane.batched_matmul(void)",
          lambda mp, sym: mp.setitem(rt._APPLE_VALUE_GPU_DISPATCH, "tessera_apple_gpu_bmm_f32", (lambda: sym, "f32")),
          lambda sym: _catching(lambda: rt._dispatch_gpu_batched_matmul(
              [np.ones((1, 2, 3), np.float32), np.ones((1, 3, 2), np.float32)],
              {"symbol": "tessera_apple_gpu_bmm_f32"}, np)),
          declined=lambda r: r is _RAISED, fail_rc=None),
]


@pytest.fixture(autouse=True)
def _probe_caches_reset(monkeypatch):
    """The `*_value_available` probes cache per process; a test must never
    inherit another's answer.

    They are also gated on `_apple_value_compile_pipeline_available`, which
    runs a real canonical compile: it is False on a host without the Apple
    value pipeline (so the probe would never reach the symbol under test, and
    these tests would pass vacuously on Linux while failing on the Mac) and
    costs ~30 s on a host where it is True. Stubbing it keeps this file
    host-free and puts the breaker, not the compile seam, under test.
    """
    monkeypatch.setattr(rt, "_apple_value_compile_pipeline_available", lambda: True)
    monkeypatch.setattr(rt, "_APPLE_VALUE_COMPILE_PIPELINE_OK", True)
    for name in dir(rt):
        if name.startswith("_APPLE_GPU_") and name.endswith("_AVAILABLE"):
            monkeypatch.setattr(rt, name, None)


@pytest.mark.parametrize("install,invoke,declined,ok_rc,fail_rc,once", DIRECT_SITES)
def test_direct_site_does_not_ask_an_open_breaker(install, invoke, declined, ok_rc, fail_rc, once, monkeypatch):
    sym = _Sym(rc=ok_rc)
    install(monkeypatch, sym)
    _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True

    result = invoke(sym)()
    assert declined(result), f"an open breaker must hand the caller to its host path, got {result!r}"
    assert sym.calls == 0, "breaker open but the site still dispatched"


@pytest.mark.parametrize("install,invoke,declined,ok_rc,fail_rc,once", DIRECT_SITES)
def test_direct_site_timeouts_count_toward_the_streak(install, invoke, declined, ok_rc, fail_rc, once, monkeypatch):
    """A reported kind-1 timeout is one streak entry per call, whatever rc the
    symbol hands back; the LIMIT-th opens the breaker and the next call must
    not pay another timeout."""
    sym = _Sym(rc=fail_rc if fail_rc is not None else ok_rc)
    install(monkeypatch, sym)
    _Channel(TIMEOUT).install(monkeypatch)
    if once:
        assert declined(invoke(sym)())
        assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 1
        assert sym.calls == 1
        return
    for index in range(LIMIT):
        assert declined(invoke(sym)())
        assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == index + 1
    assert sym.calls == LIMIT
    assert rt.apple_gpu_dispatch_breaker_state()["open"] is True
    assert declined(invoke(sym)())
    assert sym.calls == LIMIT, "the (LIMIT+1)-th call still paid a timeout"


@pytest.mark.parametrize("install,invoke,declined,ok_rc,fail_rc,once", DIRECT_SITES)
def test_direct_site_validation_decline_is_not_a_timeout(install, invoke, declined, ok_rc, fail_rc, once, monkeypatch):
    """A failure rc with no error kind is the C side declining the call. It is
    not evidence about the device and keeps the pre-routing behaviour."""
    if fail_rc is None:
        pytest.skip("a void symbol cannot decline; failure reaches Python only through the error channel")
    sym = _Sym(rc=fail_rc)
    install(monkeypatch, sym)
    _Channel(0, detail=None).install(monkeypatch)
    calls = 1 if once else LIMIT + 1
    for _ in range(calls):
        assert declined(invoke(sym)())
    assert sym.calls == calls
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0


@pytest.mark.parametrize("install,invoke,declined,ok_rc,fail_rc,once", DIRECT_SITES)
def test_direct_site_success_closes_the_streak(install, invoke, declined, ok_rc, fail_rc, once, monkeypatch):
    sym = _Sym(rc=ok_rc)
    install(monkeypatch, sym)
    channel = _Channel(TIMEOUT).install(monkeypatch)
    rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 1
    channel.detail = None
    invoke(sym)()
    assert sym.calls == 1
    assert rt.apple_gpu_dispatch_breaker_state()["consecutive_timeouts"] == 0


def test_linalg_success_rc_is_zero_not_one(monkeypatch):
    """The MPS single-matrix lanes report success as rc 0; routing them must
    not flip that (a 1 from cholesky_f32 is a failure)."""
    _install(monkeypatch, _apple_gpu_linalg_sym=lambda name, argtypes: _Sym(rc=0))
    _Channel(0, detail=None).install(monkeypatch)
    assert rt._apple_gpu_chol_2d(np.eye(2, dtype=np.float32), np) is not None
    _install(monkeypatch, _apple_gpu_linalg_sym=lambda name, argtypes: _Sym(rc=1))
    assert rt._apple_gpu_chol_2d(np.eye(2, dtype=np.float32), np) is None


def test_open_breaker_does_not_cache_a_value_probe_as_unavailable(monkeypatch):
    """A `*_value_available` probe caches per process. Answering False while
    the breaker is open must not be remembered, or the lane stays disabled
    after `reset_apple_gpu_dispatch_breaker()` says the device is back."""
    sym = _Sym(rc=1)
    _install(monkeypatch, _apple_gpu_ebm_energy_quadratic_value_f32=lambda: sym)
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt._apple_gpu_ebm_energy_quadratic_value_available() is False
    assert rt._APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE is None, "cached while the breaker was open"
    assert sym.calls == 0

    rt.reset_apple_gpu_dispatch_breaker()
    channel.detail = None
    # The fake symbol writes nothing, so the numerical check fails: the probe
    # ran (one dispatch) and its verdict is now a cached, honest False.
    assert rt._apple_gpu_ebm_energy_quadratic_value_available() is False
    assert sym.calls == 1
    assert rt._APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE is False


def test_the_escape_hatch_also_reopens_the_value_probes(monkeypatch):
    """`TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER` is read per call, so setting it
    after a streak has opened the breaker makes every lane dispatch again. A
    probe that consulted only the raw open bit would stay unavailable until an
    explicit reset -- which is not the old behaviour the opt-out promises."""
    sym = _Sym(rc=1)
    _install(monkeypatch, _apple_gpu_ebm_energy_quadratic_value_f32=lambda: sym)
    channel = _Channel(TIMEOUT).install(monkeypatch)
    for _ in range(LIMIT):
        rt._apple_gpu_run_checked("warmup", lambda: None, lambda: None)
    assert rt._apple_gpu_ebm_energy_quadratic_value_available() is False
    assert sym.calls == 0, "the breaker is open, so the probe must not dispatch"

    monkeypatch.setenv("TESSERA_APPLE_GPU_NO_DISPATCH_BREAKER", "1")
    channel.detail = None
    rt._APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE = None
    rt._apple_gpu_ebm_energy_quadratic_value_available()
    assert sym.calls == 1, "the opt-out did not reach the probe gate"


def test_a_value_probe_timeout_raises_under_strict_dispatch(monkeypatch):
    """The probes wrap their dispatch in `except Exception`; a strict-dispatch
    error from the breaker must escape that, not be cached as `False`."""
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    _install(monkeypatch, _apple_gpu_ebm_energy_quadratic_value_f32=lambda: _Sym(rc=0),
             _apple_gpu_arm_gpu_error=lambda: None,
             _apple_gpu_peek_gpu_error_kind=lambda: TIMEOUT,
             _apple_gpu_consume_gpu_error=lambda: "did not signal")
    with pytest.raises(rt.TesseraStrictDispatchError, match="ebm_energy_quadratic_value.probe"):
        rt._apple_gpu_ebm_energy_quadratic_value_available()
    assert rt._APPLE_GPU_EBM_ENERGY_QUADRATIC_AVAILABLE is None


def test_direct_timeout_is_a_named_fallback_not_a_silent_none(monkeypatch):
    _install(monkeypatch, _apple_gpu_linalg_sym=lambda name, argtypes: _Sym(rc=1),
             _apple_gpu_arm_gpu_error=lambda: None,
             _apple_gpu_peek_gpu_error_kind=lambda: TIMEOUT,
             _apple_gpu_consume_gpu_error=lambda: "did not signal")
    log = []
    monkeypatch.setattr(rt, "_note_dispatch_fallback", lambda op, reason, exc=None: log.append((op, reason)))
    assert rt._apple_gpu_chol_2d(np.eye(2, dtype=np.float32), np) is None
    assert log == [("apple_gpu.cholesky", log[0][1])] and "did not signal" in log[0][1]


# ── The gate over the whole class ────────────────────────────────────────────

BREAKER_HELPERS = {"_apple_gpu_run_checked", "_apple_gpu_device_call_checked", "_apple_gpu_symbol_call_checked"}
APPLE_SYMBOL_PREFIX = "tessera_apple_gpu"
_MM = Path(rt.__file__).resolve().parents[2] / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
# Any of these in a symbol's (transitive) body means it blocks on the device:
# the timed choke points, MPSGraph's untimed synchronous run, and the Metal 4
# shared-event wait.
_WAIT_TOKENS = re.compile(
    r"commit_and_wait_with_timeout|commit_mpsgraph_and_wait_with_timeout|runWithMTLCommandQueue"
    r"|waitUntilCompleted\]|waitUntilSignaledValue|encodeToCommandQueue")
_MM_DEF = re.compile(r'^(?:extern "C" )?(?:static )?(?:inline )?(?:[\w:<>\*&]+ )+\*?(\w+)\(')
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Sites still calling a symbol directly, found by the same enumeration and
# recorded as the third instance in docs/audit/backend/apple/todo.md. Each
# entry must currently be an offender; routing one means deleting its line.
KNOWN_UNROUTED: dict[str, str] = {
    # Empty, and that is the claim: every function that dispatches an Apple GPU
    # symbol which can block on the device routes it through the breaker. The
    # three entries this once held were not routed but reclassified, each by a
    # rule rather than by assertion -- two encode-only `_enc` entries whose
    # shared helper waits only on its nil arm, and a one-symbol wrapper whose
    # callers pass literals. The mechanism stays so a future exception must
    # still be named, and the gate below fails a line that stops being an
    # offender.
}


def _mm_signatures(text):
    """{function: [parameter names]} for every line-anchored definition.

    `_MM_DEF` anchors at a line start, so this walks lines the way
    :func:`_mm_bodies` does rather than scanning the whole text.
    """
    signatures = {}
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    for number, line in enumerate(lines):
        match = _MM_DEF.match(line)
        if not match:
            continue
        i, depth = offsets[number] + match.end() - 1, 0
        while i < len(text):
            depth += (text[i] == "(") - (text[i] == ")")
            if depth == 0:
                break
            i += 1
        params = []
        for part in text[offsets[number] + match.end():i].split(","):
            declaration = part.strip()
            names = re.findall(r"([A-Za-z_]\w*)\s*(?:\[\s*\])?\s*$", declaration)
            params.append((names[0] if names else "", declaration))
        signatures.setdefault(match.group(1), params)
    return signatures


def _guarded_wait_params(bodies, signatures):
    """{function: {parameter index whose NIL value is what reaches the wait}}.

    The encode-or-run helpers take a command buffer and branch on it:

        if (cb) [g encodeToCommandBuffer:cb ...];
        else    [g runWithMTLCommandQueue:ctx.queue ...];

    Only the else arm waits, so whether the caller waits depends on the
    argument it passes -- `nil` waits, a session's buffer does not. Classifying
    such a helper as "waits" unconditionally marks every encode-only entry
    point as a hang risk; classifying it as "does not" hides the synchronous
    one. Neither is true of the helper: it is true of the call.
    """
    guarded = {}
    for name, variants in bodies.items():
        params = signatures.get(name) or []
        # ONLY a command-buffer parameter selects encode-vs-run. An earlier
        # version accepted any parameter and wrongly exempted the int4 matmul
        # lanes, whose `tiled` flag also guards an if/else where only one arm
        # happens to hold a wait token within the window searched.
        buffers = {index for index, (_, declaration) in enumerate(params)
                   if "CommandBuffer" in declaration}
        if not buffers:
            continue
        names = [name_ for name_, _ in params]
        for body in variants:
            for match in re.finditer(r"\bif\s*\(\s*(\w+)\s*\)", body):
                guard = match.group(1)
                if guard not in names or names.index(guard) not in buffers:
                    continue
                tail = body[match.end():]
                split = re.search(r"\belse\b", tail)
                if not split:
                    continue
                then_arm, else_arm = tail[:split.start()], tail[split.end():split.end() + 400]
                if _WAIT_TOKENS.search(else_arm) and not _WAIT_TOKENS.search(then_arm[:400]):
                    guarded.setdefault(name, set()).add(names.index(guard))
    return guarded


def _call_arguments(body, callee):
    """Every argument list `callee(...)` is called with in `body`."""
    calls = []
    for match in re.finditer(r"\b" + re.escape(callee) + r"\s*\(", body):
        i, depth = match.end() - 1, 0
        while i < len(body):
            depth += (body[i] == "(") - (body[i] == ")")
            if depth == 0:
                break
            i += 1
        inner, args, depth = body[match.end():i], [], 0
        current = ""
        for ch in inner:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        calls.append(args)
    return calls


def _mm_bodies(text):
    """{function: [body]} for every line-anchored definition in the .mm."""
    bodies = {}
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    for number, line in enumerate(lines):
        match = _MM_DEF.match(line)
        if not match or match.group(1) in {"if", "for", "while", "switch", "return", "sizeof", "catch"}:
            continue
        i, depth = offsets[number] + match.end() - 1, 0
        while i < len(text):
            depth += (text[i] == "(") - (text[i] == ")")
            if depth == 0:
                break
            i += 1
        j = i + 1
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        if j >= len(text) or text[j] != "{":
            continue
        k, depth = j, 0
        while k < len(text):
            depth += (text[k] == "{") - (text[k] == "}")
            if depth == 0:
                break
            k += 1
        bodies.setdefault(match.group(1), []).append(text[j:k + 1])
    return bodies


def _symbol_waits(name, bodies, memo, stack=(), guarded=None):
    """Does `name`'s body, or anything it calls, block on the device?

    `guarded` carries the argument-sensitive helpers from
    :func:`_guarded_wait_params`: a call into one of those waits only when it
    passes `nil` for the guard parameter, so the same helper is a hang risk for
    the synchronous entry point and not for the encode-only one.
    """
    if name in memo:
        return memo[name]
    if name not in bodies:
        return None
    if name in stack:
        return False
    guarded = guarded or {}
    result = any(_WAIT_TOKENS.search(body) for body in bodies[name])
    if not result:
        for body in bodies[name]:
            for callee in sorted(set(_CALL.findall(body))):
                if callee not in bodies or callee == name:
                    continue
                if not _symbol_waits(callee, bodies, memo, stack + (name,), guarded):
                    continue
                indices = guarded.get(callee)
                if indices:
                    # Reached only on the nil arm: does THIS caller take it?
                    passes_nil = any(
                        args[i].strip() in ("nil", "NULL", "nullptr", "0")
                        for args in _call_arguments(body, callee)
                        for i in indices if i < len(args))
                    if not passes_nil:
                        continue
                result = True
                break
            if result:
                break
    memo[name] = result
    return result


def _constant(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _bound_symbol(value, params, *, loads_runtime=True):
    """The Apple symbol an assignment's right-hand side names.

    Two forms reach the same place: `getattr(runtime, "tessera_apple_gpu_x")`
    and the plain attribute `runtime.tessera_apple_gpu_x`. Missing the second
    is not academic -- it hid `_apple_gpu_rope_f32` and its siblings from an
    earlier version of this gate, and with them the f32 half of every lane that
    binds its symbol that way.
    """
    if isinstance(value, ast.Attribute) and value.attr.startswith(APPLE_SYMBOL_PREFIX):
        return value.attr
    if not isinstance(value, ast.Call):
        return None
    return _getattr_symbol(value, params, loads_runtime=loads_runtime)


def _getattr_symbol(call, params, *, loads_runtime=True):
    """The Apple symbol a `getattr(x, name)` names: a constant, an f-string
    pattern, or `<param:i>` for a parameter of the enclosing function.

    A parameter is only read as a symbol name when the enclosing function also
    loads the Apple GPU runtime -- otherwise `getattr(module, op_name)` on a
    numpy reference module would be mistaken for a device dispatch."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2):
        return None
    if not isinstance(call.args[0], (ast.Name, ast.Attribute)):
        return None
    name = call.args[1]
    if isinstance(name, ast.Name) and not loads_runtime:
        return None
    if isinstance(name, ast.Constant):
        return name.value if isinstance(name.value, str) and name.value.startswith(APPLE_SYMBOL_PREFIX) else None
    if isinstance(name, ast.JoinedStr):
        text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in name.values)
        return text if text.startswith(APPLE_SYMBOL_PREFIX) else None
    if isinstance(name, ast.Name) and name.id in params:
        return f"<param:{params.index(name.id)}>"
    return None


def _direct_symbol_sites():
    """{function: (symbols it dispatches directly, routed?)} over runtime.py.

    An accessor is any function that `getattr`s an Apple symbol and returns a
    name; a symbol bound from an accessor call or from `getattr` directly and
    then *called* is a dispatch. Accessor parameters are resolved from constant
    arguments at the binding site; anything else is `<dynamic>` and treated as
    dispatching (fail closed)."""
    tree = ast.parse(inspect.getsource(rt))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def loads_runtime(fn):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "_load_apple_gpu_runtime" for n in ast.walk(fn))

    accessors = {}
    for fn in functions:
        params, runtime_here = [a.arg for a in fn.args.args], loads_runtime(fn)
        names, bound_here = [], set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                symbol = _bound_symbol(node.value, params, loads_runtime=runtime_here)
                if symbol:
                    names.append(symbol)
                    bound_here.update(t.id for t in node.targets if isinstance(t, ast.Name))
        # An accessor RETURNS the bound symbol. Requiring that (rather than any
        # bare name) keeps a dispatching function that happens to `return out`
        # from being mistaken for one -- which would hide it from this gate.
        returns_symbol = any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id in bound_here
                             for n in ast.walk(fn))
        if names and returns_symbol and fn.name not in BREAKER_HELPERS:
            accessors[fn.name] = names

    # A one-symbol wrapper takes the name as a parameter, so its own body says
    # `<dynamic>` and fails closed. Its CALLERS often pass a literal, and when
    # every one of them does, the set of symbols it can reach is known exactly.
    # `_apple_gpu_raw_handle` is the case in point: two callers, both constant.
    constant_callers = {}
    for fn in functions:
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            literal = _constant(node.args[0]) if node.args else None
            constant_callers.setdefault(node.func.id, []).append(literal)

    def resolve(pattern, call):
        if pattern.startswith("<param:"):
            index = int(pattern[7:-1])
            arg = call.args[index] if index < len(call.args) else None
            return _constant(arg) or "<dynamic>"
        if "{}" in pattern:
            arg = _constant(call.args[0]) if call.args else None
            return pattern.replace("{}", arg) if arg else "<dynamic>"
        return pattern

    def from_callers(fn_name):
        """The symbols a `<dynamic>` wrapper can actually reach, or None when
        any caller passes something this cannot read."""
        literals = constant_callers.get(fn_name)
        if not literals or any(value is None for value in literals):
            return None
        return {value for value in literals if value.startswith(APPLE_SYMBOL_PREFIX)} or None

    sites = {}
    for fn in functions:
        if fn.name in accessors or fn.name in BREAKER_HELPERS:
            continue
        if isinstance(parents.get(fn), (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # a nested wrapper is judged as part of its parent
        params, runtime_here = [a.arg for a in fn.args.args], loads_runtime(fn)
        bound = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            values = ([v for v in node.value.values] if isinstance(node.value, ast.BoolOp)
                      else [node.value])
            symbols = []
            for value in values:
                direct = _bound_symbol(value, params, loads_runtime=runtime_here)
                if direct:
                    symbols.append("<dynamic>" if direct.startswith("<param") else direct)
                elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                      and value.func.id in accessors):
                    symbols.extend(resolve(p, value) for p in accessors[value.func.id])
            if symbols:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound.setdefault(target.id, set()).update(symbols)
        # Routing is judged PER DISPATCH, not per function, for the reason the
        # resident gate above records: a function with one wrapped call and one
        # bare one satisfies a function-wide flag. A dispatch counts as routed
        # when a breaker-helper call is among its own ancestors (the `lambda:
        # sym(...)` form), or when it sits in a nested function whose name is
        # handed to a breaker helper in this scope (the `def _run(): ...` form
        # the multi-statement lanes need).
        wrapped = {a.id for node in ast.walk(fn)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in BREAKER_HELPERS
                   for a in node.args if isinstance(a, ast.Name)}
        # `call = lambda: symbol(...)` handed to a helper is the same shape as a
        # nested def, so a Lambda carries the name it was bound to.
        lambda_names = {node.value: target.id
                        for node in ast.walk(fn) if isinstance(node, ast.Assign)
                        and isinstance(node.value, ast.Lambda)
                        for target in node.targets if isinstance(target, ast.Name)}

        def _routed(node):
            cur = parents.get(node)
            while cur is not None:
                if (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name)
                        and cur.func.id in BREAKER_HELPERS):
                    return True
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if cur is fn:
                        return False
                    if cur.name in wrapped:
                        return True
                elif isinstance(cur, ast.Lambda) and lambda_names.get(cur) in wrapped:
                    return True
                elif isinstance(cur, ast.Module):
                    return False
                cur = parents.get(cur)
            return False

        dispatched, unrouted = set(), set()
        for node in ast.walk(fn):
            symbols = set()
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in bound:
                symbols = bound[node.func.id]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith(APPLE_SYMBOL_PREFIX):
                symbols = {node.func.attr}
            if not symbols:
                continue
            dispatched |= symbols
            if not _routed(node):
                unrouted |= symbols
        # The Apple CPU (Accelerate) lane is a different backend with no
        # command buffer and no breaker; an accessor shared with it can resolve
        # to a `tessera_apple_cpu_*` name, which this gate does not govern.
        keep = lambda names: {s for s in names if s == "<dynamic>" or s.startswith(APPLE_SYMBOL_PREFIX)}  # noqa: E731
        dispatched, unrouted = keep(dispatched), keep(unrouted)
        if "<dynamic>" in dispatched:
            resolved = from_callers(fn.name)
            if resolved:
                dispatched = (dispatched - {"<dynamic>"}) | resolved
                if "<dynamic>" in unrouted:
                    unrouted = (unrouted - {"<dynamic>"}) | resolved
        if dispatched:
            sites[fn.name] = (unrouted or dispatched, not unrouted)
    return sites


@pytest.fixture(scope="module")
def _mm_waits():
    if not _MM.is_file():
        pytest.skip(f"{_MM} not present; the .mm classifier needs the source tree")
    text = _MM.read_text()
    bodies = _mm_bodies(text)
    guarded = _guarded_wait_params(bodies, _mm_signatures(text))
    memo = {}
    symbols = {n for n in bodies if n.startswith(APPLE_SYMBOL_PREFIX)}
    assert len(symbols) > 300, "the .mm parser found too few C symbols; its definition regex has drifted"

    def waits(name):
        return _symbol_waits(name, bodies, memo, guarded=guarded)

    # Self-check the classifier on one known member of each class before
    # trusting it to exempt anything.
    assert waits("tessera_apple_gpu_cholesky_f32") is True       # timed
    assert waits("tessera_apple_gpu_cf_scan_f32") is True        # untimed MPSGraph run
    assert waits("tessera_apple_gpu_mtl4_scan_f32") is True      # Metal 4 shared event
    assert waits("tessera_apple_gpu_metal4_probe") is False      # capability probe
    assert waits("tessera_apple_gpu_clear_last_error") is False  # error channel
    # ... and on the pair that shares ONE helper and splits on its argument.
    assert waits("tessera_apple_gpu_rowop_dev_f32") is True      # passes nil -> runs
    assert waits("tessera_apple_gpu_rowop_dev_f32_enc") is False  # passes s->cb -> encodes
    # A parameter that is not a command buffer must never earn that split: the
    # int4 matmul lanes branch on a `tiled` flag and both arms reach a wait.
    assert waits("tessera_apple_gpu_quantized_matmul_i4_f32") is True
    return waits


def test_the_wait_classifier_splits_one_helper_by_its_argument():
    """`encode_or_run_rowop_dev` waits on the nil arm and encodes on the other,
    so "does this symbol wait" is a property of the CALL, not the helper.
    Classifying the helper either way is wrong for half its callers: as
    waiting, every encode-only entry point reads as a hang risk; as not
    waiting, the synchronous one is hidden.
    """
    text = _MM.read_text()
    bodies = _mm_bodies(text)
    signatures = _mm_signatures(text)
    guarded = _guarded_wait_params(bodies, signatures)
    assert guarded.get("encode_or_run_rowop_dev") == {0}, (
        "the encode-or-run helper's command-buffer guard was not recognised")
    assert all("CommandBuffer" in signatures[name][index][1]
               for name, indices in guarded.items() for index in indices), (
        "only a command-buffer parameter may select encode-vs-run")
    # Without the rule both sides collapse to the same answer; with it they split.
    assert _symbol_waits("tessera_apple_gpu_rowop_dev_f32", bodies, {}) is True
    assert _symbol_waits("tessera_apple_gpu_rowop_dev_f32_enc", bodies, {}) is True
    assert _symbol_waits("tessera_apple_gpu_rowop_dev_f32", bodies, {}, guarded=guarded) is True
    assert _symbol_waits("tessera_apple_gpu_rowop_dev_f32_enc", bodies, {}, guarded=guarded) is False


def test_a_one_symbol_wrapper_is_resolved_from_its_callers():
    """`_apple_gpu_raw_handle` takes its symbol as a parameter, so its own body
    says `<dynamic>` and fails closed. Both callers pass a literal, so the set
    of symbols it can reach is known exactly -- and neither takes a command
    buffer, which is why it is exempt rather than allowlisted."""
    symbols, routed = _direct_symbol_sites()["_apple_gpu_raw_handle"]
    assert symbols == {"tessera_apple_gpu_device_handle", "tessera_apple_gpu_command_queue_handle"}
    assert routed is False, "the wrapper does not dispatch through the breaker, and does not need to"


def test_the_direct_gate_is_per_dispatch_not_per_function():
    """The direct gate inherits the resident gate's per-dispatch rule, and has
    two wrapper shapes to recognise that a plain ancestor walk does not: a
    nested `def` and an assigned `lambda`, both handed to a helper by name.

    Asserted on the real module rather than a synthetic one, because the point
    is that those two shapes — which every multi-statement lane here uses —
    are not mistaken for unrouted, while a bare second dispatch beside them
    still is. `_apple_gpu_dispatch_rope` has three wrapped dispatches and no
    bare one; the three known offenders have bare ones and nothing else.
    """
    sites = _direct_symbol_sites()
    assert sites["_apple_gpu_dispatch_rope"][1] is True, (
        "a `def _run(): sym(...)` passed to the helper by name reads as unrouted")
    assert sites["_execute_apple_compiled_norm_backward"][1] is True, (
        "a `call = lambda: symbol(...)` passed to the helper by name reads as unrouted")
    assert sites["_apple_gpu_raw_handle"][1] is False


def test_every_direct_apple_symbol_dispatch_is_breaker_routed(_mm_waits):
    """Drift gate over the whole class: a function that calls a bound
    `tessera_apple_gpu_*` symbol which can block on the device -- or a symbol
    it cannot resolve statically -- must route it through a breaker helper.
    Symbols that never reach a command-buffer wait (capability probes, the
    error channel, memory statistics, encode-only `_enc` entries whose session
    waits elsewhere) are exempt by the .mm, not by a list."""
    offenders, exempt = {}, {}
    for fn, (symbols, routed) in _direct_symbol_sites().items():
        if routed:
            continue
        blocking = sorted(s for s in symbols if s == "<dynamic>" or _mm_waits(s) is not False)
        if blocking:
            offenders[fn] = blocking
        else:
            exempt[fn] = sorted(symbols)
    assert "apple_gpu_metal4_caps" in exempt and "apple_gpu_metal4_tensor_roundtrip" in exempt, (
        "the Metal 4 probes take no command buffer and must stay exempt by evidence")
    stale = sorted(set(KNOWN_UNROUTED) - set(offenders))
    assert stale == [], f"KNOWN_UNROUTED lists functions that are routed or gone; delete them: {stale}"
    new = {fn: syms for fn, syms in offenders.items() if fn not in KNOWN_UNROUTED}
    assert new == {}, "direct dispatch(es) bypass the breaker: " + ", ".join(f"{fn} -> {syms}" for fn, syms in sorted(new.items()))
