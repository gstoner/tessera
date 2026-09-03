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
