"""Strict-dispatch mode + fallback funnel + fusion_groups consumption.

Audit 2026-06-10 (docs/audit/compiler/CODE_AUDIT_2026_06_10.md):

* Finding #5 — failure-class GPU→numpy fallbacks used to be silent, so a
  regressed Metal kernel degraded to the numpy oracle and every numerical
  test still passed. ``TESSERA_STRICT_DISPATCH=1`` makes those fallbacks
  raise ``TesseraStrictDispatchError``; without it they are recorded in
  ``dispatch_fallback_log()`` so suites can assert "zero silent fallbacks".
  Envelope-limit fallbacks (shape/dtype outside a kernel's documented range)
  are NOT failures and stay silent.

* Finding #8 — the apple_gpu executor now consumes the compiler's
  ``fusion_groups`` known_chain metadata (canonical_compile, 2026-06-07)
  instead of only re-matching chain patterns per invoke. The structural
  re-matchers remain as fallback for legacy artifacts.

All tests are host-independent: GPU symbol getters are monkeypatched, so
nothing here needs Metal.
"""

import numpy as np
import pytest

import tessera.runtime as rt


@pytest.fixture(autouse=True)
def _clean_fallback_log(monkeypatch):
    monkeypatch.delenv("TESSERA_STRICT_DISPATCH", raising=False)
    rt.reset_dispatch_fallback_log()
    yield
    rt.reset_dispatch_fallback_log()


# ---------------------------------------------------------------------------
# The funnel itself
# ---------------------------------------------------------------------------

def test_fallback_is_logged_when_strict_mode_off():
    rt._note_dispatch_fallback("tessera.example", "synthetic reason")
    assert ("tessera.example", "synthetic reason") in rt.dispatch_fallback_log()


def test_strict_mode_raises_with_op_and_reason(monkeypatch):
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError) as exc_info:
        rt._note_dispatch_fallback("tessera.example", "synthetic reason")
    assert "tessera.example" in str(exc_info.value)
    assert "synthetic reason" in str(exc_info.value)


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_strict_mode_disabled_values(monkeypatch, value):
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", value)
    assert not rt._strict_dispatch_enabled()


def test_reset_clears_the_log():
    rt._note_dispatch_fallback("tessera.example", "synthetic reason")
    rt.reset_dispatch_fallback_log()
    assert rt.dispatch_fallback_log() == []


# ---------------------------------------------------------------------------
# Real dispatch sites route through the funnel
# ---------------------------------------------------------------------------

def test_unary_symbol_missing_logs_fallback(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_unary_f32", lambda: None)
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    out = rt._apple_gpu_dispatch_unary("tessera.relu", [x], np)
    np.testing.assert_allclose(out, np.maximum(x, 0.0))
    assert any(op == "tessera.relu" for op, _ in rt.dispatch_fallback_log())


def test_unary_symbol_missing_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_unary_f32", lambda: None)
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_dispatch_unary("tessera.relu", [x], np)


def test_bmm_lane_unavailable_logs_fallback(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_dispatch_bmm", lambda a, b, np_: None)
    a = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((2, 4, 5)).astype(np.float32)
    out = rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
    np.testing.assert_allclose(out, np.matmul(a, b), rtol=1e-6)
    assert any(op == "tessera.matmul" for op, _ in rt.dispatch_fallback_log())


def test_bmm_lane_unavailable_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    monkeypatch.setattr(rt, "_apple_gpu_dispatch_bmm", lambda a, b, np_: None)
    a = np.zeros((2, 3, 4), dtype=np.float32)
    b = np.zeros((2, 4, 5), dtype=np.float32)
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)


def test_gpu_matmul_error_channel_funnels_and_recomputes(monkeypatch):
    """1d #3 — when the GEMM C-ABI symbol exists and is called but the GPU
    reports an internal failure (timeout/device-lost/cb.error) via the
    last-error channel, the matmul lane funnels (strict raises) + recomputes
    on host instead of returning the untouched (garbage) output buffer.

    Simulated by stubbing the channel consumer to report an error; the GEMM
    call itself is stubbed to a no-op so the path is host-independent."""
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    monkeypatch.setattr(rt, "_apple_gpu_mps_matmul_f32", lambda: (lambda *args: None))
    monkeypatch.setattr(rt, "_mtl4_route_matmul_f32", lambda a_, b_, np_: None)
    monkeypatch.setattr(rt, "_apple_gpu_gemm2d_call", lambda *a_, **k_: None)
    monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
    monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error",
                        lambda: "tessera.matmul: GPU dispatch did not signal")
    out = rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
    # host recompute, not the (stubbed no-op) zeros buffer
    np.testing.assert_allclose(out, a @ b)
    assert any(op == "tessera.matmul" for op, _ in rt.dispatch_fallback_log())
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)


def test_run_checked_funnels_on_error_and_recomputes(monkeypatch):
    """The shared _apple_gpu_run_checked wrapper (unary/binary/rowop/bmm lanes):
    on a reported GPU error it funnels (strict raises) and returns the host
    fallback instead of the kernel result."""
    monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
    monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: "boom")
    out = rt._apple_gpu_run_checked(
        "tessera.silu", lambda: "GPU-RESULT", lambda: "HOST-RESULT")
    assert out == "HOST-RESULT"
    assert any(op == "tessera.silu" for op, _ in rt.dispatch_fallback_log())
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_run_checked(
            "tessera.silu", lambda: "GPU-RESULT", lambda: "HOST-RESULT")


def test_run_checked_passthrough_on_success(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_arm_gpu_error", lambda: None)
    monkeypatch.setattr(rt, "_apple_gpu_consume_gpu_error", lambda: None)
    out = rt._apple_gpu_run_checked(
        "tessera.silu", lambda: "GPU-RESULT", lambda: "HOST-RESULT")
    assert out == "GPU-RESULT"
    assert rt.dispatch_fallback_log() == []


def test_gpu_error_channel_helpers_noop_without_symbols(monkeypatch):
    # Older runtime build / non-Darwin stub: consumer returns None (no error),
    # arm is a no-op. Must not raise even in strict mode.
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    class _Lib:  # a runtime image missing the channel symbols
        pass
    monkeypatch.setattr(rt, "_load_apple_gpu_runtime", lambda: _Lib())
    rt._apple_gpu_arm_gpu_error()                      # no-op, no raise
    assert rt._apple_gpu_consume_gpu_error() is None   # no channel → no error


def test_binary_symbol_missing_logs_and_raises(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_binary_f32", lambda: None)
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    out = rt._apple_gpu_dispatch_mpsgraph_binary("tessera.add", [a, b], {}, np)
    np.testing.assert_allclose(out, a + b)
    assert any(op == "tessera.add" for op, _ in rt.dispatch_fallback_log())
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_dispatch_mpsgraph_binary("tessera.add", [a, b], {}, np)


def test_binary_empty_array_is_not_a_failure(monkeypatch):
    # n == 0 is a degenerate shape, not a symbol failure — must not funnel
    # even when the symbol is missing.
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_binary_f32", lambda: None)
    a = np.zeros((0,), dtype=np.float32)
    b = np.zeros((0,), dtype=np.float32)
    rt._apple_gpu_dispatch_mpsgraph_binary("tessera.add", [a, b], {}, np)
    assert rt.dispatch_fallback_log() == []


def test_rowop_symbol_missing_logs_and_raises(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_rmsnorm_gpu_f32", lambda: None)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = rt._apple_gpu_dispatch_rowop("tessera.rmsnorm", [x], {}, np)
    assert out.shape == x.shape
    assert any(op == "tessera.rmsnorm" for op, _ in rt.dispatch_fallback_log())
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    with pytest.raises(rt.TesseraStrictDispatchError):
        rt._apple_gpu_dispatch_rowop("tessera.rmsnorm", [x], {}, np)


def test_envelope_miss_is_not_a_failure_fallback():
    # Rank-2 shape mismatch is an envelope/shape condition handled by numpy —
    # it must NOT be recorded as a failure-class fallback.
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((5, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        np.matmul(a, b)  # sanity: genuinely incompatible
    # The dispatcher routes incompatible rank-2 shapes straight to np.matmul,
    # which raises — and nothing lands in the failure log on the way.
    with pytest.raises(ValueError):
        rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)
    assert rt.dispatch_fallback_log() == []


# ---------------------------------------------------------------------------
# fusion_groups consumption (finding #8)
# ---------------------------------------------------------------------------

def _matmul_softmax_metadata(with_fusion_groups: bool) -> dict:
    # Result names are stored without the SSA "%" sigil; operand references
    # may carry it (the matchers strip it) — mirrors GraphIR metadata shape.
    metadata = {
        "arg_names": ["a", "b"],
        "output_name": "t1",
        "ops": [
            {"op_name": "tessera.matmul", "operands": ["a", "b"],
             "result": "t0", "kwargs": {}},
            {"op_name": "tessera.softmax", "operands": ["%t0"],
             "result": "t1", "kwargs": {}},
        ],
    }
    if with_fusion_groups:
        metadata["fusion_groups"] = [{
            "function": "f",
            "kind": "known_chain",
            "status": "candidate",
            "fused_kernel": "matmul_softmax",
            "ops": [{"index": 0, "op": "matmul"},
                    {"index": 1, "op": "softmax"}],
        }]
    return metadata


def test_known_chain_helper_reads_whole_program_groups():
    md = _matmul_softmax_metadata(with_fusion_groups=True)
    assert rt._apple_gpu_known_chain_from_fusion_groups(md, 2) == "matmul_softmax"
    # A group that does not cover the whole program must not match.
    assert rt._apple_gpu_known_chain_from_fusion_groups(md, 3) is None
    md_none = _matmul_softmax_metadata(with_fusion_groups=False)
    assert rt._apple_gpu_known_chain_from_fusion_groups(md_none, 2) is None


def test_executor_dispatches_fused_kernel_from_fusion_groups(monkeypatch):
    """Bare `fusion_groups` that names the chain (no `dispatch` roles) routes
    the program to the fused dispatcher via the `fused_kernel == "X"` fallback
    branch — proof the executor consumes compiler intent rather than
    re-matching (the structural re-matcher was deleted in Phase 0c)."""
    sentinel = np.full((2, 2), 42.0, dtype=np.float32)
    monkeypatch.setattr(
        rt, "_apple_gpu_dispatch_matmul_softmax",
        lambda operands, np_: sentinel)
    md = _matmul_softmax_metadata(with_fusion_groups=True)
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    out = rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b})
    np.testing.assert_array_equal(out, sentinel)


def test_executor_without_fusion_groups_runs_per_op():
    """Phase 0c: the structural re-matchers were deleted, so a truly-legacy
    artifact with no fusion_groups no longer takes a fused fast-path — it runs
    correctly per-op (the driver always emits fusion_groups today, so this only
    affects hand-built metadata). Fusion is now a carried compiler decision, not
    something the executor re-discovers."""
    md = _matmul_softmax_metadata(with_fusion_groups=False)
    assert rt._apple_gpu_resolve_authoritative_plan(md, len(md["ops"])) is None
    a = np.random.default_rng(0).standard_normal((2, 3)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    out = np.asarray(rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b}))
    sm = np.exp(a @ b - (a @ b).max(-1, keepdims=True))
    sm /= sm.sum(-1, keepdims=True)
    np.testing.assert_allclose(out, sm, rtol=1e-5, atol=1e-5)


def test_executor_dispatches_swiglu_from_fusion_groups(monkeypatch):
    """SwiGLU follow-on (2026-06-10): the DAG chain is derived by
    canonical_compile._match_swiglu_at, and the executor consumes it — the
    bare fusion_groups entry alone routes the 4-op program to the fused swiglu
    dispatcher via the `fused_kernel == "swiglu"` branch (Phase 0c deleted the
    structural re-matcher)."""
    sentinel = np.full((2, 4), 9.0, dtype=np.float32)
    monkeypatch.setattr(
        rt, "_apple_gpu_dispatch_swiglu",
        lambda x, wg, wu, wd, np_: sentinel)
    metadata = {
        "arg_names": ["x", "wg", "wu", "wd"],
        "output_name": "t3",
        "ops": [
            {"op_name": "tessera.matmul", "operands": ["x", "wg"],
             "result": "t0", "kwargs": {}},
            {"op_name": "tessera.matmul", "operands": ["x", "wu"],
             "result": "t1", "kwargs": {}},
            {"op_name": "tessera.silu_mul", "operands": ["%t0", "%t1"],
             "result": "t2", "kwargs": {}},
            {"op_name": "tessera.matmul", "operands": ["%t2", "wd"],
             "result": "t3", "kwargs": {}},
        ],
        "fusion_groups": [{
            "function": "f",
            "kind": "known_chain",
            "status": "candidate",
            "fused_kernel": "swiglu",
            "ops": [{"index": 0, "op": "matmul"},
                    {"index": 1, "op": "matmul"},
                    {"index": 2, "op": "silu_mul"},
                    {"index": 3, "op": "matmul"}],
        }],
    }
    x = np.zeros((2, 3), dtype=np.float32)
    wg = np.zeros((3, 8), dtype=np.float32)
    wu = np.zeros((3, 8), dtype=np.float32)
    wd = np.zeros((8, 4), dtype=np.float32)
    out = rt._execute_apple_gpu_mps_metadata(
        metadata, {"x": x, "wg": wg, "wu": wu, "wd": wd})
    np.testing.assert_array_equal(out, sentinel)


def test_executor_authoritative_dispatch_uses_roles(monkeypatch):
    """Phase 0b: a known_chain group carrying `dispatch` roles routes through
    the authoritative path — the fused dispatcher is called with operands bound
    from the roles, and the `fused_kernel`/cascade fallback is never reached."""
    captured = {}

    def fake_dispatch(operands, np_):
        captured["operands"] = [np.asarray(o).copy() for o in operands]
        return np.zeros((2, 2), dtype=np.float32)

    monkeypatch.setattr(rt, "_apple_gpu_dispatch_matmul_softmax", fake_dispatch)
    md = _matmul_softmax_metadata(with_fusion_groups=True)
    # Attach 0a dispatch roles so the authoritative resolver fires.
    md["fusion_groups"][0]["dispatch"] = {"a": "a", "b": "b", "out": "t1"}
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b})
    np.testing.assert_array_equal(out, np.zeros((2, 2), dtype=np.float32))
    # The authoritative path resolved operands a, b from the roles.
    assert len(captured["operands"]) == 2
    np.testing.assert_array_equal(captured["operands"][0], a)
    np.testing.assert_array_equal(captured["operands"][1], b)
