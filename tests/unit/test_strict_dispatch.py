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
    """With the structural re-matcher disabled, fusion_groups metadata alone
    must route the program to the fused dispatcher — proof the executor
    consumes compiler intent rather than re-matching."""
    sentinel = np.full((2, 2), 42.0, dtype=np.float32)
    monkeypatch.setattr(
        rt, "_apple_gpu_metadata_is_matmul_softmax_chain", lambda ops: False)
    monkeypatch.setattr(
        rt, "_apple_gpu_dispatch_matmul_softmax",
        lambda operands, np_: sentinel)
    md = _matmul_softmax_metadata(with_fusion_groups=True)
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    out = rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b})
    np.testing.assert_array_equal(out, sentinel)


def test_executor_without_fusion_groups_still_uses_rematcher(monkeypatch):
    """Legacy artifacts (no fusion_groups) keep working via the structural
    re-matcher."""
    sentinel = np.full((2, 2), 7.0, dtype=np.float32)
    monkeypatch.setattr(
        rt, "_apple_gpu_dispatch_matmul_softmax",
        lambda operands, np_: sentinel)
    md = _matmul_softmax_metadata(with_fusion_groups=False)
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    out = rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b})
    np.testing.assert_array_equal(out, sentinel)


def test_executor_dispatches_swiglu_from_fusion_groups(monkeypatch):
    """SwiGLU follow-on (2026-06-10): the DAG chain is now derived by
    canonical_compile._match_swiglu_at, and the executor consumes it —
    with the structural re-matcher disabled, the fusion_groups entry alone
    routes the 4-op program to the fused swiglu dispatcher."""
    sentinel = np.full((2, 4), 9.0, dtype=np.float32)
    monkeypatch.setattr(
        rt, "_apple_gpu_metadata_is_swiglu_chain", lambda ops: False)
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


def test_executor_fusion_groups_short_circuits_rematcher(monkeypatch):
    """When fusion_groups names the chain, the structural re-matcher is not
    even consulted (the `or` short-circuits)."""
    calls = {"rematch": 0}

    def counting_rematcher(ops):
        calls["rematch"] += 1
        return True

    monkeypatch.setattr(
        rt, "_apple_gpu_metadata_is_matmul_softmax_chain", counting_rematcher)
    monkeypatch.setattr(
        rt, "_apple_gpu_dispatch_matmul_softmax",
        lambda operands, np_: np.zeros((2, 2), dtype=np.float32))
    md = _matmul_softmax_metadata(with_fusion_groups=True)
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    rt._execute_apple_gpu_mps_metadata(md, {"a": a, "b": b})
    assert calls["rematch"] == 0
