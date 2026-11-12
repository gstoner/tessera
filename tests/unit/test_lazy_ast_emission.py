"""The decoration-time AST Graph IR is built only when something reads it.

E2E-REAL-6F: a tracer-authority call replaces ``graph_ir`` and the executed
plan with its own, so it never needs the AST module. The decorator defers the
AST emission wherever no decoration-time consumer reads it; the first reader of
``graph_ir`` (or any field derived from it) runs the emission, and a call that
is a legacy candidate is such a reader.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import tessera
import tessera as ts
from tessera.compiler.jit import TesseraJitError


def _body(x, y):
    return tessera.ops.relu(tessera.ops.add(x, y))


def _x() -> np.ndarray:
    return np.array([-1.0, 2.0, 0.5, -3.0], dtype=np.float32)


def _expected(x: np.ndarray) -> np.ndarray:
    return np.maximum(x + x, 0.0)


@pytest.mark.parametrize("target", [None, "cpu", "rocm", "x86", "nvidia_sm120"])
def test_decoration_defers_the_ast_emission(target):
    fn = ts.jit(_body) if target is None else ts.jit(target=target)(_body)
    assert fn.ast_emission_pending
    # The call ABI comes from the signature, not the unbuilt module.
    assert fn.arg_names == ["x", "y"]


def test_a_tracer_authority_call_never_builds_the_ast():
    fn = ts.jit(_body)
    x = _x()
    for _ in range(2):
        np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
        assert fn.frontend_authority == "tracer"
        assert fn.executed_plan_frontend == "tracer"
        assert fn.last_fallback_reason is None
    assert fn.ast_emission_pending
    # The artifact describes the traced plan it ran.
    assert fn.runtime_artifact().metadata["function_name"] == "_body"
    assert fn.ast_emission_pending


def test_a_legacy_candidate_call_builds_it_and_both_plans_keep_running():
    fn = ts.jit(_body)
    x = _x()
    np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
    assert fn.ast_emission_pending
    # A non-tensor signature keeps the AST candidate, which needs the module.
    np.testing.assert_allclose(fn(x.tolist(), x.tolist()), _expected(x), rtol=1e-6)
    assert not fn.ast_emission_pending
    assert fn.frontend_authority == "legacy_candidate_non_tensor_signature"
    assert fn.executed_plan_frontend == "ast"
    # Back to a tensor call: the traced plan is selected again, and the AST
    # plan is restored as the decoration plan rather than lost.
    np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
    assert fn.executed_plan_frontend == "tracer"
    decoration_plan = fn._decoration_compile[0]
    assert decoration_plan is not None and decoration_plan.target_kind == "cpu"


def test_reading_graph_ir_builds_the_ast_module():
    fn = ts.jit(_body)
    module = fn.graph_ir
    assert not fn.ast_emission_pending
    assert [op.op_name for op in module.functions[0].body] == ["tessera.add", "tessera.relu"]
    assert fn._legacy_graph_ir is module
    assert fn.cpu_plan is not None and fn.compile_bundle is not None


@pytest.mark.parametrize("decorate", [
    lambda f: ts.jit(target="apple_cpu")(f),
    lambda f: ts.jit(target="apple_gpu")(f),
    lambda f: ts.jit(autodiff="reverse", wrt=["x"])(f),
])
def test_decoration_time_readers_keep_the_eager_emission(decorate):
    # The Apple targets read the module (tracer gate, package recognizer, the
    # AST-vs-traced executable comparison); a differentiation request writes
    # its intent into it.
    assert not decorate(_body).ast_emission_pending


def _drops_a_statement(x):
    y = _passthrough(x)  # dropped by the AST front end -> dangling operand
    return tessera.ops.relu(y)


def _passthrough(v):
    return v


def test_an_emission_failure_reaches_every_reader():
    fn = ts.jit(_drops_a_statement)
    x = np.array([1.0, -2.0], np.float32)
    np.testing.assert_array_equal(fn(x), np.maximum(x, 0.0))
    # ``graph_ir`` is the traced module now; the decoration plan is the AST's.
    for _ in range(2):
        with pytest.raises(TesseraJitError, match="GRAPH_IR_UNDEFINED_OPERAND"):
            _ = fn._decoration_compile
    # A failed emission leaves the call path intact for tracer calls.
    np.testing.assert_array_equal(fn(x), np.maximum(x, 0.0))



def test_concurrent_first_readers_all_get_the_module():
    # Both readers find the slot unset; the second waits on the lock while the
    # first emits, and must then read the field rather than fail.
    fn = ts.jit(_body)
    emit = fn._ast_emission
    calls = []

    def slow_emit():
        calls.append(1)
        time.sleep(0.05)
        return emit()

    fn._ast_emission = slow_emit
    barrier = threading.Barrier(4)
    results, errors = {}, []

    def read(index, field):
        barrier.wait()
        try:
            results[index] = getattr(fn, field)
        except Exception as exc:  # noqa: BLE001 -- collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(i, f))
               for i, f in enumerate(("graph_ir", "compile_bundle", "graph_ir", "cpu_plan"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 4 and calls == [1]
    assert results[0] is results[2] is fn.graph_ir


# ── Native targets defer only when target capability admits the body ─────────

def _expands(x, s, g):
    return tessera.ops.nvfp4_expand(x, s, g)


def _syntax_only(x, y):
    return x + y


def test_a_native_target_defers_only_what_its_capability_admits():
    from tessera.compiler.jit import _scan_admits_target_capability

    assert _scan_admits_target_capability(_body, "x86", None)
    # x86 has no NVFP4 expansion: the module is built at decoration and its
    # compilation gives the refusal, so nothing is deferred past it.
    assert not _scan_admits_target_capability(_expands, "x86", None)
    with pytest.raises(TesseraJitError, match="LEGALITY_TARGET_CAPABILITY"):
        ts.jit(target="x86")(_expands)


def test_ops_lowered_from_syntax_are_checked(monkeypatch):
    from tessera.compiler import legality
    from tessera.compiler.jit import _scan_admits_target_capability

    real = legality.check_op_legality

    def refuse_add(op_name, *args, **kwargs):
        result = real(op_name, *args, **kwargs)
        if op_name != "tessera.add":
            return result
        return legality.LegalityResult(op_name, "x86", (legality.LegalityDiagnostic(
            "LEGALITY_TARGET_CAPABILITY", "add refused", "target_capability"),))

    monkeypatch.setattr(legality, "check_op_legality", refuse_add)
    # ``x + y`` is no call, so the effect scan does not list it.
    assert not _scan_admits_target_capability(_syntax_only, "x86", None)
    assert not ts.jit(target="x86")(_syntax_only).ast_emission_pending


# ── A native tracer-authority call never builds the AST either ───────────────

def _x2() -> np.ndarray:
    # Rank 2: rocm legality refuses rank-1 elementwise ops.
    return np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float32)


@pytest.mark.parametrize("target", ["rocm", "x86", "nvidia_sm120"])
def test_a_native_tracer_authority_call_never_builds_the_ast(target):
    fn = ts.jit(target=target)(_body)
    x = _x2()
    for _ in range(2):
        np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
        assert fn.frontend_authority == "tracer"
    assert fn.ast_emission_pending


def test_native_required_keeps_the_dropped_call_guard():
    # Its refusal names what the AST dropped, so the module is built.
    fn = ts.jit(target="x86", native_required=True)(_body)
    np.testing.assert_allclose(fn(_x2(), _x2()), _expected(_x2()), rtol=1e-6)
    assert not fn.ast_emission_pending


def _prefix_attention(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def _prefix_attention_unsaved(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False)


def test_the_source_attention_gate_reads_the_scan_and_the_trace():
    from tessera.compiler.jit import _is_source_attention_module

    roots = (np.zeros((1, 2, 3, 4), np.float16), np.zeros((1, 1, 5, 4), np.float16),
             np.zeros((1, 1, 5, 3), np.float16))
    fn = ts.jit(target="nvidia_sm120")(_prefix_attention)
    assert fn._scan_is_source_attention()
    assert _is_source_attention_module(fn._trace_frontend_capture(roots, {})[0])
    assert fn.ast_emission_pending
    # The scan admits the op list; only the trace sees the missing saved LSE.
    unsaved = ts.jit(target="nvidia_sm120")(_prefix_attention_unsaved)
    assert unsaved._scan_is_source_attention()
    assert not _is_source_attention_module(unsaved._trace_frontend_capture(roots, {})[0])
    assert not ts.jit(target="nvidia_sm120")(_body)._scan_is_source_attention()


def _matmul(a, b):
    return a @ b


def test_a_deferred_emission_keeps_the_decoration_host_mode(monkeypatch):
    # A host with nvcc (Super-Bear) passes the real toolchain gate, so the
    # dashboard's generic-host rendering is the only reason it fails.
    from tessera.compiler import pipeline_gates as pg

    real_which = pg.shutil.which
    monkeypatch.setattr(pg.shutil, "which",
                        lambda name, *a, **k: "/usr/bin/nvcc" if name == "nvcc"
                        else real_which(name, *a, **k))
    with pg.deterministic_host_for_dashboard():
        fn = ts.jit(target="nvidia_sm90")(_matmul)
    assert fn.ast_emission_pending
    assert fn.compile_result.first_failing_gate.gate == "toolchain"
    assert not pg._DASHBOARD_DETERMINISTIC
    # Outside the block the same decoration sees the toolchain.
    assert ts.jit(target="nvidia_sm90")(_matmul).compile_result.first_failing_gate.gate != "toolchain"
