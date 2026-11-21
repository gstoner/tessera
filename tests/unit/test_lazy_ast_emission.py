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


def test_a_differentiation_request_defers_and_writes_its_intent_on_emission():
    # The intent the C++ autodiff pass keys on, and the backward provenance
    # read from the compile, are produced by the deferred emission, so the
    # decoration builds no AST module until something reads one.
    from tessera.compiler.autodiff_request import BackwardStatus

    eager = ts.jit(autodiff="reverse", wrt=["x"])
    lazy = ts.jit(autodiff="reverse", wrt=["x"])(_body)
    assert lazy.ast_emission_pending
    assert "_differentiation_provenance" not in lazy.__dict__
    backward = lazy.backward_provenance
    assert not lazy.ast_emission_pending, "provenance comes from the compile"
    assert backward.status is not BackwardStatus.NOT_REQUESTED
    assert lazy.differentiation_provenance is backward
    assert lazy.compile_result.backward is backward
    text = lazy.graph_ir.to_mlir()
    assert "tessera.autodiff" in text
    fn_attrs = [gf.fn_attrs for gf in lazy.graph_ir.functions]
    assert any("tessera.autodiff" in attrs for attrs in fn_attrs), fn_attrs

    # Reading the module first resolves the same provenance afterwards.
    module_first = eager(_body)
    assert "tessera.autodiff" in module_first.graph_ir.to_mlir()
    assert module_first.backward_provenance.status is backward.status


def test_a_forward_request_defers_with_not_requested_backward():
    from tessera.compiler.autodiff_request import BackwardStatus

    jvp = ts.jit(autodiff="forward", wrt=["x"])(_body)
    assert jvp.ast_emission_pending
    assert jvp.backward_provenance.status is BackwardStatus.NOT_REQUESTED
    assert jvp.differentiation_provenance.status is not BackwardStatus.NOT_REQUESTED


# ── The Apple targets defer it too ───────────────────────────────────────────

def _looped(x, y):
    for _ in range(2):
        x = tessera.ops.add(x, y)
    return x


def _helper_with_a_branch(v):
    if v.ndim == 2:
        return tessera.ops.relu(v)
    return v


def _calls_a_branching_helper(x, y):
    return _helper_with_a_branch(tessera.ops.add(x, y))


def _control_call(x, y):
    return tessera.control.cond(True, lambda: x, lambda: y)


# Both Apple runtimes execute on the Darwin host only; elsewhere the traced
# bundle is artifact-only and the call builds the AST plan to compare.
@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("target", ["apple_cpu", "apple_gpu"])
def test_an_apple_tracer_authority_call_never_builds_the_ast(target):
    fn = ts.jit(target=target)(_body)
    assert fn.ast_emission_pending
    x = _x2()
    for _ in range(2):
        np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
        assert fn.frontend_authority == "tracer"
        assert fn.executed_plan_frontend == "tracer"
    assert fn.ast_emission_pending


@pytest.mark.parametrize("body", [_looped, _calls_a_branching_helper, _control_call])
def test_an_apple_gpu_body_the_tracer_gate_may_route_keeps_the_eager_emission(body):
    # The Phase-F gate reads the module's scf markers; control flow in the body,
    # in a helper it calls, or a control.* call keeps it at decoration.
    assert not ts.jit(target="apple_gpu")(body).ast_emission_pending
    # apple_cpu has no such gate.
    assert ts.jit(target="apple_cpu")(body).ast_emission_pending


@pytest.mark.parametrize("options", [
    {"dispatch_via_package": True},
    {"dispatch_via_package": "auto"},
    {"auto_batch": True},
])
def test_apple_gpu_package_and_auto_batch_routes_keep_the_eager_emission(options):
    # Both recognize their program from the module (auto_batch skips it).
    assert not ts.jit(target="apple_gpu", **options)(_body).ast_emission_pending


def test_a_deferred_apple_gpu_decoration_recognizes_its_package_on_demand():
    eager = ts.jit(target="apple_gpu", dispatch_via_package=True)(_body)
    deferred = ts.jit(target="apple_gpu")(_body)
    assert deferred.ast_emission_pending
    assert deferred.recognized_package == eager.recognized_package
    assert not deferred.ast_emission_pending


def test_a_deferred_apple_gpu_emission_failure_still_routes_to_the_tracer():
    # The gate's one remaining cause, known only once the emission runs.
    fn = ts.jit(target="apple_gpu")(_drops_a_statement)
    assert fn.ast_emission_pending
    assert fn._needs_trace is True
    assert not fn.ast_emission_pending
    assert fn._legacy_graph_ir.functions == []


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


def _prefix_attention_with_eps(q, k, v, eps=1e-5):
    normalized = ts.ops.rmsnorm(q, eps=eps, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def test_the_source_attention_gate_declines_a_scalar_parameter():
    from tessera.compiler.jit import _JIT_FALLBACK

    roots = (np.zeros((1, 2, 3, 4), np.float16), np.zeros((1, 1, 5, 4), np.float16),
             np.zeros((1, 1, 5, 3), np.float16))
    fn = ts.jit(target="nvidia_sm120")(_prefix_attention_with_eps)
    assert fn._scan_is_source_attention()
    # Declines rather than raising from ``input_signature`` on the scalar.
    assert fn._try_source_attention_call(roots, {}) is _JIT_FALLBACK
    assert fn.ast_emission_pending


@pytest.mark.parametrize("target", [None, "cpu", "apple_cpu", "apple_gpu", "x86"])
def test_the_native_descriptor_call_leaves_the_ast_unbuilt_where_it_cannot_serve(target):
    # Only the ROCm and sm_120 lanes have descriptor routes. The ROCm product
    # check used to read ``graph_ir`` before the target gate, so every other
    # call that reached it built the deferred AST module only to fall back.
    from tessera.compiler.jit import _JIT_FALLBACK

    fn = ts.jit(_body) if target is None else ts.jit(target=target)(_body)
    assert fn.ast_emission_pending
    x = _x()
    assert fn._try_native_descriptor_call((x, x), {}) is _JIT_FALLBACK
    assert fn.ast_emission_pending


def test_the_decoration_trace_is_verified_against_the_decorations_target():
    # A fp32 ``scaled_matmul`` is legal on gfx1201 and not on the CPU table.
    # Verified against the CPU table, the decoration-time trace was refused
    # and the program fell back to the AST frontend, which ``tessera.vmap``
    # then read for every scaled map.
    from tessera.compiler.graph_ir import GraphIRBuilder
    from tests.unit.test_public_floating_scaled_reverse import floating_python_source

    fn = ts.from_text(floating_python_source(), target="rocm_gfx1201")
    assert fn.ast_emission_pending
    assert fn.graph_ir.module_attrs.get("tessera.frontend.authority") == '"tracer"'
    assert fn._legacy_graph_ir is None

    def authority(legality_target):
        builder = GraphIRBuilder()
        builder.lower(fn._fn, source_text=fn._frontend_source_text,
                      prefer_abstract_trace=True, legality_target=legality_target)
        return builder.module().module_attrs.get("tessera.frontend.authority")

    assert authority("rocm_gfx1201") == '"tracer"'
    assert authority("cpu") is None  # the CPU table refuses it: AST fallback


def test_mapped_owners_share_one_ast_oracle_lowering(monkeypatch):
    # With the decoration trace owning ``graph_ir``, each ``tessera.vmap``
    # owner asks ``_ensure_legacy_graph_ir`` for the AST oracle its
    # certificate compares against. The oracle is memoized, so the second
    # owner of the same source does not lower it through the AST again, and
    # each owner still gets its own copy.
    from tessera.autodiff import vmap
    from tessera.compiler import graph_ir_cache
    from tessera.compiler.graph_ir import GraphIRBuilder
    from tests.unit.test_public_floating_scaled_reverse import floating_python_source

    graph_ir_cache.clear_graph_ir_cache()
    scalar = ts.from_text(floating_python_source(), target="rocm_gfx1201")
    calls = []
    original = GraphIRBuilder._extract_ops

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    first, second = vmap(scalar), vmap(scalar)
    # The owners carry the traced module, so each must ask for the oracle.
    for owner in (first, second):
        assert owner.graph_ir.module_attrs.get("tessera.frontend.authority") == '"tracer"'
        assert owner._legacy_graph_ir is None
    monkeypatch.setattr(GraphIRBuilder, "_extract_ops", counting)
    oracle = first._ensure_legacy_graph_ir()
    assert len(calls) == 1
    again = second._ensure_legacy_graph_ir()
    assert len(calls) == 1
    assert again == oracle and again is not oracle
    assert oracle.module_attrs.get("tessera.frontend.authority") is None


def test_the_ast_oracle_cache_keys_on_the_resolved_annotations():
    # A factory makes callables with one source text and one code location
    # whose annotations differ through a closure. ``lower`` bakes those
    # extents into the Graph module, so the memoized oracle of one must not
    # be served to the other.
    from tessera.compiler import graph_ir_cache

    def make(n):
        def f(x):
            return ts.ops.relu(x)
        # This module postpones annotations, so a closure ``n`` written in
        # the signature would stay an unresolved string; attach the object.
        f.__annotations__ = {"x": ts.Tensor[n]}
        return ts.jit(f)

    graph_ir_cache.clear_graph_ir_cache()
    eight, sixteen = make(8), make(16)
    for fn in (eight, sixteen):
        fn._legacy_graph_ir = None
    first = eight._ensure_legacy_graph_ir().to_mlir()
    second = sixteen._ensure_legacy_graph_ir().to_mlir()
    assert 'tessera.dim_names = ["8"]' in first
    assert 'tessera.dim_names = ["16"]' in second
