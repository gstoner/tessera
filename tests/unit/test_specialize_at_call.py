"""E2E-REAL-6F specialize-at-call: an undeclared signature has no compiled
form before a call.

A ``@jit`` program with an unannotated parameter defers its AST emission. The
only module a read outside a call could get is a second, AST-only lowering
typed over opaque tensors, which no call executes once the trace owns the
program. Such a read now refuses with ``JIT_UNSPECIALIZED_SIGNATURE``;
``specialize`` / ``for_signature`` give the compiled form of one concrete
signature, and reads inside a call are unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.diagnostic_codes import code_lookup as lookup
from tessera.compiler.diagnostics import JitDiagnosticCode
from tessera.compiler.jit import TesseraJitError, _undeclared_parameters

CODE = JitDiagnosticCode.UNSPECIALIZED_SIGNATURE.value


def _mm_relu(x, y):
    return ts.ops.relu(ts.ops.matmul(x, y))


def _mm_relu_declared(x: ts.Tensor["M", "K"], y: ts.Tensor["K", "N"]):
    return ts.ops.relu(ts.ops.matmul(x, y))


def _inputs():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((4, 8)).astype(np.float32),
            rng.standard_normal((8, 3)).astype(np.float32))


def _ref(a, b):
    return np.maximum(a @ b, 0.0)


def test_code_is_registered():
    entry = lookup(CODE)
    assert entry is not None and entry.severity == "error"


@pytest.mark.parametrize("read", [
    lambda f: f.graph_ir,
    lambda f: f.compile_bundle,
    lambda f: f.cpu_plan,
    lambda f: f.compile_result,
    lambda f: f.lowering_diagnostics,
    lambda f: f.runtime_artifact(),
    lambda f: f.ir_text(),
])
def test_a_pre_call_read_of_an_undeclared_signature_refuses(read):
    fn = ts.jit(_mm_relu)
    assert fn.ast_emission_pending
    with pytest.raises(TesseraJitError, match=CODE) as info:
        read(fn)
    # The message names the parameters and the remedies.
    assert "'x', 'y'" in str(info.value)
    assert "specialize" in str(info.value)
    # A refusal is not the emission's error: the program still runs.
    a, b = _inputs()
    np.testing.assert_allclose(fn(a, b), _ref(a, b), rtol=1e-5, atol=1e-6)


def test_a_declared_signature_reads_before_a_call():
    fn = ts.jit(_mm_relu_declared)
    assert fn.graph_ir.functions
    assert fn.compile_bundle is not None


def test_any_annotation_declares_a_parameter():
    def mixed(cache: object, start: int, x):
        return x

    assert _undeclared_parameters(mixed) == ("x",)
    assert _undeclared_parameters(_mm_relu_declared) == ()


def test_specialize_compiles_the_signature_and_executes_it():
    fn = ts.jit(_mm_relu)
    a, b = _inputs()
    spec = fn.specialize(a, b)
    function = spec.graph_ir.functions[0]
    # The trace names the arguments after the source parameters.
    assert [arg.name for arg in function.args] == ["x", "y"]
    assert [str(arg.ir_type) for arg in function.args] == [
        "tensor<4x8xf32>", "tensor<8x3xf32>"]
    assert spec.graph_ir.module_attrs.get("tessera.frontend.authority") == '"tracer"'
    assert spec.compile_bundle is not None and spec.cpu_plan is not None
    np.testing.assert_allclose(spec(a, b), _ref(a, b), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(spec(x=a, y=b), _ref(a, b), rtol=1e-5, atol=1e-6)
    # Specializing builds no AST module on the parent.
    assert fn.ast_emission_pending


def test_specialize_is_cached_per_signature():
    fn = ts.jit(_mm_relu)
    a, b = _inputs()
    assert fn.specialize(a, b) is fn.specialize(a, b)
    other = fn.specialize(a[:2], b)
    assert other is not fn.specialize(a, b)
    assert str(other.graph_ir.functions[0].args[0].ir_type) == "tensor<2x8xf32>"


def test_for_signature_keeps_a_declared_program_unchanged():
    a, b = _inputs()
    declared = ts.jit(_mm_relu_declared)
    assert declared.for_signature(a, b) is declared
    undeclared = ts.jit(_mm_relu)
    assert undeclared.for_signature(a, b) is undeclared.specialize(a, b)


def test_a_read_after_a_traced_call_uses_the_trace():
    fn = ts.jit(_mm_relu)
    a, b = _inputs()
    fn(a, b)
    assert fn.frontend_authority == "tracer"
    assert fn.graph_ir.module_attrs.get("tessera.frontend.authority") == '"tracer"'


def test_one_programs_call_does_not_license_anothers_pre_call_read():
    other = ts.jit(_mm_relu)

    def reads_other(x, y):
        # Touches another program's compiled form from inside this one.
        _ = other.graph_ir
        return x

    outer = ts.jit(reads_other)
    a, b = _inputs()
    with pytest.raises(TesseraJitError, match=CODE):
        outer(a, b)


def test_specialize_refuses_a_differentiation_request():
    fn = ts.jit(autodiff="reverse")(_mm_relu)
    a, b = _inputs()
    with pytest.raises(TesseraJitError, match="differentiation request"):
        fn.specialize(a, b)


def test_a_read_after_a_traced_call_describes_that_call():
    fn = ts.jit(_mm_relu)
    a, b = _inputs()
    fn(a, b)
    # Fields the traced call never set answer from its specialization.
    diagnostics = fn.lowering_diagnostics
    assert diagnostics == fn.specialize(a, b).lowering_diagnostics
    assert fn.explain_lowering()
    # ...without building the AST module or rewriting the decoration's plan.
    assert fn.ast_emission_pending


def test_a_later_call_of_another_shape_runs_its_own_program():
    fn = ts.jit(_mm_relu)
    a, b = _inputs()
    fn(a, b)
    _ = fn.compile_bundle  # answered from the (4, 8) x (8, 3) specialization
    small_a, small_b = a[:2, :5].copy(), b[:5, :].copy()
    np.testing.assert_allclose(fn(small_a, small_b), _ref(small_a, small_b),
                               rtol=1e-5, atol=1e-6)
    assert str(fn.graph_ir.functions[0].args[0].ir_type) == "tensor<2x5xf32>"


def _scaled_relu(x):
    # ``Tracer * float`` is outside the tracer's vocabulary: a call runs the
    # decoration's AST candidate for this body.
    return ts.ops.relu(x) * 2.0


def test_for_signature_answers_with_what_a_call_runs_when_the_trace_declines():
    fn = ts.jit(_scaled_relu)
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    with pytest.raises(TesseraJitError, match="tracer frontend failed"):
        fn.specialize(x)
    compiled = fn.for_signature(x)
    assert compiled is fn and not fn.ast_emission_pending
    assert fn.graph_ir.functions


def test_an_unresolvable_string_annotation_declares_nothing():
    def local_type(x: "NoSuchType", y: "int"):  # noqa: F821
        return x

    assert _undeclared_parameters(local_type) == ("x",)
