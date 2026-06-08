"""Tests for ``tessera.compiler.clifford_jit`` — the one true
compiler-integrated GA vertical slice.

These tests cover the v1 contract:

  1. **Trace + plan capture.** ``@clifford_jit(target="apple_gpu")``
     traces the function on first call, captures the op plan from
     the bridge trace, and freezes the artifact.
  2. **Manifest verification.** Every op in the plan resolves to
     ``apple_gpu / fused / <symbol>`` via the backend manifest.
  3. **Plan stability.** The plan hash is deterministic from the
     sequence of op names — same function ⇒ same hash.
  4. **Per-call route trace.** Subsequent calls produce a fresh
     route trace; ``plan_matches_routes()`` returns ``True``.
  5. **Error semantics.** Unsupported target / dtype / op raises
     :class:`CliffordJitError` at decoration or first call.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple GPU runtime only loadable on macOS",
)

from tessera.compiler.clifford_jit import (
    CliffordJitError,
    CliffordCompiledArtifact,
    CliffordCompiledCallable,
    CliffordIROpCall,
    CliffordIRProgram,
    CliffordOpPlanEntry,
    clifford_jit,
    lower_function_to_ir,
)


# ---------------------------------------------------------------------------
# Decorator contract — pre-call
# ---------------------------------------------------------------------------

def test_clifford_jit_rejects_unsupported_target() -> None:
    with pytest.raises(CliffordJitError, match="target must be one of"):
        clifford_jit(target="nvidia_sm90")


def test_clifford_jit_rejects_unsupported_dtype() -> None:
    with pytest.raises(CliffordJitError, match="dtype='f32'"):
        clifford_jit(target="apple_gpu", dtype="fp16")


def test_decorator_returns_callable_wrapper() -> None:
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f(a, b):
        return ga.inner(a, b)

    # AST → IR lowering runs at decoration time, so the artifact is
    # already populated before any call.
    assert isinstance(f, CliffordCompiledCallable)
    assert f.artifact.target == "apple_gpu"
    assert f.artifact.op_names() == ("clifford_inner",)
    assert f.artifact.plan_hash != ""
    assert f.artifact.ir is not None
    assert f.artifact.ir.arg_names == ("a", "b")


# ---------------------------------------------------------------------------
# Trace + plan capture (Darwin-only — needs the GPU runtime)
# ---------------------------------------------------------------------------

@darwin_only
def test_first_call_captures_op_plan() -> None:
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def point_cloud_rotor_invariant(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(0)
    R = rng.randn(8, 8).astype(np.float32) * 0.3
    V = rng.randn(8, 8).astype(np.float32)
    out = point_cloud_rotor_invariant(ga.Multivector(R, a),
                                       ga.Multivector(V, a))
    # Plan is now frozen. The rotor_sandwich→norm chain fuses into a single
    # clifford_rotor_sandwich_norm op (gap #6).
    plan = point_cloud_rotor_invariant.artifact.op_names()
    assert plan == ("clifford_rotor_sandwich_norm",)
    # Output shape is per-batch scalar.
    assert np.asarray(out).shape == (8,)


@darwin_only
def test_plan_hash_is_deterministic() -> None:
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f1(a, b):
        return ga.inner(a, b)

    @clifford_jit(target="apple_gpu")
    def f2(a, b):
        return ga.inner(a, b)

    a = ga.Cl(3, 0)
    A = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    B = np.random.RandomState(1).randn(4, 8).astype(np.float32)
    mv_a = ga.Multivector(A, a)
    mv_b = ga.Multivector(B, a)
    f1(mv_a, mv_b)
    f2(mv_a, mv_b)
    # Same trace plan ⇒ same hash.
    assert f1.artifact.plan_hash == f2.artifact.plan_hash
    assert f1.artifact.plan_hash != ""


@darwin_only
def test_plan_entries_match_manifest() -> None:
    """Every op in the plan must equal what the manifest resolves."""
    import tessera.ga as ga
    from tessera.compiler import jit_bridge as bridge

    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(2)
    R = rng.randn(4, 8).astype(np.float32) * 0.3
    V = rng.randn(4, 8).astype(np.float32)
    f(ga.Multivector(R, a), ga.Multivector(V, a))
    for entry in f.artifact.plan:
        expected = bridge.lookup_apple_gpu_symbol(entry.op_name)
        assert entry.symbol == expected
        assert entry.status == "fused"
        assert entry.target == "apple_gpu"


@darwin_only
def test_per_call_route_trace_matches_plan() -> None:
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    R = np.random.RandomState(3).randn(8, 8).astype(np.float32) * 0.3
    V = np.random.RandomState(4).randn(8, 8).astype(np.float32)
    mv_R = ga.Multivector(R, a)
    mv_V = ga.Multivector(V, a)
    # First call compiles + executes.
    f(mv_R, mv_V)
    # Second call: route trace ≡ plan.
    f(mv_R, mv_V)
    routes = f.last_routes()
    assert len(routes) == len(f.artifact.plan)
    assert f.plan_matches_routes()
    # Every route carries the manifest-resolved symbol.
    for route, plan_entry in zip(routes, f.artifact.plan):
        assert route.op_name == plan_entry.op_name
        assert route.symbol == plan_entry.symbol
        assert route.context == "jit:apple_gpu"


@darwin_only
def test_decorated_function_with_int_literal_runs_end_to_end() -> None:
    """``@clifford_jit`` accepts ``ga.grade_projection(a, k)`` with
    an integer literal — the IR carries the literal as ``#int:k`` and
    the executor decodes it back to a Python int at dispatch time."""
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def project_grade_2(a):
        return ga.grade_projection(a, 2)

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(7)
    A = rng.randn(8, 8).astype(np.float32)
    out = project_grade_2(ga.Multivector(A, a))
    # Plan + IR record the literal.
    assert project_grade_2.artifact.op_names() == ("clifford_grade_projection",)
    ir = project_grade_2.artifact.ir
    assert ir is not None
    assert ir.ops[0].operand_refs == ("a", "#int:2")
    # Output matches the reference projection.
    ref = ga.grade_projection(ga.Multivector(A, a), 2)
    assert np.allclose(out._coefficients, ref._coefficients, atol=1e-6)
    # Confirm at least one route fired through the bridge.
    routes = project_grade_2.last_routes()
    assert routes
    assert routes[0].op_name == "clifford_grade_projection"


@darwin_only
def test_compiled_artifact_metadata_is_json_serializable() -> None:
    """The artifact's ``as_metadata()`` output must be JSON-able for
    benchmark report embedding."""
    import json
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f(a, b):
        return ga.inner(a, b)

    a = ga.Cl(3, 0)
    A = np.random.RandomState(5).randn(4, 8).astype(np.float32)
    B = np.random.RandomState(6).randn(4, 8).astype(np.float32)
    f(ga.Multivector(A, a), ga.Multivector(B, a))
    meta = f.artifact.as_metadata()
    # Round-trips cleanly.
    blob = json.dumps(meta)
    reloaded = json.loads(blob)
    assert reloaded["target"] == "apple_gpu"
    assert reloaded["dtype"] == "f32"
    assert reloaded["plan_hash"] == f.artifact.plan_hash
    assert len(reloaded["plan"]) == len(f.artifact.plan)


# ---------------------------------------------------------------------------
# Error semantics — bad ops detected at compile time
# ---------------------------------------------------------------------------

def test_function_with_no_ga_ops_raises_at_decoration() -> None:
    """A function that doesn't fit the AST grammar (here: pure
    arithmetic without any ``ga.*`` call) is rejected at decoration
    time — that's the new compile-time guarantee."""

    with pytest.raises(CliffordJitError):
        @clifford_jit(target="apple_gpu")
        def f(x):
            return x + 1   # pure numpy; no GA op


# ---------------------------------------------------------------------------
# Frozen dataclass contract
# ---------------------------------------------------------------------------

def test_compiled_artifact_is_immutable() -> None:
    art = CliffordCompiledArtifact(
        plan=(CliffordOpPlanEntry(
            op_name="clifford_inner", target="apple_gpu",
            status="fused", symbol="tessera_apple_gpu_clifford_inner_cl30_f32",
        ),),
        target="apple_gpu", dtype="f32",
        manifest_sources=("_CLIFFORD_APPLE_GPU_FUSED",),
        plan_hash="abc123", source_name="f",
    )
    with pytest.raises((AttributeError, TypeError)):
        art.target = "rocm"   # type: ignore[misc]


def test_op_plan_entry_is_immutable() -> None:
    entry = CliffordOpPlanEntry(
        op_name="clifford_inner", target="apple_gpu",
        status="fused", symbol="x",
    )
    with pytest.raises((AttributeError, TypeError)):
        entry.target = "rocm"   # type: ignore[misc]


# ---------------------------------------------------------------------------
# AST → Clifford IR lowering — direct (no GPU dispatch)
# ---------------------------------------------------------------------------

def test_lower_function_to_ir_single_op() -> None:
    import tessera.ga as ga

    def f(a, b):
        return ga.inner(a, b)

    ir = lower_function_to_ir(f)
    assert isinstance(ir, CliffordIRProgram)
    assert ir.arg_names == ("a", "b")
    assert len(ir.ops) == 1
    op = ir.ops[0]
    assert op.op_name == "clifford_inner"
    assert op.python_attr == "inner"
    assert op.operand_refs == ("a", "b")
    assert op.result_name == ir.return_ref


def test_lower_function_to_ir_nested_call_chain() -> None:
    """rotor_sandwich → norm — the canonical vertical-slice demo."""
    import tessera.ga as ga

    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    ir = lower_function_to_ir(f)
    assert ir.arg_names == ("rotor", "points")
    assert len(ir.ops) == 2
    inner, outer = ir.ops
    assert inner.op_name == "clifford_rotor_sandwich"
    assert inner.operand_refs == ("rotor", "points")
    assert outer.op_name == "clifford_norm"
    # outer consumes inner's result.
    assert outer.operand_refs == (inner.result_name,)
    assert ir.return_ref == outer.result_name


def test_ir_text_round_trips_through_metadata() -> None:
    import json
    import tessera.ga as ga

    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    ir = lower_function_to_ir(f)
    text = ir.text()
    assert "clifford_rotor_sandwich" in text
    assert "clifford_norm" in text
    assert "return" in text
    # Metadata serializes cleanly.
    meta = ir.as_metadata()
    reloaded = json.loads(json.dumps(meta))
    assert reloaded["arg_names"] == ["rotor", "points"]
    assert [op["op"] for op in reloaded["ops"]] == [
        "clifford_rotor_sandwich", "clifford_norm",
    ]


def test_lower_rejects_non_ga_call() -> None:
    def f(a, b):
        return min(a, b)  # not a ga.* call

    with pytest.raises(CliffordJitError, match="only ``tessera.ga"):
        lower_function_to_ir(f)


def test_lower_rejects_call_on_unrelated_object_with_ga_method_name() -> None:
    """``foo.norm(a)`` must not lower as ``clifford_norm`` just
    because the attribute happens to share a name with a GA op.  The
    receiver has to be the ``ga`` namespace (``ga.<op>`` or any
    chain ending in ``.ga.<op>``)."""
    class _Foo:
        def norm(self, x):  # pragma: no cover — never called
            return x

    foo = _Foo()  # noqa: F841 — referenced inside f's source for AST

    def f(a):
        return foo.norm(a)   # bare receiver named foo, not ga

    with pytest.raises(CliffordJitError, match="only ``tessera.ga"):
        lower_function_to_ir(f)


def test_lower_rejects_numpy_linalg_norm() -> None:
    """``np.linalg.norm(a)`` — the chain ends in ``.norm`` but the
    parent segment is ``linalg``, not ``ga``.  Reject."""
    import numpy as np  # noqa: F401 — referenced inside f's source

    def f(a):
        return np.linalg.norm(a)

    with pytest.raises(CliffordJitError, match="only ``tessera.ga"):
        lower_function_to_ir(f)


def test_lower_rejects_self_dot_norm() -> None:
    """A method-style receiver (``self.norm(a)``) is rejected — the
    immediate receiver is ``self``, not ``ga``."""
    def f(self, a):
        return self.norm(a)

    with pytest.raises(CliffordJitError, match="only ``tessera.ga"):
        lower_function_to_ir(f)


def test_lower_accepts_tessera_dot_ga_chain() -> None:
    """``tessera.ga.<op>`` is still accepted — the chain ends in
    ``.ga.<op>`` with a Name root."""
    import tessera  # noqa: F401 — referenced inside f's source

    def f(a, b):
        return tessera.ga.inner(a, b)

    ir = lower_function_to_ir(f)
    assert ir.ops[0].op_name == "clifford_inner"


def test_lower_accepts_self_dot_ga_dot_op_chain() -> None:
    """A chain ending in ``.ga.<op>`` is accepted regardless of how
    deep the prefix is, as long as the root is a Name."""
    def f(self, a, b):
        return self.ga.inner(a, b)

    ir = lower_function_to_ir(f)
    assert ir.ops[0].op_name == "clifford_inner"


def test_lower_rejects_call_dot_ga_chain() -> None:
    """A chain whose prefix contains a Call (``get_lib().ga.inner``)
    is rejected — we don't reason about dynamic receivers."""
    def f(a, b):
        return get_lib().ga.inner(a, b)  # noqa: F821 — intentional

    with pytest.raises(CliffordJitError, match="only ``tessera.ga"):
        lower_function_to_ir(f)


def test_lower_rejects_binop_return() -> None:
    def f(a, b):
        return a + b

    with pytest.raises(CliffordJitError, match="unsupported expression type"):
        lower_function_to_ir(f)


def test_lower_accepts_assignment_then_return() -> None:
    """``rotated = ga.rotor_sandwich(...); return ga.norm(rotated)`` —
    the canonical vertical-slice form used by the benchmark."""
    import tessera.ga as ga

    def f(rotor, points):
        rotated = ga.rotor_sandwich(rotor, points)
        return ga.norm(rotated)

    ir = lower_function_to_ir(f)
    assert len(ir.ops) == 2
    assert ir.ops[0].op_name == "clifford_rotor_sandwich"
    assert ir.ops[1].op_name == "clifford_norm"
    # `rotated` is bound to the rotor_sandwich result; norm consumes it.
    assert ir.ops[1].operand_refs == (ir.ops[0].result_name,)


def test_lower_rejects_unknown_name_in_return() -> None:
    import tessera.ga as ga

    def f(a, b):
        return ga.inner(a, c)  # noqa: F821 — intentional

    with pytest.raises(CliffordJitError, match="not a function argument"):
        lower_function_to_ir(f)


def test_lower_rejects_branching_body() -> None:
    import tessera.ga as ga

    def f(a, b):
        if a is b:
            return ga.inner(a, b)
        return ga.inner(b, a)

    with pytest.raises(CliffordJitError, match="simple ``name = expr``"):
        lower_function_to_ir(f)


def test_lower_rejects_keyword_args_on_ga_call() -> None:
    import tessera.ga as ga

    def f(a):
        return ga.grade_projection(a, k=2)

    with pytest.raises(CliffordJitError, match="keyword arguments"):
        lower_function_to_ir(f)


def test_lower_rejects_varargs_signature() -> None:
    import tessera.ga as ga

    def f(a, b, *rest):
        return ga.inner(a, b)

    with pytest.raises(CliffordJitError, match="positional"):
        lower_function_to_ir(f)


def test_ir_op_call_is_immutable() -> None:
    op = CliffordIROpCall(
        op_name="clifford_inner",
        operand_refs=("a", "b"),
        result_name="%t0",
        python_attr="inner",
    )
    with pytest.raises((AttributeError, TypeError)):
        op.op_name = "clifford_norm"  # type: ignore[misc]


def test_lower_accepts_int_literal_for_grade_projection() -> None:
    """``ga.grade_projection(a, 2)`` — the int 2 is encoded as the
    inline operand ref ``#int:2`` in the lowered IR."""
    import tessera.ga as ga

    def f(a):
        return ga.grade_projection(a, 2)

    ir = lower_function_to_ir(f)
    assert len(ir.ops) == 1
    op = ir.ops[0]
    assert op.op_name == "clifford_grade_projection"
    assert op.operand_refs == ("a", "#int:2")


def test_lower_accepts_negative_int_literal() -> None:
    """``ga.grade_projection(a, -1)`` lowers — the unary-minus is
    folded into the literal so it survives the lowering."""
    import tessera.ga as ga

    def f(a):
        return ga.grade_projection(a, -1)

    ir = lower_function_to_ir(f)
    assert ir.ops[0].operand_refs == ("a", "#int:-1")


def test_lower_accepts_float_literal() -> None:
    """Float literals encode as ``#float:V`` for full round-trip."""
    import tessera.ga as ga

    def f(a, b):
        # ga.inner doesn't take a literal, but the lowerer should
        # still parse it — wire a synthetic case through a binary
        # GA op that accepts a literal-positional pass-through.  We
        # use grade_projection here to exercise float parsing too
        # (the executor would reject at runtime; lowering is
        # purely structural).
        return ga.grade_projection(a, 0.5)

    ir = lower_function_to_ir(f)
    assert ir.ops[0].operand_refs == ("a", "#float:0.5")


def test_lower_rejects_string_literal() -> None:
    """Non-numeric literals (strings) are still rejected — the
    structured-IR contract limits operand refs to numeric scalars."""
    import tessera.ga as ga

    def f(a):
        return ga.grade_projection(a, "two")  # type: ignore[arg-type]

    with pytest.raises(CliffordJitError, match="literal constants"):
        lower_function_to_ir(f)


def test_decorator_artifact_metadata_embeds_ir() -> None:
    """The serialized metadata carries the IR — useful for AOT export
    and benchmark-report introspection without rerunning the GPU."""
    import tessera.ga as ga

    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    meta = f.artifact.as_metadata()
    assert "ir" in meta
    assert meta["ir"]["arg_names"] == ["rotor", "points"]
    # The decorator fuses rotor_sandwich→norm into one op (gap #6).
    assert [op["op"] for op in meta["ir"]["ops"]] == [
        "clifford_rotor_sandwich_norm",
    ]
