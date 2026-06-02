"""P1 (2026-06-02) — multi-op compiler metadata: component_ops + gating.

COMPILER_AUDIT "Still Open": *program identity is too single-op-oriented*
and "gate whole programs and component ops separately." Canonical
metadata now carries the whole-program op vocabulary (``component_ops``)
and a whole-program gate answer (``program_executable``) gated
component-by-component, alongside the existing primary-op answer.

These tests pin the extractor, the component-level gating, and the
metadata surfaces — and confirm single-op programs are unchanged.
"""

from __future__ import annotations

from tessera.compiler import canonical_compile as cn
from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)


_T = IRType("tensor<8x8xf32>", ("8", "8"), "fp32")


def _single_op_module() -> GraphIRModule:
    fn = GraphIRFunction(
        name="single",
        args=[IRArg("a", _T), IRArg("b", _T)],
        result_types=[_T],
        body=[IROp(result="c", op_name="tessera.matmul",
                   operands=["%a", "%b"],
                   operand_types=["tensor<8x8xf32>", "tensor<8x8xf32>"],
                   result_type="tensor<8x8xf32>")],
        return_values=["%c"],
    )
    return GraphIRModule(functions=[fn])


def _two_op_module() -> GraphIRModule:
    """matmul → relu — a real 2-op program."""
    fn = GraphIRFunction(
        name="prog",
        args=[IRArg("a", _T), IRArg("b", _T)],
        result_types=[_T],
        body=[
            IROp(result="c", op_name="tessera.matmul",
                 operands=["%a", "%b"],
                 operand_types=["tensor<8x8xf32>", "tensor<8x8xf32>"],
                 result_type="tensor<8x8xf32>"),
            IROp(result="d", op_name="tessera.relu", operands=["%c"],
                 operand_types=["tensor<8x8xf32>"],
                 result_type="tensor<8x8xf32>"),
        ],
        return_values=["%d"],
    )
    return GraphIRModule(functions=[fn])


def _repeat_op_module() -> GraphIRModule:
    """matmul → matmul — component_ops must dedupe to a single name."""
    fn = GraphIRFunction(
        name="repeat",
        args=[IRArg("a", _T), IRArg("b", _T)],
        result_types=[_T],
        body=[
            IROp(result="c", op_name="tessera.matmul",
                 operands=["%a", "%b"],
                 operand_types=["tensor<8x8xf32>", "tensor<8x8xf32>"],
                 result_type="tensor<8x8xf32>"),
            IROp(result="d", op_name="tessera.matmul",
                 operands=["%c", "%b"],
                 operand_types=["tensor<8x8xf32>", "tensor<8x8xf32>"],
                 result_type="tensor<8x8xf32>"),
        ],
        return_values=["%d"],
    )
    return GraphIRModule(functions=[fn])


# ── component_ops vector ──────────────────────────────────────────────

def test_single_op_program_collapses_to_primary():
    r = cn.canonical_compile(_single_op_module(), target="cpu")
    assert r.component_ops == ("matmul",)
    assert r.is_single_op
    # Single-op: whole-program gate == primary-op + bundle answer.
    assert r.program_executable == r.executable


def test_two_op_program_lists_all_components_in_order():
    r = cn.canonical_compile(_two_op_module(), target="cpu")
    assert r.component_ops == ("matmul", "relu")
    assert not r.is_single_op


def test_component_ops_dedupes_repeats():
    r = cn.canonical_compile(_repeat_op_module(), target="cpu")
    assert r.component_ops == ("matmul",)
    assert r.is_single_op


# ── component-level gating ────────────────────────────────────────────

def test_program_executable_is_and_over_components():
    r = cn.canonical_compile(_two_op_module(), target="cpu")
    expected = bool(r.bundle.executable) and not r.component_blockers
    assert r.program_executable == expected
    # Every blocker references a real component op + a named gate.
    for op, gate in r.component_blockers:
        assert op in r.component_ops
        assert isinstance(gate, str) and gate


def test_component_blockers_on_hardware_gated_target():
    """On a target with no executable runtime (nvidia_sm90), the
    whole-program answer is non-executable and the blockers name the
    component ops + failing gate — not just the primary op."""
    r = cn.canonical_compile(_two_op_module(), target="nvidia_sm90")
    assert r.program_executable is False
    assert r.component_blockers  # at least one component blocks
    blocked_ops = {op for op, _g in r.component_blockers}
    assert blocked_ops <= set(r.component_ops)


# ── metadata surfaces ─────────────────────────────────────────────────

def test_to_dict_carries_component_fields():
    r = cn.canonical_compile(_two_op_module(), target="cpu")
    d = r.to_dict()
    assert d["component_ops"] == ["matmul", "relu"]
    assert "program_executable" in d
    assert isinstance(d["component_blockers"], list)


def test_runtime_artifact_metadata_carries_component_fields():
    r = cn.canonical_compile(_two_op_module(), target="cpu")
    meta = r.to_runtime_artifact().metadata
    assert meta["canonical_component_ops"] == ["matmul", "relu"]
    assert "canonical_program_executable" in meta
    assert isinstance(meta["canonical_component_blockers"], list)


def test_compile_result_from_bundle_without_module_is_single_op_safe():
    """The C.3 path (``@jit``) may pass a bundle without a module; the
    component vector then collapses to the primary op — never crashes."""
    # Build via the module path, then re-synthesize from the bundle alone.
    r = cn.canonical_compile(_single_op_module(), target="cpu")
    r2 = cn.compile_result_from_bundle(r.bundle, primary_op="matmul")
    assert r2.component_ops == ("matmul",)
    assert r2.program_executable == r2.executable
