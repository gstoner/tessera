"""Scanned inventory of inline Apple capability gates still in the test tree.

This is deliberately an inventory, not a claim that every listed test is a
native-GPU proof.  ``hardware_marked`` sites already select the shared Metal
boundary; ``inline_capability_gate`` sites are the remaining APPLE-TEST-1
migration backlog and must be split or reclassified before they can enter the
exact-device lane.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


_APPLE_CAPABILITY_TOKENS = (
    "darwin",
    "metal",
    "apple_gpu",
    "is_metal",
    "apple_gpu_available",
    "agb.is_available",
)


# APPLE-TEST-1 residency cohort.  These are deliberately separate from the
# compiler/ABI-backed ``EXACT_DEVICE_PROOFS`` registry: they prove a composed
# JIT program has stayed resident and rejected fallback, not one ABI symbol.
NATIVE_RESIDENCY_TESTS = (
    "tests/unit/test_apple_gpu_gather.py::test_gather_runs_on_metal_no_fallback",
    "tests/unit/test_apple_gpu_concat.py::test_concat_runs_on_metal_no_fallback",
    "tests/unit/test_apple_gpu_concat.py::test_concat_compounds_with_matmul_per_op_metal",
    "tests/unit/test_apple_gpu_slice.py::test_slice_runs_on_metal_no_fallback",
    "tests/unit/test_apple_gpu_slice.py::test_slice_compounds_with_matmul_per_op_metal",
    "tests/unit/test_apple_gpu_softcap.py::test_softcap_runs_on_metal_no_fallback",
    "tests/unit/test_apple_gpu_per_op_metal.py::test_mixed_program_no_fallback_on_metal",
    "tests/unit/test_apple_gpu_transpose.py::test_transpose_runs_on_metal_no_fallback",
    "tests/unit/test_apple_gpu_topk.py::test_jit_top_k_routes_to_metal_runtime_at_rung8",
    "tests/unit/test_apple_gpu_projections.py::test_jit_linear_general_metal_runtime_on_darwin",
    "tests/unit/test_apple_gpu_projections.py::test_jit_qkv_projection_metal_runtime_on_darwin",
    "tests/unit/test_apple_gpu_bmm.py::test_jit_rank3_matmul_metal_runtime_on_darwin",
    "tests/unit/test_apple_gpu_reductions.py::test_jit_mean_metal_runtime_on_darwin",
    "tests/unit/test_apple_gpu_mpsgraph_lane.py::test_jit_tier1_ops_metal_runtime_on_darwin",
    "tests/unit/test_apple_gpu_batched_mha.py::test_batched_mha_ops_metal_runtime_on_darwin",
)


# Tests that validate native runtime state directly, rather than through a JIT
# callable.  The shared collection boundary proves the Metal device exists;
# each test then checks a native runtime capability, cache, or stress behavior.
NATIVE_RUNTIME_TESTS = (
    "tests/unit/test_apple_gpu_mpsgraph_lane.py::test_runtime_reports_metal_available",
    "tests/unit/test_apple_gpu_mpsgraph_lane.py::test_mpsgraph_graph_cache_reuses_across_calls",
    "tests/unit/test_apple_gpu_control_flow_stress.py::test_cf_while_generate_after_bulk_bmm_dispatches",
    "tests/unit/test_apple_gpu_memory_budget.py::test_device_tensor_alloc_free_accounting",
    "tests/unit/test_apple_gpu_memory_budget.py::test_peak_tracks_high_water_and_resets",
)


# Offline MSL compilation needs the Apple ``metal`` command-line tool, but no
# allocated Metal device.  Keep that host-tool boundary separate from native
# runtime/residency proof.
METAL_COMPILER_TESTS = (
    "tests/unit/test_msl_gemm_emit.py::test_rung3_simdgroup_gemm_compiles_on_metal_host",
    "tests/unit/test_msl_gemm_emit.py::test_rung3_steel_gemm_compiles_on_metal_host",
    "tests/unit/test_msl_gemm_emit.py::test_rung3_steel_refinements_compile_on_metal_host",
)


@dataclass(frozen=True)
class AppleInlineGate:
    path: str
    line: int
    gate: str
    capability: str
    classification: str


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _capability(source: str) -> str:
    lowered = source.lower()
    if "darwin" in lowered:
        return "Darwin host"
    if "metal" in lowered or "is_metal" in lowered:
        return "Metal runtime/device"
    return "Apple GPU runtime"


def inline_apple_capability_gates(root: Path) -> tuple[AppleInlineGate, ...]:
    """Return every direct pytest skip/skipif mentioning Apple capability.

    The structural scan covers module-, function-, and helper-level guards, so
    a new inline capability skip becomes visible immediately in the migration
    inventory instead of disappearing behind a filename convention.
    """

    rows: list[AppleInlineGate] = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted_name(node.func)
            if name not in {"pytest.skip", "pytest.mark.skipif"}:
                continue
            source = ast.get_source_segment(text, node) or ""
            if not any(token in source.lower() for token in _APPLE_CAPABILITY_TOKENS):
                continue
            rows.append(
                AppleInlineGate(
                    path=str(path.relative_to(root)),
                    line=node.lineno,
                    gate="skipif" if name.endswith("skipif") else "skip",
                    capability=_capability(source),
                    # A direct skip remains migration debt even when a sibling
                    # test in the same module is hardware-marked.  Classify at
                    # the gate site, never from a filename/module substring.
                    classification="inline_capability_gate",
                )
            )
    return tuple(rows)
