"""C.1 — the canonical compile wrapper.

These tests pin the audit's recommendation **C** contract:

    canonical_compile(module, target) → (typed_artifacts, capability_set,
                                          executable | reason)

Properties locked here:

1. **Typed artifacts** — every IR level (graph / schedule / tile / target)
   round-trips through ``CompileResult.<level>_ir`` and matches the
   underlying ``CompileArtifactBundle`` text.
2. **Capability set** — every cell of ``gate_results`` matches what
   ``pipeline_gates.evaluate`` would have returned directly. The canonical
   wrapper does not silently rewrite the gate truth.
3. **Executable / reason agreement** — on a known-good CPU matmul,
   ``executable=True`` and ``reason==""``. On NVIDIA on this host,
   ``executable=False`` and ``reason`` leads with the named gate.
4. **Pure aggregator** — the module only imports from the existing four
   compiler surfaces + stdlib. New compiler logic doesn't sneak in.
5. **Primary op extraction** — the gate evaluation uses the first op of
   the first function; the ``tessera.`` prefix is stripped so the gate
   evaluator's registry matches.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tessera.compiler import canonical_compile as cn
from tessera.compiler import pipeline_gates as pg
from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)


def _tiny_matmul_module() -> GraphIRModule:
    """A 1-op Graph IR module the existing ladder can compile end-to-end."""
    ten_a = IRType("tensor<128x64xf32>", ("128", "64"), "fp32")
    ten_b = IRType("tensor<64x128xf32>", ("64", "128"), "fp32")
    ten_c = IRType("tensor<128x128xf32>", ("128", "128"), "fp32")
    fn = GraphIRFunction(
        name="tiny_matmul",
        args=[IRArg("a", ten_a), IRArg("b", ten_b)],
        result_types=[ten_c],
        body=[IROp(
            result="c", op_name="tessera.matmul",
            operands=["%a", "%b"],
            operand_types=["tensor<128x64xf32>", "tensor<64x128xf32>"],
            result_type="tensor<128x128xf32>",
        )],
        return_values=["%c"],
    )
    return GraphIRModule(functions=[fn])


# ---- Typed artifacts ----

def test_canonical_compile_returns_typed_result_for_cpu_matmul():
    """Every IR level is populated; the round-trip accessors mirror the
    underlying bundle text."""
    result = cn.canonical_compile(_tiny_matmul_module(), target="cpu")
    assert result.graph_ir, "graph_ir missing"
    assert result.schedule_ir, "schedule_ir missing"
    assert result.tile_ir, "tile_ir missing"
    assert result.target_ir, "target_ir missing"
    # Each accessor matches the bundle's level text.
    assert result.graph_ir == result.bundle.graph.text
    assert result.schedule_ir == result.bundle.schedule.text
    assert result.tile_ir == result.bundle.tile.text
    assert result.target_ir == result.bundle.target_ir.text


# ---- Capability set ----

def test_gate_results_match_direct_evaluator():
    """The canonical wrapper must not re-derive gate truth — the table it
    surfaces is byte-identical to what ``pipeline_gates.evaluate`` returns
    directly for the same (target, primary_op)."""
    module = _tiny_matmul_module()
    result = cn.canonical_compile(module, target="cpu")
    direct = pg.evaluate("cpu", "matmul")
    assert result.gate_results == direct


def test_gate_status_helper_resolves_by_name():
    result = cn.canonical_compile(_tiny_matmul_module(), target="cpu")
    for gate in pg.GATE_ORDER:
        # All seven canonical gates resolve, never "unknown".
        assert result.gate_status(gate) != "unknown", gate
    # An unknown gate name returns the documented sentinel.
    assert result.gate_status("not_a_real_gate") == "unknown"


# ---- Executable / reason agreement ----

@pytest.mark.skipif(sys.platform != "darwin",
                    reason="hardware_smoke evaluator requires Darwin "
                           "for apple_cpu / apple_gpu; CPU is the focus here")
def test_cpu_matmul_is_executable_with_empty_reason():
    """On a developer Mac, CPU matmul passes every gate; the canonical
    answer is executable=True with no reason text."""
    result = cn.canonical_compile(_tiny_matmul_module(), target="cpu")
    assert result.executable is True, (
        f"expected executable, got reason={result.reason!r}")
    assert result.reason == ""
    assert result.first_failing_gate is None


def test_nvidia_matmul_is_not_executable_and_names_toolchain_gate():
    """On a developer Mac without CUDA installed, the audit-named first
    failing gate is ``toolchain``. The canonical wrapper surfaces it
    directly — no parsing of the bundle's reason string required."""
    result = cn.canonical_compile(_tiny_matmul_module(), target="nvidia_sm90")
    assert result.executable is False
    assert result.first_failing_gate is not None
    assert result.first_failing_gate.gate == pg.GATE_TOOLCHAIN
    assert "nvcc" in result.first_failing_gate.detail
    # The reason string mirrors the gate.
    assert result.reason.startswith("first failing gate `toolchain`")
    assert "nvcc" in result.reason
    # Cross-link to the conformance dashboard.
    assert "op_target_conformance.md" in result.reason


def test_rocm_matmul_names_toolchain_gate():
    result = cn.canonical_compile(_tiny_matmul_module(), target="rocm")
    assert result.executable is False
    assert result.first_failing_gate.gate == pg.GATE_TOOLCHAIN
    assert "hipcc" in result.first_failing_gate.detail


def test_metalium_matmul_names_link_gate():
    result = cn.canonical_compile(_tiny_matmul_module(), target="metalium")
    assert result.executable is False
    # Metalium's toolchain probe is intentionally not_evaluated; the first
    # FAIL is `link` (artifact_only — IR emits, no linked-kernel path).
    assert result.first_failing_gate.gate == pg.GATE_LINK


# ---- Pure aggregator ----

def test_module_is_pure_aggregator():
    """``canonical.py`` must only depend on the four declared compiler
    surfaces + stdlib. New compiler logic doesn't sneak in here — it lives
    in one of the four upstream modules first."""
    src = Path(cn.__file__).read_text()
    src_no_strings = re.sub(r'"""[\s\S]*?"""', '', src)
    src_no_strings = re.sub(r"#[^\n]*", "", src_no_strings)
    bare = re.findall(r"^import\s+([\w\.]+)", src_no_strings, flags=re.M)
    froms = re.findall(r"^from\s+([\w\.]+)\s+import\s+([\w\., ]+)",
                       src_no_strings, flags=re.M)
    resolved: list[str] = list(bare)
    for pkg, names in froms:
        for raw in names.split(","):
            leaf = raw.strip().split(" as ")[0].strip()
            if leaf:
                resolved.append(f"{pkg}.{leaf}" if pkg != "__future__"
                                else pkg)
    allowed_prefixes = (
        # The upstream truth sources C reconciles. ``op_catalog`` is the
        # source of each op's intrinsic ``effect`` / ``lowering`` family,
        # which the first-class ``effects`` compile metadata reconciles
        # (``_op_effect`` / ``_lowering_family``); it is a truth source,
        # not new compiler logic.
        "tessera.compiler.driver",
        "tessera.compiler.graph_ir",
        "tessera.compiler.pipeline_gates",
        "tessera.compiler.op_catalog",
        # Stdlib / typing.
        "__future__", "dataclasses", "pathlib", "typing",
    )
    for mod in resolved:
        assert any(mod.startswith(p) for p in allowed_prefixes), (
            f"canonical.py is supposed to be a pure aggregator; "
            f"import {mod!r} not in allowed truth-source set")


# ---- Primary op extraction ----

def test_primary_op_is_first_op_of_first_function():
    result = cn.canonical_compile(_tiny_matmul_module(), target="cpu")
    assert result.primary_op == "matmul"


def test_primary_op_strips_tessera_prefix():
    """The op_name in IR carries the ``tessera.`` prefix; the gate
    evaluator's manifest registry keys *without* the prefix. The wrapper
    must strip so the lookup matches."""
    module = _tiny_matmul_module()
    # Sanity: the raw op_name has the prefix in IR.
    assert module.functions[0].body[0].op_name.startswith("tessera.")
    # And the canonical layer surfaces it without the prefix.
    result = cn.canonical_compile(module, target="cpu")
    assert result.primary_op is not None
    assert not result.primary_op.startswith("tessera.")


# ---- to_dict round-trip ----

def test_to_dict_surface_is_stable():
    """Dashboards and telemetry will consume ``to_dict()``. Lock the
    field names so a downstream consumer (compile-report, runtime
    telemetry) doesn't drift unnoticed."""
    result = cn.canonical_compile(_tiny_matmul_module(), target="nvidia_sm90")
    d = result.to_dict()
    for key in (
        "target", "primary_op", "compiler_path", "executable", "reason",
        "runtime_status", "execution_kind", "execution_mode",
        "first_failing_gate", "first_failing_gate_detail",
        "gates", "artifact_hashes",
    ):
        assert key in d, f"to_dict missing key {key!r}"
    assert isinstance(d["gates"], list)
    assert all({"gate", "status", "detail"} <= set(g) for g in d["gates"])


# ---- One typed surface — the audit's payoff ----

def test_one_surface_carries_the_whole_answer():
    """The audit's framing: 'driver.py, matmul_pipeline.py, backend
    manifests, target maps, and runtime dispatch each hold part of the
    truth.' Verify that ONE call to canonical_compile produces a result
    that names: the IR (typed artifacts), the seven gates (capability
    set), and the executable/reason answer — without needing the caller
    to import driver / pipeline_gates / execution_matrix separately."""
    result = cn.canonical_compile(_tiny_matmul_module(), target="nvidia_sm90")
    # Typed artifacts.
    assert result.graph_ir
    # Capability set.
    assert len(result.gate_results) == len(pg.GATE_ORDER)
    # Executable / reason.
    assert result.executable is False
    assert result.reason  # non-empty
    assert result.first_failing_gate is not None
