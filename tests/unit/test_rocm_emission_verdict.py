"""Workstream #15 — AMD emission rung in the Evaluator program.

rocdl_emit already emits llvm.amdgcn.wmma IR and `llc` lowers it to real gfx
AMDGCN (v_wmma_*) on this host. This wires that into the Evaluator's rung ladder
via `rocm_emission_verdict`, parallel to `nvidia_emission_verdict` — so AMD's
genuine rung-4 (ASSEMBLES) progress is scored, not invisible.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (#15) and EVALUATOR_PLAN.md §2.
"""

from __future__ import annotations

import shutil

import pytest

from tessera.compiler.evaluator import rocm_emission_verdict, Rung
from tessera.compiler import rocdl_emit


def _llc_available() -> bool:
    from tessera.compiler.rocdl_emit import _find_llc
    try:
        return _find_llc() is not None
    except Exception:
        return shutil.which("llc") is not None


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_rocm_verdict_reaches_assembles_when_llc_present(arch):
    v = rocm_emission_verdict(arch=arch, dtype="f16")
    assert v.target == f"rocm:{arch}"
    if _llc_available():
        # The LLVM 22 AMDGPU backend assembles WMMA on this host → rung 4.
        assert v.rung == Rung.ASSEMBLES, v.detail
        assert v.execution_kind == "amdgcn_assembled"
        assert v.runtime_status == "assembled"
    else:
        assert v.rung == Rung.EMITS_ASM_TEXT, v.detail


def test_rocm_verdict_never_claims_execution_or_correctness():
    # No silicon here — provenance must stay False and correctness unproven,
    # regardless of how far the host toolchain assembles.
    v = rocm_emission_verdict(arch="gfx1151", dtype="f16")
    assert v.provenance_ok is False
    assert v.correctness == "unproven"
    assert v.rung < Rung.EXECUTES   # can't claim rungs 6-7 without a GPU


def test_rocm_verdict_rung_at_least_emits_text():
    # Even without llc, structurally-valid emission earns rung 3.
    v = rocm_emission_verdict(arch="gfx1151", dtype="f16")
    assert v.rung >= Rung.EMITS_ASM_TEXT


def test_rdna4_and_gfx1250_archs_emit():
    # The newer arch families also emit structurally-valid IR (different ABIs).
    for arch in ("gfx1201", "gfx1250"):
        ir = (rocdl_emit.emit_wmma_rdna4_llvmir("f16", arch=arch)
              if arch == "gfx1201"
              else rocdl_emit.emit_wmma_gfx1250_llvmir("f16", arch=arch))
        assert "llvm.amdgcn.wmma" in ir
