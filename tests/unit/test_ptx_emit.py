"""Phase E1 rung-2.5 — NVIDIA WGMMA PTX emission (docs/audit/compiler/EVALUATOR_PLAN.md).

Portable structural tests (no toolchain) prove Tessera emits the documented
WGMMA encoding + PTX scaffolding — what earns rung 2.5. A ptxas-gated test
documents the rung-3 skip-clean behavior; real assembly of a *complete* kernel
is the named follow-up.
"""

from __future__ import annotations

import shutil

import pytest

from tessera.compiler import ptx_emit as P


def test_canonical_mnemonic_matches_inventory():
    assert P.wgmma_mnemonic(64, 256, 16) == (
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16"
    )


def test_emitted_ptx_validates_clean_for_all_documented_shapes():
    for m, n, k in [(64, 256, 16), (64, 128, 16), (64, 64, 16)]:
        ptx = P.emit_wgmma_matmul_ptx(m, n, k)
        assert P.wgmma_mnemonic(m, n, k) in ptx
        problems = P.validate_ptx_structure(ptx)
        assert problems == [], f"({m},{n},{k}): {problems}"


def test_emitted_ptx_has_version_target_and_protocol():
    ptx = P.emit_wgmma_matmul_ptx()
    assert ".version 9.3" in ptx
    assert ".target sm_90a" in ptx
    assert ".visible .entry" in ptx
    for op in ("wgmma.fence", "wgmma.commit_group", "wgmma.wait_group"):
        assert op in ptx


def test_uninventoried_shape_is_refused():
    with pytest.raises(ValueError, match="not a valid Hopper WGMMA"):
        P.emit_wgmma_matmul_ptx(13, 7, 3)


def test_validator_catches_a_broken_kernel():
    ptx = P.emit_wgmma_matmul_ptx()
    broken = ptx.replace("wgmma.fence.sync.aligned;", "")  # drop a mandatory op
    problems = P.validate_ptx_structure(broken)
    assert any("wgmma.fence" in p for p in problems)

    unbalanced = ptx.replace("ret;", "ret; {")
    assert any("brace" in p for p in P.validate_ptx_structure(unbalanced))


def test_validator_flags_wrong_arch():
    ptx = P.emit_wgmma_matmul_ptx(arch="sm_90a")
    problems = P.validate_ptx_structure(ptx, arch="sm_100a")
    assert any("sm_100a" in p for p in problems)


@pytest.mark.skipif(
    shutil.which("ptxas") is not None,
    reason="the WGMMA skeleton is not yet a complete kernel (needs smem "
    "descriptors + TMA); real ptxas assembly is the rung-3 follow-up. When the "
    "complete kernel lands, flip this to assert result.assembled.",
)
def test_ptxas_gate_skips_clean_without_toolchain():
    """On a host without ptxas (the arm64 dev Mac), the rung-3 gate must
    skip-clean rather than error — exactly like validate_nvcc_compile.py."""
    res = P.ptxas_assemble(P.emit_wgmma_matmul_ptx())
    assert res.status == "toolchain_absent"
    assert not res.assembled
