"""Machine-readable public projections of exact-device ROCm proof.

Content-addressed compiler families and scheduled native ABIs are intentionally
broader than the public capability surface.  A row belongs here only when one
public operation, one runtime compiler path, and one exact-device numerical
fixture describe the same bounded contract.  This keeps family promotion from
silently turning into a public support claim.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ROCmExactDeviceProof:
    target: str
    op_name: str
    compiler_path: str
    executor_id: str
    op_family: str
    dtypes: tuple[str, ...]
    numerical_fixture: str
    proof_build: str
    scheduled_abis: tuple[str, ...]
    reason: str


GFX1201_PUBLIC_PROOFS: tuple[ROCmExactDeviceProof, ...] = (
    ROCmExactDeviceProof(
        target="rocm_gfx1201",
        op_name="tessera.matmul",
        compiler_path="rocm_compiled",
        executor_id="rocm_gfx1201_compiled",
        op_family="matmul",
        # Public input-storage spellings proved by the scheduled packages.
        # f32/i32 are accumulator/output forms, not independently proved
        # input contracts, so they must not become supports_op() claims.
        dtypes=("bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"),
        numerical_fixture="tests/unit/test_rocm_gfx1201_scheduled.py",
        proof_build="llvm23.1.1+rocm10.0+gfx1201",
        scheduled_abis=(
            "tessera.rocm.matmul.a_b_o_m_n_k.f16_f32.v1",
            "tessera.rocm.matmul.a_b_bias_o_m_n_k.f16_f32.fused.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.bf16_f32.v1",
            "tessera.rocm.matmul.a_b_bias_o_m_n_k.bf16_f32.fused.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.e4m3_f32.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.e5m2_f32.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.e4m3_e5m2_f32.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.e5m2_e4m3_f32.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.i8_i32.v1",
            "tessera.rocm.matmul.a_b_o_m_n_k.i4_i32.v1",
        ),
        reason=(
            "RX 9070 XT exact-device scheduled matmul packages compile, launch, "
            "and match numerical oracles across the registered RDNA4 storage forms"
        ),
    ),
    ROCmExactDeviceProof(
        target="rocm_gfx1201",
        op_name="tessera.flash_attn",
        compiler_path="rocm_flash_attn_compiled",
        executor_id="rocm_gfx1201_flash_attn_compiled",
        op_family="attention",
        dtypes=("bf16", "fp16"),
        numerical_fixture="tests/unit/test_rocm_gfx1201_scheduled.py",
        proof_build="llvm23.1.1+rocm10.0+gfx1201",
        scheduled_abis=(
            "tessera.rocm.attention.q_k_v_o_dims.f16_f32out.v1",
            "tessera.rocm.attention.q_k_v_o_dims.bf16_f32out.v1",
        ),
        reason=(
            "RX 9070 XT exact-device scheduled fp16/bf16 attention packages "
            "compile, launch, and match numerical oracles"
        ),
    ),
    ROCmExactDeviceProof(
        target="rocm_gfx1201",
        op_name="tessera.softmax",
        compiler_path="rocm_softmax_compiled",
        executor_id="rocm_gfx1201_softmax_compiled",
        op_family="softmax",
        dtypes=("fp32",),
        numerical_fixture="tests/unit/test_rocm_gfx1201_scheduled.py",
        proof_build="llvm23.1.1+rocm10.0+gfx1201",
        scheduled_abis=("tessera.rocm.softmax.x_o_rows_k.f32.v1",),
        reason=(
            "RX 9070 XT exact-device scheduled f32 softmax package compiles, "
            "launches, and matches its numerical oracle"
        ),
    ),
)


def public_proofs_for(target: str) -> tuple[ROCmExactDeviceProof, ...]:
    return tuple(proof for proof in GFX1201_PUBLIC_PROOFS
                 if proof.target == target)


__all__ = ["GFX1201_PUBLIC_PROOFS", "ROCmExactDeviceProof", "public_proofs_for"]
