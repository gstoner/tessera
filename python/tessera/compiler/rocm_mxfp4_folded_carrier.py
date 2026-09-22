"""Schedule-bound Target materialization for folded gfx1201 MXFP4."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re

from .rocm_mxfp4 import FoldedRowReference
from .rocm_mxfp4_folded import (
    GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI,
    package_mxfp4_folded_prefill,
)
from .rocm_mxfp4_native import (
    _schedule_hash, _target_integer_attr, _target_string_attr,
)
from .rocm_native import ROCMNativePackage


def package_folded_scaled_wmma_target_ir(
    tile_ir: str, target_ir: str, folded: FoldedRowReference, *,
    allow_approximate: bool = False,
) -> ROCMNativePackage:
    """Materialize only the distinct, schedule-bound folded Target contract."""
    physical = "rocm_mxfp4_w4a8_folded_prefill_v1"
    directives = [
        line.strip() for line in target_ir.splitlines()
        if "tessera_rocm.scaled_wmma_gemm" in line
    ]
    carriers = [
        line.strip() for line in tile_ir.splitlines()
        if "tile.scaled_matmul_kernel" in line
    ]
    if len(directives) != 1 or len(carriers) != 1:
        raise ValueError("folded packaging requires exactly one Tile and Target carrier")
    operation, tile_operation = directives[0], carriers[0]
    for name, expected in {
        "abi": "a_bfold_sa_rowref_d_m_n_k",
        "physical_contract": physical,
        "package_abi": GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI,
        "scale_format": "e8m0_row_reference",
        "partial_combine": "row_reference_after_full_k",
        "k_step_schedule": "isolated_k_stage",
        "output": "bf16",
    }.items():
        if _target_string_attr(operation, name) != expected:
            raise ValueError(f"folded Target IR requires {name}={expected!r}")
    for name, expected in {
        "physical_contract": physical,
        "combine": "row_reference_after_full_k",
        "scope": "full_k",
        "schedule_scope": "k_stage",
        "init": "zero",
        "cross_step_motion": "forbid",
    }.items():
        if _target_string_attr(tile_operation, name) != expected:
            raise ValueError(f"folded Tile IR requires {name}={expected!r}")
    tile_hash = _schedule_hash(tile_operation, carrier="folded Tile IR")
    target_hash = _schedule_hash(operation, carrier="folded Target IR")
    if tile_hash != target_hash:
        raise ValueError("folded Tile/Target schedule hashes disagree")
    integers = {
        name: _target_integer_attr(operation, name)
        for name in (
            "m", "n", "k", "instruction_k", "scale_k", "macro_k", "stage_k",
            "block_m", "block_n", "tile_m_per_wave", "tile_n_per_wave",
        )
    }
    m, n, k = integers["m"], integers["n"], integers["k"]
    if min(m, n, k) <= 0 or m <= 64 or k % 64:
        raise ValueError("folded Target IR requires M>64, positive N, K divisible by 64")
    for name, expected_int in {
        "instruction_k": 16, "scale_k": k, "macro_k": k, "stage_k": 64,
        "block_m": 256, "block_n": 64,
        "tile_m_per_wave": 4, "tile_n_per_wave": 2,
    }.items():
        if integers[name] != expected_int:
            raise ValueError(f"folded Target IR requires {name}={expected_int}")
    for name, expected_int in {
        "instruction_steps": k // 16,
        "tessera.problem_m": m, "tessera.problem_n": n,
        "tessera.problem_k": k,
        "tessera.macro_tile_m": 256, "tessera.macro_tile_n": 64,
        "warps": 8,
    }.items():
        if _target_integer_attr(tile_operation, name) != expected_int:
            raise ValueError(f"folded Tile IR requires {name}={expected_int}")
    policy_match = re.search(r"\bnumeric_policy\s*=\s*\{([^}]*)\}", operation)
    if policy_match is None:
        raise ValueError("folded Target IR requires numeric_policy")
    for name, expected in {
        "accum": "f32", "storage": "e4m3_raw_u8",
        "execution_mode": "folded_row_reference_explicit_approximate",
    }.items():
        if _target_string_attr(policy_match.group(1), name) != expected:
            raise ValueError(f"folded Target IR requires numeric_policy.{name}={expected!r}")
    package = package_mxfp4_folded_prefill(
        m, n, k, folded, allow_approximate=allow_approximate,
    )
    digest = hashlib.sha256(target_ir.encode()).hexdigest()
    image = replace(package.image, target_ir_digest=digest)
    descriptor = replace(
        package.descriptor,
        image_digest=image.image_digest,
        provenance={
            **package.descriptor.provenance,
            "materializer": "tessera_rocm.scaled_wmma_gemm",
            "physical_contract": physical,
            "tile_ir_sha256": hashlib.sha256(tile_ir.encode()).hexdigest(),
            "target_ir_sha256": digest,
            "schedule_hash": target_hash,
        },
    )
    return ROCMNativePackage(
        tile_ir=tile_ir, target_ir=target_ir, backend_ir=package.target_ir,
        image=image, descriptor=descriptor,
    )


__all__ = ["package_folded_scaled_wmma_target_ir"]
