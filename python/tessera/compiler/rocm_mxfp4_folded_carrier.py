"""Schedule-bound Target materialization for folded gfx1201 MXFP4."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re

from .rocm_mxfp4 import FoldedRowReference
from .rocm_mxfp4_folded import (
    GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI,
    FoldedPrefillSchedule,
    _bind_folded_prefill_image,
)
from .rocm_mxfp4_native import (
    _schedule_hash, _target_integer_attr, _target_string_attr,
)
from .rocm_native import (
    ROCMNativePackage, _compile_native_tile_ir, _shape_free_target_ir, _directive_symbol,
)
from .rocm_pipeline import ROCMInputLevel
from .native_artifact import NativeImageArtifact, NativeEntryPoint


def package_folded_scaled_wmma_target_ir(
    tile_ir: str, target_ir: str, folded: FoldedRowReference, *,
    allow_approximate: bool = False, runtime_mn: bool = True, runtime_k: bool = True,
) -> ROCMNativePackage:
    """Materialize only the distinct, schedule-bound folded Target contract."""
    if not isinstance(runtime_mn, bool):
        raise TypeError("folded runtime_mn must be a bool")
    if not isinstance(runtime_k, bool):
        raise TypeError("folded runtime_k must be a bool")
    if runtime_k and not runtime_mn:
        raise ValueError("folded runtime_k requires runtime_mn")
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
    # The physical load schedule is a Target IR contract: the lowering chooses
    # it, the materializer only consumes it. Missing keys fail closed; values
    # outside the declared sets are refused by FoldedPrefillSchedule.
    schedule = FoldedPrefillSchedule(
        raster_group_m=_target_integer_attr(operation, "raster_group_m"),
        workgroup_mode=_target_string_attr(operation, "workgroup_mode"),
        staging_prefetch=_target_string_attr(operation, "staging_prefetch"),
        epilogue=_target_string_attr(operation, "epilogue_schedule"),
        row_guard=_target_string_attr(operation, "row_guard"),
    )
    # Payload/policy admission precedes compilation, including reserved E8M0
    # codes. Python binds artifacts; the native MLIR consumer owns the kernel.
    if not allow_approximate or folded.approximate_policy != "explicit_allow":
        raise ValueError("folded MXFP4 requires explicit approximate policy")
    if (folded.weight_bytes.shape != (n,k) or folded.row_reference.shape != (n,)
            or folded.weight_bytes.dtype.name != "uint8"
            or folded.row_reference.dtype.name != "uint8"
            or not folded.weight_bytes.flags.c_contiguous
            or not folded.row_reference.flags.c_contiguous
            or (folded.row_reference == 255).any()):
        raise ValueError("folded payload disagrees with its checked dimensions/storage/reference codes")
    authored_target_digest = hashlib.sha256(target_ir.encode()).hexdigest()
    image_target = (_shape_free_target_ir(target_ir, family="folded_matmul",
                        directive="tessera_rocm.scaled_wmma_gemm", runtime_k=runtime_k)
                    if runtime_mn else target_ir)
    entry = _directive_symbol(image_target, "tessera_rocm.scaled_wmma_gemm")
    native_target, backend, payload, compiler, toolchain, libraries, state = (
        _compile_native_tile_ir(image_target,
            directive="tessera_rocm.scaled_wmma_gemm", family="matmul",
            architecture="gfx1201", input_level=ROCMInputLevel.DIRECTIVE))
    digest = hashlib.sha256(native_target.encode()).hexdigest()
    image = NativeImageArtifact(
        target="rocm_gfx1201", architecture="gfx1201",
        pipeline_name="tessera-lower-to-rocm",
        compiler_fingerprint=compiler, toolchain_fingerprint=toolchain,
        target_ir_digest=digest, binary_format="hsaco", payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX_MXFP4_W4A8_FOLDED_PREFILL_ABI),),
        compile_state=state, device_libraries=libraries)
    package = _bind_folded_prefill_image(
        m, n, k, folded, image=image, entry=entry, source=native_target,
        backend_ir=backend, allow_approximate=allow_approximate, schedule=schedule)
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
            "image_input_level": "target",
            "producer_kind": "native_mlir_typed_lds",
            "native_compiler_owned": True,
            "kernel_argument_layout": "expanded_memref",
            "authored_target_ir_sha256": authored_target_digest,
            "image_shape_policy": "runtime_mnk" if runtime_k else "runtime_mn_fixed_k" if runtime_mn else "static_mnk",
            "image_m": 0 if runtime_mn else m,
            "image_n": 0 if runtime_mn else n,
            "image_k": 0 if runtime_k else k,
            "image_whole_m": m % 256 == 0,
            "image_whole_n": n % 64 == 0,
        },
    )
    return ROCMNativePackage(
        tile_ir=tile_ir, target_ir=native_target, backend_ir=backend,
        image=image, descriptor=descriptor,
    )


__all__ = ["package_folded_scaled_wmma_target_ir"]
