from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

import tessera as ts
from tessera.compiler.matmul_pipeline import build_cpu_plan, normalize_target_kind
from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.gpu_target import GPUTargetProfile, ISA


def test_target_kind_normalization_accepts_planned_backend_aliases():
    assert normalize_target_kind(None) == "cpu"
    assert normalize_target_kind("cuda") == "nvidia_sm90"
    assert normalize_target_kind("nvidia") == "nvidia_sm90"
    assert normalize_target_kind("gpu") == "nvidia_sm90"
    assert normalize_target_kind("sm90") == "nvidia_sm90"
    assert normalize_target_kind("sm100") == "nvidia_sm100"
    assert normalize_target_kind("sm120") == "nvidia_sm120"
    assert normalize_target_kind(GPUTargetProfile(isa=ISA.SM_100)) == "nvidia_sm100"
    assert normalize_target_kind("hip") == "rocm"
    assert normalize_target_kind("tt_metalium") == "metalium"
    assert normalize_target_kind("macos_cpu") == "apple_cpu"
    assert normalize_target_kind("m-series-gpu") == "apple_gpu"

    with pytest.raises(ValueError, match="unsupported Tessera target"):
        normalize_target_kind("quantum_waffle")


def test_jit_rocm_target_emits_mfma_and_async_copy_artifact():
    @ts.jit(target="rocm")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert not mm.is_executable
    assert mm.has_target_artifacts
    assert "JIT_TARGET_IR_ARTIFACT_ONLY" in mm.explain_lowering()
    assert 'target = "rocm"' in mm.target_ir
    assert "tessera_rocm.mfma" in mm.target_ir
    assert "tessera_rocm.async_copy" in mm.target_ir
    assert "tessera_rocm.wait" in mm.target_ir

    artifact = mm.runtime_artifact()
    assert artifact.metadata["target"] == "rocm"
    assert artifact.metadata["compiler_path"] == "target_ir_artifact"
    assert artifact.metadata["runtime_status"] == "artifact_only"


def test_jit_hopper_target_emits_wgmma_tma_and_mbarrier_artifact():
    @ts.jit(target=GPUTargetProfile(isa=ISA.SM_90))
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert not mm.is_executable
    assert mm.has_target_artifacts
    assert 'target = "nvidia_sm90"' in mm.target_ir
    assert 'arch = "sm_90a"' in mm.target_ir
    assert "tessera_nvidia.wgmma" in mm.target_ir
    assert "tessera_nvidia.tma_async_copy" in mm.target_ir
    assert "tessera_nvidia.mbarrier" in mm.target_ir

    artifact = mm.runtime_artifact()
    assert artifact.metadata["target"] == "nvidia_sm90"
    assert artifact.metadata["runtime_status"] == "artifact_only"


def test_jit_blackwell_target_emits_tcgen05_and_tmem_artifact():
    @ts.jit(target=GPUTargetProfile(isa=ISA.SM_100))
    def mm(A, B):
        return ts.ops.gemm(A, B)

    assert 'target = "nvidia_sm100"' in mm.target_ir
    assert 'arch = "sm_100a"' in mm.target_ir
    assert "tessera_nvidia.tmem_alloc" in mm.target_ir
    assert "tessera_nvidia.tcgen05_mma" in mm.target_ir
    assert 'block_scaled = true' in mm.target_ir


def test_jit_metalium_target_emits_dma_and_matmul_artifact():
    @ts.jit(target="metalium")
    def mm(A, B):
        return ts.ops.gemm(A, B)

    assert "tessera.matmul" in mm.ir_text()
    assert 'target = "metalium"' in mm.target_ir
    assert "tessera_metalium.dma" in mm.target_ir
    assert "tessera_metalium.matmul" in mm.target_ir
    assert "tile.mma" in mm.tile_ir


def test_jit_apple_cpu_target_emits_accelerate_artifact():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert 'target = "apple_cpu"' in mm.target_ir
    assert "tessera_apple.cpu.accelerate_gemm" in mm.target_ir
    assert 'framework = "Accelerate"' in mm.target_ir
    assert 'abi = "cblas_sgemm"' in mm.target_ir


def test_jit_apple_gpu_target_emits_mps_runtime_for_single_matmul():
    """Phase 8.3 — a single rank-2 f32 matmul flips the apple_gpu target IR
    from the artifact-only metal_kernel contract to the MPS runtime contract:
    mps_matmul + mps_dispatch with execution_mode="metal_runtime"."""

    @ts.jit(target="apple_gpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    assert 'target = "apple_gpu"' in mm.target_ir
    assert "tessera_apple.gpu.mps_matmul" in mm.target_ir
    assert "tessera_apple.gpu.mps_dispatch" in mm.target_ir
    assert 'execution_mode = "metal_runtime"' in mm.target_ir


def test_jit_apple_gpu_matmul_softmax_chain_emits_fused_msl_runtime_contract():
    """Phase 8.4.3 — the matmul -> softmax chain is the first recognized
    fusion. Both ops collapse into a single fused MSL kernel, flipping the
    program from metal_artifact to metal_runtime."""

    @ts.jit(target="apple_gpu")
    def fused(A, B):
        return ts.ops.softmax(ts.ops.matmul(A, B))

    assert 'target = "apple_gpu"' in fused.target_ir
    assert "tessera_apple.gpu.msl_kernel" in fused.target_ir
    assert 'entry_point = "matmul_softmax_f32"' in fused.target_ir
    assert 'fusion = "matmul_softmax"' in fused.target_ir
    assert 'execution_mode = "metal_runtime"' in fused.target_ir


def test_jit_apple_gpu_unrecognized_multi_op_keeps_metal_artifact_contract():
    """Multi-op programs that don't match a known fusion pattern stay on
    the artifact path. The recognized fusion set has grown over phases —
    softmax -> matmul (suffix-only, no matmul head) is currently NOT a
    recognized pattern, so it makes a stable negative case."""

    @ts.jit(target="apple_gpu")
    def chain(x, w):
        return ts.ops.matmul(ts.ops.softmax(x), w)

    assert 'target = "apple_gpu"' in chain.target_ir
    assert "tessera_apple.gpu.metal_kernel" in chain.target_ir
    assert "tessera_apple.gpu.dispatch" in chain.target_ir
    assert 'artifact = "metallib"' in chain.target_ir


def test_flash_attention_apple_gpu_gets_msl_runtime_contract():
    """Phase 8.4.1 — single flash_attn programs flip from the artifact-only
    metal_kernel contract to the metal_runtime MSL kernel contract: the
    Target IR carries the MSL source as a StringAttr on
    tessera_apple.gpu.msl_kernel and the runtime executes via a custom
    MTLComputePipelineState."""

    @ts.jit(target="apple_gpu")
    def flash(q, k, v):
        return ts.ops.flash_attn(q, k, v, causal=True)

    assert "tessera.flash_attn" in flash.ir_text()
    assert "schedule.pipeline.region" in flash.schedule_ir
    assert "tessera.attn.online_softmax" in flash.tile_ir
    assert "tessera_apple.gpu.msl_kernel" in flash.target_ir
    assert 'entry_point = "flash_attn_f32"' in flash.target_ir
    assert "kernel void flash_attn_f32" in flash.target_ir
    assert 'execution_mode = "metal_runtime"' in flash.target_ir


def test_kv_cache_target_lowering_uses_stable_unsupported_diagnostics():
    source = """
    module decode {
      func step(Cache: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>) -> tensor<?xfp32> {
        C = op.kv_cache_append(Cache, K, V);
        return C;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    plan = build_cpu_plan(module, target_kind="rocm")

    assert plan is not None
    assert "tile.kv_cache" in plan.tile_ir
    assert "tessera.target.diagnostic" in plan.target_ir
    assert 'target = "rocm"' in plan.target_ir
    assert "KV-cache target lowering is not implemented" in plan.target_ir


def test_apple_cpu_matmul_target_executes_accelerate_backend():
    @ts.jit(target="apple_cpu")
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(2, dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)

    np.testing.assert_allclose(mm(A, B), A @ B)
    assert mm.is_executable
    artifact = mm.runtime_artifact()
    assert artifact.metadata["compiler_path"] == "apple_cpu_accelerate"
    assert artifact.metadata["runtime_status"] == "ready"


def test_static_target_contract_files_define_backend_spine():
    root = Path(__file__).resolve().parents[2]
    apple = (
        root
        / "src/compiler/codegen/Tessera_Apple_Backend/include/Tessera/Target/Apple/TesseraAppleOps.td"
    ).read_text(encoding="utf-8")
    metalium_cmake = (
        root / "src/compiler/codegen/Tessera_Metalium_Backend/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    rocm_passes = (
        root / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp"
    ).read_text(encoding="utf-8")

    assert "cpu.accelerate_gemm" in apple
    assert "gpu.metal_kernel" in apple
    assert "gpu.dispatch" in apple
    assert "TesseraMetaliumOpsIncGen" in metalium_cmake
    assert "tessera-lower-to-rocm" in rocm_passes
