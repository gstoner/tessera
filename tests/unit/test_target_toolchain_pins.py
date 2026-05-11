"""Sprint G-1 + H-1 + I-1/I-2/I-3 — Target toolchain pins.

Locks the per-target hardware-free pre-work landed 2026-05-11:

  G-1: NVIDIA backend pinned to CUDA 13.2 Update 1 — per-SM feature
       matrix (wgmma_sparse / tcgen05_pair / cluster_launch /
       tma_swizzle_128b / mbarrier_arrive_tx / cp_async_bulk /
       async_proxy_fence / block_scaled_mma) + arch strings
       (sm_90a/sm_100a/sm_120a) + dtype matrix updated.

  H-1: ROCm backend pinned to ROCm 7.2.3 — per-arch feature matrix
       (mfma_f8 / mfma_xf32 / mfma_f4 / mfma_f6 / lds_async_copy /
       cluster_mode) + arch strings (gfx90a/gfx940/gfx942/gfx950/
       gfx1100) + dtype matrix + MFMA shape variants.

  I-1: Metalium kernel inventory expanded — softmax / layer_norm /
       rmsnorm lit fixtures landed under
       `src/compiler/codegen/Tessera_Metalium_Backend/test/metalium/`.

  I-2: BackendKernelManifest gains the `compileable` status + Metalium
       block-FP planned/gated entries (`bfp8`, `bfp4`) surfaced as a
       separate `metalium_blockfp` target so the audit walker correctly
       classifies them as planned_gated.

  I-3: `docs/metalium_kernel_inventory.md` documents RISC-V grid
       mapping, shipped kernel surface, dtype matrix, and execution
       gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.capabilities import TARGET_CAPABILITIES, get_target_capability


# ──────────────────────────────────────────────────────────────────────────
#               G-1: NVIDIA CUDA 13.2 U1 capability matrix
# ──────────────────────────────────────────────────────────────────────────

class TestCUDA13ToolchainPin:
    def test_target_versions_recorded(self):
        from tessera.compiler.gpu_target import (
            TESSERA_TARGET_CUDA_TOOLKIT,
            TESSERA_TARGET_CUDA_DRIVER_MIN,
            TESSERA_TARGET_PTX_ISA,
            TESSERA_TARGET_NCCL_MIN,
        )
        assert TESSERA_TARGET_CUDA_TOOLKIT == "13.2.1"
        # Driver pin should be a 3-part version string like 555.85 etc.
        assert "." in TESSERA_TARGET_CUDA_DRIVER_MIN
        assert TESSERA_TARGET_PTX_ISA == "8.6"
        assert TESSERA_TARGET_NCCL_MIN == "2.22"

    def test_nvcc_arch_strings(self):
        from tessera.compiler.gpu_target import GPUTargetProfile, ISA
        assert GPUTargetProfile(isa=ISA.SM_80).nvcc_arch == "sm_80"
        assert GPUTargetProfile(isa=ISA.SM_90).nvcc_arch == "sm_90a"
        assert GPUTargetProfile(isa=ISA.SM_100).nvcc_arch == "sm_100a"
        assert GPUTargetProfile(isa=ISA.SM_120).nvcc_arch == "sm_120a"


class TestCUDA13FeatureMatrix:
    """Per-SM feature flags under CUDA 13.2 U1."""

    def test_sm80_baseline_features(self):
        from tessera.compiler.gpu_target import GPUTargetProfile, ISA
        p = GPUTargetProfile(isa=ISA.SM_80)
        # Ampere: WMMA only.
        assert not p.supports_wgmma
        assert not p.supports_tma
        assert not p.supports_cluster_launch
        assert not p.supports_tcgen05_pair
        assert "wmma" in p.cuda_features

    def test_sm90_hopper_features(self):
        from tessera.compiler.gpu_target import GPUTargetProfile, ISA
        p = GPUTargetProfile(isa=ISA.SM_90)
        # Hopper: full WGMMA + TMA + clusters.
        assert p.supports_wgmma
        assert p.supports_wgmma_sparse
        assert p.supports_tma
        assert p.supports_tma_swizzle_128b
        assert p.supports_cluster_launch
        assert p.supports_mbarrier_arrive_tx
        assert p.supports_cp_async_bulk
        assert p.supports_async_proxy_fence
        # Not yet (Blackwell-only):
        assert not p.supports_tcgen05_pair

    def test_sm100_blackwell_features(self):
        from tessera.compiler.gpu_target import GPUTargetProfile, ISA
        p = GPUTargetProfile(isa=ISA.SM_100)
        assert p.supports_tcgen05_pair
        assert p.supports_tmem
        assert p.supports_block_scaled_mma
        # Hopper features carry forward:
        assert p.supports_wgmma_sparse
        assert p.supports_tma_swizzle_128b
        assert p.supports_cluster_launch

    def test_sm120_rubin_inherits_blackwell(self):
        from tessera.compiler.gpu_target import GPUTargetProfile, ISA
        p100 = GPUTargetProfile(isa=ISA.SM_100)
        p120 = GPUTargetProfile(isa=ISA.SM_120)
        # Rubin is at least a superset of Blackwell in CUDA 13.2 U1.
        for feature in p100.cuda_features:
            assert feature in p120.cuda_features, f"{feature} regressed on SM_120"


class TestNVIDIACapabilityRegistry:
    """capabilities.py entries pinned to CUDA 13.2 U1."""

    @pytest.mark.parametrize("name", [
        "nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120",
    ])
    def test_cuda_13_2_marker_present(self, name):
        cap = TARGET_CAPABILITIES[name]
        assert "cuda_13_2_u1" in cap.features

    def test_sm90_has_wgmma_features(self):
        cap = TARGET_CAPABILITIES["nvidia_sm90"]
        for flag in ("wgmma", "wgmma_sparse", "tma", "tma_swizzle_128b",
                     "cluster_launch", "mbarrier_arrive_tx", "cp_async_bulk"):
            assert flag in cap.features, f"{flag} missing from nvidia_sm90"

    def test_sm100_has_blackwell_features(self):
        cap = TARGET_CAPABILITIES["nvidia_sm100"]
        for flag in ("tcgen05", "tcgen05_pair", "tmem", "block_scaled_mma"):
            assert flag in cap.features

    def test_sm100_dtype_set_includes_low_precision(self):
        cap = TARGET_CAPABILITIES["nvidia_sm100"]
        for dt in ("fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2",
                   "fp4_e2m1", "nvfp4"):
            assert dt in cap.supported_dtypes, f"{dt} missing from SM_100 dtypes"

    def test_sm90_has_fp8_no_fp4(self):
        cap = TARGET_CAPABILITIES["nvidia_sm90"]
        assert "fp8_e4m3" in cap.supported_dtypes
        # Hopper doesn't have FP4/FP6 lanes.
        assert "fp4_e2m1" not in cap.supported_dtypes
        assert "nvfp4" not in cap.supported_dtypes


# ──────────────────────────────────────────────────────────────────────────
#               H-1: ROCm 7.2.3 capability matrix
# ──────────────────────────────────────────────────────────────────────────

class TestROCmToolchainPin:
    def test_target_versions_recorded(self):
        from tessera.compiler.rocm_target import (
            TESSERA_TARGET_ROCM,
            TESSERA_TARGET_HIP,
            TESSERA_TARGET_RCCL_MIN,
        )
        assert TESSERA_TARGET_ROCM == "7.2.3"
        assert TESSERA_TARGET_HIP == "7.2.3"
        assert TESSERA_TARGET_RCCL_MIN == "2.22"

    def test_hipcc_arch_strings(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        assert ROCmTargetProfile(arch=AMDArch.GFX_90A).hipcc_arch == "gfx90a"
        assert ROCmTargetProfile(arch=AMDArch.GFX_940).hipcc_arch == "gfx940"
        assert ROCmTargetProfile(arch=AMDArch.GFX_942).hipcc_arch == "gfx942"
        assert ROCmTargetProfile(arch=AMDArch.GFX_950).hipcc_arch == "gfx950"
        assert ROCmTargetProfile(arch=AMDArch.GFX_1100).hipcc_arch == "gfx1100"


class TestROCmFeatureMatrix:
    def test_gfx90a_baseline_mfma(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        p = ROCmTargetProfile(arch=AMDArch.GFX_90A)
        assert p.supports_mfma
        assert not p.supports_mfma_f8      # CDNA 2 has no FP8
        assert not p.supports_mfma_xf32
        assert not p.supports_mfma_f4
        assert not p.supports_cluster_mode

    def test_gfx942_mi300x_mfma_f8(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        p = ROCmTargetProfile(arch=AMDArch.GFX_942)
        assert p.supports_mfma
        assert p.supports_mfma_f8
        assert p.supports_mfma_xf32
        assert p.supports_lds_async_copy
        # MI300X (CDNA 3) lacks the CDNA 4-only features.
        assert not p.supports_mfma_f4
        assert not p.supports_mfma_f6
        assert not p.supports_cluster_mode

    def test_gfx950_mi325x_full_cdna4(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        p = ROCmTargetProfile(arch=AMDArch.GFX_950)
        for prop in ("supports_mfma", "supports_mfma_f8",
                     "supports_mfma_xf32", "supports_mfma_f4",
                     "supports_mfma_f6", "supports_lds_async_copy",
                     "supports_cluster_mode"):
            assert getattr(p, prop), f"{prop} expected True on gfx950"

    def test_gfx1100_rdna3_wmma_only(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        p = ROCmTargetProfile(arch=AMDArch.GFX_1100)
        assert not p.supports_mfma     # RDNA has no MFMA
        assert p.supports_wmma          # but does have WMMA
        assert p.threads_per_wave == 32   # RDNA wavefront = 32
        assert p.dtype_set == frozenset({"fp32", "bf16", "fp16", "int8"})


class TestROCmMFMAShapeTable:
    def test_cdna2_shapes_minimal(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        shapes = mfma_variants(AMDArch.GFX_90A)
        assert (32, 32, 8, 1) in shapes
        assert (16, 16, 16, 1) in shapes
        assert len(shapes) == 2

    def test_cdna3_adds_f8_xf32(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        shapes = mfma_variants(AMDArch.GFX_942)
        # K=16 / K=32 are the f8 variants
        assert (32, 32, 16, 1) in shapes
        assert (16, 16, 32, 1) in shapes
        # K=4 / K=8 are xf32 variants
        assert (32, 32, 4, 1) in shapes
        assert (16, 16, 8, 1) in shapes
        assert len(shapes) == 6

    def test_cdna4_adds_f4_lanes(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        shapes = mfma_variants(AMDArch.GFX_950)
        # CDNA 4 FP4 lanes (K=32 / K=64)
        assert (32, 32, 32, 1) in shapes
        assert (16, 16, 64, 1) in shapes
        assert len(shapes) == 8

    def test_rdna3_has_no_mfma_shapes(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        assert mfma_variants(AMDArch.GFX_1100) == frozenset()


class TestROCmCapabilityRegistry:
    @pytest.mark.parametrize("name", [
        "rocm", "rocm_gfx90a", "rocm_gfx940",
        "rocm_gfx942", "rocm_gfx950", "rocm_gfx1100",
    ])
    def test_rocm_723_marker_present(self, name):
        cap = TARGET_CAPABILITIES[name]
        assert "rocm_7_2_3" in cap.features

    def test_gfx950_has_f4_f6(self):
        cap = TARGET_CAPABILITIES["rocm_gfx950"]
        assert "mfma_f4" in cap.features
        assert "mfma_f6" in cap.features
        assert "cluster_mode" in cap.features
        for dt in ("fp4_e2m1", "fp6_e2m3", "fp6_e3m2"):
            assert dt in cap.supported_dtypes

    def test_gfx942_has_f8_no_f4(self):
        cap = TARGET_CAPABILITIES["rocm_gfx942"]
        assert "mfma_f8" in cap.features
        assert "mfma_f4" not in cap.features
        assert "fp8_e4m3" in cap.supported_dtypes
        assert "fp4_e2m1" not in cap.supported_dtypes

    def test_gfx1100_wmma_only(self):
        cap = TARGET_CAPABILITIES["rocm_gfx1100"]
        assert "wmma_f16" in cap.features
        assert "mfma" not in cap.features


# ──────────────────────────────────────────────────────────────────────────
#               I-1: Metalium kernel lit fixtures present
# ──────────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[2]
METALIUM_TEST_DIR = (
    REPO / "src" / "compiler" / "codegen" / "Tessera_Metalium_Backend"
    / "test" / "metalium"
)


class TestMetaliumLitFixtures:
    @pytest.mark.parametrize("fixture", [
        "softmax_to_metalium.mlir",
        "layer_norm_to_metalium.mlir",
        "rmsnorm_to_metalium.mlir",
    ])
    def test_fixture_exists(self, fixture):
        path = METALIUM_TEST_DIR / fixture
        assert path.exists(), f"{fixture} not found at {path}"

    @pytest.mark.parametrize("fixture", [
        "softmax_to_metalium.mlir",
        "layer_norm_to_metalium.mlir",
        "rmsnorm_to_metalium.mlir",
    ])
    def test_fixture_has_metalium_dialect_ops(self, fixture):
        body = (METALIUM_TEST_DIR / fixture).read_text()
        # Lowering target — both DMA and matmul should appear.
        assert "tessera_metalium.dma" in body
        assert "tessera_metalium.matmul" in body
        # FileCheck directive sanity
        assert "RUN: tessera-metalium-opt" in body
        assert "REQUIRES: tessera_metalium_opt" in body

    @pytest.mark.parametrize("fixture", [
        "softmax_to_metalium.mlir",
        "layer_norm_to_metalium.mlir",
        "rmsnorm_to_metalium.mlir",
    ])
    def test_fixture_uses_bf16(self, fixture):
        """Sprint I-1 fixtures all use bf16 as the canonical reasoning-
        model storage dtype."""
        body = (METALIUM_TEST_DIR / fixture).read_text()
        assert "bf16" in body
        # element_size_bytes = 2 is the bf16 DMA descriptor signature
        assert "element_size_bytes = 2" in body


# ──────────────────────────────────────────────────────────────────────────
#               I-2: Metalium dtype matrix + planned/gated entries
# ──────────────────────────────────────────────────────────────────────────

class TestMetaliumManifest:
    def test_matmul_has_metalium_artifact(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "metalium" in entries
        m = entries["metalium"]
        assert m.status == "artifact_only"
        assert "bf16" in m.dtypes
        assert "fp32" in m.dtypes
        assert "tile_local_matmul" in m.feature_flags

    def test_matmul_has_block_fp_planned_entry(self):
        from tessera.compiler.backend_manifest import manifest_for
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "metalium_blockfp" in entries
        m = entries["metalium_blockfp"]
        assert m.status == "planned"
        assert "bfp8" in m.dtypes
        assert "bfp4" in m.dtypes
        assert "block_fp" in m.feature_flags

    def test_softmax_layer_norm_rmsnorm_metalium_artifacts(self):
        from tessera.compiler.backend_manifest import manifest_for
        for op_name in ("softmax", "layer_norm", "rmsnorm"):
            entries = {e.target: e for e in manifest_for(op_name)}
            assert "metalium" in entries, f"{op_name} missing metalium entry"
            assert entries["metalium"].status == "artifact_only"
            assert "bf16" in entries["metalium"].dtypes


class TestMetaliumAuditWalker:
    def test_planned_gated_bfp_entries_classified_correctly(self):
        from tessera.compiler.backend_manifest import audit_backend_dtypes
        buckets = audit_backend_dtypes()
        # bfp8 / bfp4 should be in the planned_gated bucket, NOT unknown.
        names_seen = {dt for _, _, dt in buckets["planned_gated"]}
        assert "bfp8" in names_seen
        assert "bfp4" in names_seen
        # And none of them should leak into the unknown bucket.
        for _, _, dt in buckets["unknown"]:
            assert dt not in ("bfp8", "bfp4", "blockfp8", "blockfp4"), (
                f"{dt} should be planned_gated, not unknown"
            )

    def test_zero_unknown_dtypes_after_metalium_extension(self):
        from tessera.compiler.backend_manifest import audit_backend_dtypes
        buckets = audit_backend_dtypes()
        assert buckets["unknown"] == [], (
            f"Sprint I-2 must not introduce unknown dtypes; got: "
            f"{buckets['unknown'][:5]}"
        )


# ──────────────────────────────────────────────────────────────────────────
#               I-3: kernel inventory doc present
# ──────────────────────────────────────────────────────────────────────────

class TestMetaliumKernelInventoryDoc:
    def test_doc_exists(self):
        doc = REPO / "docs" / "metalium_kernel_inventory.md"
        assert doc.exists(), f"Sprint I-3 doc missing at {doc}"

    def test_doc_covers_required_sections(self):
        doc = REPO / "docs" / "metalium_kernel_inventory.md"
        body = doc.read_text()
        # Required sections
        for section in (
            "RISC-V grid mapping",
            "Shipped kernel inventory",
            "Phase 7",
            "Sprint I-1",
            "block-floating-point",
            "bfp8",
            "bfp4",
            "Execution gates",
        ):
            assert section in body, f"Section {section!r} missing from doc"

    def test_doc_mentions_per_core_roles(self):
        doc = REPO / "docs" / "metalium_kernel_inventory.md"
        body = doc.read_text()
        # Per-RISC-V-core role documentation
        for role in ("BRISC", "NCRISC", "TRISC0", "Tensix"):
            assert role in body, f"role {role} missing"


# ──────────────────────────────────────────────────────────────────────────
#               Cross-sprint: BackendKernelEntry compileable status
# ──────────────────────────────────────────────────────────────────────────

class TestCompileableStatus:
    """Sprint G/H follow-ups will promote NVIDIA/ROCm artifact_only
    entries to `compileable` once `nvcc -ptx` / `hipcc -S` validation
    lands.  The status itself is registered now."""

    def test_compileable_status_accepted(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        # Should construct cleanly.
        entry = BackendKernelEntry(
            target="nvidia_sm90",
            status="compileable",
            dtypes=("bf16", "fp16"),
            feature_flags=("wgmma",),
            notes="ptxas -arch=sm_90a passes",
        )
        assert entry.status == "compileable"

    def test_invalid_status_rejected(self):
        from tessera.compiler.backend_manifest import BackendKernelEntry
        with pytest.raises(ValueError, match="status must be one of"):
            BackendKernelEntry(target="cpu", status="halfway_there")
