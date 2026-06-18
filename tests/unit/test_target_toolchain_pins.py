"""Sprint G-1 + H-1 — Target toolchain pins.

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

"""

from __future__ import annotations

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

    def test_sm120_is_blackwell_consumer_not_datacenter_superset(self):
        """SM_120 is Blackwell CONSUMER (RTX 50-series), NOT a superset of the
        datacenter sm_100. It lacks tcgen05 / TMEM (sm_100a-only) and Hopper's
        wgmma; its FP4 path is warp-level mma.sync.block_scale. Grounded in
        NVIDIA/cutlass#2800 + modular#5707."""
        from tessera.compiler.gpu_target import (
            GPUTargetProfile, ISA, cuda_feature_status)
        p120 = GPUTargetProfile(isa=ISA.SM_120)
        # The datacenter-only tensor-core features must be ABSENT on consumer sm_120.
        for absent in ("tcgen05", "tcgen05_pair", "tmem", "wgmma", "wgmma_sparse"):
            assert cuda_feature_status(ISA.SM_120, absent) == "not_supported", absent
            assert absent not in p120.cuda_features
        # But the consumer FP4 path (block-scaled mma.sync) IS present.
        assert cuda_feature_status(ISA.SM_120, "block_scaled_mma") == "ready"
        assert "nvfp4" in p120.tensor_core_dtypes
        assert "fp4_e2m1" in p120.tensor_core_dtypes
        assert p120.nvcc_arch == "sm_120a"
        # sm_100 datacenter has the tensor-memory path that sm_120 lacks — the
        # asymmetry that disproves the old "superset" assumption.
        assert cuda_feature_status(ISA.SM_100, "tcgen05") == "ready"


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
        assert ROCmTargetProfile(arch=AMDArch.GFX_1200).hipcc_arch == "gfx1200"


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

    def test_gfx1151_rdna35_strix_halo_wmma_no_fp8(self):
        """RDNA 3.5 (Strix Halo / Radeon 8060S) — WMMA, wave32, NO FP8 WMMA.

        Grounded in the RDNA3.5 ISA §7.9 Table 33: WMMA combos are
        F16/BF16/IU8/IU4 only; there is no FP8 WMMA instruction (the
        load-bearing difference from gfx1200/RDNA 4)."""
        from tessera.compiler.rocm_target import (
            ROCmTargetProfile, AMDArch, rocm_feature_status)
        p = ROCmTargetProfile(arch=AMDArch.GFX_1151)
        assert p.hipcc_arch == "gfx1151"
        assert p.is_rdna
        assert not p.supports_mfma          # RDNA has no MFMA
        assert p.supports_wmma              # has WMMA (f16/bf16)
        assert p.threads_per_wave == 32     # RDNA wavefront = 32
        # The ISA-grounded distinction: NO FP8 WMMA on RDNA 3.5.
        assert rocm_feature_status(AMDArch.GFX_1151, "wmma_f8") == "not_supported"
        assert "fp8_e4m3" not in p.dtype_set and "fp8_e5m2" not in p.dtype_set
        assert p.dtype_set == frozenset({"fp32", "bf16", "fp16", "int8"})
        assert p.lds_capacity_bytes == 65536

    def test_gfx1200_rdna4_wmma_f8(self):
        from tessera.compiler.rocm_target import ROCmTargetProfile, AMDArch
        p = ROCmTargetProfile(arch=AMDArch.GFX_1200)
        assert not p.supports_mfma
        assert p.supports_wmma
        assert p.threads_per_wave == 32
        assert p.dtype_set == frozenset({
            "fp32", "bf16", "fp16",
            "fp8_e4m3", "fp8_e5m2",
            "int8", "int32", "int4",
        })


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

    def test_rdna4_has_no_mfma_shapes(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        assert mfma_variants(AMDArch.GFX_1200) == frozenset()

    def test_rdna35_has_no_mfma_shapes(self):
        from tessera.compiler.rocm_target import mfma_variants, AMDArch
        assert mfma_variants(AMDArch.GFX_1151) == frozenset()


class TestROCmWMMAShapeTable:
    """RDNA WMMA shapes (the matrix path on RDNA, parallel to MFMA on CDNA)."""

    def test_rdna_arches_expose_16x16x16(self):
        from tessera.compiler.rocm_target import wmma_variants, AMDArch
        for arch in (AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200):
            assert wmma_variants(arch) == frozenset({(16, 16, 16)}), arch.name

    def test_cdna_arches_have_no_wmma(self):
        from tessera.compiler.rocm_target import wmma_variants, AMDArch
        for arch in (AMDArch.GFX_90A, AMDArch.GFX_940,
                     AMDArch.GFX_942, AMDArch.GFX_950):
            assert wmma_variants(arch) == frozenset(), arch.name

    def test_mfma_and_wmma_are_mutually_exclusive(self):
        """Every arch has MFMA xor WMMA shapes, never both, never neither."""
        from tessera.compiler.rocm_target import (
            mfma_variants, wmma_variants, AMDArch)
        for arch in AMDArch:
            has_mfma = bool(mfma_variants(arch))
            has_wmma = bool(wmma_variants(arch))
            assert has_mfma != has_wmma, (
                f"{arch.name}: mfma={has_mfma} wmma={has_wmma} (want exactly one)")


class TestROCmCapabilityRegistry:
    @pytest.mark.parametrize("name", [
        "rocm", "rocm_gfx90a", "rocm_gfx940",
        "rocm_gfx942", "rocm_gfx950", "rocm_gfx1100", "rocm_gfx1151",
        "rocm_gfx1200",
    ])
    def test_rocm_723_marker_present(self, name):
        cap = TARGET_CAPABILITIES[name]
        assert "rocm_7_2_3" in cap.features

    def test_gfx1151_strix_halo_wmma_no_fp8(self):
        cap = TARGET_CAPABILITIES["rocm_gfx1151"]
        assert "wmma_f16" in cap.features and "wmma_bf16" in cap.features
        assert "wmma_f8" not in cap.features        # no FP8 WMMA on RDNA 3.5
        assert "mfma" not in cap.features
        assert "fp8_e4m3" not in cap.supported_dtypes
        assert "strixhalo" in cap.aliases

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

    def test_gfx1200_has_wmma_f8_dtype_matrix(self):
        cap = TARGET_CAPABILITIES["rocm_gfx1200"]
        assert "wmma_f8" in cap.features
        assert "mfma" not in cap.features
        for dt in ("fp8_e4m3", "fp8_e5m2", "int32", "int4"):
            assert dt in cap.supported_dtypes


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
