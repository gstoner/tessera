"""
Phase 3 — GPUTargetProfile validation tests.

Verifies that GPUTargetProfile correctly gates WGMMA/TMA capabilities,
validates warps_per_cta, and serialises to MLIR attribute strings.
"""
import pytest
from tessera.compiler.gpu_target import GPUTargetProfile, ISA, TesseraTargetError


class TestISACapabilities:

    def test_sm90_supports_wgmma(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        assert p.supports_wgmma is True

    def test_sm90_supports_tma(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        assert p.supports_tma is True

    def test_sm80_no_wgmma(self):
        p = GPUTargetProfile(isa=ISA.SM_80)
        assert p.supports_wgmma is False

    def test_sm80_no_tma(self):
        p = GPUTargetProfile(isa=ISA.SM_80)
        assert p.supports_tma is False

    def test_sm100_supports_wgmma(self):
        p = GPUTargetProfile(isa=ISA.SM_100)
        assert p.supports_wgmma is True

    def test_sm90_supports_mbarrier(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        assert p.supports_mbarrier is True

    def test_sm80_no_mbarrier(self):
        p = GPUTargetProfile(isa=ISA.SM_80)
        assert p.supports_mbarrier is False

    def test_rubin_placeholder_supports_low_precision_tensor_core_dtypes(self):
        p = GPUTargetProfile(isa=ISA.SM_120)
        for dtype in ("nvfp4", "fp4_e2m1", "fp6_e2m3", "fp6_e3m2", "fp8_e4m3", "fp8_e5m2"):
            assert p.supports_tensor_core_dtype(dtype)

    def test_isa_ordering(self):
        assert ISA.SM_90 > ISA.SM_80
        assert ISA.SM_100 > ISA.SM_90
        assert ISA.SM_120 > ISA.SM_100
        assert ISA.SM_86 < ISA.SM_90


class TestSharedMemory:

    def test_sm90_smem_capacity(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        assert p.max_smem_bytes == 233472  # 228 KB

    def test_sm80_smem_capacity(self):
        p = GPUTargetProfile(isa=ISA.SM_80)
        assert p.max_smem_bytes == 166912  # 163 KB

    def test_custom_smem_override(self):
        p = GPUTargetProfile(isa=ISA.SM_90, shared_mem_bytes=65536)
        assert p.max_smem_bytes == 65536

    def test_threads_per_cta(self):
        p = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)
        assert p.threads_per_cta == 128  # 4 × 32


class TestValidation:

    def test_invalid_warps_not_power_of_two(self):
        with pytest.raises(TesseraTargetError, match="power of 2"):
            GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=3)

    def test_invalid_warps_zero(self):
        with pytest.raises(TesseraTargetError):
            GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=0)

    def test_invalid_pipeline_stages(self):
        with pytest.raises(TesseraTargetError, match="pipeline_stages"):
            GPUTargetProfile(isa=ISA.SM_90, pipeline_stages=0)

    def test_valid_warps_8(self):
        p = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=8)
        assert p.warps_per_cta == 8


class TestMLIRAttrSerialization:

    def test_sm90_attr_contains_sm_version(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        attr = p.to_mlir_attr()
        assert "sm = 90" in attr

    def test_attr_contains_warps(self):
        p = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)
        attr = p.to_mlir_attr()
        assert "warps = 4" in attr

    def test_attr_contains_pipeline_stages(self):
        p = GPUTargetProfile(isa=ISA.SM_90, pipeline_stages=2)
        attr = p.to_mlir_attr()
        assert "pipeline_stages = 2" in attr

    def test_sm80_attr(self):
        p = GPUTargetProfile(isa=ISA.SM_80)
        attr = p.to_mlir_attr()
        assert "sm = 80" in attr
