"""
Phase 3 — WarpSpecializationPass Python-layer structural tests.

These verify the Python-side preconditions: that a flash_attn function
has the right Graph IR structure before C++ WarpSpecializationPass runs,
and that GPUTargetProfile properties gate the warp count correctly.
"""
import pytest
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA


class TestWarpSpecializationPreconditions:

    def test_sm90_profile_4_warps_default(self):
        p = GPUTargetProfile(isa=ISA.SM_90)
        assert p.warps_per_cta == 4

    def test_128_threads_per_cta_for_4_warps(self):
        """SM_90 warpgroup = 128 threads = 4 warps × 32."""
        p = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4)
        assert p.threads_per_cta == 128

    def test_flash_attn_ir_has_effect_attr(self):
        """WarpSpecializationPass expects tessera.effect on the func."""
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert "tessera.effect" in ir

    def test_flash_attn_ir_has_function_args(self):
        """The pass needs %Q, %K, %V as named args in the function."""
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def attn_fn(Q: tessera.Tensor["B", "S", "D"],
                    K: tessera.Tensor["B", "S", "D"],
                    V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = attn_fn.graph_ir.to_mlir()
        assert "%Q" in ir
        assert "%K" in ir
        assert "%V" in ir

    def test_sm90_wgmma_supported(self):
        """WarpSpecializationPass only emits WGMMA barriers on SM_90+."""
        assert GPUTargetProfile(isa=ISA.SM_90).supports_wgmma is True
        assert GPUTargetProfile(isa=ISA.SM_80).supports_wgmma is False

    def test_multiple_warps_per_cta_accepted(self):
        """Test 1, 2, 4, 8 warp configurations."""
        for w in [1, 2, 4, 8]:
            p = GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=w)
            assert p.warps_per_cta == w
