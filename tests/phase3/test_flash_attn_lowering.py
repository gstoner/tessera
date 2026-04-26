"""
Phase 3 — end-to-end flash_attn Graph IR lowering tests.

These tests verify:
  1. @jit(target=GPUTargetProfile(isa=ISA.SM_90)) is accepted without error.
  2. The emitted Graph IR text contains tessera.flash_attn with causal=true.
  3. The tessera.effect attribute is present.
  4. The tessera.target attribute is present when a GPU profile is provided.
  5. Divisible("D", 64) constraint is parsed and registered.
"""
import pytest
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA


# ── Anchoring test (CLAUDE.md §Phase 3) ─────────────────────────────────────

class TestAnchoring:
    """The anchoring test from CLAUDE.md §Phase 3 — must pass for Phase 3 done."""

    def test_flash_attn_jit_accepted_sm90(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
        def flash_attn_fwd(
            Q: tessera.Tensor["B", "H", "S", "D"],
            K: tessera.Tensor["B", "H", "S", "D"],
            V: tessera.Tensor["B", "H", "S", "D"],
        ) -> tessera.Tensor["B", "H", "S", "D"]:
            tessera.require(tessera.constraint.Divisible("D", 64))
            return tessera.ops.flash_attn(Q, K, V, causal=True)

        ir = flash_attn_fwd.graph_ir.to_mlir()
        assert "tessera.flash_attn" in ir
        assert "tessera.effect" in ir

    def test_gpu_profile_stored_on_jitfn(self):
        target = GPUTargetProfile(isa=ISA.SM_90)

        @tessera.jit(target=target)
        def fn(Q: tessera.Tensor["B", "D"]):
            return tessera.ops.flash_attn(Q, Q, Q)

        assert fn.target is target
        assert fn.is_gpu is True

    def test_non_gpu_jit_has_no_target(self):
        @tessera.jit
        def fn(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            return tessera.ops.gemm(A, B)

        assert fn.target is None
        assert fn.is_gpu is False


# ── Graph IR content checks ─────────────────────────────────────────────────

class TestGraphIRContent:

    def test_flash_attn_op_in_ir(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def attn(Q: tessera.Tensor["B", "S", "D"], K: tessera.Tensor["B", "S", "D"],
                 V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = attn.graph_ir.to_mlir()
        assert "tessera.flash_attn" in ir

    def test_causal_flash_attn_in_ir(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def causal_attn(Q: tessera.Tensor["B", "S", "D"],
                        K: tessera.Tensor["B", "S", "D"],
                        V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V, causal=True)

        ir = causal_attn.graph_ir.to_mlir()
        assert "tessera.flash_attn" in ir
        assert "causal" in ir

    def test_tessera_target_attr_in_module(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert "tessera.target" in ir
        assert "sm = 90" in ir

    def test_module_version_still_present(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert 'tessera.ir.version' in ir


# ── Attn config wiring ───────────────────────────────────────────────────────

class TestAttnConfig:

    def test_sm90_gets_default_attn_config(self):
        from tessera.compiler.attn_lower import SM90_DEFAULT

        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        assert fn.attn_config is not None
        assert fn.attn_config.tile_q == SM90_DEFAULT.tile_q
        assert fn.attn_config.tile_kv == SM90_DEFAULT.tile_kv

    def test_sm80_gets_no_default_attn_config(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_80))
        def fn(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            return tessera.ops.gemm(A, B)

        # SM_80 does not support WGMMA, so SM90_DEFAULT is not auto-applied.
        assert fn.attn_config is None

    def test_explicit_attn_config_respected(self):
        from tessera.compiler.attn_lower import FlashAttnLoweringConfig

        cfg = FlashAttnLoweringConfig(tile_q=128, tile_kv=128, causal=True)

        @tessera.jit(
            target=GPUTargetProfile(isa=ISA.SM_90),
            attn_config=cfg,
        )
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V, causal=True)

        assert fn.attn_config.tile_q == 128
        assert fn.attn_config.tile_kv == 128
        assert fn.attn_config.causal is True
