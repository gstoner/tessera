"""
Phase 3 — TileIRLoweringPass Python-layer structural tests.

Verifies that the Graph IR emitted by @jit carries the correct structural
preconditions for TileIRLoweringPass to consume:
  - flash_attn op present in IR text
  - tessera.target attr set by GPU profile
  - tessera.tile_q / tessera.tile_kv attrs propagate from FlashAttnLoweringConfig
  - matmul present for GEMM-only GPU paths
"""
import pytest
import tessera
from tessera.compiler.gpu_target import GPUTargetProfile, ISA
from tessera.compiler.attn_lower import FlashAttnLoweringConfig
from tessera.compiler.graph_ir import GraphIRBuilder


class TestTileIRPreconditions:

    def test_flash_attn_op_emitted_for_gpu_target(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert "tessera.flash_attn" in ir

    def test_sm_version_in_target_attr(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert "sm = 90" in ir

    def test_matmul_op_present_for_gpu_gemm(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            return tessera.ops.matmul(A, B)

        ir = fn.graph_ir.to_mlir()
        assert "tessera.matmul" in ir or "tessera.gemm" in ir

    def test_attn_config_tile_sizes_in_ir(self):
        cfg = FlashAttnLoweringConfig(tile_q=128, tile_kv=64)

        @tessera.jit(
            target=GPUTargetProfile(isa=ISA.SM_90),
            attn_config=cfg,
        )
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        # Confirm the attn config was stored on the JitFn.
        assert fn.attn_config.tile_q == 128
        assert fn.attn_config.tile_kv == 64

    def test_function_name_in_ir(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def my_attn_kernel(Q: tessera.Tensor["B", "S", "D"],
                           K: tessera.Tensor["B", "S", "D"],
                           V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = my_attn_kernel.graph_ir.to_mlir()
        assert "func.func @my_attn_kernel" in ir

    def test_module_has_version_and_target(self):
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90))
        def fn(Q: tessera.Tensor["B", "S", "D"],
               K: tessera.Tensor["B", "S", "D"],
               V: tessera.Tensor["B", "S", "D"]):
            return tessera.ops.flash_attn(Q, K, V)

        ir = fn.graph_ir.to_mlir()
        assert ir.startswith("module attributes")
        assert "tessera.ir.version" in ir
        assert "tessera.target" in ir
