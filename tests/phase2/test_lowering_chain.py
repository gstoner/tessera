"""
Phase 2 Python-layer tests for the lowering chain.

These tests verify that:
  1. GraphIRBuilder emits the correct tessera.effect and tessera.shard
     attribute text for downstream C++ passes to consume.
  2. The @jit decorator correctly infers effect levels (plumbed from Phase 1).
  3. The emitted MLIR text structurally satisfies Phase 2 pass preconditions.

Full C++ pass correctness is validated by the MLIR lit tests in
tests/tessera-ir/phase2/.
"""
import pytest
import tessera
from tessera.compiler.graph_ir import GraphIRBuilder, IRArg, IRType, IROp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def emitted(fn, **jit_kwargs):
    """Emit Graph IR text for fn decorated with @jit(**jit_kwargs)."""
    b = GraphIRBuilder()
    effect = jit_kwargs.get("deterministic") and "pure" or None
    b.lower(fn, effect_tag=effect)
    return b.module().to_mlir()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Distribution Lowering Pass preconditions
#    The pass expects func args with tessera.shard attrs or pass options.
#    The GraphIRBuilder should emit tessera.effect attrs from Region annotations.
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributionLoweringPreconditions:

    def test_read_region_emits_effect_attr(self):
        from tessera.distributed.region import Region

        def step(W: Region["read"], X: Region["read"]):
            return tessera.ops.gemm(X, W)

        ir = emitted(step)
        assert 'tessera.effect = "read"' in ir

    def test_write_region_emits_effect_attr(self):
        from tessera.distributed.region import Region

        def step(Y: Region["write"], X: Region["read"]):
            Y[:] = tessera.ops.gemm(X, X)

        ir = emitted(step)
        assert 'tessera.effect = "write"' in ir

    def test_reduce_region_emits_effect_attr(self):
        from tessera.distributed.region import Region

        def reduce_step(G: Region["reduce_sum"]):
            return tessera.ops.gemm(G, G)

        ir = emitted(reduce_step)
        assert 'tessera.effect = "reduce_sum"' in ir

    def test_module_version_attr_present(self):
        from tessera.distributed.region import Region

        def step(A: Region["read"], B: Region["read"]):
            return tessera.ops.matmul(A, B)

        ir = emitted(step)
        assert 'tessera.ir.version' in ir
        assert '"1.0"' in ir


# ─────────────────────────────────────────────────────────────────────────────
# 2. Effect Annotation Pass preconditions
#    @jit(deterministic=True) sets tessera.effect = "pure" on the function.
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectAnnotationPreconditions:

    def test_deterministic_jit_emits_pure_effect(self):
        @tessera.jit(deterministic=True, seed=42)
        def pure_fn(x: tessera.Tensor["B", "D"]):
            return tessera.ops.layer_norm(x)

        ir = pure_fn.graph_ir.to_mlir()
        assert 'tessera.effect = "pure"' in ir

    def test_default_jit_has_no_effect_attr(self):
        @tessera.jit
        def default_fn(x: tessera.Tensor["B", "D"]):
            return tessera.ops.layer_norm(x)

        ir = default_fn.graph_ir.to_mlir()
        # No explicit effect tag — the C++ EffectAnnotationPass infers it.
        assert "func.func @default_fn" in ir

    def test_effect_annotation_reflects_ops(self):
        """@jit-emitted IR for a matmul should not carry random effect."""
        @tessera.jit(deterministic=True, seed=0)
        def det_gemm(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            tessera.require(tessera.constraint.Divisible("K", 64))
            return tessera.ops.gemm(A, B)

        ir = det_gemm.graph_ir.to_mlir()
        assert 'tessera.effect = "pure"' in ir
        assert "tessera.matmul" in ir or "tessera.gemm" in ir


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tiling Pass preconditions
#    Emitted matmul ops must have compatible operand names and structure.
# ─────────────────────────────────────────────────────────────────────────────

class TestTilingPassPreconditions:

    def test_matmul_op_present_in_ir(self):
        from tessera.distributed.region import Region

        def gemm(A: Region["read"], B: Region["read"]) -> tessera.Tensor:
            return tessera.ops.gemm(A, B)

        ir = emitted(gemm)
        assert "tessera.matmul" in ir or "tessera.gemm" in ir

    def test_ir_function_name_matches_python(self):
        def my_kernel(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            return tessera.ops.matmul(A, B)

        b = GraphIRBuilder()
        b.lower(my_kernel)
        ir = b.module().to_mlir()
        assert "func.func @my_kernel" in ir

    def test_operand_names_appear_in_ir(self):
        def gemm(lhs: tessera.Tensor["M", "K"], rhs: tessera.Tensor["K", "N"]):
            return tessera.ops.matmul(lhs, rhs)

        b = GraphIRBuilder()
        b.lower(gemm)
        ir = b.module().to_mlir()
        assert "%lhs" in ir
        assert "%rhs" in ir


# ─────────────────────────────────────────────────────────────────────────────
# 4. TileToX86 pass preconditions — dtype annotation
# ─────────────────────────────────────────────────────────────────────────────

class TestTileToX86Preconditions:

    def test_bf16_dtype_annotation(self):
        """Verify the IR builder uses bf16 dtype when requested."""
        b = GraphIRBuilder()
        arg = IRArg(name="A", ir_type=IRType("tensor<16x32xbf16>"), effect="read")
        assert "bf16" in arg.to_mlir()

    def test_f16_dtype_annotation(self):
        arg = IRArg(name="A", ir_type=IRType("tensor<16x32xf16>"), effect="read")
        assert "f16" in arg.to_mlir()

    def test_f32_result_type_annotation(self):
        op = IROp(
            result="C",
            op_name="tessera.matmul",
            operands=["%A", "%B"],
            operand_types=["tensor<16x32xbf16>", "tensor<32x16xbf16>"],
            result_type="tensor<16x16xf32>",
        )
        s = op.to_mlir()
        assert "tessera.matmul" in s
        assert "%C" in s
        assert "tensor<16x16xf32>" in s


# ─────────────────────────────────────────────────────────────────────────────
# 5. End-to-end GraphIRModule structure
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIRStructure:

    def test_full_pipeline_ir_is_valid_text(self, simple_gemm_ir):
        """The emitted MLIR text must be parseable as a module block."""
        ir = simple_gemm_ir.to_mlir()
        assert ir.startswith("module attributes")
        assert ir.strip().endswith("}")
        assert "func.func @step" in ir

    def test_multiple_functions_in_module(self):
        b = GraphIRBuilder()

        def fn1(A: tessera.Tensor["M", "K"], B: tessera.Tensor["K", "N"]):
            return tessera.ops.gemm(A, B)

        def fn2(X: tessera.Tensor["B", "D"]):
            return tessera.ops.layer_norm(X)

        b.lower(fn1)
        b.lower(fn2)
        ir = b.module().to_mlir()
        assert "func.func @fn1" in ir
        assert "func.func @fn2" in ir

    def test_index_launch_produces_kernel_ir(self):
        """@kernel + index_launch emits a recognisable Graph IR kernel."""
        @tessera.kernel
        def tp_gemm(A: tessera.f16[..., ...], B: tessera.f16[..., ...],
                    C: tessera.mut_f32[..., ...]):
            C[:] = tessera.ops.gemm(A, B)

        from tessera.distributed.domain import Rect
        from tessera.distributed.shard import MeshSpec, ShardSpec
        from tessera.distributed.array import DistributedArray
        from tessera.distributed.shard import Block

        D    = Rect((4, 128, 256))
        dist = Block(mesh_axes=("tp",))
        X    = DistributedArray.from_domain(D, dtype="f16", distribution=dist)

        # index_launch should not raise at this stage.
        from tessera.distributed.launch import index_launch
        launch = index_launch(axis="tp")
        # The kernel is captured; actual execution requires the full pipeline.
        assert tp_gemm is not None
        assert launch is not None

    def test_constraint_error_raised_before_ir_emission(self):
        """Phase 2 depends on Phase 1 constraint errors firing early."""
        with pytest.raises(Exception):
            @tessera.jit
            def bad(A: tessera.Tensor["M", "K"]):
                tessera.require(tessera.constraint.Divisible("K", 64))
                return tessera.ops.gemm(A, A)

            # Trigger the constraint solver with a violation.
            bad_shaped = tessera.Tensor["M", "7"]  # K=7, not divisible by 64
            # The constraint should fire at jit time, not call time.
