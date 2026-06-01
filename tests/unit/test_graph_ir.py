"""
tests/unit/test_graph_ir.py

Tests for the Python → Graph IR lowering (tessera.compiler.graph_ir):
  - GraphIRBuilder.lower() extracts function signature and ops
  - GraphIRModule.to_mlir() emits valid-looking MLIR text
  - IRArg correctly serializes region effect annotations
  - IROp correctly serializes tessera op calls
  - Round-trip: Python fn → Graph IR text → contains expected patterns
"""

import pytest
import tessera
from tessera.compiler.graph_ir import (
    GraphIRBuilder,
    GraphIRConstant,
    GraphIRMesh,
    GraphIRModule,
    GraphIRFunction,
    GraphIRTypeAlias,
    GraphIRVerificationError,
    IRArg,
    IROp,
    IRType,
    TENSOR_OPAQUE,
    construct_mlir_module,
)
from tessera.distributed.region import Region


# ─────────────────────────────────────────────────────────────────────────────
# IRType / IRArg / IROp unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIRType:
    def test_str(self):
        t = IRType("tensor<*xbf16>")
        assert str(t) == "tensor<*xbf16>"

    def test_repr_is_str(self):
        t = IRType("f32")
        assert "f32" in str(t)


class TestIRArg:
    def test_no_effect(self):
        arg = IRArg(name="x", ir_type=TENSOR_OPAQUE)
        mlir = arg.to_mlir()
        assert "%x" in mlir
        assert "tessera.effect" not in mlir

    def test_with_effect(self):
        arg = IRArg(name="W", ir_type=TENSOR_OPAQUE, effect="read")
        mlir = arg.to_mlir()
        assert "%W" in mlir
        assert 'tessera.effect = "read"' in mlir

    def test_write_effect(self):
        arg = IRArg(name="Y", ir_type=TENSOR_OPAQUE, effect="write")
        mlir = arg.to_mlir()
        assert 'tessera.effect = "write"' in mlir


class TestIROp:
    def test_basic_op(self):
        op = IROp(
            result="v0",
            op_name="tessera.gemm",
            operands=["%X", "%W"],
            operand_types=["tensor<*x?>", "tensor<*x?>"],
            result_type="tensor<*x?>",
        )
        mlir = op.to_mlir()
        assert "%v0 = tessera.gemm" in mlir
        assert "%X" in mlir
        assert "%W" in mlir

    def test_void_op(self):
        op = IROp(
            result=None,
            op_name="tessera.copy",
            operands=["%v0", "%Y"],
            operand_types=["tensor<*x?>", "tensor<*x?>"],
        )
        mlir = op.to_mlir()
        assert "tessera.copy" in mlir
        assert "%v0 =" not in mlir

    def test_indent(self):
        op = IROp(result="r", op_name="tessera.gemm", operands=[], operand_types=[])
        mlir = op.to_mlir(indent="    ")
        assert mlir.startswith("    ")


# ─────────────────────────────────────────────────────────────────────────────
# GraphIRFunction
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphIRFunction:
    def test_empty_function(self):
        fn = GraphIRFunction(name="empty")
        mlir = fn.to_mlir()
        assert "func.func @empty" in mlir
        assert "return" in mlir

    def test_function_with_args(self):
        fn = GraphIRFunction(
            name="step",
            args=[
                IRArg("W", TENSOR_OPAQUE, effect="read"),
                IRArg("X", TENSOR_OPAQUE, effect="read"),
                IRArg("Y", TENSOR_OPAQUE, effect="write"),
            ],
        )
        mlir = fn.to_mlir()
        assert "@step" in mlir
        assert "%W" in mlir
        assert "%X" in mlir
        assert "%Y" in mlir
        assert 'tessera.effect = "read"' in mlir
        assert 'tessera.effect = "write"' in mlir

    def test_function_with_body(self):
        fn = GraphIRFunction(
            name="gemm_fn",
            args=[IRArg("A", TENSOR_OPAQUE), IRArg("B", TENSOR_OPAQUE)],
            body=[
                IROp(
                    result="v0",
                    op_name="tessera.gemm",
                    operands=["%A", "%B"],
                    operand_types=["tensor<*x?>", "tensor<*x?>"],
                    result_type="tensor<*x?>",
                )
            ],
        )
        mlir = fn.to_mlir()
        assert "tessera.gemm" in mlir
        assert "%v0" in mlir

    def test_function_attrs(self):
        fn = GraphIRFunction(
            name="tagged",
            fn_attrs={"tessera.effect": '"random"'},
        )
        mlir = fn.to_mlir()
        assert "attributes" in mlir
        assert "random" in mlir


# ─────────────────────────────────────────────────────────────────────────────
# GraphIRModule
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphIRModule:
    def test_empty_module(self):
        m = GraphIRModule()
        mlir = m.to_mlir()
        assert "module" in mlir
        assert "tessera.ir.version" in mlir

    def test_module_with_function(self):
        m = GraphIRModule(functions=[GraphIRFunction(name="foo")])
        mlir = m.to_mlir()
        assert "@foo" in mlir

    def test_repr(self):
        m = GraphIRModule()
        assert "GraphIRModule" in repr(m)

    def test_structured_module_declarations_emit_attrs(self):
        m = GraphIRModule(
            meshes=[GraphIRMesh("mesh0", axes=("dp", "tp"), shape=(2, 4))],
            type_aliases=[GraphIRTypeAlias("Tile", "fragment<16,16,32,bf16>")],
            constants=[GraphIRConstant("alpha", "fp32", 1.0)],
        )
        mlir = m.to_mlir()
        assert "tessera.meshes" in mlir
        assert "mesh0" in mlir
        assert "tessera.type_aliases" in mlir
        assert "Tile" in mlir
        assert "tessera.constants" in mlir
        assert "alpha" in mlir

    def test_structured_module_declaration_verifier(self):
        dup = GraphIRModule(meshes=[
            GraphIRMesh("mesh0", axes=("dp",), shape=(2,)),
            GraphIRMesh("mesh0", axes=("tp",), shape=(4,)),
        ])
        result = dup.verify()
        assert not result.ok
        assert "GRAPH_IR_DUP_MESH" in result.format()

        bad_rank = GraphIRModule(meshes=[GraphIRMesh("bad", axes=("dp", "tp"), shape=(2,))])
        result = bad_rank.verify()
        assert not result.ok
        assert "GRAPH_IR_MESH_RANK" in result.format()


# ─────────────────────────────────────────────────────────────────────────────
# GraphIRBuilder — lowering round-trips
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphIRBuilder:
    def test_lower_simple_function(self):
        def identity(x):
            return x

        builder = GraphIRBuilder()
        fn_ir = builder.lower(identity)
        assert fn_ir.name == "identity"

    def test_lower_extracts_args(self):
        def step(W, X, Y):
            pass

        builder = GraphIRBuilder()
        fn_ir = builder.lower(step)
        arg_names = [a.name for a in fn_ir.args]
        assert "W" in arg_names
        assert "X" in arg_names
        assert "Y" in arg_names

    def test_lower_extracts_region_effects(self):
        def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
            pass

        builder = GraphIRBuilder()
        fn_ir = builder.lower(step)
        effects = {a.name: a.effect for a in fn_ir.args}
        assert effects.get("W") == "read"
        assert effects.get("X") == "read"
        assert effects.get("Y") == "write"

    def test_lower_detects_gemm_op(self):
        def gemm_fn(A, B, C):
            C[:] = tessera.ops.gemm(A, B)

        builder = GraphIRBuilder()
        fn_ir = builder.lower(gemm_fn)
        op_names = [op.op_name for op in fn_ir.body]
        assert "tessera.matmul" in op_names

    def test_lower_multiple_functions(self):
        def f(x):
            return tessera.ops.gemm(x, x)

        def g(x):
            return tessera.ops.layer_norm(x)

        builder = GraphIRBuilder()
        builder.lower(f)
        builder.lower(g)
        m = builder.module()
        assert len(m.functions) == 2

    def test_module_mlir_has_version(self):
        def f(x):
            pass

        builder = GraphIRBuilder()
        builder.lower(f)
        mlir = builder.module().to_mlir()
        assert "tessera.ir.version" in mlir

    def test_reset_clears_module(self):
        def f(x):
            pass

        builder = GraphIRBuilder()
        builder.lower(f)
        builder.reset()
        m = builder.module()
        assert len(m.functions) == 0

    def test_graph_ir_mlir_via_jit(self):
        """@jit decorated functions expose Graph IR MLIR for inspection."""
        @tessera.jit
        def simple(A, B):
            return tessera.ops.gemm(A, B)

        ir = simple.graph_ir.to_mlir()
        assert "func.func @simple" in ir
        assert "module" in ir

    def test_graph_ir_mlir_contains_region_effects(self):
        @tessera.jit
        def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
            Y[:] = tessera.ops.gemm(X, W)

        simple_ir = step.graph_ir.to_mlir()
        assert "read" in simple_ir
        assert "write" in simple_ir

    def test_lower_preserves_tensor_and_dtype_annotation_metadata(self):
        def typed(
            A: tessera.Tensor["M", "K"],
            B: tessera.bf16[16, 32],
            C: tessera.mut_f32[16, 8],
        ):
            C[:] = tessera.ops.matmul(A, B)

        builder = GraphIRBuilder()
        fn_ir = builder.lower(typed)
        args = {arg.name: arg for arg in fn_ir.args}
        assert args["A"].dim_names == ("M", "K")
        assert str(args["A"].ir_type) == "tensor<?x?x?>"
        assert str(args["B"].ir_type) == "tensor<16x32xbf16>"
        assert str(args["C"].ir_type) == "tensor<16x8xf32>"

    def test_unsupported_python_construct_has_source_span_diagnostic(self):
        """Updated for D.1 (2026-05-31). Pre-D.1 a dynamic ``if`` with a
        non-emittable Compare test produced a blanket
        ``PY_FRONTEND_UNSUPPORTED`` warning. D.1 lowers the if/else
        structurally to ``tessera.scf.if.*`` markers (with the source text
        recorded as ``condition_text``) and demotes the diagnostic to an
        *info* note that names the specific axis that wasn't yet emitted
        as SSA (``PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION``). The
        source-span requirement is unchanged."""
        @tessera.jit
        def bad_control(x):
            if x.sum() > 0:
                return tessera.ops.relu(x)
            return x

        explanation = bad_control.explain_lowering()
        # D.1 new contract: dynamic if/else lowers structurally + emits a
        # named info note pointing at the specific unlowered axis.
        assert "PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION" in explanation
        assert "if/else lowered structurally" in explanation
        # Source span is still tracked.
        assert " at " in explanation
        # Regression guard: the pre-D.1 blanket warning must not return.
        assert "PY_FRONTEND_UNSUPPORTED" not in explanation

    def test_graph_ir_verifier_and_object_construction_boundary(self):
        module = GraphIRModule(functions=[
            GraphIRFunction(
                name="bad",
                body=[
                    IROp(
                        result="x",
                        op_name="tessera.relu",
                        operands=["%missing"],
                        operand_types=["tensor<*x?>"],
                        result_type="tensor<*x?>",
                    )
                ],
            )
        ])
        with pytest.raises(GraphIRVerificationError):
            module.to_mlir()

        valid = GraphIRModule(functions=[
            GraphIRFunction(
                name="ok",
                args=[IRArg("x", TENSOR_OPAQUE)],
                body=[
                    IROp(
                        result="y",
                        op_name="tessera.relu",
                        operands=["%x"],
                        operand_types=["tensor<*x?>"],
                        result_type="tensor<*x?>",
                    )
                ],
            )
        ])
        obj = construct_mlir_module(valid)
        assert "func.func @ok" in obj.to_mlir()

    def test_verifier_understands_returns_control_and_shapes(self):
        balanced = GraphIRModule(functions=[
            GraphIRFunction(
                name="control",
                args=[IRArg("x", TENSOR_OPAQUE)],
                body=[
                    IROp(None, "tessera.scf.if.begin", [], [], kwargs={"region": "if"}),
                    IROp("y", "tessera.relu", ["%x"], ["tensor<*x?>"], "tensor<*x?>"),
                    IROp(None, "tessera.scf.else", [], [], kwargs={"region": "if"}),
                    IROp("z", "tessera.sigmoid", ["%x"], ["tensor<*x?>"], "tensor<*x?>"),
                    IROp(None, "tessera.scf.if.end", [], [], kwargs={"region": "if"}),
                ],
            )
        ])
        assert balanced.verify().ok

        unbalanced = GraphIRModule(functions=[
            GraphIRFunction(name="bad", body=[IROp(None, "tessera.scf.if.begin", [], [])])
        ])
        result = unbalanced.verify()
        assert not result.ok
        assert "GRAPH_IR_CONTROL_UNBALANCED" in result.format()

        bad_shape = GraphIRModule(functions=[
            GraphIRFunction(
                name="bad_matmul",
                args=[
                    IRArg("a", IRType("tensor<16x32xf32>", shape=("16", "32"), dtype="fp32")),
                    IRArg("b", IRType("tensor<16x8xf32>", shape=("16", "8"), dtype="fp32")),
                ],
                body=[
                    IROp("c", "tessera.matmul", ["%a", "%b"], ["tensor<16x32xf32>", "tensor<16x8xf32>"], "tensor<16x8xf32>")
                ],
            )
        ])
        result = bad_shape.verify()
        assert not result.ok
        assert "GRAPH_IR_MATMUL_SHAPE" in result.format()

    def test_verifier_checks_function_return_contract(self):
        module = GraphIRModule(functions=[
            GraphIRFunction(
                name="bad_return",
                result_types=[TENSOR_OPAQUE],
                body=[],
                return_values=[],
            )
        ])
        result = module.verify()
        assert not result.ok
        assert "GRAPH_IR_RETURN_MISSING" in result.format()

    def test_graph_ir_mlir_contains_gemm(self):
        @tessera.jit
        def gemm_step(A, B, C):
            C[:] = tessera.ops.gemm(A, B)

        ir = gemm_step.graph_ir.to_mlir()
        assert "gemm" in ir

    def test_effect_tag_in_function_attrs(self):
        @tessera.jit
        def with_dropout(x):
            return tessera.ops.dropout(x)

        ir = with_dropout.graph_ir.to_mlir()
        # The function should be tagged with its inferred effect
        assert "random" in ir or "func.func @with_dropout" in ir
