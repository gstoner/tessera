from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.frontend import FrontendSemanticError, FrontendSyntaxError, lower_text_to_graph_ir, parse_text
from tessera.compiler.matmul_pipeline import build_cpu_plan


GRAPH_SOURCE = """
module demo {
  func main(A: tensor<?xfp32>, B: tensor<?xfp32>) -> tensor<?xfp32> {
    let C: tensor<?xfp32> = op.matmul(A, B);
    D = op.softmax(C) @{axis = -1};
    return D;
  }
}
"""


def test_parse_text_accepts_core_graph_dsl():
    program = parse_text(GRAPH_SOURCE)
    assert program.modules[0].name == "demo"
    fn = program.modules[0].funcs[0]
    assert fn.name == "main"
    assert fn.params[0].type_expr == "tensor<?xfp32>"
    assert fn.return_type == "tensor<?xfp32>"


def test_lower_text_to_graph_ir_emits_existing_graph_module():
    module = lower_text_to_graph_ir(GRAPH_SOURCE)
    text = module.to_mlir()
    assert "func.func @main" in text
    assert "tessera.matmul" in text
    assert "tessera.softmax" in text
    assert "axis = -1" in text


def test_textual_frontend_can_execute_cpu_plan():
    module = lower_text_to_graph_ir(GRAPH_SOURCE)
    plan = build_cpu_plan(module)
    assert plan is not None
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = plan.execute((a, b), {}, ["A", "B"])
    expected_raw = a @ b
    expected = np.exp(expected_raw - expected_raw.max(axis=-1, keepdims=True))
    expected = expected / expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected)


def test_textual_frontend_rejects_unknown_ops():
    with pytest.raises(FrontendSemanticError, match="unknown Tessera op"):
        lower_text_to_graph_ir("""
        module demo {
          func bad(A: tensor<?xfp32>) -> tensor<?xfp32> {
            B = op.nope(A);
            return B;
          }
        }
        """)


def test_textual_frontend_rejects_undefined_symbols():
    with pytest.raises(FrontendSemanticError, match="undefined symbol"):
        lower_text_to_graph_ir("""
        module demo {
          func bad(A: tensor<?xfp32>) -> tensor<?xfp32> {
            B = op.relu(C);
            return B;
          }
        }
        """)


def test_textual_frontend_rejects_wrong_arity_and_duplicates():
    with pytest.raises(FrontendSemanticError, match="expects 2-2 operands"):
        lower_text_to_graph_ir("""
        module demo {
          func bad(A: tensor<?xfp32>) -> tensor<?xfp32> {
            B = op.matmul(A);
            return B;
          }
        }
        """)

    with pytest.raises(FrontendSemanticError, match="duplicate symbol"):
        lower_text_to_graph_ir("""
        module demo {
          func bad(A: tensor<?xfp32>, A: tensor<?xfp32>) -> tensor<?xfp32> {
            B = op.relu(A);
            return B;
          }
        }
        """)


def test_textual_frontend_lowers_attention_and_state_reference_ops():
    source = """
    module demo {
      func stateful(Q: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>, Cache: tensor<?xfp32>) -> tensor<?xfp32> {
        A = op.flash_attn(Q, K, V) @{causal = true};
        B = op.all_gather(A);
        C = op.kv_cache_append(Cache, K, V);
        D = op.kv_cache_prune(C) @{max_entries = 1};
        return D;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    text = module.to_mlir()
    assert "tessera.flash_attn" in text
    assert "tessera.all_gather" in text
    assert "tessera.kv_cache.append" in text
    assert "tessera.kv_cache.prune" in text


def test_textual_frontend_preserves_shape_dtype_layout_and_mesh_metadata():
    source = """
    module demo {
      mesh data = mesh<axes=["dp"], shape=[2]>;
      func main(A: tensor<16x32xfp32;layout=row_major>, B: tensor<32x8xfp32>) -> tensor<16x8xfp32> {
        C = op.matmul(A, B);
        schedule.tile(C) @{m = 16, n = 8, k = 32};
        dist.all_reduce(C) @{axis = "dp", op = "sum"};
        barrier("after_reduce");
        assert(C);
        return C;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    text = module.to_mlir()
    assert module.meshes[0].name == "data"
    assert module.meshes[0].axes == ("dp",)
    assert module.meshes[0].shape == (2,)
    assert module.functions[0].return_values == ["%C"]
    assert "tensor<16x32xf32>" in text
    assert 'tessera.layout = "row_major"' in text
    assert "tensor<16x8xf32>" in text
    assert "tessera.meshes" in text
    assert "tessera.schedule.tile" in text
    assert "tessera.dist.all_reduce" in text
    assert "tessera.barrier" in text
    assert "tessera.assert" in text


def test_textual_frontend_parses_kernel_and_lowers_control_flow_markers():
    program = parse_text("""
    module demo {
      kernel k(A: tensor<?xfp32>) {
        return;
      }
    }
    """)
    assert program.modules[0].funcs[0].kind == "kernel"

    module = lower_text_to_graph_ir("""
    module demo {
      func control(A: tensor<?xfp32>) -> tensor<?xfp32> {
        if (A) {
          B = op.relu(A);
        } else {
          B = op.sigmoid(A);
        }
        for (i in 0:4:1) {
          schedule.pipeline(B) @{depth = 2};
        }
        while (B) {
          barrier("loop");
          return B;
        }
        return B;
      }
    }
    """)
    text = module.to_mlir()
    assert "tessera.scf.if.begin" in text
    assert "tessera.scf.else" in text
    assert "tessera.scf.for.begin" in text
    assert "tessera.scf.while.begin" in text
    assert "tessera.scf.yield" in text


def test_textual_frontend_accepts_full_normative_declaration_and_type_shapes():
    source = """
    module full {
      type Tile = fragment<16, 16, 32, bf16; layout=mma>;
      type Callback = fn(tensor<?xfp32>) -> tensor<?xfp32>;
      let alpha: fp32 = 1.0;
      mesh mesh0 = mesh<axes=[dp, tp], shape=[2, 4]>;

      kernel k(
        A: memref<16x32xbf16>[addrspace=shared],
        B: tensor<?x?xfp32; layout=row_major> = tensor<?x?xfp32; layout=row_major>
      ) -> (tensor<?x?xfp32>, tensor<?x?xfp32>) @{precision = strict} {
        let tmp: tensor<?x?xfp32>;
        tmp = op.arch.parameter() @{size = 4, dtype = "fp32"};
        tmp[0:1, :] = op.cast(B) @{dtype = "fp32"};
        dist.all_reduce(tmp) @{axis = "dp", op = "sum"};
        assert(tmp, "tmp exists");
        return tmp, B;
      }
    }
    """
    module = lower_text_to_graph_ir(source)
    text = module.to_mlir()
    assert module.type_aliases[0].name == "Tile"
    assert module.constants[0].name == "alpha"
    assert module.functions[0].fn_attrs["tessera.frontend.kind"] == '"kernel"'
    assert "tessera.type_aliases" in text
    assert "tessera.constants" in text
    assert "tessera.meshes" in text
    assert "tessera.frontend.kind" in text
    assert "tessera.graph.arch.parameter" in text
    assert "tessera.cast" in text
    assert "tessera.return_pack" in text


def test_textual_frontend_constructs_verified_graph_ir_object():
    module = lower_text_to_graph_ir("""
    module full_graph {
      mesh m = mesh<axes=[dp, tp], shape=[2, 4]>;
      type Pair = fn(tensor<?xfp32>) -> tensor<?xfp32>;
      let alpha: fp32 = 1.0;

      func main(A: tensor<4x8xfp32>, B: tensor<8x2xfp32>) -> tensor<4x2xfp32> {
        C = op.matmul(A, B);
        schedule.tile(C) @{m = 4, n = 2, k = 8};
        dist.all_reduce(C) @{axis = "dp", op = "sum"};
        if (C) {
          assert(C, "ok");
        }
        return C;
      }
    }
    """)
    assert module.verify().ok
    obj = module.to_mlir()
    assert "tessera.meshes" in obj
    assert "tessera.schedule.tile" in obj
    assert "tessera.scf.if.begin" in obj


def test_textual_frontend_rejects_static_matmul_shape_mismatch_at_graph_boundary():
    module = lower_text_to_graph_ir("""
    module bad {
      func main(A: tensor<4x8xfp32>, B: tensor<7x2xfp32>) -> tensor<4x2xfp32> {
        C = op.matmul(A, B);
        return C;
      }
    }
    """)
    result = module.verify()
    assert not result.ok
    assert "GRAPH_IR_MATMUL_SHAPE" in result.format()
    with pytest.raises(Exception, match="GRAPH_IR_MATMUL_SHAPE"):
        module.to_mlir()


def test_textual_frontend_lowers_tensor_and_binary_expressions():
    module = lower_text_to_graph_ir("""
    module demo {
      func exprs(A: tensor<?xfp32>) -> tensor<?xfp32> {
        X = tensor<?xfp32; layout=row_major>;
        Y = 1 + 2;
        Z = op.relu(A);
        return Z;
      }
    }
    """)
    text = module.to_mlir()
    assert "tessera.tensor.literal" in text
    assert "tessera.expr.add" in text


def test_textual_frontend_reports_syntax_line_column():
    with pytest.raises(FrontendSyntaxError) as excinfo:
        parse_text("""
        module demo {
          func bad(A: tensor<?xfp32>) {
            B = op.relu(@);
            return B;
          }
        }
        """)
    assert "at " in str(excinfo.value)
