"""Explicit BF16 load/FP32 compute/BF16 store through native parent views."""
import base64
import json
import re

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tests.unit.test_native_gate_up_view_schedule import stages

pytestmark = pytest.mark.compiler_route


def bf16_source(m=3, width=17, reverse=False):
    def original(parent):
        left = ts.ops.slice(parent, [0, 0], [m, width])
        right = ts.ops.slice(parent, [0, width], [m, width])
        left_f32 = ts.ops.cast(left, "fp32")
        right_f32 = ts.ops.cast(right, "fp32")
        value = ts.ops.silu_mul(right_f32, left_f32) if reverse else ts.ops.silu_mul(left_f32, right_f32)
        return ts.ops.cast(value, "bf16")

    parent = np.ones((m, 2 * width), dtype=ml_dtypes.bfloat16)
    traced = trace(original, parent, arg_names=["parent"])
    assert traced.output_values == (None,)
    module = to_graph_ir_module(traced, name="gate_up_bf16", target="rocm_gfx1201")
    assert [op.op_name for op in module.functions[0].body].count("tessera.cast") == 3
    module.module_attrs.update({
        "tessera.native.gate_up_view": "unit",
        "tessera.target": json.dumps("rocm"),
        "tessera.arch": json.dumps("gfx1201"),
    })
    return module.to_mlir(target="rocm_gfx1201", canonical=True)


@pytest.mark.parametrize("m,width", [(3, 17), (16, 64)])
@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_bf16_parent_profile_preserves_cast_and_storage_ssa(m, width, reverse):
    graph = bf16_source(m, width, reverse)
    schedule, tile, target, gpu = stages(graph)
    assert "family=gate_up_view" in schedule
    assert "schema = 3" in tile and "storage = bf16" in tile
    assert tile.count("arith.extf") == 2 and tile.count("arith.truncf") == 1
    assert tile.count("tile.load_element") == 2 and tile.count("tile.store_element") == 1
    assert target.count("llvm.load") == 2 and target.count("llvm.store") == 1
    assert "memref<?xbf16>" in gpu and "math.exp" in gpu
    raw = re.search(r'tessera.rocm.parent_view_member_json = "([A-Za-z0-9+/=]+)"', gpu).group(1)
    member = json.loads(base64.b64decode(raw))
    assert member["profile"] == "gfx1201_parent_view_bf16_f32_v1"
    program = json.loads(base64.b64decode(member["program_base64"]))
    assert [buffer["storage"] for buffer in program["buffers"]] == ["bf16", "bf16"]
    assert [buffer["bytes"] for buffer in program["buffers"]] == [m * width * 4, m * width * 2]


def test_explicit_bf16_cast_graph_is_semantic_without_claiming_standalone_execution():
    from tessera.compiler.capabilities import supports_op

    graph = bf16_source()
    assert graph.count("tessera.cast") == 3
    assert "tensor<3x34xbf16>" in graph and "tensor<3x17xf32>" in graph
    semantic = supports_op("rocm_gfx1201", "cast", dtype="fp32", rank=2)
    assert semantic.supported and semantic.runtime_status == "artifact_only"
