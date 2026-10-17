"""Original gate/up Graph SSA crosses native Schedule/Tile/Target boundaries."""
import base64
import json
import re

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.compiler_route


def source(m=3, width=17, reverse=False):
    def original(parent):
        left = ts.ops.slice(parent, [0, 0], [m, width])
        right = ts.ops.slice(parent, [0, width], [m, width])
        return ts.ops.silu_mul(right, left) if reverse else ts.ops.silu_mul(left, right)

    parent = np.arange(m * 2 * width, dtype=np.float32).reshape(m, 2 * width)
    traced = trace(original, parent, arg_names=["parent"])
    assert traced.output_values == (None,)
    module = to_graph_ir_module(traced, name="gate_up", target="rocm_gfx1201")
    module.module_attrs.update({
        "tessera.native.gate_up_view": "unit",
        "tessera.target": json.dumps("rocm"),
        "tessera.arch": json.dumps("gfx1201"),
    })
    return module.to_mlir(target="rocm_gfx1201", canonical=True)


def tool():
    result = find_tessera_opt()
    assert result is not None, "matching production tessera-opt required"
    return result


def stages(text):
    compiler = tool()
    schedule = run_tessera_opt(compiler, text, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(compiler, schedule, "--tessera-schedule-to-tile")
    target = run_tessera_opt(compiler, tile, "--lower-tile-to-rocm=arch=gfx1201")
    gpu = run_tessera_opt(compiler, target, "--lower-tessera-target-to-rocdl")
    return schedule, tile, target, gpu


@pytest.mark.parametrize("m,width", [(3, 17), (16, 64), (256, 12288)])
@pytest.mark.parametrize("reverse", [False, True])
def test_parent_views_reach_actual_gpu_ssa(m, width, reverse):
    schedule, tile, target, gpu = stages(source(m, width, reverse))
    assert "family=gate_up_view" in schedule and "tessera.slice" in schedule
    assert "tile.row_region_kernel" in tile
    # Original Graph text is retained as a provenance string, never an executable op.
    assert not re.search(r"^\s*%[A-Za-z0-9_]+\s*=\s*tessera\.(?:slice|dynamic_slice)\b", tile, re.MULTILINE)
    assert 'arch = "gfx1201"' in tile and "schema = 2" in tile
    assert tile.count("= tile.buffer_view ") == 2
    assert tile.count("tile.load_element") == 2 and tile.count("tile.store_element") == 1
    assert "tessera_rocm.row_region_kernel" in target and "tile.buffer_view" not in target
    assert target.count("llvm.load") == 2 and target.count("llvm.store") == 1
    assert "gpu.func" in gpu and "gpu.thread_id" in gpu and "gpu.block_id" in gpu
    assert "math.exp" in gpu and "arith.divf" in gpu and "arith.mulf" in gpu
    assert "tessera_rocm.row_region_kernel" not in gpu and "llvm.call" not in gpu
    manifest_text = re.search(r'tessera.rocm.parent_view_member_json = "([A-Za-z0-9+/=]+)"', gpu).group(1)
    manifest = json.loads(base64.b64decode(manifest_text))
    assert manifest["parent_shape"] == [m, 2 * width]
    assert manifest["output_shape"] == [m, width]
    assert manifest["scalars"] == [m * width]
    assert manifest["geometry"] == [(m * width + 255) // 256, 1, 1, 256, 1, 1]
    assert manifest["profile"] == "gfx1201_parent_view_f32_v1"
    program = json.loads(base64.b64decode(manifest["program_base64"]))
    assert program["argument_count"] == 1 and program["outputs"] == [1]
    assert [row["last_read"] for row in program["buffers"]] == [0, 1]
    assert [row["first_write"] for row in program["buffers"]] == [-1, 0]
    assert [row["ownership"] for row in program["buffers"]] == [0, 2]
    assert [row["bytes"] for row in program["buffers"]] == [m * width * 8, m * width * 4]



def test_row_count_changes_source_witness_but_reuses_physical_body():
    first = stages(source(3, 17))
    second = stages(source(16, 17))
    for label in ("source_hash", "artifact_hash"):
        a = re.search(label + r' = "([a-f0-9]+)"', first[1]).group(1)
        b = re.search(label + r' = "([a-f0-9]+)"', second[1]).group(1)
        assert a != b
    for label in ("name", "body_hash"):
        a = re.search(label + r' = "([a-zA-Z0-9_]+)"', first[1]).group(1)
        b = re.search(label + r' = "([a-zA-Z0-9_]+)"', second[1]).group(1)
        assert a == b


@pytest.mark.parametrize("change", [
    lambda text: text.replace("tessera.ir.version = \"1.0\"", "tessera.ir.version = \"2.0\""),
    lambda text: text.replace("tessera.structured_cfg.blocks = 2", "tessera.structured_cfg.blocks = 2, tessera.structured_cfg.execution = \"bounded_state_machine_v1\""),
    lambda text: text.replace('tessera.arch = "gfx1201"', 'tessera.arch = "gfx1151"'),
    lambda text: text.replace('tessera.target = "rocm"', 'tessera.target = "nvidia_sm120"'),
    lambda text: text.replace("start_indices = [0, 17]", "start_indices = [0, 0]"),
    lambda text: text.replace("tessera.effect_kind = \"pure\"", "numeric_policy = \"approximate\""),
    lambda text: text.replace("%parent: tensor<3x34xf32>", '%parent: tensor<3x34xf32> {tessera.layout = "col_major"}'),
])
def test_parent_profile_rejects_semantic_or_physical_changes(change):
    original = source()
    changed = change(original)
    assert changed != original
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool(), changed, "--tessera-graph-to-schedule")


def test_parent_profile_checks_original_schedule_witness():
    scheduled = run_tessera_opt(tool(), source(), "--tessera-graph-to-schedule")
    changed = scheduled.replace("start_indices = [0, 17]", "start_indices = [0, 0]")
    assert changed != scheduled
    with pytest.raises(RuntimeError, match="Schedule differs"):
        run_tessera_opt(tool(), changed, "--tessera-schedule-to-tile")


def test_nvidia_rejects_rocm_schema_before_target_projection():
    tile = stages(source())[1]
    with pytest.raises(RuntimeError, match="SM120 schema-1"):
        run_tessera_opt(tool(), tile, "--lower-tile-to-nvidia=sm=120")


def test_rocm_rejects_relabelled_schema_at_target_boundary():
    tile = stages(source())[1]
    changed = tile.replace("schema = 2", "schema = 1")
    assert changed != tile
    with pytest.raises(RuntimeError):
        run_tessera_opt(tool(), changed, "--lower-tile-to-rocm=arch=gfx1201")


def test_parent_view_canonical_package_preserves_native_frame():
    from tessera.compiler.rocm_parent_view_native import (
        ParentViewRecipe, PARENT_VIEW_ABI, _frame, package_parent_view_recipe,
    )
    graph = source(3, 17)
    schedule = run_tessera_opt(tool(), graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool(), schedule, "--tessera-schedule-to-tile")
    package = package_parent_view_recipe(
        ParentViewRecipe(graph, schedule, tile), pipeline_name="tessera-rocm-executable")
    assert package.image.payload.startswith(b"\x7fELF")
    assert package.image.architecture == "gfx1201"
    assert package.descriptor.abi_id == PARENT_VIEW_ABI
    program, member = _frame(package.target_ir)
    assert _frame(package.backend_ir) == (program, member)
    assert member["bindings"] == ["parent", "out"]
    assert package.descriptor.entry_symbol == member["entry"]
    assert "gpu.binary" in package.backend_ir


def test_parent_view_frontend_compiles_through_driver_without_mutating_graph():
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.rocm_parent_view_native import PARENT_VIEW_ABI

    def original(parent):
        left = ts.ops.slice(parent, [0, 0], [3, 17])
        right = ts.ops.slice(parent, [0, 17], [3, 17])
        return ts.ops.silu_mul(left, right)

    module = to_graph_ir_module(
        trace(original, np.ones((3, 34), np.float32), arg_names=["parent"]),
        name="gate_up", target="rocm_gfx1201")
    before = module.to_mlir(target="rocm_gfx1201", canonical=True)
    bundle = compile_graph_module(module, source_origin="native-parent-view-test",
        target="rocm_gfx1201", options={"package_native": True})
    assert module.to_mlir(target="rocm_gfx1201", canonical=True) == before
    assert bundle.native_image.payload.startswith(b"\x7fELF")
    assert bundle.launch_descriptor.abi_id == PARENT_VIEW_ABI
    assert bundle.request.pipeline_name == bundle.native_image.pipeline_name == "tessera-rocm-executable"
    assert tuple(item.name for item in bundle.launch_descriptor.buffers) == ("parent", "v2")
