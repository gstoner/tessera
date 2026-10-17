"""Original native Graph proof for the missing SwiGLU activation route."""

import json

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_rocm_math_widening import module


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_original_silu_mul_reaches_native_schedule_and_typed_tile(arch, storage):
    tool = find_tessera_opt()
    assert tool is not None, "matching production compiler required"
    graph = module("div", storage, shape=(3, 17), reverse=True)
    if storage == "f32":
        graph.functions[0].body = graph.functions[0].body[-1:]
        graph.functions[0].body[0].operands = ["%b", "%a"]
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    graph.module_attrs.update(
        {
            "tessera.target": json.dumps("rocm"),
            "tessera.arch": json.dumps(arch),
            "tessera.launch_bindings": json.dumps(["a", "b", "out"]),
        }
    )
    source = graph.to_mlir(target="rocm_" + arch, canonical=True)
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    assert "family=rocm_math" in schedule, "native SwiGLU Schedule contract missing"
    assert 'kind = "silu_mul"' in schedule and "roles = [1, 0]" in schedule
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tile.elementwise_kernel" in tile
    assert 'family = "binary_math"' in tile
    assert 'storage = "' + storage + '"' in tile
    assert 'output_storage = "f32"' in tile
    assert tile == run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_same_native_setup_reaches_existing_division_route(arch, storage):
    tool = find_tessera_opt()
    assert tool is not None
    graph = module("div", storage, shape=(3, 17), reverse=True)
    if storage == "f32":
        graph.functions[0].body = graph.functions[0].body[-1:]
        graph.functions[0].body[0].operands = ["%b", "%a"]
    graph.module_attrs.update(
        {
            "tessera.target": json.dumps("rocm"),
            "tessera.arch": json.dumps(arch),
            "tessera.launch_bindings": json.dumps(["a", "b", "out"]),
        }
    )
    source = graph.to_mlir(target="rocm_" + arch, canonical=True)
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    assert "family=rocm_math" in schedule
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tile.elementwise_kernel" in tile
    assert 'kind = "div"' in tile and 'storage = "' + storage + '"' in tile


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("shape", [(1,), (2, 3, 257)])
def test_silu_recipe_preserves_source_and_native_binary_abi(storage, shape):
    from copy import deepcopy
    from tessera.compiler import rocm_math_native as native

    graph = module("div", storage, shape=shape, reverse=True)
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    if storage == "f32":
        graph.functions[0].body = graph.functions[0].body[-1:]
        graph.functions[0].body[0].operands = ["%b", "%a"]
    before = deepcopy(graph)
    recipe = native.lower_math_graph(graph, "rocm_gfx1201")
    assert graph == before
    info = native._info(recipe.tile_ir)
    assert info["family"] == "binary" and info["kind"] == "silu_mul"
    assert info["roles"] == (1, 0)
    assert info["storage"] == storage and info["output_storage"] == "f32"
    assert native.math_abi(info) == native.MATH_ABIS["binary" if storage == "f32" else "binary_" + storage]


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace('kind = "silu_mul"', 'kind = "mul"'),
        lambda s: s.replace("roles = [1, 0]", "roles = [0, 1]"),
        lambda s: s.replace('numeric_policy = "f32_compute"', 'numeric_policy = "approximate"'),
        lambda s: s.replace("tessera.silu_mul ", "tessera.div "),
    ],
)
def test_silu_native_replay_rejects_semantic_mutation(edit):
    from tessera.compiler import rocm_math_native as native

    graph = module("div", "bf16", reverse=True)
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    recipe = native.lower_math_graph(graph, "rocm_gfx1201")
    changed = edit(recipe.schedule_ir)
    assert changed != recipe.schedule_ir
    with pytest.raises(RuntimeError, match="ROCm math"):
        run_tessera_opt(find_tessera_opt(), changed, "--tessera-schedule-to-tile")


@pytest.mark.parametrize("target", ["x86", "apple_gpu", "nvidia"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_silu_narrow_tile_storage_requires_owning_rocm_contract(target, storage):
    from tessera.compiler import rocm_math_native as native

    graph = module("div", storage)
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    tile = native.lower_math_graph(graph, "rocm_gfx1201").tile_ir
    changed = tile.replace('tessera.target = "rocm"', 'tessera.target = "' + target + '"')
    assert changed != tile
    with pytest.raises(RuntimeError, match="storage"):
        run_tessera_opt(find_tessera_opt(), changed, "--canonicalize")


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_silu_widening_does_not_admit_unproved_pow_storage(storage):
    from tessera.compiler import rocm_math_native as native

    graph = module("div", storage)
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    tile = native.lower_math_graph(graph, "rocm_gfx1201").tile_ir
    lines = tile.splitlines()
    for i, line in enumerate(lines):
        if "tile.elementwise_kernel" in line:
            lines[i] = line.replace('kind = "silu_mul"', 'kind = "pow"')
    changed = "\n".join(lines)
    assert changed != tile
    with pytest.raises(RuntimeError, match="storage"):
        run_tessera_opt(find_tessera_opt(), changed, "--canonicalize")


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_silu_target_and_gpu_lower_the_actual_gate_expression(arch, storage):
    from tessera.compiler import rocm_math_native as native

    graph = module("div", storage, reverse=True)
    graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    if storage == "f32":
        graph.functions[0].body = graph.functions[0].body[-1:]
        graph.functions[0].body[0].operands = ["%b", "%a"]
    recipe = native.lower_math_graph(graph, "rocm_" + arch)
    tool = find_tessera_opt()
    target = run_tessera_opt(tool, recipe.tile_ir, "--lower-tile-to-rocm=arch=" + arch)
    assert "tessera_rocm.binary" in target
    assert 'kind = "silu_mul"' in target
    assert "native_math_contract" in target
    assert "tile.elementwise_kernel" not in target
    gpu = run_tessera_opt(tool, target, "--generate-rocm-binary-kernel")
    assert "gpu.func" in gpu and "memref<?xf32>" in gpu
    for operation in ("arith.negf", "math.exp", "arith.addf", "arith.divf", "arith.mulf"):
        assert operation in gpu
    assert "arith.truncf" not in gpu
    if storage != "f32":
        assert "arith.extf" in gpu and "memref<?x" + storage + ">" in gpu


def matching_half_module(storage, shape=(3, 17)):
    graph = module("div", storage, shape=shape, reverse=True)
    fn = graph.functions[0]
    op = fn.body[-1]
    result = fn.args[0].ir_type
    op.op_name = "tessera.silu_mul"
    op.operands = ["%b", "%a"]
    op.operand_types = [str(result), str(result)]
    op.result_type = str(result)
    fn.body = [op]
    fn.result_types = [result]
    return graph


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("target", ["rocm_gfx1151", "rocm_gfx1201"])
def test_matching_half_silu_keeps_f32_math_until_final_store(storage, target):
    from copy import deepcopy
    from tessera.compiler import rocm_math_native as native

    graph = matching_half_module(storage)
    before = deepcopy(graph)
    assert native.supports_math(graph)
    recipe = native.lower_math_graph(graph, target)
    assert graph == before
    info = native._info(recipe.tile_ir)
    assert info["storage"] == info["output_storage"] == storage
    assert info["numeric_policy"] == "f32_compute"
    assert native.math_abi(info) == native.MATH_ABIS["binary_" + storage + "_store"]
    assert native.math_abi(info) != native.MATH_ABIS["binary_" + storage]
    tool = find_tessera_opt()
    physical = run_tessera_opt(tool, recipe.tile_ir, "--lower-tile-to-rocm=arch=" + target.removeprefix("rocm_"))
    assert 'output_dtype = "' + storage + '"' in physical
    gpu = run_tessera_opt(tool, physical, "--generate-rocm-binary-kernel")
    assert gpu.count("arith.truncf") == 1
    assert gpu.index("arith.mulf") < gpu.index("arith.truncf")
    assert "math.exp" in gpu and "arith.divf" in gpu


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_matching_half_silu_does_not_reinterpret_existing_widening_abi(storage):
    from tessera.compiler import rocm_math_native as native

    same = native.lower_math_graph(matching_half_module(storage), "rocm_gfx1201")
    wide_graph = module("div", storage, reverse=True)
    wide_graph.functions[0].body[-1].op_name = "tessera.silu_mul"
    wide = native.lower_math_graph(wide_graph, "rocm_gfx1201")
    assert native.math_abi(native._info(same.tile_ir)) == (
        "tessera.rocm.math.a_b_o_n." + storage + "_" + storage + ".v1"
    )
    assert native.math_abi(native._info(wide.tile_ir)) == ("tessera.rocm.math.a_b_o_n." + storage + "_f32.v1")


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("evaluate", [False, True])
def test_silu_catalog_trace_does_not_require_eager_arithmetic(storage, evaluate, monkeypatch):
    import numpy as np
    import tessera as ts
    from tessera.compiler.trace import trace

    dtype = {"f32": np.float32, "f16": np.float16, "bf16": pytest.importorskip("ml_dtypes").bfloat16}[storage]

    def call(a, b):
        return ts.ops.silu_mul(a, b)

    a, b = np.ones((3, 17), dtype), np.full((3, 17), 0.75, dtype)
    if not evaluate:

        def forbidden(*args, **kwargs):
            pytest.fail("shape-only SiLU evaluated input arithmetic")

        monkeypatch.setattr(np, "exp", forbidden)
    graph = trace(call, a, b, evaluate_catalog_outputs=evaluate)
    assert [op.op_name for op in graph.body] == ["tessera.silu_mul"]
    assert graph.output_specs == (((3, 17), storage),)
    if evaluate:
        assert len(graph.output_values) == 1 and graph.output_values[0] is not None
        assert graph.output_values[0].shape == (3, 17)
    else:
        assert all(value is None for value in graph.output_values)


@pytest.mark.parametrize(
    "storage,matching",
    [
        ("f32", False),
        ("f16", False),
        ("bf16", False),
        ("f16", True),
        ("bf16", True),
    ],
)
def test_silu_native_image_projection_preserves_store_and_reuses_shapes(storage, matching):
    from tessera.compiler import rocm_math_native as native
    from tessera.compiler.rocm_native import _shape_free_target_ir

    projected = []
    for shape in ((3, 17), (7, 259)):
        if matching:
            graph = matching_half_module(storage, shape)
        else:
            graph = module("div", storage, shape=shape, reverse=True)
            graph.functions[0].body[-1].op_name = "tessera.silu_mul"
            if storage == "f32":
                graph.functions[0].body = graph.functions[0].body[-1:]
                graph.functions[0].body[0].operands = ["%b", "%a"]
        recipe = native.lower_math_graph(graph, "rocm_gfx1201")
        target = run_tessera_opt(find_tessera_opt(), recipe.tile_ir, "--lower-tile-to-rocm=arch=gfx1201")
        physical = _shape_free_target_ir(target, family="scalar_binary", directive="tessera_rocm.binary")
        assert 'kind = "silu_mul"' in physical
        assert 'output_dtype = "' + (storage if matching else "f32") + '"' in physical
        projected.append(physical)
    assert projected[0] == projected[1]
