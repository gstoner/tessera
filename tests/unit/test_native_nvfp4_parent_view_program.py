"""Native physical partition keeps logical gate/up views inside one member."""
import pytest

from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
from tests.unit.test_native_nvfp4_expansion_program import source_graph

pytestmark = pytest.mark.compiler_route


def resident_source(m=3, width=17, k=48):
    parent = f"tensor<{m}x{2 * width}xbf16>"
    result = f"tensor<{m}x{width}xbf16>"
    compute = f"tensor<{m}x{width}xf32>"
    source = source_graph(m=m, n=2 * width, k=k, output="bf16")
    source = source.replace(f") -> {parent} {{", f") -> {result} {{", 1)
    old = f" return %out : {parent}"
    assert source.count(old) == 1
    operations = f"""
 %left = tessera.slice %out {{start_indices = [0, 0], slice_sizes = [{m}, {width}]}} : ({parent}) -> {result}
 %right = tessera.slice %out {{start_indices = [0, {width}], slice_sizes = [{m}, {width}]}} : ({parent}) -> {result}
 %left_f32 = tessera.cast %left {{dtype = "fp32"}} : ({result}) -> {compute}
 %right_f32 = tessera.cast %right {{dtype = "fp32"}} : ({result}) -> {compute}
 %product = tessera.silu_mul %left_f32, %right_f32 : ({compute}, {compute}) -> {compute}
 %activation = tessera.cast %product {{dtype = "bf16"}} : ({compute}) -> {result}
 return %activation : {result}"""
    return source.replace(old, operations)


@pytest.mark.usefixtures("production_compiler")
def test_native_three_member_program_retains_private_bf16_parent_and_view_ssa():
    program = export_native_nvfp4_program(resident_source())
    plan = program.manifest
    assert plan["schema"] == "tessera.native.nvfp4_program.v5"
    assert len(plan["steps"]) == 3 and len(plan["buffers"]) == 9
    assert [step["inputs"] for step in plan["steps"]] == [[0, 1, 2], [3, 5, 4, 6], [7]]
    assert [step["outputs"] for step in plan["steps"]] == [[5, 6], [7], [8]]
    assert plan["output"] == 8
    assert [(b["first_write"], b["last_read"], b["ownership"]) for b in plan["buffers"][5:]] == [
        (0, 1, "private_scratch"), (0, 1, "private_scratch"),
        (1, 2, "private_scratch"), (2, 3, "returned_output")]
    assert plan["buffers"][7]["shape"] == [3, 34] and plan["buffers"][7]["storage"] == "bf16"
    assert plan["buffers"][8]["shape"] == [3, 17] and plan["buffers"][8]["storage"] == "bf16"
    child = program.project_member(2)
    assert child.count("tessera.slice") == 2 and child.count("tessera.cast") == 3
    assert "tessera.native.gate_up_view" in child


@pytest.mark.usefixtures("production_compiler")
def test_static_activation_cannot_reuse_a_two_member_bounded_row_frame():
    source = resident_source().replace('tessera.arch = "gfx1201"',
                                      'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 32 : i64')
    with pytest.raises(RuntimeError, match="three-member runtime expression contract"):
        export_native_nvfp4_program(source)


def explicit_store_source(m=3, width=17, k=48):
    """Keep the frontend FP32 product and its explicit BF16 store in Graph."""
    source = resident_source(m, width, k)
    parent = f"tensor<{m}x{2 * width}xbf16>"
    accumulator = f"tensor<{m}x{2 * width}xf32>"
    marker = f"-> {parent}\n"
    assert source.count(marker) == 1
    source = source.replace(marker, f"-> {accumulator}\n")
    source = source.replace(" %out = tessera.scaled_matmul", " %acc = tessera.scaled_matmul")
    return source.replace(" %left = tessera.slice",
        f' %out = tessera.cast %acc {{dtype = "bf16"}} : ({accumulator}) -> {parent}\n %left = tessera.slice')


@pytest.mark.usefixtures("production_compiler")
def test_explicit_product_store_stays_in_native_member_without_f32_allocation():
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

    source = explicit_store_source()
    program = export_native_nvfp4_program(source)
    plan = program.manifest
    assert len(plan["buffers"]) == 9 and len(plan["steps"]) == 3
    assert plan["buffers"][7]["storage"] == "bf16"
    assert plan["buffers"][7]["shape"] == [3, 34]
    member = program.project_member(1)
    assert "tessera.cast" in member and "tensor<3x34xf32>" in member
    schedule = run_tessera_opt(find_tessera_opt(), member, "--tessera-graph-to-schedule")
    assert "schedule.matmul" in schedule
    assert "tessera.cast" not in schedule
    assert "tessera.native.fp8_store_cast" not in schedule
    tile = run_tessera_opt(find_tessera_opt(), schedule, "--tessera-schedule-to-tile")
    assert "tile.scaled_matmul_kernel" in tile
    assert "bf16" in tile
    assert source == explicit_store_source()


def frontend_activation(rows, width, reverse=False):
    import tessera as ts
    from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy

    columns = 2 * width
    policy = nvfp4_expansion_policy()
    def composed(codes, scales, globals_, activation, activation_scales):
        weight, weight_scales = ts.ops.nvfp4_expand(
            codes, scales, globals_, row_offsets=[0, 3, columns], numeric_policy=policy)
        accumulator = ts.ops.scaled_matmul(
            activation, weight, activation_scales, weight_scales, transposeB=True,
            numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
            scale_layout={"granularity": "block", "block": [1, 16], "format": "fp32"})
        parent = ts.ops.cast(accumulator, "bf16")
        left = ts.ops.slice(parent, [0, 0], [rows, width])
        right = ts.ops.slice(parent, [0, width], [rows, width])
        left_f32 = ts.ops.cast(left, "fp32")
        right_f32 = ts.ops.cast(right, "fp32")
        output = ts.ops.silu_mul(right_f32, left_f32) if reverse else ts.ops.silu_mul(left_f32, right_f32)
        return ts.ops.cast(output, "bf16")
    return composed


@pytest.mark.parametrize("reverse", [False, True])
def test_authored_frontend_preserves_explicit_product_store_and_operand_roles(reverse):
    from tessera.compiler.trace import trace, to_graph_ir_module
    from tessera.compiler.rocm_nvfp4_source_frontend import supports_source_trace
    from tests.device.rocm.test_prepared_nvfp4_expansion_program import inputs_and_oracle

    roots, _ = inputs_and_oracle(3, 34, 48, (0, 1, 2, 3, 4), "bf16")
    module = to_graph_ir_module(trace(frontend_activation(3, 17, reverse), *roots),
                               target="rocm_gfx1201")
    assert supports_source_trace(module)
    operations = module.functions[0].body
    assert len(operations) == 9
    assert "f32>" in str(operations[1].result_type)
    assert "bf16>" in str(operations[2].result_type)
    module.module_attrs.update({"tessera.target": '"rocm_gfx1201"',
                                "tessera.arch": '"gfx1201"'})
    original = module.to_mlir(target="rocm_gfx1201", canonical=True)
    plan = export_native_nvfp4_program(original).manifest
    assert len(plan["buffers"]) == 9 and plan["output"] == 8
    assert module.to_mlir(target="rocm_gfx1201", canonical=True) == original


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new", [
    ('tessera.arch = "gfx1201"', 'tessera.arch = "gfx1151"'),
    ('dtype = "bf16"', 'dtype = "bf16", rounding = "truncate"'),
    ('execution_mode = "exact_per_block"', 'execution_mode = "approximate"'),
])
def test_store_fusion_refuses_other_architecture_or_unproved_cast_policy(old, new):
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

    member = export_native_nvfp4_program(explicit_store_source()).project_member(1)
    assert old in member
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(), member.replace(old, new),
                        "--tessera-graph-to-schedule")
