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
