"""Native source-preserving NVFP4 producer/product SSA partition."""
import pytest

from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_rocm_graph_nvfp4_expand import POLICY


def source_graph(m=19, n=7, k=48, roles=(0, 1, 2, 3, 4), output="f32"):
    types = [f"tensor<{n}x{k//2}xi8>", f"tensor<{n}x{k//16}xf8E4M3FN>",
             "tensor<2xf64>", f"tensor<{m}x{k}xf8E4M3FN>", f"tensor<{m}x{k//16}xf32>"]
    physical = [""] * 5
    for role, slot in enumerate(roles):
        physical[slot] = types[role]
    args = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(physical))
    codes, scales, globals_, a, sa = (f"%arg{i}" for i in roles)
    b, sb, out = f"tensor<{n}x{k}xf8E4M3FN>", f"tensor<{k//16}x{n}xf32>", f"tensor<{m}x{n}x{output}>"
    return f'''module attributes {{tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"}} {{
 func.func @source_product({args}) -> {out} {{
 %b, %sb = "tessera.nvfp4_expand"({codes}, {scales}, {globals_}) {{
 row_offsets = [0 : i64, 3 : i64, {n} : i64], numeric_policy = {{{POLICY}}}
 }} : ({", ".join(types[:3])}) -> ({b}, {sb})
 %out = tessera.scaled_matmul {a}, %b scales({sa}, %sb) {{
 transposeB = true,
 numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
 scale_layout = {{granularity = "block", block = [1, 16], format = "fp32"}}
 }} : ({types[3]}, {b}, {types[4]}, {sb}) -> {out}
 return %out : {out}
 }}
}}'''


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("roles", [(0, 1, 2, 3, 4), (2, 0, 4, 1, 3)])
def test_native_source_preserving_edges(roles):
    source = source_graph(roles=roles)
    program = export_native_nvfp4_program(source)
    plan = program.manifest
    assert plan["schema"] == "tessera.native.nvfp4_program.v3"
    assert plan["role_indices"] == list(roles)
    assert [s["inputs"] for s in plan["steps"]] == [list(roles[:3]), [roles[3], 5, roles[4], 6]]
    assert [s["outputs"] for s in plan["steps"]] == [[5, 6], [7]]
    assert len(plan["buffers"]) == 8 and plan["output"] == 7
    assert [(b["first_write"], b["last_read"], b["ownership"]) for b in plan["buffers"][5:]] == [
        (0, 1, "private_scratch"), (0, 1, "private_scratch"), (1, 2, "returned_output")]
    tool = find_tessera_opt()
    for index, carrier in enumerate(("tile.nvfp4_expand_kernel", "tile.scaled_matmul_kernel")):
        member = program.project_member(index)
        schedule = run_tessera_opt(tool, member, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
        assert carrier in tile
    with pytest.raises(ValueError, match="outside"):
        program.project_member(2)
    assert source == source_graph(roles=roles)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_source_preserving_bound_needs_its_own_contract():
    source = source_graph().replace('tessera.arch = "gfx1201"',
                                    'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = 32 : i64')
    with pytest.raises(RuntimeError, match="static activation frame"):
        export_native_nvfp4_program(source)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
def test_source_preserving_rejects_nonproduct_consumer():
    source = source_graph()
    # A well-typed but different consumer must not be admitted as a product.
    begin = source.index(" %out =")
    end = source.index(" return %out", begin)
    source = source[:begin] + ' %out = arith.constant dense<0.0> : tensor<19x7xf32>\n' + source[end:]
    run_tessera_opt(find_tessera_opt(), source, "--verify-each")
    with pytest.raises(RuntimeError):
        export_native_nvfp4_program(source)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new", [
    ('format = "fp32"', 'format = "other"'),
    ('execution_mode = "exact_per_block"', 'execution_mode = "approximate"'),
])
def test_source_preserving_consumer_policy_is_not_normalized(old, new):
    with pytest.raises(RuntimeError, match="exact E4M3 K16"):
        export_native_nvfp4_program(source_graph().replace(old, new))
