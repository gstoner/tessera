"""Native bounded source-expansion frame, distinct from lossy bounded v2."""
import pytest

from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_nvfp4_expansion_program import source_graph

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def bounded_source(*, bound="32 : i64", roles=(0, 1, 2, 3, 4), output="f32"):
    return source_graph(roles=roles, output=output).replace(
        'tessera.arch = "gfx1201"',
        f'tessera.arch = "gfx1201", tessera.native.nvfp4_m_bound = {bound}')


@pytest.mark.parametrize("capacity", [32, 64])
@pytest.mark.parametrize("roles", [(0, 1, 2, 3, 4), (2, 0, 4, 1, 3)])
@pytest.mark.parametrize("output,output_bytes", [("f32", 4), ("bf16", 2)])
def test_source_bounded_native_frame_retains_exact_ssa(capacity, roles, output, output_bytes):
    source = bounded_source(bound=f"{capacity} : i64", roles=roles, output=output)
    program = export_native_nvfp4_program(source)
    plan = program.manifest
    assert plan["schema"] == "tessera.native.nvfp4_program.v4"
    assert (plan["active_m"], plan["m_bound"]) == (19, capacity)
    assert "tessera.native.nvfp4_m_bound" in plan["original_graph_ir"]
    assert "tessera.native.nvfp4_m_bound" not in plan["source_graph_ir"]
    assert [s["inputs"] for s in plan["steps"]] == [list(roles[:3]), [roles[3], 5, roles[4], 6]]
    assert [s["outputs"] for s in plan["steps"]] == [[5, 6], [7]]
    assert plan["buffers"][roles[3]]["shape"] == [capacity, 48]
    assert plan["buffers"][roles[4]]["shape"] == [capacity, 3]
    assert plan["buffers"][7]["shape"] == [capacity, 7]
    assert [b["shape"] for b in plan["buffers"][5:7]] == [[7, 48], [3, 7]]
    assert [(b["first_write"], b["last_read"]) for b in plan["buffers"][5:]] == [(0, 1), (0, 1), (1, 2)]
    assert plan["row_frame"] == {
        "input_slots": [roles[3], roles[4]], "output_slot": 7,
        "row_strides_bytes": [48, 12, 7 * output_bytes], "consumer_step": 1,
    }
    tool = find_tessera_opt()
    for index, carrier in enumerate(("tile.nvfp4_expand_kernel", "tile.scaled_matmul_kernel")):
        member = program.project_member(index)
        schedule = run_tessera_opt(tool, member, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
        assert carrier in tile
    assert source == bounded_source(bound=f"{capacity} : i64", roles=roles, output=output)


@pytest.mark.parametrize("bound,marker", [
    ("0 : i64", "exceed their checked storage capacity"),
    ("18 : i64", "exceed their checked storage capacity"),
    ("9223372036854775807 : i64", "exceed their checked storage capacity"),
    ("32 : i32", "static unencoded capacity types and an i64 bound"),
    ("true", "static unencoded capacity types and an i64 bound"),
])
def test_source_bounded_rejects_invalid_frame_before_projection(bound, marker):
    with pytest.raises(RuntimeError, match=marker):
        export_native_nvfp4_program(bounded_source(bound=bound))
