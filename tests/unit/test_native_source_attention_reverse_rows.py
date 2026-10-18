"""Native source AD row-correction producer and compact consumer contracts."""
import pytest

from tests.unit.test_native_source_attention_reverse_outline import source, outline
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def rows_source(storage="f16", variant="normal"):
    return source(storage, variant).replace(
        "module attributes {",
        "module attributes {tessera.native.source_attention_reverse_rows_request = unit, ", 1)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
def test_source_reverse_rows_are_typed_native_ad_members(compiler_toolchain, storage, variant):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    outlined, plan = outline(tool, rows_source(storage, variant))
    assert plan["schema"] == "tessera.native.source_attention_reverse.v2"
    assert plan["kind"] == "norm_attention_saved_reverse_rows"
    roles = [1, 2, 0] if variant == "permuted" else [0, 1, 2]
    seeds = [4, 3] if variant == "swapped" else [3, 4]
    assert [step["inputs"] for step in plan["steps"]] == [
        [roles[0]], [5, roles[1], roles[2]],
        [seeds[0], 5, roles[1], roles[2], 7, seeds[1]],
        [seeds[0], 5, roles[1], roles[2], 7, 8], [roles[0], 9]]
    assert [step["outputs"] for step in plan["steps"]] == [
        [5], [6, 7], [8], [9, 10, 11], [12]]
    assert len(plan["buffers"]) == 13 and len(plan["member_graphs"]) == 5
    assert plan["buffers"][8]["storage"] == "f32"
    assert plan["buffers"][8]["first_write"] == 2
    assert plan["buffers"][8]["last_read"] == 3
    assert plan["buffers"][8]["ownership"] == "private_scratch"
    assert plan["buffers"][6]["last_read"] == 1  # rounded O is not a correction input
    assert "tessera_attn.checkpoint_row_correction" in plan["member_graphs"][2]
    assert "tessera_attn.checkpoint_backward_rows" in plan["member_graphs"][3]
    assert "tessera.normalization_stats" in plan["member_graphs"][4]
    assert "tessera_attn.checkpoint_backward " in plan["paired_graph_ir"]
    assert outlined.count("call @") == 5
    for index, graph in enumerate(plan["member_graphs"]):
        schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
        assert schedule.returncode == 0, schedule.stderr
        tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
        assert tile.returncode == 0, tile.stderr
        assert ("tile.norm_kernel", "tile.attention_kernel",
                "tile.attention_row_correction_kernel", "tile.attention_backward_rows_kernel",
                "tile.row_region_kernel")[index] in tile.stdout


@pytest.mark.parametrize("value", ["false", "true", '"row_correction_v1"'])
def test_rows_request_cannot_be_silently_ignored(compiler_toolchain, value):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = rows_source().replace("source_attention_reverse_rows_request = unit",
                                 "source_attention_reverse_rows_request = " + value)
    result = invoke(tool, text, "--tessera-autodiff-paired")
    assert result.returncode != 0, "a malformed native rows request was silently ignored"

