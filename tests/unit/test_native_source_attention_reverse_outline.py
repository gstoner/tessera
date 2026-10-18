"""Original RMSNorm/attention SSA -> native paired AD -> typed call program.

Compiler/contract proof only. Numerical source-VJP proof belongs to the
separately gated owning SM120 suite.
"""
import base64
import json
import re

import pytest

from tests.unit.test_native_attention_prefix_program import source as primal_source
from tests.unit.test_sm120_reverse_row_schedule import invoke

pytestmark = pytest.mark.compiler_route


def source(storage="f16", variant="normal", requested=(0, 1, 2)):
    text = primal_source(storage, permuted=variant == "permuted",
                         swapped_outputs=variant == "swapped")
    text = text.replace("module attributes {",
                        "module attributes {tessera.native.source_attention_reverse_request = unit, ", 1)
    index = text.index("attributes {", text.index("func.func @")) + len("attributes {")
    return text[:index] + ('tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = '
                          + str(list(requested)) + ", ") + text[index:]


def outline(tool, text):
    result = invoke(tool, text, "--tessera-autodiff-paired")
    assert result.returncode == 0, result.stderr
    records = re.findall(r'tessera.native.source_attention_reverse_program_json = "([^"]+)"', result.stdout)
    assert len(records) == 1, "native paired source-attention plan missing"
    return result.stdout, json.loads(base64.b64decode(records[0], validate=True))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("variant", ["normal", "permuted", "swapped"])
def test_source_attention_reverse_outlines_actual_ad_members(compiler_toolchain, storage, variant):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = source(storage, variant)
    outlined, plan = outline(tool, text)
    assert plan["schema"] == "tessera.native.source_attention_reverse.v1"
    assert plan["execution_state"] == "artifact_only"
    assert plan["target"] == "nvidia_sm120" and plan["architecture"] == "sm_120"
    assert plan["kind"] == "norm_attention_saved_reverse"
    canonical = invoke(tool, text, "--verify-each")
    assert canonical.returncode == 0, canonical.stderr
    assert plan["source_graph_ir"] == canonical.stdout
    roles = [1, 2, 0] if variant == "permuted" else [0, 1, 2]
    seeds = [4, 3] if variant == "swapped" else [3, 4]
    assert plan["role_indices"] == roles and plan["cotangent_indices"] == seeds
    expected_inputs = [[roles[0]], [5, roles[1], roles[2]],
                       [seeds[0], 5, roles[1], roles[2], 6, 7, seeds[1]], [roles[0], 8]]
    assert [step["inputs"] for step in plan["steps"]] == expected_inputs
    assert [step["outputs"] for step in plan["steps"]] == [[5], [6, 7], [8, 9, 10], [11]]
    grad = {roles[0]: 11, roles[1]: 9, roles[2]: 10}
    assert plan["outputs"] == [grad[i] for i in range(3)]
    assert plan["requested_gradients"] == [0, 1, 2]
    assert [plan["buffers"][i]["last_read"] for i in (5, 6, 7, 8)] == [2, 2, 2, 3]
    assert plan["buffers"][11]["shape"] == plan["buffers"][roles[0]]["shape"]
    assert len(plan["member_graphs"]) == 4 and outlined.count("call @") == 4
    program = invoke(tool, plan["program_graph_ir"], "--verify-each")
    assert program.returncode == 0, program.stderr
    assert program.stdout.count("call @") == 4
    assert "lse_cotangent = true" in plan["member_graphs"][2]
    assert "tessera.normalization_stats" in plan["member_graphs"][3]
    assert "tessera.flash_attn" not in plan["member_graphs"][3]
    assert "tessera.autodiff.activity" not in "".join(plan["member_graphs"])
    for i, graph in enumerate(plan["member_graphs"]):
        schedule = invoke(tool, graph, "--tessera-graph-to-schedule")
        assert schedule.returncode == 0, schedule.stderr
        tile = invoke(tool, schedule.stdout, "--tessera-schedule-to-tile")
        assert tile.returncode == 0, tile.stderr
        expected = ("tile.norm_kernel", "tile.attention_kernel",
                    "tile.attention_backward_kernel", "tile.row_region_kernel")[i]
        assert expected in tile.stdout


@pytest.mark.parametrize("requested", [(0,), (1,), (2,), (2, 0), (1, 2)])
def test_source_attention_reverse_retains_requested_gradient_order(compiler_toolchain, requested):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    _, plan = outline(tool, source(requested=requested))
    grad = {0: 11, 1: 9, 2: 10}
    assert plan["requested_gradients"] == list(requested)
    assert plan["outputs"] == [grad[i] for i in requested]
    for i in (9, 10, 11):
        assert plan["buffers"][i]["ownership"] == ("returned_output" if i in plan["outputs"] else "private_scratch")


@pytest.mark.parametrize("fault", ["marker_type", "other_target", "forward_intent", "conflicting_export"])
def test_source_attention_reverse_refuses_conflicting_intent(compiler_toolchain, fault):
    tool = compiler_toolchain.require_tessera_opt("--tessera-autodiff-paired")
    text = source()
    if fault == "marker_type":
        text = text.replace("source_attention_reverse_request = unit", "source_attention_reverse_request = true")
    elif fault == "other_target":
        text = text.replace('tessera.target = "nvidia_sm120"', 'tessera.target = "apple_gpu"')
    elif fault == "forward_intent":
        text = text.replace('tessera.autodiff = "reverse"', 'tessera.autodiff = "forward"')
    else:
        text = text.replace("module attributes {", "module attributes {tessera.native.sm120_reverse_request = unit, ", 1)
    result = invoke(tool, text, "--tessera-autodiff-paired")
    assert result.returncode != 0
