"""Native AD row-residual consumer contracts, sealing and target lowering."""
import re

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_attention_row_correction_projection import project
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.compiler_route


def lower(text, option):
    tool = find_tessera_opt()
    assert tool is not None, "production compiler required"
    return run_tessera_opt(tool, text, option)


def stages(text=None, *, storage="f16", wide=False, bias=True, seeded=True,
           prune=False, compact=False, threads=128):
    text = text or source(storage, bias=bias, saved=seeded, output="f32" if wide else storage)
    flags = "--tessera-autodiff-paired=checkpoint-product=backward_rows"
    if prune:
        flags += " prune-checkpoint-gradients=true"
    if compact:
        flags += f" compact-checkpoint-gradients=true compact-checkpoint-threads={threads}"
    graph = lower(text, flags)
    schedule = lower(graph, "--tessera-graph-to-schedule")
    tile = lower(schedule, "--tessera-schedule-to-tile")
    return graph, schedule, tile


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("bias,seeded", [(False, False), (False, True), (True, False), (True, True)])
def test_native_consumer_projects_explicit_residual_without_output_or_seed(storage, wide, bias, seeded):
    original = source(storage, bias=bias, saved=seeded, output="f32" if wide else storage)
    graph, schedule, tile = stages(original)
    body = graph[graph.index("  func.func @attention_backward_rows_checkpoint"):]
    names = ["dO", "q", "k", "v"] + (["bias"] if bias else []) + ["lse", "row_correction"]
    assert "tessera.argument_bindings = [" + ", ".join('"'+name+'"' for name in names) + "]" in body
    assert body.count("bufferization.writable = false") == len(names)
    assert "tessera_attn.checkpoint_backward_rows" in body
    assert '"row_seed"' not in body and '"output"' not in body
    assert f"lse_cotangent = {str(seeded).lower()}" in body
    assert "schedule.attention_backward_rows" in schedule
    assert 'row_correction_usage = "external_input_v1"' in tile
    assert 'block_threads = 128 : i64' in tile
    producer = lower(project(original), "--tessera-graph-to-schedule")
    digest = lambda text: re.search(r'source_ad_pair_digest = "([^"]+)"', text)[1]
    assert digest(schedule) == digest(producer)
    signature = re.search(r"llvm.func @[^ (]+\(([^)]*)\)", tile)[1]
    assert signature.count("!llvm.ptr") == 9 + 2*int(bias)
    assert signature.count("i64") == 7
    target = lower(tile, "--tessera-lower-to-nvidia-sm120")
    assert "tile.attention_backward_rows_kernel" not in target
    assert "nvvm.ex2 " in target and "llvm.load" in target and "llvm.store" in target


@pytest.mark.parametrize("indices,activity", [([0], [1,0,0,0]), ([2], [0,0,1,0]), ([3,0], [1,0,0,1])])
@pytest.mark.parametrize("threads", [64,128])
def test_native_compact_consumer_seals_requested_gradient_pointer_count(indices, activity, threads):
    text = source("bf16", bias=True, saved=True).replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [' + ", ".join(map(str, indices)) + ']')
    _, schedule, tile = stages(text, prune=True, compact=True, threads=threads)
    assert "gradient_activity = array<i64: " + ", ".join(map(str,activity)) + ">" in schedule
    assert f"block_threads = {threads} : i64" in tile
    signature = re.search(r"llvm.func @[^ (]+\(([^)]*)\)", tile)[1]
    assert signature.count("!llvm.ptr") == 7+sum(activity)
    assert 'gradient_output = "compact_v1"' in tile
    assert 'gradient_launch = "packed_v1"' in tile
    assert "tile.attention_backward_rows_kernel" not in lower(tile, "--tessera-lower-to-nvidia-sm120")


@pytest.mark.parametrize("mutation", ["residual_shape", "gradient_type", "unknown_policy", "extra_seed"])
def test_consumer_graph_rejects_incoherent_residual_or_gradient_contract(mutation):
    graph, _, _ = stages()
    lines = graph.splitlines()
    index = next(i for i,line in enumerate(lines) if "tessera_attn.checkpoint_backward_rows" in line)
    if mutation == "residual_shape":
        lines[index] = lines[index].replace("tensor<1x2x3xf32>", "tensor<1x2x4xf32>")
    elif mutation == "gradient_type":
        left, right = lines[index].split(" -> ")
        lines[index] = left + " -> " + right.replace("xf16>", "xf32>", 1)
    elif mutation == "unknown_policy":
        lines[index] = lines[index].replace("{", '{row_mode = "rounded_output", ', 1)
    else:
        lines[index] = lines[index].replace("%arg6 {", "%arg6, %arg6 {", 1)
    assert "\n".join(lines) != "\n".join(graph.splitlines())
    with pytest.raises(RuntimeError):
        lower("\n".join(lines), "--canonicalize")


@pytest.mark.parametrize("mutation", ["schedule_order", "schedule_usage", "tile_usage", "tile_activity", "tile_pointer", "tile_hash"])
def test_consumer_replay_cannot_change_residual_roles_or_policy(mutation):
    original = source("f16", bias=True, saved=True, dims=(1,2,2,3,3,4,4)) if mutation == "schedule_order" else None
    _, schedule, tile = stages(original, prune=True)
    text = schedule if mutation.startswith("schedule") else tile
    lines = text.splitlines()
    marker = "schedule.attention_backward_rows" if mutation.startswith("schedule") else "tile.attention_backward_rows_kernel"
    index = next(i for i,line in enumerate(lines) if marker in line)
    if mutation == "schedule_order":
        lines[index] = lines[index].replace("%d_query, %d_key", "%d_key, %d_query")
    elif mutation.endswith("usage"):
        lines[index] = lines[index].replace('external_input_v1', 'recompute')
    elif mutation == "tile_activity":
        lines[index] = lines[index].replace('gradient_activity = array<i64: 1, 1, 1, 1>', 'gradient_activity = array<i64: 1, 0, 1, 1>')
    elif mutation == "tile_pointer":
        lines[index] = lines[index].replace('%arg5, %arg6', '%arg6, %arg5')
    else:
        index = next(i for i,line in enumerate(lines) if 'tessera.schedule_hash' in line)
        lines[index] = re.sub(r'tessera.schedule_hash = "[a-f0-9]+"', 'tessera.schedule_hash = "' + '0'*64 + '"', lines[index])
    assert "\n".join(lines) != "\n".join(text.splitlines())
    with pytest.raises(RuntimeError):
        lower("\n".join(lines), "--tessera-schedule-to-tile" if mutation.startswith("schedule") else "--canonicalize")
