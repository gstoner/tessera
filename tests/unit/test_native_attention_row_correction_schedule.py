"""Sealed native correction Schedule/Tile contracts; no device execution claim."""
import re

import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_attention_row_correction_projection import project
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.compiler_route


def lower(text, option):
    tool = find_tessera_opt()
    assert tool is not None, 'production compiler required'
    return run_tessera_opt(tool, text, option)


def stages(storage='f16', *, bias=True, saved=True, output=None):
    graph = project(source(storage, bias=bias, saved=saved, output=output))
    schedule = lower(graph, '--tessera-graph-to-schedule')
    tile = lower(schedule, '--tessera-schedule-to-tile')
    return graph, schedule, tile


@pytest.mark.parametrize('storage', ['f16', 'bf16'])
@pytest.mark.parametrize('wide_output', [False, True])
@pytest.mark.parametrize('bias,saved', [(False, False), (False, True), (True, False), (True, True)])
def test_original_graph_has_sealed_correction_schedule_and_typed_tile(storage, wide_output, bias, saved):
    graph, schedule, tile = stages(storage, bias=bias, saved=saved,
                                   output='f32' if wide_output else storage)
    assert 'tessera_attn.checkpoint_row_correction' in graph
    assert 'schedule.attention_row_correction' in schedule
    assert 'family = "attention_row_correction"' in schedule
    assert 'source_ad_pair_digest = "' in schedule
    assert 'tile.attention_row_correction_kernel' in tile
    assert f'storage = "{storage}"' in tile
    assert 'row_storage = "f32"' in tile
    assert 'row_bytes = 24 : i64' in tile
    assert 'block_threads = 32 : i64' in tile
    signature = re.search(r'\n  llvm.func @\w+\(([^)]*)\)', tile)
    assert signature is not None
    assert signature[1].count('!llvm.ptr') == 6 + int(bias) + int(saved)
    assert signature[1].count('i64') == 7
    assert 'tessera.schedule_hash' in tile
    assert 'tile.attention_row_correction_kernel' in lower(tile, '--canonicalize')


@pytest.mark.parametrize('mutation', ['scale', 'row_bytes', 'hash', 'producer', 'binding', 'pair'])
def test_schedule_cannot_disagree_with_retained_graph_or_lineage(mutation):
    _, schedule, _ = stages()
    lines = schedule.splitlines()
    index = next(i for i, line in enumerate(lines) if 'schedule.attention_row_correction' in line)
    if mutation == 'scale':
        lines[index] = re.sub(r'scale = [^,}]+ : f32', 'scale = 1.0 : f32', lines[index])
    elif mutation == 'row_bytes':
        lines[index] = lines[index].replace('row_bytes = 24 : i64', 'row_bytes = 28 : i64')
    elif mutation == 'hash':
        lines[index] = re.sub(r'artifact_hash = "[a-f0-9]+"', 'artifact_hash = "' + '0'*64 + '"', lines[index])
    elif mutation == 'producer':
        index = next(i for i, line in enumerate(lines) if 'tessera_attn.checkpoint_row_correction' in line)
        lines[index] = re.sub(r'scale = [^,}]+ : f32', 'scale = 1.0 : f32', lines[index])
    elif mutation == 'binding':
        index = next(i for i, line in enumerate(lines) if '  func.func @attention_row_correction_checkpoint' in line)
        lines[index] = lines[index].replace('["dO", "q",', '["q", "dO",')
    else:
        index = next(i for i, line in enumerate(lines) if 'tessera.attention_ad_pair =' in line)
        lines[index] = lines[index].replace('@attention', '@other_attention')
    assert '\n'.join(lines) != schedule
    with pytest.raises(RuntimeError, match='row correction.*(hash|contract)'):
        lower('\n'.join(lines), '--tessera-schedule-to-tile')


@pytest.mark.parametrize('mutation', ['scale', 'rows', 'dtype', 'bias', 'arguments', 'parent_hash'])
def test_tile_cannot_change_typed_policy_or_entry_abi(mutation):
    _, _, tile = stages()
    lines = tile.splitlines()
    index = next(i for i, line in enumerate(lines) if 'tile.attention_row_correction_kernel' in line)
    if mutation == 'scale':
        lines[index] = re.sub(r'scale = [^,}]+ : f32', 'scale = 1.0 : f32', lines[index])
    elif mutation == 'rows':
        lines[index] = lines[index].replace('row_bytes = 24 : i64', 'row_bytes = 28 : i64')
    elif mutation == 'dtype':
        lines[index] = lines[index].replace('output_storage = "f16"', 'output_storage = "bf16"')
    elif mutation == 'bias':
        lines[index] = lines[index].replace('bias_shape = array<i64: 1, 2, 1, 5>', 'bias_shape = array<i64: 1, 1, 1, 5>')
    elif mutation == 'arguments':
        lines[index] = lines[index].replace('%arg1, %arg2', '%arg2, %arg1')
    else:
        index = next(i for i, line in enumerate(lines) if 'tessera.schedule_hash =' in line)
        lines[index] = re.sub(r'tessera.schedule_hash = "[a-f0-9]+"', 'tessera.schedule_hash = "' + '0'*64 + '"', lines[index])
    assert '\n'.join(lines) != tile
    with pytest.raises(RuntimeError, match='row correction'):
        lower('\n'.join(lines), '--canonicalize')


@pytest.mark.parametrize('layout', ['row_major', 'col_major'])
def test_source_layout_is_admitted_only_when_physical_contract_supports_it(layout):
    text = source('f16', saved=True).replace('%q: tensor<1x2x3x4xf16>',
        '%q: tensor<1x2x3x4xf16> {tessera.layout = "' + layout + '"}')
    graph = project(text)
    if layout == 'row_major':
        assert 'schedule.attention_row_correction' in lower(graph, '--tessera-graph-to-schedule')
    else:
        with pytest.raises(RuntimeError, match='compact row-major'):
            lower(graph, '--tessera-graph-to-schedule')


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("threads", [32, 64, 128])
def test_direct_tile_thread_contract_matches_native_owner(storage, backward, threads):
    if backward:
        from tests.unit.test_native_attention_backward_rows import stages as consumer_stages
        _, _, tile = consumer_stages(storage=storage)
    else:
        _, _, tile = stages(storage)
    # A directly authored Tile op has no sealed parent dictionary. Exercise
    # the operation verifier itself instead of an unrelated Schedule hash guard.
    signature = next(line for line in tile.splitlines() if "llvm.func @" in line)
    signature = signature.split(" attributes ", 1)[0]
    kernel = next(line for line in tile.splitlines()
                  if "tile.attention_" in line and "_kernel" in line)
    kernel = re.sub(r"block_threads = [0-9]+ : i64",
                    f"block_threads = {threads} : i64", kernel)
    direct = "module {\n" + signature + " {\n" + kernel + "\n    llvm.return\n  }\n}\n"
    legal = threads in (64, 128) if backward else threads == 32
    if legal:
        assert "tile.attention_" in lower(direct, "--canonicalize")
    else:
        with pytest.raises(RuntimeError, match="row correction.*(threads|thread)"):
            lower(direct, "--canonicalize")


@pytest.mark.parametrize('storage',['f16','bf16'])
@pytest.mark.parametrize('role',['row_correction','backward_rows'])
def test_bounded_row_schedule_seals_capacity_without_relabeling_shapes(storage,role):
    from tests.unit.test_saved_lse_half_backward import source
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    tool=find_tessera_opt();assert tool is not None
    text=source(storage,bias=True,saved=True).replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 7, 11>')
    graph=run_tessera_opt(tool,text,'--tessera-autodiff-paired=checkpoint-product='+role)
    scheduled=run_tessera_opt(tool,graph,'--tessera-graph-to-schedule')
    assert 'schedule.attention_'+role in scheduled
    from tessera.compiler.attention_shape_contract import DYNAMIC_DIM
    assert f'shape = array<i64: 1, 2, 1, {DYNAMIC_DIM}, {DYNAMIC_DIM}, 4, 3>' in scheduled
    assert 'shape_bounds = array<i64: 1, 2, 1, 7, 11, 4, 3>' in scheduled
    assert 'shape_policy = "bounded_sequences_v1"' in scheduled
    assert 'row_bytes = 56 : i64' in scheduled
    assert 'tensor<1x2x?xf32>' in scheduled
    wrong=graph.replace('tessera.target = "nvidia_sm120"','tessera.target = "rocm_gfx1201"')
    with pytest.raises(RuntimeError,match='SM120 target contract'):
        run_tessera_opt(tool,wrong,'--canonicalize')
