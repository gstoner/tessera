"""Native ownership and policy replay for the three-result ingest Graph."""
import subprocess
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt

def graph():
    return """
module attributes {tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201"} {
 func.func @ingest(%codes: tensor<7x32xui8>, %scales: tensor<7x4xf8E4M3FN>, %globals: tensor<2xf64>) -> (tensor<7x32xui8>, tensor<2x7xui8>, tensor<7x2x2xf64>) attributes {tessera.bindings = ["codes", "scales", "globals", "packed", "exponents", "stats"]} {
 %p, %e, %s = "tessera.nvfp4_requantize"(%codes, %scales, %globals) {
 row_offsets = [0 : i64, 3 : i64, 7 : i64],
 numeric_policy = {
 execution_mode = "explicit_scale_requantization",
 source_format = "nvfp4_e2m1_e4m3_k16",
 destination_format = "mxfp4_e2m1_e8m0_k32",
 source_scale_application = "projection_global_times_e4m3",
 destination_code_selection = "nearest_signed_e2m1_by_weight_sse",
 destination_scale_order = "k_group_n",
 lossy_steps = ["nvfp4_e4m3_k16_to_mxfp4_e8m0_k32_scale_and_code_requantization"]
 }} : (tensor<7x32xui8>, tensor<7x4xf8E4M3FN>, tensor<2xf64>) -> (tensor<7x32xui8>, tensor<2x7xui8>, tensor<7x2x2xf64>)
 return %p, %e, %s : tensor<7x32xui8>, tensor<2x7xui8>, tensor<7x2x2xf64>
 }
}
"""

@pytest.fixture(scope="module")
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching compiler required")
    return str(tool)

def run(tool, source, *passes):
    return subprocess.run([tool,*passes], input=source,text=True,capture_output=True)

def test_graph_schedule_tile_target_ingest_lineage(compiler):
    schedule = run(compiler,graph(),"--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert "schedule.artifact" in schedule.stdout
    assert "tessera.nvfp4_requantize" in schedule.stdout
    tile = run(compiler,schedule.stdout,"--tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "tile.nvfp4_requantize_kernel" in tile.stdout
    assert "private_outputs_distinct_readonly_inputs" in tile.stdout
    target = run(compiler,tile.stdout,"--lower-tile-to-rocm=arch=gfx1201")
    assert target.returncode == 0, target.stderr
    assert "tessera_rocm.nvfp4_requantize" in target.stdout
    native = run(compiler,target.stdout,"--generate-rocm-fpquant-kernel")
    assert native.returncode == 0, native.stderr
    assert "gpu.func" in native.stdout
    assert "tessera_rocm.nvfp4_requantize" not in native.stdout

@pytest.mark.parametrize("old,new",[
    ('execution_mode = "explicit_scale_requantization"','execution_mode = "approximate"'),
    ('[0 : i64, 3 : i64, 7 : i64]','[0 : i64, 3 : i64, 8 : i64]'),
    ('tensor<2x7xui8>','tensor<7x2xui8>'),
    ('tensor<7x2x2xf64>','tensor<7x2x1xf64>'),
])
def test_graph_rejects_contract_mutation(compiler,old,new):
    result=run(compiler,graph().replace(old,new),"--tessera-graph-to-schedule")
    assert result.returncode != 0

def test_schedule_rejects_changed_numeric_policy(compiler):
    schedule=run(compiler,graph(),"--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    changed=schedule.stdout.replace("nearest_signed_e2m1_by_weight_sse","other_selection")
    tile=run(compiler,changed,"--tessera-schedule-to-tile")
    assert tile.returncode != 0

@pytest.mark.parametrize("old,new",[
    ('tessera.target = "rocm_gfx1201"','tessera.target = "rocm_gfx1151"'),
    ('"packed", "exponents", "stats"','"codes", "exponents", "stats"'),
    ('return %p, %e, %s','return %p, %p, %s'),
])
def test_graph_rejects_target_binding_or_return_mutation(compiler,old,new):
    result=run(compiler,graph().replace(old,new),"--tessera-graph-to-schedule")
    assert result.returncode != 0

def test_schedule_rejects_changed_ownership(compiler):
    schedule=run(compiler,graph(),"--tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert "private_outputs_distinct_readonly_inputs" in schedule.stdout
    changed=schedule.stdout.replace("private_outputs_distinct_readonly_inputs","aliased_outputs")
    tile=run(compiler,changed,"--tessera-schedule-to-tile")
    assert tile.returncode != 0
