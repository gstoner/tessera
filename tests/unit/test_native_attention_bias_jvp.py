"""Native Graph/AD and Schedule/Tile bias product contracts."""
import json
import re
from pathlib import Path
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.native_gpu_storage import _decode_image

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/"benchmarks/baselines/nvidia_public_attention_vjp_20261006/artifacts"

@pytest.fixture
def compiler():
    tool=find_tessera_opt()
    if tool is None:pytest.skip("matching native compiler required")
    return tool

def source(case):
    return json.loads((ARTIFACTS/(case+".json")).read_text())["graph_ir"].replace(
        'tessera.autodiff = "reverse"','tessera.autodiff = "forward"')

@pytest.mark.parametrize("case,activity,bias",[
    ("biasvqk_bias_5_0_1x4x1x1",[False,False,False,True],[1,4,1,1]),
    ("biasvqk_bias_v_k_q_5_1_2x4x3x5",[True,True,True,True],[2,4,3,5]),
    ("biasvqk_k_bias_5_0_2x4x3x5",[False,True,False,True],[2,4,3,5]),
    ("biasvqk_bias_5_0_2x4x3x5",[False,False,False,True],[2,4,3,5]),
])
def test_native_bias_activity_and_shape(compiler,case,activity,bias):
    product=run_tessera_opt(compiler,source(case),"--tessera-autodiff-forward=export-attention-jvp")
    fields=re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"',product)
    assert len(fields)==1
    contract=json.loads(_decode_image(fields[0]).decode())
    assert contract["schema"]==2 and contract["active"]==activity
    assert contract["bias_shape"]==bias
    assert "tessera_attn.checkpoint_jvp" in product
    # Reparse verifies the complete paired forward/bias association.
    run_tessera_opt(compiler,product,"--canonicalize")
    targeted=product
    tile=run_tessera_opt(compiler,targeted,
        "--pass-pipeline=builtin.module(tessera-graph-to-schedule,tessera-schedule-to-tile)")
    manifest_fields=re.findall(r'tessera\.native_tensor_contract = "((?:\\.|[^"\\])*)"',tile)
    manifest=json.loads(_decode_image(manifest_fields[0]).decode())
    assert [arg["name"] for arg in manifest["arguments"]]==[
        "q","k","v","primal","lse","dq","dk","dv","bias","dbias","tangent","scratch"]
    assert manifest["arguments"][8]["shape"]==bias
    assert manifest["arguments"][9]["shape"]==bias
    assert manifest["block"]==[128,1,1]
    from tessera.compiler.nvidia_native import _checkpoint_identity
    expected=_checkpoint_identity(tuple(contract["dims"]),contract["scale"],
        contract["causal"],bias=True,bias_shape=tuple(bias))
    assert f'tessera.attention_checkpoint_identity = "{expected}"' in tile

def test_bias_v_only_export_retains_bias_generation(compiler):
    text=source("biasvqk_bias_5_0_1x4x1x1")
    text=text.replace('wrt = ["bias"]','wrt = ["v"]').replace('wrt_indices = [0]','wrt_indices = [1]')
    product=run_tessera_opt(compiler,text,"--tessera-autodiff-forward=export-attention-jvp")
    fields=re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"',product)
    contract=json.loads(_decode_image(fields[0]).decode())
    assert contract["active"]==[False,False,True,False]
    assert contract["bias_shape"]==[1,4,1,1]
    run_tessera_opt(compiler,product,"--canonicalize")

def test_unbiased_export_keeps_v1_contract(compiler):
    product=run_tessera_opt(compiler,source("qkv_q_5_0"),"--tessera-autodiff-forward=export-attention-jvp")
    fields=re.findall(r'tessera\.autodiff\.attention_jvp_contract = "((?:\\.|[^"\\])*)"',product)
    contract=json.loads(_decode_image(fields[0]).decode())
    assert contract["schema"]==1 and len(contract["active"])==3
    assert "bias_shape" not in contract


@pytest.mark.parametrize("old,new",[
    ("bias_shape = array<i64: 1, 4, 1, 1>",
     "bias_shape = array<i64: 2, 4, 3, 5>"),
    ("active = [false, false, false, true]",
     "active = [false, false, false, false]"),
])
def test_bias_schedule_replay_rejects_contract_tampering(compiler,old,new):
    graph=run_tessera_opt(compiler,source("biasvqk_bias_5_0_1x4x1x1"),
        "--tessera-autodiff-forward=export-attention-jvp")
    schedule=run_tessera_opt(compiler,graph,"--tessera-graph-to-schedule")
    assert old in schedule
    with pytest.raises(RuntimeError,match="changed after hashing"):
        run_tessera_opt(compiler,schedule.replace(old,new),
            "--tessera-schedule-to-tile")


def test_bias_jvp_rejects_same_shape_different_forward_bias(compiler):
    product=run_tessera_opt(compiler,source("biasvqk_bias_5_0_1x4x1x1"),
        "--tessera-autodiff-forward=export-attention-jvp")
    lines=product.splitlines()
    for index,line in enumerate(lines):
        if "= tessera_attn.checkpoint_jvp " in line:
            assert ", %arg0, %arg4 {" in line
            lines[index]=line.replace(", %arg0, %arg4 {",", %arg4, %arg4 {")
    with pytest.raises(RuntimeError,match="generation"):
        run_tessera_opt(compiler,"\n".join(lines),"--canonicalize")
