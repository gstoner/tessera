"""Native sequence capacities survive Graph/Schedule/Tile and reject drift."""
import pytest
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")

def graph(backward=False, query=True, key=True, bias=False):
    names = (("do","q","k","v","o","bias","lse","dq","dk","dv") if backward
             else ("q","k","v","bias","o","lse")) if bias else (
        ("do","q","k","v","o","lse","dq","dk","dv") if backward
        else ("q","k","v","o","lse"))
    text = checkpoint._graph_text(names,(1,2,1,3,4,8,6),.5,True,backward,bias)
    if query:
        text=text.replace("1x2x3x","1x2x?x").replace("1x2x3xf32","1x2x?xf32")
    if key:
        text=text.replace("1x1x4x","1x1x?x")
    if bias:
        text=text.replace("1x2x3x4xf32",
                          "1x2x"+("?" if query else "3")+"x"+("?" if key else "4")+"xf32")
        # Query replacement has already changed the full logical bias prefix.
        if key:
            text=text.replace("1x2x?x4xf32","1x2x?x?xf32")
    return text.replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_shape_bounds = array<i64: 1, 2, 1, '+
        ("9" if query else "3")+", "+("11" if key else "4")+", 8, 6>")

def lower(text, pipeline):
    return run_tessera_opt(find_tessera_opt(),text,pipeline)

@pytest.mark.parametrize("backward",[False,True])
@pytest.mark.parametrize("query,key",[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("bias",[False,True])
def test_native_sequence_capacity_survives_lowering(backward,query,key,bias):
    source=graph(backward,query,key,bias)
    scheduled=lower(source,"--tessera-graph-to-schedule")
    assert 'shape_policy = "bounded_sequences_v1"' in scheduled
    assert "shape_bounds = array<i64:" in scheduled
    tile=lower(scheduled,"--tessera-schedule-to-tile")
    assert 'shape_policy = "bounded_sequences_v1"' in tile
    assert ("tile.attention_backward_kernel" if backward else "tile.attention_kernel") in tile
    assert "tessera.native_contract" in tile

@pytest.mark.parametrize("backward",[False,True])
@pytest.mark.parametrize("edit,diagnostic",[
    ("missing","explicit native shape bounds"),
    ("short","seven i64 capacities"),
    ("fixed","preserve fixed dimensions"),
    ("zero","preserve fixed dimensions"),
    ("batch","only sequence axes"),
    ("overflow","byte-address ABI"),
])
def test_native_sequence_capacity_rejects_invalid_graph(backward,edit,diagnostic):
    text=graph(backward)
    if edit=="missing":
        text=text.replace(", tessera.attention_shape_bounds = array<i64: 1, 2, 1, 9, 11, 8, 6>","")
    elif edit=="short":
        text=text.replace("1, 2, 1, 9, 11, 8, 6>","1, 2>")
    elif edit=="fixed":
        text=text.replace("1, 2, 1, 9, 11, 8, 6>","1, 3, 1, 9, 11, 8, 6>")
    elif edit=="zero":
        text=text.replace("1, 2, 1, 9, 11, 8, 6>","1, 2, 1, 0, 11, 8, 6>")
    elif edit=="batch":
        text=text.replace("tensor<1x","tensor<?x")
    else:
        text=text.replace("1, 2, 1, 9, 11, 8, 6>","1, 2, 1, 9223372036854775807, 11, 8, 6>")
    with pytest.raises(RuntimeError,match=diagnostic):
        lower(text,"--tessera-graph-to-schedule")

@pytest.mark.parametrize("backward",[False,True])
def test_native_capacity_is_bound_into_schedule_hash(backward):
    scheduled=lower(graph(backward),"--tessera-graph-to-schedule")
    # Alter module capacity only, preserving the existing Schedule hash/contract.
    prefix,rest=scheduled.split("func.func",1)
    prefix=prefix.replace("1, 2, 1, 9, 11, 8, 6","1, 2, 1, 10, 11, 8, 6")
    with pytest.raises(RuntimeError,match="contract changed"):
        lower(prefix+"func.func"+rest,"--tessera-schedule-to-tile")

@pytest.mark.parametrize("backward",[False,True])
def test_dynamic_saved_graph_import_retains_native_bounds(backward):
    source=graph(backward)
    source=source.replace("tessera_attn.checkpoint_backward","tessera.flash_attn_bwd") if backward else source.replace("tessera_attn.checkpoint_forward","tessera.flash_attn")
    source=source.replace("causal = true",'causal = true, lse_checkpoint = "saved"')
    if not backward:
        source=source.replace('lse_checkpoint = "saved"','lse_checkpoint = "saved", head_dim = 8 : i64, operandSegmentSizes = array<i32: 1, 1, 1, 0>')
    scheduled=lower(source,"--tessera-graph-to-schedule")
    assert 'shape_policy = "bounded_sequences_v1"' in scheduled
    assert "tessera_attn.checkpoint_" in scheduled
    assert "schedule.attention_checkpoint" in scheduled
