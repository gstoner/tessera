"""Native sequence projection owns the bounded public attention AD pipeline."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint
from tests.unit.test_native_attention_ad_products import source
from tessera.compiler import nvidia_native as native
pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason="matching native compiler required")

def request(text=None,bounds="array<i64: 9, 11>",target="nvidia_sm120",arch="sm_120"):
    return (source("reverse") if text is None else text).replace("module {",
        'module attributes {tessera.target = "'+target+'", tessera.arch = "'+arch+
        '", tessera.attention_sequence_bounds = '+bounds+'} {',1)

@pytest.mark.parametrize("backward",[False,True])
def test_native_projection_preserves_symbolic_graph_ad_schedule_tile(backward):
    artifact=lower_generated_checkpoint(request(),backward=backward,prune_inactive=backward)
    assert artifact.dims==(1,2,1,-(1<<63),-(1<<63),8,8)
    assert artifact.shape_bounds==(1,2,1,9,11,8,8)
    assert "tensor<1x2x?x8xf32>" in artifact.graph_ir
    assert "tessera.attention_ad_pair" in artifact.graph_ir
    assert "tessera.attention_sequence_bounds" not in artifact.graph_ir
    assert "shape_bounds = array<i64: 1, 2, 1, 9, 11, 8, 8>" in artifact.tile_ir
    assert artifact.frontend_argument_indices==(0,1,2)

@pytest.mark.parametrize("bounds",["array<i64: 9>","array<i64: 0, 11>","array<i64: 3, 11>","array<i64: 9, 5>"])
def test_native_projection_rejects_bad_capacity(bounds):
    with pytest.raises(RuntimeError,match="projection|capacity"):
        lower_generated_checkpoint(request(bounds=bounds))

def test_native_projection_is_owning_architecture_specific():
    with pytest.raises(RuntimeError,match="SM120"):
        lower_generated_checkpoint(request(target="rocm_gfx1201",arch="gfx1201"))

def test_native_projection_retains_static_saved_graph_when_not_requested():
    plain=source("reverse").replace("module {",'module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {',1)
    assert lower_generated_checkpoint(plain).dims==(1,2,1,4,6,8,8)

def test_native_projection_canonicalizes_different_trace_extents():
    first=lower_generated_checkpoint(request())
    text=source("reverse").replace("1x2x4x8","1x2x3x8").replace("1x1x6x8","1x1x5x8")
    second=lower_generated_checkpoint(request(text))
    assert first.schedule_digest==second.schedule_digest
    assert first.entry==second.entry
    assert first.tile_ir==second.tile_ir

def test_public_jit_supplies_capacity_request_without_python_type_projection(monkeypatch):
    import numpy as np,tessera as ts
    import tessera.compiler.native_attention_program as program
    @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("q","k","v"))
    def attention(q,k,v):
        return ts.ops.flash_attn(q,k,v,causal=True)
    recorded=[]
    monkeypatch.setattr(program,"compile_attention_vjp_program",lambda text,*a,**k:recorded.append(text) or object())
    q=np.zeros((1,2,3,8),np.float32);k=np.zeros((1,1,5,8),np.float32);v=np.zeros_like(k)
    attention.compile_native_attention_vjp(q,k,v,compiler=find_tessera_opt(),sequence_bounds=(9,11))
    assert "tessera.attention_sequence_bounds = array<i64: 9, 11>" in recorded[0]
    assert "tensor<1x2x3x8xf32>" in recorded[0] and "tensor<1x2x?x8xf32>" not in recorded[0]
