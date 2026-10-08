"""Native symbolic physical bias dimensions carry runtime stride/reduction extents."""
import re
import pytest
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt

pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason="matching native compiler required")

def graph(backward=False,query=True,key=True):
    names=("do","q","k","v","o","bias","lse","dq","dk","dv","dbias") if backward else ("q","k","v","bias","o","lse")
    physical=(1,1,3 if query else 1,4 if key else 1)
    text=checkpoint._graph_text(names,(1,2,1,3,4,8,6),.5,True,backward,
        bias=True,bias_gradient=backward,bias_shape=physical)
    # Logical sequences are both symbolic. A physical extent of one retains
    # its broadcast policy; a nonunit physical sequence extent is symbolic.
    text=text.replace("1x2x3x","1x2x?x").replace("1x2x3xf32","1x2x?xf32")
    text=text.replace("1x1x4x8","1x1x?x8").replace("1x1x4x6","1x1x?x6")
    actual="tensor<1x1x"+str(physical[2])+"x"+str(physical[3])+"xf32>"
    symbolic="tensor<1x1x"+("?" if query else "1")+"x"+("?" if key else "1")+"xf32>"
    text=text.replace(actual,symbolic)
    return text.replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_shape_bounds = array<i64: 1, 2, 1, 9, 11, 8, 6>')

@pytest.mark.parametrize("backward",[False,True])
@pytest.mark.parametrize("query,key",[(True,False),(False,True),(True,True)])
def test_native_symbolic_physical_bias_carries_all_runtime_extents(backward,query,key):
    scheduled=run_tessera_opt(find_tessera_opt(),graph(backward,query,key),"--tessera-graph-to-schedule")
    tile=run_tessera_opt(find_tessera_opt(),scheduled,"--tessera-schedule-to-tile")
    entry=re.search(r"llvm.func @([^ (]+)",tile)[1]
    signature=tile.split("llvm.func @"+entry,1)[1].split("attributes",1)[0]
    assert signature.count(": i64")==11
    assert "physical_owner_lexicographic_bhqk_v1" in tile
    assert "bias_shape = array<i64: 1, 1, " in tile
    assert "-9223372036854775808" in tile
    # Exercise owning native Target/PTX generation, not just parser admission.
    lowered,ptx,*_=native._compile_tile_ir(tile,entry)
    assert entry in ptx and "arith.constant -9223372036854775808" not in lowered

@pytest.mark.parametrize("backward",[False,True])
def test_symbolic_bias_tile_requires_runtime_extent_operands(backward):
    scheduled=run_tessera_opt(find_tessera_opt(),graph(backward),"--tessera-graph-to-schedule")
    tile=run_tessera_opt(find_tessera_opt(),scheduled,"--tessera-schedule-to-tile")
    kind="tile.attention_backward_kernel" if backward else "tile.attention_kernel"
    match=re.search(r"("+re.escape(kind)+r" )([^{}]+)(\{[^\n]*\} : )([^\n]+)",tile)
    assert match is not None,tile
    operands=match[2].split(",")
    types=match[4].split(",")
    assert len(operands)==len(types) and len(operands)>11
    changed=(tile[:match.start(2)]+",".join(operands[:-4])+" "+match[3]+
        ",".join(types[:-4])+tile[match.end(4):])
    with pytest.raises(RuntimeError,match="expects"):
        run_tessera_opt(find_tessera_opt(),changed,"--canonicalize")


@pytest.mark.parametrize("query,key",[(True,False),(False,True),(True,True)])
def test_dynamic_bias_guards_and_actual_physical_shapes(monkeypatch,query,key):
    from tessera.compiler.attention_shape_contract import descriptor_attention_shapes
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    text=graph(False,query,key)
    scheduled=run_tessera_opt(find_tessera_opt(),text,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(find_tessera_opt(),scheduled,"--tessera-schedule-to-tile")
    decoded=checkpoint._decode_checkpoint(text,scheduled,tile,backward=False)
    package=native.package_scheduled_checkpoint(decoded,pipeline_name="tessera-nvidia-pipeline-sm120")
    dims=(1,2,1,3,4,8,6)
    assert descriptor_attention_shapes(package.descriptor,dims)[1][3]==(1,1,3 if query else 1,4 if key else 1)
    bias_binding=package.descriptor.buffers[3].name
    guards=[g for g in package.descriptor.shape_guards if g.binding==bias_binding]
    assert all(g.value>0 for g in guards)
    for axis,dynamic,cap in ((2,query,9),(3,key,11)):
        selected={ (g.predicate,g.value) for g in guards if g.dimension==axis }
        assert selected==({("min",1),("max",cap)} if dynamic else {("eq",1)})

@pytest.mark.parametrize("bad",[-1,0,1,5,True])
def test_dynamic_physical_bias_scalar_rejected_before_cuda(monkeypatch,bad):
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.attention_shape_contract import descriptor_attention_shapes
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    text=graph(False,False,True)
    scheduled=run_tessera_opt(find_tessera_opt(),text,"--tessera-graph-to-schedule")
    tile=run_tessera_opt(find_tessera_opt(),scheduled,"--tessera-schedule-to-tile")
    decoded=checkpoint._decode_checkpoint(text,scheduled,tile,backward=False)
    package=native.package_scheduled_checkpoint(decoded,pipeline_name="tessera-nvidia-pipeline-sm120")
    dims=(1,2,1,3,4,8,6)
    shapes=descriptor_attention_shapes(package.descriptor,dims)[1]
    buffers={b.name:np.zeros(shape,"f4") for b,shape in zip(package.descriptor.buffers,shapes,strict=True)}
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv","BiasB","BiasH","BiasQ","BiasK"),(*dims,1,1,1,bad),strict=True))
    monkeypatch.setattr(rt,"_load_nvidia_ptx_launch",lambda:pytest.fail("loaded CUDA for malformed physical pitch"))
    with pytest.raises(RuntimeError,match="physical bias scalars"):
        rt._submit_nvidia_sm120_native(package.image,package.descriptor,buffers,scalars,None)
