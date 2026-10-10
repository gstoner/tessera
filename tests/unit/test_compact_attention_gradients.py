"""Native compact result projection and checked ABI refusals."""
from dataclasses import replace
import pytest
from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_attention_ad_products import source
from tessera.compiler import nvidia_native as native
from tessera.compiler.compact_attention_contract import compact_attention_contract

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires native compiler")

def artifact(indices="2",launch="packed_v1",threads=128):
    text=source("reverse", ', tessera.autodiff.wrt_indices = ['+indices+']')
    text=text.replace("module {", 'module attributes {tessera.target = "nvidia_sm120", tessera.arch = "sm_120"} {',1)
    return lower_generated_checkpoint(text,backward=True,prune_inactive=True,compact_gradients=True,compact_launch=launch,compact_threads=threads)

@pytest.mark.parametrize("launch", ["packed_v1","logical_v1"])
@pytest.mark.parametrize("indices,activity", [("2",(0,0,1)),("1, 0",(1,1,0)),("2, 0",(1,0,1)),("0, 1, 2",(1,1,1))])
@pytest.mark.parametrize("threads", [64,128])
def test_native_compact_projection_keeps_logical_but_narrows_physical_results(monkeypatch,indices,activity,launch,threads):
    a=artifact(indices,launch,threads)
    assert a.gradient_activity == activity
    assert a.compact_gradients
    assert a.compact_launch == launch
    assert '"dq", "dk", "dv"' in a.schedule_ir
    assert 'inactive_gradient = "absent_v1"' in a.tile_ir
    line=next(x for x in a.tile_ir.splitlines() if "llvm.func @" in x)
    assert line.count("!llvm.ptr") == 6 + sum(activity)
    assert 'gradient_output = "compact_v1"' in a.tile_ir
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    package=native.package_scheduled_checkpoint(a,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert package.descriptor.abi_id == native.SM120_ATTN_BWD_LSE_COMPACT_F32_ABI
    dims,shapes,input_count=compact_attention_contract(package.descriptor)
    assert dims == a.dims and input_count == 6
    assert len(shapes) == 6 + sum(activity)
    assert [x.name for x in package.descriptor.buffers[6:]] == [x for x,on in zip(("dq","dk","dv"),activity,strict=True) if on]
    with pytest.raises(ValueError):
        replace(a,compact_gradients=False).validate()
    changed=a.schedule_ir.replace('gradient_output = "compact_v1"', 'gradient_output = "complete_v1"')
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(),changed,"--tessera-schedule-to-tile")

@pytest.mark.parametrize("change",["activity","physical_roles","symbol","output","shape","ordinal","scalar","guard","direction"])
def test_compact_descriptor_cannot_relabel_the_physical_image(monkeypatch,change):
    a=artifact()
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    d=native.package_scheduled_checkpoint(a,pipeline_name="tessera-nvidia-pipeline-sm120").descriptor
    if change in ("activity","physical_roles","output","shape"):
        key,value={"activity":("gradient_activity",[1,0,0]),"physical_roles":("physical_gradient_roles",[0]),
                   "output":("gradient_output","complete_v1"),"shape":("shape",[1,2,1,4,6,9,8])}[change]
        d=replace(d,provenance={**d.provenance,key:value})
    elif change == "symbol":
        d=replace(d,entry_symbol=d.entry_symbol.replace("_m4_","_m1_"))
    elif change == "ordinal":
        bindings=list(d.buffers)
        bindings[0]=replace(bindings[0],ordinal=len(bindings)-1)
        bindings[-1]=replace(bindings[-1],ordinal=0)
        d=replace(d,buffers=tuple(bindings))
    elif change == "direction":
        d=replace(d,buffers=(*d.buffers[:-1],replace(d.buffers[-1],direction="input")))
    elif change=="scalar":
        d=replace(d,scalars=(replace(d.scalars[0],name="WrongB"),*d.scalars[1:]))
    else:
        d=replace(d,shape_guards=d.shape_guards[:-1])
    with pytest.raises(ValueError):
        compact_attention_contract(d)

@pytest.mark.parametrize("invalid", ['"complete_v1"', '1 : i64'])
def test_native_compact_policy_is_checked_before_scheduling(invalid):
    a=artifact()
    graph=a.graph_ir.replace('tessera.checkpoint_gradient_output = "compact_v1"',
                            "tessera.checkpoint_gradient_output = "+invalid)
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(),graph,"--tessera-graph-to-schedule")
    tile=a.tile_ir.replace('gradient_output = "compact_v1"', "gradient_output = "+invalid)
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(),tile,"--verify-each")

def test_native_compact_option_requires_pruned_backward_export():
    text=source("reverse")
    with pytest.raises(RuntimeError,match="pruned backward"):
        run_tessera_opt(find_tessera_opt(),text,"--tessera-autodiff-paired=checkpoint-product=backward compact-checkpoint-gradients=true")

@pytest.mark.parametrize("launch", ["invalid", "", 3])
def test_compact_launch_policy_must_be_native_and_explicit(launch):
    text=source("reverse", ', tessera.autodiff.wrt_indices = [2]')
    with pytest.raises(ValueError,match="compact launch"):
        lower_generated_checkpoint(text,backward=True,prune_inactive=True,compact_gradients=True,compact_launch=launch)

@pytest.mark.parametrize("launch", ["packed_v1","logical_v1"])
@pytest.mark.parametrize("threads", [64,128])
def test_compact_physical_bias_entry_has_complete_scalar_abi(monkeypatch,launch,threads):
    import numpy as np
    import tessera.compiler.native_attention_program as program
    from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import function
    monkeypatch.setattr(program,"compile_attention_vjp_program",lambda source,active,**kwargs:source)
    values=[np.ones(shape,np.float32) for shape in
        ((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,1,5))]
    text=function(("bias","q"),True).compile_native_attention_vjp(*values,compiler="unused")
    a=lower_generated_checkpoint(text,backward=True,prune_inactive=True,compact_gradients=True,compact_launch=launch,compact_threads=threads)
    entry=next(x for x in a.tile_ir.splitlines() if "llvm.func @" in x).split(")")[0]
    assert entry.count("!llvm.ptr")==9  # seven inputs plus dQ and physical dBias.
    assert entry.count(": i64")==11
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    pkg=native.package_scheduled_checkpoint(a,pipeline_name="tessera-nvidia-pipeline-sm120")
    assert len(pkg.descriptor.buffers)==9 and len(pkg.descriptor.scalars)==11
    compact_attention_contract(pkg.descriptor)

@pytest.mark.parametrize("threads", [False,32,256,"64"])
def test_compact_threads_reject_unsupported_geometry(threads):
    text=source("reverse", ", tessera.autodiff.wrt_indices = [2]")
    with pytest.raises(ValueError,match="threads"):
        lower_generated_checkpoint(text,backward=True,prune_inactive=True,compact_gradients=True,compact_threads=threads)
