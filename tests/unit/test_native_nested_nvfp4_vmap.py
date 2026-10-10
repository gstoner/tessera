"""Nested NVFP4 leading maps retain logical tuples through native Schedule."""
import copy,math
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tessera.compiler import scheduled_matmul as schedule,nvidia_native as native
from tests.device.nvidia.test_nvfp4_transpose_jit import oriented_product,oriented_inputs

AXES={"shared_rhs_rows":(0,None,0,None),"independent_rhs":(0,0,0,0),"shared_lhs":(None,0,None,0)}

def case(mode,ta,tb,prefix=(2,3),shape=(7,5,31),*,seed=None):
    values,wanted=oriented_inputs(mode,ta,tb,*shape,batch=math.prod(prefix),seed=seed)
    nested=[]
    for value,axis in zip(values,AXES[mode],strict=True):
        if axis is None:nested.append(value);continue
        if isinstance(value,NVFP4Tensor):
            storage=value.storage.reshape(*prefix,*value.storage.shape[1:])
            nested.append(NVFP4Tensor(storage,(*prefix,*value.shape[1:]),value.packed_axis+len(prefix)-1))
        else:nested.append(value.reshape(*prefix,*value.shape[1:]))
    scalar=ts.jit(oriented_product(ta,tb),target="nvidia_sm120")
    owner=scalar
    for _ in prefix:owner=vmap(owner,in_axes=AXES[mode])
    return scalar,owner,tuple(nested),wanted.reshape(*prefix,*wanted.shape[-2:])

@pytest.mark.parametrize("mode",tuple(AXES))
@pytest.mark.parametrize("ta,tb",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("prefix",[(2,3),(1,2,3)])
def test_frontend_and_native_schedule_preserve_nested_nvfp4_axes(mode,ta,tb,prefix):
    if schedule.find_tessera_opt() is None:pytest.skip("matching native compiler required")
    scalar,owner,values,wanted=case(mode,ta,tb,prefix)
    before=copy.deepcopy(scalar.graph_ir)
    graph=owner._trace_frontend_capture(values,{})[0]
    assert graph.functions[0].result_types[0].shape==tuple(map(str,wanted.shape))
    assert native.supports_nvfp4_matmul(graph)
    artifact=schedule.lower_scheduled_matmul(graph,target="nvidia_sm120")
    assert (artifact.m,artifact.n,artifact.k)==(math.prod(prefix)*7,5,31)
    assert artifact.graph_ir.count("!tessera.nvfp4")>=2
    assert "tile.matmul_kernel" in artifact.tile_ir
    assert scalar.graph_ir==before

def test_nested_policy_change_is_rejected_before_capture():
    scalar,owner,_,_=case("independent_rhs",False,False)
    with pytest.raises(ValueError,match="matching coupled leading policies"):
        vmap(owner,in_axes=AXES["shared_rhs_rows"])

def test_equal_product_prefix_mutation_is_rejected_before_compile(monkeypatch):
    if schedule.find_tessera_opt() is None:pytest.skip("matching native compiler required")
    from tessera.compiler.graph_ir import tensor_ir_type
    _,owner,values,_=case("independent_rhs",False,False)
    graph=owner._trace_frontend_capture(values,{})[0]
    artifact=schedule.lower_scheduled_matmul(graph,target="nvidia_sm120")
    changed=copy.deepcopy(graph)
    changed.functions[0].args[1].ir_type=tensor_ir_type((3,2,31,5),"nvfp4")
    assert not native.supports_nvfp4_matmul(changed)
    monkeypatch.setattr(native,"_compile_tile_ir",lambda *a:pytest.fail("compiled mismatched logical prefix"))
    with pytest.raises(ValueError):native.package_nvfp4_matmul(changed,pipeline_name="tessera-lower-to-nvidia-sm120",scheduled_artifact=artifact)

def test_package_retains_full_logical_tuple_and_rejects_rebinding_before_cuda(monkeypatch):
    if schedule.find_tessera_opt() is None:pytest.skip("matching native compiler required")
    from dataclasses import replace
    from tessera import runtime as rt
    _,owner,values,_=case("independent_rhs",False,False)
    graph=owner._trace_frontend_capture(values,{})[0]
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","toolchain",(),"cold"))
    package=native.package_nvfp4_matmul(graph,pipeline_name="tessera-lower-to-nvidia-sm120")
    desc=package.descriptor
    assert desc.provenance["logical_batch_shape"]==[2,3]
    assert desc.provenance["batch_rows"]==[6,7]
    assert [b.rank for b in desc.buffers]==[4,4,4,4,4]
    assert rt._nvfp4_logical_batch_prefix(desc)==(2,3)
    changed=replace(desc,provenance={**desc.provenance,"logical_batch_shape":[3,2]})
    with pytest.raises(RuntimeError,match="output guards"):rt._nvfp4_logical_batch_prefix(changed)


def test_valid_equal_product_tuple_cannot_reuse_another_schedule(monkeypatch):
    if schedule.find_tessera_opt() is None:pytest.skip("matching native compiler required")
    _,owner,values,_=case("independent_rhs",False,False,(2,3))
    graph=owner._trace_frontend_capture(values,{})[0]
    artifact=schedule.lower_scheduled_matmul(graph,target="nvidia_sm120")
    _,other,values2,_=case("independent_rhs",False,False,(3,2))
    changed=other._trace_frontend_capture(values2,{})[0]
    assert native.supports_nvfp4_matmul(changed)
    monkeypatch.setattr(native,"_compile_tile_ir",lambda *a:pytest.fail("compiled stale logical tuple"))
    with pytest.raises(ValueError,match="different logical Graph"):
        native.package_nvfp4_matmul(changed,pipeline_name="tessera-lower-to-nvidia-sm120",scheduled_artifact=artifact)


def test_nested_nvfp4_capability_is_chip_and_dtype_specific():
    from tessera.compiler.capabilities import supports_op
    for rank in (2, 3, 4, 5):
        cap = supports_op("nvidia_sm120", "scaled_matmul", dtype="nvfp4", rank=rank)
        assert cap.supported
        assert cap.runtime_status == "artifact_only"
    assert not supports_op("nvidia_sm120", "scaled_matmul", dtype="nvfp4", rank=1).supported
    assert not supports_op("nvidia_sm120", "scaled_matmul", dtype="fp16", rank=4).supported
    for target in ("nvidia_sm90", "rocm_gfx1151", "apple_gpu", "x86"):
        assert not supports_op(target, "scaled_matmul", dtype="nvfp4", rank=4).supported


@pytest.mark.parametrize("original,changed", [
    ("tensor<2x3x31x5x!tessera.nvfp4>", "tensor<3x2x31x5x!tessera.nvfp4>"),
    ("tensor<2x3x2x5xui8>", "tensor<3x2x2x5xui8>"),
])
def test_direct_native_graph_rejects_equal_product_prefix_permutation(original,changed):
    if schedule.find_tessera_opt() is None:
        pytest.skip("matching native compiler required")
    _,owner,values,_=case("independent_rhs",False,False)
    graph=owner._trace_frontend_capture(values,{})[0]
    artifact=schedule.lower_scheduled_matmul(graph,target="nvidia_sm120")
    malformed=artifact.graph_ir.replace(original,changed)
    assert malformed != artifact.graph_ir
    with pytest.raises(RuntimeError,match="NVFP4"):
        schedule.run_tessera_opt(schedule.find_tessera_opt(),malformed,"--canonicalize")
