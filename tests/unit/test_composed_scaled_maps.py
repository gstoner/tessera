"""Mapped product/sum Graphs preserve per-argument map intent and scalar owners."""
import copy,json,os
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tests.unit.test_composed_scaled_jvp import composed,case as scalar_case
from tests.unit.test_composed_scaled_vjp import shared
from tessera.compiler.native_scaled_program import package_native_scaled_jvp,package_native_scaled_vjp,package_native_scaled_primal

def case(policy="all",depth=1,mode="forward",shared_scale=False,shape=(3,5,256)):
    _,values,_=scalar_case(shape)
    names=("sa","sb0","sb1") if shared_scale else ("sa0","sb0","sa1","sb1")
    if shared_scale:values=(values[0],values[1],values[2],values[3],values[5])
    axes={"all":(0,)*len(values),
          "scales":(None,None)+((0,)* (len(values)-2)),
          "lhs":(0,None,0,None,None) if shared_scale else (0,None,0,None,0,None)}[policy]
    options={"target":"rocm_gfx1201"}
    if mode is not None:options.update(autodiff=mode,wrt=names)
    scalar=ts.jit(**options)(shared if shared_scale else composed)
    prefix=(2,) if depth==1 else (2,3)
    mapped=scalar
    for _ in range(depth):mapped=vmap(mapped,in_axes=axes)
    rng=np.random.default_rng(10889)
    arrays=[]
    for value,axis in zip(values,axes,strict=True):
        if axis is None:arrays.append(value.copy());continue
        full=np.broadcast_to(value,(*prefix,*value.shape)).copy()
        factors=rng.uniform(.8,1.2,(*prefix,1,1)).astype(np.float32)
        arrays.append((full.astype(np.float32)*factors).astype(value.dtype))
    arrays=tuple(arrays)
    seeds=(() if mapped.differentiation_request is None else
           tuple(rng.uniform(-.1,.1,arrays[i].shape).astype(np.float32) for i in mapped.differentiation_request.wrt_indices))
    return scalar,mapped,arrays,seeds,axes,prefix

@pytest.mark.parametrize("policy",["all","scales","lhs"])
@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("mode",["forward","reverse"])
def test_composed_map_frontend_certifies_full_graph_without_mutating_scalar(policy,depth,mode):
    scalar,owner,values,seeds,axes,prefix=case(policy,depth,mode)
    before=copy.deepcopy(scalar.graph_ir)
    graph=owner._specialized_autodiff_module(values,{})
    assert [op.op_name for op in graph.functions[0].body]==["tessera.scaled_matmul","tessera.scaled_matmul","tessera.add"]
    assert tuple(map(int,graph.functions[0].result_types[0].shape))==(*prefix,3,5)
    assert owner.frontend_differential(*values)
    assert scalar.graph_ir==before
    assert owner._frontend_batch_axes==axes
    assert len(owner._frontend_batch_policies)==depth

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
@pytest.mark.parametrize("mode",["forward","reverse"])
@pytest.mark.parametrize("shared_scale",[False,True])
def test_composed_mapped_native_export_owns_full_batch_and_role_frame(mode,shared_scale):
    _,owner,values,_,_,prefix=case("scales",2,mode,shared_scale)
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder=package_native_scaled_vjp if mode=="reverse" else package_native_scaled_jvp
    package=builder(graph.to_mlir(target="rocm_gfx1201"))
    p=json.loads(package.program_json)
    assert p["kind"]==("scale_vjp" if mode=="reverse" else "paired_jvp")
    assert len(package.images)==len(p["steps"])
    if mode=="reverse":
        for output,role in zip(p["outputs"],p["gradient_roles"],strict=True):
            assert p["buffers"][output]["shape"]==list(values[role].shape)
    else:
        for output in p["outputs"]:assert p["buffers"][output]["shape"]==[*prefix,3,5]


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
def test_ordinary_composed_mapped_primal_exports_all_native_members():
    _,owner,values,_,_,prefix=case("all",2,None)
    graph=owner._traced_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_primal(graph.to_mlir(target="rocm_gfx1201"))
    p=json.loads(package.program_json)
    assert p["kind"]=="primal"
    assert [step["operation"] for step in p["steps"]]==["tessera.scaled_matmul","tessera.scaled_matmul","tessera.add"]
    assert p["buffers"][p["outputs"][0]]["shape"]==[*prefix,3,5]
    assert len(package.images)==3
