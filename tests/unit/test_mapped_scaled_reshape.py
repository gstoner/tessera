"""Mapped continuous reshape SSA preserves value-specific prefixes and AD roles."""
import copy,json,os
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap

SOURCE="""def reshaped_chain(a: tessera.Tensor["M","K","fp32"], b: tessera.Tensor["K","N","fp32"],
                   sa: tessera.Tensor["M","G","fp32"], sb: tessera.Tensor["G","C","fp32"],
                   c: tessera.Tensor["J","P","fp32"], sc: tessera.Tensor["R","S","fp32"],
                   sd: tessera.Tensor["S","D","fp32"]):
    first=tessera.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[2,4],"format":"fp32"})
    edge=tessera.ops.reshape(first,shape=(6,2))
    return tessera.ops.scaled_matmul(edge,c,sc,sd,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,2],"format":"fp32"})
"""
ROLES=("a","b","sa","sb","c","sc","sd")

def case(mask=127,mode=None,prefix=(2,),out_axes=0):
    options={} if mode is None else {"autodiff":mode,"wrt":ROLES}
    scalar=ts.from_text(SOURCE,target="rocm_gfx1201",**options)
    axes=tuple(0 if mask&(1<<i) else None for i in range(7))
    owner=scalar
    for level in range(len(prefix)):
        owner=vmap(owner,in_axes=axes,out_axes=out_axes if level==len(prefix)-1 else 0)
    rng=np.random.default_rng(19301)
    shapes=((3,7),(7,4),(3,2),(2,2),(2,5),(6,1),(1,5))
    values=tuple(rng.uniform(-.4,.4,(*prefix,*shape) if axis is not None else shape).astype(np.float32)
                 for shape,axis in zip(shapes,axes,strict=True))
    return scalar,owner,values,axes

@pytest.mark.parametrize("mask",range(1,128))
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_each_root_map_policy_preserves_reshape_prefix(mask,mode):
    scalar,owner,values,axes=case(mask,mode)
    before=copy.deepcopy(scalar.graph_ir)
    graph=owner._specialized_autodiff_module(values,{})
    first,reshape,last=graph.functions[0].body
    prefix=(2,) if any(a is not None for a in axes[:4]) else ()
    assert tuple(map(int,first.inferred_type.shape))==(*prefix,3,4)
    assert tuple(map(int,reshape.inferred_type.shape))==(*prefix,6,2)
    assert tuple(reshape.kwargs["shape"])==(*prefix,6,2)
    assert tuple(map(int,last.inferred_type.shape))==(2,6,5)
    assert scalar.graph_ir==before
    assert owner.frontend_differential(*values)

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching compiler required")
@pytest.mark.parametrize("mask",[1,16,64,127])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
@pytest.mark.parametrize("out_axes",[0,-1])
def test_native_mapped_reshape_byte_and_lifetime_contract(mask,mode,out_axes):
    from tessera.compiler.native_scaled_program import (
        package_native_scaled_primal,package_native_scaled_jvp,package_native_scaled_vjp)
    _,owner,values,_=case(mask,mode,(2,3),out_axes)
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder=package_native_scaled_vjp if mode=="reverse" else package_native_scaled_jvp if mode=="forward" else package_native_scaled_primal
    package=builder(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    plan=json.loads(package.program_json)
    carriers=[step for step in plan["steps"] if step["operation"]=="tessera.reshape"]
    assert carriers
    for step in carriers:
        assert plan["buffers"][step["inputs"][0]]["bytes"]==plan["buffers"][step["output"]]["bytes"]
        assert step["lowering"]=="structured_f32_carrier"
    if mode=="reverse":
        for role,slot in zip(plan["gradient_roles"],plan["outputs"],strict=True):
            assert plan["buffers"][slot]["shape"]==list(values[role].shape)
    else:
        shape=[2,3,6,5] if out_axes==0 else [3,6,5,2]
        for slot in plan["outputs"]:assert plan["buffers"][slot]["shape"]==shape
    package.validate()


@pytest.mark.parametrize("shape", [(6, 3), (-1, 2), (True, 12), "6,2"])
def test_projection_preserves_authored_reshape_contract(shape):
    from tessera.compiler.native_vmap import project_batch
    scalar, _, values, axes = case()
    scalar_values = tuple(value[0] for value in values)
    graph = scalar._specialized_autodiff_module(scalar_values, {})
    graph.functions[0].body[1].kwargs["shape"] = shape
    before = copy.deepcopy(graph)
    with pytest.raises(ValueError, match="reshape shape differs"):
        project_batch(graph, values, axes)
    assert graph == before
