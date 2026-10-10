"""Composed scaled products preserve native SSA and scale-only JVP roles."""
import copy
import json
import os
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import supports_composed_scale_jvp
from tessera.compiler.native_scaled_program import package_native_scaled_jvp

def composed(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["K","N","fp8_e4m3"],
             sa0:ts.Tensor["M","G","fp32"], sb0:ts.Tensor["G","C","fp32"],
             sa1:ts.Tensor["M","G","fp32"], sb1:ts.Tensor["G","C","fp32"]):
    first=ts.ops.scaled_matmul(a,b,sa0,sb0,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    second=ts.ops.scaled_matmul(a,b,sa1,sb1,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    return ts.ops.add(first,second)

def case(shape=(17,19,256),wrt=("sa0","sb0","sa1","sb1"),seed=10871):
    storage=pytest.importorskip("ml_dtypes").float8_e4m3fn
    m,n,k=shape;g=(k+127)//128;c=(n+127)//128
    rng=np.random.default_rng(seed)
    values=(rng.uniform(-.5,.5,(m,k)).astype(storage),
        rng.uniform(-.5,.5,(k,n)).astype(storage),
        rng.uniform(.3,1.3,(m,g)).astype(np.float32),rng.uniform(.3,1.3,(g,c)).astype(np.float32),
        rng.uniform(.3,1.3,(m,g)).astype(np.float32),rng.uniform(.3,1.3,(g,c)).astype(np.float32))
    owner=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=wrt)(composed)
    seeds=tuple(rng.uniform(-.1,.1,values[owner.arg_names.index(name)].shape).astype(np.float32) for name in wrt)
    return owner,values,seeds

@pytest.mark.parametrize("wrt",[("sa0",),("sa1","sb1"),("sb1","sa0","sb0","sa1")])
def test_composed_graph_scale_roles_and_scalar_owner(wrt):
    owner,values,seeds=case(wrt=wrt)
    graph=owner._traced_autodiff_module(values,{})
    before=graph.to_mlir(target="rocm_gfx1201")
    assert supports_composed_scale_jvp(graph,owner.differentiation_request.wrt_indices)
    assert graph.to_mlir(target="rocm_gfx1201")==before
    assert [op.op_name for op in graph.functions[0].body]==["tessera.scaled_matmul","tessera.scaled_matmul","tessera.add"]
    assert not supports_composed_scale_jvp(graph,(0,))
    assert not supports_composed_scale_jvp(graph,(2,2))

def test_composed_admission_rejects_changed_semantics():
    owner,values,_=case()
    graph=owner._traced_autodiff_module(values,{})
    for kind in ("activation","scale_format","shape"):
        changed=copy.deepcopy(graph)
        if kind=="activation":changed.functions[0].body[-1].op_name="tessera.relu"
        elif kind=="scale_format":changed.functions[0].body[0].kwargs["scale_layout"]["format"]="e8m0"
        else:changed.functions[0].body[0].inferred_type=None
        assert not supports_composed_scale_jvp(changed,(2,3,4,5))

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
def test_composed_jvp_exports_actual_native_product_and_sum_members():
    owner,values,_=case()
    graph=owner._traced_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_jvp(graph.to_mlir(target="rocm_gfx1201"))
    manifest=json.loads(package.program_json)
    assert manifest["argument_count"]==10
    assert len(package.images)==10
    assert sum(step["operation"]=="tessera.scaled_matmul" for step in manifest["steps"])==6
    assert sum(step["operation"]=="tessera.add" for step in manifest["steps"])==4
    assert len(manifest["outputs"])==2
    assert all(buffer["ownership"]==1 for buffer in manifest["buffers"][10:] if buffer["id"] not in manifest["outputs"])
