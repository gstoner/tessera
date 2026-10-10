"""Native reverse programs retain independent and shared scale contributions."""
import base64
import copy
import json
import os
from dataclasses import replace
import numpy as np
import pytest
import tessera as ts
from tests.unit.test_composed_scaled_jvp import composed,case as forward_case
from tessera.compiler.native_scaled_program import package_native_scaled_vjp

def shared(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
           sa:ts.Tensor["M","G","fp32"],sb0:ts.Tensor["G","C","fp32"],sb1:ts.Tensor["G","C","fp32"]):
    first=ts.ops.scaled_matmul(a,b,sa,sb0,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    second=ts.ops.scaled_matmul(a,b,sa,sb1,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    return ts.ops.add(first,second)

def case(shape=(17,19,256),shared_scale=False,wrt=None):
    _,values,_=forward_case(shape)
    if shared_scale:
        values=(values[0],values[1],values[2],values[3],values[5])
        wrt=wrt or ("sb1","sa","sb0")
    else:wrt=wrt or ("sb1","sa0","sb0","sa1")
    owner=ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=wrt)(shared if shared_scale else composed)
    rng=np.random.default_rng(10873)
    dy=rng.uniform(-.2,.2,(shape[0],shape[1])).astype(np.float32)
    return owner,values,dy

def package(owner,values):
    graph=owner._traced_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    return package_native_scaled_vjp(graph.to_mlir(target="rocm_gfx1201"))

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
@pytest.mark.parametrize("shared_scale",[False,True])
def test_reverse_program_preserves_role_frame_and_shared_contributions(shared_scale):
    owner,values,_=case(shared_scale=shared_scale)
    result=package(owner,values);p=json.loads(result.program_json)
    assert p["argument_count"]==len(values)+1
    assert p["gradient_roles"]==list(owner.differentiation_request.wrt_indices)
    assert len(p["outputs"])==len(p["gradient_roles"])
    assert sum(step["operation"]=="tensor.generate" for step in p["steps"])==4
    assert sum(step["operation"]=="tessera.add" for step in p["steps"])==int(shared_scale)
    for step in p["steps"]:
        if step["operation"]=="tensor.generate":
            assert step["inputs"][-1]==len(values)
            assert step["gradient_argument"] not in step["inputs"]
        else:
            assert all(p["steps"][i-p["argument_count"]]["gradient_argument"]==step["gradient_argument"] for i in step["inputs"])

def forge(package,p):
    encoded=base64.b64encode(json.dumps(p).encode()).decode()
    members=[]
    for raw in package.members_json:
        value=json.loads(raw);value["program_base64"]=encoded
        step=p["steps"][value["step"]]
        value["inputs"]=step["inputs"]
        members.append(json.dumps(value))
    return replace(package,program_json=json.dumps(p),members_json=tuple(members))

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
def test_reverse_capture_and_shared_sum_lineage_are_checked():
    owner,values,_=case(shared_scale=True);result=package(owner,values)
    p=json.loads(result.program_json)
    bad=copy.deepcopy(p);bad["steps"][0]["inputs"][-1]=0
    with pytest.raises(ValueError,match="captured-input ABI"):forge(result,bad).validate()
    bad=copy.deepcopy(p)
    index=next(i for i,x in enumerate(bad["steps"]) if x["operation"]=="tessera.add")
    bad["steps"][index]["gradient_argument"]=bad["gradient_roles"][0]
    with pytest.raises(ValueError,match="gradient contribution lineage"):forge(result,bad).validate()
