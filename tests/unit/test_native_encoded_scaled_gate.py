"""Native encoded residual coefficients with continuous scale/gate derivatives."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import supports_scaled_reverse
from tessera.compiler.native_scaled_program import (
    package_native_scaled_primal, package_native_scaled_jvp, package_native_scaled_vjp)

from tessera.dtype import Dtype

encoded_byte = Dtype("uint8", allow_planned_gated=True)

def fp8_gated(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["K","N","fp8_e4m3"],  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
              sa:ts.Tensor["M","G","fp32"], sb:ts.Tensor["G","C","fp32"],  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
              gate:ts.Tensor["M","N","fp32"]):  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
    product=ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})
    return product*gate

def mxfp8_gated(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["K","N","fp8_e4m3"],  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
                sa:ts.Tensor["M","G",encoded_byte], sb:ts.Tensor["G","N",encoded_byte],  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
                gate:ts.Tensor["M","N","fp32"]):  # noqa: F821 - Tessera symbolic dimensions and dtype spellings
    product=ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":"e8m0"})
    return product*gate

def typed_case(fmt,mode="reverse",roles_override=None):
    storage=pytest.importorskip("ml_dtypes").float8_e4m3fn
    m,n,k=17,19,256
    rng=np.random.default_rng(18711)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(storage)
    b=rng.choice([-1,0,.5,2],(k,n)).astype(storage)
    if fmt=="fp32":
        sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
        sb=rng.uniform(.2,1,(k//128,1)).astype(np.float32)
        roles=("sa","sb","gate")
    else:
        sa=rng.integers(125,130,(m,k//32),dtype=np.uint8)
        sb=rng.integers(125,130,(k//32,n),dtype=np.uint8)
        roles=("gate",)
    if roles_override is not None:
        roles = roles_override
    gate=rng.uniform(-.5,.5,(m,n)).astype(np.float32)
    options={} if mode is None else {"autodiff":mode,"wrt":roles}
    owner=ts.jit(target="rocm_gfx1201",**options)(fp8_gated if fmt=="fp32" else mxfp8_gated)
    values=(a,b,sa,sb,gate)
    directions=tuple(rng.uniform(-.1,.1,v.shape).astype(np.float32)
                     if name in roles else None for name,v in zip(owner.arg_names,values,strict=True))
    return owner,values,directions

@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_encoded_product_gate_owns_native_residual_recomputation(fmt,mode):
    owner,values,_=typed_case(fmt,mode)
    graph=owner._specialized_autodiff_module(values,{})
    before=copy.deepcopy(graph)
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builders={None:package_native_scaled_primal,"forward":package_native_scaled_jvp,
              "reverse":package_native_scaled_vjp}
    package=builders[mode](graph.to_mlir(target="rocm_gfx1201",canonical=True))
    manifest=json.loads(package.program_json)
    assert any(step["operation"]=="tessera.scaled_matmul" for step in manifest["steps"])
    assert any(step["operation"]=="tessera.mul" for step in manifest["steps"])
    assert len(manifest["outputs"])==(1 if mode is None else 2 if mode=="forward" else 3 if fmt=="fp32" else 1)
    assert graph.functions==before.functions
    package.validate()

@pytest.mark.parametrize('fmt,roles',[('fp32',(0,)),('e8m0',(0,)),('e8m0',(2,))])
def test_encoded_gate_does_not_admit_discrete_storage_gradients(fmt,roles):
    owner,values,_=typed_case(fmt)
    graph=owner._specialized_autodiff_module(values,{})
    assert not supports_scaled_reverse(graph,roles)


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"), reason="matching native compiler required")
@pytest.mark.parametrize("fmt,role", [("fp32", "a"), ("e8m0", "a"), ("e8m0", "sa")])
def test_native_export_rejects_encoded_storage_derivatives(fmt, role):
    owner, values, _ = typed_case(fmt, roles_override=(role,))
    graph = owner._specialized_autodiff_module(values, {})
    graph.module_attrs.update({"tessera.target": '"rocm"', "tessera.arch": '"gfx1201"'})
    with pytest.raises(RuntimeError, match="scaled transpose|derivative interface"):
        package_native_scaled_vjp(graph.to_mlir(target="rocm_gfx1201", canonical=True))
