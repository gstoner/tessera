"""Ordinary frontend scale-only batch axes retain native scaled ownership."""
import numpy as np
import ml_dtypes
import tessera as ts

def scaled(a,b,sa,sb):
    return ts.ops.scaled_matmul(a,b,sa,sb,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":"fp32"})

def case(role,mode=None):
    options={} if mode is None else {"autodiff":mode,"wrt":("sa","sb")}
    owner=ts.jit(target="rocm_gfx1201",**options)(scaled)
    rng=np.random.default_rng(9210+role)
    shapes=[(3,35),(35,5),(3,2),(2,2)]
    shapes[role]=(2,3,*shapes[role])
    values=tuple(rng.uniform(-.5,.5,size=shape).astype(ml_dtypes.float8_e4m3fn)
        if i<2 else rng.uniform(.3,1.3,size=shape).astype(np.float32)
        for i,shape in enumerate(shapes))
    return owner,values

def oracle(values):
    a,b,sa,sb=(np.asarray(v).astype(np.float64) for v in values)
    out=np.zeros((2,3,3,5),np.float64)
    for k in range(35):
        out+=a[..., :,k,None]*b[...,None,k,:]*sa[..., :,k//32,None]*sb[...,k//32,np.arange(5)//3][...,None,:]
    return out

def test_scale_only_frontend_joins_operand_axes():
    from tessera.compiler.rocm_typed_scaled_native import supports_typed_scaled
    for role in range(4):
        owner,values=case(role)
        graph=owner._specialized_autodiff_module(values,{})
        assert graph.functions[0].result_types[0].shape==("2","3","3","5")
        assert supports_typed_scaled(graph)
