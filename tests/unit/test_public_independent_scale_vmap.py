"""Public typed map preserves independent scale prefixes and source semantics."""
import itertools
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.rocm_typed_scaled_native import supports_scale_transpose
from tessera.compiler.native_vmap import batch_specs

def product_nn(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
               sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,4],"format":"fp32"})

def product_nt(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
               sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,4],"format":"fp32"})

def product_tn(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
               sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,4],"format":"fp32"})

def product_tt(a:ts.Tensor["K","M","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
               sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeA=True,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,4],"format":"fp32"})

def case(mask,ta,tb,prefix,wrt=("sa","sb"),seed=1007):
    fn={(False,False):product_nn,(False,True):product_nt,
        (True,False):product_tn,(True,True):product_tt}[ta,tb]
    scalar=ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=wrt)(fn)
    axes=tuple(0 if mask&(1<<i) else None for i in range(4))
    owner=scalar
    for _ in prefix:owner=vmap(owner,in_axes=axes)
    rng=np.random.default_rng(seed)
    suffix=((7,3) if ta else (3,7),(5,7) if tb else (7,5),(3,2),(2,2))
    values=tuple(rng.uniform(-.5,.5,size=(*(prefix if axis is not None else ()),*shape)).astype(ml_dtypes.float8_e4m3fn)
                 if i<2 else rng.uniform(.3,1.3,size=(*(prefix if axis is not None else ()),*shape)).astype(np.float32)
                 for i,(axis,shape) in enumerate(zip(axes,suffix)))
    return scalar,owner,values,axes

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix",[(2,),(2,3),(2,1,3)])
def test_public_independent_reverse_projection_and_certificate(mask,ta,tb,prefix):
    scalar,owner,values,axes=case(mask,ta,tb,prefix)
    before=scalar.graph_ir.to_mlir(target="rocm_gfx1201")
    graph=owner._specialized_autodiff_module(values,{})
    assert supports_scale_transpose(graph)
    assert graph.functions[0].result_types[0].shape==tuple(map(str,(*prefix,3,5)))
    assert owner.frontend_differential(*values) is owner.frontend_differential(*values)
    assert scalar._frontend_batch_axes is None
    assert scalar.graph_ir.to_mlir(target="rocm_gfx1201")==before

def test_nonleading_and_boolean_axes_remain_invalid():
    _,_,values,_=case(15,False,False,(2,))
    for axes in ((False,0,0,0),(1,None,None,None)):
        with pytest.raises(ValueError):batch_specs(values,axes)

def test_source_bounds_precede_independent_projection(monkeypatch):
    from tessera.compiler.constraints import Range,TesseraConstraintError
    scalar,_,values,axes=case(4,False,False,(2,))
    scalar.constraints.add(Range("M",1,2))
    owner=vmap(scalar,in_axes=axes)
    def forbidden(*args,**kwargs):raise AssertionError("invalid source reached frontend")
    monkeypatch.setattr(owner,"_specialized_autodiff_module",forbidden)
    with pytest.raises(TesseraConstraintError):
        owner.native_backward(*values,out_cotangents=np.ones((2,3,5),np.float32))
