"""Direct frontend broadcast retains all four independent prefixes."""
import itertools
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera.dtype import Dtype
encoded_byte = Dtype("uint8", allow_planned_gated=True)
from tessera.compiler.graph_ir import _shape_scaled_matmul, tensor_ir_type
from tessera.compiler.rocm_typed_scaled_native import supports_typed_scaled, supports_scale_transpose

def direct_0_00(a:ts.Tensor['P','U','M','K','fp8_e4m3'],b:ts.Tensor['Q','K','N','fp8_e4m3'],
                  sa:ts.Tensor['M','G','fp32'],sb:ts.Tensor['R','Q','G','C','fp32']):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=False,transposeB=False,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":'fp32'})

def direct_0_01(a:ts.Tensor['P','U','M','K','fp8_e4m3'],b:ts.Tensor['Q','N','K','fp8_e4m3'],
                  sa:ts.Tensor['M','G','fp32'],sb:ts.Tensor['R','Q','G','C','fp32']):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=False,transposeB=True,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":'fp32'})

def direct_0_10(a:ts.Tensor['P','U','K','M','fp8_e4m3'],b:ts.Tensor['Q','K','N','fp8_e4m3'],
                  sa:ts.Tensor['M','G','fp32'],sb:ts.Tensor['R','Q','G','C','fp32']):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=True,transposeB=False,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":'fp32'})

def direct_0_11(a:ts.Tensor['P','U','K','M','fp8_e4m3'],b:ts.Tensor['Q','N','K','fp8_e4m3'],
                  sa:ts.Tensor['M','G','fp32'],sb:ts.Tensor['R','Q','G','C','fp32']):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=True,transposeB=True,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[3,32],"format":'fp32'})

def direct_1_00(a:ts.Tensor['P','U','M','K','fp8_e4m3'],b:ts.Tensor['Q','K','N','fp8_e4m3'],
                  sa:ts.Tensor['M','G',encoded_byte],sb:ts.Tensor['R','Q','G','N',encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=False,transposeB=False,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":'e8m0'})

def direct_1_01(a:ts.Tensor['P','U','M','K','fp8_e4m3'],b:ts.Tensor['Q','N','K','fp8_e4m3'],
                  sa:ts.Tensor['M','G',encoded_byte],sb:ts.Tensor['R','Q','G','N',encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=False,transposeB=True,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":'e8m0'})

def direct_1_10(a:ts.Tensor['P','U','K','M','fp8_e4m3'],b:ts.Tensor['Q','K','N','fp8_e4m3'],
                  sa:ts.Tensor['M','G',encoded_byte],sb:ts.Tensor['R','Q','G','N',encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=True,transposeB=False,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":'e8m0'})

def direct_1_11(a:ts.Tensor['P','U','K','M','fp8_e4m3'],b:ts.Tensor['Q','N','K','fp8_e4m3'],
                  sa:ts.Tensor['M','G',encoded_byte],sb:ts.Tensor['R','Q','G','N',encoded_byte]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        transposeA=True,transposeB=True,batching="broadcast",
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,32],"format":'e8m0'})

DIRECT={(encoded,ta,tb):globals()[f"direct_{int(encoded)}_{int(ta)}{int(tb)}"]
        for encoded,ta,tb in itertools.product((False,True),repeat=3)}

def case(encoded=False,ta=False,tb=False,mode=None,seed=8108):
    options={} if mode is None else {"autodiff":mode,"wrt":("sa","sb")}
    owner=ts.jit(target="rocm_gfx1201",**options)(DIRECT[encoded,ta,tb])
    rng=np.random.default_rng(seed)
    shapes=((2,1,35,3) if ta else (2,1,3,35),
            (3,5,35) if tb else (3,35,5),(3,2),(1,3,2,5 if encoded else 2))
    values=tuple(rng.uniform(-.5,.5,size=shape).astype(ml_dtypes.float8_e4m3fn)
                 if i<2 else (rng.integers(126,129,size=shape,dtype=np.uint8)
                 if encoded else rng.uniform(.3,1.3,size=shape).astype(np.float32))
                 for i,shape in enumerate(shapes))
    return owner,values

def oracle(values,ta,tb,encoded=False):
    a,b,sa,sb=(np.asarray(v).astype(np.float64) for v in values)
    if ta:a=a.swapaxes(-1,-2)
    if tb:b=b.swapaxes(-1,-2)
    if encoded:sa=np.exp2(sa-127);sb=np.exp2(sb-127)
    out=np.zeros((2,3,3,5),np.float64)
    for k in range(35):
        out+=(a[..., :,k,None]*b[...,None,k,:]*
              sa[..., :,k//32,None]*sb[...,k//32,np.arange(5)//(1 if encoded else 3)][...,None,:])
    return out

@pytest.mark.parametrize("encoded,ta,tb",tuple(itertools.product((False,True),repeat=3)))
def test_direct_broadcast_trace_joins_all_operand_prefixes(encoded,ta,tb):
    owner,values=case(encoded,ta,tb)
    graph=owner._specialized_autodiff_module(values,{})
    assert tuple(map(int,graph.functions[0].result_types[0].shape))==(2,3,3,5)
    assert supports_typed_scaled(graph)
    if not encoded:assert supports_scale_transpose(graph)
    before=owner.graph_ir.to_mlir(target="rocm_gfx1201")
    assert 'batching = "broadcast"' in graph.to_mlir(target="rocm_gfx1201")
    assert owner.graph_ir.to_mlir(target="rocm_gfx1201")==before

@pytest.mark.parametrize("prefixes",[
    ((2,3),(3,2),(),()),((2,1),(3,),(),(1,4)),((0,3),(),(),())])
def test_direct_broadcast_rejects_incompatible_or_empty_prefixes(prefixes):
    types=[tensor_ir_type((*p,*suffix),"fp8_e4m3" if i<2 else "fp32")
           for i,(p,suffix) in enumerate(zip(prefixes,((3,35),(35,5),(3,2),(2,2))))]
    with pytest.raises(ValueError,match="prefixes do not broadcast|positive extents"):
        _shape_scaled_matmul(types,{"batching":"broadcast"})

def test_broadcast_never_reinterprets_a_named_packed_contract():
    types=[tensor_ir_type(shape,"fp8_e4m3") for shape in ((3,35),(35,5),(3,2),(2,2))]
    with pytest.raises(ValueError,match="semantic typed product"):
        _shape_scaled_matmul(types,{"batching":"broadcast","physical_contract":"nvidia_sm120_nvfp4_blockscale_v1"})


@pytest.mark.parametrize("role",range(4))
def test_any_operand_can_supply_the_result_batch_prefix(role):
    # In particular, scale-only batching must not disappear from the output.
    suffixes=((3,35),(35,5),(3,2),(2,2))
    types=[tensor_ir_type((*((2,3) if i==role else ()),*suffix),
                         "fp8_e4m3" if i<2 else "fp32")
           for i,suffix in enumerate(suffixes)]
    result=_shape_scaled_matmul(types,{"batching":"broadcast"})
    assert result.shape==("2","3","3","5")
