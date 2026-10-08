"""Saved attention results and reference AD semantics, independent of GPU availability."""
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.trace import trace, to_graph_ir_module
from tessera.compiler.graph_ir import _infer_result_types, tensor_ir_type
from tessera.autodiff.jvp import jvp_flash_attn
from tessera.autodiff.vjp import vjp_flash_attn


def saved(q,k,v):
    return ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")


def oracle(values, causal, dropout=0.0, seed=None):
    q,k,v,bias=values
    groups=q.shape[1]//k.shape[1]
    k,v=np.repeat(k,groups,axis=1),np.repeat(v,groups,axis=1)
    scores=q@np.swapaxes(k,-1,-2)/np.sqrt(q.shape[-1])+bias
    if causal:
        sq,sk=scores.shape[-2:]
        mask=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
        scores=np.where(mask,scores,-np.inf)
    maximum=scores.max(axis=-1,keepdims=True)
    exponential=np.exp(scores-maximum)
    denominator=exponential.sum(axis=-1,keepdims=True)
    probabilities=exponential/denominator
    if dropout:
        probabilities=probabilities*np.random.default_rng(seed).binomial(
            1,1-dropout,probabilities.shape)/(1-dropout)
    return probabilities@v,maximum[...,0]+np.log(denominator[...,0])


@pytest.mark.parametrize("dtype",["fp16","fp32","fp64"])
def test_saved_lse_shape_preserves_primary_storage_and_f32_auxiliary(dtype):
    types=[tensor_ir_type(shape,dtype) for shape in
           (("B","Hq","Sq","D"),("B","Hkv","Sk","D"),("B","Hkv","Sk","Dv"))]
    output,lse=_infer_result_types("tessera.flash_attn",types,{"lse_checkpoint":"saved"})
    assert output.shape==("B","Hq","Sq","Dv") and output.dtype==dtype
    assert lse.shape==("B","Hq","Sq") and lse.dtype=="fp32"


def test_direct_multi_result_return_is_retained_in_ast_and_trace():
    fn=ts.jit(saved,target="nvidia_sm120")
    assert len(fn.graph_ir.functions[0].return_values)==2
    traced=trace(saved,((1,2,3,4),"fp32"),((1,1,5,4),"fp32"),((1,1,5,3),"fp32"))
    module=to_graph_ir_module(traced,name="saved",target="nvidia_sm120")
    assert [t.shape for t in module.functions[0].result_types]==[("1","2","3","3"),("1","2","3")]
    assert module.functions[0].body[0].kwargs["lse_checkpoint"]=="saved"


@pytest.mark.parametrize("dtype",[np.float16,np.float32,np.float64])
def test_eager_saved_lse_auxiliary_width(dtype):
    q=np.zeros((1,2,3,4),dtype)
    k=np.zeros((1,1,5,4),dtype)
    v=np.ones((1,1,5,3),dtype)
    output,lse=saved(q,k,v)
    assert output.dtype==dtype and lse.dtype==np.float32
    np.testing.assert_allclose(output,1,rtol=0,atol=0)
    np.testing.assert_allclose(lse,np.log(5),rtol=1e-6,atol=0)


@pytest.mark.parametrize("causal",[False,True])
@pytest.mark.parametrize("heads",[(2,1),(4,2)])
@pytest.mark.parametrize("dropout",[0.0,0.25])
def test_saved_attention_adjoint_and_tangent_match_independent_direction(causal,heads,dropout):
    hq,hkv=heads
    rng=np.random.default_rng(120619)
    shapes=((1,hq,3,4),(1,hkv,5,4),(1,hkv,5,2),(1,hq,1,5))
    values=tuple(rng.normal(size=s)*.2 for s in shapes)
    tangents=tuple(rng.normal(size=s)*.1 for s in shapes)
    expected=oracle(values,causal,dropout,19)
    cotangents=tuple(rng.normal(size=x.shape) for x in expected)
    epsilon=1e-6
    plus=oracle(tuple(p+epsilon*t for p,t in zip(values,tangents,strict=True)),causal,dropout,19)
    minus=oracle(tuple(p-epsilon*t for p,t in zip(values,tangents,strict=True)),causal,dropout,19)
    numerical=tuple((p-m)/(2*epsilon) for p,m in zip(plus,minus,strict=True))
    kwargs={"causal":causal,"dropout_p":dropout,"seed":19,"lse_checkpoint":"saved"}
    primal,directional=jvp_flash_attn(values,tangents,**kwargs)
    for actual,reference in zip(primal,expected,strict=True):
        np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-6)
    for actual,reference in zip(directional,numerical,strict=True):
        np.testing.assert_allclose(actual,reference,rtol=2e-7,atol=2e-9)
    gradients=[vjp_flash_attn(c,*values,_output_index=i,**kwargs) for i,c in enumerate(cotangents)]
    assert [g.shape for g in gradients[0]]==[v.shape for v in values]
    assert [g.shape for g in gradients[1]]==[v.shape for v in values]
    np.testing.assert_array_equal(gradients[1][2],np.zeros_like(values[2]))
    lhs=sum(np.vdot(c,t) for c,t in zip(cotangents,directional,strict=True))
    rhs=sum(np.vdot(g0+g1,t) for g0,g1,t in zip(*gradients,tangents,strict=True))
    np.testing.assert_allclose(lhs,rhs,rtol=2e-10,atol=2e-10)



def test_attention_trace_avoids_eager_math_and_explicit_oracle_can_evaluate(monkeypatch):
    from tessera.autodiff.tape import _make_wrapper
    arrays=(np.zeros((1,2,3,4),np.float32),np.zeros((1,1,5,4),np.float32),
            np.ones((1,1,5,3),np.float32))
    def forbidden(*args,**kwargs):
        pytest.fail("typed production trace evaluated eager attention")
    original=ts.ops.flash_attn
    monkeypatch.setattr(ts.ops,"flash_attn",_make_wrapper("flash_attn",forbidden))
    typed=trace(saved,*arrays)
    assert len(typed.output_values)==2 and all(value is None for value in typed.output_values)
    monkeypatch.setattr(ts.ops,"flash_attn",original)
    evaluated=trace(saved,*arrays,evaluate_catalog_outputs=True)
    assert all(value is not None for value in evaluated.output_values)
    np.testing.assert_allclose(evaluated.output_values[1],np.log(5),rtol=1e-6,atol=0)


def saved_bound(q,k,v):
    pair=ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")
    return pair


def saved_copied(q,k,v):
    pair=ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")
    copy=pair
    pair=q
    return copy


def saved_unpacked(q,k,v):
    pair=ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")
    output,lse=pair
    return output,lse


def saved_selected(q,k,v):
    pair=ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")
    output=pair[0]
    lse=pair[-1]
    return output,lse


def saved_rebound(q,k,v):
    pair=ts.ops.flash_attn(q,k,v,lse_checkpoint="saved")
    pair=q
    return pair


@pytest.mark.parametrize("function",[saved_bound,saved_copied,saved_unpacked,saved_selected])
def test_tuple_aliases_preserve_all_producer_ssa_components(function):
    fn=ts.jit(function,target="nvidia_sm120")
    graph=fn.graph_ir.functions[0]
    assert graph.return_values==["%"+n for n in graph.body[0].result_names]
    assert len(graph.result_types)==2
    traced=trace(function,((1,2,3,4),"fp32"),((1,1,5,4),"fp32"),((1,1,5,3),"fp32"))
    typed=to_graph_ir_module(traced,name="aliased",target="nvidia_sm120").functions[0]
    assert [t.shape for t in typed.result_types]==[("1","2","3","3"),("1","2","3")]


def test_tuple_rebinding_does_not_retain_stale_auxiliary_results():
    graph=ts.jit(saved_rebound,target="nvidia_sm120").graph_ir.functions[0]
    assert graph.return_values==["%q"]
    assert len(graph.result_types)==1
