"""Ordinary primal JIT must execute the compiler-owned SM120 attention image."""
from itertools import permutations
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.benchmark_jvp_argument_order import function
from tests._support.nvidia import nvidia_cuda_host_ready


def forward_oracle(q,k,v,causal,bias=None):
    q,k,v=(x.astype(np.float64) for x in (q,k,v))
    groups=q.shape[1]//k.shape[1]
    k,v=np.repeat(k,groups,axis=1),np.repeat(v,groups,axis=1)
    scores=(q@np.swapaxes(k,-1,-2))/np.sqrt(q.shape[-1])
    if bias is not None:scores+=bias.astype(np.float64)
    if causal:
        sq,sk=q.shape[-2],k.shape[-2]
        scores=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),scores,-np.inf)
    shifted=np.exp(scores-scores.max(axis=-1,keepdims=True))
    return (shifted/shifted.sum(axis=-1,keepdims=True))@v


@pytest.mark.parametrize("order",list(permutations(("q","k","v"))))
@pytest.mark.parametrize("causal",[False,True])
@pytest.mark.parametrize("shape",[(1,2,1,3,5,4,3),(2,4,2,5,3,4,6)])
def test_ordinary_attention_jit_permutations(order,causal,shape,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 device/toolchain required")
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(5070)
    arrays={name:rng.normal(size=s).astype(np.float32)*.2 for name,s in zip(
        ("q","k","v"),((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)),strict=True)}
    expected=forward_oracle(*(arrays[n] for n in ("q","k","v")),causal)
    fn=ts.jit(target="nvidia_sm120")(function(order,causal))
    fn._trace_frontend_capture(tuple(arrays[n] for n in order),{})
    def forbidden(*args,**kwargs):pytest.fail("ordinary compiled attention called eager arithmetic")
    monkeypatch.setattr(ts.ops,"flash_attn",forbidden)
    output=fn(**arrays)
    assert fn.execution_kind=="native_gpu"
    np.testing.assert_allclose(output,expected,rtol=3e-5,atol=3e-5)
    artifact=rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
    assert artifact.launch_descriptor.provenance["route"]=="canonical_scheduled_tile_consumer"
    provenance=artifact.launch_descriptor.provenance
    assert provenance["schedule_digest"]
    for key in ("graph_ir_digest","schedule_ir_digest","tile_ir_digest","target_ir_digest"):
        assert len(provenance[key])==64
    assert provenance["target_ir_digest"]==artifact.native_image.target_ir_digest
    assert "tile.attention_kernel" in artifact.tile_ir
    bindings={arg.name:arrays[name] for arg,name in zip(fn._trace_frontend_capture(tuple(arrays[n] for n in order),{})[0].functions[0].args,order,strict=True)}
    bindings[artifact.launch_descriptor.buffers[-1].name]=np.empty(output.shape,np.float32)
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True))
    receipt=rt.launch(artifact,{"buffers":bindings,"scalars":scalars})
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    np.testing.assert_array_equal(bindings[artifact.launch_descriptor.buffers[-1].name],output)
    import importlib
    canonical=importlib.import_module("tessera.compiler.canonical_compile")
    monkeypatch.setattr(canonical,"canonical_compile",forbidden)
    arrays["v"]*=np.float32(.5)
    changed=fn(*(arrays[n] for n in order))
    np.testing.assert_allclose(changed,expected*.5,rtol=3e-5,atol=3e-5)


def biased_function(order,causal):
    def first(bias,v,q,k):return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=causal)
    def middle(k,bias,v,q):return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=causal)
    return first if order==("bias","v","q","k") else middle


@pytest.mark.parametrize("order",[("bias","v","q","k"),("k","bias","v","q")])
@pytest.mark.parametrize("causal",[False,True])
def test_ordinary_attention_full_bias_operand_roles(order,causal,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 device/toolchain required")
    rng=np.random.default_rng(5071)
    arrays={n:rng.normal(size=s).astype(np.float32)*.1 for n,s in zip(
        ("q","k","v","bias"),((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,3,5)),strict=True)}
    expected=forward_oracle(arrays["q"],arrays["k"],arrays["v"],causal,arrays["bias"])
    fn=ts.jit(target="nvidia_sm120")(biased_function(order,causal))
    fn._trace_frontend_capture(tuple(arrays[n] for n in order),{})
    def forbidden(*args,**kwargs):pytest.fail("compiled biased attention invoked eager arithmetic")
    monkeypatch.setattr(ts.ops,"flash_attn",forbidden)
    output=fn(**arrays)
    assert fn.execution_kind=="native_gpu"
    np.testing.assert_allclose(output,expected,rtol=3e-5,atol=3e-5)
    descriptor=fn.runtime_artifact().launch_descriptor
    assert descriptor.provenance["bias"] is True
    assert [x.direction for x in descriptor.buffers]==["input"]*4+["output"]


def saved_function(order, causal, bias=False):
    def direct(q,k,v):
        return ts.ops.flash_attn(q,k,v,causal=causal,lse_checkpoint="saved")
    def reversed_args(v,k,q):
        return ts.ops.flash_attn(q,k,v,causal=causal,lse_checkpoint="saved")
    def bias_first(bias,v,q,k):
        return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=causal,lse_checkpoint="saved")
    return bias_first if bias else direct if order==("q","k","v") else reversed_args


def saved_forward_oracle(q,k,v,causal,bias=None):
    groups=q.shape[1]//k.shape[1]
    keys=np.repeat(k.astype(np.float64),groups,axis=1)
    scores=(q.astype(np.float64)@np.swapaxes(keys,-1,-2))/np.sqrt(q.shape[-1])
    if bias is not None: scores+=bias.astype(np.float64)
    if causal:
        sq,sk=q.shape[-2],k.shape[-2]
        scores=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),scores,-np.inf)
    row_max=scores.max(axis=-1)
    lse=row_max+np.log(np.exp(scores-row_max[...,None]).sum(axis=-1))
    return forward_oracle(q,k,v,causal,bias),lse


@pytest.mark.parametrize("causal",[False,True])
@pytest.mark.parametrize("order,bias",[(("q","k","v"),False),(("v","k","q"),False),(("bias","v","q","k"),True)])
@pytest.mark.parametrize("shape",[(1,2,1,3,5,4,3),(2,4,2,5,3,4,6)])
def test_ordinary_saved_lse_attention_jit(order,bias,causal,shape,monkeypatch):
    if not nvidia_cuda_host_ready(): pytest.skip("exact SM120 device/toolchain required")
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(5072)
    arrays={name:rng.normal(size=s).astype(np.float32)*.2 for name,s in zip(
        ("q","k","v"),((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)),strict=True)}
    if bias: arrays["bias"]=rng.normal(size=(1,hq,1,sk)).astype(np.float32)*.1
    oracle=saved_forward_oracle(arrays["q"],arrays["k"],arrays["v"],causal,arrays.get("bias"))
    fn=ts.jit(target="nvidia_sm120")(saved_function(order,causal,bias))
    result=fn(**arrays)
    assert isinstance(result,tuple) and len(result)==2
    assert fn.execution_kind=="native_gpu"
    for actual,expected in zip(result,oracle,strict=True):
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
    artifact=rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
    assert "tile.attention_kernel" in artifact.tile_ir
    assert artifact.launch_descriptor.provenance["schedule_digest"]
    outputs=[x for x in artifact.launch_descriptor.buffers if x.direction=="output"]
    assert [x.rank for x in outputs]==[4,3]
    module,_=fn._trace_frontend_capture(tuple(arrays[n] for n in order),{})
    assert len(module.functions[0].result_types)==2
    def forbidden(*args,**kwargs): pytest.fail("saved-LSE warm call used eager arithmetic or recompilation")
    import importlib
    monkeypatch.setattr(importlib.import_module("tessera.compiler.canonical_compile"),"canonical_compile",forbidden)
    monkeypatch.setattr(fn,"_fn",forbidden)
    arrays["v"]*=np.float32(.5)
    changed=fn(**arrays)
    np.testing.assert_allclose(changed[0],oracle[0]*.5,rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(changed[1],oracle[1],rtol=3e-5,atol=3e-5)


def saved_alias_function(order,causal,bias=False):
    def direct(q,k,v):
        pair=ts.ops.flash_attn(q,k,v,causal=causal,lse_checkpoint="saved")
        return pair
    def reversed_args(v,k,q):
        pair=ts.ops.flash_attn(q,k,v,causal=causal,lse_checkpoint="saved")
        copy=pair
        pair=q
        return copy
    def bias_first(bias,v,q,k):
        pair=ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=causal,lse_checkpoint="saved")
        output,lse=pair
        return output,lse
    return bias_first if bias else direct if order==("q","k","v") else reversed_args


@pytest.mark.parametrize("order,bias",[(("q","k","v"),False),(("v","k","q"),False),(("bias","v","q","k"),True)])
@pytest.mark.parametrize("causal",[False,True])
def test_saved_lse_tuple_aliases_execute_native(order,bias,causal,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 device/toolchain required")
    rng=np.random.default_rng(507063)
    arrays={name:rng.normal(size=s).astype(np.float32)*.2 for name,s in zip(
        ("q","k","v"),((2,4,5,4),(2,2,3,4),(2,2,3,6)),strict=True)}
    if bias: arrays["bias"]=rng.normal(size=(1,4,1,3)).astype(np.float32)*.1
    expected=saved_forward_oracle(arrays["q"],arrays["k"],arrays["v"],causal,arrays.get("bias"))
    fn=ts.jit(target="nvidia_sm120")(saved_alias_function(order,causal,bias))
    assert len(fn.graph_ir.functions[0].result_types)==2
    result=fn(**arrays)
    assert fn.execution_kind=="native_gpu"
    assert fn.runtime_artifact().launch_descriptor.provenance["schedule_digest"]
    for actual,reference in zip(result,expected,strict=True):
        np.testing.assert_allclose(actual,reference,rtol=3e-5,atol=3e-5)
    def forbidden(*args,**kwargs):pytest.fail("aliased saved tuple executed Python after compilation")
    monkeypatch.setattr(fn,"_fn",forbidden)
    arrays["v"]*=np.float32(.5)
    result=fn(**arrays)
    np.testing.assert_allclose(result[0],expected[0]*.5,rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(result[1],expected[1],rtol=3e-5,atol=3e-5)
