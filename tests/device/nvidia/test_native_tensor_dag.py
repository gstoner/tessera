"""Exact SM120 native ownership of both operand chains."""
from contextlib import closing
from copy import deepcopy
import subprocess
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.unit.test_native_sm120_tensor_dag import dag_module


def module(dtype, rhs_first=False, depth=1):
    graph=dag_module(dtype,rhs_first=rhs_first)
    fn=graph.functions[0];producers=fn.body[:-1];consumer=fn.body[-1]
    if depth==2:
        rows=[]
        for first in producers:
            second=deepcopy(first)
            second.op_name="tessera.layer_norm" if first.op_name=="tessera.softmax" else "tessera.softmax"
            second.kwargs={"eps":1e-5} if second.op_name=="tessera.layer_norm" else {"axis":-1}
            second.operands=["%"+first.result];second.result=first.result+"_second"
            consumer.operands=[("%"+second.result if v=="%"+first.result else v) for v in consumer.operands]
            rows.extend((first,second))
        fn.body=[*rows,consumer]
    return graph


def oracle(a,b,depth):
    def norm(x,kind):
        x=x.astype(np.float64)
        if kind=="softmax":
            e=np.exp(x-x.max(axis=-1,keepdims=True))
            return (e/e.sum(axis=-1,keepdims=True)).astype(a.dtype)
        c=x-x.mean(axis=-1,keepdims=True) if kind=="layernorm" else x
        return (c/np.sqrt(np.mean(c*c,axis=-1,keepdims=True)+1e-5)).astype(a.dtype)
    left=norm(a,"rmsnorm");right=norm(b,"softmax")
    if depth==2:left=norm(left,"softmax");right=norm(right,"layernorm")
    return left.astype(np.float64)@right.astype(np.float64)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("rhs_first",[False,True])
@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("bounded",[False,True])
def test_native_two_operand_program_replay(dtype,rhs_first,depth,bounded,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    graph=module(dtype,rhs_first,depth)
    original=graph.to_mlir(canonical=True,target="nvidia_sm120")
    program=lhs.package_traced_lhs(graph,shape_bounds={"M":32,"N":24,"K":64} if bounded else None)
    assert graph.to_mlir(canonical=True,target="nvidia_sm120")==original
    data=program.manifest()
    assert data["schema"]=="tessera.nvidia.lhs_tensor_program.v4"
    def forbidden(*args,**kwargs):pytest.fail("warm DAG replay invoked compiler or Graph constructor")
    monkeypatch.setattr(subprocess,"run",forbidden)
    restored=lhs.from_manifest(data)
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(923)
    frames=[(17,11,32),(32,24,64),(1,1,1)] if bounded else [(17,11,32)]*3
    with closing(PreparedLhsCall(restored)) as prepared:
        retained=[];snapshots=[];stats=None
        for m,n,k in frames:
            a=rng.normal(0,.2,(m,k)).astype(storage)
            b=rng.normal(0,.2,(k,n)).astype(storage)
            output,receipt=prepared([a,b])
            assert receipt["execution_kind"]=="native_gpu"
            assert len(receipt["component_receipts"])==depth*2+1
            np.testing.assert_allclose(output,oracle(a,b,depth),rtol=.015,atol=.015)
            retained.append(output);snapshots.append(output.copy())
            if stats is None:stats=prepared.scratch_stats()
            assert prepared.scratch_stats()==stats
            profile=prepared.profile(4)
            assert profile["program_ms"]>0
            assert len(profile["grouped_stage_ms"])==depth*2+1
            np.testing.assert_allclose(profile["output"],oracle(a,b,depth),rtol=.015,atol=.015)
        for output,snapshot in zip(retained,snapshots,strict=True):
            np.testing.assert_array_equal(output,snapshot)

import tessera as ts


@ts.jit(target="nvidia_sm120")
def public_dag(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),
                         ts.ops.softmax(rhs,axis=-1),output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def public_dag_reordered(rhs,source):
    b=ts.ops.softmax(rhs,axis=-1)
    a=ts.ops.rmsnorm(source,eps=1e-5)
    return ts.ops.matmul(a,b,output_dtype="fp32")


@ts.jit(target="nvidia_sm120")
def public_dag_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),
                         ts.ops.softmax(rhs,axis=-1),bias=bias,activation="relu",
                         residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120")
def public_dag_deep(source,rhs):
    a=ts.ops.softmax(ts.ops.rmsnorm(source,eps=1e-5),axis=-1)
    b=ts.ops.layer_norm(ts.ops.softmax(rhs,axis=-1),eps=1e-5)
    return ts.ops.matmul(a,b,output_dtype="fp32")


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("variant",["ordinary","reordered","fused","bounded"])
def test_public_two_operand_jit_and_portable_replay(dtype,variant,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    base=public_dag_reordered if variant=="reordered" else public_dag_fused if variant=="fused" else public_dag
    function=ts.jit(target="nvidia_sm120",**(
        {"shape_bounds":{"M":32,"N":24,"K":64}} if variant=="bounded" else {}))(base._fn)
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(882)
    def frame(m,n,k):
        a=rng.normal(0,.2,(m,k)).astype(storage);b=rng.normal(0,.2,(k,n)).astype(storage)
        expected=oracle(a,b,1)
        if variant=="fused":
            bias=rng.normal(0,.2,n).astype(np.float32);residual=rng.normal(0,.2,(m,n)).astype(np.float32)
            expected=(np.maximum(expected+bias,0)+residual).astype(np.float16)
            args=(a,b,bias,residual)
        else:args=(b,a) if variant=="reordered" else (a,b)
        return args,expected
    args,expected=frame(17,19,35)
    first=function(*args);saved=first.copy()
    np.testing.assert_allclose(first,expected,rtol=.015,atol=.015)
    assert function.execution_kind=="native_gpu"
    packages=function.native_lhs_packages()
    assert len(packages)==3
    assert packages[-1].descriptor.provenance["b_layout"]=="row_major"
    portable=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
    def forbidden(*args,**kwargs):pytest.fail("warm public DAG used eager arithmetic or compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"compile_native_lhs_matmul",forbidden)
    args,expected=frame(*( (32,24,64) if variant=="bounded" else (17,19,35)))
    output=function(*args)
    np.testing.assert_allclose(output,expected,rtol=.015,atol=.015)
    receipt=rt.launch(portable,args)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    np.testing.assert_array_equal(receipt["output"],output)
    np.testing.assert_array_equal(first,saved)
    assert function.native_lhs_packages()==packages


@pytest.mark.hardware_nvidia
def test_native_profile_rejects_stale_arena_lease():
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    program=lhs.package_traced_lhs(module("fp16"))
    a=np.full((17,32),.2,np.float16);b=np.full((32,11),.3,np.float16)
    with closing(PreparedLhsCall(program)) as first, closing(PreparedLhsCall(program)) as second:
        first([a,b])
        second([a*.5,b*.75])
        with pytest.raises(RuntimeError,match="lease is stale"):first.profile(4)
        output,_=first([a,b])
        profile=first.profile(4)
        assert profile["program_ms"]>0 and len(profile["grouped_stage_ms"])==3
        np.testing.assert_array_equal(profile["output"],output)


@pytest.fixture(scope="module")
def packaged_dag():
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    return lhs.package_traced_lhs(module("fp16"))


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("mutation",["drop","kind","geometry","capacity","digest"])
def test_two_operand_component_corruption_rejected_before_cuda(packaged_dag,mutation):
    from dataclasses import replace
    from tessera.compiler.native_artifact import LaunchGeometry
    program=deepcopy(packaged_dag)
    right=program.rhs_chain[0];descriptor=right.descriptor
    if mutation=="drop":program=replace(program,rhs_chain=())
    elif mutation=="kind":
        sem=deepcopy(program.semantics);sem["rhs_chain"][0]={"producer":"tessera.rmsnorm","producer_attrs":{"eps":1e-5}}
        program=replace(program,semantics=sem)
    elif mutation=="geometry":
        descriptor=replace(descriptor,geometry=LaunchGeometry(policy="sm120_norm_serial_rows"))
    elif mutation=="capacity":
        guards=list(descriptor.shape_guards);guards[0]=replace(guards[0],value=guards[0].value+1)
        descriptor=replace(descriptor,shape_guards=tuple(guards))
    else:descriptor=replace(descriptor,provenance={**descriptor.provenance,"native_tensor_program_digest":"0"*64})
    if mutation not in {"drop","kind"}:
        program=replace(program,rhs_chain=(replace(right,descriptor=descriptor),))
    with pytest.raises(ValueError):program.validate()


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("bounded",[False,True])
def test_owned_resident_result_computes_both_chains(dtype,depth,bounded,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    program=lhs.package_traced_lhs(module(dtype,depth=depth),**(
        {"shape_bounds":{"M":32,"N":24,"K":64}} if bounded else {}))
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(1913)
    def frame(m,n,k):
        return (rng.normal(0,.2,(m,k)).astype(storage),
                rng.normal(0,.2,(k,n)).astype(storage))
    a,b=frame(17,11,32)
    first=program.execute_resident(a,b)
    try:
        assert first.intermediate is None
        assert first.consumer_receipt["native_call_binding"]=="prepared_cpp_owned_resident_tensor_dag"
        assert len(first.producer_receipt["component_receipts"])==2*depth
        saved=first.output.numpy()
        np.testing.assert_allclose(saved,oracle(a,b,depth),rtol=.015,atol=.015)
        def forbidden(*args,**kwargs):pytest.fail("resident replay invoked compiler")
        monkeypatch.setattr(subprocess,"run",forbidden)
        a,b=frame(*( (32,24,64) if bounded else (17,11,32)))
        with program.execute_resident(a,b) as second:
            np.testing.assert_allclose(second.output.numpy(),oracle(a,b,depth),rtol=.015,atol=.015)
            np.testing.assert_array_equal(first.output.numpy(),saved)
        with pytest.raises(RuntimeError,match="closed"):
            second.output.numpy()
    finally:first.close()
    first.close()
    with pytest.raises(RuntimeError,match="closed"):first.output.numpy()
