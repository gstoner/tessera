"""Owning SM120 long-capacity short/long/short resident softmax lifetime."""
from contextlib import ExitStack,closing
import subprocess
import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.device.nvidia.test_native_tensor_dag import public_dag_deep,oracle
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed,queue_delayed_upload

pytestmark=pytest.mark.hardware_nvidia

@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_bounded_cooperative_dag_reuses_capacity_and_retires_borrowed_reads(dtype,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact sm120 required")
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(19251)
    graph=public_dag_deep._traced_autodiff_module((
        np.zeros((17,35),storage),np.zeros((35,19),storage)),{})
    program=lhs.package_traced_lhs(graph,shape_bounds={"M":257,"N":128,"K":1024})
    program=lhs.from_manifest(program.manifest())
    assert program.producer_chain[1].descriptor.provenance["schedule"]=="cooperative_128"
    assert program.rhs_chain[0].descriptor.provenance["schedule"]=="serial"
    def forbidden(*args,**kwargs):pytest.fail("warm bounded resident DAG invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession());right=stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        owner=stack.enter_context(closing(PreparedLhsCall(program)))
        control=stack.enter_context(closing(PreparedLhsCall(program)))
        ac=left.upload(np.zeros((257,1024),storage));bc=right.upload(np.zeros((1024,128),storage))
        oc=launch.empty((257,128),np.float32)
        assert not left.synchronize() and not right.synchronize()
        owner.invoke_resident([Borrowed(ac.view(0,(17,35),storage)),Borrowed(bc.view(0,(35,19),storage))],
            oc.view(0,(17,19),np.float32),stream=launch.stream)
        stats=owner.scratch_stats();retained=[]
        for m,n,k in ((17,19,35),(129,65,513),(1,1,1),(17,19,35)):
            a=rng.normal(0,.2,(m,k)).astype(storage);b=rng.normal(0,.2,(k,n)).astype(storage)
            av=ac.view(0,(m,k),storage);bv=bc.view(0,(k,n),storage);output=oc.view(0,(m,n),np.float32)
            complete=[]
            callbacks=[queue_delayed_upload(left,av,a,left.stream,complete),
                       queue_delayed_upload(right,bv,b,right.stream,complete)]
            assert not complete
            roots=[Borrowed(av),Borrowed(bv)]
            owner.invoke_resident(roots,output,stream=launch.stream)
            assert len(complete)==len(callbacks)
            assert owner.scratch_stats()==stats
            np.testing.assert_allclose(output.numpy(),oracle(a,b,2),rtol=.015,atol=.015)
            host,_=control([a,b]);np.testing.assert_array_equal(output.numpy(),host)
            result=program.execute_resident(*roots);stack.callback(result.close)
            snapshot=result.output.numpy()
            np.testing.assert_array_equal(snapshot,host)
            retained.append((result,snapshot))
        left.close();right.close()
        for result,snapshot in retained:np.testing.assert_array_equal(result.output.numpy(),snapshot)
