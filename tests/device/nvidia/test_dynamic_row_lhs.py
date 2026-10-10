"""Owning SM120 dynamic row-RHS Schedule/Tile and native owner proof."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle,
)

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning SM120 required")
FUNCTIONS={"rmsnorm":(rms_lhs,rms_lhs_fused),"layernorm":(layer_lhs,layer_lhs_fused),
           "softmax":(softmax_lhs,softmax_lhs_fused)}
AXES=(("M",),("N",),("K",),("M","N"),("M","K"),("N","K"),("M","N","K"))


@pytest.mark.parametrize("axes",AXES)
@pytest.mark.parametrize("kind",FUNCTIONS)
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
def test_dynamic_row_rhs_public_and_portable(axes,kind,dtype,fused,monkeypatch):
    from tessera.compiler import nvidia_tensor_lhs as lhs
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    bounds={axis:value for axis,value in zip(("M","N","K"),(33,23,49),strict=True) if axis in axes}
    function=ts.jit(target="nvidia_sm120",shape_bounds=bounds,rhs_storage_order="row_major")(FUNCTIONS[kind][fused]._fn)
    rng=np.random.default_rng(125070)
    storage=_storage(dtype)
    x=(rng.normal(size=(33,49))*.2).astype(storage)
    b=(rng.normal(size=(49,23))*.2).astype(storage)
    bias=(rng.normal(size=23)*.2).astype(np.float32)
    residual=(rng.normal(size=(33,23))*.2).astype(np.float32)
    first=function(x,b,bias,residual) if fused else function(x,b)
    program=function._nvidia_lhs_last_program
    assert program.edge.consumer.descriptor.provenance["b_layout"]=="row_major"
    assert ".a_row_b_d_m_n_k_lda_ldb_ldd." in program.edge.consumer.descriptor.abi_id
    assert 'order = "row_major", leading_dim = 0' in program.edge.consumer.tile_ir
    artifact=rt.RuntimeArtifact.from_json(lhs.runtime_artifact(program).to_json())
    np.testing.assert_allclose(first,_oracle(x,b,kind,bias if fused else None,residual if fused else None),rtol=.015,atol=.015)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm row-RHS shape reuse reached tracing/compiler")
    monkeypatch.setattr(function,"_traced_autodiff_module",forbidden)
    monkeypatch.setattr(lhs,"find_tessera_opt",forbidden)
    owner=PreparedLhsCall(program)
    try:
        # Capacity is warmed first; subsequent padded C/F frames cannot grow scratch.
        owner((x,b,bias,residual) if fused else (x,b))
        stats=owner.scratch_stats()
        for m,k,n in ((17,35,19),(1,1,1),(33,49,23)):
            m=m if "M" in axes else 33
            n=n if "N" in axes else 23
            k=k if "K" in axes else 49
            source=x[:m,:k]
            # Switch caller physical pitch/order while keeping the sealed row image.
            rhs=np.array(b[:k,:n],order="F")
            args=(source,rhs,bias[:n],residual[:m,:n]) if fused else (source,rhs)
            expected=_oracle(source,rhs,kind,bias[:n] if fused else None,residual[:m,:n] if fused else None)
            actual=function(*args)
            assert function._nvidia_lhs_last_program is program
            prepared,receipt=owner(args)
            replay=rt.launch(artifact,args)
            assert replay["ok"] and receipt["native_call_binding"]=="prepared_cpp_dynamic_tensor_matmul"
            np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
            np.testing.assert_array_equal(actual,prepared)
            np.testing.assert_array_equal(actual,replay["output"])
            assert owner.scratch_stats()==stats
    finally:
        owner.close()
        function.close_native_storage()


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_dynamic_row_resident_pitch_validation(dtype):
    from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    storage=_storage(dtype)
    x=np.ones((17,35),storage)
    b=np.ones((35,19),storage)
    program=rms_lhs.compile_native_lhs_matmul(x,b,dynamic_axes=("M","N","K"),rhs_storage_order="row_major")
    package=program.edge.consumer
    artifact=NvidiaNormRhsProgram.runtime_artifact(package)
    with NvidiaDeviceSession() as session:
        # Genuinely padded row-major allocation is admitted with matching LDB.
        backing=np.zeros((35,24),storage)
        backing[:,:19]=1
        allocation=session.upload(backing)
        class PaddedRowView:
            shape=(35,19)
            dtype=allocation.dtype
            tessera_layout="strided"
            @property
            def __cuda_array_interface__(self):
                interface=dict(allocation.__cuda_array_interface__)
                interface.update(shape=self.shape,strides=(24*self.dtype.itemsize,self.dtype.itemsize))
                return interface
        db=PaddedRowView()
        da=session.upload(x,layout="strided")
        dd=session.empty((17,19),np.float32,layout="strided")
        values={program.edge.consumer_input_name:da,program.edge.consumer_rhs_name:db,
                program.edge.output_name:dd,"M":17,"N":19,"K":35,"LDA":35,"LDB":24,"LDD":19}
        receipt=rt.launch(artifact,values,stream=session.stream)
        assert receipt["ok"],str(receipt)
        np.testing.assert_allclose(session.download(dd),35,rtol=1e-6)
        values["LDB"]=19
        with pytest.raises(RuntimeError,match="storage disagrees"):
            rt._submit_nvidia_sm120_native(package.image,package.descriptor,
                {name:value for name,value in values.items() if not isinstance(value,int)},
                {name:value for name,value in values.items() if isinstance(value,int)},session.stream)


@pytest.mark.parametrize("malformation",["zero","overflow","rhs_k","pitch","bytes","output","dtype"])
def test_native_row_guard_preserves_output(malformation):
    from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
    from tests.device.nvidia.test_prepared_dynamic_lhs_owner import _views
    x=np.ones((17,35),np.float16)
    b=np.ones((35,19),np.float16)
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(x,b,dynamic_axes=("M","N","K"),rhs_storage_order="row_major"))
    try:
        output=np.full((17,19),-123,np.float32)
        views=_views((x,b,output))
        before=owner.scratch_stats()
        if malformation=="zero":views[0].shape[0]=0
        elif malformation=="overflow":views[1].shape[1]=20
        elif malformation=="rhs_k":views[1].shape[0]=34
        elif malformation=="pitch":views[1].strides[0]+=2
        elif malformation=="bytes":views[1].bytes-=2
        elif malformation=="output":views[2].shape[0]-=1
        else:views[1].dtype=1
        assert owner.lib.tessera_nvidia_matmul_invoke(owner.handle,views,3)!=0
        np.testing.assert_array_equal(output,-123)
        assert owner.scratch_stats()==before
        actual,_=owner((x[:3,:11],b[:11,:7]))
        np.testing.assert_allclose(actual,_oracle(x[:3,:11],b[:11,:7],"rmsnorm"),rtol=.015,atol=.015)
    finally:
        owner.close()


@pytest.mark.parametrize("dtype",["fp16","bf16"])
def test_row_abi_fresh_process_without_compiler(dtype,tmp_path):
    import os
    import subprocess
    import sys
    from tessera.compiler import nvidia_tensor_lhs as lhs
    storage=_storage(dtype)
    x=np.ones((17,35),storage)
    b=np.ones((35,19),storage)
    program=rms_lhs.compile_native_lhs_matmul(x,b,dynamic_axes=("M","N","K"),rhs_storage_order="row_major")
    path=tmp_path/"row.json"
    path.write_text(lhs.runtime_artifact(program).to_json())
    script="""
import sys
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs,scheduled_matmul,nvidia_native
def forbidden(*args,**kwargs):
    raise AssertionError("portable row replay invoked compiler")
for module in (nvidia_tensor_lhs,scheduled_matmul):
    module.find_tessera_opt=forbidden
    module.run_tessera_opt=forbidden
nvidia_native.package_scheduled_tensor_matmul=forbidden
if sys.argv[2]=="bf16":
    import ml_dtypes
    dtype=ml_dtypes.bfloat16
else: dtype=np.float16
artifact=rt.RuntimeArtifact.from_json(Path(sys.argv[1]).read_text())
for m,k,n in ((17,35,19),(3,11,7),(1,1,1)):
    receipt=rt.launch(artifact,(np.ones((m,k),dtype),np.ones((k,n),dtype)))
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    expected=float(np.array(1/np.sqrt(1+1e-5),dtype=dtype))*k
    np.testing.assert_allclose(receipt["output"],expected,rtol=.015,atol=.015)
"""
    env=dict(os.environ,TESSERA_OPT="/nonexistent/compiler")
    result=subprocess.run([sys.executable,"-c",script,str(path),dtype],env=env,
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
