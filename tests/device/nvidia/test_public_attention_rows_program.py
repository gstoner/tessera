"""Public runtime VJP v3 replay over the native row owner on exact SM120."""
import os
import subprocess

import ml_dtypes
import numpy as np
import pytest

import tessera.runtime as rt
from tessera.compiler.native_attention_program import compile_attention_vjp_program
from tessera.compiler.prepared_attention_vjp import clear_prepared
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark=pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
@pytest.mark.parametrize("bias,seeded",[(False,False),(False,True),(True,False),(True,True)])
def test_public_v3_restored_program_executes_without_compiler(storage,wide,bias,seeded,monkeypatch):
    assert rt._nvidia_device_name()=="sm_120", "owning GPU is required"
    dtype=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
    out_dtype=np.dtype(np.float32) if wide else dtype
    original=source(storage,bias=bias,saved=seeded,output="f32" if wide else storage)
    active=tuple(reversed(range(3+int(bias))))
    original=original.replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = ['+', '.join(map(str,active))+']')
    program=compile_attention_vjp_program(original,active,compiler=os.environ["TESSERA_OPT"],
        compact_gradients=True,row_correction=True)
    artifact=rt.RuntimeArtifact(graph_ir=original,target_ir=program.pair.backward.target_ir,
        metadata=dict(target="nvidia_sm120",compiler_path="nvidia_sm120_attention_vjp_compiled",
            execution_kind="native_gpu",execution_mode="cuda_runtime",executable=True,
            arg_names=[f"primal_{i}" for i in range(3+int(bias))]+["cotangent"]+(["lse_cotangent"] if seeded else []),
            program_json=program.to_json(),program_digest=program.program_digest))
    rng=np.random.default_rng(20261016)
    try:
        rt.backend_capabilities("nvidia_sm120")
        for _ in range(2):
            q=rng.normal(0,.4,(1,2,3,4)).astype(dtype)
            k=rng.normal(0,.4,(1,1,5,4)).astype(dtype)
            v=rng.normal(0,.4,(1,1,5,3)).astype(dtype)
            do=rng.normal(0,.4,(1,2,3,3)).astype(out_dtype)
            physical=rng.normal(0,.1,(1,2,1,5)).astype(np.float32) if bias else None
            seed=rng.normal(0,.2,(1,2,3)).astype(np.float32) if seeded else np.zeros((1,2,3),np.float32)
            _,expected=reference(q,k,v,do,physical,seed)
            values=(q,k,v)+((physical,) if bias else ())+(do,)+((seed,) if seeded else ())
            def forbidden(*args,**kwargs):pytest.fail("portable public v3 launch invoked compiler")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(subprocess,"check_output",forbidden)
                guard.setattr(subprocess,"Popen",forbidden)
                receipt=rt.launch(artifact,values)
            assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
            for actual,index in zip(receipt["output"],active,strict=True):compare(actual,expected[index])
    finally:clear_prepared()


@pytest.mark.parametrize("storage",["float16","bfloat16"])
@pytest.mark.parametrize("saved,bias",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("value_only",[False,True])
@pytest.mark.parametrize("sq,sk",[(3,5),(1,5),(3,1)])
def test_original_jit_frontend_reaches_v3_and_warm_replay_uses_all_member_identity(storage,saved,bias,value_only,sq,sk,monkeypatch):
    import hashlib
    import json
    import tessera as ts
    from tests.device.nvidia.test_public_typed_attention_vjp import function
    assert rt._nvidia_device_name()=="sm_120", "owning GPU required"
    monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","prepass")
    dtype=np.dtype(ml_dtypes.bfloat16 if storage=="bfloat16" else storage)
    rng=np.random.default_rng(5070)
    values={name:rng.normal(0,.3,shape).astype(dtype) for name,shape in {
        "q":(1,2,sq,4),"k":(1,1,sk,4),"v":(1,1,sk,3)}.items()}
    if bias:values["bias"]=rng.normal(0,.1,(1,2,1,sk)).astype(np.float32)
    do=rng.normal(0,.2,(1,2,sq,3)).astype(dtype)
    seed=rng.normal(0,.2,(1,2,sq)).astype(np.float32) if saved else np.zeros((1,2,sq),np.float32)
    _,expected=reference(values["q"],values["k"],values["v"],do,values.get("bias"),seed)
    wrt=("v",) if value_only else (("bias","q") if bias else ("v","k","q"))
    roles={"q":0,"k":1,"v":2,"bias":3}
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(function(saved,bias))
    seeds=(do,seed) if saved else do
    try:
        first=fn.native_backward(**values,out_cotangents=seeds)
        for actual,name in zip(first,wrt,strict=True):compare(actual,expected[roles[name]])
        held=tuple(x.copy() for x in first)
        artifact=rt.RuntimeArtifact.from_json(fn.native_backward_runtime_artifact().to_json())
        data=json.loads(artifact.metadata["program_json"])["program"]
        assert data["schema"]=="tessera.native_attention_vjp_program.v3"
        assert (data["correction"] is None)==value_only
        members=[data["forward"]]+([data["correction"]] if data["correction"] is not None else [])+[data["backward"]]
        receipt=fn.last_backward_execution
        assert receipt["tile_program_digest"]==hashlib.sha256("".join(x["tile_ir"] for x in members).encode()).hexdigest()
        ordered=tuple(values[name] for name in fn.arg_names)
        def forbidden(*args,**kwargs):pytest.fail("warm v3 frontend executed compiler or concrete function")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess,"run",forbidden)
            guard.setattr(subprocess,"check_output",forbidden)
            guard.setattr(subprocess,"Popen",forbidden)
            guard.setattr(fn,"_fn",forbidden)
            warm=fn.native_backward(*ordered,out_cotangents=seeds)
            replay=rt.launch(artifact,(*ordered,do,*((seed,) if saved else ())))
        assert replay["ok"] and replay["execution_kind"]=="native_gpu",replay
        for collection in (warm,replay["output"]):
            for actual,name in zip(collection,wrt,strict=True):compare(actual,expected[roles[name]])
        for actual,prior in zip(first,held,strict=True):np.testing.assert_array_equal(actual,prior)
        assert fn.last_frontend_differential.contract["structural_match"]
        assert fn.last_frontend_differential.contract["numerical_match"]
    finally:
        fn.close_native_storage();clear_prepared()


@pytest.mark.parametrize("storage",["float16","bfloat16"])
def test_public_resident_v3_orders_pending_roots_and_keeps_old_outputs(storage,monkeypatch):
    from contextlib import ExitStack
    import tessera as ts
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_public_typed_attention_vjp import function
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    assert rt._nvidia_device_name()=="sm_120"
    monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","prepass")
    dtype=np.dtype(np.float16 if storage=="float16" else ml_dtypes.bfloat16)
    rng=np.random.default_rng(9124)
    hosts={name:rng.normal(0,.3,shape).astype(dtype) for name,shape in {
        "q":(1,2,3,4),"k":(1,1,5,4),"v":(1,1,5,3)}.items()}
    hosts["bias"]=rng.normal(0,.1,(1,2,1,5)).astype(np.float32)
    do=rng.normal(0,.2,(1,2,3,3)).astype(dtype)
    seed=rng.normal(0,.2,(1,2,3)).astype(np.float32)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("bias","q"))(function(True,True))
    try:
        first=fn.native_backward(**hosts,out_cotangents=(do,seed));held=tuple(x.copy() for x in first)
        with ExitStack() as stack:
            ordered=tuple(hosts[name] for name in fn.arg_names)+(do,seed)
            sessions=tuple(stack.enter_context(NvidiaDeviceSession()) for _ in ordered)
            resident=tuple(session.upload(host) for session,host in zip(sessions,ordered,strict=True))
            for session in sessions:assert session.synchronize()==0
            index=fn.arg_names.index("k")
            changed=(hosts["k"].astype(np.float32)*.7).astype(dtype)
            completed=[]
            callback=queue_delayed_upload(sessions[index],resident[index],changed,sessions[index].stream,completed)
            actual=fn.native_backward(*resident[:4],out_cotangents=resident[4:])
            assert callback is not None and completed
            _,expected=reference(hosts["q"],changed,hosts["v"],do,hosts["bias"],seed)
            compare(actual[0],expected[3]);compare(actual[1],expected[0])
            assert fn.last_frontend_differential.contract["concrete_executions"]==0
        for output,prior in zip(first,held,strict=True):np.testing.assert_array_equal(output,prior)
    finally:fn.close_native_storage();clear_prepared()


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("value_only",[False,True])
def test_fresh_process_restores_v3_with_compilers_unavailable(storage,value_only,tmp_path):
    import json
    import sys
    assert rt._nvidia_device_name()=="sm_120"
    dtype=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
    active=(2,) if value_only else (3,0)
    original=source(storage,bias=True,saved=True,output="f32").replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = ['+', '.join(map(str,active))+']')
    program=compile_attention_vjp_program(original,active,compiler=os.environ["TESSERA_OPT"],compact_gradients=True,row_correction=True)
    artifact=rt.RuntimeArtifact(graph_ir=original,target_ir=program.pair.backward.target_ir,
        metadata=dict(target="nvidia_sm120",compiler_path="nvidia_sm120_attention_vjp_compiled",
            execution_kind="native_gpu",execution_mode="cuda_runtime",executable=True,
            arg_names=["primal_0","primal_1","primal_2","primal_3","cotangent","lse_cotangent"],
            program_json=program.to_json(),program_digest=program.program_digest))
    rng=np.random.default_rng(999)
    q=rng.normal(0,.3,(1,2,3,4)).astype(dtype);k=rng.normal(0,.3,(1,1,5,4)).astype(dtype)
    v=rng.normal(0,.3,(1,1,5,3)).astype(dtype);bias=rng.normal(0,.1,(1,2,1,5)).astype(np.float32)
    do=rng.normal(0,.2,(1,2,3,3)).astype(np.float32);seed=rng.normal(0,.2,(1,2,3)).astype(np.float32)
    values=(q,k,v,bias,do,seed)
    _,expected=reference(q,k,v,do,bias,seed)
    types=[x.dtype.name for x in values]
    arrays={f"input_{i}":x.view(np.uint16) if x.dtype.name=="bfloat16" else x for i,x in enumerate(values)}
    arrays.update({f"expected_{i}":expected[role] for i,role in enumerate(active)})
    np.savez(tmp_path/"values.npz",**arrays)
    (tmp_path/"artifact.json").write_text(artifact.to_json())
    (tmp_path/"types.json").write_text(json.dumps(types))
    script=tmp_path/"restore.py"
    script.write_text('''import json,sys,subprocess
from pathlib import Path
import numpy as np
import ml_dtypes
import tessera.runtime as rt
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare
root=Path(sys.argv[1])
rt.backend_capabilities("nvidia_sm120")
def forbidden(*args,**kwargs):raise AssertionError("fresh replay accessed compiler/subprocess")
subprocess.run=forbidden;subprocess.check_output=forbidden;subprocess.Popen=forbidden
artifact=rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
blob=np.load(root/"values.npz")
types=json.loads((root/"types.json").read_text())
values=tuple(blob[f"input_{i}"].view(ml_dtypes.bfloat16) if dtype=="bfloat16" else blob[f"input_{i}"] for i,dtype in enumerate(types))
receipt=rt.launch(artifact,values)
assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
for i,actual in enumerate(receipt["output"]):compare(actual,blob[f"expected_{i}"])
print("fresh-v3-native-replay-passed")
''')
    environment=dict(os.environ)
    for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_LLVM_BIN","LLVM_BIN"):
        environment[name]="/compiler-unavailable"
    completed=subprocess.run([sys.executable,str(script),str(tmp_path)],capture_output=True,text=True,env=environment,timeout=120)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert "fresh-v3-native-replay-passed" in completed.stdout


def test_public_policy_switch_has_distinct_compiler_and_prepared_cache_identity(monkeypatch):
    import json
    import tessera as ts
    from tests.device.nvidia.test_public_typed_attention_vjp import function
    assert rt._nvidia_device_name()=="sm_120"
    rng=np.random.default_rng(20261016)
    q=rng.normal(0,.3,(1,2,3,4)).astype(np.float16)
    k=rng.normal(0,.3,(1,1,5,4)).astype(np.float16)
    v=rng.normal(0,.3,(1,1,5,3)).astype(np.float16)
    do=rng.normal(0,.2,(1,2,3,3)).astype(np.float16)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("q","k","v"))(function())
    try:
        monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","inline")
        baseline=fn.native_backward(q,k,v,out_cotangents=do)
        old=fn.native_backward_runtime_artifact()
        assert json.loads(old.metadata["program_json"])["program"]["schema"]=="tessera.native_attention_vjp_program.v2"
        monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","prepass")
        changed=fn.native_backward(q,k,v,out_cotangents=do)
        new=fn.native_backward_runtime_artifact()
        assert json.loads(new.metadata["program_json"])["program"]["schema"]=="tessera.native_attention_vjp_program.v3"
        assert old.metadata["program_digest"]!=new.metadata["program_digest"]
        for a,b in zip(baseline,changed,strict=True):np.testing.assert_array_equal(a,b)
        monkeypatch.setenv("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION","inline")
        restored=fn.native_backward(q,k,v,out_cotangents=do)
        assert fn.native_backward_runtime_artifact().metadata["program_digest"]==old.metadata["program_digest"]
        for a,b in zip(baseline,restored,strict=True):np.testing.assert_array_equal(a,b)
    finally:fn.close_native_storage();clear_prepared()
