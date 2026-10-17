"""Public bounded row VJP packages execute native images across runtime shapes."""
import os
import json
import subprocess

import ml_dtypes
import numpy as np
import pytest

import tessera.runtime as rt
from tessera.compiler.native_attention_program import compile_attention_vjp_program
from tessera.compiler.prepared_attention_vjp import clear_prepared, prepared
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark=pytest.mark.hardware_nvidia


def make_program(storage,wide,bias_kind,active=None,threads=128,launch="packed_v1"):
    trace_sq,trace_sk=(1,4) if bias_kind=="query" else (3,1) if bias_kind=="key" else (3,4)
    biased=bias_kind!="none"
    active=active or tuple(reversed(range(3+int(biased))))
    original=source(storage,bias=biased,saved=True,output="f32" if wide else storage,
        dims=(1,2,1,trace_sq,trace_sk,4,3))
    if bias_kind in ("key","dense"):
        physical=f"1x2x{trace_sq}x{1 if bias_kind=='key' else trace_sk}"
        original=original.replace(f"tensor<1x2x1x{trace_sk}xf32>",f"tensor<{physical}xf32>")
    original=original.replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 9, 11>')
    original=original.replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = ['+', '.join(map(str,active))+']')
    program=compile_attention_vjp_program(original,active,compiler=os.environ["TESSERA_OPT"],
        compact_gradients=True,compact_threads=threads,compact_launch=launch,row_correction=True)
    artifact=rt.RuntimeArtifact(graph_ir=original,target_ir=program.pair.backward.target_ir,
        metadata=dict(target="nvidia_sm120",compiler_path="nvidia_sm120_attention_vjp_compiled",
            execution_kind="native_gpu",execution_mode="cuda_runtime",executable=True,
            arg_names=[f"primal_{i}" for i in range(3+int(biased))]+["cotangent","lse_cotangent"],
            program_json=program.to_json(),program_digest=program.program_digest))
    assert json.loads(program.to_json())["program"]["schema"]=="tessera.native_attention_vjp_program.v4"
    return program,rt.RuntimeArtifact.from_json(artifact.to_json()),(trace_sq,trace_sk)


def inputs(storage,wide,bias_kind,sq,sk,seed=975):
    rng=np.random.default_rng(seed)
    dtype=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
    out_dtype=np.dtype(np.float32) if wide else dtype
    q=rng.normal(0,.3,(1,2,sq,4)).astype(dtype)
    k=rng.normal(0,.3,(1,1,sk,4)).astype(dtype)
    v=rng.normal(0,.3,(1,1,sk,3)).astype(dtype)
    do=rng.normal(0,.2,(1,2,sq,3)).astype(out_dtype)
    bias_shape={"query":(1,2,1,sk),"key":(1,2,sq,1),"dense":(1,2,sq,sk)}.get(bias_kind)
    bias=rng.normal(0,.1,bias_shape).astype(np.float32) if bias_shape else None
    row_seed=rng.normal(0,.2,(1,2,sq)).astype(np.float32)
    _,expected=reference(q,k,v,do,bias,row_seed)
    values=(q,k,v)+((bias,) if bias is not None else ())+(do,row_seed)
    return values,expected


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
@pytest.mark.parametrize("bias_kind",["none","query","key","dense"])
def test_public_v4_reuses_images_across_shapes_without_compiler(storage,wide,bias_kind,monkeypatch):
    assert rt._nvidia_device_name()=="sm_120", "owning GPU required"
    program,artifact,trace=make_program(storage,wide,bias_kind)
    pin=program.program_digest
    held=[]
    try:
        rt.backend_capabilities("nvidia_sm120")
        for index,(sq,sk) in enumerate((trace,(3,5),(9,11),(9,1),(3,5))):
            values,expected=inputs(storage,wide,bias_kind,sq,sk,seed=975+index)
            def forbidden(*args,**kwargs):pytest.fail("public bounded replay accessed compiler/subprocess")
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(subprocess,"check_output",forbidden)
                guard.setattr(subprocess,"Popen",forbidden)
                receipt=rt.launch(artifact,values)
            assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
            for actual,role in zip(receipt["output"],program.active,strict=True):compare(actual,expected[role])
            held.append((receipt["output"],tuple(x.copy() for x in receipt["output"])))
            assert artifact.metadata["program_digest"]==pin
        for outputs,prior in held:
            for actual,copy in zip(outputs,prior,strict=True):np.testing.assert_array_equal(actual,copy)
    finally:clear_prepared()


@pytest.mark.parametrize("storage",["f16","bf16"])
def test_bounded_value_only_elides_row_member(storage):
    assert rt._nvidia_device_name()=="sm_120"
    program,artifact,_=make_program(storage,False,"query",active=(2,))
    assert program.correction is None
    try:
        for sq,sk in ((1,4),(9,11)):
            values,expected=inputs(storage,False,"query",sq,sk)
            receipt=rt.launch(artifact,values)
            assert receipt["ok"],receipt
            compare(receipt["output"][0],expected[2])
    finally:clear_prepared()


def test_bounded_invalid_frame_is_rejected_before_native_setup(monkeypatch):
    program,artifact,_=make_program("f16",False,"query")
    values,_=inputs("f16",False,"query",3,5)
    owner=prepared(artifact.metadata)
    def forbidden(*args,**kwargs):pytest.fail("invalid frame reached native setup")
    from tessera.compiler.prepared_attention_vjp import PreparedAttentionVJP
    monkeypatch.setattr(PreparedAttentionVJP,"_prepare",forbidden)
    try:
        malformed=list(values);malformed[0]=np.zeros((1,2,10,4),np.float16)
        with pytest.raises(ValueError,match="envelope"):
            owner.runtime_dimensions(malformed)
        malformed=list(values);malformed[2]=np.zeros((1,1,4,3),np.float16)
        from tessera.compiler.prepared_attention_vjp import execute
        with pytest.raises(ValueError,match="storage"):
            execute(artifact.metadata,malformed)
    finally:clear_prepared()


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("asynchronous",[False,True])
@pytest.mark.parametrize("bias_kind",["none","query","key","dense"])
def test_bounded_resident_capture_retains_native_rows_and_gradients(storage,asynchronous,bias_kind,monkeypatch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from benchmarks.nvidia.benchmark_jit_attention_vjp import download
    assert rt._nvidia_device_name()=="sm_120"
    program,_,trace=make_program(storage,True,bias_kind)
    def forbidden(*args,**kwargs):pytest.fail("bounded capture reentered compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(subprocess,"Popen",forbidden)
    with NvidiaDeviceSession() as session:
        for sq,sk in (trace,(3,5),(9,11),(9,1)):
            values,expected=inputs(storage,True,bias_kind,sq,sk)
            roots=values[:-2]
            resident=tuple(session.upload(value) for value in roots)
            with program.capture(*resident,asynchronous=asynchronous) as frame:
                frame.wait_on(session.stream)
                expected_primal,_=reference(values[0],values[1],values[2],values[-2],
                    values[3] if bias_kind!="none" else None,values[-1])
                for actual,wanted in zip(frame.primal,expected_primal,strict=True):compare(download(session,actual),wanted)
                seeds=tuple(session.upload(value) for value in values[-2:])
                first=frame.backward(seeds)
                frame.wait_on(session.stream)
                held=tuple(download(session,value) for value in first)
                for actual,role in zip(held,program.active,strict=True):compare(actual,expected[role])
                second=frame.backward(seeds)
                frame.wait_on(session.stream)
                for old,snapshot,new in zip(first,held,second,strict=True):
                    np.testing.assert_array_equal(download(session,old),snapshot)
                    np.testing.assert_array_equal(download(session,new),snapshot)
            with pytest.raises(ValueError,match="closed"):
                _=first[0].__cuda_array_interface__


@pytest.mark.parametrize("storage",["f16","bf16"])
def test_bounded_resident_value_only_keeps_producer_elided(storage):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from benchmarks.nvidia.benchmark_jit_attention_vjp import download
    program,_,_=make_program(storage,False,"query",active=(2,))
    values,expected=inputs(storage,False,"query",9,11)
    with NvidiaDeviceSession() as session:
        with program.capture(*(session.upload(x) for x in values[:-2]),asynchronous=True) as frame:
            got=frame.backward(tuple(session.upload(x) for x in values[-2:]))
            frame.wait_on(session.stream)
            compare(download(session,got[0]),expected[2])


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("asynchronous",[False,True])
def test_bounded_row_capture_orders_pending_writes_and_preserves_snapshot(storage,asynchronous):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
    from benchmarks.nvidia.benchmark_jit_attention_vjp import download
    program,_,_=make_program(storage,True,"query")
    values,_=inputs(storage,True,"query",3,5)
    changed=(values[0].astype(np.float32)*.7).astype(values[0].dtype)
    _,expected=reference(changed,values[1],values[2],values[-2],values[3],values[-1])
    with NvidiaDeviceSession() as session:
        roots=tuple(session.upload(value) for value in values[:-2])
        completed=[]
        before=queue_delayed_upload(session,roots[0],changed,session.stream,completed)
        with program.capture(*roots,asynchronous=asynchronous) as frame:
            after=queue_delayed_upload(session,roots[0],np.zeros_like(changed),session.stream,completed)
            assert before is not None and after is not None
            actual=frame.backward(tuple(session.upload(value) for value in values[-2:]))
            frame.wait_on(session.stream)
            for value,role in zip(actual,program.active,strict=True):compare(download(session,value),expected[role])


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("value_only",[False,True])
def test_original_frontend_exports_bounded_row_capture(storage,bias,value_only):
    import tessera as ts
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_public_typed_attention_vjp import function
    from benchmarks.nvidia.benchmark_jit_attention_vjp import download
    kind="query" if bias else "none"
    values,_=inputs(storage,False,kind,1,4)
    roots=dict(zip(("q","k","v","bias") if bias else ("q","k","v"),values[:-2],strict=True))
    wrt=("v",) if value_only else (("bias","q") if bias else ("v","k","q"))
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(function(True,bias,True))
    program=fn.compile_native_attention_vjp(**roots,compiler=os.environ["TESSERA_OPT"],
        sequence_bounds=(9,11),compact_gradients=True,row_correction=True)
    assert json.loads(program.to_json())["program"]["schema"].endswith(".v4")
    with NvidiaDeviceSession() as session:
        for sq,sk in ((1,4),(9,11)):
            values,expected=inputs(storage,False,kind,sq,sk)
            roots=dict(zip(("q","k","v","bias") if bias else ("q","k","v"),values[:-2],strict=True))
            with program.capture(*(session.upload(roots[name]) for name in fn.arg_names)) as frame:
                got=frame.backward(tuple(session.upload(x) for x in values[-2:]))
                for value,role in zip(got,program.active,strict=True):compare(download(session,value),expected[role])


@pytest.mark.parametrize("storage",["f16","bf16"])
def test_fresh_process_bounded_v4_replay_without_compiler(storage,tmp_path):
    import sys
    program,artifact,_=make_program(storage,True,"query")
    values,expected=inputs(storage,True,"query",9,11)
    arrays={f"input_{i}":x.view(np.uint16) if x.dtype.name=="bfloat16" else x for i,x in enumerate(values)}
    arrays.update({f"expected_{i}":expected[role] for i,role in enumerate(program.active)})
    np.savez(tmp_path/"values.npz",**arrays)
    (tmp_path/"artifact.json").write_text(artifact.to_json())
    (tmp_path/"types.json").write_text(json.dumps([x.dtype.name for x in values]))
    script=tmp_path/"restore.py"
    script.write_text('''import json,sys,subprocess
from pathlib import Path
import numpy as np
import ml_dtypes
import tessera.runtime as rt
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare
root=Path(sys.argv[1])
rt.backend_capabilities("nvidia_sm120")
def forbidden(*args,**kwargs):raise AssertionError("fresh bounded replay accessed compiler")
subprocess.run=forbidden;subprocess.check_output=forbidden;subprocess.Popen=forbidden
artifact=rt.RuntimeArtifact.from_json((root/"artifact.json").read_text())
blob=np.load(root/"values.npz")
types=json.loads((root/"types.json").read_text())
values=tuple(blob[f"input_{i}"].view(ml_dtypes.bfloat16) if dtype=="bfloat16" else blob[f"input_{i}"] for i,dtype in enumerate(types))
receipt=rt.launch(artifact,values)
assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
for i,value in enumerate(receipt["output"]):compare(value,blob[f"expected_{i}"])
print("fresh-bounded-v4-passed")
''')
    env=dict(os.environ)
    for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_LLVM_BIN","LLVM_BIN"):
        env[name]="/compiler-unavailable"
    completed=subprocess.run([sys.executable,str(script),str(tmp_path)],text=True,capture_output=True,env=env,timeout=120)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert "fresh-bounded-v4-passed" in completed.stdout


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("launch",["packed_v1","logical_v1"])
def test_bounded_64_thread_compact_geometry_has_public_and_resident_parity(storage,launch):
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from benchmarks.nvidia.benchmark_jit_attention_vjp import download
    program,artifact,_=make_program(storage,True,"query",active=(3,0),threads=64,launch=launch)
    values,expected=inputs(storage,True,"query",9,11)
    try:
        receipt=rt.launch(artifact,values)
        assert receipt["ok"],receipt
        for value,role in zip(receipt["output"],program.active,strict=True):compare(value,expected[role])
        with NvidiaDeviceSession() as session:
            with program.capture(*(session.upload(x) for x in values[:-2]),asynchronous=True) as frame:
                outputs=frame.backward(tuple(session.upload(x) for x in values[-2:]))
                frame.wait_on(session.stream)
                for value,control in zip(outputs,receipt["output"],strict=True):
                    np.testing.assert_array_equal(download(session,value),control)
    finally:clear_prepared()
