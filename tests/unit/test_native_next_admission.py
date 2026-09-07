"""Physical scalar residuals, nonlinear budgets and asynchronous ownership."""
from pathlib import Path
import threading
import subprocess
from types import SimpleNamespace
import pytest
import numpy as np
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.native_ann import prepare_native_ann, affine_error_bound, evaluate_native_ann
from tessera.compiler.native_persistent_tape import materialize_persistent_tape
from benchmarks.record_native_tape_extensions import data_while_source, predicate_source
from test_native_ann_composition import source


def relu_source():
    return source().replace('return %o : tensor<3x2xf32>',
        '%r = "tessera.relu"(%o) : (tensor<3x2xf32>) -> tensor<3x2xf32>\nreturn %r : tensor<3x2xf32>')




def compiler():
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    return tool


def test_terminal_relu_native_analytic_admission():
    compiler()
    from tessera import _jit_boundary as jit
    import platform
    if platform.machine().lower() not in ("x86_64", "amd64") or jit._find_dylib() is None:
        pytest.skip("owning x86 native JIT required")
    pair=prepare_native_ann(relu_source(),allow_reassociation=True)
    assert pair.original.count('tessera.relu')==pair.transformed.count('tessera.relu')==1
    linear=prepare_native_ann(source(),allow_reassociation=True)
    assert affine_error_bound(pair,1.0)==affine_error_bound(linear,1.0)
    assert evaluate_native_ann(pair,[np.ones((3,2),np.float32),-np.ones((3,2),np.float32)],
                               input_bound=1.0,absolute_budget=.001).admitted


def test_data_dependent_while_preserves_native_checkpoint_and_scalar_storage():
    pair=materialize_persistent_tape(data_while_source(),compiler=compiler(),
        llvm_bin=Path('/usr/lib/llvm-23/bin'),backend='rocm',chip='gfx1151')
    f,b=pair.validate()
    assert any('i64>' in t for t in f['results'])
    assert 'scf.while' not in pair.forward.arena_ir.split('gpu.module',1)[1]
    assert 'scf.if' in pair.forward.arena_ir.split('gpu.module',1)[1]
    assert f['results'][1:]==b['inputs'][3:]


def test_data_while_requires_actual_conjunctive_capacity():
    text=data_while_source().replace('arith.andi %bounded, %active','arith.ori %bounded, %active')
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        materialize_persistent_tape(text,compiler=compiler(),llvm_bin=Path('/usr/lib/llvm-23/bin'),backend='rocm',chip='gfx1151')


def test_predicate_residual_is_losslessly_boxed():
    text=predicate_source()
    result=run_tessera_opt(compiler(),text,'--tessera-autodiff-paired=box-product-scalars=true export-product=forward')
    assert 'tensor<i8>' in result


@pytest.mark.parametrize('status',[600,700,0])
def test_completion_poll_retains_owners_until_success(status):
    from tessera.compiler.native_gpu_storage import NativeSubmission
    released=[]
    owner=SimpleNamespace(_lock=threading.RLock(),_pending=[],
        _event_query=lambda event:status,_event_sync=lambda event:0,
        _event_destroy=lambda event:released.append(event) or 0)
    def check(value):
        if value: raise RuntimeError('driver failure')
    owner._check=check
    ticket=NativeSubmission(owner,123,[123],(object(),),4)
    owner._pending.append(ticket)
    try:
        if status==700:
            with pytest.raises(RuntimeError):ticket.poll()
        else:
            assert ticket.poll()==(status==0)
        assert bool(ticket._keepalive)==(status!=0)
        assert released==([123] if status==0 else [])
    finally:
        owner._event_query=lambda event:0
        ticket.wait()


def test_gpu_ann_source_replay_rejects_modified_lowered_program():
    from dataclasses import replace
    from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu
    tool=compiler()
    if not Path("/usr/lib/llvm-23/bin/mlir-opt").exists():
        pytest.skip("LLVM 23 tools required")
    logical=prepare_native_ann(relu_source(),allow_reassociation=True)
    pair=materialize_native_ann_gpu(logical,compiler=tool,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend='rocm',chip='gfx1151')
    changed=replace(pair.original,arena_ir=pair.original.arena_ir+'\n')
    changed=replace(changed,binding_digest=changed._digest())
    with pytest.raises(ValueError,match='replay'):
        replace(pair,original=changed).validate()


def test_rank_zero_predicate_materializes_native_device_descriptor():
    pair=materialize_persistent_tape(predicate_source(),compiler=compiler(),llvm_bin=Path('/usr/lib/llvm-23/bin'),backend='rocm',chip='gfx1151')
    f,_=pair.validate()
    assert 'tensor<i8>' in f['results']
    assert 'array<0 x i64>' not in pair.forward.arena_ir


def test_measurement_gate_uses_raw_samples_and_retains_incumbent():
    from copy import deepcopy
    from tessera.compiler.native_ann_gpu import summarize_native_ann_measurements
    base=dict(backend='rocm',chip='gfx1151',pair='p',original='o',transformed='t',
        input_bound=1.0,absolute_budget=.001,bounds=['a','b'],
        timing_domain='warm_package_h2d_dispatch_d2h_host_wall',sources={'a':'b'},
        recorder_sha256='r',numerical_verified=True,samples_ms=[[1.0]*31,[1.0]*31],
        medians_ms=[1.0,1.0],speedup=1.0)
    reports=[dict(deepcopy(base),pid=i+1) for i in range(9)]
    assert not summarize_native_ann_measurements(reports)['performance_eligible']
    edited=deepcopy(reports);edited[0]['samples_ms'][0][0]=True
    with pytest.raises(ValueError,match='positive finite'):summarize_native_ann_measurements(edited)
    edited=deepcopy(reports);edited[0]['speedup']=1.1
    with pytest.raises(ValueError,match='summary'):summarize_native_ann_measurements(edited)
    with pytest.raises(ValueError,match='nine'):summarize_native_ann_measurements(reports[:5])


def test_measurement_gate_rejects_boolean_median_summary():
    from tessera.compiler.native_ann_gpu import summarize_native_ann_measurements
    base=dict(backend='rocm',chip='gfx1151',pair='p',original='o',transformed='t',
        input_bound=1.0,absolute_budget=.001,bounds=['a','b'],
        timing_domain='warm_package_h2d_dispatch_d2h_host_wall',sources={'a':'b'},
        recorder_sha256='r',numerical_verified=True,samples_ms=[[1.0]*31,[1.0]*31],
        medians_ms=[True,True],speedup=1.0)
    with pytest.raises(ValueError,match='summary'):
        summarize_native_ann_measurements([dict(base,pid=i+1) for i in range(9)])


@pytest.mark.parametrize('fail_completion',[False,True])
def test_derivative_release_retains_allocations_if_completion_fails(fail_completion):
    import ctypes as ct
    from tessera.compiler.native_persistent_tape import PersistentDerivativeSubmission
    freed=[]
    buffer=SimpleNamespace(pointer=ct.c_void_p(100))
    other=SimpleNamespace(pointer=ct.c_void_p(200))
    def check(status):
        if status:raise RuntimeError('completion failed')
    frame=SimpleNamespace(_lock=threading.RLock(),_ready=lambda:None,
        check=check,sync=lambda:700 if fail_completion else 0,
        free=lambda ptr:freed.append(ptr.value) or 0,buffers=[other,buffer],_submissions=[])
    result=PersistentDerivativeSubmission(frame,SimpleNamespace(wait=lambda:None),(buffer,),1)
    frame._submissions.append(result)
    if fail_completion:
        with pytest.raises(RuntimeError,match='completion'):result.release()
        assert frame.buffers==[other,buffer] and not freed
        assert frame._submissions==[result]
    else:
        result.release();result.release()
        assert frame.buffers==[other] and freed==[100] and not frame._submissions
        with pytest.raises(ValueError,match='released'):result.wait()


def test_fixed_nine_run_bound_tolerates_one_outlier_but_not_two():
    from copy import deepcopy
    from tessera.compiler.native_ann_gpu import summarize_native_ann_measurements
    base=dict(backend='rocm',chip='gfx1151',pair='p',original='o',transformed='t',
        input_bound=1.0,absolute_budget=.001,bounds=['a','b'],
        timing_domain='warm_package_h2d_dispatch_d2h_host_wall',sources={'a':'b'},
        recorder_sha256='r',numerical_verified=True,samples_ms=[[1.1]*31,[1.0]*31],
        medians_ms=[1.1,1.0],speedup=1.1)
    reports=[dict(deepcopy(base),pid=i+1) for i in range(9)]
    for index in (0,1):
        reports[index].update(samples_ms=[[1.1]*31,[10.0]*31],medians_ms=[1.1,10.0],speedup=1.1/10.0)
        result=summarize_native_ann_measurements(reports)
        assert result['performance_eligible']==(index==0)
        assert not result['production_promoted']
