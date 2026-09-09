"""Checked persistent GPU guard ABI and scoped consumer integration."""
from types import SimpleNamespace
import inspect
import threading
import pytest
from tessera.compiler.scheduled_matmul import run_tessera_opt, find_tessera_opt
from benchmarks.record_dynamic_cfg_storage import dynamic_source


def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('native compiler required')
    return tool


@pytest.mark.parametrize('backend', ['nvidia', 'rocm'])
def test_checked_guard_retains_failed_status_and_gates_suffix(backend):
    source = dynamic_source().replace('%value = arith.constant 7.0 : f32',
        '%value = arith.constant 7.0 : f32\n%ok = arith.constant false\ncf.assert %ok, "exhausted"')
    text = run_tessera_opt(compiler(), source,
        '--tessera-native-tape-to-gpu=backend='+backend+' status-buffer=true')
    assert 'tessera.autodiff.gpu_status = "guard-v1"' in text
    assert 'cf.assert' not in text
    assert 'scf.if' in text and 'else' in text
    assert 'memref<1xi64>' in text


def test_nested_assertions_propagate_status():
    source = dynamic_source().replace('%n = arith.addi %i, %one : index',
        '%ok = arith.constant true\ncf.assert %ok, "nested"\n%n = arith.addi %i, %one : index')
    text = run_tessera_opt(compiler(),source,'--tessera-native-tape-to-gpu=status-buffer=true')
    assert 'cf.assert' not in text
    assert 'scf.if' in text and 'memref<1xi64>' in text


def test_synchronous_binding_refuses_scoped_reader():
    from tessera.compiler.native_gpu_tensor import NativeTensorCall
    call=SimpleNamespace(_lock=threading.RLock(),signature=inspect.Signature([
        inspect.Parameter('x',inspect.Parameter.POSITIONAL_ONLY)]))
    with pytest.raises(ValueError,match='explicit stream'):
        NativeTensorCall.__call__(call,SimpleNamespace(_tessera_reader_stream=21))


def test_checked_frame_refuses_invalid_stream_before_allocating():
    from tessera.compiler.native_persistent_tape import PersistentTapeFrame
    frame=SimpleNamespace(_lock=threading.RLock(),_ready=lambda:None,_checked_status=True)
    with pytest.raises(ValueError,match='non-null stream'):
        PersistentTapeFrame.backward_async(frame,0,tracked=True)


from benchmarks.record_product_status import function_cfg_source


@pytest.mark.parametrize('role',['forward','backward'])
def test_function_body_cfg_uses_native_recovery(role):
    text=run_tessera_opt(compiler(),function_cfg_source(),
        '--tessera-autodiff-paired=box-product-scalars=true export-product='+role)
    body=text.split('func.func',1)[1]
    assert 'cf.switch' not in body
    assert 'bounded native CFG exhausted' in body


def test_function_body_cfg_without_bound_refuses():
    with pytest.raises(RuntimeError):
        run_tessera_opt(compiler(),function_cfg_source().replace('max_steps = 2','max_steps = 0'),
            '--tessera-autodiff-paired=export-product=forward')


@pytest.mark.parametrize('backend', ['nvidia', 'rocm'])
def test_two_statuses_are_serialized_and_consumed(backend):
    text=run_tessera_opt(compiler(), dynamic_source(),
        '--tessera-native-tape-to-gpu=backend='+backend+' status-buffer=true input-status=true input-status-count=2')
    assert 'tessera.autodiff.input_status_count = 2 : i64' in text
    assert 'cf.assert' not in text
    from tessera.compiler.native_persistent_tape import _input_status_count, _status_specs
    assert _input_status_count(SimpleNamespace(arena_ir=text)) == 2
    assert [s.name for s in _status_specs(2)] == ['dependency_status','dependency_status_1']


@pytest.mark.parametrize('count', [0, 9])
def test_unsupported_status_count_refuses(count):
    with pytest.raises(RuntimeError):
        run_tessera_opt(compiler(), dynamic_source(),
            f'--tessera-native-tape-to-gpu=status-buffer=true input-status=true input-status-count={count}')


@pytest.mark.parametrize('count',[3,4,8])
def test_wider_status_count_has_a_native_consumer(count):
    text=run_tessera_opt(compiler(),dynamic_source(),
        f'--tessera-native-tape-to-gpu=status-buffer=true input-status=true input-status-count={count}')
    from tessera.compiler.native_persistent_tape import _input_status_count
    assert _input_status_count(SimpleNamespace(arena_ir=text))==count
    assert 'cf.assert' not in text
