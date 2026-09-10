"""Native runtime slice indices must follow Python clipping, not trace samples."""
import os
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.trace import trace
from tessera.compiler.source_control_flow import to_native_source_ir


def runtime_slice(x,start,stop,step):
    return x[start:stop:step]


def runtime_columns(x,start,stop,step):
    return x[:,start:stop:step]


def source(fn=runtime_slice,shape=(8,)):
    return to_native_source_ir(trace(fn,np.ones(shape,np.float32),*[np.ones(1,np.int64)]*3,source_control_flow=True))


@pytest.mark.parametrize('bounds',[(1,7,2),(-6,-1,2),(-100,100,3),(7,1,1),(0,8,100),(-(1<<63),(1<<63)-1,1),(7,0,-2),(100,-100,-3),(7,0,-(1<<63))])
def test_runtime_slice_reuses_native_program(bounds):
    if not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native CPU required')
    from tessera import _jit_boundary as jit
    data=np.arange(8,dtype=np.float32)
    expected=data[slice(*bounds)]
    output=np.empty_like(expected)
    handle=jit.compile_module(source())
    try:
        jit.invoke(handle,'source_program',[data,*[np.array([n],np.int64) for n in bounds]],[output])
        np.testing.assert_array_equal(output,expected)
    finally:jit.destroy(handle)


def test_runtime_slice_is_serialized_not_specialized():
    ir=source()
    assert 'tensor<?xf32>' in ir
    assert 'arith.index_cast' in ir
    assert 'runtime slice step cannot be zero' in ir


def test_float_runtime_index_never_silently_converts():
    from tessera.compiler.source_control_flow import SourceControlFlowError
    with pytest.raises(SourceControlFlowError,match='int64'):
        trace(runtime_slice,np.ones(8,np.float32),*[np.ones(1,np.float32)]*3,source_control_flow=True)


def test_runtime_indices_keep_integer_storage():
    traced=trace(runtime_slice,np.ones(8,np.float32),*[np.ones(1,np.int64)]*3,source_control_flow=True)
    assert [dtype for _,_,dtype in traced.args]==['f32','i64','i64','i64']


def test_read_only_slice_cannot_be_written_via_another_state_group():
    from tessera.compiler.source_control_flow import SourceControlFlowError
    def write(state,readonly):
        part=readonly[::2]
        part[:]=part+part
        return state
    with pytest.raises(SourceControlFlowError,match='declared state alias'):
        trace(write,np.ones(4,np.float32),np.ones(4,np.float32),source_control_flow=True,source_state_groups=((0,),))


def test_read_only_slice_survives_local_root_rebinding():
    if not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native CPU required')
    from tessera import _jit_boundary as jit
    def retain(x):
        part=x[::2]
        x=x+x
        return part
    data=np.arange(8,dtype=np.float32)
    ir=to_native_source_ir(trace(retain,data,source_control_flow=True))
    handle=jit.compile_module(ir)
    try:
        output=np.empty(4,np.float32)
        jit.invoke(handle,'source_program',[data],[output])
        np.testing.assert_array_equal(output,data[::2])
    finally:jit.destroy(handle)


def nested_runtime(x,start,stop,step):
    part=x[start:stop:step]
    return part[::-1]


@pytest.mark.parametrize('bounds',[(7,0,-2),(-100,100,3),(7,1,1),(1,7,2)])
def test_dynamic_cpu_jit_allocates_nested_slice_outputs(bounds):
    if not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native CPU required')
    from tessera.compiler.jit import jit
    data=np.arange(8,dtype=np.float32)
    with jit(source_control_flow=True)(nested_runtime) as program:
        actual=program(data,*bounds)
        np.testing.assert_array_equal(actual,data[slice(*bounds)][::-1])
        assert actual.shape==data[slice(*bounds)][::-1].shape
        program(data,0,8,2)
        assert len(program._programs)==1
        with pytest.raises(RuntimeError,match='guard failed'):program(data,0,8,0)


def test_dynamic_cpu_multiple_multidimensional_results():
    if not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native CPU required')
    from tessera.compiler.jit import jit
    def slices(x,start,stop,step):
        part=x[:,start:stop:step]
        return part,part[::-1,::2]
    x=np.arange(24,dtype=np.float32).reshape(3,8)
    with jit(source_control_flow=True)(slices) as program:
        a,b=program(x,7,0,-2)
        np.testing.assert_array_equal(a,x[:,7:0:-2])
        np.testing.assert_array_equal(b,x[:,7:0:-2][::-1,::2])


def test_memref_signature_admission_excludes_nonidentity_layouts():
    from tessera._jit_boundary import _parse_sig_type
    assert _parse_sig_type('memref<8xf32>')==((8,),'f32')
    assert _parse_sig_type('memref<8xf32, strided<[2]>>')[0] is None
    assert _parse_sig_type('memref<8xf32, 1>')[0] is None


def test_dynamic_gather_reverse_emits_scatter():
    from tessera.compiler.native_gpu_storage import _run
    compiler=Path(os.environ.get('TESSERA_OPT','/missing'))
    if not compiler.is_file():pytest.skip('native compiler required')
    ir=to_native_source_ir(trace(runtime_slice,np.ones(8,np.float32),
        *[np.ones(1,np.int64)]*3,source_control_flow=True),autodiff='reverse')
    result=_run(compiler,'--tessera-autodiff-paired',source=ir)
    assert 'gather cotangent shape mismatch' in result
    assert 'tensor.insert' in result


@pytest.mark.parametrize('bounds',[(7,0,-2),(0,8,2),(7,1,1)])
def test_runtime_gather_native_cpu_vjp(bounds):
    import json
    from tessera import _jit_boundary as jit
    from tessera.compiler.native_gpu_storage import _run
    from tessera.compiler.native_persistent_tape import _attribute
    compiler=Path(os.environ.get('TESSERA_OPT','/missing'))
    if not compiler.is_file() or not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native compiler/JIT required')
    data=np.arange(8,dtype=np.float32)
    ir=to_native_source_ir(trace(nested_runtime,data,*[np.ones(1,np.int64)]*3,source_control_flow=True),autodiff='reverse')
    module=_run(compiler,'--tessera-autodiff-paired=export-product=backward',source=ir)
    abi=json.loads(_attribute(module,'tessera.autodiff.product_abi'))
    seed=np.arange(data[slice(*bounds)].size,dtype=np.float32)+1
    # RECOMPUTE_ALL carries the original arguments and the output cotangent.
    inputs=[data,*[np.array([n],np.int64) for n in bounds],seed]
    output=np.zeros_like(data)
    handle=jit.compile_module(module)
    try:jit.invoke(handle,abi['entry'],inputs,[output,*[np.zeros(1,np.int64) for _ in bounds]])
    finally:jit.destroy(handle)
    expected=np.zeros_like(data);expected[slice(*bounds)]=seed[::-1]
    np.testing.assert_array_equal(output,expected)


def test_gather_adjoint_accumulates_repeated_indices():
    import json
    from tessera import _jit_boundary as jit
    from tessera.compiler.native_gpu_storage import _run
    from tessera.compiler.native_persistent_tape import _attribute
    compiler=Path(os.environ.get('TESSERA_OPT','/missing'))
    if not compiler.is_file() or not Path(os.environ.get('TESSERA_JIT_LIB','/missing')).is_file():pytest.skip('native compiler/JIT required')
    source='''module {
      func.func @repeat(%x: tensor<2xf32>) -> tensor<3x2xf32>
          attributes {tessera.autodiff = "reverse"} {
        %r = tensor.generate {
        ^bb0(%i: index, %j: index):
          %v = tensor.extract %x[%j] : tensor<2xf32>
          tensor.yield %v : f32
        } : tensor<3x2xf32>
        return %r : tensor<3x2xf32>
      }
    }'''
    module=_run(compiler,'--tessera-autodiff-paired=export-product=backward',source=source)
    abi=json.loads(_attribute(module,'tessera.autodiff.product_abi'))
    seed=np.arange(6,dtype=np.float32).reshape(3,2)
    output=np.zeros(2,np.float32)
    handle=jit.compile_module(module)
    try:jit.invoke(handle,abi['entry'],[np.ones(2,np.float32),seed],[output])
    finally:jit.destroy(handle)
    np.testing.assert_array_equal(output,seed.sum(axis=0))
    # The gather rule must not silently discard arithmetic in a generator.
    import subprocess
    nonlinear=source.replace('tensor.yield %v : f32',
        '%square = arith.mulf %v, %v : f32\n          tensor.yield %square : f32')
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run(compiler,'--tessera-autodiff-paired',source=nonlinear)
    assert 'unsupported nested-region' in caught.value.stderr
