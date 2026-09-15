"""Execute the production C++ allocator ABI, not a Python emulation."""
from pathlib import Path
import shutil
import subprocess
import pytest
from tessera.compiler.native_exception_producer import NativeExceptionProducer


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('native C++ compiler required')
    root = Path(__file__).resolve().parents[2]
    target = tmp_path_factory.mktemp('exception-native') / 'heap.so'
    subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-pthread','-O2',
                    '-I'+str(root/'src/runtime/include'),
                    str(root/'src/runtime/src/exception_heap.cpp'),'-o',str(target)],check=True)
    return target


def test_native_allocation_cycle_collection_and_stale_generations(library):
    with NativeExceptionProducer(library,capacity=2,payload_capacity=8) as heap:
        a = heap.allocate(1,b'one',root=True)
        b = heap.allocate(2,b'two',cause=a)
        heap.set_edges(a,cause=b,context=b)
        assert heap.read(a) == (1,b'one',b,b)
        assert heap.collect() == 0
        with pytest.raises(MemoryError):
            heap.allocate(3,b'')
        heap.root(a,False)
        assert heap.collect() == 2
        c = heap.allocate(3,b'again',root=True)
        assert c != a and (c & 0xffffffff) == (a & 0xffffffff)
        with pytest.raises(ValueError,match='stale'):
            heap.root(a)
        with pytest.raises(ValueError,match='stale'):
            heap.set_edges(c,cause=b)
        assert heap.read(c)[1] == b'again'
    with pytest.raises(ValueError,match='closed'):
        heap.read(c)


def test_native_partial_collection_reuses_payload_and_preserves_live_roots(library):
    with NativeExceptionProducer(library,capacity=3,payload_capacity=12) as heap:
        a = heap.allocate(1,b'left',root=True)
        b = heap.allocate(2,b'hole',root=True)
        c = heap.allocate(3,b'end!',root=True)
        heap.root(b,False)
        assert heap.collect() == 1
        for _ in range(200):
            transient = heap.allocate(4,b'next',root=True)
            assert heap.read(a)[1] == b'left' and heap.read(c)[1] == b'end!'
            heap.root(transient,False)
            assert heap.collect() == 1
        with pytest.raises(MemoryError):
            heap.allocate(1,b'too large')
        assert heap.read(a)[1] == b'left'


@pytest.mark.parametrize('capacity,bytes_', [(0,4),(65537,4),(2,0),(2,1<<29),(True,4)])
def test_native_heap_rejects_unbounded_configuration(library,capacity,bytes_):
    with pytest.raises(ValueError):
        NativeExceptionProducer(library,capacity=capacity,payload_capacity=bytes_)


def test_source_ir_emits_native_allocation_roots_and_cycle_edges(library):
    import json
    from tessera.compiler.native_exception_ir import NativeExceptionIRProgram
    from tessera.compiler.source_exception_heap import pack_exception_table
    from tessera.compiler.llvm_tools import llvm_bin_dir
    llvm = llvm_bin_dir() or Path('/usr/lib/llvm-23/bin')
    if not (llvm/'clang').exists():
        pytest.skip('LLVM 23 toolchain required')
    table = pack_exception_table([('ValueError',['first']),('RuntimeError',['second'])])
    table['roots'] = [0]
    table['nodes'][0]['cause'] = 1
    table['nodes'][1]['context'] = 0
    source = 'module attributes {tessera.source_state = '+json.dumps(json.dumps({'exception_heap':table}))+'} {}'
    program = NativeExceptionIRProgram(source,llvm_bin=llvm,runtime=library)
    from tessera.compiler.native_source_state import decode_source_exception
    import numpy as np
    error = decode_source_exception({'exception_heap':table},[np.array([1],np.float32)],heap_program=program)
    assert isinstance(error,ValueError) and error.args == ('first',)
    assert isinstance(error.__cause__,RuntimeError)
    assert error.__cause__.__context__ is error
    assert 'func.call @tsr_exception_heap_alloc' in program.mlir
    assert 'call i32 @tsr_exception_heap_root' in program.llvm_ir
    for _ in range(3):
        heap, roots = program.instantiate()
        with heap:
            root = roots[0]
            _, payload, cause, _ = heap.read(root)
            assert json.loads(payload)['args'] == ['first']
            assert heap.read(cause)[3] == root
            assert heap.collect() == 0
            heap.root(root,False)
            assert heap.collect() == 2
    # A native allocation failure after the first node never publishes a heap.
    with pytest.raises(MemoryError):
        program.instantiate(capacity=1)
    heap,roots = program.instantiate()
    program.close()
    assert heap.read(roots[0])[0] == 0
    heap.close()
    with pytest.raises(ValueError,match="closed"):
        program.instantiate()


def test_heap_ir_refuses_undeclared_dynamic_exception_payloads():
    import json
    from tessera.compiler.native_exception_ir import emit_exception_heap_ir
    from tessera.compiler.source_exception_heap import pack_exception_table
    table = pack_exception_table([('ValueError',['@tensor','slot'])])
    source = 'module attributes {tessera.source_state = '+json.dumps(json.dumps({'exception_heap':table}))+'} {}'
    with pytest.raises(ValueError,match='declared dynamic'):
        emit_exception_heap_ir(source)


def test_native_source_program_uses_compiled_heap_on_exception(library):
    import numpy as np
    from tessera import _jit_boundary as jit
    from tessera.compiler.native_source_state import compile_source_state
    from tessera.compiler.llvm_tools import llvm_bin_dir
    llvm = llvm_bin_dir() or Path('/usr/lib/llvm-23/bin')
    if jit._find_dylib() is None or not (llvm/'clang').exists():
        pytest.skip('native CPU JIT and LLVM required')
    def check(x,limit):
        if x > limit:
            raise ValueError('native table')
        return x + x
    x = np.ones(1,np.float32)
    with compile_source_state(check,x,np.zeros_like(x),mutable=(),error_specs=(((1,),'f32'),)) as program:
        program.enable_native_exception_heap(runtime=library,llvm_bin=llvm)
        for _ in range(2):
            with pytest.raises(ValueError,match='native table'):
                program.run(x,np.zeros_like(x))
        np.testing.assert_array_equal(program.run(-x,np.zeros_like(x)),-2*x)


def test_native_dynamic_payload_allocation_is_owned_and_retryable(library):
    import json
    import numpy as np
    from tessera.compiler.native_exception_ir import NativeExceptionIRProgram
    from tessera.compiler.source_exception_heap import pack_exception_table
    from tessera.compiler.llvm_tools import llvm_bin_dir
    llvm = llvm_bin_dir() or Path('/usr/lib/llvm-23/bin')
    if not (llvm/'clang').exists():
        pytest.skip('LLVM 23 toolchain required')
    table = pack_exception_table([('ValueError',['@tensor','slot'])])
    contract = dict(exception_heap=table,error_dynamic=True,error_payload_sites=['slot'])
    source = 'module attributes {tessera.source_state = '+json.dumps(json.dumps(contract))+'} {}'
    with NativeExceptionIRProgram(source,llvm_bin=llvm,runtime=library) as program:
        for shape in ((2,3),(7,), (0,)):
            value = np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            with pytest.raises(MemoryError):
                program.instantiate(payloads={'slot':np.ones(3,np.float32)},payload_capacity=1)
            heap,roots = program.instantiate(payloads={'slot':value})
            expected = value.copy()
            value.fill(-10)
            with heap:
                np.testing.assert_array_equal(np.frombuffer(heap.read(roots[0])[1],dtype=np.float32).reshape(shape),expected)
            error = program.decode(1,contract,[expected],{})
            expected.fill(99)
            np.testing.assert_array_equal(error.args[0],np.arange(np.prod(shape),dtype=np.float32).reshape(shape))
        with pytest.raises(ValueError,match='numeric'):
            program.instantiate(payloads={'slot':np.array([object()])})
        with pytest.raises(ValueError,match='sites'):
            program.instantiate()


def test_native_source_dynamic_exception_payload_uses_heap(library):
    import numpy as np
    from tessera import _jit_boundary as jit
    from tessera.compiler.native_source_state import compile_source_state
    from tessera.compiler.llvm_tools import llvm_bin_dir
    llvm = llvm_bin_dir() or Path('/usr/lib/llvm-23/bin')
    if jit._find_dylib() is None or not (llvm/'clang').exists():
        pytest.skip('native CPU JIT and LLVM required')
    def check(x,limit):
        if x > limit:
            raise ValueError(x + x)
        return x + x
    x = np.ones(1,np.float32)
    with compile_source_state(check,x,np.zeros_like(x),mutable=(),error_specs=(((1,),'f32'),)) as program:
        program.enable_native_exception_heap(runtime=library,llvm_bin=llvm)
        for value in (1.,3.):
            with pytest.raises(ValueError) as caught:
                program.run(x*value,np.zeros_like(x))
            np.testing.assert_array_equal(caught.value.args[0],x*value*2)
