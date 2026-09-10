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
