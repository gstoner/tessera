"""The explicit JIT ABI rejects incompatible tensor bindings before dispatch."""
from dataclasses import asdict
import inspect
from types import SimpleNamespace
import pytest
from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage
from tessera.compiler.native_gpu_tensor import NativeTensorCall, TensorSpec, IndexSpec


def binding():
    p = NativeGPUStoragePackage('nvidia', 'sm_120', 'entry', 'size', ('pointer', 'index'),
                                'module {}', b'image', b'host', 'c' * 64, 'd' * 64, '')
    p = NativeGPUStoragePackage(**{**asdict(p), 'binding_digest': p._digest()})
    return NativeTensorCall(p, inspect.signature(lambda output, n: None),
        (TensorSpec('output', 'float32', ('n',), True), IndexSpec('n')),
        grid=(1, 1, 1), block=('n', 1, 1))


def tensor(**changes):
    interface = dict(version=3, shape=(32,), typestr='<f4', data=(4096, False), strides=None)
    interface.update(changes)
    return SimpleNamespace(__cuda_array_interface__=interface)


def test_resolves_tensor_shape_and_native_geometry():
    output = tensor()
    raw, allocations, grid, block, outputs = binding().prepare(output, 32)
    assert (raw, allocations, grid, block, outputs) == ((4096, 32), [(4096, 128)], (1, 1, 1), (32, 1, 1), (output,))


@pytest.mark.parametrize('changes', [dict(shape=(64,)), dict(typestr='<f8'),
    dict(data=(4096, True)), dict(data=(4097, False)), dict(strides=(8,)), dict(shape=(True,))])
def test_rejects_incompatible_tensor_before_loading_package(changes):
    with pytest.raises(ValueError):
        binding()(tensor(**changes), 32)


@pytest.mark.parametrize('n', [True, 0, -1, 1 << 31, 32.0])
def test_rejects_non_native_index(n):
    with pytest.raises(ValueError, match='bounds'):
        binding()(tensor(), n)


def test_rejects_host_array():
    import numpy as np
    with pytest.raises(ValueError, match='resident'):
        binding()(np.zeros(32, dtype=np.float32), 32)


def test_jit_binding_bypasses_graph_and_exposes_native_identity(monkeypatch):
    from tessera.compiler.jit import JitFn
    from tessera.compiler.graph_ir import GraphIRModule
    from tessera.compiler.effects import Effect
    from tessera.compiler.constraints import ConstraintSolver
    def body(output, n):
        raise AssertionError('Python body must not execute')
    jit = JitFn(body, GraphIRModule(), Effect.memory, ConstraintSolver())
    original = binding()
    jit.bind_native_storage(original.package, original.specs, grid=original.grid, block=original.block)
    monkeypatch.setattr(jit, '_establish_tracer_authority', lambda *a: pytest.fail('unexpected Graph trace'))
    monkeypatch.setattr(NativeTensorCall, '__call__', lambda self, *a, **kw: 'native result')
    assert jit(tensor(), 32) == 'native result'
    assert jit.execution_kind == 'native_gpu'
    assert jit.runtime_artifact().metadata['package_digest'] == original.package.binding_digest
    jit.differentiation_request = object()
    with pytest.raises(ValueError, match='paired differentiation'):
        jit(tensor(), 32)


@pytest.mark.parametrize('owner,length,message', [(1, 128, 'another device'), (0, 64, 'exceeds')])
def test_rejects_wrong_device_or_short_allocation_before_launch(monkeypatch, owner, length, message):
    import ctypes as ct
    class Function:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *args):
            return self.fn(*args)
    def current(output):
        ct.cast(output, ct.POINTER(ct.c_int))[0] = 0
        return 0
    def attribute(output, kind, pointer):
        ct.cast(output, ct.POINTER(ct.c_int))[0] = owner
        return 0
    def extent(base, size, pointer):
        ct.cast(base, ct.POINTER(ct.c_void_p))[0] = 4096
        ct.cast(size, ct.POINTER(ct.c_size_t))[0] = length
        return 0
    native = SimpleNamespace(_driver=SimpleNamespace(cuCtxGetDevice=Function(current),
        cuPointerGetAttribute=Function(attribute), cuMemGetAddressRange_v2=Function(extent)),
        _check=lambda code: None, launch=lambda *a, **kw: pytest.fail('unexpected dispatch'))
    monkeypatch.setattr(NativeGPUStoragePackage, 'bind', lambda self: native)
    with pytest.raises(ValueError, match=message):
        binding()(tensor(), 32)
