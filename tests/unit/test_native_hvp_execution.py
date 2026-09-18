"""Exact public compiler HVP execution; NumPy supplies analytic oracles only."""
import numpy as np
import pytest
import tessera as ts
from tessera import _jit_boundary as boundary
from tessera.compiler.jit import TesseraJitError


@ts.jit(target='cpu', autodiff='reverse')
def cubic(x):
    return ts.ops.mul(ts.ops.mul(x, x), x)


@ts.jit(target='cpu', autodiff='reverse', wrt=('y',))
def coupled(x, y):
    return ts.ops.mul(x, ts.ops.mul(y, y))


def test_composed_exact_hvp_preserves_curvature_at_zero():
    dtype = np.float32
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    x = np.array([0, -2, .25, 3], dtype=dtype)
    v = np.array([3, .5, -2, 1], dtype=dtype)
    cot = np.array([2, -1, .5, 3], dtype=dtype)
    gradients, products = cubic.native_hvp(x, tangents=v, out_cotangents=cot)
    np.testing.assert_allclose(gradients[0], 3*x*x*cot)
    np.testing.assert_allclose(products[0], 6*x*v*cot)
    assert cubic.last_hvp_execution['execution_kind'] == 'native_cpu'


def test_subset_wrt_zeros_inactive_directions_and_projects_results():
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    x = np.array([2, 3, -1], np.float32)
    y = np.array([0, 2, -3], np.float32)
    v = np.array([1, -.5, 2], np.float32)
    gradients, products = coupled.native_hvp(x, y, tangents=v, out_cotangents=np.ones_like(y))
    assert len(gradients) == len(products) == 1
    np.testing.assert_allclose(gradients[0], 2*x*y)
    np.testing.assert_allclose(products[0], 2*x*v)
    assert products[0][0] != 0


def test_hvp_rejects_wrong_direction_before_compilation():
    with pytest.raises(TesseraJitError, match='shape and dtype'):
        cubic.native_hvp(np.ones(3, np.float32), tangents=np.ones(4, np.float32),
                         out_cotangents=np.ones(3, np.float32))


@ts.jit(target='cpu', autodiff='reverse')
def repeated_square(x):
    return ts.control.fori_loop(0, 2, lambda i, c: ts.ops.mul(c, c), x)


def test_control_loop_hvp():
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    x = np.array([0, -.5, 1, 2], np.float32)
    g, h = repeated_square.native_hvp(x, tangents=np.ones_like(x), out_cotangents=np.ones_like(x))
    np.testing.assert_allclose(g[0], 4*x**3)
    np.testing.assert_allclose(h[0], 12*x*x)


@ts.jit(target='cpu', autodiff='reverse', wrt=('x',))
def nested_branch(flag, x):
    return ts.control.cond(flag,
        lambda: ts.control.fori_loop(0, 2, lambda i, c: ts.ops.mul(c, c), x),
        lambda: ts.ops.mul(x, x))


def test_nested_branch_hvp_uses_runtime_predicate():
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    x = np.array([[0, -.5], [1, 2]], np.float32)
    for flag in (1, -1, 1):
        g, h = nested_branch.native_hvp(np.array([flag], np.float32), x,
            tangents=np.ones_like(x), out_cotangents=np.ones_like(x))
        np.testing.assert_allclose(g[0], 4*x**3 if flag > 0 else 2*x)
        np.testing.assert_allclose(h[0], 12*x*x if flag > 0 else np.full_like(x, 2))


def test_bad_cotangent_never_enters_native_code(monkeypatch):
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    def forbidden(*args):
        pytest.fail('invalid cotangent entered native code')
    monkeypatch.setattr(boundary, 'invoke', forbidden)
    with pytest.raises(TesseraJitError, match='shape mismatch'):
        cubic.native_hvp(np.ones(3, np.float32), tangents=np.ones(3, np.float32),
                         out_cotangents=np.ones(4, np.float32))
    assert cubic.last_hvp_execution is None


def test_saved_nested_product_hvp_captures_residual_tangents():
    if boundary._find_dylib() is None:
        pytest.skip('native compiler execution library unavailable')
    import subprocess
    from benchmarks.record_persistent_split_tape import source
    generated = subprocess.run([boundary._find_tessera_opt(), '--tessera-autodiff-hvp-pipeline'],
        input=source(), text=True, capture_output=True, check=True).stdout
    assert 'state_tape' in generated
    x = np.array([0, -.5, 1, 2], np.float32)
    w = np.array([1, .75, -.5, 1.25], np.float32)
    vx = np.array([1, -.5, .25, 0], np.float32)
    vw = np.array([.5, .25, -1, .5], np.float32)
    outputs = [np.empty_like(x) for _ in range(4)]
    handle = boundary.compile_module(generated)
    try:
        boundary.invoke(handle, 'nested__hvp', [x, w, np.ones_like(x), vx, vw], outputs)
    finally:
        boundary.destroy(handle)
    for actual, expected in zip(outputs, [w**6, 6*x*w**5, 6*w**5*vw,
                                           6*w**5*vx + 30*x*w**4*vw]):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.fixture
def gpu_device_factory():
    import ctypes as ct
    from benchmarks.record_device_ring_protocol import Device
    devices = []
    def create(backend):
        device = Device(backend)
        devices.append(device)
        return device
    yield create
    for device in reversed(devices):
        if device.cuda:
            destroy = device.lib.cuCtxDestroy_v2
            destroy.argtypes, destroy.restype = [ct.c_void_p], ct.c_int
            device.check(destroy(device.context))


@pytest.mark.parametrize('backend,chip,gate', [
    ('rocm', 'gfx1201', 'TESSERA_GFX1201_DEVICE_PROOF'),
    ('nvidia', 'sm_120', 'TESSERA_SM120_DEVICE_PROOF'),
])
@pytest.mark.parametrize('shape', [(4,), (2, 2)])
@pytest.mark.parametrize('function,power', [(cubic, 3), (repeated_square, 4)])
def test_gpu_hvp_executes_compiler_product(function, power, shape, backend, chip, gate, gpu_device_factory):
    import os
    if os.environ.get(gate) != '1':
        pytest.skip(f'requires owning {chip} proof')
    import ctypes as ct
    import inspect
    from pathlib import Path
    from types import SimpleNamespace
    from tessera.compiler.native_storage_contract import generate_tensor_binding, tensor_contract_specs, read_tensor_contract
    from tessera import runtime
    if backend == 'rocm':
        assert runtime._rocm_live_arch() == chip
    device = gpu_device_factory(backend)
    if device.cuda:
        ordinal, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
        current = device.lib.cuCtxGetDevice
        current.argtypes, current.restype = [ct.POINTER(ct.c_int)], ct.c_int
        capability = device.lib.cuDeviceComputeCapability
        capability.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int]
        capability.restype = ct.c_int
        device.check(current(ct.byref(ordinal)))
        device.check(capability(ct.byref(major), ct.byref(minor), ordinal))
        assert f'sm_{major.value}{minor.value}' == chip
    # The compiler and LLVM tools come from the explicit env pins when set,
    # else from the repo's own discovery; a host with neither skips by name
    # (until 2026-09-18 this raised a bare KeyError inside the sweep).
    from tessera.compiler.llvm_tools import llvm_bin_dir
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    compiler = os.environ.get('TESSERA_OPT') or find_tessera_opt()
    llvm_bin = os.environ.get('TESSERA_LLVM_BIN') or llvm_bin_dir()
    if not compiler or not llvm_bin:
        pytest.skip('requires tessera-opt and the matched LLVM tools (TESSERA_OPT / TESSERA_LLVM_BIN or the build tree)')
    x = np.array([0, -.5, 1, 2], np.float32).reshape(shape)
    package = function.compile_native_hvp(x, compiler=str(compiler),
        llvm_bin=Path(llvm_bin), backend=backend, chip=chip)
    specs = tensor_contract_specs(read_tensor_contract(package))
    signature = inspect.Signature([inspect.Parameter(s.name, inspect.Parameter.POSITIONAL_ONLY) for s in specs])
    binding = generate_tensor_binding(package, signature)
    arrays = [x, np.ones_like(x), np.ones_like(x), np.zeros_like(x), np.zeros_like(x)]
    pointers, tensors = [], []
    try:
        for a in arrays:
            p = ct.c_void_p()
            device.check(device.alloc(ct.byref(p), a.nbytes))
            pointers.append(p)
            device.check(device.htod(p, a.ctypes.data, a.nbytes) if device.cuda
                         else device.copy(p, a.ctypes.data, a.nbytes, 1))
            tensors.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,
                shape=a.shape, typestr=a.dtype.str, data=(p.value, False))))
        binding(*tensors, 1)
        for a,p in zip(arrays[-2:], pointers[-2:]):
            device.check(device.dtoh(a.ctypes.data, p, a.nbytes) if device.cuda
                         else device.copy(a.ctypes.data, p, a.nbytes, 2))
        np.testing.assert_allclose(arrays[-2], power*x**(power-1))
        np.testing.assert_allclose(arrays[-1], power*(power-1)*x**(power-2))
    finally:
        binding.close()
        for p in pointers:
            device.check(device.free(p))


@ts.jit(target='cpu', autodiff='reverse', wrt=('x',))
def bounded_while(x, limit):
    return ts.control.while_loop(lambda c: ts.ops.sub(limit, c),
        lambda c: ts.ops.mul(c, c), x, max_steps=5)


def test_effectful_while_refuses_instead_of_using_empty_ast_candidate():
    if boundary._find_tessera_opt() is None:
        pytest.skip('native compiler unavailable')
    x = np.array([1.5], np.float32)
    with pytest.raises(TesseraJitError, match='AUTODIFF_NESTED_REGION'):
        bounded_while.native_hvp(x, np.array([8], np.float32),
            tangents=np.ones_like(x), out_cotangents=np.ones_like(x))


def test_hvp_export_rejects_unrelated_suffix_function():
    import subprocess
    compiler = boundary._find_tessera_opt()
    if compiler is None:
        pytest.skip('native compiler unavailable')
    source = 'module { func.func @unrelated__hvp(%x: tensor<4xf32>) -> tensor<4xf32> { return %x : tensor<4xf32> } }'
    result = subprocess.run([compiler, '--tessera-autodiff-forward=export-hvp=true'],
                            input=source, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'requires an HVP preparation pass' in result.stderr
