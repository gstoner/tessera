"""Public bounded capture uses native axis identities rather than trace sizes."""
import numpy as np
import pytest
import tessera as ts

from tessera.compiler.native_tensor_jit import input_signature, shape_signature, matches_shape_signature
from tests.unit.test_native_tensor_shape_package import shape_package  # noqa: F401

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def norm_lhs(a, b):
    return ts.ops.matmul(ts.ops.rmsnorm(a), b, output_dtype="fp32")


def rhs_add(a, b, r):
    return ts.ops.matmul(ts.ops.relu(a), ts.ops.add(b, r), output_dtype="fp32")


def chain(a, b, r):
    return ts.ops.matmul(ts.ops.relu(ts.ops.add(a, r)), b, output_dtype="fp32")


def two_matmuls(a, w, b):
    hidden = ts.ops.matmul(a, w, output_dtype="fp32")
    return ts.ops.matmul(ts.ops.relu(hidden), b, output_dtype="fp32")


@pytest.mark.parametrize("function", [norm_lhs, rhs_add, chain, two_matmuls])
def test_public_compile_projects_independent_bounds_in_native_ir(function):
    if function is two_matmuls:
        args = (np.ones((17, 32), np.float16), np.ones((32, 32), np.float16), np.ones((32, 11), np.float16))
    else:
        args = (np.ones((17, 32), np.float16), np.ones((32, 11), np.float16))
        if function in (rhs_add, chain):
            args += (np.ones_like(args[1] if function is rhs_add else args[0]),)
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65, "N": 47, "K": 64})(function)
    program = fn.compile_native_tensor_program(*args)
    assert program.frame["schema"] == "tessera.native.tensor_program_frame.v3"
    assert program.shape_companion is not None
    assert "tessera.native.tensor_program_shape_bounds_request" in program.source_ir


def test_variable_compact_columns_share_compiler_derived_cache_key(shape_package):
    bounds = {"M": 65, "N": 47, "K": 64}
    def signature(m, n, k):
        return input_signature((np.empty((m, k), np.float16), np.empty((k, n), np.float16)))
    key = shape_signature(signature(17, 17, 17), shape_package.frame, bounds)
    for m, n, k in ((1, 1, 1), (65, 47, 64), (5, 31, 7)):
        assert matches_shape_signature(signature(m, n, k), key, bounds)
    assert not matches_shape_signature(signature(17, 11, 23), key, {"M": 64, "N": 47, "K": 64})
    wrong_dtype = input_signature((np.empty((17, 23), np.float32), np.empty((23, 11), np.float16)))
    assert not matches_shape_signature(wrong_dtype, key, bounds)


def test_fixed_contraction_axis_remains_in_compiler_cache_key():
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": 65, "N": 47})(norm_lhs)
    program = fn.compile_native_tensor_program(np.ones((17, 32), np.float16), np.ones((32, 11), np.float16))
    assert program.frame["shape_bindings"][0][1] == -1
    original = input_signature((np.empty((17, 32), np.float16), np.empty((32, 11), np.float16)))
    key = shape_signature(original, program.frame, {"M": 65, "N": 47})
    changed = input_signature((np.empty((5, 32), np.float16), np.empty((32, 31), np.float16)))
    assert matches_shape_signature(changed, key, {"M": 65, "N": 47})
    different_k = input_signature((np.empty((5, 23), np.float16), np.empty((23, 31), np.float16)))
    assert not matches_shape_signature(different_k, key, {"M": 65, "N": 47})



def test_partial_v3_provider_leaves_route_unclaimed_before_tracing(monkeypatch):
    from types import SimpleNamespace
    from tessera import runtime
    from tessera.compiler.jit import _JIT_FALLBACK

    fn = ts.jit(target="nvidia_sm120")(norm_lhs)
    provider = SimpleNamespace(tessera_nvidia_native_frame_prepare_v3=object())
    monkeypatch.setattr(runtime, "_load_nvidia_ptx_launch", lambda: provider)
    def forbidden(*args, **kwargs):
        pytest.fail("incomplete v3 provider traced frontend source")
    monkeypatch.setattr(fn, "_traced_autodiff_module", forbidden)
    args = (np.ones((17, 32), np.float16), np.ones((32, 11), np.float16))
    assert fn._try_native_tensor_program_call(args, {}, shape_bounds={"M": 65, "N": 47, "K": 64}) is _JIT_FALLBACK
    assert not fn._native_tensor_program_cache
