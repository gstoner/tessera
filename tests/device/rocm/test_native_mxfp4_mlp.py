"""Owning execution of the native MXFP4-weight seven-member MLP.

ROCM-MXFP4-NATIVE-MLP-20261102 (gfx1201), ROCM-MXFP4-NATIVE-MLP-GFX1151-20261102
(gfx1151). The weight expansion reads E8M0 K32 scales; everything after it is
the accepted NVFP4 program's scaled FP8 MLP on the live owning architecture.
Each architecture's result is its own proof.
"""
import ml_dtypes
import numpy as np
import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp, mxfp4_inputs
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain
from tessera.compiler.nvfp4_expansion import mxfp4_expansion_policy, reference_nvfp4_expand
from tessera.compiler.rocm_nvfp4_expand_native import execute_nvfp4_expansion
from tests.device.rocm.test_native_quantized_mlp_bounded_rows import owning_target, provider

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.compiler_rocm, pytest.mark.compiler_route]




@pytest.mark.parametrize("shape", [(1, 32, 32, 1), (3, 32, 64, 19), (128, 128, 4096, 256)])
def test_public_mxfp4_mlp_executes_and_certifies(shape):
    target = owning_target()
    lib = provider()
    m, h, k, n = shape
    fn = ts.jit(target=target, native_required=True)(make_mlp(m, h, k, n, source="mxfp4"))
    roots = mxfp4_inputs(m, h, k, n)
    before = [x.copy() for x in roots]
    result = np.asarray(fn(*roots))
    artifact = fn.runtime_artifact()
    assert fn.execution_kind == "native_gpu"
    assert artifact.metadata["compiler_path"] == "canonical_rocm_quantized_mlp_program"
    native = fn._rocm_quantized_product_last_program.native
    assert native.gate.expansion.mxfp4 and native.down.expansion.mxfp4
    assert result.dtype == np.dtype(ml_dtypes.bfloat16) and result.shape == (m, n)
    assert np.isfinite(result.astype(np.float32)).all()
    certificate = assert_compiled_chain(native, roots, result, lib)
    assert certificate["resident_staged_bitwise_equal"] and certificate["propagated_bound_passed"]
    for a, b in zip(roots, before, strict=True):
        np.testing.assert_array_equal(a, b)
    # A second call with new weights reuses the program and still certifies.
    changed = mxfp4_inputs(m, h, k, n, seed=902)
    again = np.asarray(fn(*changed))
    assert_compiled_chain(fn._rocm_quantized_product_last_program.native, changed, again, lib)


@pytest.mark.parametrize("n,k,offsets", [(64, 64, [0, 32, 64]), (256, 4096, [0, 128, 256])])
def test_mxfp4_expansion_image_is_bitwise_the_reference(n, k, offsets):
    target = owning_target()
    fn = ts.jit(target=target, native_required=True)(
        make_mlp(1, offsets[1], k, 32, source="mxfp4"))
    roots = mxfp4_inputs(1, offsets[1], k, 32)
    fn(*roots)
    package = fn._rocm_quantized_product_last_program.native.gate.expansion
    rng = np.random.default_rng(n + k)
    codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    # Every non-reserved exponent, including 0 (2^-127), within f32 range.
    exponents = rng.integers(0, 255, size=(n, k // 32), dtype=np.uint8)
    exponents.flat[:3] = (0, 1, 254)
    globals_ = np.array([1.0, 1.0])
    values, plane = execute_nvfp4_expansion(package, codes, exponents, globals_)
    expected_values, expected_plane = reference_nvfp4_expand(
        codes, exponents, globals_, row_offsets=offsets, numeric_policy=mxfp4_expansion_policy())
    np.testing.assert_array_equal(values.view(np.uint8), expected_values.view(np.uint8))
    np.testing.assert_array_equal(plane.view(np.uint32), expected_plane.view(np.uint32))


def test_mxfp4_mlp_refuses_reserved_exponent_before_launch():
    target = owning_target()
    fn = ts.jit(target=target, native_required=True)(make_mlp(3, 32, 64, 19, source="mxfp4"))
    roots = mxfp4_inputs(3, 32, 64, 19)
    fn(*roots)
    roots[2] = roots[2].copy()
    roots[2][0, 0] = 255
    with pytest.raises(Exception, match="255"):
        fn(*roots)
