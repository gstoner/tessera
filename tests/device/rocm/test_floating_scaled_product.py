"""Exact gfx1201 proof for continuous native primal and four-operand JVP."""
import itertools
import os

import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram, package_native_scaled_primal, package_native_scaled_jvp)
from tests.unit.test_native_floating_scaled_product import product_source
from tests.device.rocm.test_floating_scaled_adjoint import inputs, batch_inputs

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def primal(values, ta, tb):
    a, b, sa, sb = [value.astype(np.float64) for value in values]
    if ta:
        a = a.swapaxes(-1, -2)
    if tb:
        b = b.swapaxes(-1, -2)
    prefix = np.broadcast_shapes(*(value.shape[:-2] for value in values))
    result = np.zeros((*prefix, a.shape[-2], b.shape[-1]), dtype=np.float64)
    for group in range(sa.shape[-1]):
        span = slice(group*4, min((group+1)*4, a.shape[-1]))
        dot = np.matmul(a[..., :, span], b[..., span, :])
        columns = np.take(sb[..., group, :], np.arange(b.shape[-1])//4, axis=-1)
        result += dot * sa[..., :, group, None] * columns[..., None, :]
    return result


def reference(values, ta, tb, jvp):
    expected = [primal(values[:4], ta, tb)]
    if jvp:
        tangent = np.zeros_like(expected[0])
        for index in range(4):
            operands = list(values[:4])
            operands[index] = values[4+index]
            tangent += primal(operands, ta, tb)
        expected.append(tangent)
    return expected


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("policy", [None, "shared_rhs_rows", "shared_lhs", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("jvp", [False, True])
def test_continuous_product_numerics_and_warm_lifetime(ta, tb, policy, jvp):
    assert runtime._rocm_live_arch() == "gfx1201"
    values = list((inputs(ta, tb) if policy is None else batch_inputs(policy, ta, tb))[:4])
    if jvp:
        rng = np.random.default_rng(938)
        values.extend(rng.uniform(-.5, .5, value.shape).astype(np.float32)
                      for value in tuple(values))
    source = product_source(ta, tb, policy, jvp)
    package = (package_native_scaled_jvp(source) if jvp else package_native_scaled_primal(source))
    with PreparedScaledProgram(package, values,
                               runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as owner:
        generation, _ = owner.invoke()
        actual = owner.read(generation)
        for got, want in zip(actual, reference(values, ta, tb, jvp), strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        retained = tuple(value.copy() for value in actual)
        changed = [np.ascontiguousarray(value*np.float32(-.625)) for value in values]
        owner.update(changed)
        generation, _ = owner.invoke()
        for got, want in zip(owner.read(generation), reference(changed, ta, tb, jvp), strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        for got, old in zip(actual, retained, strict=True):
            np.testing.assert_array_equal(got, old)
