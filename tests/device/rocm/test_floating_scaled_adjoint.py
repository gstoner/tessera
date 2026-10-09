"""Owning gfx1201 native adjoints for continuous scaled-product operands."""
import os

import numpy as np
import pytest

from tessera import runtime
from tessera.compiler.native_scaled_program import package_native_scaled_vjp, PreparedScaledProgram
from tests.unit.test_native_floating_scaled_adjoint import floating_source

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def inputs(ta, tb):
    rng = np.random.default_rng(911)
    a = rng.uniform(-.5, .5, (2, 9)).astype(np.float32)
    b = rng.uniform(-.5, .5, (9, 5)).astype(np.float32)
    sa = rng.uniform(.25, 1.25, (2, 3)).astype(np.float32)
    sb = rng.uniform(.25, 1.25, (3, 2)).astype(np.float32)
    dy = rng.uniform(-.5, .5, (2, 5)).astype(np.float32)
    return (np.ascontiguousarray(a.T if ta else a),
            np.ascontiguousarray(b.T if tb else b), sa, sb, dy)


def oracle(values, ta, tb):
    a, b, sa, sb, dy = [value.astype(np.float64) for value in values]
    a = a.T if ta else a
    b = b.T if tb else b
    da, db = np.zeros_like(a), np.zeros_like(b)
    dsa, dsb = np.zeros_like(sa), np.zeros_like(sb)
    columns = np.arange(5) // 4
    for group in range(3):
        selected = slice(group*4, min(group*4+4, 9))
        weighted = dy * sa[:, group, None] * sb[group, columns][None, :]
        da[:, selected] = weighted @ b[selected, :].T
        db[selected, :] = a[:, selected].T @ weighted
        dot = a[:, selected] @ b[selected, :]
        dsa[:, group] = np.sum(dot * dy * sb[group, columns][None, :], axis=1)
        for column in range(2):
            mask = columns == column
            dsb[group, column] = np.sum(dot[:, mask] * dy[:, mask] * sa[:, group, None])
    return (da.T if ta else da, db.T if tb else db, dsa, dsb)


@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [(0,), (1,), (2,), (3,), (0, 1, 2, 3), (3, 1, 0, 2)])
@pytest.mark.parametrize("placed", [False, True])
def test_native_floating_adjoints_match_oracle_and_retain_outputs(ta, tb, roles, placed):
    assert runtime._rocm_live_arch() == "gfx1201"
    values = inputs(ta, tb)
    package = package_native_scaled_vjp(floating_source(ta, tb, roles, placed))
    seed = np.ascontiguousarray(values[-1].T) if placed else values[-1]
    expected = oracle(values, ta, tb)
    with PreparedScaledProgram(package, (*values[:4], seed),
                               runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as owner:
        generation, _ = owner.invoke()
        actual = owner.read(generation)
        for result, role in zip(actual, roles, strict=True):
            np.testing.assert_allclose(result, expected[role], rtol=4e-5, atol=3e-6)
        retained = tuple(result.copy() for result in actual)
        changed = tuple(np.ascontiguousarray(value * np.float32(1.125)) for value in values)
        owner.update((*changed[:4], np.ascontiguousarray(changed[-1].T) if placed else changed[-1]))
        generation, _ = owner.invoke()
        repeated = owner.read(generation)
        expected = oracle(changed, ta, tb)
        for result, role in zip(repeated, roles, strict=True):
            np.testing.assert_allclose(result, expected[role], rtol=4e-5, atol=3e-6)
        for result, old in zip(actual, retained, strict=True):
            np.testing.assert_array_equal(result, old)


def batch_inputs(policy, ta=False, tb=False):
    from tests.unit.test_native_floating_scaled_adjoint import floating_batch_shapes
    rng = np.random.default_rng(912)
    shapes = (*floating_batch_shapes(policy, ta, tb), (2, 3, 2, 5))
    return tuple(rng.uniform(.25, 1.25, shape).astype(np.float32) if slot in (2, 3)
                 else rng.uniform(-.5, .5, shape).astype(np.float32)
                 for slot, shape in enumerate(shapes))


def batch_oracle(values, ta=False, tb=False):
    # Independently evaluate each logical plane, then sum each result into its
    # original operand's broadcast coordinate. No compiler indexing is reused.
    gradients = [np.zeros_like(value, dtype=np.float64) for value in values[:4]]
    for plane in np.ndindex(values[-1].shape[:-2]):
        coordinates = []
        selected = []
        for value in values[:4]:
            prefix = value.shape[:-2]
            coordinates.append(tuple(0 if extent == 1 else axis
                                     for extent, axis in zip(prefix, plane[len(plane)-len(prefix):], strict=True)))
            selected.append(value[coordinates[-1]])
        expected = oracle((*selected, values[-1][plane]), ta, tb)
        for result, coordinate, gradient in zip(gradients, coordinates, expected, strict=True):
            result[coordinate] += gradient
    return tuple(gradients)


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "shared_lhs", "independent_rhs", "broadcast"])
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("roles", [(0, 1, 2, 3), (3, 1, 0, 2)])
def test_floating_batch_adjoints_reduce_shared_operands(policy, ta, tb, roles):
    from tests.unit.test_native_floating_scaled_adjoint import floating_batch_source
    assert runtime._rocm_live_arch() == "gfx1201"
    values = batch_inputs(policy, ta, tb)
    package = package_native_scaled_vjp(floating_batch_source(policy, ta, tb, roles))
    expected = batch_oracle(values, ta, tb)
    with PreparedScaledProgram(package, values,
                               runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as owner:
        generation, _ = owner.invoke()
        actual = owner.read(generation)
        for result, role in zip(actual, roles, strict=True):
            np.testing.assert_allclose(result, expected[role], rtol=4e-5, atol=3e-6)
        retained = tuple(result.copy() for result in actual)
        changed = tuple(np.ascontiguousarray(value * np.float32(1.125)) for value in values)
        owner.update(changed)
        generation, _ = owner.invoke()
        repeated = owner.read(generation)
        expected = batch_oracle(changed, ta, tb)
        for result, role in zip(repeated, roles, strict=True):
            np.testing.assert_allclose(result, expected[role], rtol=4e-5, atol=3e-6)
        for result, old in zip(actual, retained, strict=True):
            np.testing.assert_array_equal(result, old)
