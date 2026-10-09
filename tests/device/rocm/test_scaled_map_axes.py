"""Exact gfx1201 proof for non-leading scaled-map primal, tangent and adjoint."""
import os
import subprocess

import numpy as np
import pytest

from tests.unit.test_native_scaled_map_axes import axis_case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required"
)


def axis_oracle(values, policies, fmt, nk):
    """Independent scalar mapping and float64 K-group arithmetic."""
    if policies:
        axes = policies[0]
        size = next(value.shape[axis] for value, axis in zip(values, axes, strict=True)
                    if axis is not None)
        return np.stack([axis_oracle(
            tuple(np.take(value, plane, axis=axis) if axis is not None else value
                  for value, axis in zip(values, axes, strict=True)),
            policies[1:], fmt, nk) for plane in range(size)])
    a, b, sa, sb = values
    b = b.T if nk else b
    sk, sn = (128, 128) if fmt == "fp32" else (32, 1)
    sa = sa.astype(np.float64) if fmt == "fp32" else np.exp2(sa.astype(np.float64)-127)
    sb = sb.astype(np.float64) if fmt == "fp32" else np.exp2(sb.astype(np.float64)-127)
    result = np.zeros((a.shape[0], b.shape[1]), np.float64)
    for group in range((a.shape[-1] + sk - 1) // sk):
        result += (a[:, group*sk:(group+1)*sk].astype(np.float64) @
                   b[group*sk:(group+1)*sk].astype(np.float64)) * (
                       sa[:, group, None] * sb[group, np.arange(b.shape[1])//sn])
    return result


def forbid_warm_recovery(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("warm execution called a compiler or a numerical reference")
    monkeypatch.setattr(subprocess, "run", forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference, "reference_typed_scaled_matmul", forbidden)


@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("profile", ["single", "nested", "cartesian"])
def test_nonleading_native_primal_and_changed_input(fmt, nk, profile, monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    scalar, owner, values, expected = axis_case(fmt, nk, nested=profile != "single",
                                               cartesian=profile == "cartesian")
    before = scalar.compile_result
    np.testing.assert_allclose(axis_oracle(values, owner._frontend_batch_policies, fmt, nk),
                               expected, rtol=4e-5, atol=2e-5)
    result = owner(*values)
    saved = result.copy()
    np.testing.assert_allclose(result, expected, rtol=4e-5, atol=2e-5)
    assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    assert "tile.scaled_matmul_kernel" in owner.compile_result.tile_ir
    assert scalar.compile_result is before
    changed = list(values)
    changed[2] = values[2] * .5 if fmt == "fp32" else values[2] - np.uint8(1)
    forbid_warm_recovery(monkeypatch)
    np.testing.assert_allclose(owner(*changed), expected * .5, rtol=4e-5, atol=2e-5)
    np.testing.assert_array_equal(result, saved)


@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("profile", ["single", "nested", "cartesian"])
def test_nonleading_native_jvp_seed_axes(nk, profile, monkeypatch):
    _, owner, values, expected = axis_case("fp32", nk, mode="forward",
                                           nested=profile != "single",
                                           cartesian=profile == "cartesian")
    seeds = (values[2] * .125, values[3] * -.0625)
    primal, tangent = owner.native_jvp(*values, tangents=seeds)
    np.testing.assert_allclose(primal, expected, rtol=4e-5, atol=2e-5)
    np.testing.assert_allclose(tangent, expected * .0625, rtol=4e-5, atol=2e-5)
    assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    forbid_warm_recovery(monkeypatch)
    np.testing.assert_allclose(owner.native_jvp(*values, tangents=tuple(-seed for seed in seeds))[1],
                               expected * -.0625, rtol=4e-5, atol=2e-5)


@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("profile", ["single", "nested", "cartesian"])
def test_nonleading_native_vjp_restores_axis_order(nk, profile, monkeypatch):
    _, owner, values, expected = axis_case("fp32", nk, mode="reverse",
                                           nested=profile != "single",
                                           cartesian=profile == "cartesian")
    cot = np.random.default_rng(1808).uniform(-1, 1, expected.shape).astype(np.float32)
    wanted = []
    for role in (2, 3):
        gradient = np.empty(values[role].shape, np.float64)
        for index in np.ndindex(gradient.shape):
            plus, minus = list(values), list(values)
            plus[role] = values[role].astype(np.float64)
            minus[role] = values[role].astype(np.float64)
            plus[role][index] += .0001
            minus[role][index] -= .0001
            gradient[index] = np.sum(cot * (
                axis_oracle(plus, owner._frontend_batch_policies, "fp32", nk) -
                axis_oracle(minus, owner._frontend_batch_policies, "fp32", nk))) / .0002
        wanted.append(gradient)
    actual = owner.native_backward(*values, out_cotangents=cot)
    for role, got, want in zip((2, 3), actual, wanted, strict=True):
        assert got.shape == values[role].shape
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=2e-5)
    assert owner.last_backward_execution["execution_kind"] == "native_gpu"
    forbid_warm_recovery(monkeypatch)
    for got, want in zip(owner.native_backward(*values, out_cotangents=-cot), wanted, strict=True):
        np.testing.assert_allclose(got, -want, rtol=4e-5, atol=2e-5)
