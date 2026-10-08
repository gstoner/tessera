"""Owning gfx1201 mixed map execution and compiler-free warm replay."""
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_native_mixed_scaled_maps import mixed_case
pytestmark = pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
                                reason="owning gfx1201 required")


@pytest.mark.parametrize("fmt", ["fp32", "e8m0"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("cartesian", [False, True])
def test_mixed_nested_public_native_execution(fmt, nk, cartesian, monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    scalar, inner, owner, values, expected = mixed_case(fmt, nk, cartesian)
    scalar_result, inner_result = scalar.compile_result, inner.compile_result
    np.testing.assert_allclose(owner(*values), expected, rtol=4e-5, atol=2e-5)
    assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    assert "tile.scaled_matmul_kernel" in owner.compile_result.tile_ir
    assert scalar.compile_result is scalar_result and inner.compile_result is inner_result
    def forbidden(*args, **kwargs):
        raise AssertionError("warm mixed map called compiler or numerical reference")
    monkeypatch.setattr(subprocess, "run", forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference, "reference_typed_scaled_matmul", forbidden)
    changed = list(values)
    changed[2] = values[2] * .5 if fmt == "fp32" else values[2] - np.uint8(1)
    np.testing.assert_allclose(owner(*changed), expected * .5, rtol=4e-5, atol=2e-5)

@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("cartesian", [False, True])
def test_mixed_nested_scale_jvp_seed_views(nk, cartesian, monkeypatch):
    _, _, owner, values, expected = mixed_case("fp32", nk, cartesian, mode="forward")
    seeds = (values[2] * .125, values[3] * -.0625)
    primal, tangent = owner.native_jvp(*values, tangents=seeds)
    np.testing.assert_allclose(primal, expected, rtol=4e-5, atol=2e-5)
    np.testing.assert_allclose(tangent, expected * .0625, rtol=4e-5, atol=2e-5)
    assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    def forbidden(*args, **kwargs):
        raise AssertionError("warm mixed JVP invoked compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_jvp(*values, tangents=tuple(-seed for seed in seeds))
    np.testing.assert_allclose(repeated[1], expected * -.0625, rtol=4e-5, atol=2e-5)


@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("cartesian", [False, True])
def test_mixed_nested_scale_vjp_unbroadcasts_to_original_views(nk, cartesian, monkeypatch):
    from tests.unit.test_native_mixed_scaled_maps import mixed_oracle
    _, _, owner, values, expected = mixed_case("fp32", nk, cartesian, mode="reverse")
    cot = np.random.default_rng(10822).uniform(-1, 1, size=expected.shape).astype(np.float32)
    wanted = []
    for role in (2, 3):
        gradient = np.empty_like(values[role], dtype=np.float64)
        for index in np.ndindex(gradient.shape):
            plus, minus = list(values), list(values)
            plus[role] = values[role].astype(np.float64)
            minus[role] = values[role].astype(np.float64)
            plus[role][index] += .0001
            minus[role][index] -= .0001
            gradient[index] = np.sum(cot * (
                mixed_oracle(plus, owner._frontend_batch_policies, "fp32", nk) -
                mixed_oracle(minus, owner._frontend_batch_policies, "fp32", nk))) / .0002
        wanted.append(gradient)
    actual = owner.native_backward(*values, out_cotangents=cot)
    for role, got, want in zip((2, 3), actual, wanted, strict=True):
        assert got.shape == values[role].shape
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=2e-5)
    assert owner.last_backward_execution["execution_kind"] == "native_gpu"
    def forbidden(*args, **kwargs):
        raise AssertionError("warm mixed VJP invoked compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    for got, want in zip(owner.native_backward(*values, out_cotangents=-cot), wanted, strict=True):
        np.testing.assert_allclose(got, -want, rtol=4e-5, atol=2e-5)

@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
def test_existing_nested_k129_primal_partial_groups(policy, nk):
    from tests.unit.test_native_nested_typed_vmap import nested_case
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    _, _, owner, values, expected = nested_case(policy, "fp32", nk, shape=(2,3,7,19,129))
    np.testing.assert_allclose(owner(*values), expected, rtol=4e-5, atol=2e-5)
    assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
