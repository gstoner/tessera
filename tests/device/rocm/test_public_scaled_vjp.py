"""Owning gfx1201 public scale-VJP numerical and compiler-free replay proof."""
import os
import subprocess
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tests.unit.test_native_typed_scaled_vmap import case
from tests.unit.test_native_nested_typed_vmap import nested_case
from tests.support.scaled_product_transpose_oracle import scale_adjoint

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit owning gfx1201 execution required")

@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("wrt", [("sa",), ("sb",), ("sa","sb"), ("sb","sa")])
def test_public_scale_vjp_native_map_and_warm_replay(policy, nk, depth, wrt, monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch() == "gfx1201"
    scalar, mapped, values, expected = case(policy, "fp32", nk)
    axes = mapped._frontend_batch_axes
    owner = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=wrt)(scalar._fn)
    if depth == 0:
        values = tuple(value[0] if axis == 0 else value
                       for value, axis in zip(values, axes, strict=True))
        expected = expected[0]
    else:
        owner = vmap(owner, in_axes=axes)
    if depth == 2:
        owner = vmap(owner, in_axes=axes)
        _, _, _, values, expected = nested_case(policy, "fp32", nk)
    dy = np.random.default_rng(1007).uniform(-.5, .5, expected.shape).astype(np.float32)
    oracle = scale_adjoint(*values, dy, scale_k=128, scale_n=128,
                           batching=policy if depth else None, transpose_b=nk)
    actual = owner.native_backward(*values, out_cotangents=dy)
    for got, name in zip(actual, wrt, strict=True):
        np.testing.assert_allclose(got, oracle[0 if name=="sa" else 1], rtol=4e-5, atol=2e-4)
    receipt = owner.last_backward_execution
    assert receipt["compiler_path"] == "rocm_scaled_vjp_program_compiled"
    assert receipt["frontend_authority"] == "tracer"
    assert receipt["execution_certificate"]["evidence_scope"] == "exact_device"
    assert receipt["physical_attestation"]["device_arch"] == "gfx1201"
    assert owner.native_backward_runtime_artifact().target == "rocm_gfx1201"
    def forbidden(*args, **kwargs):
        raise AssertionError("warm public reverse invoked compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_backward(*values, out_cotangents=dy*-.5)
    for got, want in zip(repeated, actual, strict=True):
        np.testing.assert_allclose(got, want*-.5, rtol=4e-5, atol=2e-4)
