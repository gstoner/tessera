"""Native JVP against independent transpose action; not native VJP proof."""
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_native_nested_typed_vmap import nested_case
from tests.support.scaled_product_transpose_oracle import scale_adjoint

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="owning gfx1201 required")


@pytest.mark.parametrize("policy", ["shared_rhs_rows", "independent_rhs", "shared_lhs"])
@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("shape", [(2, 3, 7, 19, 256), (2, 2, 17, 129, 1536)])
def test_native_scale_jvp_transpose_duality(policy, nk, shape, monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch() == "gfx1201"
    _, _, owner, values, _ = nested_case(policy, "fp32", nk, shape, jvp=True)
    a, b, sa, sb = values
    rng = np.random.default_rng(709)
    seeds = tuple(rng.uniform(-.15, .15, value.shape).astype(np.float32)
                  for value in (sa, sb))
    dy = rng.uniform(-.5, .5, shape[:-1]).astype(np.float64)
    # shape is B0,B1,M,N,K: cotangent retains B0,B1,M,N.
    adjoints = scale_adjoint(a, b, sa, sb, dy, scale_k=128, scale_n=128,
                            batching=policy, transpose_b=nk)
    _, tangent = owner.native_jvp(*values, tangents=seeds)
    assert owner.last_jvp_execution["execution_kind"] == "native_gpu"
    def compare(tangent, sign):
        lhs = np.sum(tangent.astype(np.float64) * dy)
        rhs = sign * sum(np.sum(seed.astype(np.float64) * gradient)
                         for seed, gradient in zip(seeds, adjoints, strict=True))
        np.testing.assert_allclose(lhs, rhs, rtol=4e-5, atol=2e-3)
    compare(tangent, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("warm native duality invoked compiler/reference")
    monkeypatch.setattr(subprocess, "run", forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference, "reference_typed_scaled_matmul", forbidden)
    _, changed = owner.native_jvp(*values, tangents=tuple(-x for x in seeds))
    compare(changed, -1)
