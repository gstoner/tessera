"""Owning gfx1201 public reverse maps retain native inverse-cotangent storage."""
import json
import os
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tessera.autodiff import vmap
from tests.unit.test_native_typed_scaled_vmap import case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def scale_oracle(values, axes, seed):
    a, b, sa, sb = values
    result = [np.zeros_like(sa, dtype=np.float64), np.zeros_like(sb, dtype=np.float64)]
    n = seed.shape[-1]
    columns = np.arange(n) // 128
    for plane in range(seed.shape[0]):
        aa, bb, ss, tt = [value[plane] if axis is not None else value
                          for value, axis in zip(values, axes, strict=True)]
        da = result[0][plane] if axes[2] is not None else result[0]
        db = result[1][plane] if axes[3] is not None else result[1]
        for group in range((aa.shape[-1] + 127) // 128):
            dot = aa[:, group*128:(group+1)*128].astype(np.float64) @ bb[group*128:(group+1)*128].astype(np.float64)
            weighted = dot * seed[plane].astype(np.float64)
            da[:, group] += np.sum(weighted * tt[group, columns][None, :], axis=1)
            for column_group in range(db.shape[-1]):
                selected = columns == column_group
                db[group, column_group] += np.sum(weighted[:, selected] * ss[:, group, None])
    return tuple(result)


@pytest.mark.parametrize("policy", ["independent_rhs", "shared_lhs", "shared_rhs_rows"])
@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("wrt", [("sa",), ("sb",), ("sa", "sb"), ("sb", "sa")])
def test_public_reverse_map_executes_inverse_seed_and_requested_gradients(policy, axis, wrt, monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    scalar, leading, values, primal = case(policy, "fp32", False, (2, 7, 19, 256))
    reverse = ts.jit(target="rocm_gfx1201", autodiff="reverse", wrt=wrt)(scalar._fn)
    owner = vmap(reverse, in_axes=leading._frontend_batch_axes, out_axes=axis)
    seed = np.random.default_rng(906).uniform(-.5, .5, primal.shape).astype(np.float32)
    dy = np.ascontiguousarray(np.moveaxis(seed, 0, axis))
    references = scale_oracle(values, leading._frontend_batch_axes, seed)
    expected = tuple(references[0 if name == "sa" else 1] for name in wrt)
    actual = owner.native_backward(*values, out_cotangents=dy)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=4e-5, atol=3e-5)
    assert owner.last_backward_execution["execution_kind"] == "native_gpu"
    assert owner.last_backward_execution["evidence_target"] == "rocm_gfx1201"
    program = json.loads(owner._native_backward_artifact.program_json)
    step = program["steps"][0]
    assert step["operation"] == "tessera.transpose"
    assert step["cotangent_source"] == 4
    assert all(member["inputs"][-1] == step["output"] for member in program["steps"][1:])
    retained = tuple(result.copy() for result in actual)
    def forbidden(*args, **kwargs):
        raise AssertionError("warm reverse map invoked a compiler")
    monkeypatch.setattr(subprocess, "run", forbidden)
    repeated = owner.native_backward(*values, out_cotangents=dy * np.float32(-.5))
    for result, reference in zip(repeated, expected, strict=True):
        np.testing.assert_allclose(result, reference * -.5, rtol=4e-5, atol=3e-5)
    for result, reference in zip(actual, retained, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("n", [19, 129])
def test_nested_reverse_map_restores_nonuniform_seed_before_scale_reduction(n):
    assert runtime._rocm_live_arch() == "gfx1201"
    scalar, _, flat, primal = case("independent_rhs", "fp32", False, (6, 7, n, 256))
    reverse = ts.jit(target="rocm_gfx1201", autodiff="reverse",
                     wrt=("sa", "sb"))(scalar._fn)
    owner = vmap(vmap(reverse, in_axes=0, out_axes=1), in_axes=0, out_axes=2)
    values = tuple(value.reshape(2, 3, *value.shape[1:]) for value in flat)
    seed = np.random.default_rng(907).uniform(-.5, .5, primal.shape).astype(np.float32)
    dy = np.ascontiguousarray(seed.reshape(2, 3, 7, n).transpose(2, 1, 0, 3))
    references = scale_oracle(flat, (0, 0, 0, 0), seed)
    actual = owner.native_backward(*values, out_cotangents=dy)
    for result, reference in zip(actual, references, strict=True):
        np.testing.assert_allclose(result, reference.reshape(result.shape), rtol=4e-5, atol=3e-5)
    assert owner.last_backward_execution["execution_kind"] == "native_gpu"
    program = json.loads(owner._native_backward_artifact.program_json)
    assert program["steps"][0]["permutation"] == [2, 1, 0, 3]
    assert program["buffers"][5]["shape"] == [2, 3, 7, n]
