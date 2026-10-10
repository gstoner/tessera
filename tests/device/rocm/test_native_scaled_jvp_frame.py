"""Owning gfx1201 JVP preparation is native for mapped and composed frames."""
import os
import subprocess

import numpy as np
import pytest

from tests.device.rocm.test_public_scaled_jvp import oracle
from tests.unit.test_composed_scaled_jvp import case
from tests.unit.test_native_scaled_map_axes import axis_case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required"
)


def padded(value):
    storage = np.empty((*value.shape[:-1], value.shape[-1]*2), value.dtype)
    view = storage[..., ::2]
    view[...] = value
    return view


def forbid_python_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("warm native JVP invoked Python compaction, compiler or reference")
    monkeypatch.setattr(np, "ascontiguousarray", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference, "reference_typed_scaled_matmul", forbidden)


@pytest.mark.parametrize("nk", [False, True])
@pytest.mark.parametrize("profile", ["single", "nested", "cartesian"])
def test_nonleading_jvp_native_frame_without_python_packing(nk, profile, monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    _, owner, values, expected = axis_case(
        "fp32", nk, mode="forward", nested=profile != "single", cartesian=profile == "cartesian"
    )
    seeds = tuple(padded(value * factor) for value, factor in
                  zip(values[2:], (.125, -.0625), strict=True))
    actual = owner.native_jvp(*values, tangents=seeds)
    for got, wanted in zip(actual, (expected, expected*.0625), strict=True):
        np.testing.assert_allclose(got, wanted, rtol=4e-5, atol=2e-5)
    assert owner.last_jvp_execution["host_preparation"] == "native_checked_view_pack"
    saved = tuple(value.copy() for value in actual)
    forbid_python_preparation(monkeypatch)
    repeated = owner.native_jvp(*values, tangents=tuple(-seed for seed in seeds))
    np.testing.assert_allclose(repeated[0], expected, rtol=4e-5, atol=2e-5)
    np.testing.assert_allclose(repeated[1], -expected*.0625, rtol=4e-5, atol=2e-5)
    for original, previous in zip(actual, saved, strict=True):
        np.testing.assert_array_equal(original, previous)


@pytest.mark.parametrize("shape", [(17, 19, 256), (3, 5, 37)])
@pytest.mark.parametrize("wrt", [("sa0",), ("sa1", "sb1"), ("sb1", "sa0", "sb0", "sa1")])
def test_composed_jvp_strided_primals_seeds_and_reordered_roles(shape, wrt, monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == "gfx1201"
    owner, values, seeds = case(shape, wrt)
    values, seeds = tuple(map(padded, values)), tuple(map(padded, seeds))
    def expected(inputs, tangents):
        a, b, sa0, sb0, sa1, sb1 = inputs
        mapping = dict(zip(wrt, tangents, strict=True))
        first = oracle(a, b, sa0, sb0, mapping.get("sa0", np.zeros_like(sa0)),
                       mapping.get("sb0", np.zeros_like(sb0)))
        second = oracle(a, b, sa1, sb1, mapping.get("sa1", np.zeros_like(sa1)),
                        mapping.get("sb1", np.zeros_like(sb1)))
        return tuple(left+right for left, right in zip(first, second, strict=True))
    wanted = expected(values, seeds)
    actual = owner.native_jvp(*values, tangents=seeds)
    for got, target in zip(actual, wanted, strict=True):
        np.testing.assert_allclose(got, target, rtol=4e-5, atol=3e-5)
    assert owner.last_jvp_execution["host_preparation"] == "native_checked_view_pack"
    saved = tuple(value.copy() for value in actual)
    changed = list(values)
    changed[2] = padded(values[2]*.75)
    changed_seeds = tuple(-seed for seed in seeds)
    repeated_wanted = expected(changed, changed_seeds)
    forbid_python_preparation(monkeypatch)
    repeated = owner.native_jvp(*changed, tangents=changed_seeds)
    for got, target in zip(repeated, repeated_wanted, strict=True):
        np.testing.assert_allclose(got, target, rtol=4e-5, atol=3e-5)
    for original, previous in zip(actual, saved, strict=True):
        np.testing.assert_array_equal(original, previous)
