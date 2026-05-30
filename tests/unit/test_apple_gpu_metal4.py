"""Apple GPU — Metal 4 capability probe (M0) + MTLTensor round-trip (M1).

Metal 4 is an *additive* lane alongside MPSGraph (which still runs on the
classic command model). These tests lock the live capability probe — which
actually creates the Metal 4 objects on-device — and the native ``MTLTensor``
typed-resource round-trip. Everything degrades cleanly to ``available=False`` /
a numpy copy off Tahoe / non-Darwin, so the contract is checked everywhere. See
docs/apple_gpu_metal4_adoption.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R


def test_metal4_probe_reports_consistent_caps():
    caps = R.apple_gpu_metal4_caps()
    for k in ("available", "command_queue", "command_allocator", "compiler",
              "tensor", "msl4"):
        assert isinstance(caps[k], bool), k
    assert isinstance(caps["bits"], int)
    # If the Metal 4 stack is up at all, the command queue + MTLTensor are the
    # core bits the lane depends on.
    if caps["available"]:
        assert caps["command_queue"]
        assert caps["tensor"]


def test_metal4_probe_matches_bits():
    caps = R.apple_gpu_metal4_caps()
    expect = (caps["command_queue"] * 1 + caps["command_allocator"] * 2
              + caps["compiler"] * 4 + caps["tensor"] * 8 + caps["msl4"] * 16)
    assert caps["bits"] == expect
    assert caps["available"] == (caps["bits"] != 0)


@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_metal4_tensor_roundtrip_is_exact(dtype):
    if dtype == "bf16":
        ml = pytest.importorskip("ml_dtypes")
        np_dtype = ml.bfloat16
    else:
        np_dtype = {"f32": np.float32, "f16": np.float16}[dtype]
    rng = np.random.default_rng(0)
    a = rng.standard_normal(19).astype(np_dtype)
    rt = R.apple_gpu_metal4_tensor_roundtrip(a, np)
    # Round-trip through the native MTLTensor (or numpy fallback) is a storage
    # copy — bit-exact, dtype preserved.
    assert rt.dtype == a.dtype
    assert np.array_equal(rt.astype(np.float32), a.astype(np.float32))


def test_metal4_tensor_roundtrip_preserves_shape_size():
    a = np.arange(33, dtype=np.float32) * 0.5
    rt = R.apple_gpu_metal4_tensor_roundtrip(a, np)
    assert rt.shape == a.shape
    np.testing.assert_array_equal(rt, a)
