"""Native orientation of logical codes and K16 scales on owning SM120."""
import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.skipif(
    not nvidia_cuda_host_ready(), reason="owning SM120 required")]


def oriented_product(ta, tb, batching=None):
    def product(a, b, sa, sb):
        return ts.ops.scaled_matmul(a, b, sa, sb,
            physical_contract="nvidia_sm120_nvfp4_blockscale_v1",
            numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
            scale_layout={"granularity": "block", "block": [1, 16], "format": "ue4m3"},
            transposeA=ta, transposeB=tb, batching=batching)
    return product


def oriented_inputs(mode, ta, tb, rows, n, k, *, batch=3, seed=None):
    rng = np.random.default_rng(120615 + k if seed is None else seed)
    a_shape = (rows, k) if mode in {"rank_two", "shared_lhs"} else (batch, rows, k)
    b_shape = (batch, k, n) if mode in {"independent_rhs", "shared_lhs"} else (k, n)
    ac = rng.integers(0, 16, a_shape, dtype=np.uint8)
    bc = rng.integers(0, 16, b_shape, dtype=np.uint8)
    sk = k // 16 + (k % 16 != 0)
    choices = np.array([0x30, 0x33, 0x38, 0x3a, 0x40], np.uint8)
    sa = choices[rng.integers(0, len(choices), (*a_shape[:-1], sk))]
    sb_shape = (batch, sk, n) if mode in {"independent_rhs", "shared_lhs"} else (sk, n)
    sb = choices[rng.integers(0, len(choices), sb_shape)]
    def expected():
        av = _decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=-1)[..., :k]
        bs = np.repeat(_decode_ue4m3(sb), 16, axis=-2)
        bs = bs[..., :k, :]
        return av.astype(np.float64) @ (_decode_e2m1(bc) * bs).astype(np.float64)
    logical_a = np.ascontiguousarray(np.swapaxes(ac, -2, -1) if ta else ac)
    logical_b = np.ascontiguousarray(np.swapaxes(bc, -2, -1) if tb else bc)
    axis_a = logical_a.ndim - (2 if ta else 1)
    axis_b = logical_b.ndim - (1 if tb else 2)
    a = NVFP4Tensor(_pack_nvfp4(logical_a, axis_a), logical_a.shape, axis_a)
    b = NVFP4Tensor(_pack_nvfp4(logical_b, axis_b), logical_b.shape, axis_b)
    physical_sa = np.ascontiguousarray(np.swapaxes(sa, -2, -1) if ta else sa)
    physical_sb = np.ascontiguousarray(np.swapaxes(sb, -2, -1) if tb else sb)
    return (a, b, physical_sa, physical_sb), expected()


CASES = [(mode, ta, tb) for mode in ("rank_two", "independent_rhs", "shared_rhs_rows", "shared_lhs")
         for ta in (False, True) for tb in (False, True)]


@pytest.mark.parametrize("mode,ta,tb", CASES)
@pytest.mark.parametrize("rows,n,k", [(16, 8, 64), (17, 19, 129)])
def test_native_nvfp4_transpose_matches_oracle(mode, ta, tb, rows, n, k, monkeypatch):
    values, expected = oriented_inputs(mode, ta, tb, rows, n, k)
    call = ts.jit(oriented_product(ta, tb, None if mode == "rank_two" else mode), target="nvidia_sm120")
    result = call(*values)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-3)
    descriptor = call._cached_artifact.launch_descriptor
    assert descriptor.provenance["transposeA"] is ta
    assert descriptor.provenance["transposeB"] is tb
    assert call._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
    assert "tile.matmul_kernel" in call.compile_bundle.tile.text
    assert "mma.sync.aligned.m16n8k64" in call._cached_artifact.native_image.payload.decode()
    import importlib
    compiler = importlib.import_module("tessera.compiler.canonical_compile")
    def forbidden(*args, **kwargs):
        raise AssertionError("warm orientation must reuse the package without eager arithmetic")
    monkeypatch.setattr(compiler, "canonical_compile", forbidden)
    monkeypatch.setattr(call, "_fn", forbidden)
    np.testing.assert_allclose(call(*values), expected, rtol=0, atol=2e-3)


@pytest.mark.parametrize("mode", ["independent_rhs", "shared_rhs_rows", "shared_lhs"])
@pytest.mark.parametrize("ta,tb", [(True, False), (False, True), (True, True)])
def test_native_vmap_keeps_operand_orientation(mode, ta, tb):
    values, expected = oriented_inputs(mode, ta, tb, 7, 5, 31)
    axes = (None, 0, None, 0) if mode == "shared_lhs" else (0 if mode == "independent_rhs" else (0, None, 0, None))
    call = vmap(ts.jit(oriented_product(ta, tb), target="nvidia_sm120"), in_axes=axes)
    np.testing.assert_allclose(call(*values), expected, rtol=0, atol=2e-3)
    assert call._cached_artifact.launch_descriptor.provenance["transposeA"] is ta
    assert call._cached_artifact.launch_descriptor.provenance["transposeB"] is tb
