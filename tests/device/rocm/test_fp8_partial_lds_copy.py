"""Exact gfx1201 proof for partial thread rounds in native LDS copies."""
import os
import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import (
    BlockScaleShape, blockscale_reference, lower_blockscale, package_blockscale,
)
from tests.device.rocm.test_fp8_blockscale_w8a8 import _inputs, _launch
from tests._support import rocm_isa

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit gfx1201 owning-device gate")]


@pytest.mark.parametrize("shape", [(128, 4096, 128), (200, 2048, 2048),
                                   (200, 2049, 128), (200, 8192, 1024)])
@pytest.mark.parametrize("scale_k", [16, 32])
@pytest.mark.parametrize("output", ["f32", "bf16"])
@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_partial_lds_copy_preserves_all_scale_groups(shape, scale_k, output, prefetch):
    assert rt._rocm_live_arch() == "gfx1201"
    profile = BlockScaleShape(*shape, scale_k, 1, "nk", output)
    # K16 has one instruction panel; its explicit legal performance key is 1.
    # The existing auto default of 2 is a separate selector-policy obligation.
    package = package_blockscale(lower_blockscale(profile),
        scale_group_panels=1 if scale_k == 16 else -1, blockscale_prefetch=prefetch)
    assert package.descriptor.provenance["staging"] == "lds"
    # Deliberately exact operands and power-of-two, nonuniform group scales.
    # Omitted copy vectors or scale groups must not hide behind rounding.
    a, b, sa, sb = _inputs(profile, exact=True, seed=sum(shape) + scale_k)
    want = blockscale_reference(a, b, sa, sb, scale_k=scale_k, scale_n=1)
    dtype = ml_dtypes.bfloat16 if output == "bf16" else np.float32
    got = _launch(package, a, b, sa, sb, profile)
    np.testing.assert_array_equal(got, want.astype(dtype))
    rocm_isa.assert_selected(package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8", what="partial LDS FP8 copy")
