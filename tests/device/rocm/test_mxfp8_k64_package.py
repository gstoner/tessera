"""Explicit native K64 slab: independent K32 scale joins and checked K."""
import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from tests.device.rocm.test_mxfp8_checked_package import _arguments, _artifact
from tests.device.rocm.test_mxfp8_scheduled_scale import pytestmark


@pytest.mark.parametrize("mnk", [(17, 19, 64), (200, 257, 192), (256, 1024, 1536)])
@pytest.mark.parametrize("output", ["f32", "bf16"])
@pytest.mark.parametrize("project", [False, True])
def test_k64_slab_preserves_independent_scale_groups(mnk, output, project):
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(*mnk, 32, 1, "nk", output)
    candidate = compile_mxfp8(shape, schedule_policy="lds_k64",
                              project_image_identity=project)
    control = compile_mxfp8(shape, schedule_policy="lds",
                            project_image_identity=project)
    assert candidate.descriptor.provenance["macro_k"] == 64
    assert candidate.image.payload != control.image.payload
    args, expected = _arguments(shape)
    want = expected.astype(ml_dtypes.bfloat16 if output == "bf16" else np.float32)
    for package in (control, candidate):
        receipt = rt.launch(_artifact(package), args)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        np.testing.assert_array_equal(args["buffers"]["o"], want)


def test_k64_runtime_image_reuse_keeps_physical_slab_recipe():
    shapes = [BlockScaleShape(17, 19, k, 32, 1, "nk") for k in (64, 128, 192)]
    packages = [compile_mxfp8(s, schedule_policy="lds_k64") for s in shapes]
    assert len({p.image.image_digest for p in packages}) == 1
    assert len({p.descriptor.descriptor_digest for p in packages}) == 3


def test_k64_runtime_rejects_partial_slab_before_hip(monkeypatch):
    package = compile_mxfp8(BlockScaleShape(17, 19, 64, 32, 1, "nk"),
                            schedule_policy="lds_k64")
    args, _ = _arguments(BlockScaleShape(17, 19, 96, 32, 1, "nk"))
    def forbidden(*args, **kwargs):
        pytest.fail("partial physical slab reached HIP")
    monkeypatch.setattr(rt, "_load_hip_for_launch", forbidden)
    with pytest.raises(RuntimeError, match="MXFP8 descriptor"):
        rt._submit_rocm_gfx1151_native(package.image, package.descriptor,
                                      args["buffers"], args["scalars"], None)


@pytest.mark.parametrize("k", [1024, 1088, 1536, 1984, 2048])
def test_automatic_narrow_slab_route(k):
    shape = BlockScaleShape(200, 2049, k, 32, 1, "nk", "bf16")
    package = compile_mxfp8(shape)
    explicit = compile_mxfp8(shape, schedule_policy="lds_k64")
    assert package.image.payload == explicit.image.payload
    args, expected = _arguments(shape)
    receipt = rt.launch(_artifact(package), args)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_array_equal(args["buffers"]["o"],
                                  expected.astype(ml_dtypes.bfloat16))


@pytest.mark.parametrize("mnk", [(200, 1024, 128), (200, 8192, 1024),
                                (256, 4096, 5120), (512, 4096, 1024)])
def test_automatic_route_retains_other_profiles(mnk):
    shape = BlockScaleShape(*mnk, 32, 1, "nk")
    package = compile_mxfp8(shape)
    assert package.descriptor.provenance["macro_k"] == 32


@pytest.mark.parametrize("mnk", [(200,1025,3136), (400,512,3584),
                                (256,1024,2560), (200,2048,5120)])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_automatic_long_k_route_keeps_independent_groups_and_ragged_edges(mnk, output):
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(*mnk,32,1,"nk",output)
    automatic = compile_mxfp8(shape)
    explicit = compile_mxfp8(shape,schedule_policy="lds_k64")
    assert automatic.image.payload == explicit.image.payload
    assert automatic.descriptor.provenance["macro_k"] == 64
    args,expected = _arguments(shape)
    receipt = rt.launch(_artifact(automatic),args)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    want = expected.astype(ml_dtypes.bfloat16 if output == "bf16" else np.float32)
    np.testing.assert_array_equal(args["buffers"]["o"],want)
