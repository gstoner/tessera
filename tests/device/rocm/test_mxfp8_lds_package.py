"""Exact gfx1201 E8M0 LDS package, lifetime and runtime image proof."""
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from tests.device.rocm.test_mxfp8_checked_package import _arguments, _artifact
from tests.device.rocm.test_mxfp8_scheduled_scale import _reference, pytestmark
from tests._support import rocm_isa


@pytest.mark.parametrize("mnk", [(128, 4096, 1024), (200, 2048, 2048),
    (200, 2049, 1536), (256, 4096, 1536), (257, 4097, 1024), (512, 4096, 1024)])
@pytest.mark.parametrize("output", ["f32", "bf16"])
@pytest.mark.parametrize("project", [False, True])
def test_mxfp8_lds_isolated_groups_match_oracle(mnk, output, project):
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(*mnk, 32, 1, "nk", output)
    package = compile_mxfp8(shape, project_image_identity=project)
    prov = package.descriptor.provenance
    assert prov["staging"] == "lds" and prov["warps"] == 8
    assert prov["macro_tile"] in ([128, 64], [128, 128])
    assert prov["runtime_mn_image"] is project
    assert prov["lds_runtime_k"] is project
    assert "scale_format = \"e8m0\"" in package.target_ir
    args, expected = _arguments(shape)
    want = expected.astype(ml_dtypes.bfloat16 if output == "bf16" else np.float32)
    for _ in range(2):
        receipt = rt.launch(_artifact(package), args)
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        np.testing.assert_array_equal(args["buffers"]["o"], want)
    rocm_isa.assert_selected(package.image.payload, chip="gfx1201",
        pattern=r"v_wmma_f32_16x16x16_\w+",
        require="v_wmma_f32_16x16x16_fp8_fp8", what="MXFP8 E8M0 LDS")


@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_lds_extreme_scales_and_nan(output):
    shape = BlockScaleShape(200, 2049, 1024, 32, 1, "nk", output)
    package = compile_mxfp8(shape)
    args, _ = _arguments(shape)
    buffers = args["buffers"]
    buffers["a_scale"][0, :] = 0
    buffers["b_scale"][:, 0] = 254
    buffers["a_scale"][1, :] = 254
    buffers["b_scale"][:, 1] = 0
    buffers["a_scale"][2, :] = 0
    buffers["b_scale"][:, 2] = 127
    buffers["a_scale"][3, 0] = 255
    buffers["b_scale"][1, 3] = 255
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        want = _reference(buffers["a"], buffers["b"].T,
                          buffers["a_scale"], buffers["b_scale"]).astype(
                              ml_dtypes.bfloat16 if output == "bf16" else np.float32)
        receipt = rt.launch(_artifact(package), args)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    actual = buffers["o"]
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(want))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_array_equal(actual[finite], want[finite])


@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_lds_image_reuses_mnk_with_edge_class(output):
    profiles = [BlockScaleShape(200, 2048, 1024, 32, 1, "nk", output),
                BlockScaleShape(201, 2112, 1536, 32, 1, "nk", output)]
    packages = [compile_mxfp8(s) for s in profiles]
    assert packages[0].image.payload == packages[1].image.payload
    assert packages[0].image.image_digest == packages[1].image.image_digest
    assert packages[0].descriptor.descriptor_digest != packages[1].descriptor.descriptor_digest
    for shape, package in zip(profiles, packages):
        args, expected = _arguments(shape)
        result = rt.launch(_artifact(package), args)
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        dtype = ml_dtypes.bfloat16 if output == "bf16" else np.float32
        np.testing.assert_array_equal(args["buffers"]["o"], expected.astype(dtype))


@pytest.mark.parametrize("field,value", [
    ("macro_tile", [64, 128]), ("workgroup", [32, 1, 1]), ("warps", True),
    ("warps", 4), ("macro_k", 64), ("pipeline_depth", 2), ("staging", "global"),
    ("runtime_shape_image", True), ("scale_format", "fp32"), ("scale_n", 2),
])
def test_bad_mxfp8_lds_descriptor_refused_before_hip(monkeypatch, field, value):
    shape = BlockScaleShape(200, 2048, 1024, 32, 1, "nk", "f32")
    package = compile_mxfp8(shape, schedule_policy="lds")
    args, _ = _arguments(shape)
    descriptor = replace(package.descriptor,
        provenance={**package.descriptor.provenance, field: value})
    def forbidden(*args, **kwargs):
        pytest.fail("invalid E8M0 LDS profile reached HIP")
    monkeypatch.setattr(rt, "_load_hip_for_launch", forbidden)
    with pytest.raises(RuntimeError, match="MXFP8 descriptor"):
        rt._submit_rocm_gfx1151_native(package.image, descriptor,
            args["buffers"], args["scalars"], None)


def test_mxfp8_lds_identity_keeps_scale_semantics():
    shape = BlockScaleShape(200, 2048, 1024, 32, 1, "nk", "f32")
    mx, fp = compile_mxfp8(shape), compile_blockscale(shape)
    assert mx.image.payload != fp.image.payload
    assert mx.descriptor.abi_id != fp.descriptor.abi_id
    assert mx.image.image_digest != fp.image.image_digest

@pytest.mark.parametrize("runtime_k", [32, 64, 128, 192, 256, 512])
def test_mxfp8_lds_reuses_image_at_small_runtime_k(runtime_k):
    compiled = BlockScaleShape(200, 2049, 1024, 32, 1, "nk", "f32")
    package = compile_mxfp8(compiled, schedule_policy="lds")
    runtime_shape = BlockScaleShape(200, 2049, runtime_k, 32, 1, "nk", "f32")
    rebound = compile_mxfp8(runtime_shape, schedule_policy="lds")
    assert rebound.image.payload == package.image.payload
    assert rebound.image.image_digest == package.image.image_digest
    assert rebound.descriptor.descriptor_digest != package.descriptor.descriptor_digest
    args, expected = _arguments(runtime_shape)
    # The original Graph-specific descriptor must continue to refuse this K.
    rejected = rt.launch(_artifact(package), args)
    assert not rejected["ok"] and rejected["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
    receipt = rt.launch(_artifact(rebound), args)
    assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
    np.testing.assert_array_equal(args["buffers"]["o"], expected)


@pytest.mark.parametrize("k", [128, 256, 512])
def test_short_mxfp8_selector_keeps_seed(k):
    package = compile_mxfp8(BlockScaleShape(128, 4096, k, 32, 1, "nk", "f32"))
    assert package.descriptor.provenance["staging"] == "global"
    assert package.descriptor.provenance["macro_tile"] == [16, 16]
