"""Checked runtime proof for the native gfx1201 E8M0 K32 package."""
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8, MXFP8_PACKAGE_ABIS
from tests.device.rocm.test_mxfp8_scheduled_scale import _inputs, _reference, pytestmark


def _artifact(package):
    return rt.RuntimeArtifact(metadata={"target": package.image.target},
                              native_image=package.image, launch_descriptor=package.descriptor,
                              tile_ir=package.tile_ir, target_ir=package.target_ir)


def _arguments(shape):
    a, b, sa, sb = _inputs(shape)
    output = np.full((shape.m, shape.n), np.nan,
                     dtype=ml_dtypes.bfloat16 if shape.output == "bf16" else np.float32)
    weight = np.ascontiguousarray(b.T) if shape.weight_layout == "nk" else b
    return {"buffers": {"a": a, "b": weight, "a_scale": sa, "b_scale": sb, "o": output},
            "scalars": {"M": shape.m, "N": shape.n, "K": shape.k}}, _reference(a, b, sa, sb)


@pytest.mark.parametrize("mnk", [(17, 19, 64), (16, 16, 128), (16, 2, 64)])
@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_checked_mxfp8_multigroup(mnk, layout, output):
    assert rt._rocm_live_arch() == "gfx1201"
    shape = BlockScaleShape(*mnk, scale_k=32, scale_n=1, weight_layout=layout, output=output)
    package = compile_mxfp8(shape)
    assert package.descriptor.abi_id == MXFP8_PACKAGE_ABIS[(layout, output)]
    assert [b.dtype for b in package.descriptor.buffers[2:4]] == ["uint8", "uint8"]
    arguments, expected = _arguments(shape)
    result = rt.launch(_artifact(package), arguments)
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    actual = arguments["buffers"]["o"]
    if output == "bf16":
        np.testing.assert_array_equal(actual.view(np.uint16),
                                      expected.astype(ml_dtypes.bfloat16).view(np.uint16))
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_runtime_image_reuses_mnk(layout, output):
    first = BlockScaleShape(17, 19, 64, 32, 1, layout, output)
    second = BlockScaleShape(31, 7, 128, 32, 1, layout, output)
    p, q = compile_mxfp8(first), compile_mxfp8(second)
    assert p.image.image_digest == q.image.image_digest
    assert p.descriptor.descriptor_digest != q.descriptor.descriptor_digest
    for shape, package in [(first, p), (second, q)]:
        arguments, expected = _arguments(shape)
        result = rt.launch(_artifact(package), arguments)
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        actual = arguments["buffers"]["o"].astype(np.float32)
        rounded = expected.astype(ml_dtypes.bfloat16).astype(np.float32) if output == "bf16" else expected
        np.testing.assert_array_equal(actual, rounded)


@pytest.mark.parametrize("field,value", [
    ("scale_k", 16), ("scale_format", "fp32"), ("staging", "lds"),
    ("warps", 2), ("macro_k", 64), ("output_storage", "bf16"),
    ("split_k", 2), ("runtime_mn_image", True),
    ("numeric_policy", {"storage": "e4m3", "accum": "f16", "execution_mode": "exact_per_block"}),
])
def test_mxfp8_rejects_conflicting_scale_descriptor_before_hip(monkeypatch, field, value):
    shape = BlockScaleShape(17, 19, 64, 32, 1)
    package = compile_mxfp8(shape)
    arguments, _ = _arguments(shape)
    descriptor = replace(package.descriptor,
                         provenance={**package.descriptor.provenance, field: value})
    def forbidden(*args, **kwargs):
        pytest.fail("invalid E8M0 contract reached HIP")
    monkeypatch.setattr(rt, "_load_hip_for_launch", forbidden)
    with pytest.raises(RuntimeError, match="MXFP8 descriptor"):
        rt._submit_rocm_gfx1151_native(package.image, descriptor,
                                      arguments["buffers"], arguments["scalars"], None)


@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_special_scales_nan_subnormal(output):
    shape = BlockScaleShape(17, 19, 64, 32, 1, "nk", output)
    package = compile_mxfp8(shape)
    arguments, _ = _arguments(shape)
    buffers = arguments["buffers"]
    buffers["a_scale"][1, 0] = 255
    buffers["b_scale"][0, 1] = 255
    buffers["a_scale"][2, :] = 0
    buffers["b_scale"][:, 2] = 127
    expected = _reference(buffers["a"], buffers["b"].T,
                          buffers["a_scale"], buffers["b_scale"])
    result = rt.launch(_artifact(package), arguments)
    assert result["ok"] and result["execution_kind"] == "native_gpu", result
    actual = buffers["o"]
    rounded = expected.astype(ml_dtypes.bfloat16) if output == "bf16" else expected
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(rounded))
    finite = np.isfinite(rounded)
    np.testing.assert_array_equal(actual[finite], rounded[finite])


def test_mxfp8_image_identity_distinguishes_fp32_scales():
    from tessera.compiler.rocm_fp8_blockscale import compile_blockscale
    shape = BlockScaleShape(17, 19, 64, 32, 1)
    mx, fp = compile_mxfp8(shape), compile_blockscale(shape)
    assert mx.image.image_digest != fp.image.image_digest
    assert mx.image.payload != fp.image.payload
    assert mx.descriptor.abi_id != fp.descriptor.abi_id
