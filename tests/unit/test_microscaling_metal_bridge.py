"""The macOS 27.0 MTLTensorDataType + auxiliary-plane bridge for microscaling.

Grounds Tessera's hardware-free low-precision contract in the concrete macOS 27.0
Metal API (Decision #27): the fp8/fp4/e8m0 element+scale types and the
multi-plane auxiliary-plane (blockFactors) machinery that is the runtime image of
a ScaleLayout. Hardware-free — no Metal call; just the mapping.
"""

from __future__ import annotations

from tessera.compiler import microscaling as M


def test_element_types_map_to_270_symbols():
    for d, sym, ver in [
        ("fp8_e4m3", "MTLTensorDataTypeFloat8E4M3", "27.0"),
        ("fp8_e5m2", "MTLTensorDataTypeFloat8E5M2", "27.0"),
        ("fp4_e2m1", "MTLTensorDataTypeFloat4E2M1", "27.0"),
        ("e8m0", "MTLTensorDataTypeFloat8UE8M0", "27.0"),
    ]:
        t = M.mtl_tensor_data_type(d)
        assert t is not None and t.mtl_symbol == sym and t.min_macos == ver


def test_int_types_track_their_real_availability():
    # int8 long-standing; int4/uint4 since 26.4; int2/uint2 land 27.0 — keep the
    # availability honest so the runtime gate is correct per type.
    assert M.mtl_tensor_data_type("int8").min_macos == "26.0"
    assert M.mtl_tensor_data_type("int4").min_macos == "26.4"
    assert M.mtl_tensor_data_type("int2").min_macos == "27.0"


def test_unknown_dtype_has_no_metal_image():
    assert M.mtl_tensor_data_type("bf16") is None
    assert M.mtl_tensor_data_type("nope") is None


def test_mxfp8_plane_plan_has_e8m0_scale_plane_block32():
    fmt = M.FORMATS["mxfp8_e4m3"]
    plan = M.metal_plane_plan(fmt, (64, 256))
    assert plan is not None
    assert plan.element.mtl_symbol == "MTLTensorDataTypeFloat8E4M3"
    assert len(plan.aux_planes) == 1
    aux = plan.aux_planes[0]
    assert aux.data_type == "MTLTensorDataTypeFloat8UE8M0"   # E8M0 MX scale
    assert aux.block_factors == (1, 32)                       # block-32 on last axis
    assert aux.scale_shape == (64, 8)                         # 256 / 32 = 8
    assert plan.min_macos == "27.0"


def test_nvfp4_plane_plan_has_e4m3_scale_block16():
    fmt = M.FORMATS["nvfp4"]
    plan = M.metal_plane_plan(fmt, (16, 64))
    assert plan is not None
    assert plan.element.mtl_symbol == "MTLTensorDataTypeFloat4E2M1"
    aux = plan.aux_planes[0]
    assert aux.data_type == "MTLTensorDataTypeFloat8E4M3"     # NVFP4 uses E4M3 scales
    assert aux.block_factors == (1, 16)
    assert aux.scale_shape == (16, 4)                         # 64 / 16 = 4


def test_per_tensor_int8_needs_no_auxiliary_plane():
    # A per-tensor scalar scale is applied in-kernel — no multi-plane tensor, so
    # this lane does not require the 27.0 auxiliary-plane machinery.
    fmt = M.FORMATS["int8"]
    plan = M.metal_plane_plan(fmt, (32, 32))
    assert plan is not None
    assert plan.aux_planes == ()
    assert plan.element.mtl_symbol == "MTLTensorDataTypeInt8"
    assert plan.min_macos == "26.0"


def test_plane_plan_round_trips_block_count_with_scale_shape():
    # The auxiliary plane's scale_shape must equal the contract's scale_shape —
    # one source of truth for "how many scales".
    fmt = M.FORMATS["mxfp4_e2m1"]
    shape = (8, 96)
    plan = M.metal_plane_plan(fmt, shape)
    assert plan.aux_planes[0].scale_shape == fmt.layout.scale_shape(shape)
