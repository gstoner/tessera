"""The diagnostic control must preserve codes, block axes and independent globals."""
from types import SimpleNamespace
import ml_dtypes
import numpy as np
import pytest
from benchmarks.rocm.benchmark_nvfp4_source_preserving_control import expand_projection_control, measure
from tessera.compiler.rocm_nvfp4_ingest import NVFP4Projection
from tessera.compiler.rocm_mxfp4 import pack_e2m1_codes


def projection(name="gate", global_scale=0.125, n=2, k=32):
    codes = np.broadcast_to(np.arange(16, dtype=np.uint8), (n, k // 16, 16)).reshape(n, k).copy()
    scales = np.resize(np.asarray([.25, .75, 1.5, 3.5], dtype=ml_dtypes.float8_e4m3fn), (n, k // 16))
    return NVFP4Projection(name, pack_e2m1_codes(codes), scales, global_scale)


def test_all_codes_signed_zeros_and_distinct_projection_globals_are_preserved():
    first, second = projection(), projection("up", global_scale=0.5)
    values, scales = expand_projection_control([first, second])
    expected = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    np.testing.assert_array_equal(values.astype(np.float32).view(np.uint32), np.tile(expected, (4, 2)).view(np.uint32))
    want = np.concatenate([first.e4m3_scales.astype(np.float32) * .125, second.e4m3_scales.astype(np.float32) * .5]).T
    np.testing.assert_array_equal(scales, want)
    assert values.flags.c_contiguous and scales.flags.c_contiguous


@pytest.mark.parametrize("items", [[], [projection(), projection()], [projection(), projection("up", k=64)]])
def test_invalid_projection_collection_rejects(items):
    with pytest.raises(ValueError):
        expand_projection_control(items)


@pytest.mark.parametrize("global_scale", [1e300, 1e-300, -1, np.inf, np.nan])
def test_invalid_or_unrepresentable_scale_products_reject(global_scale):
    with pytest.raises(ValueError):
        expand_projection_control([projection(global_scale=global_scale)])


def test_invalid_arguments_reject_before_capture_device_or_network(monkeypatch):
    import benchmarks.rocm.benchmark_nvfp4_source_preserving_control as recorder
    monkeypatch.setattr(recorder, "load_capture", lambda *a: pytest.fail("capture must not run"))
    with pytest.raises(ValueError, match="positive row counts"):
        measure(SimpleNamespace(windows=2, m=[128]))


@pytest.mark.parametrize("k", [16, 48, 80])
def test_control_keeps_native_nvfp4_k16_geometry_without_mx_k32_restriction(k):
    source = projection(k=k)
    values, scales = expand_projection_control([source])
    assert values.shape == (2, k) and scales.shape == (k // 16, 2)


@pytest.mark.parametrize("kind", ["codes_dtype", "scale_dtype", "scale_shape", "nan_scale"])
def test_source_storage_and_invalid_scale_planes_reject(kind):
    source = projection()
    codes, scales = source.packed_codes, source.e4m3_scales.copy()
    if kind == "codes_dtype":
        codes = codes.astype(np.int32)
    elif kind == "scale_dtype":
        scales = scales.astype(np.float32)
    elif kind == "scale_shape":
        scales = scales[:, :1]
    else:
        scales[0, 0] = np.nan
    with pytest.raises((TypeError, ValueError)):
        expand_projection_control([NVFP4Projection("gate", codes, scales, 0.125)])


def test_negative_zero_scale_is_retained_by_the_reference_control():
    source = projection()
    scales = np.full(source.e4m3_scales.shape, -0.0, dtype=ml_dtypes.float8_e4m3fn)
    _, expanded = expand_projection_control([NVFP4Projection("gate", source.packed_codes, scales, 0.125)])
    np.testing.assert_array_equal(expanded.view(np.uint32), np.full(expanded.shape, 0x80000000, np.uint32))
