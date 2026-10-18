"""Host-free: exact NVFP4 scale admission against the elementwise value predicate."""
import ctypes as C
from functools import partial
from pathlib import Path
import shutil
import subprocess

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.nvfp4_expansion import checked_expansion_inputs, nvfp4_expansion_policy



@pytest.fixture(scope="module")
def bounds_library(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/nvfp4_scale_bounds.cpp"
    output = tmp_path_factory.mktemp("nvfp4-scale-bounds") / "bounds.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
                    str(source), "-o", str(output)], check=True, capture_output=True)
    return C.CDLL(str(output))


@pytest.fixture(params=("legacy", "native"))
def admission(request):
    bounds = None
    if request.param == "native":
        from tessera.compiler.nvfp4_scale_admission import native_scale_bounds
        bounds = native_scale_bounds(request.getfixturevalue("bounds_library"))
        assert bounds is not None
    return partial(checked_expansion_inputs, _scale_bounds=bounds)


def _reference_admits(scales, globals_, offsets):
    decoded = scales.astype(np.float64)
    if not np.isfinite(decoded).all() or np.any(decoded < 0):
        return False
    if not np.isfinite(globals_).all() or np.any(globals_ <= 0):
        return False
    for i, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rounded = (decoded[start:stop] * globals_[i]).astype(np.float32)
        if not np.isfinite(rounded).all() or np.any((decoded[start:stop] > 0) & (rounded == 0)):
            return False
    return True


def _ask(scales, globals_, offsets, admission):
    codes = np.zeros((scales.shape[0], scales.shape[1] * 8), np.uint8)
    if _reference_admits(scales, globals_, offsets):
        checked = admission(codes, scales, globals_, row_offsets=offsets,
                                           numeric_policy=nvfp4_expansion_policy())
        assert checked[0] is codes and checked[1] is scales and checked[2] is globals_
    else:
        with pytest.raises(ValueError):
            admission(codes, scales, globals_, row_offsets=offsets,
                                     numeric_policy=nvfp4_expansion_policy())


@pytest.mark.parametrize("code", range(256))
def test_every_e4m3_encoding_has_the_same_value_admission(code, admission):
    scales = np.array([[code]], np.uint8).view(ml_dtypes.float8_e4m3fn)
    _ask(scales, np.ones(1, np.float64), (0, 1), admission)


@pytest.mark.parametrize("code", range(1, 127))
@pytest.mark.parametrize("boundary", ("underflow", "overflow"))
@pytest.mark.parametrize("adjacent", (-1, 0, 1))
def test_every_positive_scale_at_f32_rounding_boundaries(code, boundary, adjacent, admission):
    scales = np.full((2, 2), code, np.uint8).view(ml_dtypes.float8_e4m3fn)
    value = np.float64(scales[0, 0])
    threshold = (np.ldexp(np.float64(1), -150) if boundary == "underflow"
                 else np.float64(np.finfo(np.float32).max) + np.ldexp(np.float64(1), 103))
    global_ = threshold / value
    if adjacent:
        global_ = np.nextafter(global_, np.float64(0 if adjacent < 0 else np.inf))
    _ask(scales, np.array([global_], np.float64), (0, 2), admission)


@pytest.mark.parametrize("global_", (np.nextafter(0., 1.), 1., np.finfo(np.float64).max))
def test_both_signed_zeros_remain_admitted_with_extreme_positive_globals(global_, admission):
    scales = np.array([[0, 128], [128, 0]], np.uint8).view(ml_dtypes.float8_e4m3fn)
    before = scales.view(np.uint8).copy()
    _ask(scales, np.array([global_], np.float64), (0, 2), admission)
    np.testing.assert_array_equal(scales.view(np.uint8), before)


@pytest.mark.parametrize("bad", (127, 129, 255))
@pytest.mark.parametrize("index", (0, 17, 63))
def test_invalid_encoding_is_not_hidden_by_other_extrema(bad, index, admission):
    codes = np.linspace(0, 126, 64).astype(np.uint8).reshape(8, 8)
    codes.flat[index] = bad
    _ask(codes.view(ml_dtypes.float8_e4m3fn), np.ones(2, np.float64), (0, 3, 8), admission)


@pytest.mark.parametrize("globals_", ((1., 1.), (1e-300, 1.), (1., 1e300), (0., 1.), (-1., 1.), (np.inf, 1.), (np.nan, 1.)))
def test_projection_boundaries_and_strided_scale_planes(globals_, admission):
    codes = (np.arange(48, dtype=np.uint8).reshape(6, 8) * 2) % 127
    scales = codes.view(ml_dtypes.float8_e4m3fn)[::2, ::2]
    assert not scales.flags.c_contiguous
    _ask(scales, np.array(globals_, np.float64), (0, 1, 3), admission)

def test_native_bounds_reject_late_bad_encoding_without_partial_commit(bounds_library):
    from tessera.compiler.nvfp4_scale_admission import native_scale_bounds
    assert native_scale_bounds(bounds_library) is not None
    source = (C.c_uint8 * 2)(55, 255)
    offsets = (C.c_int64 * 3)(0, 1, 2)
    output = (C.c_uint8 * 4)(81, 82, 83, 84)
    rc = bounds_library.tessera_rocm_program_nvfp4_scale_bounds(
        C.cast(source, C.c_void_p), 2, 2, 1, offsets, 3, output, 4)
    assert rc == 2 and list(output) == [81, 82, 83, 84]


@pytest.mark.parametrize("field", ("bytes", "rows", "columns", "offset_count", "output_bytes", "offsets", "alias"))
def test_native_bounds_refuse_malformed_frames_atomically(bounds_library, field):
    from tessera.compiler.nvfp4_scale_admission import native_scale_bounds
    assert native_scale_bounds(bounds_library) is not None
    source = (C.c_uint8 * 4)(55, 63, 48, 0)
    offsets = (C.c_int64 * 3)(0, 1, 2)
    output = (C.c_uint8 * 4)(81, 82, 83, 84)
    values = [C.cast(source, C.c_void_p), 4, 2, 2, offsets, 3, output, 4]
    positions = {"bytes": 1, "rows": 2, "columns": 3, "offset_count": 5, "output_bytes": 7}
    if field in positions:
        values[positions[field]] = 0
    elif field == "offsets":
        offsets[1] = -1
    else:
        values[6] = source
    assert bounds_library.tessera_rocm_program_nvfp4_scale_bounds(*values) == 1
    assert list(source) == [55, 63, 48, 0] and list(output) == [81, 82, 83, 84]
