"""Owning-device numerical policy tests for native ROCm math recipes."""
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import rocm_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from benchmarks.rocm.benchmark_native_math_schedule import graph, MathCase, expected

@pytest.mark.parametrize("kind", ["sqrt", "exp", "add", "div", "cumsum", "cummax"])
@pytest.mark.parametrize("columns", [17, 257])
def test_native_math_nan_infinity_and_zero_semantics(kind, columns):
    arch = rt._rocm_live_arch()
    if arch not in {"gfx1151", "gfx1201"}:
        pytest.skip("requires owning gfx1151/gfx1201 ROCm device")
    hip = rt._load_hip_for_launch()
    assert hip is not None and hip.hipInit(0) == 0
    source = graph(kind, arch, (3, columns))
    tool = find_tessera_opt()
    schedule = run_tessera_opt(tool, source, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    scan = kind in {"cumsum", "cummax"}
    binary = kind in {"add", "div"}
    family = "scan" if scan else "scalar_binary" if binary else "scalar_unary"
    directive = "tessera_rocm."+("scan" if scan else "binary" if binary else "unary")
    target, _, image, *_ = native._compile_native_tile_ir(
        tile, directive=directive, family=family, architecture=arch)
    a = np.full((3, columns), np.float32(.25))
    a[0, :7] = [0., -0., -1., np.inf, -np.inf, np.nan, 1.]
    a[1, -3:] = [np.inf, -np.inf, np.nan]
    arrays = [a]
    if binary:
        b = np.full_like(a, 2.)
        b[0, :7] = [1., 1., -0., 2., np.inf, 0., np.nan]
        arrays.append(b)
    with np.errstate(all="ignore"):
        oracle = expected(kind, arrays)
    case = MathCase(hip, image, native._directive_symbol(target, directive), arrays, scan)
    try:
        actual = case.download()
    finally:
        case.close()
    np.testing.assert_allclose(actual, oracle, rtol=2e-5, atol=2e-5, equal_nan=True)
    # IEEE NaN and infinity classification is part of the policy, not a tolerance.
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(oracle))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(oracle))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(oracle))
    if kind in {"sqrt", "div"}:
        zero = oracle == 0
        np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(oracle[zero]))
