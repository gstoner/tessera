"""x86 inclusive prefix scan (cumsum/cumprod/cummax/cummin) — the CPU analog of
the ROCm block-scan lane, loaded from libtessera_x86_elementwise.so.

Reachable through `runtime.launch()` via `compiler_path="x86_scan_compiled"`;
op names tessera.cumsum / cumprod / cummax / cummin; f32; same-shape output;
axis=None flattens (numpy semantics).

Validated vs numpy. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, axis):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_scan_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": {"axis": axis}}],
    })


_CASES = {
    "tessera.cumsum": (np.cumsum, (-2.0, 2.0)),
    "tessera.cumprod": (np.cumprod, (0.85, 1.15)),
    "tessera.cummax": (np.maximum.accumulate, (-3.0, 3.0)),
    "tessera.cummin": (np.minimum.accumulate, (-3.0, 3.0)),
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("shape,axis", [
    ((4, 50), -1), ((4, 50), 1), ((8, 300), -1), ((130,), 0), ((3, 5, 7), -1),
])
def test_x86_scan_matches_numpy(op_name, shape, axis):
    rt = _x86_or_skip()
    ref, (lo, hi) = _CASES[op_name]
    rng = np.random.default_rng(89 + len(shape) + int(np.prod(shape)))
    x = (rng.random(shape) * (hi - lo) + lo).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name, axis), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_scan_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    expect = ref(x, axis=axis)
    assert out.shape == expect.shape
    np.testing.assert_allclose(out, expect, atol=2e-4, rtol=2e-4)


def test_x86_scan_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_scan_compiled executor"):
        rt._execute_x86_compiled_scan(_artifact(rt, "tessera.softmax", -1), (x,))


def test_x86_scan_rejects_non_f32():
    rt = _x86_or_skip()
    x = np.zeros((4, 8), np.float64)
    with pytest.raises(ValueError, match="f32 only"):
        rt._execute_x86_compiled_scan(_artifact(rt, "tessera.cumsum", -1), (x,))


@pytest.mark.parametrize(
    ("op_name", "reference"),
    [
        ("tessera.cummax", np.maximum.accumulate),
        ("tessera.cummin", np.minimum.accumulate),
    ],
)
def test_x86_extrema_scan_propagates_nan_and_preserves_zero_sign(
    op_name, reference
):
    rt = _x86_or_skip()
    x = np.asarray(
        [
            [-0.0, 0.0, -0.0, 1.0, np.nan, -2.0],
            [0.0, -0.0, 0.0, -1.0, 2.0, np.nan],
        ],
        dtype=np.float32,
    )
    result = rt.launch(_artifact(rt, op_name, -1), (x,))
    assert result["ok"] is True, result.get("reason")
    output = np.asarray(result["output"], dtype=np.float32)
    expected = reference(x, axis=-1)
    np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(output[finite], expected[finite])
    zeros = finite & (expected == 0.0)
    np.testing.assert_array_equal(np.signbit(output[zeros]), np.signbit(expected[zeros]))
