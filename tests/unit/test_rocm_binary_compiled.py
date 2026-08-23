"""Compiler-generated elementwise binary arithmetic (sub/div/pow/max/min) on
gfx1151 — the S2 binary-arithmetic family, the binary sibling of the unary-math
lane.

The `tessera_rocm.binary` directive expands (via `generate-rocm-binary-kernel`)
into a flat 2-operand per-element kernel (one thread per element). Reachable
through `runtime.launch()` via `compiler_path="rocm_binary_compiled"`; op names
tessera.sub / div / pow / maximum / minimum; f16/bf16/f32 storage, f32 compute.
maximum/minimum are IEEE NaN-propagating.

Validated vs numpy. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._support.compiler_tool import run_tessera_opt


def _binary_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_binary_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"]}],
    })


# op_name -> (numpy reference, sampler producing (a, b) in a valid domain)
_ANY = lambda rng, shp: (rng.standard_normal(shp) * 1.5).astype(np.float32)
_POS = lambda rng, shp: (rng.random(shp) * 4.0 + 0.05).astype(np.float32)


def _s_sub(rng, shp):  return _ANY(rng, shp), _ANY(rng, shp)
def _s_div(rng, shp):  # divisor away from zero (both signs)
    a = _ANY(rng, shp)
    b = _POS(rng, shp) * rng.choice([-1.0, 1.0], size=shp).astype(np.float32)
    return a, b
def _s_pow(rng, shp):  # positive base, modest exponent
    return _POS(rng, shp), (_ANY(rng, shp) * 0.5)
def _s_minmax(rng, shp):  return _ANY(rng, shp), _ANY(rng, shp)
def _s_mod(rng, shp):
    a = _ANY(rng, shp) * np.float32(3.0)
    b = _POS(rng, shp) * rng.choice([-1.0, 1.0], size=shp).astype(np.float32)
    return a, b


_CASES = {
    "tessera.sub": (lambda a, b: a - b, _s_sub),
    "tessera.div": (lambda a, b: a / b, _s_div),
    "tessera.pow": (lambda a, b: np.power(a, b), _s_pow),
    "tessera.maximum": (np.maximum, _s_minmax),
    "tessera.minimum": (np.minimum, _s_minmax),
    "tessera.mod": (np.mod, _s_mod),
    "tessera.floor_div": (np.floor_divide, _s_mod),
}


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 2e-5), (np.float16, 4e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_binary_matches_numpy(op_name, dtype, tol, shape):
    rt = _binary_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    ref, sampler = _CASES[op_name]
    rng = np.random.default_rng(23 + len(shape) + int(np.prod(shape)))
    a, b = sampler(rng, shape)
    a, b = a.astype(dtype), b.astype(dtype)
    res = rt.launch(_artifact(rt, op_name), (a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_binary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    expect = np.asarray(
        ref(a.astype(np.float32), b.astype(np.float32))).astype(np.float32)
    np.testing.assert_allclose(out, expect, atol=tol, rtol=tol)


def test_binary_max_min_nan_propagating():
    rt = _binary_or_skip()
    a = np.array([1.0, np.nan, 3.0, -1.0], np.float32)
    b = np.array([2.0, 5.0, np.nan, -2.0], np.float32)
    mx = rt.launch(_artifact(rt, "tessera.maximum"), (a, b))
    mn = rt.launch(_artifact(rt, "tessera.minimum"), (a, b))
    assert mx["ok"] is True and mn["ok"] is True
    # IEEE maximumf/minimumf propagate NaN, matching np.maximum/np.minimum.
    np.testing.assert_array_equal(
        np.asarray(mx["output"]).astype(np.float32), np.maximum(a, b))
    np.testing.assert_array_equal(
        np.asarray(mn["output"]).astype(np.float32), np.minimum(a, b))


def test_binary_max_min_signed_zero_ties_are_ieee_ordered():
    """Pins the fleet-wide IEEE-754-2019 contract (decision 2026-08-23):
    signed zeros are ordered — maximum tie -> +0.0, minimum tie -> -0.0 —
    matching arith.maximumf/minimumf and the x86 CPU JIT lane
    (test_jit_tensor_elementwise_totality.py). The emitter previously
    wrapped the op in `select(a == b, b, ...)` to emulate numpy-on-x86
    (second operand wins); numpy's tie sign is an ISA accident (SSE vs
    NEON), not a portable contract, so it is NOT a valid oracle here —
    expectations are explicit."""
    rt = _binary_or_skip()
    a = np.array([0.0, -0.0, 5.0], np.float32)
    b = np.array([-0.0, 0.0, 5.0], np.float32)
    expected_signbit = {"tessera.maximum": False, "tessera.minimum": True}
    for op, tie_sign in expected_signbit.items():
        res = rt.launch(_artifact(rt, op), (a, b))
        assert res["ok"] is True, res.get("reason")
        out = np.asarray(res["output"]).astype(np.float32)
        np.testing.assert_array_equal(
            out == 0.0, np.array([True, True, False]))
        np.testing.assert_array_equal(
            np.signbit(out[:2]), np.array([tie_sign, tie_sign]))


@pytest.mark.parametrize("op_name,reference", [
    ("tessera.pow", np.power),
    ("tessera.mod", np.mod),
    ("tessera.floor_div", np.floor_divide),
])
def test_binary_difficult_domain_contract(op_name, reference):
    rt = _binary_or_skip()
    if op_name == "tessera.pow":
        a = np.array([-2, -2, -0.0, -0.0, 0.0, 2.0, np.inf, np.nan], np.float32)
        b = np.array([0.5, 3.0, 3.0, -3.0, -1.0, np.inf, -1.0, 2.0], np.float32)
    else:
        a = np.array([-5, -5, 5, 5, -0.0, 0.0, np.inf, 1.0, np.nan], np.float32)
        b = np.array([2, -2, 2, -2, 2, -2, 2, np.inf, 2], np.float32)
    with np.errstate(all="ignore"):
        expected = reference(a, b).astype(np.float32)
    result = rt.launch(_artifact(rt, op_name), (a, b))
    assert result["ok"] is True, result.get("reason")
    out = np.asarray(result["output"], np.float32)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(out), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(out), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(out[finite], expected[finite], atol=2e-5, rtol=2e-5)
    zeros = finite & (expected == 0)
    np.testing.assert_array_equal(np.signbit(out[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("op_name,tie_sign", [
    ("tessera.maximum", False), ("tessera.minimum", True),
])
def test_binary_minmax_signed_zero_contract(op_name, tie_sign):
    # Explicit IEEE-754-2019 expectations, NOT a numpy oracle: numpy's ±0
    # tie sign is ISA-dependent (SSE second operand, NEON IEEE), so it only
    # agreed with the old numpy-emulating kernel on x86 hosts. Contract
    # (fleet decision 2026-08-23): max tie -> +0.0, min tie -> -0.0;
    # same-sign ties keep their sign; NaN propagates.
    rt = _binary_or_skip()
    a = np.array([0.0, -0.0, 0.0, -0.0, np.nan, 1.0], np.float32)
    b = np.array([-0.0, 0.0, 0.0, -0.0, 1.0, np.nan], np.float32)
    result = rt.launch(_artifact(rt, op_name), (a, b))
    assert result["ok"] is True, result.get("reason")
    out = np.asarray(result["output"], np.float32)
    np.testing.assert_array_equal(
        np.isnan(out), [False, False, False, False, True, True])
    np.testing.assert_array_equal(
        np.signbit(out[:4]), [tie_sign, tie_sign, False, True])


def test_binary_shape_mismatch_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_rocm_compiled_binary(_artifact(rt, "tessera.sub"), (a, b))


def test_binary_mixed_dtype_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 8), np.float16)
    with pytest.raises(ValueError, match="matching operand dtypes"):
        rt._execute_rocm_compiled_binary(_artifact(rt, "tessera.sub"), (a, b))


def test_binary_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_binary_compiled executor"):
        rt._execute_rocm_compiled_binary(_artifact(rt, "tessera.softmax"), (a, b))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────

_KINDS = ["sub", "div", "pow", "maximum", "minimum", "add", "mul", "mod",
          "floor_div"]


def _opt(directive, *passes):
    """Skips when this build lacks a requested pass (see _support.compiler_tool)."""
    return run_tessera_opt(directive, *passes)


@pytest.mark.parametrize("kind", _KINDS)
def test_binary_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.binary"() {name = "b", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-binary-kernel")
    assert ir.returncode == 0, ir.stderr
    m = re.search(r"gpu\.func @b\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 4
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-binary-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_binary_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.binary"() {name = "b", kind = "floof"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-binary-kernel")
    assert r.returncode != 0 and "ROCm binary math kind" in r.stderr
