"""Compiler-generated elementwise unary math (exp/log/sqrt/erf/…) on gfx1151 —
the S2 scalar-math / stability family, the unary sibling of the activation lane.

The `tessera_rocm.unary` directive expands (via `generate-rocm-unary-kernel`)
into a flat per-element kernel (one thread per element). Reachable through
`runtime.launch()` via `compiler_path="rocm_unary_compiled"`; op names
tessera.exp / log / sqrt / rsqrt / reciprocal / absolute (abs) / sign / erf /
tanh / sigmoid / log1p / expm1 / softplus; f16/bf16/f32 storage, f32 compute.

Validated vs numpy. Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _unary_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_unary_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"]}],
    })


def _np_softplus(x):
    # Stable: log1p(exp(-|x|)) + max(x, 0)  (matches the kernel)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


# op_name -> (numpy reference, sampler producing a valid domain)
_DOMAIN_POS = lambda rng, shp: (rng.random(shp) * 4.0 + 0.05).astype(np.float32)
_DOMAIN_ANY = lambda rng, shp: (rng.standard_normal(shp) * 1.5).astype(np.float32)

_CASES = {
    "tessera.exp": (np.exp, _DOMAIN_ANY),
    "tessera.log": (np.log, _DOMAIN_POS),
    "tessera.sqrt": (np.sqrt, _DOMAIN_POS),
    "tessera.rsqrt": (lambda x: 1.0 / np.sqrt(x), _DOMAIN_POS),
    "tessera.reciprocal": (np.reciprocal, _DOMAIN_POS),
    "tessera.absolute": (np.abs, _DOMAIN_ANY),
    "tessera.abs": (np.abs, _DOMAIN_ANY),
    "tessera.sign": (np.sign, _DOMAIN_ANY),
    "tessera.erf": (None, _DOMAIN_ANY),  # erf filled in below if scipy present
    "tessera.tanh": (np.tanh, _DOMAIN_ANY),
    "tessera.sigmoid": (lambda x: 1.0 / (1.0 + np.exp(-x)), _DOMAIN_ANY),
    "tessera.log1p": (np.log1p, _DOMAIN_POS),
    "tessera.expm1": (np.expm1, _DOMAIN_ANY),
    "tessera.softplus": (_np_softplus, _DOMAIN_ANY),
    # tail: trig / special / rounding (2026-06-26)
    "tessera.cos": (np.cos, _DOMAIN_ANY),
    # tan: stay inside (-π/2, π/2) to avoid the poles
    "tessera.tan": (np.tan, lambda rng, shp:
                    (rng.random(shp) * 2.8 - 1.4).astype(np.float32)),
    "tessera.sinh": (np.sinh, _DOMAIN_ANY),
    "tessera.cosh": (np.cosh, _DOMAIN_ANY),
    # asin/acos need |x| <= 1
    "tessera.asin": (np.arcsin, lambda rng, shp:
                     (rng.random(shp) * 1.8 - 0.9).astype(np.float32)),
    "tessera.acos": (np.arccos, lambda rng, shp:
                     (rng.random(shp) * 1.8 - 0.9).astype(np.float32)),
    "tessera.atan": (np.arctan, _DOMAIN_ANY),
    "tessera.erfc": (None, _DOMAIN_ANY),  # erfc filled below if scipy present
    "tessera.floor": (np.floor, _DOMAIN_ANY),
    "tessera.ceil": (np.ceil, _DOMAIN_ANY),
    "tessera.round": (np.round, _DOMAIN_ANY),   # round-half-to-even
    "tessera.trunc": (np.trunc, _DOMAIN_ANY),
}


def _erf_ref():
    try:
        from scipy.special import erf  # type: ignore
        return erf
    except Exception:
        # vectorized math.erf fallback — no scipy dependency required
        import math
        return np.vectorize(math.erf)


def _erfc_ref():
    try:
        from scipy.special import erfc  # type: ignore
        return erfc
    except Exception:
        import math
        return np.vectorize(math.erfc)


@pytest.mark.parametrize("op_name", list(_CASES))
@pytest.mark.parametrize("dtype,tol", [
    (np.float32, 2e-5), (np.float16, 4e-3), ("bf16", 3e-2),
])
@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7)])
def test_unary_matches_numpy(op_name, dtype, tol, shape):
    rt = _unary_or_skip()
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    ref, sampler = _CASES[op_name]
    if ref is None:
        ref = _erfc_ref() if op_name == "tessera.erfc" else _erf_ref()
    rng = np.random.default_rng(11 + len(shape) + int(np.prod(shape)))
    x = sampler(rng, shape).astype(dtype)
    res = rt.launch(_artifact(rt, op_name), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_unary_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    expect = np.asarray(ref(x.astype(np.float32))).astype(np.float32)
    np.testing.assert_allclose(out, expect, atol=tol, rtol=tol)


def test_unary_sign_at_zero():
    rt = _unary_or_skip()
    x = np.array([-2.0, -0.0, 0.0, 3.0], np.float32)
    res = rt.launch(_artifact(rt, "tessera.sign"), (x,))
    assert res["ok"] is True, res.get("reason")
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_array_equal(out, np.array([-1.0, 0.0, 0.0, 1.0], np.float32))


def test_unary_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="rocm_unary_compiled executor"):
        rt._execute_rocm_compiled_unary(_artifact(rt, "tessera.softmax"), (x,))


# ── GPU-free codegen gate (needs only tessera-opt, not a GPU) ────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"

_KINDS = ["exp", "log", "sqrt", "rsqrt", "reciprocal", "abs", "neg", "sign",
          "erf", "tanh", "sigmoid", "log1p", "expm1", "softplus",
          "cos", "tan", "sinh", "cosh", "asin", "acos", "atan", "erfc",
          "floor", "ceil", "round", "trunc"]


def _opt(directive, *passes):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    return subprocess.run([str(_OPT), "-", *passes], input=directive,
                          capture_output=True, text=True)


@pytest.mark.parametrize("kind", _KINDS)
def test_unary_codegen_and_lowers(kind):
    import re
    d = ('module {\n  "tessera_rocm.unary"() {name = "u", '
         f'kind = "{kind}", dtype = "f32"}} : () -> ()\n}}\n')
    ir = _opt(d, "--generate-rocm-unary-kernel")
    assert ir.returncode == 0, ir.stderr
    m = re.search(r"gpu\.func @u\(([^)]*)\)", ir.stdout)
    assert m and len([a for a in m.group(1).split(",") if a.strip()]) == 3
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-unary-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_unary_codegen_bad_kind_rejected():
    d = ('module {\n  "tessera_rocm.unary"() {name = "u", kind = "floof"} '
         ': () -> ()\n}\n')
    r = _opt(d, "--generate-rocm-unary-kernel")
    assert r.returncode != 0 and "unknown kind" in r.stderr
