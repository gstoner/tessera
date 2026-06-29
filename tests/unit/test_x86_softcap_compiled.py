"""Compiler-generated softcap on x86 AVX-512 (P2e of S_SERIES_GAP_CLOSURE_PLAN)
— Gemma-style logit soft-cap cap*tanh(x/cap), composed on the transcendental
tanh lane (no new kernel; scalar cap broadcast on host). Reachable via
`compiler_path="x86_softcap_compiled"`. Validated vs cap*tanh(x/cap). Skip-clean:
x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, cap):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_softcap_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.softcap", "result": "o", "operands": ["a"],
                 "kwargs": {"cap": cap}}],
    })


@pytest.mark.parametrize("shape", [(64,), (4, 9), (2, 3, 5)])
@pytest.mark.parametrize("cap", [1.0, 30.0, 50.0])
def test_softcap(shape, cap):
    rt = _rt_or_skip()
    # span well past the cap so the tanh saturation is exercised
    x = (np.random.default_rng(0).standard_normal(shape) * 40).astype(np.float32)
    res = rt.launch(_art(rt, cap), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_softcap_compiled"
    ref = (cap * np.tanh(x / cap)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=2e-5)


def test_softcap_bounds_output():
    """Output is smoothly bounded to (-cap, cap)."""
    rt = _rt_or_skip()
    cap = 5.0
    x = (np.linspace(-1e3, 1e3, 257)).astype(np.float32)
    out = np.asarray(rt.launch(_art(rt, cap), (x,))["output"])
    assert np.all(np.abs(out) <= cap + 1e-4)


def test_softcap_rejects_nonpositive_cap():
    rt = _rt_or_skip()
    res = rt.launch(_art(rt, -1.0), (np.ones(4, np.float32),))
    assert res["ok"] is False
