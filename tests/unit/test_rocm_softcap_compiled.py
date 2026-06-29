"""Compiler-generated softcap on gfx1151 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
Gemma-style logit soft-cap cap*tanh(x/cap), composed on the COMPILER-GENERATED
unary tanh kernel (no new kernel; scalar cap broadcast on host). Reachable via
`compiler_path="rocm_softcap_compiled"`. Validated vs cap*tanh(x/cap) on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, cap):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_softcap_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.softcap", "result": "o", "operands": ["a"],
                 "kwargs": {"cap": cap}}],
    })


@pytest.mark.parametrize("shape", [(64,), (4, 9), (2, 3, 5)])
@pytest.mark.parametrize("cap", [1.0, 30.0, 50.0])
def test_softcap(shape, cap):
    rt = _rocm_or_skip()
    x = (np.random.default_rng(0).standard_normal(shape) * 40).astype(np.float32)
    res = rt.launch(_art(rt, cap), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_softcap_compiled"
    ref = (cap * np.tanh(x / cap)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, atol=1e-4)


def test_softcap_rejects_nonpositive_cap():
    rt = _rocm_or_skip()
    res = rt.launch(_art(rt, 0.0), (np.ones(4, np.float32),))
    assert res["ok"] is False
