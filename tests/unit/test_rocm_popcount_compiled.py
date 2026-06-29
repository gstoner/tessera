"""Compiler-generated popcount on gfx1151 (P2e of S_SERIES_GAP_CLOSURE_PLAN) —
set-bit count per i32 element via math.ctpop (RDNA v_bcnt), on the
COMPILER-GENERATED bitwise lane (kind "popcount", unary). Reachable via
`compiler_path="rocm_bitwise_compiled"`. Validated vs a numpy bit-count
reference on gfx1151. Skip-clean: tessera-opt not built / no GPU.
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


def _artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_bitwise_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.popcount", "result": "o",
                 "operands": ["a"]}],
    })


def _popref(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    return np.array([bin(int(v) & 0xFFFFFFFF).count("1") for v in flat],
                    np.int32).reshape(x.shape)


# (257,) crosses the 256-thread block, exercising the strided grid.
@pytest.mark.parametrize("shape", [(8, 64), (257,), (3, 5, 7)])
def test_rocm_popcount_matches_numpy(shape):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7 + int(np.prod(shape)))
    a = rng.integers(0, 1 << 32, size=shape, dtype=np.uint32).view(np.int32)
    res = rt.launch(_artifact(rt), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_bitwise_compiled"
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, _popref(a))


def test_rocm_popcount_edge_values():
    rt = _rocm_or_skip()
    a = np.array([0, -1, 1, 2**31 - 1, -(2**31), 0x55555555], np.int32)
    out = np.asarray(rt.launch(_artifact(rt), (a,))["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, _popref(a))
    assert out[1] == 32 and out[0] == 0
