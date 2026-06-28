"""Compiler-generated popcount on x86 AVX-512 (P2e of S_SERIES_GAP_CLOSURE_PLAN)
— set-bit count per i32 element via the VPOPCNTDQ instruction
(_mm512_popcnt_epi32), on the bitwise lane (kind 4, unary). Reachable via
`compiler_path="x86_bitwise_compiled"`. Validated vs a numpy bit-count
reference. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_bitwise_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.popcount", "result": "o",
                 "operands": ["a"]}],
    })


def _popref(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    return np.array([bin(int(v) & 0xFFFFFFFF).count("1") for v in flat],
                    np.int32).reshape(x.shape)


# (130,) and (17,) exercise the scalar tail past the 16-lane __m512i body.
@pytest.mark.parametrize("shape", [(8, 64), (130,), (17,), (3, 5, 7)])
def test_x86_popcount_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(7 + int(np.prod(shape)))
    a = rng.integers(0, 1 << 32, size=shape, dtype=np.uint32).view(np.int32)
    res = rt.launch(_artifact(rt), (a,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_bitwise_compiled"
    out = np.asarray(res["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, _popref(a))


def test_x86_popcount_edge_values():
    """0 / all-ones / INT_MIN / INT_MAX / alternating-bit patterns."""
    rt = _x86_or_skip()
    a = np.array([0, -1, 1, 2**31 - 1, -(2**31), 0x55555555, 0x7FFFFFFF],
                 np.int32)
    out = np.asarray(rt.launch(_artifact(rt), (a,))["output"]).astype(np.int32)
    np.testing.assert_array_equal(out, _popref(a))
    # all-ones int32 has 32 set bits; zero has 0.
    assert out[1] == 32 and out[0] == 0
