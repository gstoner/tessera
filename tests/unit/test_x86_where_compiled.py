"""x86 ternary select where(cond, a, b) — the CPU analog of the ROCm
where lane, loaded from libtessera_x86_elementwise.so.

Reachable through `runtime.launch()` via `compiler_path="x86_where_compiled"`;
op name tessera.where; cond i8 normalized != 0, a/b/out f32
(_mm512_cmpneq_epi8_mask + _mm512_mask_blend_ps).

Validated vs np.where. Skip-clean: libtessera_x86_elementwise.so absent.
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
        "target": "x86", "compiler_path": "x86_where_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["cond", "a", "b"], "output_name": "o",
        "ops": [{"op_name": "tessera.where", "result": "o",
                 "operands": ["cond", "a", "b"]}],
    })


@pytest.mark.parametrize("shape", [(8, 64), (130,), (3, 5, 7), (5,), (1,)])
def test_x86_where_matches_numpy(shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(83 + len(shape) + int(np.prod(shape)))
    cond = rng.integers(0, 4, size=shape).astype(np.uint8)
    a = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(shape).astype(np.float32)
    res = rt.launch(_artifact(rt), (cond, a, b))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_where_compiled"
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_array_equal(out, np.where(cond != 0, a, b))


def test_x86_where_shape_mismatch_rejected():
    from tessera import runtime as rt
    cond = np.zeros((4, 8), np.uint8)
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching operand shapes"):
        rt._execute_x86_compiled_where(_artifact(rt), (cond, a, b))


def test_x86_where_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    art = rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_where_compiled",
        "arg_names": ["a"], "output_name": "o",
        "ops": [{"op_name": "tessera.softmax", "result": "o",
                 "operands": ["a"]}],
    })
    with pytest.raises(ValueError, match="x86_where_compiled executor"):
        rt._execute_x86_compiled_where(art, (a,))
