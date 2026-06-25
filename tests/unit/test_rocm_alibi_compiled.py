"""Compiler-generated ALiBi positional-bias generator on gfx1151 — the
positional sibling of the rope lane.

The `tessera_rocm.alibi` directive expands (via `generate-rocm-alibi-kernel`)
into a flat elementwise kernel generating `bias[h,i,j] = slope[h]·(j−i)` of
shape `[H, S, S]`, computed in f32. Reachable through `runtime.launch()` via
`compiler_path="rocm_alibi_compiled"`, op name `tessera.alibi`, with
`num_heads`/`seq_len` kwargs and an optional `slopes` operand; f32/f16/bf16.

Validated vs the `nn.functional.alibi` numpy reference.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _alibi_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, num_heads, seq_len, *, with_slopes=False, dtype="f32"):
    operands = ["slopes"] if with_slopes else []
    arg_names = ["slopes"] if with_slopes else []
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_alibi_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": arg_names, "output_name": "o",
        "ops": [{"op_name": "tessera.alibi", "result": "o",
                 "operands": operands,
                 "kwargs": {"num_heads": num_heads, "seq_len": seq_len,
                            "dtype": dtype}}],
    })


def _ref(num_heads, seq_len, slopes=None):
    if slopes is None:
        slopes = 2.0 ** (-8.0 * np.arange(1, num_heads + 1, dtype=np.float32)
                         / num_heads)
    slopes = np.asarray(slopes, dtype=np.float32).reshape(num_heads, 1, 1)
    pos = np.arange(seq_len, dtype=np.float32)
    distance = pos.reshape(1, seq_len) - pos.reshape(seq_len, 1)
    return slopes * distance.reshape(1, seq_len, seq_len)


@pytest.mark.parametrize("dtype,tol", [
    ("f32", 1e-5), ("f16", 5e-2), ("bf16", 4e-1),
])
@pytest.mark.parametrize("num_heads,seq_len", [
    (4, 8), (8, 16), (2, 300),    # seq_len*seq_len*H spanning >1 block
])
def test_launch_alibi_matches_numpy(dtype, tol, num_heads, seq_len):
    rt = _alibi_or_skip()
    res = rt.launch(_artifact(rt, num_heads, seq_len, dtype=dtype),
                    tuple())
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_alibi_compiled"
    out = res["output"].astype(np.float32)
    # half precision: the (j−i) range grows to seq_len, so compare relative too.
    np.testing.assert_allclose(out, _ref(num_heads, seq_len), atol=tol, rtol=tol)


def test_explicit_slopes_operand():
    """A caller-supplied slopes buffer overrides the default ramp."""
    rt = _alibi_or_skip()
    h, s = 3, 12
    slopes = np.array([0.5, 0.25, 0.125], np.float32)
    res = rt.launch(_artifact(rt, h, s, with_slopes=True), (slopes,))
    assert res["ok"] is True, res.get("reason")
    out = res["output"].astype(np.float32)
    np.testing.assert_allclose(out, _ref(h, s, slopes), atol=1e-5, rtol=0)


def test_diagonal_is_zero():
    """bias[h,i,i] = slope·0 = 0 on the diagonal — an anchor independent of slope."""
    rt = _alibi_or_skip()
    out = rt.launch(_artifact(rt, 4, 16), tuple())["output"].astype(np.float32)
    diag = np.einsum("hii->hi", out)
    np.testing.assert_allclose(diag, np.zeros((4, 16)), atol=1e-6, rtol=0)


def test_slopes_length_mismatch_rejected():
    from tessera import runtime as rt
    bad = np.ones((5,), np.float32)  # num_heads=3 but 5 slopes
    with pytest.raises(ValueError, match="slopes must have length"):
        rt._execute_rocm_compiled_alibi(
            _artifact(rt, 3, 8, with_slopes=True), (bad,))
