"""The COMPILER-GENERATED ROCm flash_attn forward through ``runtime.launch()``.

The attention analog of the device_verified_jit-GEMM launch lane (test_rocm_compiled_launch_
execute.py): an artifact stamped ``compiler_path = "rocm_flash_attn_compiled"``
routes through ``runtime.launch()`` → the matrix → the
``rocm_flash_attn_compiled`` executor, which generates + serializes the FA-2
forward kernel in-process (tessera-opt, no mlir-opt) and launches the hsaco via
HIP. This is the additive "L4" step for attention: the kernel already executed
and was validated (test_rocm_flash_attn_compiled.py); here it becomes reachable
through the runtime executor table the way matmul is.

Skip-clean: tessera-opt not built, or no usable AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def test_execution_matrix_has_rocm_flash_attn_row():
    from tessera.compiler import execution_matrix as em
    row = em.lookup("rocm", "rocm_flash_attn_compiled")
    assert row is not None
    assert row.executor_id == "rocm_flash_attn_compiled"
    assert row.execution_kind == "native_gpu"
    assert "rocm_flash_attn_compiled" in em.KNOWN_EXECUTORS


def test_rocm_flash_attn_executor_registered():
    from tessera import runtime as rt
    assert "rocm_flash_attn_compiled" in rt._executor_table()


def test_mismatched_v_is_rejected_not_oob():
    """V is paired with K: `sk`/`bh_kv` (from K) drive the V device copy size, so
    a V with a shorter sequence length (or wrong head dim / head count) would
    read past the host buffer. The lane must reject it with a clean ValueError,
    never read OOB. Pure shape validation — runs before any GPU/tessera-opt
    call, so it needs no device."""
    from tessera import runtime as rt
    q = np.zeros((1, 1, 32, 16), np.float16)
    k = np.zeros((1, 1, 64, 16), np.float16)   # K: 64 keys
    v = np.zeros((1, 1, 32, 16), np.float16)   # V: only 32 — would OOB at sk=64
    art = _artifact(rt, q, k, v, causal=False, scale=0.25)
    with pytest.raises(ValueError, match="V to match K"):
        rt._execute_rocm_compiled_flash_attn(art, (q, k, v))


def _fa_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, q, k, v, causal, scale, op_name="tessera.flash_attn",
              extra_kwargs=None):
    kwargs = {"causal": bool(causal), "scale": scale}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["q", "k", "v"], "kwargs": kwargs}],
    })


# ── op-name acceptance (the `device_verified_jit` rocm_target_map rows must launch) ───────
# The flash-attn WMMA kernel realizes the whole multi-head family; the executor
# must accept the registry op names marked `device_verified_jit` in rocm_target_map, else
# the dashboard overstates runtime.launch() support. These are GPU-free: they
# assert the op-name gate is passed (a downstream validation fires), not the
# "handles exactly one" rejection.

@pytest.mark.parametrize("op_name", [
    "tessera.multi_head_attention", "tessera.gqa_attention",
    "tessera.mqa_attention", "tessera.attn_sliding_window",
])
def test_flash_attn_family_op_names_accepted(op_name):
    from tessera import runtime as rt
    # head_dim=15 (not a multiple of 16) trips the head_dim check, which is AFTER
    # the op-name gate — so reaching it proves the op name was accepted.
    bad = np.zeros((1, 1, 16, 15), np.float16)
    art = _artifact(rt, bad, bad, bad, causal=False, scale=0.25,
                    op_name=op_name, extra_kwargs={"window": 8})
    with pytest.raises(ValueError, match="head_dim a positive multiple of 16"):
        rt._execute_rocm_compiled_flash_attn(art, (bad, bad, bad))


def test_unknown_op_name_still_rejected():
    from tessera import runtime as rt
    z = np.zeros((1, 1, 16, 16), np.float16)
    art = _artifact(rt, z, z, z, causal=False, scale=0.25,
                    op_name="tessera.linear_attn")
    with pytest.raises(ValueError, match="handles exactly one"):
        rt._execute_rocm_compiled_flash_attn(art, (z, z, z))


def test_attn_sliding_window_op_name_requires_window():
    from tessera import runtime as rt
    z = np.zeros((1, 1, 16, 16), np.float16)
    art = _artifact(rt, z, z, z, causal=True, scale=0.25,
                    op_name="tessera.attn_sliding_window")  # no window kwarg
    with pytest.raises(ValueError, match="requires a positive `window`"):
        rt._execute_rocm_compiled_flash_attn(art, (z, z, z))


def _ref(q, k, v, scale, causal):
    qf, kf, vf = q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)
    s = scale * np.einsum("bhqd,bhkd->bhqk", qf, kf)
    _, _, sq, sk = s.shape
    if causal:
        i = np.arange(sq)[:, None]; j = np.arange(sk)[None, :]
        s = np.where((j > i)[None, None], -1e30, s)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s); p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, vf).astype(np.float32)


def _grouped_ref(q, k, v, scale, causal):
    """Reference for H query heads sharing G key/value heads."""
    groups = k.shape[1]
    ratio = q.shape[1] // groups
    expanded_k = np.repeat(k, ratio, axis=1)
    expanded_v = np.repeat(v, ratio, axis=1)
    return _ref(q, expanded_k, expanded_v, scale, causal)


@pytest.mark.parametrize("D,B,H,Sq,Sk,causal", [
    (16, 1, 2, 32, 48, 0),
    (64, 2, 2, 48, 48, 1),     # causal
    (64, 1, 2, 33, 17, 1),     # ragged + causal
])
def test_launch_rocm_flash_attn_matches_numpy(D, B, H, Sq, Sk, causal):
    rt = _fa_or_skip()
    rng = np.random.default_rng(11 + D + Sq + Sk + causal)
    q = (rng.standard_normal((B, H, Sq, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, H, Sk, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))

    res = rt.launch(_artifact(rt, q, k, v, causal, scale), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["runtime_status"] == "success"
    assert res["compiler_path"] == "rocm_flash_attn_compiled"
    assert res["execution_kind"] == "native_gpu"
    out = res["output"]
    assert out.shape == q.shape

    ref = _ref(q, k, v, scale, causal)
    maxerr = float(np.max(np.abs(out - ref)))
    assert maxerr < 2e-2, f"flash_attn launch maxerr={maxerr} " \
        f"D={D} {B}x{H}x{Sq}x{Sk} causal={causal}"


def test_launch_multi_head_attention_op_name_on_gpu():
    """End-to-end on gfx1151: the `multi_head_attention` op name launches via the
    flash_attn kernel (proves the `device_verified_jit` row, not just op-name acceptance)."""
    rt = _fa_or_skip()
    D, B, H, S = 16, 1, 2, 32
    rng = np.random.default_rng(123)
    q = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    res = rt.launch(_artifact(rt, q, k, v, False, scale,
                              op_name="tessera.multi_head_attention"), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    out = res["output"]
    maxerr = float(np.max(np.abs(out - _ref(q, k, v, scale, False))))
    assert maxerr < 2e-2, f"multi_head_attention op-name launch maxerr={maxerr}"


@pytest.mark.parametrize("op_name,kv_heads", [
    ("tessera.gqa_attention", 2),
    ("tessera.mqa_attention", 1),
])
def test_launch_grouped_attention_op_names_on_gpu(op_name, kv_heads):
    """Exact runtime-matrix path and numerical proof for GQA/MQA on gfx1151."""
    rt = _fa_or_skip()
    D, B, H, S = 16, 1, 8, 32
    rng = np.random.default_rng(321 + kv_heads)
    q = (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((B, kv_heads, S, D)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((B, kv_heads, S, D)) * 0.3).astype(np.float16)
    scale = 1.0 / float(np.sqrt(D))
    res = rt.launch(
        _artifact(rt, q, k, v, False, scale, op_name=op_name), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_flash_attn_compiled"
    assert res["execution_kind"] == "native_gpu"
    out = np.asarray(res["output"], np.float32)
    ref = _grouped_ref(q, k, v, scale, False)
    np.testing.assert_allclose(out, ref, rtol=0, atol=2e-2)
