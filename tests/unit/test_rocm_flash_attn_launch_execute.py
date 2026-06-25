"""The COMPILER-GENERATED ROCm flash_attn forward through ``runtime.launch()``.

The attention analog of the compiled-GEMM launch lane (test_rocm_compiled_launch_
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


def _fa_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, q, k, v, causal, scale):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v"],
                 "kwargs": {"causal": bool(causal), "scale": scale}}],
    })


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
