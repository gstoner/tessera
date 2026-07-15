import time

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.stdlib import attention


def _artifact(op_name, arg_names, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_sparse_attn_compiled",
        "executable": True,
        "arg_names": list(arg_names),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(arg_names),
            "kwargs": dict(kwargs),
        }],
    })


def _qkv(seed=51, *, B=1, Hq=4, Hkv=2, Sq=8, Sk=8, D=4, Dv=5):
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal((B, Hq, Sq, D)) * 0.2).astype(np.float32),
        (rng.standard_normal((B, Hkv, Sk, D)) * 0.2).astype(np.float32),
        (rng.standard_normal((B, Hkv, Sk, Dv)) * 0.2).astype(np.float32),
    )


def test_rocm_msa_selected_blocks_matches_stdlib():
    Q, K, V = _qkv()
    block_size = 2
    top_k = 2
    scores = attention.msa_index_scores(Q, K, block_size=block_size)
    selected = attention.msa_select_blocks(scores, top_k=top_k, block_size=block_size)
    art = _artifact(
        "tessera.msa_sparse_attention",
        ["Q", "K", "V", "selected"],
        {"block_size": block_size, "top_k": top_k, "causal": True},
    )

    res = rt.launch(art, (Q, K, V, selected))

    assert res["ok"], res.get("reason")
    assert res["compiler_path"] == "rocm_sparse_attn_compiled"
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    ref = attention.msa_sparse_attention(
        Q, K, V, block_size=block_size, top_k=top_k,
        selected_block_ids=selected)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-5, atol=2e-5)


def test_rocm_msa_select_all_matches_dense_gqa():
    Q, K, V = _qkv(seed=52, Sq=8, Sk=8)
    block_size = 2
    top_k = K.shape[2] // block_size
    art = _artifact(
        "tessera.msa_sparse_attention",
        ["Q", "K", "V"],
        {"block_size": block_size, "top_k": top_k, "causal": True},
    )

    res = rt.launch(art, (Q, K, V))

    assert res["ok"], res.get("reason")
    dense = attention.dense_causal_attention(Q, K, V).astype(np.float32)
    np.testing.assert_allclose(res["output"], dense, rtol=2e-5, atol=2e-5)


def test_rocm_msa_auto_gpu_select_tiled_path_matches_stdlib():
    Q, K, V = _qkv(seed=56, Sq=8, Sk=8, Dv=6)
    kw = {
        "block_size": 2,
        "top_k": 2,
        "causal": True,
        "selection_strategy": "auto",
        "attention_strategy": "tiled",
    }
    res = rt.launch(_artifact("tessera.msa_sparse_attention", ["Q", "K", "V"], kw),
                    (Q, K, V))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    ref = attention.msa_sparse_attention(Q, K, V, block_size=2, top_k=2, causal=True)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-5, atol=2e-5)


def test_rocm_dsa_selected_layout_matches_stdlib():
    Q, K, V = _qkv(seed=53, Sq=8, Sk=8)
    kw = {"block_size": 2, "top_k_blocks": 2, "causal": True}
    art = _artifact("tessera.dsa_block_sparse_attention", ["Q", "K", "V"], kw)

    res = rt.launch(art, (Q, K, V))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    ref = attention.dsa_block_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-5, atol=2e-5)


def test_rocm_deepseek_sparse_attention_matches_ops_reference():
    import tessera as ts

    Q, K, V = _qkv(seed=57, Hq=2, Hkv=2, Sq=8, Sk=8, D=4, Dv=4)
    gate = np.random.default_rng(58).standard_normal((1, 2, 8, 3)).astype(np.float32)
    kw = {"window_size": 4, "block_size": 2, "top_k": 2, "causal": True}
    res = rt.launch(
        _artifact("tessera.deepseek_sparse_attention", ["Q", "K", "V", "gate"], kw),
        (Q, K, V, gate),
    )

    assert res["ok"], res.get("reason")
    assert res["compiler_path"] == "rocm_sparse_attn_compiled"
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    ref = ts.ops.deepseek_sparse_attention(Q, K, V, gate, **kw)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-5, atol=2e-5)


def test_rocm_deepseek_topk_branch_does_not_apply_msa_token_mask(monkeypatch):
    import tessera as ts

    Q, K, V = _qkv(seed=60, Hq=2, Hkv=2, Sq=8, Sk=8, D=4, Dv=4)
    gate = np.zeros((1, 2, 8, 3), dtype=np.float32)
    kw = {"window_size": 4, "block_size": 4, "top_k": 1, "causal": True}
    seen: dict[str, object] = {}

    def fake_topk(scores, np_mod, *, top_k, block_size, causal=True,
                  force_local_block=True, q_positions=None):
        sc = np_mod.asarray(scores, dtype=np.float64).copy()
        B, H, Sq, nb = sc.shape
        if causal:
            q_block = np_mod.arange(Sq) // int(block_size)
            future = np_mod.arange(nb)[None, None, None, :] > q_block[None, None, :, None]
            sc = np_mod.where(future, -np.inf, sc)
        idx = np_mod.argpartition(-sc, int(top_k) - 1, axis=-1)[..., :int(top_k)]
        return np_mod.sort(idx, axis=-1).astype(np.int64)

    def fake_selected(q, k, v, selected, np_mod, *, block_size, causal=True,
                      q_positions=None, scale=None, tiled=True):
        seen["causal"] = causal
        assert causal is False
        out = np_mod.zeros(q.shape[:-1] + (v.shape[-1],), dtype=np.float64)
        attn_scale = (1.0 / np_mod.sqrt(q.shape[-1])) if scale is None else float(scale)
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                for s in range(q.shape[2]):
                    rows = []
                    val_rows = []
                    for blk in selected[b, h, s]:
                        lo = int(blk) * int(block_size)
                        hi = lo + int(block_size)
                        rows.append(k[b, h, lo:hi])
                        val_rows.append(v[b, h, lo:hi])
                    K_sel = np_mod.concatenate(rows, axis=0)
                    V_sel = np_mod.concatenate(val_rows, axis=0)
                    logits = (q[b, h, s].astype(np.float64) @ K_sel.T.astype(np.float64)) * attn_scale
                    weights = np_mod.exp(logits - logits.max())
                    weights = weights / weights.sum()
                    out[b, h, s] = weights @ V_sel
        return out.astype(np.result_type(q, k, v), copy=False)

    monkeypatch.setattr(rt, "_rocm_block_sparse_topk_select_native", fake_topk)
    monkeypatch.setattr(rt, "_rocm_selected_block_attention_native", fake_selected)

    out, kind = rt._deepseek_sparse_attention_rocm(Q, K, V, gate, np, **kw)

    assert kind == "native_gpu"
    assert seen["causal"] is False
    ref = ts.ops.deepseek_sparse_attention(Q, K, V, gate, **kw)
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.performance
def test_dk2_rocm_sparse_attention_perf_baseline_is_bounded():
    Q, K, V = _qkv(seed=54, Sq=8, Sk=8, D=4)
    kw = {"block_size": 2, "top_k": 2, "causal": True}
    art = _artifact("tessera.msa_sparse_attention", ["Q", "K", "V"], kw)

    direct_vals = []
    launch_vals = []
    for _ in range(7):
        t0 = time.perf_counter()
        attention.msa_sparse_attention(Q, K, V, **kw)
        direct_vals.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        rt.launch(art, (Q, K, V))
        launch_vals.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(launch_vals)) < max(75.0, float(np.median(direct_vals)) * 4.0)


@pytest.mark.hardware_rocm
def test_rocm_sparse_attention_native_gpu_matches_reference_on_hardware():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    Q, K, V = _qkv(seed=55, Sq=8, Sk=8)
    kw = {"block_size": 2, "top_k": 2, "causal": True}
    res = rt.launch(_artifact("tessera.msa_sparse_attention", ["Q", "K", "V"], kw),
                    (Q, K, V))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    ref = attention.msa_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-4, atol=2e-4)


@pytest.mark.hardware_rocm
def test_rocm_deepseek_sparse_attention_native_gpu_matches_reference_on_hardware():
    import tessera as ts

    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    Q, K, V = _qkv(seed=59, Hq=2, Hkv=2, Sq=8, Sk=8, D=4, Dv=4)
    kw = {"window_size": 4, "block_size": 2, "top_k": 2, "causal": True}
    res = rt.launch(
        _artifact("tessera.deepseek_sparse_attention", ["Q", "K", "V"], kw),
        (Q, K, V),
    )

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    ref = ts.ops.deepseek_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(res["output"], ref, rtol=2e-4, atol=2e-4)


@pytest.mark.compiler_tool
def test_rocm_block_sparse_attention_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.block_sparse_attention"() {name = "bsa"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-block-sparse-attn-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout


@pytest.mark.compiler_tool
def test_rocm_block_sparse_attention_tiled_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.block_sparse_attention_tiled"() {name = "bsat"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-block-sparse-attn-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout


@pytest.mark.compiler_tool
def test_rocm_block_sparse_topk_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.block_sparse_topk_select"() {name = "btopk"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-block-sparse-topk-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout


@pytest.mark.hardware_rocm
def test_rocm_large_topk_cooperative_matches_serial_on_hardware():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    rng = np.random.default_rng(61)
    scores = rng.standard_normal((1, 2, 32, 2048), dtype=np.float32)
    kwargs = {
        "top_k": 8, "block_size": 16, "causal": True,
        "q_positions": np.arange(32, dtype=np.int64) * 64,
    }
    serial = rt._rocm_block_sparse_topk_select_native(
        scores, np, cooperative=False, **kwargs)
    cooperative = rt._rocm_block_sparse_topk_select_native(
        scores, np, cooperative=True, **kwargs)

    np.testing.assert_array_equal(cooperative, serial)
