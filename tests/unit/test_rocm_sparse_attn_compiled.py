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
