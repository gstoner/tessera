import time

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.cache import LatentKVCacheHandle
from tessera.stdlib import attention


def _weights(seed=41):
    rng = np.random.default_rng(seed)
    hidden, heads, d_nope, d_rope, d_v, d_c = 8, 2, 3, 2, 4, 5
    return attention.MLAWeights(
        w_dkv=(rng.standard_normal((hidden, d_c)) * 0.12).astype(np.float32),
        w_uk=(rng.standard_normal((d_c, heads * d_nope)) * 0.11).astype(np.float32),
        w_uv=(rng.standard_normal((d_c, heads * d_v)) * 0.10).astype(np.float32),
        w_q=(rng.standard_normal((hidden, heads * (d_nope + d_rope))) * 0.09).astype(np.float32),
        w_kr=(rng.standard_normal((hidden, d_rope)) * 0.08).astype(np.float32),
        num_heads=heads,
        d_nope=d_nope,
        d_rope=d_rope,
        d_v=d_v,
    )


def _caches(weights, *, max_seq=16):
    return (
        LatentKVCacheHandle(weights.d_c, max_seq=max_seq, dtype="fp32"),
        LatentKVCacheHandle(weights.d_rope, max_seq=max_seq, dtype="fp32"),
    )


def _seed_prompt(latent_cache, rope_cache, prompt, weights):
    c, kr = attention.compress_latent(prompt, weights)
    latent_cache.append(c)
    rope_cache.append(kr)


def _artifact():
    names = ["x_t", "latent_cache", "rope_cache", "weights"]
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_exotic_attn_compiled",
        "executable": True,
        "arg_names": names,
        "output_name": "o",
        "ops": [{
            "op_name": "tessera.mla_decode_step",
            "result": "o",
            "operands": names,
            "kwargs": {"absorb": True},
        }],
    })


def _case(seed=42):
    rng = np.random.default_rng(seed)
    weights = _weights(seed + 1)
    prompt = (rng.standard_normal((4, weights.w_dkv.shape[0])) * 0.2).astype(np.float32)
    x_t = (rng.standard_normal((2, weights.w_dkv.shape[0])) * 0.2).astype(np.float32)
    ref_lat, ref_rope = _caches(weights)
    rt_lat, rt_rope = _caches(weights)
    _seed_prompt(ref_lat, ref_rope, prompt, weights)
    _seed_prompt(rt_lat, rt_rope, prompt, weights)
    return weights, x_t, (ref_lat, ref_rope), (rt_lat, rt_rope)


def test_rocm_mla_decode_step_matches_stdlib_and_cache_side_effects():
    weights, x_t, ref_caches, rt_caches = _case()
    ref_lat, ref_rope = ref_caches
    got_lat, got_rope = rt_caches

    expected = attention.mla_decode_step(x_t, ref_lat, ref_rope, weights)
    res = rt.launch(_artifact(), (x_t, got_lat, got_rope, weights))

    assert res["ok"], res.get("reason")
    assert res["compiler_path"] == "rocm_exotic_attn_compiled"
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    np.testing.assert_allclose(res["output"], expected, rtol=2e-5, atol=2e-5)
    assert got_lat.current_seq == ref_lat.current_seq
    assert got_rope.current_seq == ref_rope.current_seq
    np.testing.assert_allclose(got_lat.read(0, got_lat.current_seq),
                               ref_lat.read(0, ref_lat.current_seq),
                               rtol=0, atol=0)
    np.testing.assert_allclose(got_rope.read(0, got_rope.current_seq),
                               ref_rope.read(0, ref_rope.current_seq),
                               rtol=0, atol=0)


@pytest.mark.performance
def test_dk1_rocm_mla_decode_step_perf_baseline_is_bounded():
    weights, x_t, _, _ = _case(seed=44)
    art = _artifact()

    def direct():
        lat, rope = _caches(weights)
        _seed_prompt(lat, rope, np.zeros((4, weights.w_dkv.shape[0]), np.float32), weights)
        return attention.mla_decode_step(x_t, lat, rope, weights)

    def launched():
        lat, rope = _caches(weights)
        _seed_prompt(lat, rope, np.zeros((4, weights.w_dkv.shape[0]), np.float32), weights)
        return rt.launch(art, (x_t, lat, rope, weights))

    direct_vals = []
    launch_vals = []
    for _ in range(7):
        t0 = time.perf_counter()
        direct()
        direct_vals.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        launched()
        launch_vals.append((time.perf_counter() - t0) * 1000.0)
    assert float(np.median(launch_vals)) < max(75.0, float(np.median(direct_vals)) * 4.0)


@pytest.mark.hardware_rocm
def test_rocm_mla_decode_step_native_gpu_matches_reference_on_hardware():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    weights, x_t, ref_caches, rt_caches = _case(seed=46)
    expected = attention.mla_decode_step(x_t, ref_caches[0], ref_caches[1], weights)
    res = rt.launch(_artifact(), (x_t, rt_caches[0], rt_caches[1], weights))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    np.testing.assert_allclose(res["output"], expected, rtol=2e-4, atol=2e-4)


@pytest.mark.compiler_tool
def test_rocm_mla_absorb_decode_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.mla_absorb_decode"() {name = "mla"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-mla-absorb-decode-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
