import numpy as np
import pytest

from tessera import runtime as rt
from tessera.stdlib import dspark


def _weights(vocab=8, hidden=6, seed=0):
    rng = np.random.default_rng(seed)
    return dspark.DSparkWeights(
        token_embedding=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.2,
        hidden_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1,
        token_proj=rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1,
        out_proj=rng.standard_normal((hidden, vocab)).astype(np.float32) * 0.15,
        confidence_proj=rng.standard_normal(hidden).astype(np.float32) * 0.1,
        markov=rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05,
    )


def _artifact(cfg):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_dspark_draft_block_compiled",
        "executable": True,
        "arg_names": ["target_hidden", "prev_tokens", "anchors", "weights"],
        "dspark_config": {
            "num_anchors": cfg.num_anchors,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "confidence_threshold": cfg.confidence_threshold,
        },
        "ops": [{"op_name": "tessera.dspark.draft_block"}],
    })


def test_rocm_dspark_draft_block_runtime_matches_reference_oracle():
    cfg = dspark.DSparkConfig(num_anchors=2, block_size=3, vocab_size=8)
    weights = _weights()
    rng = np.random.default_rng(11)
    target_hidden = rng.standard_normal((2, 7, 6)).astype(np.float32)
    prev_tokens = np.array([1, 5], dtype=np.int64)
    anchors = np.array([0, 3], dtype=np.int64)

    ref = dspark.draft_block_forward(target_hidden, prev_tokens, anchors, weights, cfg)
    res = rt.launch(_artifact(cfg), (target_hidden, prev_tokens, anchors, weights))

    assert res["ok"]
    assert res["compiler_path"] == "rocm_dspark_draft_block_compiled"
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    out = res["output"]
    np.testing.assert_allclose(out["logits"], ref.logits)
    np.testing.assert_allclose(out["confidence_logits"], ref.confidence_logits)
    np.testing.assert_array_equal(out["tokens"], ref.tokens)
    np.testing.assert_allclose(out["hidden"], ref.hidden)


def test_rocm_dspark_draft_block_accepts_weight_mapping_and_infers_static_shape():
    cfg = dspark.DSparkConfig(num_anchors=1, block_size=2, vocab_size=8)
    weights = _weights(seed=2)
    weight_map = {
        "token_embedding": weights.token_embedding,
        "hidden_proj": weights.hidden_proj,
        "token_proj": weights.token_proj,
        "out_proj": weights.out_proj,
        "confidence_proj": weights.confidence_proj,
        "markov": weights.markov,
    }
    rng = np.random.default_rng(13)
    target_hidden = rng.standard_normal((1, 4, 6)).astype(np.float32)
    prev_tokens = np.array([3], dtype=np.int64)
    anchors = np.array([1], dtype=np.int64)
    art = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_dspark_draft_block_compiled",
        "executable": True,
        "block_size": cfg.block_size,
        "arg_names": ["target_hidden", "prev_tokens", "anchors", "weights"],
        "ops": [{"op_name": "tessera.dspark.draft_block"}],
    })

    res = rt.launch(art, {
        "target_hidden": target_hidden,
        "prev_tokens": prev_tokens,
        "anchors": anchors,
        "weights": weight_map,
    })

    assert res["ok"]
    assert res["execution_kind"] in {"native_gpu", "reference_cpu"}
    assert res["output"]["logits"].shape == (1, 1, 2, 8)


def test_rocm_dspark_draft_block_requires_executable_metadata():
    cfg = dspark.DSparkConfig(num_anchors=1, block_size=2, vocab_size=8)
    art = _artifact(cfg)
    art.metadata["executable"] = False
    res = rt.launch(art, ())
    assert not res["ok"]
    assert res["runtime_status"] in {"unimplemented", "missing_backend"}


def test_rocm_dspark_draft_block_native_gpu_matches_reference_on_hardware():
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")

    cfg = dspark.DSparkConfig(num_anchors=2, block_size=3, vocab_size=8)
    weights = _weights(seed=21)
    rng = np.random.default_rng(22)
    target_hidden = rng.standard_normal((2, 7, 6)).astype(np.float32)
    prev_tokens = np.array([2, 4], dtype=np.int64)
    anchors = np.array([1, 3], dtype=np.int64)

    ref = dspark.draft_block_forward(target_hidden, prev_tokens, anchors, weights, cfg)
    res = rt.launch(_artifact(cfg), (target_hidden, prev_tokens, anchors, weights))

    assert res["ok"], res.get("reason")
    assert res["execution_kind"] == "native_gpu"
    out = res["output"]
    np.testing.assert_allclose(out["logits"], ref.logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out["confidence_logits"], ref.confidence_logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out["tokens"], ref.tokens)
    np.testing.assert_allclose(out["hidden"], ref.hidden, rtol=1e-5, atol=1e-5)


def test_rocm_dspark_draft_block_codegen_lowers():
    import subprocess
    from pathlib import Path

    opt = Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt"
    if not opt.is_file():
        pytest.skip("build tessera-opt")
    directive = (
        'module {\n'
        '  "tessera_rocm.dspark_draft_block"() {name = "ds"} : () -> ()\n'
        '}\n')
    low = subprocess.run(
        [str(opt), "-",
         "--pass-pipeline=builtin.module(generate-rocm-dspark-draft-block-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout
