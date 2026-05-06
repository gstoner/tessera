from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "advanced" / "mla"


def _load_example():
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))
    import mla

    return mla


def test_mla_numpy_smoke_and_current_compiler_artifact():
    mla = _load_example()
    cfg = mla.tiny_config()
    rng = np.random.default_rng(0)
    hidden = rng.normal(0.0, 0.02, size=(cfg.batch_size, cfg.seq_len, cfg.model_dim)).astype(np.float32)

    result = mla.MultiLatentAttentionNumpy(cfg, seed=1).forward(hidden)

    assert result.output.shape == hidden.shape
    assert result.kv_latent.shape == (cfg.batch_size, cfg.seq_len, cfg.latent_dim)
    assert result.attn_probs.shape == (cfg.batch_size, cfg.num_q_heads, cfg.seq_len, cfg.seq_len)
    assert result.kv_cache_reduction > 0.0
    assert np.isfinite(result.output).all()

    bundle = mla.compile_toy_graph(target="apple_cpu")

    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert bundle.target_ir is not None
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text


def test_mla_graph_ir_carries_metadata_and_result_return():
    mla = _load_example()

    text = mla.build_toy_graph_ir().to_mlir()

    assert "func.func @flash_mla_tiny_prefill" in text
    assert "tessera.mla.latent_dim" in text
    assert "tessera.softmax" in text
    assert "return %norm_out : tensor<16x64xf32>" in text
