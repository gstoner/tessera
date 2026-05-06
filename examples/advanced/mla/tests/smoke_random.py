#!/usr/bin/env python3
"""Run FlashMLA against the current Tessera compiler."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
for path in (ROOT, REPO / "python"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from mla import MultiLatentAttentionNumpy, build_toy_graph_ir, compile_toy_graph, tiny_config


def main() -> int:
    cfg = tiny_config()
    rng = np.random.default_rng(0)
    hidden = rng.normal(0.0, 0.02, size=(cfg.batch_size, cfg.seq_len, cfg.model_dim)).astype(np.float32)
    result = MultiLatentAttentionNumpy(cfg, seed=1).forward(hidden)

    assert result.output.shape == hidden.shape
    assert result.kv_latent.shape == (cfg.batch_size, cfg.seq_len, cfg.latent_dim)
    assert result.attn_probs.shape == (cfg.batch_size, cfg.num_q_heads, cfg.seq_len, cfg.seq_len)
    assert result.kv_cache_reduction > 0.0
    assert np.isfinite(result.output).all()

    graph_text = build_toy_graph_ir(cfg).to_mlir()
    assert "tessera.softmax" in graph_text
    assert "tessera.mla.latent_dim" in graph_text

    bundle = compile_toy_graph(target="apple_cpu")
    assert bundle.target_ir is not None
    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text

    print(
        "OK mla tiny:",
        tuple(result.output.shape),
        "kv_reduction",
        round(result.kv_cache_reduction, 4),
        bundle.request.target,
        bundle.execution_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
