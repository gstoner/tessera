#!/usr/bin/env python3
"""Run Fast dLLM v2 against the current Tessera compiler."""

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

from fast_dllm_v2 import FastDLLMNumpy, build_toy_graph_ir, compile_toy_graph, tiny_config


def main() -> int:
    cfg = tiny_config()
    model = FastDLLMNumpy(cfg, seed=0)
    prompt = np.arange(8, dtype=np.int64) % cfg.vocab_size
    result = model.decode(prompt)

    assert result.tokens.shape == (cfg.branch_count, prompt.size + cfg.decode_steps)
    assert result.confidences.shape == (cfg.branch_count, cfg.decode_steps)
    assert result.cache_blocks.shape == (cfg.branch_count, cfg.decode_steps, cfg.block_tokens)
    assert 0 <= result.accepted_prefix <= cfg.decode_steps
    assert np.isfinite(result.cache_blocks).all()

    graph_text = build_toy_graph_ir(cfg).to_mlir()
    assert "tessera.softmax" in graph_text
    assert "tessera.rmsnorm_safe" in graph_text

    bundle = compile_toy_graph(target="apple_cpu")
    assert bundle.target_ir is not None
    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text

    print(
        "OK fast_dllm tiny:",
        tuple(result.tokens.shape),
        "accepted",
        result.accepted_prefix,
        bundle.request.target,
        bundle.execution_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
