from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "advanced" / "Fast_dLLM_v2"


def _load_example():
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))
    import fast_dllm_v2

    return fast_dllm_v2


def test_fast_dllm_numpy_smoke_and_current_compiler_artifact():
    fast_dllm = _load_example()
    cfg = fast_dllm.tiny_config()
    model = fast_dllm.FastDLLMNumpy(cfg, seed=0)
    prompt = np.arange(8, dtype=np.int64) % cfg.vocab_size

    result = model.decode(prompt)

    assert result.tokens.shape == (cfg.branch_count, prompt.size + cfg.decode_steps)
    assert result.confidences.shape == (cfg.branch_count, cfg.decode_steps)
    assert result.cache_blocks.shape == (cfg.branch_count, cfg.decode_steps, cfg.block_tokens)
    assert 0 <= result.accepted_prefix <= cfg.decode_steps
    assert np.isfinite(result.cache_blocks).all()

    bundle = fast_dllm.compile_toy_graph(target="apple_cpu")

    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert bundle.target_ir is not None
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text


def test_fast_dllm_graph_ir_carries_decode_metadata_and_return():
    fast_dllm = _load_example()

    text = fast_dllm.build_toy_graph_ir().to_mlir()

    assert "func.func @fast_dllm_v2_confidence_decode_step" in text
    assert "tessera.decode.branches" in text
    assert "tessera.kv.block_tokens" in text
    assert "tessera.softmax" in text
    assert "return %norm1 : tensor<32x64xf32>" in text
