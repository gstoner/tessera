from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "advanced" / "Nemotron_Nano_12B_v2"


def _load_example():
    if str(EXAMPLE) not in sys.path:
        sys.path.insert(0, str(EXAMPLE))
    import nemotron_nano

    return nemotron_nano


def test_nemotron_numpy_smoke_and_current_compiler_artifact():
    nemotron = _load_example()
    cfg = nemotron.tiny_config()
    model = nemotron.NemotronNanoNumpy(cfg, seed=0)
    input_ids = (np.arange(32, dtype=np.int64).reshape(2, 16) * 7) % cfg.vocab_size

    logits = model.forward(input_ids)

    assert logits.shape == (2, 16, cfg.vocab_size)
    assert np.isfinite(logits).all()

    bundle = nemotron.compile_toy_graph(target="apple_cpu")

    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert bundle.target_ir is not None
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text


def test_nemotron_graph_ir_emits_result_return_for_mlir_validation():
    nemotron = _load_example()

    text = nemotron.build_toy_graph_ir().to_mlir()

    assert "func.func @nemotron_nano_tiny_m_star_dash" in text
    assert "return %norm : tensor<32x64xf32>" in text
    assert "tessera.rmsnorm_safe" in text
