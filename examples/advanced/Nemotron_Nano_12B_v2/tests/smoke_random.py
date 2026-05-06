#!/usr/bin/env python3
"""Run the Nemotron Nano sample against the current Tessera compiler.

This smoke intentionally avoids PyTorch so it can run in the repository venv.
It validates:

* tiny NumPy reference forward path for the M/*/- hybrid stack
* Graph IR object construction and verification
* Graph -> Schedule -> Tile -> Apple Target IR artifact plumbing
"""

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

from nemotron_nano import NemotronNanoNumpy, build_toy_graph_ir, compile_toy_graph, tiny_config


def main() -> int:
    cfg = tiny_config()
    model = NemotronNanoNumpy(cfg, seed=0)
    input_ids = (np.arange(32, dtype=np.int64).reshape(2, 16) * 7) % cfg.vocab_size
    logits = model.forward(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size), logits.shape
    assert np.isfinite(logits).all()

    graph = build_toy_graph_ir(cfg)
    graph_text = graph.to_mlir()
    assert "tessera.matmul" in graph_text
    assert "tessera.rmsnorm_safe" in graph_text

    bundle = compile_toy_graph(target="apple_cpu")
    assert bundle.target_ir is not None
    assert bundle.runtime_status == "ready"
    assert bundle.execution_mode == "cpu_accelerate"
    assert "tessera_apple.cpu.accelerate_gemm" in bundle.target_ir.text

    print("OK nemotron tiny:", tuple(logits.shape), bundle.request.target, bundle.execution_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
