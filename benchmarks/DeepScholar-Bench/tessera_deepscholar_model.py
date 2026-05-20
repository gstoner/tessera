"""CPU smoke benchmark for the DeepScholar-Bench concept.

This file intentionally uses only the current public Tessera surface:

* ``@tessera.jit``
* ``tessera.ops.matmul``
* ``tessera.ops.softmax``
* ``tessera.ops.layer_norm``

It is not a claim that the original LOTUS/DeepScholar research stack is
ported.  The goal is narrower and load-bearing: keep this benchmark
directory runnable while preserving a JSON proof that the current compiler
can build a small research-synthesis scoring kernel and execute the CPU
reference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import tessera as ts  # noqa: E402


KERNEL_SOURCE = """\
def deepscholar_score(query, sources, w_q, w_s):
    q = ts.ops.layer_norm(ts.ops.matmul(query, w_q))
    s = ts.ops.layer_norm(ts.ops.matmul(sources, w_s))
    logits = ts.ops.matmul(q, s.T)
    return ts.ops.softmax(logits, axis=-1)
"""


@dataclass(frozen=True)
class SmokeConfig:
    dim: int = 32
    hidden: int = 16


def _text_embedding(text: str, dim: int) -> np.ndarray:
    """Hash-token embedding used as a deterministic source/text proxy."""

    out = np.zeros((dim,), dtype=np.float32)
    tokens = [tok.strip(".,;:!?()[]{}").lower() for tok in text.split()]
    for tok in tokens:
        if not tok:
            continue
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        out[idx] += sign
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def _kernel():
    namespace = {"ts": ts}
    exec(KERNEL_SOURCE, namespace)
    return ts.jit(namespace["deepscholar_score"], source=KERNEL_SOURCE)


def run_smoke(config: SmokeConfig | None = None) -> dict[str, Any]:
    cfg = config or SmokeConfig()
    query_text = "efficient citation grounded synthesis for long context research"
    source_texts = [
        "retrieval augmented related work generation with citation precision",
        "long context attention and source attribution for research synthesis",
        "operator benchmarks and compiler artifacts for tensor programs",
        "unrelated cooking recipe with no academic citation evidence",
    ]

    query = _text_embedding(query_text, cfg.dim).reshape(1, cfg.dim)
    sources = np.stack([_text_embedding(text, cfg.dim) for text in source_texts])
    rng = np.random.default_rng(7)
    w_q = rng.normal(scale=0.2, size=(cfg.dim, cfg.hidden)).astype(np.float32)
    w_s = rng.normal(scale=0.2, size=(cfg.dim, cfg.hidden)).astype(np.float32)

    kernel = _kernel()
    start = time.perf_counter()
    scores = np.asarray(kernel(query, sources, w_q, w_s), dtype=np.float32)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    artifact = kernel.runtime_artifact()

    q_ref = ts.ops.layer_norm(ts.ops.matmul(query, w_q))
    s_ref = ts.ops.layer_norm(ts.ops.matmul(sources, w_s))
    ref = ts.ops.softmax(ts.ops.matmul(q_ref, s_ref.T), axis=-1)
    max_abs_err = float(np.max(np.abs(scores - ref)))

    metadata = artifact.metadata or {}
    return {
        "schema": "tessera.deepscholar_smoke.v1",
        "benchmark": "DeepScholar-Bench",
        "query_count": 1,
        "source_count": len(source_texts),
        "operator_chain": ["matmul", "layer_norm", "matmul", "layer_norm", "matmul", "softmax"],
        "compiler": {
            "frontend": "tessera.jit",
            "compiler_path": metadata.get("compiler_path", "unknown"),
            "runtime_status": metadata.get("runtime_status", "unknown"),
            "execution_kind": metadata.get("execution_kind", "unknown"),
            "artifact_hash": artifact.artifact_hash,
            "artifact_levels": {
                "graph": bool(artifact.graph_ir),
                "schedule": bool(artifact.schedule_ir),
                "tile": bool(artifact.tile_ir),
                "target": bool(artifact.target_ir),
            },
        },
        "correctness": {
            "max_abs_err": max_abs_err,
            "passed": bool(max_abs_err <= 1e-6),
        },
        "profile": {
            "cpu_wall_ms": elapsed_ms,
        },
        "scores": scores.round(6).tolist(),
        "status": "reference" if metadata.get("execution_kind") == "reference_cpu" else "artifact",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    args = parser.parse_args(argv)

    payload = run_smoke(SmokeConfig(dim=args.dim, hidden=args.hidden))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if payload["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
