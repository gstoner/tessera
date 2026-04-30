# Tessera Energy-Based Transformer (EBT) — v1

This drop maps the **Energy‑Based Transformer (EBT)** architecture into the Tessera Programming Model.

> Sources: Energy‑Based Transformers site, paper, and reference PyTorch code (see docs/EBT_refs.md).

## What’s inside
- `docs/EBT_in_Tessera.md` — design + mapping across Tessera IR levels (Graph/Schedule/Tile/Target).
- `models/ebt/ir/ebt_ir_samples.mlir` — canonical IR snippets for energy evaluation, inner‑loop “thinking”, and self‑verification.
- `models/ebt/passes/` — pass scaffolds (CMake + headers) for `tessera-ebt-canonicalize`, `tessera-ebt-lower` pipelines.
- `models/ebt/runtime/ebt_runner.h/.cc` — a tiny CPU/GPU‑agnostic runner for iterative sampling (placeholder; shows APIs).
- `models/ebt/tests/` — FileCheck tests for the IR patterns and pipelines.
- `cmake/FindTesseraEBT.cmake` — optional helper for integration.
- `tools/lit/lit.cfg.py` — run `llvm-lit` on the tests directory.

## Quick start
```bash
# From repo root after unzipping under tessera/models/
lit models/ebt/tests
# or
mlir-opt models/ebt/ir/ebt_ir_samples.mlir   -tessera-ebt-canonicalize -tessera-ebt-lower | FileCheck models/ebt/tests/ebt_lowering.check
```
