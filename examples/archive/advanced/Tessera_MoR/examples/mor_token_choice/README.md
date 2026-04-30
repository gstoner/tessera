# examples/mor_token_choice — Tessera Mixture‑of‑Recursions Test App (Token‑Choice)

A **pure Tessera-language** MoR sample usable as a test and a runnable example.

## Highlights
- Router assigns a **discrete depth** per token (`1..S`).
- Shared **recursion block** (attention+MLP) reused across steps.
- **KV cache** policy selectable: `recursion` or `share_first`.
- Emits `MOR_DEPTH_COUNTS` tag for CI assertions.

## Quick run
```bash
bash examples/mor_token_choice/scripts/run_example.sh
```

## Pass pipeline (suggested)
```
-tessera-mor-route-assign -tessera-mor-kv-plan -tessera-mor-depth-batching -canonicalize -cse
```

## Files
- `mor_token_choice.tsr` — Tessera source
- `mlir/mor_token_choice.mlir` — reference/golden Target-IR skeleton
- `tests/mor_token_choice.mlir.test` — FileCheck test
- `scripts/run_example.sh` — tiny run helper
- `CMakeLists.txt` — lit wiring (optional)
