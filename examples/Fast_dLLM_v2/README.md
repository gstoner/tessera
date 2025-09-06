# Fast‑dLLM v2 → Tessera Mapping (Starter Package)

This starter contains a concrete mapping plan for **Fast‑dLLM v2** into the Tessera Programming Model, plus example IR snippets, a confidence‑aware parallel decoding policy stub, and suggested `tessera-opt` pipelines.

Contents
- `docs/Fast_dLLM_to_Tessera.md` — full write‑up and design notes.
- `ir/fast_dllm_ops.mlir` — end‑to‑end example (Graph IR → Schedule IR → Tile IR stubs).
- `ir/tests/*.mlir` — FileCheck‑style examples for block‑wise approximate KV Cache and parallel decoding forks/joins.
- `runtime/policy_confidence.*` — C++-style pseudocode for the confidence policy + approximate KV block manager.
- `pipelines/pipelines.md` — recommended `tessera-opt` / `tessera-compile` invocations and pass order.

> This is a *starter* drop: it is intentionally minimal and aligned to your house style so we can iterate quickly.
