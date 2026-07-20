# Fast-dLLM v2 -> Tessera Mapping

This example contains a current-compiler Fast dLLM v2 smoke path plus the
original mapping notes for a full block-wise approximate KV cache and
confidence-aware parallel decoding implementation.

Contents
- `docs/Fast_dLLM_to_Tessera.md` — full write‑up and design notes.
- `fast_dllm_v2/` — dependency-light NumPy reference and current Graph IR compiler smoke.
- `ir/fast_dllm_ops.mlir` — parser-valid current-dialect Graph IR tensor skeleton.
- `ir/tests/*.mlir` — parser-valid fixtures for current compiler contracts.
- `runtime/policy_confidence.*` — C++-style pseudocode for the confidence policy + approximate KV block manager.
- `pipelines/pipelines.md` — recommended `tessera-opt` / `tessera-compile` invocations and pass order.

## Quick Start

From the repository root:

```bash
PYTHONPATH=python /Users/gregorystoner/venv/bin/python \
  examples/advanced/Fast_dLLM_v2/tests/smoke_random.py

PATH="$PWD/build/tools/tessera-opt:/opt/homebrew/opt/llvm@23/bin:$PATH" \
  tessera-opt examples/advanced/Fast_dLLM_v2/ir/fast_dllm_ops.mlir >/tmp/fast_dllm_graph.mlir
```

Expected smoke output:

```text
OK fast_dllm tiny: (4, 14) accepted 0 apple_cpu cpu_accelerate
```

## Current Compiler Contract

The current smoke intentionally separates two concerns:

- `fast_dllm_v2.compiler_smoke` builds Graph IR with the Python object model and
  lowers it through Graph IR -> Schedule IR -> Tile IR -> Apple Target IR artifacts.
- `ir/fast_dllm_ops.mlir` uses quoted registered `tessera.*` ops so `tessera-opt`
  can parse and verify the checked-in textual fixture.

The full Fast dLLM semantics still live in the docs and runtime policy sketch:
branch fork/join, confidence stats, approximate KV pack/read, COW cache pages,
and validated-prefix merge. Those should become native Graph/Schedule/Tile ops
as the compiler grows beyond the current straight-line tensor core.
