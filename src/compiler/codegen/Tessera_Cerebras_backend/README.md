# Tessera → Cerebras WSE-3 Starter (v1)

This is a **scaffold** that shows how to target the Cerebras SDK from the Tessera Programming Model.
It includes:
- a small **target adapter skeleton** (`tessera/targets/cerebras/`) with lowering hooks,
- a **bank-aware vectorizer** stub,
- a **layout planner** that partitions a 2D tile grid into rectangular regions,
- **example kernels** (GEMM and tiny FlashAttention) in CSL-like scaffolding,
- **host launch stubs** that illustrate `SdkRuntime` usage (wrapped in `try/except` so the files run even without the SDK).

> Status: scaffolding and templates. You will need the Cerebras SDK to compile and run the `.csl` kernels.

## Quick Start

1. Install the Cerebras SDK (see https://sdk.cerebras.net).
2. From this bundle's root, review `examples/gemm/` and `examples/flashattn_tiny/`.
3. Use your SDK-provided compiler/launcher to build the `.csl` and invoke via the host stubs.
4. Integrate this adapter into Tessera by pointing your build to `tessera/targets/cerebras` and calling:

```python
from tessera.targets.cerebras import compile_to_cerebras, plan_layout, emit_csl
artifacts = compile_to_cerebras(tessera_module, execution_mode="pipeline")  # or "weight_streaming"
```

## What’s inside

- `tessera/targets/cerebras/lowering.py` — maps Tessera IR ops → CSL kernel templates
- `tessera/targets/cerebras/layout.py` — rectangle partitioner over a tile grid → `SdkLayout`-like JSON
- `tessera/targets/cerebras/bank_vectorizer.py` — helper to pick vector widths that avoid SRAM bank conflicts
- `tessera/targets/cerebras/runtime.py` — host-launch helpers with graceful fallbacks
- `tessera/targets/cerebras/csl_codegen.py` — Jinja-style emission for CSL-like code
- `examples/gemm/` — GEMM kernel scaffold + host launcher
- `examples/flashattn_tiny/` — FlashAttention(very small) scaffold + host launcher

> NOTE: The `.csl` here is **scaffolding** intended to be a starting point. Adjust to the real CSL syntax from the SDK examples/tutorials you have installed.

---

Generated: 2025-09-03 00:23:01


## Build (CMake)

```bash
cmake -S . -B build -DTESSERA_ENABLE_MLIR=OFF
cmake --build build -j
```

If you have LLVM/MLIR installed:
```bash
cmake -S . -B build -DTESSERA_ENABLE_MLIR=ON -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --build build -j
```

Tools produced:
- `tessera-cerebras-codegen` — stub that would emit CSL + layout.json
- `tessera-cerebras-opt` — stub "opt" that wires pass factories

## Tests (lit)

If you have `lit`:
```bash
python -m lit -sv tests
```

Generated: 2025-09-03 00:40:48


## v5 additions (2025-09-03 00:50:09)
- Added **Matmul** op in `ttarget` and `cerebras` + lowering.
- Added **extra rewrite patterns**: memcpy no-op fold, load/store fuse, route dedup.
- Implemented **real emit**: writes `layout.json` (regions/routes) and a tiny `out.csl` based on ops.
- Test now asserts files appear under **%t** (lit temp directory).


## v6 additions
- Richer `.csl` emission grouped by region blocks.
- Each `load_sram`/`store_sram` is accompanied by a normalized `memcpy(dst_space,src_space)` line.
- `cerebras.memcpy` now prints `memcpy(dst_space,src_space)` explicitly.
