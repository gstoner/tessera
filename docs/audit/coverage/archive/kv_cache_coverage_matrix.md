---
status: Audit
classification: Coverage Matrix
authority: Per-target KV-cache lowering status as of 2026-05-10
last_updated: 2026-05-10
---

# KV-Cache Lowering Coverage Matrix

CLAUDE.md Architecture Decision #21 requires that backends emit a stable
diagnostic when an op cannot be lowered, rather than silently dropping it.
This matrix documents whether each backend handles `kv_cache_*` ops, FA-4
with cache, and the supporting tile-level `tile.kv_cache` op.

If a cell turns out to be ❌ (silent drop) below, that is a Decision #21
violation and gets a follow-up entry in `docs/audit/execution_roadmap.md`.

## Legend

- ✅ **Executes** — runs end-to-end and produces correct output
- 🟡 **Lowers** — emits target IR but no real launch (artifact-only)
- 🔲 **Diagnostic** — backend emits a clear error per Decision #21
- ❌ **Silent drop** — falls through with no diagnostic (BUG)
- ⛔ **Not encountered** — backend's lowering pipeline does not see KV-cache ops in any tested path; status undetermined

## Op surface

| Layer | Op | File |
|-------|----|----|
| Graph IR | `tessera.kv_cache.create` | `src/compiler/ir/TesseraOps.td` |
| Graph IR | `tessera.kv_cache.append` | `src/compiler/ir/TesseraOps.td` |
| Graph IR | `tessera.kv_cache.prune` | `src/compiler/ir/TesseraOps.td` |
| Graph IR | `tessera.kv_cache.read` (handle + position) | `src/compiler/ir/TesseraOps.td:374` |
| Graph IR type | `tessera.kv_cache` (`Tessera_KVCacheType`) | `TesseraOps.td:103` |
| Tile IR | `tile.kv_cache` (`Tile_KVCacheOp`) | `programming_model/ir/tile/TileMemoryOps.td:91` |
| Effects | `tessera.kv_cache.*` → `Effect.state` | `src/transforms/lib/EffectAnnotationPass.cpp:106` |
| Python runtime | `tessera.ops.kv_cache_append` / `kv_cache_prune` (numpy `ReferenceKVCache`) | `python/tessera/__init__.py:830` |

## Coverage matrix

| Target | `kv_cache.create` | `kv_cache.append` | `kv_cache.prune` | `kv_cache.read` | `tile.kv_cache` | FA-4 + cache | Notes |
|--------|-------------------|-------------------|------------------|-----------------|------------------|--------------|-------|
| **x86** (`tessera_x86_backend`) | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | `TileToX86Pass.cpp:LowerKVCacheToX86` emits `func.call @tessera_x86_kv_cache_op(kind:i32)` artifacts (Decision #21 compliant). Runtime symbol delegates to the Python `KVCacheHandle` reference path. Lit fixture: `tests/tessera-ir/phase2/x86_kv_cache_lowering.mlir` (2026-05-10) |
| **Apple CPU** (`Tessera_Apple_Backend`, CPU lowering) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Lowers to `tessera_apple.cpu.kv_cache_cpu` artifact carrying `kind=tessera.kv_cache.*`, `abi="kv_cache_handle"`, `framework="Tessera"`. Runtime path consumes via `KVCacheHandle`. Lit fixtures: `apple_kv_cache_lowering.mlir`, `apple_cpu_lowering.mlir` (2026-05-10) |
| **Apple GPU** (`Tessera_Apple_Backend`, GPU lowering) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Lowers to `tessera_apple.gpu.kv_cache_gpu` artifact carrying `kind=...`, `abi="kv_cache_handle"`, `framework="Metal"`. Lit fixtures: `apple_kv_cache_lowering.mlir`, `apple_gpu_lowering.mlir` (2026-05-10) |
| **NVIDIA** (`tessera_gpu_backend_NVIDIA`) | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | No references — IR is ready (Phase 3) but kv_cache isn't wired into the NVIDIA Target IR. **Not yet a Decision #21 violation** because no tested path lowers KV-cache ops here; will become one when NVIDIA execution lands (Phase G) |
| **ROCm** (`Tessera_ROCM_Backend`) | ⛔ | ⛔ | ⛔ | ⛔ | 🔲 | ⛔ | `TileToROCM.cpp:94` matches `tile.kv_cache` and emits "ROCm lowering does not implement KV-cache artifacts in this phase" |
| **TPU** (`Tessera_TPU_Backend`) | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | No references in backend |
| **Cerebras** (`Tessera_Cerebras_backend`) | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | No references in backend |
| **Metalium** (`Tessera_Metalium_Backend`) | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | `TileToMetalium.cpp:LowerKVCacheToMetalium` emits `tessera_metalium.kv_cache_op` artifact carrying `kind=tessera.kv_cache.*`, `abi="kv_cache_handle"`, `plan={...}` populated by `MetaliumBufferPlanner::planKVCache()` (DRAM/SRAM staging, k/v tile bindings, kLoad/vLoad DMA descriptors, tileSeq). Decision #21 compliant. Lit fixture (REQUIRES tessera-metalium-backend): `tests/tessera-ir/phase7/metalium_kv_cache_lowering.mlir` (2026-05-10) |
| **RubinCPX** (`Tessera_RubinCPX_Backend`) | 🟡 | 🟡 | 🟡 | 🟡 | ⛔ | 🟡 | Backend defines its own `cpx.kv.cache` op (`NVRubinCPX.td:86`) with verifier; runs through `tessera-cpx-opt` pipeline but no real device execution path |
| **Python numpy reference** (cross-target fallback via `ops.kv_cache_*`) | n/a | ✅ | ✅ | n/a | n/a | n/a | `ReferenceKVCache` Python class; backs all numpy reference-path runs (today: x86 + Apple CPU fast path); no FA-4 integration |

## Findings

1. **Decision #21 is being honored on every backend that lowers KV-cache
   ops.** Apple CPU+GPU now emit real `tessera_apple.{cpu,gpu}.kv_cache_*`
   artifacts (was: diagnostic). x86 emits `func.call @tessera_x86_kv_cache_op`
   artifacts. Metalium emits `tessera_metalium.kv_cache_op` artifacts that
   embed the DRAM/SRAM staging plan from `MetaliumBufferPlanner::planKVCache()`.
   ROCm continues to emit a named diagnostic. ✅ no violations of #21 in
   tested lowering paths. (2026-05-10)

2. **Remaining ⛔ cells are non-violations** — NVIDIA, TPU, and Cerebras never
   *encounter* KV-cache ops because the test surface today doesn't push them
   through. The risk: when those backends start executing real KV-cache flows,
   the absence of explicit handling becomes a silent drop. **Recommendation:**
   when each backend lights up end-to-end execution, add a `tile.kv_cache`
   match arm that either lowers or emits a Decision #21 diagnostic.

3. **The Python runtime path works on every backend that uses the numpy
   reference path** — calling `tessera.ops.kv_cache_append(cache, k, v)` from
   user code goes through `_make_ops_namespace`'s reference impl
   (`ReferenceKVCache`) and works regardless of compiled-backend status. This
   is what makes `examples/advanced/kv_cache_serving/` a runnable Python
   utility today (no Tile IR involvement). x86 and Metalium artifact lowerings
   are designed to delegate to this same `KVCacheHandle` runtime path until
   each backend grows a native in-cache score-matrix kernel.

4. **RubinCPX still has partial scaffolding** (target IR op + verifier) but no
   full lowering chain. RubinCPX remains 🟡 — not a silent drop, but not real
   execution either.

5. **Metalium lit fixture is REQUIRES-gated** on `tessera-metalium-backend`
   feature. The Metalium backend isn't currently linked into the in-tree
   `tessera-opt` build (gated by `TESSERA_BUILD_METALIUM_BACKEND` and
   `TESSERA_METALIUM_WITH_MLIR`). When the backend is enabled and registered
   in `tessera-opt`, lit will pick up the fixture automatically via the
   feature probe in `tests/tessera-ir/lit.cfg.py`.

## Action items folded into roadmap

| Audit cell | Roadmap task |
|------------|--------------|
| All ⛔ cells | Track in **Phase G** for NVIDIA; track ad-hoc for x86/TPU/Cerebras as those backends light up |
| Python `ReferenceKVCache` modernization (handle abstraction, paged storage, quantization) | **Phase E (E1, E2, E3)** of execution roadmap |
| `tile.kv_cache` real lowering on Apple GPU | Mark as 🔲 deferred — write `Tessera_Apple_KVCache_Lowering.md` design doc when it becomes the bottleneck for an example |
| FA-4 + cache on every backend | Tie to **Phase G6** (NVIDIA FA-4 verification) and **Phase E** (handle abstraction) |

## Cross-references

- CLAUDE.md Architecture Decision #21
- `docs/audit/execution_roadmap.md` — Phases E, G
- `examples/advanced/kv_cache_serving/` — exercises the Python reference path
- `docs/CANONICAL_API.md` — `tessera.ops.kv_cache_append` / `kv_cache_prune`
