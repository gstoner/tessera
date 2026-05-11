---
status: Informative
classification: Reference / Kernel Inventory
authority: Companion to `src/compiler/codegen/Tessera_Metalium_Backend/` Phase 7/8 work
last_updated: 2026-05-11
---

# Tenstorrent Metalium Kernel Inventory

> Hardware-free reference recording every kernel the Tessera Metalium
> backend currently lowers (Phase 7 + Sprint I-1/I-2). Companion to
> `docs/apple_gpu_kernel_inventory.md` for parallel coverage tracking.

This document is the **authoritative kernel inventory** for the
`tessera_metalium` Target IR dialect. It captures:

1. The **RISC-V grid mapping** used to dispatch kernels across the
   Wormhole/Blackhole RISC-V tile mesh.
2. The **shipped kernel surface** (matmul + DMA from Phase 7; softmax /
   LayerNorm / RMSNorm from Sprint I-1).
3. The **dtype matrix** including the planned/gated `bfp8` / `bfp4`
   Tenstorrent block-FP formats (per
   `docs/reference/tessera_tensor_attributes.md`).
4. The **execution gates** — Metalium kernels are `artifact_only` until
   real Wormhole/Blackhole hardware lights up.

---

## 1. RISC-V grid mapping

A Tenstorrent device is organized as a 2-D grid of **Tensix cores**.
Each Tensix core contains 5 RISC-V "baby cores" (BRISC, NCRISC, two
TRISC compute cores, plus a packetizer core) plus on-core SRAM. DRAM is
attached via a separate set of cores on the periphery of the grid.

Tessera's `tessera_metalium` dialect maps the abstract tile space onto
the grid via the `CoreRangeAttr` attribute (defined in
`TesseraMetaliumOps.td`):

```mlir
#tessera_metalium.core_range<[0, 0]..[7, 7]>
```

— an inclusive 2-D rectangle `[x0, y0]..[x1, y1]`. The compiler infers
the rectangle from the launched tile space; the Metalium runtime
dispatches one kernel instance per Tensix core in the range.

### Per-core responsibilities

| RISC-V core | Role | Tessera dispatch |
|---|---|---|
| **BRISC** | DRAM ↔ NoC reads, host coordination | Origin of `tessera_metalium.dma` ops with `direction = "dram_to_sram"` |
| **NCRISC** | NoC ↔ DRAM writes | Origin of `tessera_metalium.dma` ops with `direction = "sram_to_dram"` |
| **TRISC0 / TRISC1** | Matrix compute (FMA tile engine) | Executes `tessera_metalium.matmul` |
| **Packetizer** | Network packet routing | Implicit; managed by Metalium runtime |

### Memory hierarchy

```
DRAM (host-attached, ~64 GB)
  ↓ [tessera_metalium.dma direction="dram_to_sram"]
SRAM (on-core, ~1.5 MB per Tensix)
  ↓ [tessera_metalium.load — SRAM → registers]
Vector registers (per RISC-V core)
  ↓ [tessera_metalium.matmul — tile-local FMA]
```

The compiler is responsible for ensuring SRAM doesn't overflow.
`MetaliumBufferPlanner` (C++ side) reserves per-core SRAM regions and
fails the lowering early if the requested tile shape doesn't fit.

---

## 2. Shipped kernel inventory

All entries are **artifact-only** (lit-testable IR; execution gated on
real Wormhole/Blackhole hardware). Lit fixtures live under
`src/compiler/codegen/Tessera_Metalium_Backend/test/metalium/`.

### Phase 7 — Base kernels

| Op | Lowering pattern | Lit fixture | dtypes |
|---|---|---|---|
| `tessera.tile.gemm` | → `tessera_metalium.matmul` with `tile_shape` attr | `gemm_to_matmul_opt.mlir` | bf16, fp32 |
| `tessera.tile.copy` (DRAM↔SRAM) | → `tessera_metalium.dma` with `direction` + `element_size_bytes` attrs | `tile_to_dma_opt.mlir` | bf16, fp32 |
| no-silent-erase contract | Verifier — rejects ops that would silently elide DMA boundaries | `no_silent_erase.mlir` | n/a |

### Sprint I-1 — Reduction / normalization (2026-05-11)

| Op | Decomposition | Lit fixture | dtypes |
|---|---|---|---|
| `tessera.tile.softmax` | DMA → row-local reduce (via matmul) → exp/scale → DMA | `softmax_to_metalium.mlir` | bf16 |
| `tessera.tile.layer_norm` | 2× row-local reduce (Σx, Σ(x−μ)²) → elementwise compose → DMA | `layer_norm_to_metalium.mlir` | bf16 |
| `tessera.tile.rmsnorm` | 1× row-local reduce (Σx²) → elementwise compose → DMA | `rmsnorm_to_metalium.mlir` | bf16 |

**Why decompose through matmul for reductions?** Metalium has no
dedicated reduce intrinsic — the Tensix FMA engine is matrix-shaped.
A row-local reduction lowers as a `1×N · N×1` matmul against a
broadcast identity vector, which the FMA engine already supports
efficiently. This is the same idiom DeepSeek's `dpsk-flash-mla` uses
for the softmax denominator pass on Hopper.

---

## 3. Dtype matrix (per Tenstorrent block-FP family)

Per `docs/reference/tessera_tensor_attributes.md`, Tenstorrent's
**block-floating-point** formats (`bfp8` / `bfp4` / `blockfp8` /
`blockfp4`) are in the **planned/gated** set. Registry entries that
reference them must declare `metadata.dtype_status="planned_gated"`.

| Family | Canonical names | Status under ROCm 7.2.3 / Wormhole / Blackhole |
|---|---|---|
| Standard FP | `fp32`, `bf16` | shipped under `artifact_only` |
| Tenstorrent block-FP | `bfp8`, `bfp4` | **planned/gated** — separate `metalium_blockfp` target in the backend manifest, populated lazily via `_METALIUM_PLANNED_GATED_KERNELS` |
| Alias-reject | `blockfp8`, `blockfp4` | rejected by `tessera.dtype.canonicalize_dtype` (do not alias to OCP FP8/FP4 or AMD MXFP) |

The block-FP entries are surfaced separately so the audit walker
(`backend_manifest.audit_backend_dtypes`) correctly classifies them as
`planned_gated` rather than `unknown`. Sprint I-2 wires this into the
existing `tessera.compiler.primitive_coverage.audit_canonical_dtypes`
walker.

---

## 4. Execution gates

| Gate | What it means | When it lifts |
|---|---|---|
| `artifact_only` (current state) | Target IR is well-formed; lit fixtures validate the lowering; no actual Tensix execution | Once a Wormhole or Blackhole grid is available + TT-Metal SDK is on the dev box |
| `compileable` (Sprint I follow-up) | `tt-metalium-compiler` can lower the artifact to a deployable binary; **without execution** | Requires TT-Metal SDK 0.40+ on the dev box (no hardware needed) |
| `executable` | The deployable binary runs on a real device; numerical correctness verified vs. CPU reference | Requires Wormhole/Blackhole hardware |
| `fused` | Performance characterized; achieves ≥85% of roofline TFLOPS | Requires hardware + perf tuning sprint |

Today the Tessera Metalium backend sits firmly at `artifact_only`.
Lit-testable lowering of matmul + DMA + softmax + LayerNorm + RMSNorm
is achievable hardware-free (and tested in
`tests/unit/test_metalium_backend_inventory.py`).

---

## 5. Source map

| Component | Path |
|---|---|
| ODS dialect definition | `src/compiler/codegen/Tessera_Metalium_Backend/include/Tessera/Target/Metalium/TesseraMetaliumOps.td` |
| Pass implementations | `src/compiler/codegen/Tessera_Metalium_Backend/lib/` (C++) |
| Lit fixtures (Phase 7 base) | `src/compiler/codegen/Tessera_Metalium_Backend/test/metalium/{gemm_to_matmul,tile_to_dma,no_silent_erase}_opt.mlir` |
| Lit fixtures (Sprint I-1) | `src/compiler/codegen/Tessera_Metalium_Backend/test/metalium/{softmax,layer_norm,rmsnorm}_to_metalium.mlir` |
| Backend manifest entries | `python/tessera/compiler/backend_manifest.py` — `_METALIUM_KERNELS` + `_METALIUM_PLANNED_GATED_KERNELS` |
| Capability registry | `python/tessera/compiler/capabilities.py` — `"metalium"` entry under `TARGET_CAPABILITIES` |
| Buffer planning | `src/compiler/codegen/Tessera_Metalium_Backend/include/Tessera/Target/Metalium/MetaliumBufferPlanner.h` |
| Pipeline alias | `tessera-lower-to-metalium` (registered in `tessera-opt`) |

---

## 6. Roadmap — what's hardware-free vs. blocked

### Hardware-free (this kernel inventory can grow further)
- Add `tessera.tile.gelu` → `dma + elementwise` Metalium lowering + lit fixture
- Add `tessera.tile.matmul_softmax_fused` → single-pass `dma + matmul + softmax` chain (mirrors Apple GPU's `matmul→softmax` fusion)
- Add `tessera.tile.flash_attn` → tile-local online-softmax matmul chain
- Expand BFP support: `bfp8`/`bfp4` MFMA-style matmul lit fixtures (still planned/gated)

### Blocked on Wormhole / Blackhole hardware
- End-to-end execution + numerical correctness
- Performance characterization (TFLOPS, MFU)
- Multi-grid scaling tests (4×4 → 8×8 Tensix grids)
- Block-FP `bfp8` / `bfp4` performance vs. bf16 baseline
