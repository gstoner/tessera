<<<MERGE_START: Tessera_TPU_Backend_Design>>>
# Tessera → Google TPU (OpenXLA) Backend: Design & Base Mapping

> Status: **prototype scaffold**. This document explains how Tessera’s tile abstractions lower to StableHLO/XLA for Cloud TPU **v3/v4** via PJRT, and ships a minimal pass/runner scaffold.

## 0. Scope & Goals
- Produce a **lowering path** from Tessera Target‑IR to **StableHLO** (and XLA HLO) that runs on **TPU** through **PJRT** (`libtpu`).
- Map Tessera’s **tile** & **collective** operations to TPU‑efficient HLO forms (`dot_general`, `convolution`, fusions, and `all_reduce`/`all_gather`/`reduce_scatter`/`collective_permute`).
- Provide **GSPMD/Shardy** annotations from Tessera partition specs.
- Provide a tiny **CMake + mlir-opt plugin** and **PJRT device lister** for TPU bring‑up.
- Keep the surface **BF16‑first** with FP32 accum (INT8 hooks noted).

## 1. OpenXLA pipeline overview (where Tessera fits)
```
Tessera Frontend  →  Tessera Target‑IR  →  (this repo) Lower to StableHLO
                                          ↓
                                      XLA/TPU Backend (HLO passes, GSPMD)
                                          ↓
                                     PJRT (TPU plugin: libtpu)
                                          ↓
                                         TPU
```
**StableHLO** is our interchange IR. **GSPMD/Shardy** carry sharding metadata. **PJRT** provides device/runtime API.

## 2. TPU v3/v4 facts that drive mapping (TL;DR)
- **MXU tile:** v3/v4 TensorCores use **128×128** systolic arrays per MXU; v3 has **two MXUs per core**. Favor `K,N,M` multiples of 128 (padding or packing) and **BF16→FP32 accumulate**.
- **HBM:** v3 cores have **~16 GiB** each; v4 has **32 GiB unified per chip** (two cores). Prefer large, divisible batch/feature dims; keep fusion to maximize on‑chip reuse.
- **Vector regs:** TPU cores expose **8×128** vector lanes; shapes divisible by **8** and **128** tend to schedule well.
- **Collectives:** Use StableHLO `all_reduce`, `all_gather`, `reduce_scatter`, `collective_permute` for DP/TP/PP; annotate shardings for GSPMD.
- **Pods:** v4 pods scale to **4096 chips**; partition specs must compose (data × model × pipeline) with mesh axes names.

> All of the above inform defaults in `passes/TesseraShardingAnnotator.cpp` and tiling in the matmul/conv lowerings.

## 3. Tessera → StableHLO op mapping (initial cut)
| Tessera op               | StableHLO/XLA target                           | Notes |
|---|---|---|
| `tessera.matmul`         | `stablehlo.dot_general`                         | Contract dims = last of LHS & second‑last of RHS; set precision to BF16 with FP32 accum. |
| `tessera.conv{nd}`       | `stablehlo.convolution`                         | Strides/dilations window attrs; prefer NHWC to match TPU layouts. |
| `tessera.add/mul`        | elementwise `stablehlo.add/mul`                 | Prefer **fusion** with producers/consumers to keep data on‑chip. |
| `tessera.bias_gelu`      | `fusion(compute)` or composite                  | Start as composite (bias → tanh‑GELU); consider `custom_call` for fast path later. |
| `tessera.transpose/reshape/slice` | like‑named StableHLO ops               | Canonicalize around `dot_general` and `convolution` to avoid transposes on hot path. |
| `tessera.softmax`        | `stablehlo.reduce` + `exp` + `div` or composite | Use composite to preserve fusion boundaries. |
| `tessera.all_reduce`     | `stablehlo.all_reduce`                          | Add `mhlo.sharding`/Shardy attrs for partitioned dims. |
| `tessera.all_gather`     | `stablehlo.all_gather`                          |  |
| `tessera.reduce_scatter` | `stablehlo.reduce_scatter`                      |  |
| `tessera.permute`        | `stablehlo.collective_permute`                  |  |

### Data types
- **BF16 as default** for compute; **FP32 accum** for MXU paths.
- INT8 (quant) is sketched via StableHLO quantization and `dot_general`/`convolution` with quantized types; wire later.

## 4. Tiling rules for TPU MXU
- Prefer **M,N,K** multiples of **128** (pad if needed). For heads/dmodel, bias towards {M,N} divisible by **128**, K by **128/256** depending on pack.
- Batch × Heads typically shard on **data mesh axis**; sequence sharding often flows with attention block’s `all_gather`/`reduce_scatter` pattern.
- Emit **composites/fusions** around matmul/conv + epilogues (bias, GELU) to keep values in on‑chip buffers.

## 5. Sharding & GSPMD (with Shardy)
- Attach sharding via `mhlo.sharding` string attr or Shardy’s `sdy.tensor_sharding` (preferred forward path).
- Mesh axes example: `{data, model, fsdp}`. Tessera’s `tile.distribute(mesh={"data","model"})` maps to `TensorShardingAttr` and StableHLO collectives inserted by Shardy export passes.
- For incompatible shardings, insert explicit reshard ops (pass provided).

## 6. Runtime & PJRT
- Use **PJRT** to enumerate TPU devices and hand compiled executable blobs to **libtpu**. In this starter we:
  - Build a **device lister** (`runtime/pjrt_runner.cc`) to confirm TPU visibility.
  - Provide a **placeholder** compile‑and‑run hook for StableHLO once you point CMake to OpenXLA headers/libs.
- Environment hints: `PJRT_DEVICE=TPU`, `LIBTPU_INIT_ARGS=--xla_tpu_enable_micro_machine_learning=true` (as needed), `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` (not relevant to TPU).

## 7. What’s in the starter
- **Pass:** `TesseraToStableHLO` – lowers a minimal subset (`matmul`, `add`) and leaves TODOs for attention/conv.
- **Pass:** `TesseraShardingAnnotator` – attaches prototype shardings to HLO ops.
- **Tool:** `tessera-tpu-opt` – mlir‑opt‑style driver to run the passes.
- **Runtime:** `pjrt_runner.cc` – PJRT device discovery stub, ready to grow into load/compile/execute.
- **Examples/tests:** sample StableHLO matmul; lit file with FileCheck.

## 8. Bring‑up recipe
1) Build `tessera-tpu-opt` and run:  
   `tessera-tpu-opt examples/attention_stub.mlir -tessera-lower-to-stablehlo -tessera-annotate-sharding`
2) Inspect StableHLO; verify `dot_general` contracts and collective shape.
3) Run `pjrt_runner --list-devices` on a TPU VM to confirm `libtpu` is visible.
4) (Next) Hook compile‑and‑execute once you point the PJRT includes in CMake.

## 9. Next steps
- Lower **attention** (qkv pack, scaled dot‑product, mask, dropout) with composites/fusions.
- Add **conv** lowering with NHWC defaults.
- Wire **quantized** `dot_general`/`convolution` and int‑epilogues.
- Optional: **TPU v4 embedding** via custom call to TPU embedding libraries; SparseCore hand‑off as a future extension.

---
**Appendix A – Shapes that feed MXU**
- Favor `[M×K] · [K×N]` with `M,N,K % 128 == 0` (pad otherwise). Sequence lengths not divisible by 128: pad/mask and trim after softmax.
- Try to keep **batch** divisible by **8**; TPU vector regs are 8×128.

**Appendix B – Dialect/Pass inventory**
- `stablehlo` ops, `sdy` sharding attrs, XLA custom calls (future), Tessera Target‑IR (this project).

<<<MERGE_END: Tessera_TPU_Backend_Design>>>
