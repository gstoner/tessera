# Real Qwen3-8B gate/up NVFP4 ingest on gfx1201

Owner ROCM-NVFP4-INGEST-1. Sync ROCM-NVFP4-INGEST-1-QWEN3-GATE-UP-2026-10-03.
Implementation remains unpublished.

## Source and native boundary

The recorder fetches only layer-0 gate_proj and up_proj tensor ranges from
pinned nvidia/Qwen3-8B-NVFP4 and Qwen/Qwen3-8B revisions. gate-up.json records
the revisions, tensor/scale/global-scale byte hashes, index hashes, dtype/shape
specifications and both independent projection entries. Each matrix has
12288×4096 weights; merged row offsets are 0,12288,24576.

Both real global scales happen to equal 0.00021943592582829297. This real packet
does not stress unequal scales. The exact-device synthetic gate/up test retains
separate 0.5 and 2.0 globals, along with independent host ingest properties.
Neither real projection's global is collapsed or discarded.

Host ingest converts NVFP4 K16/E4M3/global scaling to MXFP4 K32/E8M0 and explicitly
records code/scale requantization loss and per-projection SQNR. The destination
packed MXFP4 operands enter the textual frontend, typed Graph, native Schedule,
Tile and ROCm Target lowering, then HSACO with the checked runtime ABI.
Ingest itself is host checkpoint preprocessing, not a native conversion op;
the recorded conversion policy does not claim MLIR executes that conversion.

## Exact-device numerical and timing evidence

Tajasaurus RX 9070 XT, gfx1201, matching compiler
fd8acb4877bc5e2cea58f280d8f31d882f6ff3618e16e7e10bb283fb17675640.
At M/N/K = 16/24576/4096, native ingested and direct-BF16 MXFP4 outputs both
match their decoded-weight references with zero absolute error. Activations
are deterministic synthetic FP8, not real model activations.

| Weight path versus pinned BF16 | Relative RMS error | SQNR dB |
| --- | ---: | ---: |
| Shipped NVFP4 | 9.50% | 20.45 |
| NVFP4 → MXFP4 ingest | 14.97% | 16.49 |
| BF16 → MXFP4 direct | 11.30% | 18.94 |

Ingest differs from shipped NVFP4 by 11.29% relative RMS; each projection's
separate quality metrics are retained. These are weight errors, not whole-model
accuracy or a format-selection decision. FP8, MXFP8 and MXFP4 remain mandatory
comparison gates before a strategy decision.

Seven device-event samples (20 launches each) have median 0.16075 ms.
Seven checked end-to-end samples have median 9.53975 ms and CV 1.55%.
Host ingest takes 14.818 s; checkpoint fetch and compiler packaging are separate.
Device timing uses resident buffers/module and includes GPU dispatch. It does not
include host staging, transfers, allocation or module lifecycle. No comparison
speedup, isolated instruction-time or persistent-kernel claim follows.

## Fixture and recorder repair

The previous ingest fixture included a stale W8A8 function with f32 scale tensors
labeled E8M0. The current verifier correctly refused it before the packed ingest
function lowered. Ingest and exact packed-MXFP4 callers now use a dedicated
e2e_mxfp4_ingest_rocm.mlir. The combined fixture uses raw i8 E8M0 scales, the
per-column K32 contract and verified WMMA generator/LLVM ldexp lowering.
Both native FileCheck routes pass. This preserves both format checks.

The shared resident recorder now requires an expected result and checks the
first native output before creating timing events. A fault-injection test proves
bad output aborts before any timing event and releases allocations/module.
Quality reductions and direct quantization bound temporary f64 storage so merged
source matrices fit the owning host's 15 GiB RAM.

## Validation

final-tests.txt: 38 host/exact-gfx1201 tests pass, including independent unequal
globals and generic packed-MXFP4 native execution. host-contracts.txt: 46 host
contracts pass. Both native format fixtures pass FileCheck. Eleven audit tests,
compiler-plan ownership/links, all 32 generated views, Ruff and git diff --check
pass. validation.json binds the retained source files and confirms native image
identity with the earlier q_proj packet. Graphify is unavailable in WSL.

## Reproduce and remaining work

    source scripts/_rocm_env.sh
    export TESSERA_OPT=$PWD/.build-gfx1201-current/tools/tessera-opt/tessera-opt
    export TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1
    export PYTHONPATH=$PWD/python:$PWD
    /home/angstorms/scratch/gfx1201-scheduled-norm-edge/.venv/bin/python benchmarks/rocm/benchmark_rocm_nvfp4_checkpoint.py --projection-group gate_up --output gate-up.json

Default q_proj CLI remains supported. Unsupported tensor groups and invalid
timing arguments are rejected before network/device access. Source-code and
compiler hashes bind the packet to its implementation; raw failed-fixture test
logs are retained separately from final validation.

Remaining: native conversion-op integration if ingest is to execute below Graph,
whole-model/source-activation quality, other packing variants, model-derived
FP8/MXFP8 comparisons, and launch movement overhead. The NVIDIA W1.1 and attention
programs keep their own exact-device proof and uncovered envelopes. No sibling
physical schedule or dtype capability is promoted.
