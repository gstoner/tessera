# Native normalization accuracy closure

Owner: W1.1; sibling FRONTEND-IR-MEDIUM-1. Sync: NVIDIA-NORM-ACCURACY-2026-10-05.

The SM120 C++ materializer now uses MLIR math.sqrt and arith.divf instead of approximate reciprocal square root followed by multiplication. Serial normalization uses compensated FP32 summation for the mean/squared sum and centered variance. A finite-value guard preserves Inf/NaN propagation. Accumulation remains FP32, storage remains fp16/BF16/fp32, and the Graph/Schedule/Tile and runtime ABI contracts are unchanged.

## Numerical proof

- Frozen original compiler: three of 48 composed arms fail the existing float64-oracle rtol/atol=.015 gate.
- Current compiler: all 48 arms pass, across FP16/BF16, RMSNorm/LayerNorm, K1024/4096/8192, two seeds and both physical schedules.
- Named BF16 M128/K4096/N64 RMSNorm problem: serial and cooperative stored norm now exactly match the rounded independent oracle for this seed; final output max error is 0.00032467 versus 0.02533446/0.02263976 before.
- Consumer PTX payload and Target IR are identical across all 24 paired semantic inputs. Whole-image identities differ because compiler fingerprints correctly change.
- 435 focused shared/audit tests pass, 17 skip; compiler-plan and 32 generated-document drift gates pass. Two native FileCheck fixtures pass, including serial/cooperative norm materialization and Graph→Schedule→Tile norm/attention.
- 136 exact RTX 5070 device tests pass, including four long-row regressions and six Inf/NaN cases. The archived unguarded compensated compiler fails the three serial non-finite cases; the guarded compiler passes them.
- 40 ordinary composed @jit cases pass before timing, including the previously excluded BF16 K4096 cases, fused epilogues, cached calls and portable replay. The original tolerance is unchanged.

## Physical and timing evidence

Graph, Schedule and Tile snapshots are byte-identical between frozen/current arms; the change belongs to native NVIDIA Target materialization. Eight complete stage/image+ABI snapshots and both native norm source functions are retained. PTX replaces rsqrt.approx with sqrt.rn. RMSNorm regular barriers remain 9 and LayerNorm 18 under the cooperative schedule; no split/async barriers are introduced. Shared storage remains 512 bytes and measured arms have no spills. Representative BF16 serial registers change 18→19 (RMSNorm) and 22→24 (LayerNorm); cooperative registers change 23→23 and 29→34.

Thirty-six norm cases compare frozen/current implementations with five balanced trials per arm, separate CUDA-event resident dispatch and checked host wall windows. Performance is envelope-specific; the packet retains short and long cases.

| Storage | Producer | Shape M/K | Baseline cooperative event ms | Current cooperative event ms |
| --- | --- | --- | ---: | ---: |
| bf16 | rmsnorm | 1/32 | 0.009036 | 0.008919 |
| bf16 | rmsnorm | 128/1024 | 0.009131 | 0.008581 |
| bf16 | rmsnorm | 256/4096 | 0.011202 | 0.011215 |
| bf16 | rmsnorm | 2/4097 | 0.009113 | 0.009720 |
| bf16 | layernorm | 1/32 | 0.009354 | 0.008228 |
| bf16 | layernorm | 128/1024 | 0.009293 | 0.008175 |
| bf16 | layernorm | 256/4096 | 0.012127 | 0.011358 |
| bf16 | layernorm | 2/4097 | 0.009525 | 0.009794 |

The composed program packet separately compares current native-selected and forced-serial schedules with identical consumer images. BF16 M128/K4096/N64 selected producer dispatch is about 0.010 ms versus 0.80–1.00 ms for serial in this run. Full checked wall time remains about 5 ms and is separately reported.

## Remaining work

This closes the named long-row BF16 gate in the measured envelope. General W1.1 producers, dynamic/composed AD, broader attention families, ROCm route/performance closure, and independent FP8/MXFP8/MXFP4 gates remain open. Attention JVP still constructs its GPU MLIR body in python/tessera/compiler/native_attention_jvp.py; a native Schedule/Tile migration remains required.

The source change is confined to the SM120 NVIDIA materializer. Apple, x86 and ROCm physical normalization have no new execution proof from this packet and require their own IEEE/accuracy assessment. No sibling performance transfer or format-policy promotion.

Historical intermediate packets preserve the sqrt-only and unguarded-compensation measurements. nonfinite-counterexample-during-link.txt is an infrastructure failure and is not numerical evidence; performance-overlap-aborted.txt is an aborted run and is not timing evidence. Only the final accuracy.json, timing.json, program-timing.json and device-tests.txt support the current claim. Graphify is unavailable in the authoritative WSL checkout; no refreshed graph is claimed.
