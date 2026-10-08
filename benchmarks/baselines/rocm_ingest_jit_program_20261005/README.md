# Ordinary frontend NVFP4 resident program

Owner: ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1.
Synchronization key: ROCM-INGEST-JIT-PROGRAM-2026-10-05.
Exact device: AMD Radeon RX 9070 XT / gfx1201 on Tajasaurus.
Matching native compiler SHA256: `7d064f19ce0478bc713f1eeb6887e5f586a9c4bef876a8046fdf93354f74f159`.

## Proven contract

Ordinary Python `@jit(target="rocm_gfx1201")` captures
`nvfp4_requantize -> mxfp4_folded_storage -> scaled_matmul`.
Public catalog inference derives the BF16 [M,N] result from the packed profile.
The complete caller Graph is verified by tessera-opt before partitioning.
The caller's operation types, policy and SSA edges are checked and preserved.
Each stage runs native Graph -> Schedule -> Tile -> ROCm Target -> LLVM -> HSACO.
Python performs tracing, guards and orchestration; compiled execution does not
call the CPU conversion, storage or matmul oracles.

Three packages execute with one owned HIP stream and eleven private allocations.
Weights remain resident between native stages. Argument order survives ordinary
calls and serialized RuntimeArtifact replay. Fresh-process replay needs no compiler.
Content hashes check integrity; they do not authenticate compiler origin.

Scope: static primal M>64, N multiple of 16, K multiple of 64; named gfx12 folded
physical profile with explicit approximate numerical policy. General AD,
dynamic shapes, alternate layouts/sharding/effects/model metadata, arbitrary
producer graphs and model-quality acceptance remain open. General uint8 and
format defaults are not promoted. gfx1151 cannot inherit RDNA4 FP8 WMMA proof.

## Validation

- 30 gfx1201 tests: first and cached ordinary calls, two frontend argument orders,
  numerical/bitwise stage checks, fresh-process common-runtime replay, manifest
  tamper and stream/lifetime refusal, incompatible policy before compiler/GPU use.
- 882 shared registry, dtype, Graph shape, manifest, diagnostics, pass, native
  artifact, execution matrix and coverage gates pass.
- 76 selected RTX 5070 regression tests pass with the owning SM120 device gate.
  This validates sibling regression; it does not prove NVIDIA ingest parity.
- Three exact-target native Schedule/Tile/Target FileCheck stages pass.
- 11 audit tests pass. Lint and generated-document results are retained alongside.

See [device tests](device-tests.txt), [shared tests](shared-tests.txt),
[NVIDIA regression](nvidia-regression.txt), [native fixture](native-fixture.txt),
[audit tests](audit-tests.txt) and [generated checks](generated-check.txt).

## Timing: pinned gate/up, M=256 N=24576 K=4096

The source is the pinned layer-zero gate/up checkpoint recorded in
[checkpoint-format-controls.json](checkpoint-format-controls.json).
Activations are seeded synthetic FP32 values, not captured model activations.
Independent conversion/storage/folded-float64 arithmetic checks run before and
after timing. All five fresh format control arms have zero numerical-bound violations.

| Ordinary frontend measurement | ms |
| --- | ---: |
| Cold JIT, including compilation and checked execution | 529.795 |
| Warm JIT, cached images with fresh owned HIP resources | 173.247 |
| RuntimeArtifact JSON restore only | 1.668 |
| Portable checked launch, allocations/uploads/readback/cleanup | 175.758 |
| Resident converter graph window per iteration | 40.242 |
| Resident storage graph window per iteration | 0.342 |
| Resident matmul graph window per iteration | 0.388 |
| Resident combined graph window per iteration | 40.892 |

Serialized artifact: 286749 bytes.
Graph windows use repeated resident execution and include GPU dispatch; host wall
times also include orchestration and movement. Compilation/restore/device/wall
values are separate. Conversion dominates this full-chain workload.
The direct resident control is recorded separately in the same packet.

Six synthetic rows in [jit.json](jit.json) cover M/N/K = 128/32/256,
257/80/1024 and 256/64/64, each in two argument orders.
Warm JIT medians span 20.857–
27.857 ms.

## Independent format gates

Same source weights and seeded activation values; each arm has its own explicit
quantization/scale policy. These are consumer measurements after preparation,
so they cannot be compared directly with the ingest chain's conversion-inclusive time.

| Format arm | Device + dispatch ms | Host E2E ms | Output relative RMS vs source | Bound violations |
| --- | ---: | ---: | ---: | ---: |
| fp8_k128_n128 | 0.547 | 14.544 | 3.686% | 0 |
| fp8_k32_n1_control | 0.809 | 20.955 | 3.388% | 0 |
| mxfp8_k32_n1 | 0.829 | 17.298 | 3.751% | 0 |
| mxfp4_folded | 0.393 | 100.703 | 12.049% | 0 |
| mxfp4_folded_native_packed | 0.385 | 57.130 | 12.049% | 0 |

Ingested folded-chain output relative RMS vs source:
15.177%.
Folded weight relative RMS: 14.980%.
Arithmetic correctness is established for the declared approximate profile;
these source-loss values require model-quality evaluation before acceptance.

The packet verifies source/compiler fingerprints, actual GPU identity and ISA
hashes. Historical packets retain their own original measurements.
Compiler-plan and whitespace checks pass. Graphify update could not run because
Graphify is unavailable in the WSL scratch environment (exit 127); no refreshed
knowledge graph is claimed. The original five-slice goal remains active.
