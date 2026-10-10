# Exact native NVFP4 candidate normalization

Owner: ROCM-NVFP4-INGEST-1; sibling ROCM-MXFP4-W4A8-1.
Sync: ROCM-INGEST-RECIPROCAL-2026-10-05.
Device: AMD Radeon RX 9070 XT / gfx1201 on Tajasaurus.

## Engineering change

Native Graph/Schedule/Tile/ROCm Target/LLVM converter candidate normalization
multiplies by an exact reciprocal power of two instead of dividing by the scale.
The candidate scale exponent is clamped to [-126,127]; its reciprocal exponent
is [-127,126]. Both values are normal, exactly representable FP64 powers.
IEEE multiplication by that reciprocal and division by the scale have identical
rounding. The nine-candidate search, strict midpoint comparisons, reduction
order and tie-breaking stay intact. No semantic policy or ABI changes.

The complete native converter retains conversion signal/error statistics.
The Graph package, checked descriptors, private resident buffers, storage bridge
and packed native consumer remain in the execution route.
No Python GPU source or eager arithmetic enters that route.

## Paired evidence

[paired.json](paired.json) uses identical inputs and two compiled Graph programs.
Baseline/candidate order alternates; stage order reverses each trial.
Five windows per arm use resident HIP graphs with at least eight repeated
iterations, including GPU graph dispatch. Full host walls include allocations,
uploads, three native stages, readback and cleanup.

Every row compares conversion and lossless storage against independent CPU
oracles, final output against the folded arithmetic oracle, and checks bitwise
A/B equality of bytes, exponents, FP64 statistics and final outputs.
Untouched storage/consumer native image digests must match across compilers.

| M/N/K | Baseline converter ms | Reciprocal converter ms | Speedup | Baseline combined ms | Reciprocal combined ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/32/256 | 0.152 | 0.098 | 1.559x | 0.177 | 0.119 |
| 257/80/1024 | 0.195 | 0.145 | 1.349x | 0.226 | 0.175 |
| 256/64/64 | 0.181 | 0.142 | 1.274x | 0.198 | 0.158 |
| 256/24576/4096 | 40.217 | 18.521 | 2.171x | 40.859 | 18.955 |

Pinned M256/N24576/K4096 checked host wall:
168.012 -> 145.158 ms. This is separate from device graph timing.

The frozen baseline compiler SHA256 is
7d064f19ce0478bc713f1eeb6887e5f586a9c4bef876a8046fdf93354f74f159.
The rebuilt reciprocal compiler SHA256 is
f5dd9037f1b70cf4407274faf2a05546d1d9c8c7efad836671f52177e45220b9.
Its baseline source is archived alongside. The deliberately older compiler
emits the existing stale-source warning during A/B; its identity matches the
preceding JIT packet. It is not presented as a current-source compiler.

## Native attribution

[normalization-counts.json](normalization-counts.json) and archived GPU MLIR
show candidate divisions removed: 289 FP64 divides -> 1 (weighted seed mean).
[isa-attribution.json](isa-attribution.json), HSACO/disassembly and metadata
show native division fixups 289 -> 1, reciprocal instructions 289 -> 1,
fixed scratch bytes per work-item 920 -> 180.
Both variants still allocate 256 VGPRs and 46 SGPRs; remaining spills are open.
These are static native resource/instruction observations, not hardware counters.

## Final ordinary JIT and independent format gates

[checkpoint-format-controls.json](checkpoint-format-controls.json) reruns the
complete ordinary Python frontend chain on the rebuilt compiler.
Warm ordinary JIT wall: 149.100 ms.
Resident combined frontend graph window:
18.990 ms.
Compiler/native/source hashes and actual device identity are retained.

All fresh FP8/MXFP8/MXFP4 controls pass elementwise arithmetic bounds before and
after timing. They share pinned source weights and seeded synthetic FP32
activations. Consumer-only times exclude ingest conversion and cannot be
compared directly with the full chain.

| Arm | Device + dispatch ms | Output relative RMS vs source | Bound violations |
| --- | ---: | ---: | ---: |
| fp8_k128_n128 | 0.545 | 3.686% | 0 |
| fp8_k32_n1_control | 0.809 | 3.388% | 0 |
| mxfp8_k32_n1 | 0.826 | 3.751% | 0 |
| mxfp4_folded | 0.385 | 12.049% | 0 |
| mxfp4_folded_native_packed | 0.387 | 12.049% | 0 |

Ingested folded-chain source output relative RMS remains
15.177%,
identical to the previous packet. This optimization changes cost, not quality.
The activations are synthetic, not captured model activations; model-quality
acceptance stays open. No format selector/default or general dtype promotion.

## Validation and remaining scope

47 exact gfx1201 converter/resident tests pass, including finite accepted
scale cases near the lower/upper candidate exponent limits, signed zero,
raw E4M3 scales and all E2M1 values. See [device tests](device-tests.txt).
350 focused native/registry gates pass with the ROCm-built compiler on Tajasaurus;
NVIDIA-only tessera-opt does not contain lower-tile-to-rocm.
[Shared gates](shared-tests.txt), [build](build.txt) and generated-document
results are retained alongside. Eleven audit tests and all 32 generated-document
checks pass; lint, compiler-plan and whitespace gates pass. Graphify update
is unavailable in this WSL environment (exit 127); no fresh graph is claimed.

The physical edit is ROCm-only; shared IR/ABI/diagnostics/dtypes and sibling
materializers are unchanged. All four backend plans record that scope.
General frontend/AD, dynamic/layout, model-quality and broader five-slice
closure remain open. Remaining register spills are the next attribution target.
