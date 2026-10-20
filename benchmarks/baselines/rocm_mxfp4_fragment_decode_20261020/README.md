# Exact packed MXFP4 fragment decoder — 2026-10-20

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync: ROCM-MXFP4-FRAGMENT-DECODE-20261020.
Base: merged PR996, 11a26a9c303a1d1e578ed3e1be35e5afa74072e0.

## Native change and numerical proof

The ROCm C++ Tile materializer replaces seven magnitude lookup comparisons
with the exact E2M1-to-E4M3 bit mapping: 48 + 4*m for magnitudes 2 through 7,
with explicit encodings for zero and 0.5. The sign is applied afterward,
preserving negative zero. Addressing, scale semantics, rounding, schedules,
launch geometry and checked runtime ABI are unchanged. Python remains the
frontend, oracle and measurement harness; native MLIR lowering owns the decoder.

tests/unit/test_rocm_mxfp4_fragment_decode.py and
tests/device/rocm/test_mxfp4_fragment_decode.py expose the packed fragment bits
before an MMA can erase signed zero. Sixteen gfx1201 cases cover every nibble,
negative zero, tight capacities, pitch 16/17, odd/even nonzero K origins,
partial K/N, exact invalid-lane zero, input immutability and output canaries.
The independent oracle uses signed numerical E2M1 levels encoded by ml_dtypes,
rather than duplicating the compiler's bit formula. The target-specific test
cast exposes hardware registers; it is not a portable Graph operation.

Baseline compiler 276e09dfe344a910b345a71de4d39eeaf430a3a18a290fc052b1489cc648adb2
passes the 18 unit/device byte-oracle cases. Candidate compiler
01922c4ad30f86d754318d4c51ded63ee306c44af351a9763bb209554df1ba82
passes 55 focused cases, including ordinary exact packed primal/JVP/reverse
execution on the RX 9070 XT/gfx1201. candidate-native.patch records the source
difference from the merged baseline. frozen-source-identity.json ties the
compiler binary to the native source and patch hashes; its private-experiment
label describes the freeze time, before delivery gates.

## Separate timing domains

Recorded by benchmarks/rocm/record_paired_exact_mxfp4.py.
All shapes below are M,K,N. Arms share quantized operands, independent exact
FP64/group-scale -> BF16 oracle checks before and after windows, native owner
lifetime, duration admission and alternating A/B then B/A order.

Captured-member mode has its own schema and records the first matmul only.
Capture/instantiation and transfers are excluded; device graph dispatch remains
included. It removes per-kernel host enqueue gaps. Short cast/gate member times
are diagnostic and are not promoted as fine performance differences.

| M,K,N | First candidate/control matmul | Reverse-order repeat |
| --- | ---: | ---: |
| 256,128,80 | 0.872 | 0.873 |
| 256,128,160 | 0.890 | 0.888 |
| 17,64,19 | 0.902 | 0.894 |
| 256,4096,1024 | 0.645 | 0.643 |
| 256,4096,2048 | 0.680 | 0.678 |
| 256,4096,8192 | 0.882 | 0.884 |

All admitted captured windows exceed 10 ms. The program digest is unchanged;
only the matmul image changes, while cast and gate images remain byte-identical.
candidate-program-spans.json separately records native program HIP spans,
which include host enqueue gaps and must not be labeled kernel-only.

Completed ordinary warmed @jit calls include frontend binding, checked native
preparation, upload, execution and readback. Compilation and numerical validation
are excluded. Compiler subprocesses are forbidden during measured calls; both
arms pay the diagnostic environment-selection cost. The repeated run reverses
shape order and uses 16 windows with a 40 ms admission threshold.

| M,K,N | First candidate/reference ms | Repeat candidate/reference ms |
| --- | ---: | ---: |
| 256,128,80 | 3.734 / 3.662 | 3.705 / 3.696 |
| 256,4096,1024 | 2.843 / 3.139 | 2.906 / 3.252 |
| 256,4096,2048 | 3.650 / 4.250 | 3.595 / 4.196 |
| 256,4096,8192 | 15.357 / 15.341 | 13.876 / 14.537 |

The small public case is essentially unchanged in the repeat; the N8192 gain
is not consistent across runs. No universal public-call gain is claimed.

## Attribution and controls

image-attribution.json records 89 registers/thread, zero local bytes, zero
static LDS and actual 32-thread launch geometry for both images. The HIP helper
reports 64 active blocks per multiprocessor for both; this is a recorded API
result, not an inferred physical occupancy guarantee. isa-comparison.json
retains full opcode counts and suffix-aware prefix totals: static instruction
count falls from 8389 to 7923; equality compares from 80 to 64; conditional
masks from 778 to 636. Both retain 70 global loads and four WMMA instructions.
These are whole-image static counts, including cold paths, not dynamic hot-loop
counts. Unchanged resources, loads, WMMA and geometry support arithmetic
simplification as the measured cause without attributing a new LDS strategy.

FP8 K128/N128, FP32-scaled FP8 K32/N1, MXFP8 K32/N1 and approximate unpacked
folded MXFP4 control packages produce byte-identical images between compilers.
This exact packed optimization does not close the approximate folded Radiance
gap. Broader shapes/layouts, representative model quality, generic producer/AD
composition and all five aggregate programs remain open. gfx1151, NVIDIA,
Apple and x86 receive no transferred physical proof.

## Reproduction

Use the owning gfx1201 WSL host, matching LLVM/MLIR 23.1.1 and frozen compiler
executables. Set TESSERA_OPT, TESSERA_NVIDIA_OPT, native runtime libraries and
TESSERA_GFX1201_DEVICE_PROOF=1. Never rebuild a binary used by a running job.
Run the two fragment-decode test files above, test_rocm_mxfp4_native.py and
test_mxfp4_residual_native_execution.py for numerical proof.

For captured timings use record_paired_exact_mxfp4 with --captured-members
--windows 16 --min-window-ms 10 and --shapes in both listed orders. For public
calls use --public-windows 16 --public-min-window-ms 40. Supply --compiler and
--reference as distinct frozen executables and --output for each packet.
Keep GPU timing isolated from builds and broad host validation.
formatting_receipt.json preserves original and compact JSON hashes and checks
that formatting changes no recorded values.

The original resource/image attribution harness is preserved verbatim as
image-attribution-source.txt with its owning scratch paths. It uses the public
program owner, independent coefficient oracle, actual member descriptor block
size and HIP resource helper. For reproduction, copy it into an owning scratch
folder, update its compiler/source roots and run with the same environment.
The two ELF note transcripts preserve the metadata independently of HIP.
ISA counts were obtained with llvm-objdump -d --mcpu=gfx1201 on the first
member HSACO; counts are whole-image opcode tokens, including encoding
suffixes. The original binaries and full disassemblies remain in owning
scratch storage; image hashes identify them in image-attribution.json.

isa-summary-correction.json records removal of two erroneous zero-valued
unsuffixed mnemonic summaries. Full opcode counts are unchanged; the
suffix-aware prefix totals above are authoritative.

## Final local gates

Host WSL CPU lane: 30,818 passed, 3,721 declared skips.
Production compiler-route lane: 2,694 passed, zero skips, executed-case gate
passed. Focused diagnostic/pass/audit/recorder drift: 329 passed. Ruff passes,
mypy has zero errors, all 33 generated documents and all five surface audits
pass, and Graphify update completed. The benchmark surface initially found a
stale generated /tmp CMake cache from the previous checkout; regenerating that
build resolves it without changing source or expectations.
validation-summary.json records JUnit hashes/counts and owning receipt paths.
Public exact-head CI and review remain separate publication gates.

Review follow-up: optional public windows must be zero (disabled) or a positive
even count. Odd/one-window requests are rejected before the hardware probe;
valid 0/2/4 boundaries reach that probe. The committed measurements use 12 and
16 windows and are unaffected. Native-program/captured-member windows also require a positive even count,
with odd requests rejected before hardware. This seals both paired domains
at the recorder CLI.

public-window-review.xml/.txt record 21 passing follow-up timing-boundary
tests. The full local suites above ran on original PR head 642444207; the
follow-up changes CLI admission and tests only. The native compiler source,
binary and recorded timing windows remain unchanged. Exact-head public CI
reruns the full lanes on the correction.
