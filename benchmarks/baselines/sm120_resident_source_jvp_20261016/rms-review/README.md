# RMS numerical review repair — 2026-10-16

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1 / AD-HIGHER-1.
Synchronization key W11-RESIDENT-SOURCE-20261016.

The shared RMS JVP/VJP repair multiplies x by each inverse factor before
the reductions, and scales epsilon by each inverse factor before applying the
direction. It avoids materializing an overflowing inverse square when inverse
is finite. Thirty-two native tiny-epsilon regressions reproduce NaNs before the
repair; all 153 native LLVM precision cases now pass, including zero seeds in
FP16/BF16->FP32 and FP32->FP64, affine/permuted/static/dynamic cases and nonzero
BF16 radial inputs. Twenty RTX5070 cases cover nonzero tiny-epsilon BF16
through public host/resident, portable and event/captured profile routes.

Frozen 46a5dd8da721055a51b672adde1eb6e534789c3b passes 26,889 non-slow unit passes (7,678 skips),
912 required compiler-route passes and 416 owning RTX5070 passes, with no skips
in either required proof lane. All phase source/tool/helper witnesses match.
Eight timed packages also pass independent FP64 checks and standalone replay
with compiler subprocesses disabled. The latest same-owner resident comparison
shows about 19–28% lower public latency at the larger synthetic M256 cases,
and 40–57% higher latency at smaller cases. No default, quantized-route or
sibling physical promotion follows. The earlier d1ba6e98d and e164fd4d8
records retain their original source scope; they do not validate this changed
arithmetic. The five-slice aggregate and wider integration gates remain open.

## Latest measurements

Recorded by benchmarks/nvidia/benchmark_active_source_jvp.py, using the same
seven alternating windows and five completed calls/window as the parent packet.
Wall times include completed host outputs; device stage/program CUDA events
remain separate in corrected-timings.json.gz. Initial resident uploads occur
outside invocation timing. These measurements are a new source epoch, not a
controlled speedup comparison against the earlier arithmetic.

| Storage | M x N x K | Host roots ms | Resident roots ms | Resident / host |
|---|---|---|---|---|
| f16 | 17 x 19 x 35 | 0.8389 | 1.1830 | 1.4103 |
| bf16 | 17 x 19 x 35 | 0.7352 | 1.1560 | 1.5724 |
| f16 | 129 x 33 x 257 | 0.9013 | 1.2854 | 1.4263 |
| bf16 | 129 x 33 x 257 | 0.8803 | 1.2321 | 1.3996 |
| f16 | 256 x 1024 x 1536 | 3.3832 | 2.7518 | 0.8134 |
| bf16 | 256 x 1024 x 1536 | 3.4249 | 2.7456 | 0.8017 |
| f16 | 256 x 1024 x 2048 | 4.1300 | 3.0500 | 0.7385 |
| bf16 | 256 x 1024 x 2048 | 4.1578 | 3.0042 | 0.7225 |

## Evidence and reproduction

The parent compression-manifest.json checks every raw and compressed record.
corrected-validation.json.gz binds the exact source, compiler/provider binaries,
helper hashes, commands, exit codes and raw log hashes for each phase. Core,
NVIDIA and ROCm compilers were rebuilt; tiny-epsilon-build.log.gz retains the
build log. tiny-epsilon-before.log.gz retains the 32 numerical failures.

programs/index.json and eight compiled packages retain actual Graph/Schedule/
Tile/Target/native LLVM/PTX, independently checked outputs, identical timed
images and compiler-free replay receipts. Capture used the unchanged parent
capture_programs.py; its hash is bound by the validation receipt. The frozen
validate_corrected.py.txt.gz is archived separately with the new expected head.
Run the parent's benchmark/capture recipe from the recorded source and matching
tools; production compiler-route and device JUnit are checked by
scripts/ci_require_executed.py.

source-snapshot.json.gz inherits byte-identical files from the parent's immutable
source-snapshot-d1ba6e98d.json.gz and stores only changed/additional source bytes.
Every reconstructed file has its own SHA256; no current workspace hash pin is
introduced. This avoids copying unchanged source history into another epoch.

Tests gate the named numerical envelopes; no universal overflow-free derivative
algorithm, sibling physical parity or completed aggregate compiler is claimed.
The compiler retains explicitly selected working types and existing unspecified/
storage-typed behavior. Generic producers, dynamic/pitched/mixed resident roots,
resident outputs and reverse composition remain open.
