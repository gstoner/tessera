---
audit_role: plan
plan_state: landing
owner: Apple backend
target: apple_gpu
last_updated: 2026-07-16
---

# Apple compiler, exact-device, and performance plan

This plan brings the proof discipline established by the CUDA and ROCm work to
the Apple backend. It complements [`APPLE_AUDIT.md`](APPLE_AUDIT.md), the
generated execution inventory, and the durable architecture under
[`docs/backends/apple/`](../../../backends/apple/). Those documents remain the
status and architecture authorities; this file owns the active execution order
and completion gates.

The goal is not to transplant CUDA warps or AMD waves into Metal. Apple route
selection must be measured across MPS, MPSGraph, synthesized MSL,
`simdgroup_matrix`, Metal 4 cooperative tensors/MPP, and authored package
subgraphs. Logical fixtures, ABI contracts, numerical oracles, diagnostic
rules, and benchmark schemas should be shared with CUDA and ROCm; physical
fragments, threadgroup shapes, residency strategy, and command-buffer schedules
remain Apple-owned.

## Current state and immediate risk

- The repository contains 181 Apple/Metal/MPS-oriented unit-test modules. The
  first APPLE-TEST-1 migration cohort raises `pytest -m hardware_apple_gpu`
  collection from **3 to 12 of 15,331** unit tests: the MPSGraph warmup and
  MegaMoE measured paths plus exact native proofs for f32 CSR/COO SpMM, SDDMM,
  BSMM, scatter, optimizer, local MoE, MoE transport, and RNG. The shared
  pytest boundary now supplies the Darwin/Metal skip; the marked proofs retain
  their explicit `native_gpu` assertions. Most exact-device candidates still
  use inline Darwin, runtime-symbol, or Metal-availability skips and need the
  same migration.
- A fallback result can prove semantics, but it cannot prove `native_gpu`, GPU
  residency, Metal ordering, resource lifetime, or performance. Device tests
  must assert their execution state and provenance explicitly.
- Apple already has broad MPS/MPSGraph/MSL execution, Metal 4 probes,
  `simdgroup_matrix` and cooperative-matrix candidates, fused GELU/SiLU
  epilogues, online-softmax attention, resident block-paged KV, ReplaySSM,
  command-buffer batching, route characterization, and a hot-path baseline.
  The work below strengthens, compares, and retunes these paths rather than
  reimplementing them blindly.
- The committed Apple hot-path ratchet is predominantly f32 and end-to-end
  wall-clock. It does not yet provide the square/rectangular/ragged/dtype matrix
  or per-candidate GPU-counter/resource evidence now required for CUDA/ROCm.
- Attention backward does not have a CUDA/ROCm-equivalent native Apple proof.
  It is an implementation gap, not merely a missing benchmark.
- FP8/FP4/MX execution remains gated by the macOS 27 SDK/runtime surface. The
  compiler-side scale-layout and multi-plane contracts already exist; do not
  claim hardware execution until the public Metal tensor path runs natively.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus now
  carries additive compiler/resource, cold/warm, cache, and two-run stability
  evidence. Existing v1/v2 rows migrate without changing Apple selection, and
  no CUDA schedule or selector is transferred to Metal. Apple follow-up is to
  populate the same logical evidence fields from Metal-native counters during
  its own performance work; current Apple plan state is otherwise unaffected.

## Completion definition

This plan reaches `closed` only when all of the following are true:

1. Host-free, compiler-artifact, Apple exact-device correctness, Metal 4, and
   measured-performance tests are separate selectable lanes with retained
   reports.
2. Every device test proves `native_gpu` placement on the intended route. A
   non-Darwin stub, NumPy fallback, symbol-presence check, or reference
   recomputation cannot earn a device pass.
3. Dtype, op, target, diagnostic, runtime-symbol, execution-state, and generated
   documentation registries are drift-gated. Every newly emitted diagnostic is
   registered and every live plan uses `open`, `landing`, or `closed`.
4. Portable Tile fixtures execute without test-authored physical fragments and
   select an Apple-owned layout/schedule from observed device capabilities.
5. Performance records use repeated medians after warmup, separate GPU/kernel
   time from end-to-end time where Metal counters permit it, and retain route,
   compiler, OS/SDK, device, residency, and resource evidence.
6. Paged KV and ReplaySSM pass the same non-identity, rollback, ordering, stress,
   and lifecycle closure used on CUDA/ROCm.
7. Production route changes consume only matching native-and-correct evidence;
   stale reports, reference rows, or records from another Apple GPU family
   cannot change selection.
8. The complete exact-device correctness lane passes twice from a fresh runtime
   image, and the isolated performance lane produces stable winner decisions.

## Apple-host preflight

Run decisive tests outside a sandbox in a fresh process. Record the exact host
before interpreting a skip or timing change:

```bash
sw_vers
system_profiler SPDisplaysDataType
xcodebuild -version
xcrun --sdk macosx --show-sdk-version
xcrun --find metal
python3 --version
git rev-parse HEAD
```

Also record Apple GPU family/capability probe output, macOS deployment target,
Metal language version, power mode, thermal state, and whether another process
is using the GPU. Use at least one established Metal 3 host and one Metal 4 host
for capability-dependent promotion; never generalize a winner across Apple GPU
families without a matching record.

For compiler artifacts, build the Apple backend and portable MLIR tools:

```bash
cmake -S . -B build-apple -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTESSERA_BUILD_APPLE_BACKEND=ON \
  -DTESSERA_BUILD_EXAMPLES=ON
cmake --build build-apple --target tessera-opt tessera-translate-mlir \
  TesseraAppleRuntime
export TESSERA_OPT="$PWD/build-apple/tools/tessera-opt/tessera-opt"
export PYTHONPATH="$PWD/python:$PWD"
```

Use the actual Ninja output path if the local LLVM/MLIR build lays out
`tessera-opt` differently. Build or load one fresh Apple runtime image for the
device lane; duplicate or stale dylibs invalidate symbol and placement proof.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | APPLE-TEST-1 | Reclassify Apple tests by execution state | Inventory every inline Darwin/Metal/runtime skip. Split mixed fallback/device tests where needed and apply `hardware_apple_gpu`, `compiler_tool`, `integration`, and `performance` at the actual boundary. Centralize Metal/runtime discovery and fresh-process environment setup in shared test support. | Marker collection accounts for every exact-device candidate; the CPU PR lane cannot execute Metal work; every skip names one unavailable capability; no hardware test can pass through the stub or NumPy fallback. |
| 2 | APPLE-TEST-2 | Rebuild the exact-device proof ladder | Separate source/registry, compiler artifact, native dispatch, residency, execute/compare, multi-op ordering, stress, and performance assertions. Replace broad source scans and duplicated runtime probes only after stronger semantic or object/runtime proof exists. | Each family has an explicit strongest proof rung; native rows assert placement and numerical oracle; fresh-runtime and fallback-injection negatives prove the lane cannot mislabel execution. |
| 3 | APPLE-REG-1 | Close registry and state drift | Audit every Apple dtype, op, target feature, runtime symbol, diagnostic code, package form, and execution state against central registries and generated tables. Require the same registration discipline that caught ROCm's unregistered dtype diagnostic. | Adding an emitted diagnostic, dtype, op, or execution state without registration fails a focused unit test; Apple runtime/envelope/generated docs remain synchronized. |
| 4 | APPLE-TILE-1 | Portable Tile materialization and architecture-owned fragments | Run the shared logical Tile fixture through Apple lowering and materialize Apple-owned fragments for the selected MPS, `simdgroup_matrix`, or cooperative-tensor route. Cover packing, execute, unpack/store, ragged edges, grid shapes, alignment, and supported storage/accumulator combinations. | No fixture authors lane maps or physical fragments; exact-device output matches the shared oracle; the selected instruction/API family, threadgroup geometry, and resource record agree with the target descriptor. |
| 5 | APPLE-GEMM-1 | Repeated-median GEMM schedule ratchet | Extend route characterization and the hot-path recorder across square, rectangular, ragged, dtype, and bias/ReLU/GELU/SiLU epilogues. Compare MPS, MPSGraph/package, synthesized MSL, `simdgroup_matrix`, and Metal 4 candidates only where semantics match. Capture GPU time and end-to-end time separately plus threadgroup memory, occupancy/concurrency proxy, compiler statistics, and spill evidence available through Metal tooling. | Two stable runs select a device-family-specific winner per timing domain; correctness and native dispatch precede timing; no production GEMM selector changes without retained before/after evidence. |
| 6 | APPLE-EPILOGUE-1 | Unify the Tile epilogue contract | Drive bias, ReLU, GELU, and SiLU from the backend-neutral epilogue fixture. Verify accumulator precision, activation order, optional bias/residual guards, large-activation stability, ragged stores, and f32/f16/bf16 routes across MSL and Metal 4. | Every supported fusion executes natively and matches the common oracle; unsupported dtype/op pairs produce a registered diagnostic or explicit non-fused route; no silent host epilogue is labeled fused. |
| 7 | APPLE-ATTN-FWD-1 | Retune online-softmax forward attention | Apply the ROCM-6 G6-B/CUDA methodology to Apple without assuming a two-wave or multi-warp shape. Evaluate MPSGraph, synthesized online MSL, command-buffer-resident, and cooperative-matrix candidates for D=128 plus ragged, causal/window, bias, softcap, MHA/GQA/MQA, and long-context cases. | Candidates share one oracle and placement proof; memory traffic, residency, resource evidence, GPU time, and end-to-end time explain the selected per-device route. |
| 8 | APPLE-ATTN-BWD-1 | Implement split/reduced attention backward | Design a native Apple backward path for dQ/dK/dV. Compare atomics with split workspace plus reduction, preserve f32 accumulation, define determinism and workspace policy, and compose it with the same forward variants. | Gradients match the shared CUDA/ROCm oracle across dtype, ragged, causal, GQA/MQA, and boundary cases; determinism and workspace caps are explicit; measured resource and latency rows select the production route. |
| 9 | APPLE-PAGED-KV-1 | Exact-device paged-KV consumer | Strengthen `ResidentBlockPagedKVCache` beyond on-GPU gather: prove permuted/non-contiguous page tables, remap/reuse, concurrent sequences, causal offsets, boundary lengths, exhaustion, and leak-free teardown. Compare gather/stage-to-attention with a direct resident page-table attention candidate under the unchanged ABI. | Both candidates match the same non-identity oracle; no hidden host copy occurs; GPU and end-to-end timing rows may choose different winners and cache them under distinct timing-domain keys. |
| 10 | APPLE-REPLAY-1 | Persistent ReplaySSM serving closure | Extend the fused and block ReplaySSM paths through long decode, forced flush, rollback, partial speculative rejection, block submit, ordered asynchronous command-buffer/ring operation, backpressure, and cleanup. Replace the benchmark's fallback-capable fused row with explicit placement proof and GPU timing. | All transitions match `SSMStateHandle`; rejected tokens cannot mutate committed state; ordering and teardown survive stress; traffic, GPU latency, and end-to-end latency are committed for a wider B/D/N/capacity matrix. |
| 11 | APPLE-RETUNE-1 | Retune older compiler paths | Re-evaluate f32/f16/bf16 GEMM, grouped GEMM/SwiGLU, KV movement, MoE transport, reductions, and decode chains using the improved compiler. Measure primitive kernels separately from command-buffer batching, host synchronization, unified-memory faults, and package authoring/cold compile. | Compiled and incumbent candidates match one oracle; kernel/GPU and end-to-end winners are recorded separately; grouped/transport rows include byte counts, achieved bandwidth, dispatch count, and residency. |
| 12 | APPLE-ROUTE-1 | Harden device-keyed autotuning | Align Apple route reports with CUDA/ROCm corpus fields: physical device/family, OS/SDK/compiler fingerprint, route, timing domain, native proof, correctness, resources, cold/warm state, and cache behavior. Keep package-subgraph selection separate from single-op selection. | Stale, reference, mismatched-device, or wrong-domain evidence is rejected; selector decisions cite retained rows; cold/warm and package-cache behavior are reproducible. |
| 13 | APPLE-DTYPE-1 | Finish low-precision execution when the public SDK permits | On macOS 27+, turn the existing scale-layout and auxiliary-plane descriptor contracts into native FP8/FP4/MX tensors and GEMM/epilogue execution. Keep int4/int8 and f16/bf16 regression coverage on older supported hosts. | Native multi-plane buffers execute on-device and match bit-accurate references; scale layout round-trips; unsupported OS/SDK pairs return the registered toolchain-gated state without falling through to a hardware claim. |
| 14 | APPLE-CI-1 | Own Apple release lanes | Add Apple-host jobs with per-device concurrency, fresh runtime images, retained JUnit/compiler/Metal artifacts, and isolated performance scheduling. Keep the broad host-free Apple lowering tests in ordinary CI. | Release output reports CPU, compiler artifact, Metal 3 correctness, Metal 4 correctness, and performance independently; required Apple promotion cannot pass when exact-device proof is skipped. |
| 15 | APPLE-CI-2 | Validate Apple host-free compiler ownership | On the Apple build host, construct the compiler artifact used by host-free tests and prove which Apple/ROCm/NVIDIA pass families it contains. Split or capability-gate foreign-backend compiler tests when that host intentionally builds only Apple; do not treat a foreign pass absence as an Apple device or test-location failure. Record command, build flags, tool path, collected node IDs, and diagnostic for each unavailable foreign capability. | The Apple host-free lane is green for its declared compiler capability set, every excluded foreign compiler test has an explicit owner/selection rule, and no Apple migration is blocked by a CUDA/ROCm-only build assumption. |

## Canonical validation lanes

After APPLE-TEST-1 establishes complete marker coverage, the Apple host should
run these as independent commands:

```bash
# Host-free compiler, selector, validation, rejection, and fallback contracts.
python3 -m pytest tests/unit -q \
  -m "not hardware_apple_gpu and not performance"

# Apple compiler artifacts; this lane does not claim device execution.
python3 -m pytest tests/unit -q \
  -m "compiler_tool and not hardware_apple_gpu" --durations=50

# Native Metal correctness, twice from the same fresh build/runtime image.
python3 -m pytest tests/unit -q \
  -m "hardware_apple_gpu and not performance" --durations=100 \
  --junitxml=/tmp/apple-device-correctness.xml

# Measured lane: serial execution only.
python3 -m pytest tests/unit -q -n 0 \
  -m "hardware_apple_gpu and performance" --durations=0 \
  --junitxml=/tmp/apple-performance.xml
```

The first focused parity and characterization loop is:

```bash
python3 -m pytest -q \
  tests/unit/test_apple_gemm_schedules.py \
  tests/unit/test_apple_sdpa_schedules.py \
  tests/unit/test_apple_gpu_metal4.py \
  tests/unit/test_apple_gpu_mpsgraph_lane.py \
  tests/unit/test_apple_gpu_resident_block_paged.py \
  tests/unit/test_ssm_apple_gpu_fused.py

python3 benchmarks/apple_gpu/benchmark_route_characterization.py \
  --matmul-shapes 64x64x64 128x256x64 257x129x65 256x256x256 \
  --softmax-shapes 64x64 128x257 256x256 \
  --reps 30 --output /tmp/apple-routes.json

python3 benchmarks/apple_gpu/benchmark_ssm_replay.py \
  --shapes 1x128x128 1x256x128 4x128x64 \
  --tokens 512 --capacity 16 --reps 20 \
  --output /tmp/apple-ssm-replay.json

python3 benchmarks/apple_gpu/record_hot_path_baseline.py --reps 20 --margin 2.0
```

Focused tests are edit-loop aids, not substitutes for the full marker lanes.
Files under `/tmp` are review artifacts only. Update a committed baseline or
route corpus only after two stable runs, explicit native-placement review, and
before/after resource inspection.

## Failure and benchmark evidence contract

For each failure or candidate record retain:

- test node, proof layer, Apple GPU family, macOS/SDK/compiler, dtype, shape,
  seed, selected route, and observed placement;
- fresh-runtime identity and whether the result reproduces alone, serially, and
  on the second clean run;
- named diagnostic or runtime error kind, compiler output, and Metal validation
  messages;
- maximum absolute/relative error, first failing index, non-finite policy, and
  the exact shared oracle;
- GPU/kernel time versus end-to-end time, warmup/repetition policy, cold compile
  or package-authoring cost, and command-buffer/dispatch count;
- residency and traffic bytes, threadgroup memory, occupancy/concurrency proxy,
  compiler statistics, and spill evidence available from the Metal toolchain;
- disposition: product defect, test-state defect, stale route/baseline, duplicate
  proof, unsupported capability, or exact external environment blocker.

Do not widen numerical tolerances or latency caps solely to turn the lane green.
Derive numerical policy from storage/accumulation semantics and performance
policy from stable repeated-median evidence.

## Next update

After the first Apple-host collection, replace the provisional marker count
with the migrated exact-device totals and append a failure table by execution
family and device generation. Set `plan_state: landing` once implementation or
test migration begins. Move this plan to the Apple archive only after every
completion gate is met.
