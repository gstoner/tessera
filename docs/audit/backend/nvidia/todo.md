---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-07-16
---

# NVIDIA compiler test-suite evaluation and rearchitecture

This is the execution plan for evaluating, repairing, and then restructuring
the CUDA compiler tests on the NVIDIA box. It complements
[`NVIDIA_AUDIT.md`](NVIDIA_AUDIT.md); it does not reopen completed sm_120 feature
work unless a test exposes a real defect.

Baseline state on the NVIDIA box (2026-07-15, commit `ecf9483f`):

- The repository collects **264 exact-device CUDA tests** under
  `pytest -m hardware_nvidia`: **246 correctness** cases and **18 measured
  performance** cases. Eight of the correctness cases also require external
  compiler tools.
- The required CPU PR lane now excludes hardware and measured-performance
  states while retaining host-free CUDA emit, selector, validation, rejection,
  registry, and source-contract tests.
- Live CUDA tests carry `hardware_nvidia`; measured tests additionally carry
  `performance`; compiler/toolchain crossings carry `compiler_tool`.
- The WSL NVIDIA host is an RTX 5070 Ti (UUID
  `GPU-5072cda5-509a-008c-93c8-dc06e105f307`, CC 12.0), driver 610.62, CUDA
  13.3.73, LLVM/MLIR 22.1.8, and Python 3.14.4. At collection it was idle;
  observed graphics clock/power were 375 MHz / 23.18 W with a 300 W limit.
- The compiler-artifact lit suite passed **19/19**. The exact-device
  non-performance lane passed twice: **224 selected, 0 failed, 0 errored, 1
  Apple-only skip** (54.783 s and 53.658 s). This covers the required
  execute/compare layer, but it does not constitute performance evidence.
- The serial measured lane passed **18 CUDA tests** (plus the same unrelated
  Apple-only collection skip; JUnit: `/tmp/nvidia-performance-ecf9483f.xml`).
  The production hot-path ratchet, device-resident event timing, convolution
  routes, and Tile/autotune selection were included. Verbose `ptxas` and
  `cuobjdump --dump-resource-usage` for the compiler-produced Tile kernels
  report zero spills: 36 registers and no shared/local memory for direct,
  GELU, and SiLU; 42 registers and 2 KiB static shared memory from `ptxas`
  (3 KiB in the cubin resource table) for shared-staging f32/bf16/bias-ReLU.
  Nsight Compute 2026.2.1 profiled the f16 device-resident GEMM proof: the
  19x13x29 case launches four one-warp blocks (2x2 grid), uses 40
  registers/thread and no kernel shared memory, and reports 50% theoretical
  versus 2.08% achieved occupancy. The low achieved value is expected for this
  deliberately tiny ABI fixture: four blocks cannot occupy all 70 SMs. It is
  launch-shape evidence, not a production-throughput claim.
- Nsight captures for the production hot-path, convolution-route, and Tile
  schedule tests are retained under `/tmp/nvidia-*-ncu.ncu-rep`. Hot-path
  GEMM/fused/attention use 40 registers/thread and no shared memory (the
  attention kernel has 22.92% theoretical occupancy); fp quantization uses 18
  registers/thread, no shared memory, and 74.36% achieved occupancy. The
  direct/shared convolution routes use 40/30 registers and 0/32 bytes static
  shared memory, respectively. The selected Tile candidates report direct:
  36 registers, 0 shared, 50% theoretical occupancy; shared: 42 registers,
  2.05 KiB shared, 83.33% theoretical occupancy. The measured achieved
  occupancies are shape-dependent and low for the micro-grid fixtures, so they
  are resource evidence rather than selector-retuning evidence.
- The hot-path ratchet intentionally fails when run under Nsight Compute:
  profiler replay raises the wall-clock samples above its uninstrumented
  repeated-median caps. Its ordinary serial run remains the timing proof; the
  Nsight run is resource-only and must never update or relax the ratchet.
- The serving benchmark completed 20-repetition device-event and end-to-end
  rows for ReplaySSM `1x128x64`/`1x256x128` and fused/staged paged-KV decode
  at 128/512/2048 tokens (`/tmp/nvidia-sm120-serving-test5.json`). A temporary
  candidate corpus records both timing domains for rectangular `128x256x64`
  and ragged `127x259x63` f16 GEMM: end-to-end selects shared Tile (0.506 and
  0.490 ms), while device timing selects direct Tile (0.00657 and 0.00703 ms).
  No committed selector or timing cap changed.
- Nsight reduction coverage passed 49 native cases: f32/f16 kernels use 28
  registers and 1.02 KiB static shared memory with 100% theoretical occupancy.
  MoE transport passed three native cases: gather/combine use 20 registers and
  no shared memory; grouped GEMM uses 40 registers and no shared memory. The
  two live MoE transport tests were missing `hardware_nvidia` and are now in
  the canonical exact-device collection; its host-only rejection test remains
  unmarked. Their repeated-median rows and resource evidence are now committed
  under the TEST-5 baselines described below.
- NVIDIA-TEST-4 now has a shared storage/accumulation tolerance contract for
  f32, f16, bf16, TF32, FP8, int8, and NVFP4 semantics. The shipped MMA and
  compiler-produced Tile proofs consume that contract; exact integer/NVFP4
  contracts remain bit-exact. The reduction matrix now also proves f16/f32
  non-finite propagation and rejects empty, rank-invalid, unsupported-storage,
  and unknown-operation contracts before launch. Two WSL exact-device runs
  recorded 243 tests with zero failures/errors (one expected Apple-only skip).
- NVIDIA-TEST-5 now productizes repeated-median reduction and MoE transport
  rows in `record_reduction_transport_baseline.py`: every route records both
  end-to-end and CUDA-event timing through the production generated kernel.
  Two 20-sample sm_120 runs were recorded, the committed ratchet baseline
  covers reduction sum/mean/max plus MoE dispatch/combine/grouped-GEMM, and
  the first expanded serial performance lane passed 19 tests (one expected
  skip). The wider corpus and parsed resource evidence have since landed.
- The TEST-5 D2 corpus now includes measured square `512x512x512`, rectangular
  `128x256x64`, and ragged `127x259x63` f16/bf16 GEMM rows in both timing
  domains, alongside fused GELU, forward attention, gated MLP, and convolution
  routes. Two 20-sample WSL runs were taken before retaining the second corpus;
  the initial end-to-end winners varied between runs, so no selector was
  promoted from that evidence.
  Serving was likewise refreshed from two 20-sample runs for ReplaySSM and
  fused/staged paged-KV at 128/512/2048 tokens, retaining device-event and
  end-to-end medians separately.
- **NVIDIA-TEST-5 is closed (2026-07-16).** Two fresh high-sample sweeps
  (50 end-to-end repetitions after 10 warmups; 200 device-event repetitions
  after 20 warmups) converge for all 20 retained D2 rows under the declared 3%
  noise policy. Every row is selector-eligible only because both runs share a
  near-winner consensus and the selected route has a committed resource
  fingerprint. Backward attention adds regular `1x8x128x64` and ragged
  `1x8x257x64` dual-domain ratchets. Parsed Nsight evidence records registers,
  static/dynamic shared memory, theoretical/achieved occupancy, and explicit
  local-load/store spill counters for GEMM/Tile, fused and forward/backward
  attention, convolution, reductions, MoE transport, paged-KV, and ReplaySSM.
  The backward VJP uses 48 registers and measurable local-memory traffic; this
  is retained evidence, not hidden by a zero-spill claim. All other selected
  rows in the resource manifest recorded zero local spill traffic. The final
  serial performance lane passed 20 tests with one expected Apple-only skip.
- NVIDIA-TEST-6 has begun with `tests/_support/nvidia.py` (with a retained
  `tests/unit/_nvidia_testutil.py` compatibility import): it centralizes
  CUDA-toolchain, MMA-runtime, and bare CUDA-host probes without conflating
  their skip semantics, and supplies a common native-provenance assertion.
  The MoE transport, reductions, paged-KV, and ReplaySSM families migrated in
  the first batch; 70 focused tests passed and the canonical device collection
  remains 243 nodes. This is NVIDIA-only test infrastructure; Apple and ROCm
  plan states are unaffected.
- The second batch removed the same local MMA-runtime probe from norm, softmax,
  matmul-ReLU, matmul-softmax, compiled KV-cache, forward/backward Flash
  Attention, and convolution tests. Their 89 focused exact-device tests passed
  on the RTX 5070 Ti. Specialized compiler and Tile availability probes remain
  local until their stronger capability contracts can be preserved explicitly.
- The third batch migrated control flow, DeltaNet, dequant GEMM, FP quant,
  local collectives, optimizers, positional encoding, and SSM to the shared
  MMA-runtime probe. It also classified their live CUDA tests with
  `hardware_nvidia` while leaving host-only negative tests unmarked. The 32
  focused tests passed; collection increased from 243 to 264 exact-device
  nodes (246 correctness, 18 performance).
- The next helper-deduplication batch replaced private ordinary MMA-runtime
  probes in linear attention, MLA decode, and sparse attention with the shared
  capability-specific helper. The E3 hand-tuned GEMM proof now uses the shared
  MMA-plus-PTX-launch predicate; Tile tool/runtime checks remain local because
  they prove a stronger compiler-path capability.
- The second physical relocation split the mixed NVIDIA MMA launch file into
  two host-free execution-matrix contracts and five exact-device launch/JIT
  proofs under `tests/device/nvidia/`. The mapped cohort passed 21 focused
  tests, 19/19 compiler lit plus its compiler pytest contract, exact-device
  correctness twice (246 passed, one Apple-only skip each), and serial
  performance (18 passed, one Apple-only skip).
- The third physical relocation moved the two device-only DSA sparse-attention
  proofs to `tests/device/nvidia/test_sparse_attention.py`. Its node map,
  focused execute/compare run, compiler artifact lane, two exact-device runs
  (246 passed, one Apple-only skip each), and serial performance lane (18
  passed, one Apple-only skip) all passed without changing the 264-node
  NVIDIA marker topology.
- NVIDIA compiler-artifact selection no longer relies on the
  `test_nvidia_*.py` filename pattern: the `compiler_nvidia` marker owns the
  CUDA artifact lane and its release-gate selection. `NvidiaDeviceSession`
  now frees all tracked buffers and destroys its stream even after a
  synchronization failure, and destroys a successfully-created timing event
  if its partner event cannot be created. Host-free fault-injection tests pass
  (2/2); the marker artifact lane passed and the real stream/event ABI fixture
  passed 15/15 on the RTX 5070 Ti.
- The TEST-6 closure audit found no remaining ordinary private MMA/PTX probe
  implementation: plugin and hot-path-ratchet compatibility names now delegate
  to shared predicates, while the Tile probe remains intentionally specialized.
  Running the hot-path ratchet immediately after the broad plugin matrix
  exceeded two f16 caps (512³ and 1024³); two isolated serial reruns both
  passed. The disposition is test-state contamination outside the canonical
  isolated performance lane, not a tolerance change or performance regression.
  Keep TEST-6 `landing`: additional mature device families still require the
  same mapped relocation and four-layer proof contract.
- The control-flow cohort is now accepted: source/rejection contracts remain
  host-free under `tests/unit`, while the bounded-control and runtime-binding
  execute/compare proofs moved to `tests/device/nvidia/test_control_flow.py`.
  Its mapped nodes passed focused validation, 19/19 compiler lit plus the
  NVIDIA compiler marker lane, exact-device correctness twice (246 passed,
  one Apple-only skip each), and serial performance (18 passed, one skip).
- The first device run exposed a product defect in Tile GELU: NVPTX could not
  select LLVM's `ftanh`; after its arithmetic lowering, SiLU exposed the same
  issue for `fexp`. Both now lower through a bounded Pade tanh expression, so
  no unsupported transcendental libcall is emitted. Focused Tile and related
  CUDA correctness tests passed **38/38** after the fix.

## Completion definition

This plan reaches `closed` only when all of the following are true on the
NVIDIA box:

1. Host-free CPU PR tests, NVIDIA compiler-artifact tests, CUDA device
   correctness tests, and CUDA performance tests run as separate commands with
   separate reports.
2. Every exact-device test proves `native_gpu` provenance and compares against
   the same numerical oracle used by the CPU/ROCm paths; no fallback earns a
   pass.
3. The full non-performance CUDA device matrix passes twice from a clean build.
4. Performance tests run serially after warmup and commit repeated-median
   kernel-only and end-to-end evidence. Timing under xdist is forbidden.
5. Tool, dtype, diagnostic, op, target, execution-state, and generated-doc
   registries remain green.
6. Duplicate/source-scan tests are removed only after an equal or stronger
   semantic, FileCheck, object/SASS, or execute/compare proof replaces them.
7. The NVIDIA release gate owns this lane and preserves logs plus machine
   identity for each proof run.

## NVIDIA-box preflight

Record this before interpreting any failure:

```bash
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total \
  --format=csv,noheader
nvcc --version
ptxas --version
python3 --version
git rev-parse HEAD
```

Required target is RTX 5070 Ti / compute capability 12.0. NVFP4/block-scale
tests compile the architecture-specific `sm_120a` target. Record driver, CUDA
toolkit, LLVM/MLIR, Python, GPU UUID, clocks/power mode, and whether another
process is using the device.

Build the compiler and CUDA runtime from a clean NVIDIA build directory:

```bash
cmake -S . -B build-nvidia-cuda -G Ninja \
  -DTESSERA_BUILD_NVIDIA_BACKEND=ON \
  -DTESSERA_ENABLE_CUDA=ON \
  -DTESSERA_CUDA_ARCH=sm_120a \
  -DTESSERA_BUILD_EXAMPLES=OFF
ninja -C build-nvidia-cuda tessera-opt tessera-nvidia-opt \
  tessera_nvidia_gemm tessera_runtime
```

Export explicit tool paths rather than relying on a previous build:

```bash
export TESSERA_OPT="$PWD/build-nvidia-cuda/tools/tessera-opt/tessera-opt"
export MLIR_OPT=/usr/lib/llvm-22/bin/mlir-opt
export PYTHONPATH="$PWD/python:$PWD"
```

Adjust `TESSERA_OPT` to the actual Ninja output reported by the build if the
generator places it under `build-nvidia-cuda/tools/tessera-opt/` differently.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-TEST-1 | Establish a reproducible baseline | Run collection and each proof layer separately; save JUnit, skip reasons, duration report, machine identity, and the exact commit. Classify every failure as product defect, test defect, environment defect, or stale claim. | Two collections return the same node set; no unknown markers; every skip has an explicit unavailable capability. |
| 2 | NVIDIA-TEST-2 | Compiler-artifact layer | Run `check-tessera-nvidia` plus CUDA pytest files carrying `compiler_tool`; migrate private tool probes to `compiler_toolchain`; split artifact assertions from the eight tests that currently continue into device execution; replace large textual snapshots with named diagnostics, FileCheck, or focused IR/object invariants. | Clean build passes without a GPU; missing-tool simulation skip-cleans; no compiler test invokes a nonexistent path. |
| 3 | NVIDIA-TEST-3 | Exact-device correctness | Run `hardware_nvidia and not performance`; group failures by GEMM/Tile, attention, reductions/norms, control flow, KV/ReplaySSM, collectives, and ABI/conformance. Require native provenance and execute/compare. | Entire correctness matrix passes twice; fallback-injection negatives fail to earn native proof. |
| 4 | NVIDIA-TEST-4 | Numerical policy | Centralize dtype/op tolerances from accumulation/storage behavior. Add ragged, rectangular, boundary, non-finite, misalignment, and invalid-contract cases where absent. | f16/bf16/tf32/FP8/int8/NVFP4 cases use documented tolerances; no default zero-`atol` checks near zero. |
| 5 | NVIDIA-TEST-5 | Measured performance | Run `hardware_nvidia and performance` serially. Warm up compilation and caches; use repeated medians; measure kernel-only and end-to-end separately; record registers, shared memory, occupancy, spills, and selected route. | Stable baselines cover square/rectangular/ragged GEMM, fused epilogues, attention, paged KV, ReplaySSM, reductions, and transport. Each ratchet identifies the selected implementation. |
| 6 | NVIDIA-TEST-6 | Refactor and deduplicate | Move mature families toward `tests/compiler/`, `tests/device/nvidia/`, `tests/integration/`, and `tests/performance/nvidia/`. Consolidate repeated CUDA availability, compilation, launch, oracle, and cleanup code. | No central filename allowlist; no duplicated private CUDA probe/loader; process trees and device allocations clean up on failure. |
| 7 | NVIDIA-TEST-7 | CI/release ownership | Add the NVIDIA-box workflow/release-gate command with concurrency control and retained artifacts. Keep device correctness required for NVIDIA promotion; run performance on an isolated/scheduled lane. | A clean branch run reports CPU, artifact, device, and performance states independently and links retained evidence. |

### High-risk NVIDIA-TEST-6 migration

**NVIDIA-TEST-6-HIGH — Relocate mature CUDA families without breaking the
proof contract.** Move mature compiler, device, integration, and performance
families toward `tests/compiler/`, `tests/device/nvidia/`,
`tests/integration/`, and `tests/performance/nvidia/`. This is high risk because
pytest node IDs, import roots, marker collection, CI selection, and retained
JUnit history can all change even if individual assertions still pass.

Before accepting the migration, record an old-to-new node map, preserve every
`hardware_nvidia`/`performance`/`compiler_tool` classification, prove that the
old paths have no duplicate collection, and run the host-free, artifact,
exact-device, and serial-performance layers. Do not combine this migration with
backend behavior, tolerance, or selector changes.

**Pilot evidence (2026-07-15, `landing`).** MoE transport is the first
relocated family. Its two native CUDA execute/compare nodes now live in
`tests/device/nvidia/test_moe_transport.py`; its host-free invalid-partition
contract remains in `tests/unit/test_nvidia_moe_transport_contract.py`. The
checked-in old-to-new map is `tests/device/nvidia/node_migrations.json`, and
`tests/unit/test_nvidia_test_location_migration.py` prevents restoration of
the old file or duplicate destinations. The second cohort applies the same
contract to the former mixed `test_nvidia_launch_execute.py`: two host-free
execution-matrix nodes remain under `tests/unit/`, and five native launch/JIT
nodes move to `tests/device/nvidia/test_launch_execute.py`. The combined roots
collect exactly **264** `hardware_nvidia` nodes (246 correctness, 18
performance). The two device-only DSA sparse-attention nodes are also mapped
to `tests/device/nvidia/test_sparse_attention.py`. Every relocated node
preserves its `hardware_nvidia` classification; none gained `performance` or
`compiler_tool` classification.

The compiler-artifact proof passed (19/19 lit and 1 compiler-tool pytest
contract), exact-device correctness passed twice (246 passed, 1 unrelated
Apple-only skip, zero failures/errors on each run), and the serial performance
lane passed (18 passed, 1 unrelated Apple-only skip). The second cohort
repeated those artifact, two-run correctness, and serial-performance proofs;
its executable node-map and retained host-free contracts passed (4/4). The
complete host-free PR command is **not an
NVIDIA-host acceptance gate** when it exercises Apple/ROCm compiler passes:
this WSL checkout's generic `build/` is intentionally NVIDIA-only, so 274
foreign-backend compiler tests cannot run here. This is not a relocation
failure or an NVIDIA-TEST-6-HIGH blocker. `APPLE-CI-2` and `ROCM-TEST-1` own
validation of their respective host-free compiler configurations on the correct
backend hosts; the NVIDIA host retains the focused host-free migration guard
plus its artifact and exact-device proof layers.

**Completion evidence (2026-07-15, `landing`).** The executable map now covers
**286** relocated node IDs. Mature execute/compare families are collected from
`tests/device/nvidia/`; paged-KV, ReplaySSM, and the MMA bridge are in
`tests/integration/`; and hot-path, Conv2D, MMA-symbol, and plugin timing
proofs are in `tests/performance/nvidia/`. The mixed plugin implementation is
shared through a non-discovered support module, while its 20 host-free
contracts, 53 native nodes, and 8 measured nodes are collected only from their
respective unit/device/performance entry points. The only remaining
`hardware_nvidia` references under `tests/unit/` are release-gate and
marker-policy structural assertions.

The final migrated plugin cohort passed its focused mapping/architecture guard
(93 tests), compiler artifacts (19/19 lit; one compiler pytest pass and one
hardware-excluded skip), exact-device correctness twice (246 passed and one
Apple-only skip each), and serial performance (18 passed and one skip).
Relocating Conv2D exposed an order-dependent product defect: automatic f32
dispatch admitted the explicit `im2col_tf32` candidate under a looser internal
tolerance. Automatic dispatch now selects only f32-accurate direct/shared
routes; explicitly requested TF32 performance coverage remains intact. The
post-fix exact-device matrix passed twice (246 passed and one skip each), and
the serial performance lane passed (18 passed and one skip). This is a product
correctness fix with retained before/after numerical evidence, not a tolerance
relaxation.

## Canonical commands on the NVIDIA box

```bash
# 0. State/collection contract (currently 264 nodes)
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m hardware_nvidia --collect-only -q --no-header

# 1. Host-free PR contract, including CUDA emit/validation/rejection tests
python3 scripts/run_unit_tests.py --timeout=180 -q

# 2. Compiler artifacts without claiming device execution
ninja -C build-nvidia-cuda check-tessera-nvidia
python3 -m pytest tests/unit tests/device/nvidia \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml=/tmp/nvidia-compiler-tool.xml

# 3. Exact-device correctness; run twice from the same clean build
python3 -m pytest tests/unit tests/device/nvidia \
  -m "hardware_nvidia and not performance" -q --durations=100 \
  --junitxml=/tmp/nvidia-device-correctness.xml

# 4. Measured lane: serial only
python3 -m pytest tests/unit tests/device/nvidia \
  -m "hardware_nvidia and performance" -q -n 0 --durations=0 \
  --junitxml=/tmp/nvidia-performance.xml
```

Do not use `-x` for the first baseline: the complete failure topology is needed
to design the migration. After triage, use focused files for the edit loop and
rerun the complete layer before marking an item complete.

## Failure triage contract

For every failure, record:

- node id, proof layer, target/dtype/shape, seed, selected route, and native
  provenance;
- whether it reproduces alone, serially, and on the second clean run;
- compiler stdout/stderr and named diagnostic code;
- numerical maximum absolute/relative error and first failing index;
- kernel-only versus end-to-end latency for performance cases;
- register/shared-memory/occupancy/spill evidence when a kernel changes;
- disposition: fix product, fix test state, replace weak test, merge duplicate,
  or document an exact environment blocker.

Never relax a tolerance or timing cap solely to make the lane green. Recompute
it from dtype semantics or a stable repeated-median baseline, and retain the
before/after evidence.

## Initial family matrix

| Family | Representative coverage | Required follow-up on NVIDIA box |
|---|---|---|
| Tile/GEMM | compiler-generated SM120 fragments, shipped MMA symbols, ragged/grid GEMM, f16/bf16/tf32/FP8/int8, NVFP4 OMMA | Verify exact SASS/instruction family, lane maps, ragged stores, allocation cleanup, and kernel/device timing separation. |
| Fusion | bias, ReLU, GELU, SiLU, gated SwiGLU, matmul-softmax | Cross-check epilogue order and dtype accumulation against CUDA and ROCm shared oracles. |
| Attention | MHA/GQA/MQA, backward, sparse/DSA, window/bias/softcap | Separate compiler artifact from live execution; cover global decode positions and non-finite policy. |
| Reductions/norms | sum/mean/min/max, softmax, RMSNorm/LayerNorm | Validate non-power-of-two/ragged widths, NaN policy, large-offset variance, and dtype tolerances. |
| Stateful serving | paged KV and ReplaySSM async ring | Long decode, flush, rollback, rejection/backpressure, remapped pages, native provenance, and leak-free teardown. |
| Control/collectives | bounded for/if/while/scan and single-device collectives | Validate one-launch ABI, bad-shape rejection before launch, and explicit multi-rank deferral. |
| Performance | GEMM routes, convolution routes, device timing, hot-path ratchet | Run isolated and serial; record winner, resource evidence, kernel-only, and end-to-end rows. |

## ROCm-derived CUDA parity work

The completed ROCm work raised the proof standard for several features that
already exist on CUDA. These are CUDA audits and measured retunes, not literal
ports of AMD schedules. Share logical fixtures, ABI contracts, numerical
oracles, benchmark schemas, and decision rules across backends; keep physical
fragments and schedules architecture-owned.

In particular, an RDNA wave is not a CUDA warp, LDS is not evidence about
shared-memory behavior, VGPR pressure does not predict the CUDA register file,
and WMMA/MFMA winners do not select `mma.sync` or OMMA winners. Every production
selector change below requires fresh `sm_120a` measurements on the NVIDIA box.

| Order | ID | ROCm lesson and CUDA work | Current CUDA state | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-PARITY-TILE | Re-run the same logical portable-Tile fixture through the NVIDIA architecture-owned fragment selector. Cover direct/shared schedules, grid and ragged edges, supported f16/bf16/tf32/FP8/int8/NVFP4 forms, and bias/ReLU/GELU/SiLU epilogues. Add a CUDA fragment resource record containing registers, shared memory, occupancy, spills, and the selected SASS instruction family. | Compiler-generated SM120 fragments, layout oracles, and direct/shared execution tests exist. | Fixtures never author physical fragments; pack/execute/unpack/store matches the shared oracle; emitted instructions and resource rows match the selected `sm_120a` contract. |
| 2 | NVIDIA-PARITY-GEMM-RATCHET | Extend the hot-path recorder into a repeated-median schedule matrix covering square, rectangular, ragged, dtype, and fused-epilogue cases. Record kernel/device-event and end-to-end time separately, then capture registers, shared memory, occupancy, and spills before changing the production tile selector. | `record_hot_path_baseline.py` provides a useful but narrow latency ratchet. | A committed device-keyed baseline identifies every candidate and winner; two stable runs agree within the declared noise policy; no selector change lands without before/after resource evidence. |
| 3 | NVIDIA-PARITY-LEGACY-RETUNE | Re-evaluate older f32/tf32 GEMM, grouped GEMM, grouped SwiGLU, KV movement, and MoE transport now that the compiler and fragment selection are stronger. Compare compiled, shipped, and staged/direct candidates without conflating launch/transfer cost with kernel time. | Individual CUDA paths and hot-path rows exist, but there is no ROCm-equivalent wide retune corpus. | All candidates match one oracle; kernel-only and end-to-end winners are recorded independently; grouped and transport rows include launch collapse and achieved-bandwidth evidence. |
| 4 | NVIDIA-PARITY-ATTN-FWD | Apply the G6-B methodology to CUDA forward attention: evaluate occupancy-aware multi-warp CTA schedules with online softmax at D=128, plus ragged, causal/window, bias, softcap, and MHA/GQA/MQA cases. Do not assume ROCm's two-wave shape is the CUDA winner. | Compiled CUDA forward-attention paths and exact-device tests exist. | Candidate schedules match the shared oracle; traffic and resource evidence explain the winner; the selected route wins repeated-median kernel timing without regressing end-to-end timing. |
| 5 | NVIDIA-PARITY-ATTN-BWD | Apply the G6-C methodology to dK/dV backward. Measure the existing path against atomic and split-workspace/reduction candidates, including deterministic behavior and workspace limits. | Compiled CUDA backward attention is covered, but has not been re-ratcheted against the split/reduced design space. | Forward-derived gradients pass the shared tolerance matrix; determinism and workspace caps are explicit; resource, kernel-only, and end-to-end rows select the production route. |
| 6 | NVIDIA-PARITY-PAGED-KV | Re-prove the stable paged-KV ABI with non-identity/permuted pages, remaps, causal offsets, and boundary lengths. Compare direct resident page-table attention with staged/gather-to-FA using the same oracle and retain both timing domains. | Direct fused and staged paged-attention candidates plus an SM120 serving baseline already exist. | Every candidate consumes the same ABI and matches the same permuted-page oracle; device-event and end-to-end rows may choose different winners and the cache keys preserve that distinction. |
| 7 | NVIDIA-PARITY-REPLAY | Re-run CUDA ReplaySSM against the closure matrix exposed by ROCm: long decode, flush, rollback, speculative rejection, block submit, ordered async ring, backpressure, and teardown. Expand B/D/N/M shapes and record state traffic as well as latency. | CUDA is the reference persistent ReplaySSM implementation and has serving rows, but needs the wider proof and benchmark matrix. | All transitions match `SSMStateHandle`; rejected work cannot mutate committed state; ring ordering and cleanup survive stress; traffic plus kernel/end-to-end latency are committed. |
| 8 | NVIDIA-PARITY-EPILOGUE | Make the common Tile epilogue contract explicit for bias, ReLU, GELU, and SiLU. Check accumulator precision, operation order, optional bias/residual guards, ragged stores, and all supported storage dtypes against shared CUDA/ROCm fixtures. | CUDA emits fused epilogues and plugin tests cover representative forms. | One backend-neutral oracle drives both backends; every supported fusion executes natively; unsupported dtype/op pairs reject with registered diagnostic codes rather than silently de-fusing. |
| 9 | NVIDIA-PARITY-AUTOTUNE | Align CUDA and ROCm corpus schemas around device-keyed candidates, timing domain, compiler/resource fingerprint, cold/warm compile state, and cache behavior. Promote a winner only after the relevant correctness and schedule ratchets pass. | CUDA has autotune and serving corpus writers, but their evidence must be reconciled with the newer ROCm records. | Corpus validation rejects stale devices, compilers, resources, and timing domains; cold/warm behavior is reproducible; selector decisions cite a retained measurement row. |
| 10 | NVIDIA-PARITY-TRANSPORT | Close KV-movement and MoE-transport parity with direct/staged routes, ragged/grouped loads, bandwidth attainment, and launch-amortization measurements. Feed any winner into the legacy retune only after ABI and correctness closure. | CUDA transport operations exist but lack one consolidated exact-device performance proof. | Byte counts and achieved bandwidth are auditable; kernel-only and end-to-end winners are separate; awkward sizes and grouped routes match their reference without leaks or hidden host staging. |

The first focused CUDA parity proof on the NVIDIA box is:

```bash
python3 -m pytest -q \
  tests/device/nvidia/test_tile_fragment_compiler_path.py \
  tests/unit/test_nvidia_fragment_layout.py \
  tests/integration/test_nvidia_paged_kv_native.py \
  tests/integration/test_nvidia_replay_ssm.py \
  tests/device/nvidia/test_flash_attention.py \
  tests/device/nvidia/test_flash_attention_backward.py

python3 benchmarks/nvidia/benchmark_serving.py \
  --shapes 1x128x64 1x256x128 \
  --tokens 64 --chunk 4 --slots 4 \
  --kv-tokens 128 512 2048 --heads 8 --dim 64 --page-size 16 \
  --reps 20 --output /tmp/nvidia-sm120-serving.json

python3 benchmarks/nvidia/record_hot_path_baseline.py --reps 20 --margin 2.0
```

The focused pytest command is a correctness loop, not a substitute for the
marker-separated full CUDA lanes above. Benchmark outputs under `/tmp` are
review artifacts only; update committed baselines or autotune corpora only
after two stable runs and an explicit before/after review.

## Next update

The collection contract, compiler-artifact, exact-device correctness, and
serial measured lanes now have recorded baselines. NVIDIA-TEST-5 is closed;
continue the test-layout migrations under NVIDIA-TEST-6. Keep `plan_state:
landing` while those implementation, migration, or re-run items remain.
Move this plan to the NVIDIA archive only after every completion gate is met.
