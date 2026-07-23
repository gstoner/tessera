---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-07-22
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
  13.3.73, LLVM/MLIR 23, and Python 3.14.4. At collection it was idle;
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
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: NVIDIA exact-device parity is
  now validated on the RTX 5070 Ti after the shared LLVM/MLIR 23 migration.
  A clean `sm_120a` build required the MLIR bytecode interface include and the
  `NVVM::Barrier0Op` to `NVVM::BarrierOp` API migration. NVIDIA lit passes
  19/19, two stable collections contain the same 268 nodes, the host-free
  compiler-artifact proof passes, exact-device correctness passes 248/248
  twice, TEST-4/TEST-6 focused gates pass 190/190, and the isolated TEST-5
  lane passes 20/20. Explicit Tile tool paths now take precedence over stale
  build-tree binaries. ROCm receives only the LLVM 23 lit-shell compatibility
  update; Apple has no affected physical schedule or runtime contract.
- **NVIDIA-TEST-7 is closed as local WSL release ownership; GitHub runners are
  intentionally not used.** The release command exposes independent `cpu`,
  `compiler`, `device`, and `performance` layers, rejects overlapping runs with
  a host lock, writes a fail-closed status record, retains timestamped machine,
  JUnit, and baseline bundles, and keeps performance serial. The finalized
  all-layer invocation passed 410 host-free/shared-registry tests (one explicit
  skip), 20/20 lit, 1/1 compiler artifact, 268/268 correctness twice, and 20/20
  performance. Its retained bundle is
  `artifacts/nvidia-release/20260717T003224Z-18866bbb/all/`.
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
- **NVIDIA-TEST-6 is complete (2026-07-16).** The closure audit found no
  remaining ordinary private MMA/PTX probe
  implementation: plugin and hot-path-ratchet compatibility names now delegate
  to shared predicates, while the Tile probe remains intentionally specialized.
  Running the hot-path ratchet immediately after the broad plugin matrix
  exceeded two f16 caps (512³ and 1024³); two isolated serial reruns both
  passed. The disposition is test-state contamination outside the canonical
  isolated performance lane, not a tolerance change or performance regression.
  An AST ratchet now rejects any future exact-device test under `tests/unit`.
  The final topology collects 333 NVIDIA nodes; compiler artifacts pass 20/20,
  exact-device correctness passes 313/313 twice, and the serial measured lane
  passes 20/20. New backward-attention and epilogue families landed directly
  in `tests/device/nvidia`, and the backward nodes are recorded in the
  executable post-migration map.
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

### Install LLVM/MLIR 23 on Ubuntu

Use the repository bootstrap on Ubuntu 24.04; it installs one matched LLVM,
Clang, LLD, MLIR, and Polly 23 toolchain from apt.llvm.org:

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

For a toolchain-only manual installation, use a dedicated versioned source
file rather than replacing the distribution LLVM packages:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/apt.llvm.org.gpg

. /etc/os-release
LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}-23"
if ! wget -q --spider \
  "https://apt.llvm.org/${VERSION_CODENAME}/dists/${LLVM_SUITE}/Release"; then
  LLVM_SUITE="llvm-toolchain-${VERSION_CODENAME}"
fi
echo "deb [signed-by=/etc/apt/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/${VERSION_CODENAME}/ ${LLVM_SUITE} main" \
  | sudo tee /etc/apt/sources.list.d/llvm-23.list >/dev/null
sudo apt-get update
sudo apt-get install -y \
  clang-23 lld-23 llvm-23 llvm-23-dev llvm-23-tools \
  mlir-23-tools libmlir-23-dev libpolly-23-dev

export LLVM_ROOT=/usr/lib/llvm-23
export PATH="$LLVM_ROOT/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

llvm-config --version
mlir-opt --version
mlir-tblgen --version
FileCheck --version
```

All four commands must report major version 23. Remove or disable any stale
pre-23 toolchain source selection from the build environment; keeping
multiple apt repositories installed is acceptable, but Tessera's CMake cache,
compiler executables, MLIR tools, and CMake package directories must all resolve
to `/usr/lib/llvm-23`.

Build the compiler and CUDA runtime from a clean NVIDIA build directory:

```bash
cmake -S . -B build-nvidia-cuda -G Ninja \
  -DCMAKE_C_COMPILER=/usr/lib/llvm-23/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib/llvm-23/bin/clang++ \
  -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir \
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
export MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt
export PYTHONPATH="$PWD/python:$PWD"
```

Adjust `TESSERA_OPT` to the actual Ninja output reported by the build if the
generator places it under `build-nvidia-cuda/tools/tessera-opt/` differently.

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and updates portable Tile/NVIDIA TableGen plus greedy-rewrite
compatibility. The shared sources compile in the LLVM/MLIR 23 ROCm build, and
NVIDIA exact-device parity is now validated independently on the `sm_120`
host. No CUDA execution status was inferred or promoted from the ROCm run.

## Ordered work

| Order | ID | Work | Engineering action | Completion gate |
|---:|---|---|---|---|
| 1 | NVIDIA-TEST-1 | Establish a reproducible baseline | Run collection and each proof layer separately; save JUnit, skip reasons, duration report, machine identity, and the exact commit. Classify every failure as product defect, test defect, environment defect, or stale claim. | Two collections return the same node set; no unknown markers; every skip has an explicit unavailable capability. |
| 2 | NVIDIA-TEST-2 | Compiler-artifact layer | Run `check-tessera-nvidia` plus CUDA pytest files carrying `compiler_tool`; migrate private tool probes to `compiler_toolchain`; split artifact assertions from the eight tests that currently continue into device execution; replace large textual snapshots with named diagnostics, FileCheck, or focused IR/object invariants. | Clean build passes without a GPU; missing-tool simulation skip-cleans; no compiler test invokes a nonexistent path. |
| 3 | NVIDIA-TEST-3 | Exact-device correctness | Run `hardware_nvidia and not performance`; group failures by GEMM/Tile, attention, reductions/norms, control flow, KV/ReplaySSM, collectives, and ABI/conformance. Require native provenance and execute/compare. | Entire correctness matrix passes twice; fallback-injection negatives fail to earn native proof. |
| 4 | NVIDIA-TEST-4 | Numerical policy | Centralize dtype/op tolerances from accumulation/storage behavior. Add ragged, rectangular, boundary, non-finite, misalignment, and invalid-contract cases where absent. | f16/bf16/tf32/FP8/int8/NVFP4 cases use documented tolerances; no default zero-`atol` checks near zero. |
| 5 | NVIDIA-TEST-5 | Measured performance | Run `hardware_nvidia and performance` serially. Warm up compilation and caches; use repeated medians; measure kernel-only and end-to-end separately; record registers, shared memory, occupancy, spills, and selected route. | Stable baselines cover square/rectangular/ragged GEMM, fused epilogues, attention, paged KV, ReplaySSM, reductions, and transport. Each ratchet identifies the selected implementation. |
| 6 | NVIDIA-TEST-6 | Refactor and deduplicate | Move mature families toward `tests/compiler/`, `tests/device/nvidia/`, `tests/integration/`, and `tests/performance/nvidia/`. Consolidate repeated CUDA availability, compilation, launch, oracle, and cleanup code. | No central filename allowlist; no duplicated private CUDA probe/loader; process trees and device allocations clean up on failure. |
| 7 | NVIDIA-TEST-7 | Local release ownership | Own the NVIDIA-box release gate locally in WSL with a host concurrency lock and retained artifacts; GitHub runners are intentionally not used. Keep two-run device correctness required for NVIDIA promotion and performance serial. | A clean branch run reports NVIDIA host-free/shared registries, compiler artifact, device correctness, and performance independently and retains the fail-closed evidence bundle. |
| 8 | NVIDIA-E2E-1 | Canonical SM120 compiler spine | Under sync `E2E-SPINE-2026-07-18`, compose Graph/Schedule/Tile lowering with `LowerTileToNVIDIA(sm=120)`, NVVM/PTX/native-image packaging, and the existing register/invoke launch bridge. Prove f16 and NVFP4 first, including non-origin scale tiles and general-shape dispatch. | One canonical driver request returns a typed image artifact plus launch descriptor, registers and launches on `sm_120`, compares numerically, and retains compiler/ABI/device/resource evidence without a selector change. |
| 9 | NVIDIA-E2E-2 | Per-SM and operation breadth | Replace shared-alias/hardcoded target behavior with architecture-specific pipelines, then move supported CUDA families through the same typed image/launch seam. | Every enabled SM/family has the four-layer proof on its exact device or an explicit unsupported/planned terminal state; `sm_90` and `sm_100` are never inferred from `sm_120`. |

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

**Completion evidence (2026-07-16, `complete`).** The executable map now covers
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

The final static audit finds no `hardware_nvidia` test function under
`tests/unit`; structural marker/release assertions remain host-free. The
expanded map contains 292 relocations plus 23 post-migration nodes. The final
four-layer proof is 20/20 compiler lit, 313/313 exact-device correctness twice,
and 20/20 serial performance on the RTX 5070 Ti. This closes the migration;
future native tests must land directly in device, integration, or performance
roots and satisfy the same AST/node-map ratchets.

## Canonical commands on the NVIDIA box

```bash
# 0. State/collection contract (currently 334 nodes)
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
  -m hardware_nvidia --collect-only -q --no-header

# 1. Host-free PR contract, including CUDA emit/validation/rejection tests
python3 scripts/run_unit_tests.py --timeout=180 -q

# 2. Compiler artifacts without claiming device execution
ninja -C build-nvidia-cuda check-tessera-nvidia
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "compiler_nvidia and not hardware_nvidia" -q --durations=50 \
  --junitxml=/tmp/nvidia-compiler-tool.xml

# 3. Exact-device correctness; run twice from the same clean build
python3 -m pytest tests/unit tests/device/nvidia tests/integration \
  -m "hardware_nvidia and not performance" -q --durations=100 \
  --junitxml=/tmp/nvidia-device-correctness.xml

# 4. Measured lane: serial only
python3 -m pytest tests/unit tests/device/nvidia tests/performance/nvidia tests/integration \
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

### CUDA parity execution record

The parity queue uses ROCm's logical coverage and proof methodology, not its
physical schedules. CUDA owns warp/register packing, `HMMA`/`QMMA`/`IMMA`/OMMA
selection, shared-memory staging, barriers, occupancy limits, and every selector
winner. An AMD wave shape, LDS strategy, or VGPR result is never a CUDA default.

- **NVIDIA-PARITY-TILE — complete on sm_120a.** The architecture-owned SM120 fragment
  selector now describes f16 (f16/f32 accumulation), bf16, TF32, FP8 E4M3/E5M2,
  int8, and block-scaled NVFP4 separately. C++ lowering consumes the descriptor
  for physical input packing and per-lane register-count validation. The exact
  compiler path passes the shared numerical oracle for f16/bf16/TF32/FP8/int8,
  direct/shared grids, ragged edges, and bias/ReLU/GELU/SiLU. The reproducible
  13-row `nvidia_sm120_tile_fragment_resources.json` record retains cubin hashes,
  registers, shared memory, theoretical occupancy, spills, and observed SASS:
  `HMMA` for f16/bf16/TF32, `QMMA` for FP8, `IMMA` for int8, and block-scaled
  `OMMA` for NVFP4. Portable typed Tile now carries NVFP4's two logical UE4M3
  scale tiles. C++ consumes nibble-packed logical A/B storage, materializes the
  backend-owned scale selectors, emits real block-scaled inline PTX, assembles
  for `sm_120a`, and passes the non-uniform-scale numerical oracle without
  fixture-authored physical fragments. The resource row now comes from that
  typed compiler artifact rather than the original CUDA spike.
- **NVIDIA-PARITY-GEMM-RATCHET — measured complete; no promotion.** The
  device-keyed `nvidia_sm120_gemm_schedule_matrix.json` contains 34 exact-case
  rows spanning square, rectangular, ragged, f16/bf16, and bias plus
  none/ReLU/GELU/SiLU epilogues. Every row has two stable repeated-median runs,
  separate CUDA-event and rotated-interleaved end-to-end timing, complete
  per-candidate resource fingerprints, and an explicit 3% noise policy. The
  smallest ragged rows require 50 untimed device warmups to remove clock-ramp
  drift. The record intentionally leaves the production selector unchanged.
  CUDA 13.3's renamed local/shared spill-request metrics are normalized alongside
  the legacy Nsight metrics, and the synthesized fused fallback now has retained
  production-sized Nsight evidence.
- **NVIDIA-PARITY-LEGACY-RETUNE — stable complete; no selector promotion.** The
  device-keyed `nvidia_sm120_legacy_retune.json` now compares compiled exact-f32
  and shipped TF32 GEMM on square/ragged rows, one grouped GEMM launch against
  the retained per-expert decomposition, and a new grouped SwiGLU route whose
  four launches are independent of expert count against the legacy `4E` route.
  All candidates use one f32 oracle and retain separate event/end-to-end rows,
  byte and achieved-bandwidth accounting, launch counts, and linked resources.
  SwiGLU rows retain both the grouped-GEMM cubin fingerprint and the exact
  generated SiLU-gate registers, occupancy, and spill record.
  The final corpus uses production-scale 512-square and 509x773x257 ragged
  GEMM, 1024x384x256x5 grouped GEMM, and 512x256x384x8 grouped SwiGLU rows.
  Exact-route resident warmup occurs after allocation in the timed session,
  and two disjoint interleaved cohorts retain every device and end-to-end batch.
  All eight rows pass 3% (maximum device/end-to-end deltas 2.47%/0.77%). TF32
  and the launch-collapsed grouped routes win both retained runs, but this
  evidence intentionally leaves selectors unchanged.
- **NVIDIA-PARITY-ATTN-FWD — stable complete; CUDA 4-warp candidate leads kernel time.**
  CUDA-owned 4- and 8-warp CTA candidates now cover D=128 MHA, causal sequence
  1009, ragged GQA windowing, and MQA bias+softcap. Each warp owns one query,
  uses warp shuffles for QK, and keeps distributed online-softmax/PV state;
  this is not ROCm's two-wave LDS schedule. All rows match the shared oracle
  within `7e-8`. Both candidates use 56 registers with zero spills; modeled
  occupancy is 75% for four warps and 66.67% for eight. Four warps win CUDA-event
  timing on both retained runs for every case. Ten disjoint, sample-interleaved
  end-to-end batches remove run-order aliasing without sharing observations;
  all eight rows now pass 3% with maximum device/end-to-end deltas of
  0.22%/1.84%. Small end-to-end rows do not have unanimous winner consensus,
  so production selection remains unchanged.
- **NVIDIA-PARITY-ATTN-BWD — measured complete; atomic retained.** The atomic
  incumbent and deterministic two-part split/workspace/fixed-order-reduction
  candidate share one forward-derived oracle across D64 MHA, causal D128 MHA,
  and ragged windowed GQA. The split route is bitwise repeatable, rejects
  unsupported f16 storage, and enforces an exact one-extra-dK+dV f32 workspace
  cap (524,288 bytes on MHA rows; 134,144 on ragged GQA). All six candidate
  rows pass 3% with maximum device/end-to-end deltas of 0.36%/1.67%. Resources
  retain atomic 48-register/83.33%-occupancy and split dQ 48-register,
  dK/dV 56-register/75%-occupancy, and reduction 12-register/100%-occupancy
  fingerprints plus spill evidence. Atomic wins both timing domains in every
  case by a large margin, so `selector_changed` remains false.
- **NVIDIA-PARITY-PAGED-KV — correctness and timing complete; no promotion.** Both fused
  and staged routes now pass the same permuted-page oracle at lengths 1, 3, 4,
  5, 7, 8, 9, and 13, including non-monotonic logical indices and global causal
  offsets. The 13-row transport corpus covers 127/128/129/511-token boundaries
  with separate device/end-to-end keys, byte formulas, resources, and no
  selector change. Repeated event batches now remain inside one warmed resident
  session; all eight fused/staged rows pass the 3% two-run policy, with maximum
  device and end-to-end deltas of 1.89% and 1.85% respectively.
- **NVIDIA-PARITY-REPLAY — canonical state contract and correctness complete;
  timing characterization retained.** Exact-device
  tests cover long decode across flushes, rollback, speculative rejection,
  block submit, reset, ordered ring backpressure, rejected-submit immutability,
  and teardown over wider B/D/N shapes. The 10-row replay corpus spans five
  geometries and 16/64 tokens with traffic, resources, and both timing domains.
  Each runtime handle now carries the shared `tessera.replayssm.state.v1`
  descriptor: exact persistent device and pinned-host byte formulas, session
  lifetime with preserved initialization, ordered stream/event slot ownership,
  consumer-wait-before-release, and teardown draining. Span checks reject before
  CUDA submission. The CPU oracle is outside the end-to-end interval; each
  retained run has disjoint four-route batch medians with recorded out-of-band
  clock conditioning. All errors remain below `1.5e-8`. Under the WSL 4%
  foundation policy 5/10 refreshed rows satisfy both domains; the remaining
  small/multi-batch rows range from 4.05% to 8.75%. No selector decision consumes
  these unstable rows.
- **NVIDIA-PARITY-EPILOGUE — execution matrix complete.** `FusedRegion` is the
  backend-neutral bias/activation/residual/order oracle and now emits registered
  `E_FUSED_EPILOGUE_*` diagnostics for unsupported dtype/op/order and missing
  operands. The exact-device matrix now executes all 43 supported combinations
  over f32/f16/bf16/FP8 E4M3/FP8 E5M2, optional bias, no activation or
  ReLU/GELU/SiLU, and f32 residual-after-activation ordering. Accumulation is
  f32; low-precision residual, activation-before-bias, repeated activation,
  and unsupported dtype/op pairs reject with the registered diagnostics.
- **NVIDIA-PARITY-AUTOTUNE — strict admission complete; no promotion.** Corpus
  admission can require exact device, timing domain, compiler fingerprint,
  resource fingerprints, compile state, and cache state. The committed
  reproducibility record admits all 20 selector-eligible NVIDIA rows, rejects
  stale device/timing/compiler/resource mutations, and reproduces one kernel
  cache key across two cold builds and warm hits (about 0.05 ms warm lookup).
- **NVIDIA-PARITY-TRANSPORT — correctness, evidence, and timing complete.**
  The consolidated 13-row paged-KV/MoE/grouped corpus retains auditable traffic
  formulas, achieved bandwidth, launch-amortization keys, exact resources, and
  independent timing domains. MoE dispatch/combine now consume one canonical
  `tessera.moe_transport.v1` int32/fp32 descriptor with stable expert grouping,
  capacity/drop semantics, and dispatch-before-compute-before-combine ordering;
  grouped GEMM consumes canonical ragged sizes/offsets and retains empty experts.
  Local-device scope is explicit; multi-rank collective execution remains a
  separate backend/runtime item. Maximum oracle error is below `3e-7`; all 13
  rows pass the WSL 4% foundation policy. MoE CUDA-event samples retain one
  native allocation set across repeated batches, and the tiny routes use 101
  medians per run. No selector or legacy-retune winner is promoted.

- **NVIDIA-SM120-LOWP-PRODUCTIZATION — complete (2026-07-18).** The shipped
  CUDA ABI adds general-shape block-scaled NVFP4: packed E2M1 A/B, raw UE4M3
  scale views, M16/N8 grid dispatch, K64 accumulation, ragged zero fill, and
  pre-launch shape/view rejection. Fixed 16x8x64, multi-tile 33x19x129, and
  sub-tile 7x5x31 non-uniform-scale cases match the exact NVFP4 oracle on the
  RTX 5070 Ti. Native one-kernel TF32 and FP8 E4M3/E5M2 fused-epilogue,
  QK-softmax-PV attention, and gated routes now coexist with the composed
  candidates. Two fresh runs use 20 end-to-end medians and 100 CUDA-event
  repetitions per route. The cross-domain 3% gate promotes 11 of 18 retained
  shape/dtype rows; long attention and disagreement rows remain unpromoted.
  The linked 12-row cubin record reports 40-register fused/attention kernels,
  47–48-register gated kernels, 8/32 KiB attention dynamic shared memory,
  shape-dependent 22.92%/6.25% modeled attention occupancy, zero compiler spill
  storage, and the expected TF32 HMMA / FP8 QMMA SASS. Evidence:
  `nvidia_sm120_low_precision_native_{routes,resources}.json`.
- **Audit-document reconciliation — complete (2026-07-18).** This plan,
  `NVIDIA_AUDIT.md`, and `sm120-kernel-guide.md` now agree that mature SM120
  fragments lower for real, general NVFP4 dispatch is executable, and native
  TF32/FP8 transformer candidates exist. The plan remains `landing` only for
  unrelated architecture-specific follow-ons such as sm_90 WGMMA and sm_100
  tcgen05 exact-device proof; landed SM120 work is no longer described as open.

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16` changes the shared typed Tile
operand contract only. NVIDIA supplies exact-device materialization evidence;
Apple and ROCm do not inherit its physical schedule and record their outcomes
in their own plans.

Cross-backend sync `PR420-REVIEW-2026-07-17` corrects the NVIDIA-owned NVFP4
scale materializer to apply both `tile.view` origins using the declared
row-major A-scale and column-major B-scale layouts. A live `sm_120a` fixture
selects nonzero A-row and B-column scale tiles and matches the NumPy oracle;
the NVIDIA compiler lit suite passes 21/21. The SM120 Target IR selector also
accepts canonical `fp16` as the existing f16 fragment contract. This is a
correctness/dispatch repair only: no physical fragment, resource record,
timing row, or production selector changes. The same sync makes Ubuntu LLVM
repository setup install its probe prerequisites before first use; sibling
backend outcomes are recorded in their plans.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18` is NVIDIA-owned. It changes no
shared dtype spelling, portable Tile scale layout, backend-neutral epilogue
order, or generic autotune schema. Apple has no enabled NVFP4 cooperative-matrix
route and ROCm gfx1151 has no FP8/FP4 WMMA instruction; neither inherits CUDA
packing, HMMA/QMMA/OMMA schedules, resource values, timings, or selector rows.

Cross-backend sync `E2E-SPINE-2026-07-18`: NVIDIA owns **NVIDIA-E2E-1** and
**NVIDIA-E2E-2**. Shared code owns only the image/launch schemas and canonical
orchestration; NVIDIA retains PTX/SASS generation, physical fragments, launch
geometry, resources, and route selection. Existing NVRTC, shipped-library, and
PTX-register/invoke paths remain valid candidates while the typed spine lands.
Host-free IR/object evidence cannot promote an SM or selector, and exact-device
proof for `sm_90`, `sm_100`, and `sm_120` remains architecture-specific. The
completed E2E-SPINE-0 foundation records SM80 as lacking an exact registered
pipeline and SM100/SM120 as shared-builder aliases; it also corrects the Python
pass inventory to match that builder without changing CUDA runtime selection.
E2E-SPINE-1 adds the portable image/descriptor and rejection contract only;
PTX/cubin contents, warp schedules, launch geometry policy, resources, and CUDA
selectors remain NVIDIA-owned and unchanged until NVIDIA-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no CUDA hook and
does not reinterpret `nvidia_mma` or any shipped/NVRTC candidate; NVIDIA-E2E-1
still owns PTX packaging, `sm_120` registration/submission, numerical proof,
resources, cleanup, and the first Level-C row.

NVIDIA-E2E-1 is **complete**. The f16 slice makes an explicit canonical
driver request own the typed `tile.matmul_kernel`, runs the production
`LowerTileToNVIDIA(sm=120)` and NVVM/LLVM/PTX pipeline, validates the image with
`ptxas`, and returns the shared native-image plus exact A/B/D/M/N/K descriptor.
The descriptor registers and launches through the shipped PTX bridge on the RTX
5070 Ti; aligned `16x8x16` and ragged `37x29x23` rows match the f32 NumPy oracle.
The image retains compiler/toolchain fingerprints, cold/warm state, and ptxas
register/shared-memory/spill fields. This slice changes no production selector.
The same driver now selects a CUDA-owned general-shape NVFP4 descriptor with
packed E2M1 A/B, logical UE4M3 `scale_a`/`scale_b`, f32 output, and M/N/K. The
typed lowering owns M16/N8 origins, K64 accumulation, ragged zero fill, scale
word materialization, and guarded stores before LLVM 23 emits `sm_120a` PTX.
Exact RTX 5070 Ti rows `16x8x64`, `33x19x129`, and `7x5x31` match the block-scale
oracle; the multi-tile row uses nonuniform row/column scales to prove non-origin
scale views. Missing/malformed scales, wrong scale storage, bias, and malformed
launch shapes reject before CUDA submission. Both f16 and NVFP4 retain stable
cold/warm image identity and ptxas register/shared-memory/spill evidence. The
shared Tile verifier change is limited to the explicit eight-operand NVFP4
launch ABI; it transfers no CUDA schedule, layout, resources, or selector.

NVIDIA-E2E-2 is **closed for the available SM120 host**, with the unavailable
multi-GPU, SM90, and SM100 boundaries assigned the deferred terminal states
below. Its first dependency slice replaces the former
shared SM90 alias with exact SM90/SM100/SM120 Graph→Tile builders and registered
Tile→`tessera_nvidia`→NVVM producers. The exact target now reaches Tile IR,
the control-flow guard, async-copy lowering, and the target producer without
being rewritten to SM90. Hopper alone consumes the proven WGMMA and Hopper
FlashAttention markers; SM100 and SM120 retain target-tagged typed carriers for
architecture-owned lowering. Straight-line async copies mint typed completion
tokens, the matching wait retires them, and matrix consumers preserve those
edges through TMA lowering. Host WSL FileCheck proves the three distinct IR
routes; native SM90 and SM100 remain unsupported-by-evidence until exact-device
runs exist. No selector changes. That breadth statement described the first
landing slice and is now superseded by the implementation record below: SM120
canonical execution covers the complete matmul dtype matrix plus softmax,
reductions, fused epilogues, attention, paged-KV, ReplaySSM, and local MoE.

The next NVIDIA-E2E-2 family slice now gives static f16/f32 last-axis softmax a
canonical Level-C path. `tile.softmax_kernel` carries source/destination,
flattened Rows/K, `storage="f16"|"f32"`, `accum="f32"`, and `axis=-1`; the SM120
materializer emits a stable max-shifted row loop and target-native `nvvm.ex2`
instead of an unavailable NVPTX `fexp` libcall. LLVM 23 emits and ptxas
validates `sm_120a` PTX, while the typed descriptor registers and launches it
through the shipped CUDA-driver bridge. Exact RTX 5070 Ti proof covers shapes
`1x16`, `8x64`, `4x300`, and `2x3x48`, extreme logits, malformed output shape
rejection, stable cold/warm image identity, and resource/spill fields for both
storage types. f16 loads extend before the max/sum/normalization loops and
truncate only at output storage. This is a correctness-first 128-thread,
one-thread-per-row candidate. The existing cooperative CUDA-C route remains
selected. All four final canonical/production rows are stable in both timing
domains, and production wins both domains for the production-sized and ragged
cases.

The following NVIDIA-E2E-2 dtype-totality slice centralizes consumer-Blackwell
storage, math-mode, scalar/vector, Tensor Core, compiler, and runtime states in
`nvidia_dtype_contract.py`. Every canonical float storage type now has an
explicit row. CUDA 13.3 compile proof covers scalar/vector forms for fp64,
fp32, fp16, bf16, FP8 E4M3/E5M2, FP6 E2M3/E3M2, and packed FP4; TF32 remains
strictly an fp32 `math_mode`, never storage. Tensor Core Target IR/PTX rows now
cover the required TF32, bf16, fp16, FP8, FP6, FP4, and int8 families. The
canonical descriptor lane now executes BF16, explicit fp32-storage TF32 math,
FP8 E4M3/E5M2, and INT8 with int32 accumulation. FP64 m8n8k4 DMMA now owns a
distinct Tile lane map and f64 descriptor/bridge ABI; aligned and ragged RTX
5070 Ti rows match the f64 oracle with masked tails.
FP6 E2M3/E3M2 now assemble as `kind::mxf8f6f4`, m16n8k32,
UE8M0/`scale_vec::1X`; OCP/MXFP4 assembles as `kind::mxf4`, m16n8k64,
UE8M0/`scale_vec::2X`. Compiler-owned packed-memory Tile materializers,
five-buffer descriptors, CUDA-driver launch ABIs, and aligned/ragged numerical
proof now cover both FP6 encodings and MXFP4. In particular,
`fp4_e2m1` does not alias NVFP4: MXFP4's UE8M0 scale contract cannot reuse
NVFP4's UE4M3/`scale_vec::4X` scale words.
The shared MMA selector now requires explicit `math_mode="tf32"` for fp32 and
retains distinct `nvfp4` and `fp4_e2m1` K64 identities. No selector promotion
or production route changes.

The canonical dtype execution matrix records two disjoint, sample-interleaved
runs for square and ragged fp64/fp16/bf16/TF32/FP8/FP6/MXFP4/INT8 routes, with separate
CUDA-event and allocation/copy-inclusive timing, cold/warm image identity, and
ptxas register/shared-memory/spill fields. The retained 20-row collection
changes no selector. The final 31-sample, 10,000-device-launch and
50-end-to-end-launch run has 19/20 rows stable in both timing domains. The only
terminal miss is TF32 `256x256x256`: its device cohorts remain bimodal at 7.02%
while end-to-end is stable at 0.59%. That row is explicitly non-promoting; the
existing selector is retained rather than hiding the exact-device result.

The broader-family NVIDIA-E2E-2 reduction slice now carries
`tile.reduce_kernel(X,O,Outer,AxisExtent,Inner)` and an SM120-owned v2
materializer/descriptor ABI for f16/f32 sum, mean, and NaN-propagating max.
Normalized arbitrary axes and keepdims shape contracts execute through both a
single-owner serial schedule and a 128-thread cooperative shared-memory
candidate. Exact RTX 5070 Ti proof covers axes 0/1/2, keepdims on/off,
rectangular/ragged rank-3 inputs, f32 accumulation, non-finite values,
image/resource retention, and 42 numerical rows. The earlier last-axis record
remains historical evidence; the new comparative record applies the WSL 4%
foundation policy in both timing domains and changes no selector.

The canonical epilogue slice carries f16/bf16/TF32/FP8 E4M3/E5M2 bias,
ReLU/GELU/SiLU, optional f32
residual, and the explicit `matmul -> bias -> activation -> residual` order in
the Tile kernel plus launch descriptor. The CUDA materializer consumes distinct
bias/residual buffers and rejects unsupported dtype/order/shape contracts
instead of silently dropping epilogue semantics. The original 32 f16/bf16 rows
and the 48-case TF32/FP8 matrix pass exact-device execution. The comparative
record measures canonical single-kernel images against the existing production
composed routes, retaining both timing domains, cold/warm state,
image/resource fingerprints, spills, and raw disjoint cohorts. Production
selectors remain unchanged unless both domains select the same stable winner.

The first canonical attention slice adds a shared typed
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
f16/f32 storage, f32 accumulation/output, positive scale, and causal semantics.
The SM120 correctness-first materializer and four-buffer descriptor launch
through the shipped PTX bridge; exact RTX 5070 Ti proof passes 8/8
MHA/MQA, rectangular/ragged, causal/non-causal cases with zero spills. The
entry symbol includes the scale/causal semantic digest so incompatible images
cannot alias in the driver cache. Bias, window, softcap, dropout, and backward
are completed below. The retained eight-row
two-cohort baseline records CUDA-event and allocation/copy-inclusive timings,
cold/warm image identity, resources, and raw samples. A higher-amortization
rerun now has 8/8 rows within 3% in both domains. It remains historical
evidence; the final comparison below owns the production disposition.

The forward carrier now also owns optional dense f32 bias, signed left/right
window bounds, arithmetic softcap, and deterministic `lcg32_counter_v1`
dropout. These semantics participate in the image digest and descriptor
provenance. An exact-device advanced row proves causal+window+bias+softcap,
bitwise dropout replay, and malformed-bias rejection; the earlier 8-row
MHA/MQA matrix remains green. The f32 backward reference now also crosses the
compiler-owned seam through `tile.attention_backward_kernel` and a seven/eight-
buffer native descriptor. It assigns one dQ/dK/dV element to one thread,
performs fixed-order single-owner dK/dV reduction, requires
`deterministic=true`, and declares zero workspace. The exact-device GQA row
proves causal+window+bias+softcap derivatives, bitwise replay, descriptor-shape
rejection, and agreement with the shared Pade-softcap oracle. The final semantic
slice below adds matching f16 storage and dropout-mask replay.

The refreshed backward candidate matrix passes 6/6 exact-device oracle,
determinism, and workspace cases. All six atomic/split rows are stable in both
timing domains. Atomic wins both domains for MHA D64, causal MHA D128, and
ragged GQA; split/reduced remains the bitwise-repeatable option with one extra
dK+dV f32 workspace (134,144--524,288 bytes in the retained shapes). Production
already selects atomic, so the evidence retains that selector. The canonical
deterministic reference carrier is now landed; production selection continues
to be governed by the stable atomic/split corpus rather than the intentionally
serial reference materializer.

The paged-KV landing slice adds `tile.paged_kv_read_kernel` and a compiler-owned
f32-pages/i32-table direct descriptor ABI. Four exact-device boundary ranges,
two non-identity physical-page permutations, remap/reuse, and invalid-table
rejection pass. The existing 12-case fused/staged suite also remains green,
including causal offsets and page boundaries. The committed
`nvidia_sm120_e2e_spine_paged_kv.json` corpus compares canonical Tile-direct
against legacy CUDA staged gather at 128, 512-ragged, and 2048-ragged tokens.
It retains two repeated medians in both timing domains, cold/warm image and
cache state, registers, shared memory, occupancy, spills, and resource
fingerprints. This WSL foundation lane uses a 4% repeatability policy because
its graphics clocks are host-managed. All six candidate rows are accepted; the
legacy 2048 device-event row uses an explicit five-basis-point WSL margin at
4.02%, and margin-accepted rows are selector-ineligible. Timing-domain winners
also disagree at 512/2048, so the selector
remains unchanged. The SM120 foundation disposition is closed as retain-existing;
a future native-Linux controlled-host promotion attempt is a separate
hardware-environment follow-up, not an open migration dependency.

The stateful/MoE image slice adds compiler-owned Tile→NVIDIA→PTX packages for
ReplaySSM decode/flush and local f16/bf16/f32 MoE dispatch/combine/ragged
grouped GEMM.
The resident Replay handle no longer embeds those device kernels in its CUDA
host bridge: it loads the compiler-produced PTX functions while retaining the
session-persistent allocations, asynchronous ring, events, and ordering
contract. Compiler-owned MoE candidates launch through the generic descriptor
submission path. Exact RTX 5070 Ti tests cover
dispatch/combine numerical order,
zero-sized expert groups, ragged grouped GEMM, Replay transitions, persistent
workspace metadata, image identity, and resource retention.

The final comparative record contains 14 strict-stable rows for cooperative
softmax, four-warp forward attention, and local MoE dispatch/combine/grouped
GEMM. Every row retains two CUDA-event and allocation/copy-inclusive cohorts,
the discarded first lifecycle launch, 100-launch end-to-end amortization,
per-candidate clock conditioning, cold/warm image state, and exact resource
fingerprints. Production softmax and attention win both domains. MoE does not
produce cross-domain consensus across all three routes, so the existing MoE
selector is retained. ReplaySSM's higher-amortization 10-row matrix is now
10/10 stable in both timing domains. No selector changes.

The collective follow-on adds an explicit content-addressed rank/device
topology and a one-process/multiple-device NCCL executor for all-reduce,
all-gather, reduce-scatter, and grouped send/receive all-to-all. This host
exposes one CUDA device, so it proves deterministic topology/rejection and
records the two-device request as unavailable; it cannot supply the required
two-or-more-GPU numerical, topology, resource, or timing evidence. RCCL and
Apple mappings remain architecture-owned follow-ups. No collective or MoE
selector changes.

The remaining-dtype/reduction performance corpus uses production-sized square
and ragged TF32/FP8 fused epilogues plus f16/f32 arbitrary-axis reductions.
For every candidate it records first-use compilation/cache fill separately,
discards the first launch, and amortizes each device-event and end-to-end sample
over the next ten launches. Two disjoint time-interleaved 100-sample cohorts
retain raw samples, cold/warm or first/second-use state, image/resource
fingerprints, registers, shared memory, and spill fields. All 30 rows are
accepted under the WSL foundation rule: 29 pass the strict 4% gate and the
production fp16-mean reduction end-to-end row is explicitly margin-accepted at
4.099% under the user-approved 4.15% rounding bound. That row is
selector-ineligible. Seven strict rows have cross-domain winner consensus, but
the record changes no selector because stable consensus alone does not establish
a promotion policy or required material benefit.

The final SM120 semantic slice removes the remaining execution limitations.
The deterministic attention VJP now accepts matching f16 or f32 dO/Q/K/V and
gradient storage, accumulates in f32, and replays the forward
`lcg32_counter_v1` dropout mask from the semantic seed without a saved-mask
workspace. A compiler-owned `tile.paged_attention_kernel` consumes Q, K/V
pages, the i32 remap table, i64 logical token indices, and an explicit causal
offset in one fused descriptor; the offset is never inferred from allocation
capacity. MoE dispatch, deterministic combine, and ragged grouped GEMM now
accept f16, bf16, or f32 storage with int32 metadata, f32 combine weights, and
f32 grouped accumulation. Exact RTX 5070 Ti tests prove numerical agreement,
dropout bitwise replay, page remapping/causal boundaries, malformed metadata
rejection, and low-precision MoE execution. This shared Tile-carrier extension
transfers no CUDA schedule: Apple is not applicable because it owns a separate
resident paged-attention ABI and mature low-precision dispatch paths; ROCm
requires architecture-owned lowering before claiming these carrier variants.

Two hardware boundaries now have formal **deferred terminal** states for this
work item:

- exact two-or-more-GPU NCCL topology, numerical, resource, and timing proof is
  deferred because the available SM120 WSL host exposes one GPU;
- exact SM90 Hopper and SM100 datacenter-Blackwell Level-C evidence is deferred
  because neither exact target is available. Their compile-only Level-B
  artifacts do not inherit SM120 execution evidence.

Deferred hardware terminals do not authorize selector changes and do not hide
missing evidence. A future hardware follow-up must reopen its own exact-device
item under synchronization key `E2E-SPINE-2026-07-18`.

E2E-SPINE-3 is a shared-contract follow-up under the same synchronization key.
It is applicable to NVIDIA only as a family-granular evidence envelope around
the existing SM120 results: fixture identity, Level-C provenance, cold/warm
cache identity, benchmark metadata, and hash-sealed release-packet validation.
It changes no CUDA schedule, ABI, dtype capability, or selector. SM90, SM100,
and exact multi-GPU rows remain explicit hardware-deferred terminals and may
not inherit the SM120 packet.

The E2E-SPINE-3 exact-host recorder packages shared f32 softmax and reduction
fixtures through the existing SM120 compiler-owned image/descriptor seam,
proves cold/warm identity, retains selected route plus ptxas resource
fingerprints, and records interleaved repeated-median device-event and
end-to-end rows. The hash-sealed WSL RTX 5070 Ti packet is checked in against
landed source commit `9f3757ef2dda2dd61ff94f1aefe0244f1b80f064`; all four
rows pass the unchanged 4% stability gate after increasing short-kernel and
end-to-end amortization. The fleet dashboard therefore marks only SM120
softmax and reduction release-ready. Matmul, epilogue, attention, paged-KV,
ReplaySSM, and MoE packet scope remains explicit `packet_pending`. No CUDA
selector changed.

The LLVM-stage device-library follow-on makes CUDA `libdevice` an explicit
compiler dependency rather than accidental driver behavior. Native-image
identity now retains logical device-library name, content digest, and link mode
without serializing host paths. The SM120 packager fingerprints
`nvvm/libdevice/libdevice.10.bc` and uses `llvm-link --only-needed` whenever
translated LLVM IR retains an unresolved `__nv_*` call. A real `__nv_sinf`
fixture links through CUDA 13.3 libdevice, lowers with LLVM 23 `llc`, and
assembles with `ptxas -arch=sm_120a`. Intrinsic-only kernels retain an empty
linked-library set, while the available libdevice digest still participates in
the toolchain/cache fingerprint. This changes no runtime selector.

The CUDA floating-point follow-on separates three semantic routes: IEEE
arithmetic operators, function-specific CUDA libdevice calls, and explicit PTX
approximations. The shared softmax envelope now carries
`exp_mode="approx_exp2"` and `ftz=false`; SM120 accepts only that proven mode
and lowers it to `ex2.approx.f32`. The contract records PTX's full-range 2-ULP
bound, requires a nonzero near-zero comparison budget, and versions native
cache identity independently of `-O3`. It does not reuse the `__expf` accuracy
table for a different instruction and does not enable global fast math.
The semantic authority is NVIDIA's
[floating-point computation appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html);
instruction-specific accuracy comes from the
[PTX `ex2` specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2).

The CUDA Math API scalar/integer follow-on records representative integer math,
bit, packed-dot, numeric/bit-cast, and 2x16/4x8 packed-SIMD families. A CUDA
13.3 `nvcc -arch=sm_120a` fixture proves the documented symbols compile, while
the SM120 contract keeps their Tessera Target-IR and runtime states `planned`.
The shared rounding vocabulary now represents CUDA's four conversion suffixes
RN/RD/RU/RZ exactly; nearest-away and stochastic modes cannot silently map to a
CUDA cast. Undefined signed-min absolute value, out-of-range float-to-integer
conversion, funnel-shift wrap/clamp, signedness, lane width, and saturation are
retained as contract boundaries. No public op, runtime route, or selector is
added. Sources: [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html),
[integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html),
[integer math](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html),
[casts](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html),
and [packed SIMD](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

The PTX 9.3 truth audit now separates CUDA C++ storage spelling from physical
PTX typing. Fundamental storage rows are fp64 `.f64`, fp32 `.f32`, fp16 `.f16`,
and int8 `.s8`; BF16, TF32, FP8, FP6, FP4, and NVFP4 are alternate instruction
formats carried in same-width bit registers. Tensor fragments explicitly name
`.f64` or packed `.b32` operands. This corrects BF16 scalar/vector status to
`conversion_only` and prevents CUDA header types from implying fundamental PTX
types. PTX operand compatibility never performs automatic numeric conversion.
Direct `ptxas -arch=sm_120a` proof assembles the fundamental register surface
and rejects `bf16`, `tf32`, `e4m3`, and `u8x4` as register declarations.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It adapts the
shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, packages
HSACO, and submits through an exact gfx1151 HIP descriptor hook. NVIDIA's
`tessera_nvidia` lowering, `ex2.approx.f32` math contract, PTX ABI, SM120
schedule, resource/timing evidence, and selectors are unchanged. No AMD
wave/LDS or OCML behavior transfers to CUDA. ROCm's subsequent use of the
shared device-library record for its driver-selected OCML/OCKL/OCLC set is
parity validated at the schema boundary and requires no CUDA record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to NVIDIA target state. It adds no canonical dtype or alias and does
not change the SM120 PTX storage/Tensor Core contract, fragment ABI, runtime
readiness, or selector; it only prevents RDNA3.5 ISA formats from being
conflated with Tessera gfx1151 execution support.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. NVIDIA parity is validated at that
logical contract; NVFP4 and NVIDIA packed-weight/Tensor Core ABIs remain
distinct and backend-owned. No PTX capability, fragment ABI, runtime route, or
selector is promoted by the gfx1151 proof, and unsigned packed-4 remains
unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so CUDA schema parity is validated; no NVIDIA ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens its portable verifier to
admit bf16. NVIDIA's backend-specific materializer still explicitly accepts
only f16/f32, so its op registry, `Outer/AxisExtent/Inner` schema, serial/cooperative-128
lowerings, PTX ABI, resources, exact-device evidence, routes, and selectors are
unchanged; the ROCm five-argument HSACO ABI transfers no CUDA claim.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. NVIDIA's existing direct PTX mapping remains parity validated; no ROCm
gather schedule, HSACO ABI, page-table validation evidence, timing, readiness,
or selector state transfers to CUDA.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. NVIDIA's typed PTX mapping remains
parity validated at the carrier boundary; no AMD gather schedule, HSACO ABI,
gfx1151 evidence, timing, readiness, or selector state transfers to CUDA.

The accompanying PTX memory contract records CTA/cluster/GPU/system scopes and
relaxed/acquire/release/acq_rel atomic semantics. Vector and packed memory
accesses are sets of scalar accesses in unspecified element order, not one
atomic unit; mixed-size races fall outside the model; `red` does not form an
acquire pattern; texture/`ld.global.nc` accesses are excluded; ordered CUDA
submission does not establish intra-kernel memory order. Sources:
[types and state spaces](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#state-spaces-types-and-variables),
[instruction operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-operands),
and [memory consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

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

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
CUDA remains `native_gpu` with its existing event and end-to-end timing domains;
no PTX ABI, SM schedule, device evidence, readiness, or selector state transfers.
The x86 pilot consumes existing Tile softmax/reduction carriers without changing
their shared dtype or operation registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. NVIDIA inherits no
x86 ABI, vector schedule, host timing, readiness, or selector state. SM120
GQA/dropout and dtype breadth remain governed by NVIDIA-owned Target IR and
exact-device evidence; x86's narrower descriptor contract changes no CUDA row.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. NVIDIA parity is not applicable; no NVIDIA pipeline,
PTX ABI, schedule, capability, or selector changes. X86-E2E-2 subsequently
closed the remaining inventory and reassessed NVIDIA at each shared-contract
boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. NVIDIA parity is assessed at the carrier boundary only;
the AVX-512 ABI, CPU schedule/timing, 16K binary selector threshold, and exact
x86 evidence transfer no PTX implementation or CUDA selector claim. Existing
NVIDIA elementwise target and execution rows are unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. NVIDIA inherits no C ABI, null-operand convention,
32K selector threshold, CPU timing, PTX implementation, or CUDA selector
claim; NVIDIA target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
NVIDIA parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, PTX routes, and CUDA selectors
do not transfer. Existing NVIDIA rows remain unchanged.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. NVIDIA already owns
independent dtype, MMA, accumulator, PTX, and runtime contracts; no CUDA target,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, NVIDIA target capabilities, PTX ABIs, schedules, and
selector state are unchanged; NVIDIA parity is validated by the shared
attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; NVIDIA uses the
versioned apt LLVM 23 packages alongside CUDA 13. NVIDIA target semantics, PTX
ABIs, and selectors are unchanged, and the LLVM 23 compiler/lit build validates
host-free parity; exact-device claims remain NVIDIA-owned.

The collection contract, compiler-artifact, exact-device correctness, and
serial measured lanes now have recorded baselines. NVIDIA-TEST-5 and
NVIDIA-TEST-6 are closed; the requested attention, epilogue, and legacy-retune
parity records are stable without a selector promotion. Keep `plan_state:
landing` while unrelated implementation or exact-device follow-ons remain.
Move this plan to the NVIDIA archive only after every completion gate is met.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes the NVIDIA families as a **lead performance target** (Decision #28 — its
`wgmma`/`mma.sync` candidates set the ceiling and are never capped by the shared
mixer framework). It adds candidates under existing families, opening no new
NVIDIA-TEST item: channel-wise KDA/GDN decode → **NVIDIA-TEST-3/-5 KV/ReplaySSM**;
`sliding_window`/full mixer fwd + backward → **attention** (split/reduced dK/dV,
G6-C-style); chunkwise-scan inner GEMMs → **GEMM/Tile** (`wgmma` sm_90 / `mma.sync`
sm_120, preferably via the NVIDIA Tile IR lowering target); NVFP4/MXFP8 mixer GEMMs
→ **NVIDIA-TEST-4** numerical policy (this is the executing FP4 lane — sm_120
`mma.sync`, not `tcgen05`). Inherits the TEST-3 native-provenance / TEST-5
kernel-vs-E2E evidence contract unchanged. Direction pointer only; no NVIDIA gate,
route, or exact-device claim changes here.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. NVIDIA parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no PTX/CUDA implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, PTX/CUDA ABI, NVIDIA schedule, dtype capability,
execution row, or selector. NVIDIA parity is therefore not applicable.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the NVIDIA not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` hardens the LLVM 23 and
Linux TSAN host environment and repairs an Apple-only capability row. NVIDIA
parity is not applicable: no CUDA dtype, PTX ABI, schedule, execution row,
selector, or exact-device evidence changes.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus. NVIDIA fixture-schema parity
is validated, but the gfx1151 HSACO, HIP ABI, resources, timing, and
exact-device evidence do not transfer to CUDA. The ROCm-owned compiler lane
explicitly excludes `compiler_nvidia`; no NVIDIA capability, test ownership,
schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` closes shared Graph/Neighbors
verifier gaps and records the shared `sm_120` MMA selection in NVIDIA manifest
rows. Equal-tier candidates may use its analytical accumulator footprint only
after route-tier precedence. This is parity validated at the host-free
compiler/manifest boundary; it changes no PTX instruction schedule, automatic
selector promotion, CUDA ABI, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` makes compute dtype
legalization the default in NVIDIA named pipelines. Terminal storage
legalization remains intentionally opt-in because the generic CUDA route has no
packed-storage consumer; that consumer is follow-up required before a sub-byte
default is honest. The executable row-major layout materializer and guarded
dynamic launch are x86-only and transfer no PTX schedule, CUDA ABI, bucket
policy, selector, or exact-device evidence.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. NVIDIA remains **follow-up required** for an architecture-owned
Graph-cast materializer; the pass stays opt-in and transfers no PTX layout,
schedule, selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to bucketed tensor-core routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no CUDA backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to bucketed
tensor-core routes; no CUDA ABI, schedule, selector, backward runtime, or
exact-device claim transfers. NVIDIA's architecture-owned Graph-cast consumer
is host-validated: after shared legality it accepts row/column-major/BHSD/NHWC,
removes the Graph marker, and carries the binding into `tile.async_copy`.
This changes staging metadata only and claims no PTX schedule or device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. NVIDIA is **follow-up required** for backward execution:
no PTX/CUDA ABI, affine gamma/beta contract, tensor-core schedule, selector,
runtime binding, performance result, or exact-device proof is added here.
Dynamic statistics remain Graph IR until an NVIDIA-owned materializer lands.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. NVIDIA is **follow-up
required** for an architecture-owned dynamic affine normalization materializer
and backward runtime: the gfx1151 HSACO and AVX-512 ABIs, schedules, timing,
and exact-device evidence do not transfer to CUDA/PTX. Shared static/dynamic
linalg and CPU-oracle proof validate the Graph contract only; no NVIDIA
selector, execution row, or device claim changes.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. NVIDIA still
requires its own CUDA/PTX backward materializer and exact-device proof; the
gfx1151 two-kernel schedule, bitwise evidence, and timing do not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. NVIDIA remains **follow-up required**: neither the gfx1151 HSACO ABI nor
the AVX-512 f32 ABI, schedule, dtype-accumulation contract, timing, or device
evidence transfers to CUDA/PTX. The shared Graph adjoint and dynamic Linalg
contract remain parity validated; no NVIDIA execution row or selector changes.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. NVIDIA parity is host-validated through the shared linalg contract.
All NVIDIA backend variants now execute Tile buffer reuse and materialize one
address-space-3 shared-memory arena with typed planned-offset views before
Tile-to-NVIDIA/NVVM lowering. Function-budgeted liveness-aware
rematerialization also runs in the shared production post-autodiff pipeline.
Exact CUDA/PTX shared-allocation assembly, occupancy, backward reduction
launch, and performance evidence remain follow-up required; no device or
selector promotion is claimed.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. NVIDIA parity is host-validated
at the shared IR boundary. The gfx1151 HIP composition/module cache and
AVX-512 execution do not transfer to CUDA/PTX; an NVIDIA-owned compiled MSE
backward launch, tensor-core policy, and exact-device evidence remain
follow-up required.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized address-space-3 arena planning and a benchmark-fed
rematerialization cost contract. NVIDIA retains opt-in Graph layout assignment
through its existing materializer. The new MSE backward launch and numerical
proof are ROCm gfx1151-only; CUDA/PTX still needs an architecture-owned VJP,
dynamic shared-allocation assembly/occupancy proof, and selector evidence.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. NVIDIA is **follow-up required** for architecture-owned CUDA/PTX
backward materialization and exact-device evidence. The gfx1151 generated HIP
kernel, AVX-512 C ABI, boundary timing, caches, and selector state do not
transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. NVIDIA is **follow-up required** for CUDA/PTX backward
materializers and exact-device evidence; the gfx1151 and AVX-512 loss and
optimizer ABIs do not transfer. No NVIDIA selector or support claim changes.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. NVIDIA
parity is validated only at the shared Graph/Linalg contract. NVIDIA remains
**follow-up required** for an architecture-owned CUDA/PTX fused training
materializer and exact-device evidence; gfx1151 HIP and AVX-512 ABIs, cache
identities, timings, and selector decisions do not transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. NVPTX is expected to lower this form to
shared memory, but exact CUDA assembly, resource, occupancy, and performance
evidence remain follow-up required and are not inferred from gfx1151. The
measured rematerialization corpus has gfx1151 and AVX-512 rows only; no NVIDIA
selector or default policy changes.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. NVIDIA remains **follow-up
required** for CUDA measurements and policy selection. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no PTX ABI, shared-memory allocation,
packed consumer, performance, or selector claim. The existing NVIDIA
architecture-owned layout consumer remains unchanged.
