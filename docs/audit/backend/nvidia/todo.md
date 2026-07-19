---
audit_role: plan
plan_state: landing
owner: NVIDIA backend
target: nvidia_sm120
last_updated: 2026-07-18
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
`llvm-toolchain-noble-22` source selection from the build environment; keeping
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
- **NVIDIA-PARITY-REPLAY — correctness and timing complete.** Exact-device
  tests cover long decode across flushes, rollback, speculative rejection,
  block submit, reset, ordered ring backpressure, rejected-submit immutability,
  and teardown over wider B/D/N shapes. The 10-row replay corpus spans five
  geometries and 16/64 tokens with traffic, resources, and both timing domains.
  The CPU oracle is outside the end-to-end interval; each retained run has 100
  disjoint four-route batch medians with recorded out-of-band clock conditioning.
  All errors are below `1.5e-8`; maximum device and end-to-end two-run deltas are
  0.93% and 1.58%.
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
  independent timing domains. Maximum oracle error is below `3e-7`; all 13 rows
  pass the 3% two-run policy. MoE CUDA-event samples retain one native allocation
  set across repeated batches, and the tiny routes use 101 medians per run. No
  selector or legacy-retune winner is promoted by this evidence.

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
