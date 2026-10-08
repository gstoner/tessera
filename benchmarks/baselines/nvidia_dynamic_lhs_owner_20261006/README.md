# SM120 native ownership for bounded producer-to-matmul frames

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-DYNAMIC-LHS-OWNER-2026-10-06.

## Architectural increment

The named FP16/BF16 RMSNorm, LayerNorm and last-axis softmax -> matmul programs
retain their verified Graph/Schedule/Tile/Target/NVVM/LLVM component images.
C++ now retains both modules, one ordered stream, pinned/device staging and
private intermediate storage across independently bounded M/N/K frames.

Prepare binds immutable capacity extents and exact/static versus bounded axes.
A private native setter admits dynamic mode only before producer attachment/
first invocation, for column-major storage and a loaded parameter ABI with
ordered pointer arguments followed by M/N/K/LDA/LDB/LDD. Driver parameter
reflection verifies widths, offsets and exact count. A static image cannot be
reinterpreted as a dynamic kernel. No image or Tile arithmetic is reconstructed.

Each invocation derives active M/N/K from checked host views. Every unset axis
remains exact, every bounded extent must be positive and within capacity, RHS K
must agree, and all dtype/rank/byte/output/compact-pitch contracts are checked.
Active sizes determine transfers, private intermediate span and native leading
dimensions. Capacities and module identity never change. Grow-only shared
scratch is leased under the native mutex through upload, both kernels,
readback and final caller output copy. Shrinking/expanding within a previously
warmed capacity requires no new allocation.

The existing portable typed witness and unique live-context owner cache now
admit these dynamic owners. Warm shape changes reuse one handle without
manifest readmission, module construction or a Python device session. The
serialized artifact still contains no CUDA handles. Explicit clear retires
owners; context mismatch, fork and closed-handle guards remain in force.
Failure after a real producer queues drains its stream before retirement and
leaves caller output untouched. Python binds/compacts host views and the
compiler certificate; native kernels compute all intermediate values.

## Exact-device validation

Super-Bear's RTX 5070 UUID/driver/compute capability 12.0 are queried by the
recorder. 589 owning/native-contract/registry checks pass, plus 97 existing
dynamic frontend checks and two targeted portable-cache tests: 688 total,
including 332 device cases. No skips in these lanes. Captured logs retain
three deprecation warnings, including intentional multithreaded fork guards.
The new owner suite has 100 cases: all seven axis subsets, all producers,
FP16/BF16, plain/fused output, large first frames, shrinking/ragged/singleton/
changed inputs, independent output lifetime, no scratch growth, mixed-shape
concurrency, raw native byte/pitch/dtype/capacity guards, static ABI refusal,
context recovery, failed-consumer retirement, compiler-free warm portable
reuse and two live contexts. Four fresh compiler-disabled dynamic replays
remain included in the existing frontend suite.

## Matched benchmark

Four independent processes run control/prepared then reverse process order.
Each has 48 correctness-gated rows: 12 producer/dtype/epilogue groups, each
reusing one capacity M,K,N = 128,1024,64 at four active shapes:
128,1024,64; 17,35,19; 1,1,1; and 63,511,31. Three measured warm portable wall
samples follow an untimed checked call. Matching image/Schedule/ABI/certificate/
compiler/runtime/source hashes are required by the analyzer. Prepared receipts
must name the dynamic C++ owner, control must use the checked resident route,
and scratch capacity/allocation counts must remain unchanged across shapes.

Compile wall and separate producer/consumer CUDA-event dispatch windows remain
recorded. Event windows include driver gaps. Warm portable wall includes the
route's CPU binding/admission, transfers, two launches, completion and readback.
This evaluates host ownership, not a kernel speedup or quantized format choice.
Final median per-case prepared/control warm-wall ratios are 0.120738 forward
and 0.120296 reverse, approximately 88% lower portable host-call overhead.
All 192 correctness-gated rows pass; per-case ratios and separate component
dispatch medians are retained in analysis.json.

## Reproduction and remaining scope

On Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run tests/device/nvidia/test_prepared_dynamic_lhs_owner.py and the captured
regression/frontend lanes. Run benchmarks/nvidia/benchmark_dynamic_lhs_frontend.py
with TESSERA_NVIDIA_PREPARED_LHS_REPLAY=0 or 1 and TESSERA_DYNAMIC_LHS_PACKET set
to the JSON destination. Run this directory's analyze.py from repository root.

This supersedes the prior dynamic-frontend packet's descriptor-only ownership
boundary for the named host-array envelope. Ordinary JIT bounded-shape cache
selection, dynamic row-major RHS, general producer composition/AD,
asynchronous/resident native ownership, wider formats and sibling physical
consumers remain open. The full five-slice objective remains active.
