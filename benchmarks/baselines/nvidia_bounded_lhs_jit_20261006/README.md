# Ordinary bounded SM120 producer-to-matmul JIT

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-BOUNDED-LHS-JIT-2026-10-06.

## Compiler and runtime contract

The public jit shape_bounds option declares positive M/N/K capacities for
the named FP16/BF16 RMSNorm, LayerNorm and last-axis softmax -> matmul route.
A first call below capacity traces its actual operands and verifies the
original semantic Graph before projecting capacity and dynamic dimensions.
Native Graph/Schedule/Tile/Target/NVVM/LLVM packages compute the producer and
consumer; native C++ retains their modules, ordered stream and intermediate
storage. Warm bounded axes reuse the same program without tracing or compiling.

A straight-line source certificate and the live Python bytecode constrain
this admission to canonical direct operations, assignments and a final return.
Shape-dependent Python branches, hidden helpers, generic control flow and AD
require further frontend integration. The certificate checks callable identity
and code on reuse, rather than assuming one trace proves all shapes.

Bounds are copied immutably. Active shape, rank, dtype, K agreement and fused
bias/residual roles are checked. Dtype and undeclared static axes specialize
separately. Per-function programs and native owners are limited to 24 entries;
eviction retires associated native ownership. Unique live CUDA context identity,
fork refusal, close/rebind and independent returned-output lifetime are retained.
Host C/F RHS arrays are packed to the existing dynamic column-major ABI.
This does not implement a native dynamic row-major RHS image.

## Exact-device tests

Super-Bear WSL reports NVIDIA GeForce RTX 5070, UUID
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, compute capability 12.0.
The matching compiler/runtime are fingerprinted in each benchmark packet.

547 regression tests pass without skips, including 211 device cases and
336 host tests. The new bounded ordinary-call suite contains 44 device cases.
All seven axis subsets, FP16/BF16, all three producers, plain/fused epilogues,
C/F host RHS, first calls below capacity, expanding/ragged/singleton shapes,
changed inputs, argument permutations, static-axis specialization, capacity
guards, 24-entry retirement and two-context reuse are covered.
Warm-call tests forbid tracing/compiler reconstruction. Host tests reject
false supplied source certificates and malformed original Graph result shapes.
See regression-tests.txt, focused-tests.txt and device-tests.txt for lane logs.
Generated compiler progress/freshness documents are regenerated; 11 audit
tests pass, focused Ruff and diff whitespace checks pass. Graphify update was
attempted but its CLI is absent from this WSL host.

## Matched measurements

Four independent processes execute control/prepared and reverse process order.
Each records 48 cases: two storage types, three producers, two epilogue modes
and four active M/K/N frames under capacity 128/1024/64.
Every group starts with a cold ordinary call at 17/35/19.
An untimed checked call precedes three measured warm samples per frame.

All 192 numerical rows pass, with public output also compared bitwise against
portable replay. analyze.py verifies the exact declared case set, GPU,
compiler/runtime/source hashes and matching image/Schedule/ABI/contract identity.
One native owner is reused across shapes; warmed scratch witnesses stay fixed.

Median per-case prepared/control warm ordinary-call wall ratios are
0.0765821 forward and 0.0761356 reverse, approximately 92.4% lower wall time.
Wall includes route-specific binding, host packing, transfers, both launches,
completion and readback. Cold trace/compile wall and separate producer/consumer
CUDA event dispatch samples remain in the packets. Event windows include
dispatch gaps. These measurements establish host orchestration improvement;
they do not establish a GPU algorithm gain or a quantized strategy promotion.

## Reproduction and remaining work

In Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh and set
PYTHONPATH to the repository python directory and root.
Run benchmarks/nvidia/benchmark_bounded_lhs_jit.py with both
TESSERA_NVIDIA_PREPARED_LHS and TESSERA_NVIDIA_PREPARED_LHS_REPLAY set to 0
for control or 1 for prepared, and TESSERA_BOUNDED_LHS_PACKET naming the output.
Run this directory's analyze.py from the repository root.

This closes ordinary bounded selection for the named straight-line host-array
envelope. General producer composition and bufferization, shape-dependent
control flow, AD, dynamic row-RHS, resident/asynchronous ownership, wider
formats and sibling physical consumers remain open. The full five-slice
objective remains active. No PR publication is claimed by this evidence.
