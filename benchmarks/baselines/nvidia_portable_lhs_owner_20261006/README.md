# SM120 portable producer-to-matmul replay ownership

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-PORTABLE-LHS-2026-10-06.

## Architecture and admission

The static FP16/BF16 RMSNorm, LayerNorm and last-axis softmax -> matmul
program reuses its compiler-owned Graph/Schedule/Tile/Target/LLVM component
images. Portable manifests contain no CUDA handles. Full manifest admission
constructs an immutable CPU witness; repeated replay compares a type-strict
snapshot before reuse. Booleans, integers, float types and signed zero cannot
alias distinct certificates. Parent Graph, target, argument order, input shape
and dtype are checked before native context discovery.

A bounded process-local LRU retains at most 24 admitted programs and 24
context-specific native owners. The owner key includes CUDA's unique context
identity, so one artifact cannot borrow another live context's modules.
C++ owns both modules, ordered stream, private intermediate and synchronous
pinned/device staging. The Python cache lock spans execution and retirement;
eviction and explicit clear cannot release an in-flight frame. Fork refusal
occurs before the Python/native locks or CUDA access. Native handles remain
process-local and are never serialized.

## Exact-device validation

Super-Bear reports NVIDIA GeForce RTX 5070, compute capability 12.0.
498 focused checks pass, including 142 device cases and 356 host cases.
The 35 new portable owner cases cover all three producers, FP16/BF16,
C/F RHS and plain/fused epilogues, changed inputs, independent output lifetime,
malformed seals, guards before context access, bounded eviction/clear/rebind,
two live contexts, OrderedDict manifests, concurrent launches and fork while
a different thread holds the Python cache lock. Existing fresh-process
compiler-disabled replay tests remain included. See regression-tests.txt.
Native build and focused Ruff checks also pass.

## Matched portable replay benchmark

Eight separate processes cover control/prepared and reverse process order
for C/F RHS. Public JIT preparation remains enabled in both arms; only
TESSERA_NVIDIA_PREPARED_LHS_REPLAY changes. Every process checks 24 cases:
FP16/BF16, three producers, plain/fused, M,K,N = 17,35,19 and 128,1024,64.
Five repeated portable wall samples follow the cold replay. Correctness is
required before timing and repeated outputs match the public call bitwise.
The analyzer checks matching image/ABI/semantic/compiler/runtime/source
identities, current source hashes and actual portable owner receipt bindings.
Separate producer and consumer CUDA-event dispatch windows remain recorded.

Median per-case prepared/control warm portable-wall ratios:

| RHS | Forward | Reverse |
|---|---:|---:|
| row-major | 0.168013 | 0.168429 |
| column-major | 0.163194 | 0.165866 |

This is approximately 83–84% lower warm portable replay overhead.
All eight processes and 192 correctness-gated rows pass; analysis.json retains
each matched row, wall ratio and separate producer/consumer event medians.
These wall measurements include binding, transfers, dispatch and completion;
they do not establish a kernel speedup, compile-latency gain or format promotion.

## Reproduction and remaining scope

From Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run benchmarks/nvidia/benchmark_lhs_jit_dispatch.py with
TESSERA_NVIDIA_PREPARED_LHS=1, TESSERA_NVIDIA_PREPARED_LHS_REPLAY=0 or 1,
TESSERA_LHS_RHS_ORDER=C or F and TESSERA_LHS_PACKET set to the JSON destination.
Run this directory's analyze.py from the repository root.
Use clear_portable_lhs_owners() for explicit retirement.

This supersedes the earlier prepared-LHS packet's statement that common
portable replay retains only its descriptor control route for this named
static envelope. General producer composition, dynamic layouts, AD,
asynchronous/resident ownership, wider formats and sibling physical consumers
remain open. The full five-slice objective remains active.
