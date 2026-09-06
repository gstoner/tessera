# Native storage: generated AD, ownership recurrence and Apple materialization

Owner **W2.4a / CAKE / SO-2**; synchronization **IR-NATIVE-FOUNDATION-1**.
This is the bounded follow-up to [PR #730's integration](NATIVE_STORAGE_INTEGRATION.md).
Python remains the frontend, artifact binder and numerical oracle; the C++
compiler and native backends execute the generated programs.

## Compiler-generated AD child

`JitFn.compile_native_storage_jvp` specializes the actual tracer-owned forward
program. The existing C++ `tessera-autodiff-forward` pass's opt-in
`emit-storage-child=true` consumes its generated primal/tangent SSA program and
materializes one fused physical GPU child. Python supplies no derivative formula.
The parent retains source/paired-program fingerprints and the serialized child's
image, ABI, sizing companion and manifest digest. Reloading executes through
`NativeStorageJVP` with caller-owned resident input, tangent and output buffers.

Admission: one equal-shape rank-one f32 product of add/mul operations, static
width 1–1024, one workgroup, no regions or unsupported operation attributes.
The compiler's ordered `wrt` inputs feed the child; output order is primal then
tangent. This is an explicit compile API with host example inputs, followed by a
physical ABI distinct from the original Python function's call signature.
It does not replace general family planning, synthesize reverse AD, infer
arbitrary Schedule manifests or allocate output tensors at execution time.

The [gfx1151](baselines/native_storage_ad_rocm.json) and
[RTX 5070](baselines/native_storage_ad_nvidia.json) packets each prove four widths
for `x*x+x`, with varying signed primal/tangent values and exact output oracles.
Both execute serialized/reloaded native children generated from real tracing.
Run `record_native_storage_ad.py --backend {rocm,nvidia} --compiler PATH --output FILE`.
No performance claim is attached to this correctness-oriented storage child.

## Generation-sensitive fixed-slot ownership

The shared lifetime analysis proves a changing-token recurrence for a fixed
allocation: initial copy/group; exact carried-token full wait and publication;
reads; collective release; refill/group yielded to the next iteration; and a
final wait/barrier on the loop result. The last result selects the seed on zero
trips and the last refill otherwise. Only then may a later allocation reuse the
slot. Different fixed slots are analyzed independently; this is not a proof for
permuted memref aliases, dynamically selected ring slots or arbitrary nested CFGs.

Regressions ensure stale seeds, partial waits, missing release/final barriers and
an identity backedge cannot justify coalescing; forged reuse groups are rejected
by the arena consumer. The [RTX packet](baselines/native_storage_rotating_nvidia.json)
proves five cases including zero trips. Each generation reads different source
values; later synchronous scratch reuses the released dynamic slot, with a native
size of one slot rather than two. Run `record_rotating_storage.py --compiler PATH
--output FILE` on the CUDA owner. NVGPU token completion is not a ROCm protocol.

## Apple-owned native materialization

`TileBufferArena`'s `emit-apple-msl=true` option translates a bounded typed GPU
arena to MSL: device pointers/index scalars, one dynamic `threadgroup(0)` byte
arena, typed f32 views, arithmetic, loads/stores, uniform counted loops and
collective barriers. Unknown operations, NVGPU copies, incompatible descriptors
and unsupported launch ABIs fail closed. Loop yields use parallel assignments.
The **native sizing companion** rounds Metal's allocation length to 16 bytes
and rejects oversized arithmetic; Python does not recreate the sizing formula.

`materialize_apple_arena` exports the paired MSL and LLVM companion. The owning
Mac recorder compiles the companion to arm64 and binds its result with Metal's
`setThreadgroupMemoryLength`. The [M1 Max packet](baselines/native_storage_arena_apple.json)
proves six widths, including 1 and 17 (16/80 bytes), cross-lane reads, 17 loop
iterations, 32 workgroups and oversized-size rejection. This ran unsandboxed on
Mac with the user's explicit exception to the WSL-only test rule.

Reproduce: on the compiler host, run `export_apple_native_arena.py --compiler PATH
--output DIR`; transfer that directory to the Mac and run
`record_apple_native_arena.py --artifacts DIR --output FILE` with `PYTHONPATH=python`.
The probe has a bounded 30-second completion wait. Production Metal package/JIT
binding, broader operations/control flow and selector integration remain open.
No existing Apple runtime source or sealed E2E packet changed.

## Stronger ROCm wait evidence

`repeat_rocm_prefetch.py` runs five fresh processes, each with eleven randomized
samples of fifty launches for three variants. The additional nonblocking
`s_waitcnt vmcnt(63)` control has the same instruction footprint as the immediate
VMEM drain. RDNA3.5's checked-in ISA archive defines 63 as no wait on VMCNT.

The [packet](baselines/rocm_overlap_independent.json) verifies compiler, source,
image and device identity, distinct processes, raw samples and derived medians.
[Drain](baselines/rocm_overlap_immediate.disasm) versus
[nonblocking](baselines/rocm_overlap_nonblocking.disasm) disassembly has **348
instructions each**, with **only five wait thresholds changed**. Both use 28
SGPRs, 16 VGPRs, wave32, zero private/scratch segment and the same dynamic LDS
bytes. Raw native resource metadata is included in each process packet.

| Blocks × width × generations | Median drain/nonblocking ratio | Runs favoring nonblocking |
|---|---:|---:|
| 32 × 64 × 7 | 1.0147 | 3/5 |
| 256 × 256 × 33 | 1.0171 | 5/5 |
| 256 × 256 × 65 | 1.0221 | 5/5 |

The larger cases support a repeatable benefit from deferring VMEM completion
with an otherwise identical instruction schedule. The small case is unstable.
This is stronger wait-placement evidence, not universal overlap or promotion
proof. The installed ROCm profiler produced HIP API traces but no kernel trace;
[counter discovery](baselines/rocm_overlap_counter_support.txt) reports no
supported PMC metrics on this WSL GPU stack. Hardware stall/counter attribution
still needs a supported profiler environment. No timing threshold or default
selector was changed.

## Backend disposition

- NVIDIA: generated AD and fixed-slot rotating ownership have owning-device
  correctness proof. Arbitrary rotating aliases and sanitizer proof remain open.
- ROCm: generated AD has device proof; wait placement has independent matched
  evidence. GPU counter attribution and a generic AMD completion-token protocol
  remain open.
- Apple: native arena-to-MSL and native sizing have M1 Max proof. Production
  packaging/JIT binding and broader materialization remain open; AD child
  execution is not inherited from CUDA/HIP.
- x86: native LLVM host sizing executes on WSL x86 and on arm64 for Apple.
  GPU kernels and timing establish no x86 compute or CPU-native AD parity.

## Validation

Host WSL: 339 focused tests passed; 276 shared lifetime/forward-AD native fixtures
passed (7 unsupported); the rebuilt production NVIDIA compiler passed all 61
fixtures. Ruff and the zero-error mypy ratchet pass; all 30 generated audit
documents are synchronized. The code knowledge graph was refreshed. Device
packets above share the final core-compiler fingerprint; AD packets additionally
fingerprint their runtime source files. Metal validation used the explicitly
authorized unsandboxed Mac exception. No full-unit-suite or sanitizer result is
claimed by this loop.

## Expanded AD, slot aliases and resident Apple binding — 2026-09-06

Owner W2.4a / CAKE / SO-2; sync **IR-NATIVE-FOUNDATION-1**.

The physical AD child now admits subtraction, stopped primals and exact f32
splat constants emitted by the forward pass. A review also found that forward
AD declared tangent parameters in requested `wrt` order but mapped them in source
argument order. It now maps by the declared order. Native tests pin this with
`wrt_indices = [1, 0]`; numerical device oracles use distinct tangents and check
`dx-dy`. Both [CUDA](baselines/native_storage_ad_expanded_nvidia.json) and
[ROCm](baselines/native_storage_ad_expanded_rocm.json) execute 16 exact cases:
four widths each of polynomial, two-input subtraction, stopped polynomial, and
reversed `wrt`. These packets extend, rather than replace, the earlier evidence.
Activation/transcendental, reduction, attention and reverse-AD families still
need their own scalar/Schedule lowering and device proof.

Dynamic memref aliases now have a bounded **two-slot permutation** proof. Roots
must be distinct allocations with dominating markers, the loop carries must swap
bijectively, and returned memref aliases cannot escape. Every asynchronous copy
must fully complete and publish before another slot access, followed by a
collective release before the backedge. Zero trips retain the initial slots.
The arena consumer repeats this proof; forged reuse groups, duplicated carries,
partial waits, missing publication and missing release cannot authorize reuse.
The [CUDA packet](baselines/native_storage_slot_alias_nvidia.json) proves seven
cases with zero/odd/even trips and varying generation data. Two physical slots
remain distinct, while later scratch reuses a released slot (native size `8*n`).
This deliberately does not prove a token that remains pending across a slot swap,
a dynamically indexed N-slot ring, nested permutations or arbitrary CFGs.

`AppleArenaPackage` binds the compiler-owned MSL, native arm64 sizing companion,
and separately compiled Metal bridge with one digest. Deserialization requires
an independently pinned digest. Its explicit resident-buffer API uses Tessera's
existing Metal device and queue, checks f32 ownership/extent, rejects aliasing,
checks native sizing and hardware geometry, and binds dynamic threadgroup slot 0.
A 30-second wait reports timeout and disables the binding; submitted Metal buffers
remain retained through completion. Calls serialize within a binding. Other
queues/producers require caller synchronization; output extents and scheduling
remain explicit raw-ABI responsibilities, as on the raw CUDA/HIP package API.
Non-owning views are rejected because their owner's lifetime is not tracked.

The [M1 Max packet](baselines/native_storage_binding_apple.json) executes a
serialized/reloaded package on actual resident `DeviceTensor` buffers across six
width/round combinations, including zero trips, plus invalid sizing inputs. The
packet fingerprints the compiler, runtime, binding and bridge. Run
`record_apple_arena_binding.py --artifact DIR/artifact.json --output FILE` with
`TESSERA_APPLE_GPU_RUNTIME_LIB` selecting the owning runtime. This replaces the
standalone probe at the raw package binding layer. Automatic tensor descriptors,
JIT/arbiter registration, Apple AD children and cross-binding stream ownership
remain open. The existing sealed Apple runtime source is unchanged.

[Structured ROCm discovery](baselines/rocm_counter_capability.json) records
rocprofiler SDK 1.3.5 / ROCm 10.0.0, the actual WSL host, gfx1151 agent details,
stdout/stderr and alternatives. It reports no PMC or SPM counters and no
PC-sampling agents. The installed Windows AMD directory exposed Qt debug profiler
plugins, not a Radeon GPU Profiler installation. This is a host capability
blocker, not zero stalls and not hardware-counter attribution. The new recorder
keeps discovery `unverified` even if a counter name appears; successful counter
capture matched to the dispatched images is still required. Prior HIP-event and
instruction-matched evidence remains valid in its narrower scope. No default
selector or performance threshold changed.

## Nonlinear AD, pending swaps and typed Apple JIT — 2026-09-06

Owner W2.4a / CAKE / SO-2; sync **IR-NATIVE-FOUNDATION-1**.

**Nonlinear forward products.** The native child producer now lowers sigmoid,
tanh and their compositions from the actual paired SSA program. Derivatives
remain owned by `TangentInterface`. Scalar sigmoid evaluates the exponential on
a nonpositive argument and selects the appropriate ratio, avoiding overflow in
both tails. Tanh uses that stable ratio away from zero and an odd polynomial for
`abs(x) < 1/8`, avoiding subtractive cancellation; `copysign` retains negative
zero. Backend GPU conversion owns target-specific math before generic Math-to-LLVM
conversion. No host libm fallback or Python derivative is introduced. RDNA3.5's
archive records the exp instruction's 1-ULP and denormal-flush boundary.

The [CUDA](baselines/native_storage_nonlinear_nvidia.json) and
[ROCm](baselines/native_storage_nonlinear_rocm.json) packets each cover nine
cases: three widths for sigmoid, tanh and sigmoid(tanh(x)). They include finite
saturation tails, infinities, NaNs and signed zero. Primal/tangent comparisons
against float64 oracles use `atol=rtol=2e-6`; near-zero tanh additionally requires
`atol=0, rtol=2e-6`. These are bounded numerical contracts, not correctly-rounded
transcendental or subnormal-preservation claims. Reductions, attention,
reverse-mode children, broader shapes and Schedule-family selection remain open.

**Pending tokens across a swap.** The lifetime proof couples a two-slot
permutation with its copy token. The initial token must commit the initial read
slot's seed; the next token must commit the refill of the slot that becomes the
next read slot. Each iteration fully waits/publishes before reading, collectively
releases the consumed slot before the backedge, and finally drains the exact
loop-result token. Zero trips drain the seed. Returned memref aliases cannot
escape. The producer can prefetch into the other, already released slot **before
consuming the current slot**, preserving an opportunity for overlap rather than
requiring a serialized refill. Both physical slots remain distinct throughout.
The arena consumer repeats the proof before accepting a reuse assignment.

The [CUDA packet](baselines/native_storage_pending_swap_nvidia.json) proves seven
zero/odd/even-trip cases with generation-varying input and two-slot native sizing;
later scratch reuses one slot only after the final drain. Negative native tests
reject stale tokens, unswapped aliases, a wrong refill destination, partial waits,
and missing release. This is correctness and ownership evidence, not a measured
speedup. General N-slot indexing, nested/conditional recurrences and a generic
AMD completion-token producer remain follow-ups.

**Typed Apple JIT.** Shared tensor signature validation and manifest decoding
are now separated from CUDA/HIP address handling. `AppleTensorCall` consumes the
same compiler-preserved tensor manifest with its own Metal validation. Shapes,
f32 dtype, scalar bounds, grid/block geometry and writable outputs come from that
manifest. `JitFn.bind_apple_native_arena` dispatches directly to resident Metal
buffers and reports the explicit package identity; calls retain the existing JIT
constraint checks. No Graph IR reconstruction, eager fallback, automatic Python
semantic equivalence or arbiter promotion is implied. Requested AD is rejected
because the Apple package does not yet carry a paired AD contract.

The [M1 Max packet](baselines/native_storage_typed_jit_apple.json) covers six
shape/round combinations, positional and keyword calls, close/lazy reopening,
invalid scalar bounds, wrong tensor shapes, non-owning views and forged resident
extents. Reproduce by adding `--typed` to the export and Apple binding recorders.
The kernel, native sizer, runtime and JIT fingerprints are retained. Caller-owned
output allocation and external producer synchronization remain explicit; Apple
AD, automatic family/arbiter selection and cross-queue event ownership remain open.

## N-slot/nested ownership and paired AD queues — 2026-09-06

Owner W2.4a / CAKE / SO-2, synchronization IR-NATIVE-FOUNDATION-1.

Released arbitrary bijective slot permutations now coalesce after complete
per-iteration publication/release. Uniform enclosing loops/branches are admitted;
nonuniform nesting is rejected. Pending recurrence remains two-slot with exact
seed, destination, backedge and final-drain checks. CUDA packets
`native_storage_ring{3,4,8}_nvidia.json` and
`native_storage_nested_pending_nvidia.json` each contain seven exact cases,
including zero/odd/even trips and post-loop reuse. These prove correctness, not speedup.

The compiler emits cooperative power-of-two sum/mean JVP reductions with barriers
outside lane branches and lane-zero scalar stores. Reverse children fuse the
real compiler forward/backward SSA with an explicit output cotangent. This slice
supports residual-free single-input equal-shape elementwise reverse pairs.
The public pair package pins paired-program lineage and derives its typed ABI
from the compiler manifest. Scalar tensor rank and canonical reduce kind were
fixed after real traced reduction IR failed native parsing.

`native_storage_pair_{nvidia,rocm,apple}.json` each records 15 cases: sum/mean JVP,
tanh JVP, square/tanh VJP, widths 32/64/256, serialized package roundtrip and
independent numeric oracles (2e-6 absolute/relative). Owning devices are RTX5070,
gfx1151 and M1 Max respectively. Metal additionally exercises explicit JIT pair
binding and 12 producer blit → fresh shared event → consumer queue wait → AD
handoffs. Native fences retain their producer device and event; queue/fence
close is serialized against submission. Consumer completion is bounded by the
existing 30-second package wait. No runtime seal or automatic selector changed.
Apple Clang cannot parse LLVM23's optimizer-only nocreateundeforpoison attribute;
packaging conservatively omits that promise while retaining the pinned original
LLVM companion and all sizing instructions.

Remaining: N-slot pending generation maps; mixed-shape reduction VJP; attention
Tangent/AdjointInterface and paired mask/saved-LSE contracts; general reverse
tapes; automatic family/arbiter generation; CUDA/HIP event parity and resource
ownership across queues; measured overlap. Attention currently has neither AD
interface, so hand-written attention backward packages do not establish compiler
AD support. Avoid a quadratic-score lane-scalarized implementation: dense attention
needs row/tile cooperative native lowering. x86 has host-companion evidence only.
ROCm counters remain unavailable on the installed gfx1151 WSL profiler.

## Pending rings, reduction VJP, attention interfaces and queue intervals — 2026-09-06

Owner W2.4a / CAKE / SO-2; synchronization **IR-NATIVE-FOUNDATION-1**.

- Pending ownership now accepts arbitrary bijective N-slot permutations with
  **one pending copy generation**, provided the yielded read slot is exactly the
  refill destination. Seeds, per-iteration wait/publication, release and final
  drain remain mandatory. Nested 3/4/8-slot CUDA packets each pass seven exact
  zero/odd/even cases (`native_storage_pending_ring{3,4,8}_nvidia.json`). Multiple
  simultaneously outstanding slot generations remain outside this proof.
- Native pairs carry per-input and per-output widths. Sum/mean VJP loads a scalar
  output cotangent and broadcasts the compiler-produced gradient to N lanes;
  primal and gradient output extents differ. The 21-case
  `native_storage_reduction_vjp_{nvidia,rocm,apple}.json` packets cover forward
  reductions/nonlinear pairs, elementwise reverse and reduction reverse at
  widths 32/64/256. Metal needed constant-index LLVM pointer materialization.
- Straight-line compiler SAVE residuals are connected from forward result SSA to
  the matching backward arguments inside the fused native child. Residual source
  metadata distinguishes saved results from additional public primal outputs;
  the latter are rejected. This is native fused residual forwarding, not an
  externally persistent tape or general control-flow tape implementation.
- FlashAttnOp now implements bounded dense-f32 reverse AD through registered
  checkpoint_forward/checkpoint_backward operations, with one explicit natural-log
  row-LSE SSA value and matching scale/end-aligned causal policy. The forward
  tangent interface supports V-only activity (attention is linear in V).
  Q/K activity, dropout, bias, cache, numeric-policy and unknown semantic attrs
  fail closed. Compiler IR tests prove production interface dispatch and valid
  registered checkpoint operations. **No new end-to-end attention device proof**
  is claimed: native scheduling of compound AD products and persisted forward-LSE
  reuse remain follow-ups. The backward interface currently recomputes LSE.
- Metal command timestamps are exposed only after successful bounded completion.
  Five independent processes recorded 270 matched serial/concurrent measurements
  of the same native shader on two queues, with independent outputs and exact
  oracles (`native_storage_queue_overlap_apple.json`). For 2048 rounds, all 45
  concurrent samples have overlapping GPU command intervals; median serial versus
  concurrent GPU-span ratios range 1.945–2.071 across runs. Spans include gaps
  between submissions, so these are not kernel-throughput speedups. Command
  interval overlap does not prove simultaneous instruction issue. No selector
  or automatic concurrency policy changed; ROCm counter availability is unchanged.

Validation workflow: broad tests must use an immutable compiler executable.
Re-linking tessera-opt during tests caused transient permission failures in an
invalidated run; subsequent validation uses tessera-opt-loop5-validated with no
concurrent builds. Do not count the invalidated run as evidence.
