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


## Multiple pending cohorts and persisted attention LSE — 2026-09-06

W2.4a / CAKE / SO-2; synchronization key **IR-NATIVE-FOUNDATION-1**.

The shared memref proof now tracks a set of matched seed/refill/token/slot
relationships. Every member must complete before collective publication;
all reads must release before the backedge, and every final token must drain
before arena reuse. This covers multiple independent pending cohorts under
uniform nesting, not arbitrary CFG joins or a FIFO that waits only its head.
The arena repeats the proof before consuming a reuse assignment.

`record_rotating_storage.py --outstanding {2,3,4} --nested` passed seven cases
per cohort count on RTX5070: zero, one, odd/even and longer trips. Each cohort
uses a distinct input slab; the oracle sums all consumed generations and checks
scratch reuse after the final drain. [Packets](baselines/native_storage_loop6/)
bind source, compiler and package digests. These are correctness measurements,
not evidence that additional outstanding groups improve throughput.

Paired dense-f32 attention AD now returns natural-log row LSE from forward and
accepts it as a named backward residual. The standalone attention backward no
longer invokes checkpoint forward to rediscover LSE. Causal alignment and the
f32 scale remain shared between producer and consumer. This is an MLIR ABI
change with native compiler tests; persisted device allocation ownership and
AD-to-device package integration are still required. The in-place adjoint
interface retains its recompute behavior because it has no external tape ABI.

### Queue measurements and attribution

`record_native_queue_overlap.py` compiles one cooperative native MLIR ring
workload and compares serial/parallel launches using disjoint inputs/outputs,
a common GPU event anchor, alternating measurement order and exact oracles.
Each backend ran five repetitions at 8/32/128 blocks, for 30 measured pairs.
These are driver-level protocol measurements, not production tensor/JIT binding
validation. Every packet identifies its actual compiler; CUDA used LLVM23
`mlir-opt`, while HIP used the core compiler with its registered ROCm pipeline.

| Backend | Blocks | Median serial span ms | Median parallel span ms | Median interval intersection ms |
|---|---:|---:|---:|---:|
| RTX5070 | 8 | 0.132384 | 0.098080 | 0.032256 |
| RTX5070 | 32 | 0.118208 | 0.113728 | 0.025888 |
| RTX5070 | 128 | 0.243968 | 0.170176 | 0.085120 |
| gfx1151 | 8 | 0.218220 | 0.133234 | 0.072196 |
| gfx1151 | 32 | 0.788756 | 0.439157 | 0.381118 |
| gfx1151 | 128 | 0.941002 | 0.576973 | 0.517147 |

This single-process sample is exploratory; event intervals include scheduling
and event overhead. It does not establish a repeatable application speedup.
A separate Nsight Systems run captured 30 CUDA kernel intervals (including
warmups), with 17 cross-stream intersections. Among the three measured parallel
pairs, kernel intersection was 0 / 16032 / 99008 ns at 8 / 32 / 128 blocks;
the smallest profiled pair did not overlap. The extracted intervals and raw
SQLite digest are in `baselines/native_storage_loop6/nsight_systems.json`.
Nsight Compute 2026.2.1 separately profiled one 8-block kernel: duration
102880 ns, SM throughput 0.78% and DRAM throughput 1.55% of sustained peak.
Its replayed, isolated counters are workload characterization, not attribution
of concurrent instruction issue. No selector was promoted.

HIP counter attribution remains unavailable on the current gfx1151 WSL stack;
GPU events do not fill that gap. Apple and x86 do not inherit either GPU result.

### Next architecture boundaries

1. Persistent tapes need owned device snapshots, immutable generation and
   package identity, repeated-backward lifetime rules, and completion-aware
   reclamation. Keeping raw input pointers alive does not protect against input
   mutation. Separate forward/backward packages must replace the current fused
   native child's two-output assumption before this can be called persistent.
2. Nested tapes need per-invocation frames with branch identity, executed trip
   counts and saved state; nested Python recording is not a device tape proof.
3. Attention Q/K JVP needs a cooperative product lowering for
   `dP = P * (dS - rowsum(P*dS))`, with
   `dS = scale * (dQ*K^T + Q*dK^T)` and `dO = dP*V + P*dV`.
   Reuse the same causal/LSE convention and avoid materializing a quadratic
   score tensor. Ship the registered producer and native consumer together.
4. Bind compiler-produced attention forward/backward functions to physical
   packages with the saved-LSE owner/generation checked at consumption. Measure
   each backend independently; existing hand-authored packages do not prove it.
5. Extend queue evidence through production submissions and fresh-process runs;
   obtain ROCm tracing/counters on a supporting host. GPU timestamp overlap is
   insufficient to attribute which execution units overlapped.

Validation on Princess-Luna WSL: 17977 non-slow tests passed, 2309 skipped,
870 deselected; 125 native lit fixtures passed and 3 were unsupported. The
subsequently added two-attention residual-slot case also passed in the final
18-test attention/audit run. Registry-focused tests, Ruff, the zero-error mypy
ratchet and all 30 generated-document checks pass. Tests used an immutable
`build/tools/tessera-opt/tessera-opt-loop6-validated` executable.


## Persistent snapshots and generated attention exports — 2026-09-06

`record_native_device_tape.py` validates 12 cases each on RTX5070 and gfx1151:
square/tanh/sum/mean, widths32/64/256, input mutation after capture, two backward
cotangents with independently retained results, and nested-frame cleanup.
`NativeStoragePair.capture` owns a device copy; results remain frame-owned until
close. Calls are synchronous and recompute the paired product. This does not
claim higher-order AD, arbitrary control-flow tapes, stream overlap or reduced
compute. Explicit close remains required; package close releases its frames.

`record_generated_attention_ad.py` validates six RTX5070 cases starting from a
fresh reverse-marked MLIR function. `checkpoint-product=forward|backward` exports
one compiler-generated product with canonical arguments and full paired lineage;
`lower_generated_checkpoint` then uses native Schedule/Tile and the existing
NVIDIA descriptor packages. No historical Graph module is reconstructed.
Forward output, natural-log LSE and Q/K/V gradients match independent float64
oracles within3e-5 for Sq/Sk3/5,5/3,4/4, each causal/noncausal. The named
`end_aligned_v1` policy is `key <= q + max(Sk-Sq,0)`; negative offsets are not
part of this contract. LSE crosses the host-buffer runtime bridge; resident
attention tape ownership is still open. Q/K JVP remains unimplemented.

Packets live in [native_storage_loop7](baselines/native_storage_loop7/).
The incoming native gfx1201 host has a read-only commissioning probe and
[profiling plan](../docs/audit/backend/rocm/NATIVE_RDNA4_COMMISSIONING.md).
No counter validation is claimed before that machine is available.

The streaming Q/K/V JVP spike has 17 passing reference cases, including block
sizes1/3/8/32, directional finite differences, max rescaling and empty masks.
It establishes an algorithm for the next cooperative lowering; it supplies no
native tangent registration, package, timing or device evidence. MSW-9 now
separates dense, unique storage and unique trainable slot counts, including
frozen/shared snapshots. Its automatic fusion and executable-identity bindings
remain open.

Loop7 validation: 17990 non-slow tests passed, 2309 skipped, 870 deselected;
125 native fixtures passed, 3 unsupported. The final focused run passed87 tests,
including the subsequently added streaming/ANN cases and tape lifecycle checks.
The lifecycle test was then rerun with bounded daemon-thread joins (2 passed).
Ruff, zero-error mypy and all30 generated-doc checks pass. The 12 cases on each
GPU were rerun after serializing pair capture/calls against close; packets record
the Python implementation hashes. Counts overlap and must not be added.

## Loop 8: native frozen ANN composition and resident LSE

Owners: W2.4a / CAKE / SO-2 and MSW-9; sync **IR-NATIVE-FOUNDATION-1**.

`AttentionCheckpointPair.capture` now supplies a synchronous CUDA frame owning
private Q/K/V snapshots and the generated forward's LSE allocation. Its backward
launch directly consumes those pointers, bypassing the host tensor bridge.
Multiple backward results have separate allocations. Context checks precede
use/free; allocation bounds, complete descriptor ABI/policy and shape guards
are validated before execution. Closing invalidates exposed read-only views.
An injected failed backward releases only that attempt's allocations.

The [resident attention packet](baselines/native_storage_loop8/resident_attention_nvidia.json)
records six RTX 5070 cases: Sq/Sk = 3/5, 5/3 and 4/4, each causal and noncausal,
with grouped heads, mutated caller inputs after capture and repeated backward
with doubled cotangents. Independent float64 output/gradient oracles use 3e-5
absolute and relative tolerance. This is correctness/ownership evidence only;
there is no timing comparison or overlap claim. The native kernel/compiler is
unchanged from loop7; the new frame binds its images directly.

MSW-9's native consumer now lives in `tessera-canonicalize`, enabled explicitly
with `ann-reassociate=true`. It discovers a two-affine constant chain and folds
weights/biases with deterministic APFloat arithmetic. Registered `tessera.add`
requires equal-rank tensors, so the native rewrite consumes matrix biases and
preserves per-row values. It refuses runtime parameters, intervening activations,
shared intermediates, transpose/policy overrides, nonfinite folded results and
excessive compile-time work. Default pipelines retain existing association.
The rewrite emits existing native MLIR operations; it does not rebuild Graph IR.

General nested control-flow tapes and native Q/K JVP remain open. The integrated
plan now specifies the split product/residual ABI and same-generation O/LSE
score-tangent consumer needed to close them. Automatic JIT/arbiter ANN promotion
and native original/fused performance evidence also remain open. CUDA allocation
ownership does not establish HIP, Metal or x86 execution support.

Validation: 261 focused ownership/ANN/lifecycle/audit/registry tests on WSL;
67 native AD and canonicalization fixtures; six resident CUDA cases. Ruff,
zero-baseline mypy and all 30 generated-document checks passed. The shared
compiler was rebuilt on Princess-Luna and tested through an immutable copy.
No full-suite or new ROCm/Metal/x86 device execution result is claimed for loop8.


## Loop 9: typed nested products and native resident Q/K JVP

Owners: W2.4a / CAKE / SO-2 and MSW-9; sync **IR-NATIVE-FOUNDATION-1**.

The split native AD exporter preserves full tensor/index/bool residual types,
common paired lineage and native control-flow regions. Saved if/while products,
nested SAVE loops, zero outer loops and zero inner loops have compiler tests.
Nested replay forwards its own residual results into the registered pullback.
This is typed artifact proof; general persistent nested device tape allocation
and execution remain open. Replay of a nested pure region from saved outer
state is not a claim that every inner tape persists across device invocations.

The [resident JVP packet](baselines/native_storage_loop9/resident_attention_jvp_nvidia.json)
records eight RTX 5070 cases, causal/noncausal Sq/Sk = 3/5, 5/3, 4/4 and 3/129.
Each tests four tangent modes: Q, K, Q+K, Q+K+V (32 total). Independent float64
analytic derivatives agree with centered finite differences; native results use
3e-5 absolute/relative tolerance. The native GPU MLIR consumer reconstructs P
from the owned LSE, reduces with bounded shared storage and uses the same grouped
heads and end-aligned mask. Its explicit resident API requires `prepare_jvp`
before `jvp`. It does not change the automatic V-only TangentInterface. The
129-key backward oracle also caught and fixed a launch grid using the maximum
gradient range instead of their concatenated total.

The packet was generated using the loop9 compiler built on Princess-Luna and
copied immutably to Super-Bear. It contains source hashes and package identities.
There is no latency or overlap comparison. This bounded kernel recomputes scores
per output column; no performance promotion is justified by these results.

Shared arbiter cache admission now requires explicit admissibility both for
exact hits and retained incumbents. Three regressions reject ineligible or
unseparated cached records. ANN candidate registration and native original/fused
comparison remain open; no ANN route was promoted. HIP, Metal and x86 execution
proof must be established by their own consumers and owning hosts.

Validation: 307 focused native-product, ownership, arbiter and registry tests;
11 audit tests; 125 native AD/IR fixtures passed (three unsupported fixtures).
Ruff and the zero-error mypy ratchet passed on Princess-Luna WSL. Eight resident
CUDA cases and 32 tangent directions passed on Super-Bear. No full-suite or new
ROCm, Metal or x86 device result is claimed for this loop.


## Loop 10: automatic Q/K product binding and persistent failure isolation

The [automatic attention packet](baselines/native_storage_loop10/automatic_attention_jvp_nvidia.json)
uses native `TangentInterface` generation and `export-attention-jvp`, then binds
the resulting product to the resident forward generation. Eight RTX 5070 cases
exercise 32 directional probes including 129 keys, both causal policies and
grouped heads. Analytic and centered finite-difference oracles match within the
same tolerances as loop9. This is correctness evidence only. The core compiler
was rebuilt on Princess-Luna and copied immutably to Super-Bear; Python source
hashes in the packet match the local implementation.

The new internal checkpoint JVP verifier enforces the same forward SSA producer
for O/LSE and identical Q/K/V and policy. The physical export rejects composed
functions or indirect argument/return mappings; inactive tangent slots are
zeroed in native IR. General JIT composition and HIP/Metal/x86 lowering remain
open. V-only retains its linear checkpoint-forward implementation.

Persistent snapshot allocation now rejects invalid/overflowing extents before
calling the driver and rolls back partial backward-result allocation failure.
A fault-injection regression verifies that existing frame allocations survive.
This is a shared allocation contract test, not new HIP/CUDA tape execution
proof. General persistent nested tensor tapes still need split bufferized native
products; the current snapshot consumer recomputes the backward residuals.

Validation: 298 distinct focused AD/ownership/operator/dtype/registry tests,
135 additional dialect/native binding tests, 17 audit/frontend tests, and 99
native AD/control-flow fixtures passed on Princess-Luna WSL. Ruff and the
zero-error mypy ratchet passed. No full-suite run is claimed for loop10.


## Loop 11: split persistent tensor products and JIT-owned attention

The native producer exports two independently callable products. Upstream
one-shot bufferization turns full tensor results/residuals into output buffers;
input arguments are explicitly readonly to prevent in-place reuse of captured
snapshots. `tessera-native-tape-to-gpu` consumes the bufferized body and preserves
bounded for/if control. Each temporary receives an entry-owned byte allocation
with separate slices for its entire enclosing iteration path. Zero-trip bodies
still count reserved storage against the 4096-byte logical budget.

[CUDA](baselines/native_storage_loop11/persistent_tape_nvidia.json) and
[ROCm](baselines/native_storage_loop11/persistent_tape_rocm.json) each pass widths
4, 8 and 16 for a two-level SAVE loop (two outer and three inner iterations).
Forward residuals have full `tensor<1xNxf32>` shape. Both recorders verify private
input snapshots after caller mutation, repeated backward with different seeds,
unchanged retained residuals, and invalid views after close. Deliberately zeroing
the retained outer residual changes the derivative as predicted, establishing
that backward consumes that allocation instead of silently rerunning the outer
forward. The inner state is still replayed by the compiler's backward product.

Physical lowering differs: AMDGPU needs explicit private address space 5 to
select its FrameIndex; NVVM uses generic allocas and owns their local addressing.
A forced common private representation failed the CUDA backward execution check.
The final backend-specific implementation passes on RTX 5070 and gfx1151 using
the same immutable core compiler. These serial, one-thread entries provide
correctness evidence, not latency, overlap or production performance evidence.

The [JIT attention packet](baselines/native_storage_loop11/jit_attention_nvidia.json)
contains ten RTX 5070 cases: Q, K, Q/K, reversed K/Q and Q/K/V requests, each with
5 and 129 keys, grouped heads and aligned causal masks. The recorder starts with
an ordinary decorated function and calls `compile_native_attention_jvp`; callers
supply no native IR. A captured resident forward owns O/LSE, and `jvp` accepts
only active directions in request order. Primal and centered float64 finite-
difference checks use 3e-5 absolute/relative tolerance. Native reverse/forward
products use the same selected compiler. Dense frontend attention now supplies
its required head width and a precise pure effect only for the known dropout-
free three-tensor form. Cache and unrecognized variants remain conservative.

Both APIs remain explicit native compilation entries. General JIT compositions,
dynamic or mixed-type tape slots (including saved predicates), persistent while
tapes, asynchronous backward retirement and parallel tape scheduling remain
open. Apple requires MSL/buffer ownership integration; x86 currently supplies
the host sizing companion, not execution proof. Source hashes and immutable
compiler digests are included in the packets; no sibling performance claim or
ANN promotion is made.

Validation: 333 focused product, ownership, registry, dtype, frontend and audit
tests plus 99 native AD/control-flow fixtures passed on Princess-Luna WSL.
Ruff and the zero-error mypy ratchet passed. Three tape cases passed on each GPU
and ten JIT attention cases on RTX 5070. All 30 generated-document checks passed;
no full-suite run or new Apple/x86 execution evidence is claimed.
