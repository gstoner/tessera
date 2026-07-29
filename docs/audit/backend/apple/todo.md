---
audit_role: plan
plan_state: landing
owner: Apple backend
target: apple_gpu
last_updated: 2026-07-29
---

# Apple compiler, exact-device, and performance plan

## APPLE-CALIB-1: contribute op breadth to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **follow-up required, owning
host M1 Max (apple7).** Apple is the *breadth* axis, not the sole site.

**What is being calibrated.** Two static, device-free quality metrics found in
production AMD code and recorded in
[`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8: a step-distance locality histogram over a materialized access order,
and an N-way bank-conflict analyzer computed from a descriptor alone. Both are
computable on any target with no silicon. The question is whether either
*predicts measured latency* — which decides how much weight the arbiter's
hardware-free tier can carry, per
[`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §2.

**Apple's role.** The widest F4-verified op-family envelope in the fleet, so it
answers *does the score generalize across op kinds* — norm chains, attention with
online softmax, pointwise-reduce, gated matmul, coopmat `simdgroup_matrix`, not
just GEMM. NVIDIA and ROCm supply shape depth within GEMM/attention
(`NVIDIA-CALIB-1`, `ROCM-CALIB-1`). Both axes are required: a score fitted on one
architecture reproduces the overfit that assessment §5.2 records for NeuSight,
which led on the A100 inside its training distribution and lost that lead on
every newer part.

**Apple-specific caveat.** The bank-conflict half was derived for LDS with a
known bank count and a 4-phase wave64 access pattern. Metal threadgroup memory is
not LDS and its banking is not documented to the same level (Decision #27 — do
not assert a Metal hardware detail without a real source), so the conflict metric
may be **not applicable** on Apple even where the locality metric is not. Report
that split rather than one blended verdict.

**Missing exact-device evidence.** Rank correlation between each score and
recorded M1 Max latency, per op family, over the families the Apple lane already
measures. A score that does not rank measured Apple kernels correctly is not
trustworthy for unmeasured kernels anywhere.
## APPLE-RASTER-1: reconcile the MLX-inherited swizzle with the shared contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **follow-up required, owning
host M1 Max (apple7).** Apple's follow-up is *reconciliation*, not
implementation: it is the one backend that already had a threadblock swizzle
before the shared contract existed.

**Shared contract changed.** Schedule IR gained `raster_order` (`row_major` |
`column_major` | `grouped_m` | `grouped_n`) and `raster_group` on
`schedule.tile` / `schedule.knob`, mirrored by `TuningConfig` and the tuning
cache, over the arch-neutral `compiler/tile_rasterization.py`. Rationale:
[`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**The pre-existing divergence.** `compiler/apple_gemm_schedules.py` carries
`swizzle_log`, inherited from MLX as a Metal function constant with a hardcoded
heuristic — `swizzle_log = 0 if tm <= 3 else 1`, where `tm = ceil(M/bm)`. That is
a two-valued, shape-derived rule, not a tuned axis, and it is expressed as a
power-of-two tile block rather than the contract's panel height. So Apple is not
missing the lever; it has a **second, incompatible spelling of it**, which is
exactly the kind of drift the shared contract exists to stop.

**Decision required (not yet made).** Either (a) express `swizzle_log` as
`raster_order="grouped_m"` with `raster_group = 1 << swizzle_log` and retire the
hardcode, or (b) record why the MLX form stays — e.g. if the Metal
function-constant specialization is load-bearing for pipeline-state caching in a
way the generic emission is not. Option (a) is preferred *only* if it measures
neutral-or-better; MLX's heuristic is tuned against real Apple silicon and must
not be displaced by a generic default on tidiness grounds (Theory §1 rule 2
applies to inherited hand-tuning as much as to our own).

**Note the shape of the win differs here.** M1 Max is unified-memory with a
48 MB SLC, not a discrete L2 — the cache tier a swizzle protects behaves
differently, so a group size ported from a discrete GPU is not evidence for this
part. Measure locally or not at all.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property and compiles the emitted C against the Python
reference for every block id. The emitted form is C, so it validates the ROCm and
NVIDIA lanes; Apple's MSL synthesizer would need its own emission if option (a)
is chosen — the C snippet is *not* MSL.

**Missing exact-device evidence.** An M1 Max A/B of the MLX heuristic against
`grouped_m` at matched group sizes across the GEMM shape buckets, with Metal
counter evidence, before either spelling is declared canonical.

## APPLE-AOT-4: S1 probe — what an MLIR → AIR emitter would actually cost

**Status: probe complete 2026-07-28, owning host (M1 Max / apple7, Metal
toolchain 32023.883). Capability findings, executed; not a perf claim.**

S0 showed a GPU runs hand-written scalar AIR IR. The open question was whether
that extends to the path that matters — `simdgroup_matrix`, the ceiling-setter
and the reason SPIR-V was rejected. It does.

### simdgroup_matrix is ordinary code, not a special form

Dumping the real synthesized coopmat kernel with `metal -S -emit-llvm`:

| MSL | AIR IR |
|---|---|
| `simdgroup_float8x8` | `<64 x float>` — a plain LLVM vector |
| `make_filled_simdgroup_matrix` | `declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)` |
| `simdgroup_multiply_accumulate` | `declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32(...)` |
| `threadgroup_barrier` | `@air.wg.barrier(i32, i32)` |

These are **external function declarations, not intrinsics needing backend
support**. Hand-written IR calling them — plus an `addrspace(3)` threadgroup
global and a barrier — compiles, packages, loads, and **executes correctly**
(fixture `tests/data/apple/handwritten_air_simdgroup.ll`, all 64 lanes exact).
Everything here is expressible in MLIR's LLVM dialect without extension.

### The builtin surface is 11 declarations

Compiling every synthesizer family and collecting `air.*` references:

| family | IR lines | builtins |
|---|---|---|
| coopmat | 408 | 7 |
| tiled | 277 | 3 |
| attention / attention-online | 244 / 235 | 2 / 2 |
| gated-matmul | 194 | 1 |
| matmul-epilogue | 183 | 2 |
| norm-chain | 155 | 2 |
| pointwise | 78 | 1 |

**Union across all eight families: 11 distinct builtins** — four simdgroup
matrix ops, six math (`convert`, `fast_clamp`, `fast_exp`, `fast_fmax`,
`fast_rsqrt`, `fast_tanh`), one barrier. That is the entire `air.*` dependency
an emitter must know how to name.

### So what S1 costs

Not the builtins (11 declarations) and not the codegen (MLIR's LLVM dialect
already emits functions, calls, address spaces, vector types). The work is the
**metadata emitter**: `!air.kernel` naming the function, one `!air.buffer`
descriptor per argument (location index, access, address space, element
size/align/type/name), builtin descriptors for `thread_position_in_grid` and
friends, plus module flags and `!air.version` / `!air.language_version`. Roughly
five node kinds, all declarative.

On that basis it looks *week-shaped* rather than quarter-shaped — but flag
that as an estimate, not a measurement. The 11 builtins and the IR sizes are
counted; the effort figure is a judgement with no prototype behind it, and it
sits next to measured numbers where it can borrow their credibility. Treat it
as "small enough to try", not as a schedule.

### What still argues against doing it

Feasibility is no longer the constraint; **supported-ness is**. `.ll` input to
`metal` is undocumented, `-x ir` is undocumented, AIR is undocumented by
deliberate Apple choice, and there is no man page. An emitter would rest on an
input path Apple can change or remove in any toolchain update, with no contract
and no deprecation warning — while the MSL lane (`apple_gpu_air`) already
captures the whole front-end saving through the supported input.

The case for building it is therefore *architectural*: it puts Apple where ROCm
already is — device code produced by the compiler rather than by a shell-out
over synthesized source — and it is the only way Apple joins the MLIR/LLVM spine
the other three backends share. Decision #26a names exactly that condition for
revisiting. This probe supplies the missing cost and risk numbers; the call is
a judgement about risk appetite, not about difficulty.

Reproduce: `xcrun metal -S -emit-llvm <kernel>.metal` on any synthesizer output.

## APPLE-AOT-3: S0 result — a GPU executes hand-written AIR IR

**Status: PASSED 2026-07-28 on the owning host (Apple M1 Max / apple7, Metal
toolchain 32023.883). Host-verified numerics, not a perf claim.**

The four things APPLE-AOT-2 listed as unverified are now settled, three by
experiment and one against me:

| question | result |
|---|---|
| does a GPU *run* hand-written AIR IR? | **yes** — `o[i] = a[i]*3.0f` written directly as LLVM IR, no MSL front end, dispatched via `newLibraryWithURL:`; output bit-exact vs `x*3` over 1024 elements |
| is `.ll` input to `metal` supported? | **no** — it works (`-x ir` too), but Apple documents MSL as the only supported input and deliberately does not document AIR |
| is a `.metallib` portable across GPU families? | **not family-tagged** — `metal-lipo -info` reports `architecture: air64_v28`; the tag is the *AIR version*, which tracks deployment target (`-mmacos-version-min=14.0` → `air64_v26`, `15.0` → `v27`). GPU-specific compilation happens later, at pipeline creation — which is also why ~15.2 ms remains in the AOT lane. Cross-family *execution* untested: one machine. |
| does the shared-dispatch refactor shrink runtime code? | **no — my prediction was wrong.** Measured: 58 lines before, 84 after (48 shared + 18 + 18) for two lanes. Duplication would have been 116, so the *marginal* cost per lane drops 58 → 18 lines (3.2×). It grows in absolute terms; it is cheaper than duplicating. |

### What S0 changes

The AIR path needs **no reverse engineering**. The stalled LLVM `air64` RFC was
blocked because it reimplemented Apple's bitcode writer and container; emitting
IR *into* `xcrun metal` requires neither. The metadata contract is declarative
and legible — `!air.kernel` naming the function, one `!air.buffer` per argument
(location index, access, address space, element size/align/type/name), the
builtin descriptor for `thread_position_in_grid`, `addrspace(1)` device
pointers. The fixture is `tests/data/apple/handwritten_air.ll`, exercised by
`test_gpu_executes_hand_written_air_ir`.

The shape is also ordinary rather than exotic: NVIDIA is MLIR → NVVM → PTX →
**ptxas** → cubin. Apple would be MLIR → AIR IR → **metal/metallib** →
metallib. A vendor assembler in the chain is normal.

### The risk that decides it

`.ll` input is **unsupported**. It is not in any Apple documentation, there is
no man page, and AIR is undocumented by deliberate choice. So an MLIR → AIR
emitter would rest on an input path Apple can change or remove in any toolchain
update, with no contract and no deprecation warning. The MSL lane has no such
exposure — MSL is the documented, supported input, and `apple_gpu_air` already
captures the whole front-end saving through it.

That is the trade to decide, and it is now a clean one: **structure (compiler-
produced code, parity with ROCm's tessera-opt-emitted hsaco) against supported-
ness.** Not, as Decision #26a assumed, feasibility — feasibility is settled.

### C1 landed alongside

`AppleAIRRunner` is registered, closing the gap that made `apple_gpu_air` the
only registered target without a runner. It registers with `default=False` so it
cannot become the process default by import side effect — that would silently
move every F4 verification onto the AOT lane. Only `run_fused_region` has an AOT
dispatch; the other three return a `REFERENCE_EXECUTIONS` tag so the oracle
trusts the reference rather than comparing numpy against itself.

New C ABI: `tessera_apple_gpu_metallib_elementwise_f32` — a generic 1-in/1-out
metallib dispatch. Written for S0, but it is the shape most synthesized
pointwise kernels take, so it is the first of the APPLE-AOT-2 phase-B entries
rather than scaffolding.

## APPLE-AOT-2: close out the `apple_gpu_air` lane

**Status: open, plan of record 2026-07-28.** APPLE-AOT-1 proved the lane works
and is worth ~14.5 ms per cold kernel. This is what it takes to make it a lane
the compiler can actually *use* rather than a demonstrated capability.

### Where it stands, measured not assumed

| | emitter | compiler (`compile_fn`) | runner | dispatch symbols |
|---|---|---|---|---|
| `apple_gpu` | ✅ | deferred (`None`) | ✅ registered | 17 |
| `apple_gpu_air` | ✅ (delegates) | ✅ real `.metallib` | ❌ **none** | **1** |
| `nvidia` / `rocm` / `x86` | ✅ | ✅ real `.so` | ✅ registered | n/a |

Two gaps carry everything else. `apple_gpu_air` is the **only registered target
without a `KernelRunner`**, so it is invisible to `build()`'s execute half, to
the F4 oracle, and to any future arbiter — the ad-hoc
`apple_air.run_fused_region_aot` is the only way in. And the runtime has **1 of
17** dispatch entry points in an AOT form, so the lane can only run a coopmat
matmul-epilogue.

### A1 — register an `AppleAIRRunner` *(small, unblocks everything else)*

Implement the four `KernelRunner` methods over the metallib path and
`register_runner(...)`. Set `accuracy_atol` (ROCm sets `0.005` for its f16
budget; Apple's f16 coopmat measured 1.2e-4 against the f32 reference, so the
f32 default is probably right — confirm, do not inherit by omission).

Until this exists, nothing generic can select the AOT lane, which makes A3 and
C untestable.

### A2 — make the `artifact` / `deferred` contract safe

Nothing outside `apple_air.py` and its tests currently reads
`CompiledKernel.artifact` or `.deferred`. The moment an arbiter iterates targets
and assumes `artifact` is a loadable path, it breaks on `apple_gpu` as a `None`
surprise — and the three `.so`-returning backends make that assumption easy.
Add an accessor that forces both cases to be handled, plus a guard test that
every registered target either returns a path or sets `deferred`.

### B — coverage: 1 → 17, without 17 copies

The expensive way is a hand-written AOT twin per symbol. Do not do that. The
pattern already used for coopmat is the cheap one: extract the dispatch body
(`dispatch_matmul_epilogue_coopmat`) so the JIT and AOT entries differ *only* in
how they obtain the pipeline, via `compile_msl_kernel` or `load_metallib_kernel`.
Applying it to the remaining families makes the JIT entries thinner too, so
total runtime code goes down rather than up.

Families, in the order their value lands:

1. `matmul_epilogue` scalar / tiled (f16, f32) — completes the region the lane
   already serves.
2. `pointwise` + `pointwise_reduce` (f16, f32) — the largest op population.
3. `norm_chain` (f16, f32).
4. `attention` (f16, f32) — the one where cold-compile cost is felt most, since
   attention kernels are the biggest source the synthesizer emits.
5. `gated_matmul` (f16, f32).
6. `tile_simdgroup_gemm` (f16, bf16).

Each new C ABI symbol needs its non-Darwin stub; the ratchet in
`test_apple_runtime_stub_parity.py` fails the build if one is missed.

### C — the arbiter *(the actual payoff, and fleet-wide)*

Selection between `apple_gpu` and `apple_gpu_air` per
`(op, shape-bucket, dtype, target)` on measured evidence — Decision #28's
measured, accuracy-budgeted arbiter, for which this is the first backend with
two genuinely comparable candidates. It needs a persisted decision record, and
it must treat the offline build cost as amortised (~5 cold launches) rather than
per-launch.

This is where the Apple work stops being Apple-specific: the arbiter is shared
infrastructure, and ROCm/CUDA will feed it candidates too.

### D — cache maturity

Artifacts live in `$TMPDIR/tessera-apple-air` with no eviction and no sharing.
Before this is load-bearing: a durable location, an eviction policy, and a
decision on whether artifacts are fleet-shareable (they are host-ISA-specific,
so probably per-machine — but the `kernel_cache` key does *not* include host
identity today, which is the same latent collision X86-1 flags for `-march=native`).

### The real gap to ROCm and CUDA (corrected 2026-07-28)

An earlier draft of this plan claimed ROCm and CUDA "have no JIT lane at all"
and so had "nothing to catch up on". **That was wrong**, and wrong from
absence-of-evidence: it was inferred from `nvrtc`/`hiprtc` not appearing in
`emit/nvidia_cuda.py` and `emit/rocm_hip.py`. Those two files indeed have none —
but the shipping runtime lanes do. `runtime.py` documents the ROCm WMMA lane as
"HIPRTC-compiled for the device arch (gfx1151/gfx1100) **at load**" and the
NVIDIA lane as "NVRTC-compiled warp-level mma.sync". There is a dedicated
`nvrtc_jit.cpp` in the NVIDIA backend. Both vendors JIT.

What each backend actually does, counted rather than assumed:

| backend | AOT artifact | produced by | JIT path | weight |
|---|---|---|---|---|
| ROCm | **hsaco** | **`tessera-opt`** — `convert-gpu-to-rocdl` → `rocdl-attach-target` → `gpu-module-to-binary` | HIPRTC at load (WMMA lane) | hsaco dominant: 601 references |
| NVIDIA | prebuilt `.so`, kernel NVRTC'd inside at load | cmake + NVRTC | NVRTC at load | JIT-dominant; no cubin/fatbin path in `runtime.py` |
| Apple | `.metallib` | **`xcrun metal` shell-out from Python** | `newLibraryWithSource:` at launch | JIT-dominant; AOT at 1 of 17 |

Three corrections follow, and they change the plan's framing:

1. **Apple is behind ROCm on AOT, not ahead of it.** ROCm's precompiled lane is
   the dominant one and has been for a long time; Apple's is one kernel old.
2. **The gap to ROCm is architectural, not coverage.** ROCm's AOT artifact comes
   *out of the MLIR pipeline* — the compiler produces the binary. Apple's comes
   out of a Python subprocess calling a vendor CLI. Closing B gets Apple to
   ROCm's *coverage*; it does not get Apple to ROCm's *structure*.
3. **NVIDIA is the backend closest to Apple's position**, not the distant one —
   its device code is NVRTC-compiled at load and it has no precompiled lane in
   `runtime.py`. The AOT-vs-JIT question is genuinely open there, and the
   measurement method (with its cache control) transfers directly.

This also reframes the AIR deferral recorded in Decision #26a. That deferral
rests on AIR saving no more than the ~15 ms `apple_gpu_air` already captures,
which stands. But the *architectural* case is stronger than that framing
suggested: an MLIR → AIR path would put Apple's AOT artifact where ROCm's
already is — produced by the compiler rather than post-processed by a shell-out.
Revisit on that basis, which is exactly the "architecture, not performance"
condition #26a names.

### Sequencing

A1 → A2 → B1-B2 → C, with D before C ships. A1 is hours and unblocks the rest;
B is mechanical but the bulk of the work; C is the only part that needs design
discussion, and it should be designed fleet-wide rather than for Apple alone.

## APPLE-AOT-1: `.metallib` pipeline creation measured against the JIT lane

**Status: measured 2026-07-28 on the owning host (Apple M1 Max / apple7,
macOS 26.5.2, SDK 26.5, Metal toolchain 32023.883, `air64-apple-darwin25.5.0`).
Host-wall timing, not device-event evidence; not selector-eligible.**

`apple_gpu_air` (`emit/apple_air.py`) compiles synthesized MSL ahead of time —
`xcrun metal -c` → `.air` → `xcrun metallib` → `newLibraryWithURL:` — against
the default `apple_gpu` lane's `newLibraryWithSource:`. Both run the same
synthesized coopmat kernel and the same
`dispatch_matmul_epilogue_coopmat` in the runtime, verified **bit-identical**
(max |diff| exactly 0.0, both 1.2168e-4 from the f32 reference at f16 storage).

Cold pipeline creation + one dispatch, 256×256×256 f16 coopmat, n=25, a
never-before-compiled kernel per sample, device pre-warmed, lanes interleaved:

| lane | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| JIT `newLibraryWithSource:` | 28.7 | 29.3 | **29.7** | 30.0 | 30.5 |
| AOT `newLibraryWithURL:` | 14.9 | 15.1 | **15.2** | 15.4 | 15.8 |
| AOT offline build (excluded) | 72.2 | 73.0 | **73.7** | 74.4 | 77.2 |

**AOT roughly halves pipeline creation — ~14.5 ms saved, 1.95×.** The offline
build costs ~73.7 ms once per kernel per machine and repays after ~5 cold
launches. Warm steady state is a wash (0.36/0.39, 0.69/0.72, 1.61/1.53 ms at
128/512/1024 cubes) — expected, since both are then a cache lookup into the
same dispatch.

**The measurement needs a control, and the obvious one is wrong.** Metal keeps
an on-disk shader cache that survives process exit: the same kernel measured
140.8 ms in one process and 0.5 ms in the next. Timing "first launch in a fresh
process" therefore measures whether that kernel was ever built on this machine.
A first attempt controlled it with a unique *unused `constant`* — which the
Metal compiler drops as dead code, producing byte-identical metallibs, so the
AOT lane reloaded one artifact and reported 1.2 ms (a 13× win). Renaming the
kernel **entry point** per sample makes each library genuinely distinct and
moves both lanes: JIT 15.7 → 29.7, AOT 1.2 → 15.2. The *saving* was stable
across both methods (14.6 vs 14.5 ms); the *ratio* was not (13× vs 1.95×).
`test_apple_air_target.py::test_nonce_control_defeats_metals_shader_cache`
asserts the artifacts differ, so a toolchain change cannot silently restore the
flattering number.

**Strategic read for Decision #26a.** The ~15 ms AOT removes is the MSL
front end. The ~15.2 ms that remains is AIR → GPU-ISA, which *any* AIR-based
path still pays — so emitting AIR directly from LLVM IR would save the same
~15 ms and no more. The ceiling on this whole direction is about half of cold
pipeline creation, which should temper how much the undocumented-format and
legal exposure of direct AIR emission is worth.

Reproduce: `python3 benchmarks/apple_gpu/benchmark_aot_vs_jit.py --samples 25`.

**Decision (2026-07-28): ship the AOT metallib lane; defer a direct AIR
emitter.** `apple_gpu_air` is the fast path and stays on supported tooling. A
direct LLVM IR → AIR emitter is not scheduled: it would save the same ~15 ms
this lane already captures, because the residual cost is AIR → GPU-ISA which any
AIR path pays. Its case is architectural — sharing the LLVM lowering with
CUDA/ROCm/x86 — and should be reopened on that basis, with a measured need, not
on compile-time grounds. SPIR-V → SPIRV-Cross → MSL is rejected: it cannot
express `simdgroup_matrix`, so it would cap the Apple ceiling.

**Cross-backend note.** This is the fleet's fast-path shape, not an Apple
special case: a precompiled artifact behind `register_compiler(target,
compile_fn)` plus the content-addressed cache. NVIDIA, ROCm, and x86 already
return real artifacts (`.so` via nvcc / hipcc / clang); Apple was the only
`deferred` compile-on-launch lane until now. The same AOT-vs-JIT question is
expected on ROCm and CUDA as their performance work ramps — reuse this harness,
and reuse its **cache control**: a never-before-compiled kernel per sample, or
the number is the vendor's shader cache rather than the compile strategy.


Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` moves the last 43
self-resolving test files onto the shared `tests/_support/compiler_tool.py`
driver contract and folds `CompilerToolchain` onto the same resolver and
capability check, leaving one of each in the tree. Apple is **parity
validated**, not merely unaffected: the Apple compiler-tool fixtures
(`test_apple_canonical_gemm.py`, `test_apple_threadgroup_pipeline.py`) reach
the driver through `CompilerToolchain.require_tessera_opt`, whose bare-pass-name
spelling is preserved, and which now also discovers
`build-apple/tools/tessera-opt/tessera-opt` — a candidate the old
`CompilerToolchain` search order did not carry, so a `build-apple`-only tree
that previously skipped as "not built" now resolves. Selection prefers an
in-repo build but takes the first candidate registering the requested passes.
No Metal registration, MSL/MPS schedule, runtime ABI, selector, storage policy,
device evidence, or timing gate changed, and **no exact-device evidence is
claimed or required** for this host-free infrastructure change. Apple's
separately owned package and exact-device gates are untouched.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` adds no Apple capability
claim. It proves exact optimized BF16 forward and deterministic five-entry
backward attention on gfx1151 for the shared ragged-GQA,
bias+softcap+causal-window+dropout contracts. AMD BF16 WMMA, LDS scheduling,
HSACO packaging, HIP launch workspace, numerical results, and resident
host-wall timing are architecture-owned and do not transfer to Metal. Apple
retains its separately owned storage policy, package, exact-device, and timing
gates while shared semantic parity remains unchanged.

Cross-backend sync `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` is **closed**.
The shared lit resolver now accepts `TESSERA_OPT_BIN`, `TESSERA_OPT_PATH`, and
`TESSERA_OPT_CPP` after the canonical `TESSERA_OPT` override, and the validation
script forwards its selected binary through that contract. Exact gfx1151
verification proves the full ROCm driver, legitimate lean ROCm artifact
driver, conflict rejection, both named streaming-attention fixtures, the
seven-fixture filter, and the complete 50-test ROCm backend lit suite. This is
shared test/build infrastructure only; no Metal registration, schedule,
runtime ABI, device evidence, or selector changes.

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` makes ROCm gfx1151 the
first direct physical consumer of the shared tensor-valued attention backward
phase loops. Apple remains **follow-up required** to validate the same
dQ/split-dK/dV/fixed-reduction contract and map it to a Metal-owned package.
The AMD WMMA schedule, five-entry HSACO, HIP launch workspace, gradient
evidence, and host-wall timing do not transfer. No shared IR or Apple
capability state changed in this ROCm-owned closure.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the deterministic split/reduced backward contract as tensor-valued
shared `scf.for` bodies: dQ is query-head/block owned, dK/dV partials are
launch-owned `[split,B,Hkv,Sk,D]` tensors, and reduction is fixed ascending
split order. The shared forward KV recurrence now carries registered additive
bias and softcap operations in canonical
`softcap(scale*QK^T + bias)` order, including rank-4 per-head bias. Apple is
**follow-up required** to lower these shared phase operations into its Metal
backward package and direct forward schedule. The gfx1151 ABI repair, HSACO,
numerical result, and resident timing do not transfer.

Cross-backend sync `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified
split count, launch-owned workspace, block-loop metadata, ascending reduction
order, and canonical `softcap(scale*QK^T + bias)` semantics to the shared
carrier/oracle. Apple is **follow-up required** to map this form to Metal and
validate dropout replay; gfx1151 code, evidence, and timing do not transfer.

Cross-backend sync `ROCM-E2E-ATTENTION-BACKWARD-2026-07-26` is not applicable
to Apple physical execution. It adds a ROCm-owned five-entry HSACO and
gfx1151 split/reduced launch workspace without changing the shared launch
descriptor schema or canonical backward loop. AMD WMMA kernels, workspace
topology, exact-device gradients, timings, and selector state do not transfer.

The ROCm optimized-attention feature follow-up under
`ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` adds AMD-only deterministic dropout
replay and combined bias+softcap consumption to the gfx1151 WMMA schedule,
plus a host-wall resident performance ratchet. The semantic combinations are
already represented by the shared carrier; no shared ABI or Apple capability
changed. Apple parity therefore remains unchanged and its Metal lowering,
counter implementation, numerical proof, and timing evidence do not inherit
from this ROCm result.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` retires every active
`#tile.buffer_ref` compatibility reader after migrating the shared
barrier-reuse and WarpSpec lifetime fixtures plus the ROCm LDS fixture to
`!tile.buffer` def-use. The deprecated attribute remains parser-visible only
for migration diagnostics and archived IR. Apple/shared IR therefore no longer
depends on name-based allocation identity; Metal threadgroup scheduling remains
Apple-owned follow-up. The same sync generalizes the proven Apple ReplaySSM
lifecycle schema to target-keyed resident ABIs and adds explicit MoE launch
workspace ownership plus optional rank/device topology binding. Apple retains
its existing session-private ring, flush/rollback, ordered submission, and
drain-before-release semantics; ROCm execution and evidence do not transfer.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` is a ROCm-owned
physical consumer and exact gfx1151 evidence landing for the already-shared
forward/backward attention carriers. Apple requires follow-up for any carrier
variant not already covered by its Metal execution contract; AMD wave32 WMMA,
LDS ownership, HIP descriptors, scalar recurrence, resources, timings, and
selector state are not applicable to Metal and do not transfer. The ROCm v2
benchmark's separate operation-total and resident
`hipModuleLaunchKernel`/`hipDeviceSynchronize` host-wall domains are likewise
not applicable to Metal timing or selector policy. No Apple readiness or
exact-device row changes.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` lands an AMD-owned
consumer of the existing shared `!tile.buffer`, `!tile.async_token`, and
`!tile.pipeline_state` vocabulary without changing those shared definitions.
Apple parity is validated at the portable IR boundary only. AMD LDS layouts,
waitcnt/s_barrier sequencing, gfx1151 structural evidence, compiler timings,
and selectors are not applicable to Metal and do not transfer. Apple retains
its separately recorded follow-up for architecture-owned threadgroup
allocation and pipeline-state lowering.
**Resolved 2026-07-27 by APPLE-PIPE-1 (row 19).** Apple is now a real consumer
of the same vocabulary: `tessera-apple-threadgroup-pipeline` places `smem`
`tile.alloc` into one capacity-bounded Metal threadgroup arena and claims
`!tile.pipeline_state` rings as ping-pong staging, with the NVIDIA-only
TMA/mbarrier/TMEM vocabulary rejected by named diagnostic.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` makes terminal
sub-byte storage a target + operation + physical-descriptor + complete
def-use-consumer decision. The newly admitted packed load/unpack, supported
round trip, packed matmul, and explicit conversion paths are NVIDIA SM120
consumers only. Apple remains disabled for generic terminal FP4/FP6
legalization until architecture-owned Metal physical consumers and exact
device proof land; no CUDA schedule or evidence transfers.
**Rejection proven 2026-07-27 by APPLE-DTYPE-1-REJECT (row 22).** The SDK gate
is now enforced rather than incidental: `apple_gpu` stamps no
`tessera.storage_packed` where `nvidia_sm120` does, and an unrouted
cooperative-matrix descriptor is refused with `APPLE_MMA_STORAGE_UNSUPPORTED`.
Apple remains disabled for the legalization itself; this proves the block, it
does not lift it. The deprecated
`#tile.buffer_ref` attribute remains parser-only for archived IR; no
Apple/shared fixture or active pass consumes it. Apple capabilities, execution
rows, and selectors are unchanged.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`
carrying the FP32 output accumulator, running maximum, normalization sum,
producer/consumer `!tile.pipeline_state` values, and absolute boundary offset.
The shared TMA-shaped seam now retains typed block coordinates and logical
source extents for ragged zero fill; NVIDIA WarpSpecialization no longer emits
name-based `#tile.buffer_ref` or annotation-only `#tile.pipeline_state`
metadata. Apple is **follow-up required** to map the same recurrence onto an
architecture-owned Metal/MPS attention schedule and threadgroup allocation
identity.
**Rank-2 resolved 2026-07-27 by APPLE-ATTN-STREAM-1 (row 21).**
`tessera-apple-streaming-attention` re-forms the rank-2 recurrence as one Metal
flash-attention dispatch carrying `causal` / `logical_sk` / `window_left/right`
/ `kv_block` read off `tessera_attn.boundary_mask` rather than re-derived.
Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` adds shared rank-4 batch/head
distribution and a direct ROCm consumer. Apple remains **follow-up required**
for its architecture-owned rank-4 Metal/MPS consumer — APPLE-ATTN-STREAM-1
covers rank-2 only, and row 24 owns the rank-4 gap; the gfx1151 schedule,
HSACO, resources, wall timing, and selector evidence do not transfer. Deterministic
backward workspace materialization remains open shared work; no Apple
capability or selector changes in this synchronization slice.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` changes the shared
Graph/Schedule→Tile GEMM contract to explicit M/N/K `scf.for`, FP32/INT32
loop-carried accumulation, zero-pad ragged guards, structured layouts, and SSA
pipeline dependencies. Apple is **follow-up required** for an
architecture-owned Metal/AMX/SIMD consumer where that loop is preferable to
Accelerate; the existing value-mode Accelerate GEMM remains intentionally
unexpanded. NVIDIA Tensor Core fragments, PTX, SM120 resource/cache/timing
evidence, and selectors do not transfer to Apple CPU or GPU. No Apple
capability, execution state, schedule, or selector changes in this slice.
**Resolved 2026-07-27 by APPLE-TILE-2 (row 20).** `tessera-apple-canonical-gemm`
recognizes the three-deep nest and re-forms it as one `simdgroup_matrix`
dispatch carrying the loop's tile decision, fp32 accumulation, and the ragged
zero-pad guarantee, with exact-device execute-and-compare on Apple7. The
incumbent rule stands: recognition is not promotion, and value-mode
Accelerate/MPS remains the production route.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` is **parity validated**
for Apple. The shared Tile change is limited to preserving the canonical
ragged-zero-fill guarantee across `tessera.matmul` → `tile.mma`; APPLE-TILE-2
already consumes the same loop and guarantee. ROCm's address-space-3 LDS
schedule, barriers, gfx1151 WMMA, HSACO resources, and host-wall results do not
transfer to Metal. No Apple route, capability, execution state, or selector
changes in this slice.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: shared lit feature
hygiene now rejects undefined requirements and obsolete global GPU target
flags. The retired fixtures and decoupled CUDA/HIP instruction probes were
NVIDIA/ROCm-owned; Apple has no unsupported fixtures in the LLVM23 suite, and
no Metal IR, runtime route, schedule, evidence, or selector changed. This is
not applicable beyond parity validation of the shared test infrastructure.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: compiler-owner
markers now skip a foreign compiler proof with its required system named in the
pytest summary (Apple, CUDA, ROCm, X86, or AVX512), including a per-system
count. Apple owns the host-free gate integration; this changes no Metal IR,
runtime route, schedule, evidence, or selector.

Cross-backend sync `STATEFUL-TRANSPORT-FOUNDATION-2026-07-19`: the shared launch
workspace schema now distinguishes per-launch scratch from session-persistent,
preserved state. ReplaySSM and MoE metadata contracts are portable, but this
NVIDIA slice changes no Metal allocation, command-buffer ordering, physical
schedule, resource claim, timing row, or selector. Apple resident ReplaySSM
must map its already-proven handle lifecycle to the shared fields in an
Apple-owned follow-up; local and distributed MoE routes retain their existing
Apple evidence and cannot inherit CUDA bandwidth.

Cross-backend sync `NVIDIA-E2E2-STATEFUL-REDUCE-2026-07-19` extends the shared
Tile surface with explicit ReplaySSM decode/flush, MoE dispatch/combine/grouped
GEMM, and `Outer/AxisExtent/Inner` reduction carriers, plus a backend-neutral
rank/device topology fingerprint. Apple does not inherit PTX images, CUDA
workspace residency, the serial/cooperative-128 reduction schedules, NCCL
submission, resources, timings, or selectors. Mapping these semantics to the
existing Metal/MPS and distributed-MoE routes is Apple-owned follow-up; FP8
epilogue execution remains SDK-gated and TF32 is not applicable to Apple.

This plan brings the proof discipline established by the CUDA and ROCm work to
the Apple backend. It complements [`APPLE_AUDIT.md`](APPLE_AUDIT.md), the
generated execution inventory, and the durable architecture under
[`docs/backends/apple/`](../../../backends/apple/). The generated execution
inventory is the authority for exact-target execution state (including
`native_gpu` versus `reference_cpu`); the durable backend documents are the
architecture authority. This file owns only the active execution order and
completion gates.

The goal is not to transplant CUDA warps or AMD waves into Metal. Apple route
selection must be measured across MPS, MPSGraph, synthesized MSL,
`simdgroup_matrix`, Metal 4 cooperative tensors/MPP, and authored package
subgraphs. Logical fixtures, ABI contracts, numerical oracles, diagnostic
rules, and benchmark schemas should be shared with CUDA and ROCm; physical
fragments, threadgroup shapes, residency strategy, and command-buffer schedules
remain Apple-owned.

## Current state and immediate risk

- APPLE-TEST-1 now has a structural inventory at
  `tests/_support/apple_inventory.py`: its current scan records **0** direct
  Apple/Darwin/Metal capability gates. Apple device, integration, compiler-tool,
  and portable simulated-host cases are classified at their actual proof
  boundary. Offline MSL
  compiler checks are now `compiler_tool` tests with a shared `metal`-tool
  boundary, rather than device-gated tests. The first
  cohorts raise `pytest -m hardware_apple_gpu` collection from **3 to 976 of
  15,374** unit tests: the MPSGraph warmup and MegaMoE measured paths, exact
  native proofs for f32 CSR/COO SpMM, SDDMM, BSMM, scatter, optimizer, local
  MoE, MoE transport, and RNG, plus gather/concat/slice/softcap/transpose,
  mixed-program residency, TopK, projections, BMM, reduction, MPSGraph
  Tier-1, composed MHA, MPSGraph-runtime/cache, control-flow stress, and
  memory-budget residency proofs, quantized matmul, TopK, complex-runtime,
  evaluator/native-required, value-executor, fusion-synthesis, GA/EBM benchmark,
  control-flow/tracing, attention, delta, LDT, MoE, and other JIT-route proofs.
  The shared pytest boundary now supplies the Darwin/Metal skip; the marked
  proofs retain their explicit `native_gpu` and `metal_runtime` assertions
  where JIT provenance is available.
  APPLE-TEST-2 binds the first cohort's
  execution-matrix row,
  generic-envelope ownership where applicable, runtime ABI symbols, marked
  native node, and explicit fallback node in one registry; the shared native
  assertion rejects a semantic `reference_cpu` result. The f32 MPS matmul,
  MPSGraph BSMM/gather, and Philox symbols used by the cohort are now
  ABI-registered, so APPLE-REG-1 rejects an unregistered replacement.
- **2026-07-18 APPLE-TEST-1 closure:** a fresh full unit-tree collection found
  **976 of 15,374** nodes behind the centralized `hardware_apple_gpu` boundary,
  while the structural inventory still found **0** direct Apple/Darwin/Metal
  capability skips. The residency, runtime, and offline Metal-compiler cohorts
  retain their marker/provenance ratchets, so a newly added inline gate or
  misclassified compiler test fails the portable inventory suite. Ongoing
  classification enforcement is maintenance, not an open implementation rung.
  **APPLE-TEST-1 is closed.** No shared marker semantics or sibling-backend
  ownership changed; NVIDIA and ROCm are not applicable.
- APPLE-CI-2 now has an executable host-free ownership gate:
  `scripts/run_apple_host_free_compiler_tests.py`. It reads the CMake backend
  declarations, probes Apple/NVIDIA/ROCm pass registration, then selects only
  `compiler_tool` tests owned by the declared compiler capability set. On the
  current Apple-only build, Apple lowering is registered while the NVIDIA and
  ROCm probes are explicitly unregistered; the selected Apple artifact lane is
  green. The gate accepts any CMake cache type for `LLVM_DIR`, resolves and
  verifies the matching MLIR runner-utils dylib, and exports that exact path to
  the selected tests. Foreign compiler tests carry `compiler_nvidia` or
  `compiler_rocm`. This closes APPLE-CI-2. NVIDIA and ROCm are not applicable:
  their compiler ownership expressions and toolchain runners are unchanged.
- Cohort ledger: **APPLE-TEST-2-C1 / APPLE-REG-1-C1** records f32 sparse
  transport (CSR/COO SpMM and SDDMM), BSMM, scatter, optimizer, local MoE,
  MoE transport, and Philox RNG. Each row binds its execution-matrix path,
  native and fallback node, and runtime ABI symbols in
  `apple_exact_device_proofs.py`; complex/conformal remains outside this cohort
  until a hardware-marked execute/compare proof replaces its fallback-capable
  portable tests.
- Cohort ledger: **APPLE-TEST-2-C2 / APPLE-REG-1-C2** records only the fused
  interleaved-f32 complex/conformal subset (`complex_mul`, `complex_exp`,
  `mobius`, and `stereographic`). The device proof requires a traced fused ABI
  route and numerical oracle; its bridge-miss negative is explicitly
  `reference_cpu`. The long-tail complex/certificate operations remain outside
  C2 because they are intentionally host structured or lack a fused ABI route.
- Cohort ledger: **APPLE-TEST-2-C3 / APPLE-REG-1-C3** records only f32
  MPSGraph `sum`, `mse_loss`, and `mae_loss` (binary subtraction plus
  multiply/absolute-value plus reduction).
  Their exact-device nodes execute and compare on Metal; a forced missing
  MPSGraph binding must return `reference_cpu` from `runtime.launch`, rather
  than retaining the execution-matrix default. Huber, smooth-L1, log-cosh, and
  the loss-family lane remain outside C3 because their middle computations are
  host structured. NVIDIA/ROCm require no plan change: their loss/reduction
  paths have separate exact-device owners and no shared ABI changed.
- **2026-07-17 C1–C3 exact-device evidence:** all 12 distinct ledger-native
  nodes passed twice on Metal from separate freshly compiled runtime images;
  the 12 corresponding fallback-injection nodes passed and asserted
  `reference_cpu`. The two C2 rows intentionally share one fused
  complex/conformal native node and one bridge-miss negative. This closes the
  first cohort's placement, oracle, fresh-runtime, and fallback-negative
  evidence only; **APPLE-TEST-2 remains landing/open** until the same proof
  ladder covers the remaining Apple families, ordering/stress, and performance
  layers.
- **2026-07-17 broader exact-device evidence:** two independent fresh-runtime
  runs collected 853 nodes and each completed **849 passed, 4 skipped, 0
  failed** (97.4 s / 99.2 s). The four skipped legacy hand-written synthesis
  comparisons have now been explicitly reclassified as retired, non-native ABI
  contracts; their live synthesized replacements carry the Metal-placement and
  numerical-oracle evidence, and a forced missing-synthesis binding must return
  the reference route. A third fresh-runtime post-reclassification run completed
  **850 passed, 0 skipped, 0 failed** (100.5 s). The LLVM/MLIR 23 migration also
  fixed the JIT engine transformer's dangling callback and bounded the
  process-wide ExecutionEngine cache, which had previously made serial device
  validation segfault after accumulated JIT compiles.
- **2026-07-17 stateful and performance ladder evidence:** a separate fresh
  runtime passed the package/session-cache, resident block-paged KV, ReplaySSM,
  command-buffer, MPSGraph-LRU, and control-flow cohort (**63 passed**), with
  the bulk-MPSGraph/control-flow ordering stress raised to 75 iterations. Two
  independent route-characterization runs (21 rows each) and two independent
  ReplaySSM runs (12 rows each) reported native dispatch and numerical
  validation for every row. The temporary artifacts are
  `/private/tmp/apple-routes-proof-{a,b}.json` and
  `/private/tmp/apple-ssm-replay-proof-{a,b}.json`; they are evidence, not a
  committed performance ratchet. The remaining proof-ledger work is to add the
  same explicit fallback-injection record to the other native family owners;
  the closure update immediately below records the final family set and
  corrected serial performance selection.
- **2026-07-17 APPLE-TEST-2 closure:** the proof ledger now includes the C1--C3
  ABI cohort, synthesized matmul/reduction replacement, paged-KV attention, and
  fused ReplaySSM. ReplaySSM's C ABI now returns an explicit dispatch bit: its
  exact-device node requires `native_gpu` and its forced missing-binding
  negative requires `reference_cpu`, so a numerically identical host reference
  can no longer earn the native rung. The final fresh-runtime correctness lane
  passed **850/850**; the serial measured lane passed **69/69**. Two simulated
  distributed-MoE wall-clock tests were removed from the Apple hardware marker
  because they use modeled communication and do not assert an Apple route; the
  JIT-bridge benchmark fixture typo was corrected. **APPLE-TEST-2 is closed.**
  The plan remains `landing` because APPLE-REG-1, TILE, retuning, paged-KV,
  ReplaySSM serving expansion, and device-keyed performance selection are
  separate owning items.
- **2026-07-17 APPLE-REG-1 closure:** the canonical Apple ABI registry,
  runtime-header ABI, target-map, exact-device proof, and Tile-envelope drift
  gates passed against the LLVM/MLIR 23 `build-apple` compiler. The Tile status
  test now honors `$TESSERA_OPT` before the stale default build path, preventing
  an ABI-incompatible LLVM dylib from masquerading as a lowering failure.
  **APPLE-REG-1 is closed.** No dtype/op/diagnostic/target state was added in
  this slice, so NVIDIA and ROCm are not applicable.
- **2026-07-17 APPLE-TILE-1 start:** the real Tile-to-Apple status/materialized
  artifact gate passes with the LLVM 23 compiler, but it is not yet an
  exact-device fragment proof: the current fixture uses `tile.mock` and asserts
  runtime status only. TILE-1 remains open until a shared logical value fixture
  selects an Apple-owned fragment/layout from target capabilities and proves
  packing, ragged store, geometry/resource record, and native execute/compare.
- **2026-07-17 APPLE-TILE-1 value/ragged evidence:** the value-preserving
  `tile.batched_gemm` path now runs both aligned `2x4x8 @ 2x8x16` and ragged
  `2x5x7 @ 2x7x9` fixtures for f32/f16/bf16. Each exact-device case asserts
  `native_gpu` + `metal_runtime` and compares against the NumPy oracle; the
  fixture supplies only logical shapes, while Apple lowering owns BMM packing
  and route selection. **8 passed.** TILE-1 remains open for an explicit
  selected physical fragment/layout and threadgroup/resource record; the MPS
  BMM value route must not be relabeled as simdgroup-fragment materialization.
- **2026-07-17 APPLE-TILE-1 fragment-materialization landing rung:** Apple7+
  Tile selection now owns an exact `simdgroup_matrix` descriptor: fp16/bf16
  storage, fp32 accumulation, an 8x8x8 MMA fragment, 32 lanes, and a
  `(32,1,1)` threadgroup. The target-selected materializer consumes that
  descriptor to emit the existing steel-shaped MSL artifact with cooperative
  packing, bounds zero-padding, partial-edge store, and double buffering.
  Its host-free structure and target limits gates passed **85 passed, 9
  compiler-tool skips**. At that point this was artifact evidence only; the
  source-backed ABI and exact-device evidence are recorded below.
- **2026-07-17 APPLE-TILE-1 resource-contract landing rung:** each selected
  simdgroup artifact now carries a target-owned record for its `(32,1,1)`
  launch geometry, 32 lanes, staged-A/B bytes, ragged-store scratch, buffering
  mode, and total threadgroup-memory demand. Materialization rejects a tile
  that exceeds the selected target's threadgroup-memory capacity (the
  double-buffered 32x32x16 fp16/bf16 case records 4,352 bytes). The focused
  fragment/emitter/feature suite passed **67 passed, 9 compiler-tool skips**.
  This completed resource planning for the artifact path; runtime evidence is
  recorded below.
- **2026-07-17 APPLE-TILE-1 native single-fragment rung:** a distinct,
  registered `tessera_apple_gpu_tile_simdgroup_gemm_f16` C ABI now accepts the
  selected steel MSL source and entry, binds fp16 A/B and fp32 output, and
  dispatches exactly one 32-lane `(32,1,1)` threadgroup per 8x8 output tile.
  It is separate from the MPS BMM ABI and rejects any other threadgroup size;
  the non-Darwin stub returns 0. A fresh runtime image compiled and ran the
  8x8 fp16 fragment on Metal with zero fp32-oracle error; the focused proof
  test also forces the ABI binding missing and observes an explicit non-native
  result. The follow-on expanded this exact-device proof to bf16 8x8x8 and
  ragged/multi-fragment fp16 `13x16 @ 16x11`; both remain native and match the
  fp32 oracle (**46 passed, 9 compiler-tool skips**). A 30-repetition warm
  end-to-end characterization retained at
  `/private/tmp/apple-tile-simdgroup-characterization.json` reports medians of
  0.310 ms (8x8x8), 0.311 ms (13x16x11), and 0.315 ms (32x16x32); it has no
  device-event timing or MPS comparison, so it is not a selector decision.
  The C++ full pipeline now selects this ABI only for strict static rank-2
  `tile.matmul`/`tile.gemm` with fp16 or bf16; rank-3 `tile.batched_gemm`
  deliberately remains on MPS BMM. The Python value executor materializes the
  selected source and rejects a non-native result rather than using MPS/NumPy.
  The compiler/runtime ABI regression passed **18 passed**. TILE-1 remains open
  for retained runtime resource/provenance telemetry and comparative device-time
  performance selection.
- **2026-07-17 APPLE-TILE-1 telemetry/first comparison rung:** every direct
  source-backed dispatch can now return a record containing the ABI symbol,
  source SHA-256, native/reference result, execution mode, selected resource
  record, and runtime MSL pipeline-cache size. Its focused regression passed
  **17 passed**. A warm 30-repetition `32x16 @ 16x32` end-to-end comparison
  retained at `/private/tmp/apple-tile-simdgroup-vs-mps.json` recorded 0.314 ms
  median for native fp16 simdgroup and 0.229 ms for the existing f32 MPS route.
  These are not equivalent dtype paths and have no device-event timing, so they
  are explicitly **not** a selector decision. Remaining work is equal-dtype MPS
  comparison plus Metal device-time/resource telemetry and a two-run stability
  gate before any production-route change.
- **2026-07-17 APPLE-TILE-1 equal-dtype stability rung:** two independent warm
  30-repetition fp16 `32x16 @ 16x32` comparisons passed their respective fp16
  numerical oracles (the MPS route uses documented `rtol=1e-2` accumulation
  tolerance). Retained evidence at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-two-run.json` measured
  simdgroup medians of 0.336/0.293 ms versus MPS medians of 0.234/0.226 ms.
  MPS is the end-to-end winner for this one shape; no selector changed because
  the runtime at that rung exposed neither command-buffer GPU timestamps nor
  Metal counter sampling. The Tile record supplies selected static resource
  bytes and pipeline-cache state, but not measured occupancy/spills. The next
  required implementation is a dedicated runtime timing/counter ABI, followed
  by a broader shape/dtype corpus and an explicit promotion threshold.
  **Superseded — read as dated provenance, not current capability.** The two
  rungs immediately below implemented that timing/counter ABI. The runtime now
  exposes `tessera_apple_gpu_tile_last_device_time_ns` (completed
  `kernelStartTime`/`kernelEndTime`, `GPUStartTime`/`GPUEndTime` fallback),
  `tessera_apple_gpu_tile_counter_sampling_supported` +
  `tessera_apple_gpu_tile_last_counter_delta` (named
  `MTLCommonCounterSetTimestamp` set, dispatch-boundary sampling), and the
  7-bit `tessera_apple_gpu_profiling_capabilities` matrix, which additionally
  reports the statistic and stage-utilization counter sets, stage-boundary
  sampling, and the Metal 4 timestamp heap. What remains genuinely
  unavailable is narrower and is a **public-Metal limit, not missing Tessera
  work**: register count, scratch bytes, spill count, and true occupancy have
  no public query, so those fields must stay absent rather than inferred.
- **2026-07-17 APPLE-TILE-1 kernel-time rung:** the runtime now records the
  completed command buffer's `kernelStartTime`/`kernelEndTime` (falling back to
  GPU start/end only when available) through
  `tessera_apple_gpu_tile_last_device_time_ns`. The exact-device proof requires
  a positive measured value (**17 passed**). Two 30-repetition equal-dtype fp16
  kernel-time runs retained at
  `/private/tmp/apple-tile-simdgroup-vs-mps-f16-device-two-run.json` measured
  simdgroup medians 23.1/21.4 us and MPS medians 21.8/18.8 us for `32x16 @
  16x32`; MPS wins this shape in both domains. The following bounded-counter
  rung replaces the then-missing capability-gated counter path; no selector
  changed.
- **2026-07-17 APPLE-TILE-1 bounded counter/corpus rung:** the runtime now
  discovers the named `MTLCommonCounterSetTimestamp` set only when the device
  supports dispatch-boundary samples, allocates a two-sample buffer, and
  samples immediately before/after the source-backed Tile compute encoder.
  The dispatch record retains either its measured timestamp delta or explicit
  `counter_sampling_supported: false`; it never manufactures occupancy or
  spill values. This M1 Max reports the latter while retaining positive
  command-buffer kernel timing. The new
  `benchmark_tile_simdgroup.py` corpus made two independent 30-repetition
  warm runs for fp16/bf16, aligned `8x8x8`/`32x16x32`/`256x256x256`, and ragged
  `127x63x129` shapes in both end-to-end and kernel domains. All eight
  end-to-end rows retain MPS. Kernel-only microcase movement is not a
  production promotion: the selector's production domain is end-to-end, where
  MPS remains the route. The selector contract requires native placement plus
  numerical proof, retained resource/counter evidence, and a 5% win in both
  intended-domain runs; no production route changed.
- **2026-07-17 APPLE-GEMM-1 capture-telemetry landing rung:** an opt-in,
  thread-local dispatch record now spans the owned Apple command-buffer paths.
  Legacy MPS/MSL records use completed `kernelStartTime`/`kernelEndTime`
  (command-buffer time only as an explicit fallback); the shared MTL4 encoder
  uses a reusable two-entry timestamp heap and converts its raw tick delta with
  the device timestamp frequency. The same record retains the live MTL4
  threads-per-threadgroup, execution width, maximum threads, and static
  threadgroup-memory properties. Capture is disabled by default so precise
  timestamp sampling cannot perturb production dispatch. The standalone
  MPSGraph row-op path now encodes into an owned `MPSCommandBuffer`, commits its
  live `rootCommandBuffer`, and reports a whole-dispatch interval only when
  MPSGraph did not auto-flush and replace the supplied Metal command buffer;
  occupancy and spill fields remain null rather than inferred.
  `select_stable_gemm_routes.py` aggregates two or more current-schema reports
  by exact Apple GPU family and emits separate device/end-to-end decisions. A
  promotion requires native placement, numerical proof, repeated samples,
  retained resources/counters, at most 15% cross-run drift, and a 5% win in
  every run. Two fresh 30-repetition Apple7 reports at
  `/private/tmp/apple-gemm-stable-{c,d}.json` cover square, rectangular,
  ragged, fp16/f32, MPS, simdgroup, cooperative-tensor, MSL, and MPSGraph
  routes; `/private/tmp/apple-gemm-stable-ledger.json` records **0 promotions,
  13 incumbent retentions, and 9 inconclusive timing-domain rows**. MPSGraph
  device intervals are present in both reports; its three device decisions are
  inconclusive because cross-run drift exceeded the 15% bound. No production
  selector changed. NVIDIA and ROCm are not applicable: this is an Apple-only
  Metal ABI and Apple-only report extension, with no shared IR, schedule, or
  cross-backend benchmark schema change.
- **2026-07-17 APPLE-GEMM-1 paired-winner/resource evidence rung:** absolute
  cross-process latency is now diagnostic rather than a promotion veto. Each
  report runs nine alternating route blocks of 30 repetitions; a candidate
  must win at least 75% of paired blocks, clear 5% median speedup in both fresh
  processes, and keep cross-process speedup spread within five percentage
  points. The committed Apple7 ledger is
  `benchmarks/baselines/apple7_gemm_route_ledger.json`: **3 timing-domain
  promotions, 19 incumbent retentions, 0 inconclusive rows**. Only
  end-to-end winners affect production: f32 `128x257` and `256x256` softmax
  select MPSGraph instead of MSL on Apple7 (24.2--28.0% and 36.9--40.2%
  paired median wins, respectively, winning all 18 blocks). The f32
  `64x64x64` simdgroup route wins device time by 38.5--40.1% but loses end to
  end, so MPS remains production. All other measured matmul shapes retain MPS.
  The new profiling-capability ABI records what public Metal actually exposes
  on this M1 Max: compiled-pipeline limits, stage-boundary timestamp sampling,
  and the Metal 4 timestamp heap are available; statistic/stage-utilization
  counter sets and dispatch-boundary sampling are unavailable. Live MSL/MTL4
  records retain execution width, maximum threads, static threadgroup memory,
  simdgroups per threadgroup, and a clearly named threadgroup-capacity proxy.
  The runtime ABI exposes no register count, scratch bytes, spill count, or
  true occupancy metric, so those per-dispatch fields remain null rather than
  inferred from pipeline limits. A separate bounded Instruments `Metal System
  Trace` now supplies genuine compiler/spill evidence, summarized reproducibly
  by `benchmarks/apple_gpu/summarize_metal_trace.py` in
  `benchmarks/baselines/apple7_gemm_metal_trace_evidence.json`. The exact
  Apple7 process trace retained four compute-shader compile intervals (2.356 ms
  total, 1.486 ms maximum), two MTLibrary creation intervals (0.258 ms total),
  and seven named compute shaders. Exact command-buffer joins observed one
  64-byte spill event on each of ten `tessera.rowop.mpsgraph` submissions and
  zero spill events on ten f32 MPS GEMMs, ten f16 MPS GEMMs, ten explicit MSL
  softmax submissions, and twenty reusable MTL4 submissions. The MTL4 command
  buffer is intentionally reused and Instruments retains it as `Command Buffer
  0`, so its zero-event row is an aggregate MTL4 observation rather than a
  per-kernel claim. The default system-trace template recorded
  `counter-profile=0`, but the standalone `Metal GPU Counters` instrument is
  available on this M1 Max and two bounded captures enabled profile 3 with
  shader profiler 1. Its genuine `Compute Occupancy` counter (ID 24) produced
  376 command-buffer-correlated samples: f32 MPS GEMM retained 144 samples
  (one nonzero sample, 0.282% maximum), the reusable MTL4 buffer retained 12
  zero-valued samples, MPSGraph retained 216 zero-valued samples, and explicit
  MSL softmax retained four zero-valued samples. Those zeros are the measured
  counter values for this small characterization workload, not synthesized
  occupancy estimates; f16 MPS had no in-interval sample and remains null.
  The live threadgroup-capacity/concurrency proxy remains alongside the raw
  counter evidence. NVIDIA and ROCm are not applicable because no shared IR,
  schedule, or cross-backend ABI changed.
- **2026-07-17 APPLE-EPILOGUE-1 native/resource/selection rung:** synthesized
  f32, f16, and bf16 epilogues already had common-oracle coverage for bias,
  ReLU, GELU, SiLU, residual guards, ragged stores, large reductions, and a
  forced symbol-missing negative. The runtime now labels every synthesized
  command buffer and retains its live pipeline limits, actual threadgroup, and
  total pipeline-static plus encoder-requested threadgroup memory. A ragged
  `64x64x2049` tiled softmax proof records at least `2049 * sizeof(float)`
  scratch; an fp16 bias+SiLU proof records the selected cooperative-matrix
  threadgroup and both match the backend-neutral `FusedRegion` oracle.
  MPSGraph unary and binary epilogue dispatches now use an explicitly owned
  `MPSCommandBuffer` and expose status-returning ABI variants, so native
  placement is independent of numerical success. Before the later Metal 4
  envelope closure below, MPSGraph could legally call `commitAndContinue` and
  replace the supplied root command buffer, so timing remained null rather
  than reporting a partial interval.
  `benchmark_epilogue_routes.py` collected two fresh Apple7 runs with seven
  alternating trials of 15 repetitions for aligned `64x64x64`, ragged
  `65x63x67`, and `256x256x256` f32/f16 ReLU plus f32 bias+SiLU. The committed
  `benchmarks/baselines/apple7_epilogue_route_ledger.json` records a stable
  end-to-end synthesized-fusion win for all nine comparable rows (49.8--71.6%
  paired median speedup and 100% paired-block wins in both processes). Device
  decisions remain explicitly inconclusive because the unfused MPSGraph
  segments do not expose complete command-buffer intervals. Production already
  selects the synthesized fused route for these supported regions, so this
  evidence ratifies rather than changes that selector. GELU and bf16 remain
  native correctness/resource proofs but are not compared against a false
  mixed-dtype or missing-MPSGraph incumbent. NVIDIA and ROCm are not applicable:
  the new ABI and schedule evidence are Apple Metal-only and no shared IR or
  numerical contract changed.
- **2026-07-18 APPLE-TILE-1 closure:** the shared logical fixture now selects
  an Apple-owned descriptor and schedule without test-authored physical maps;
  the selected f16/bf16 fragment path has packing, ragged-edge, resource,
  provenance, native execute/compare, device-time, and counter-capability
  evidence. The two-run aligned/ragged corpus retains MPS in every end-to-end
  row. That measured non-promotion is a valid selector outcome, not unfinished
  Tile work. **APPLE-TILE-1 is closed.**
- **2026-07-18 APPLE-GEMM-1 closure:** the paired Apple7 ledger records a
  stable decision for every measured timing-domain row: three promotions and
  nineteen incumbent retentions, with no inconclusive rows. Native placement,
  numerical validation, timing-domain separation, resources, and bounded
  Instruments compiler/spill evidence are retained. New device families or
  candidate routes require a new corpus; they do not keep this Apple7 ratchet
  open. **APPLE-GEMM-1 is closed.**
- **2026-07-18 APPLE-EPILOGUE-1 closure:** every supported f32/f16/bf16
  epilogue has native placement, common-oracle, resource, ragged-store, and
  fallback-negative proof. The two-run Apple7 ledger records stable
  synthesized-fusion end-to-end wins for all nine comparable rows. Unsupported
  pairs retain an explicit non-fused route or registered diagnostic.
  **APPLE-EPILOGUE-1 is closed.**
- **2026-07-18 MPSGraph device-interval closure:** the telemetry-only Metal 4
  bracket writes a timestamp before graph execution, makes MPSGraph wait on
  that event, signals a second event at its documented completed stage, then
  writes the final timestamp after that signal. It therefore spans every
  internal `commitAndContinue` root rotation without treating a partial root
  interval as a graph interval. Unary (including the prior queue-owned
  epilogue path), binary, row-op, transpose, paged gather, and BSMM now encode
  through the owned descriptor path. The result is labeled
  `metal4_mpsgraph_envelope`, deliberately distinct from direct MTL4 encoder
  timing; without a Metal 4 timestamp heap telemetry stays unavailable rather
  than fabricated. Fresh exact-device softmax and epilogue smoke evidence has
  complete MPSGraph interval coverage. The historical Apple7 selector ledger
  is unchanged: a new two-run corpus is required before any device-domain
  selector decision can use this new timing domain. NVIDIA and ROCm are not
  applicable because this is an Apple Metal runtime telemetry path only.
- **2026-07-17 APPLE-ATTN-FWD-1 placement/resource landing rung:** the f32 and
  f16 online-softmax MSL command buffers now carry stable route labels, retain
  their actual `Sq x B`-derived threadgroup and live pipeline limits, and expose
  status-returning ABI variants. The exact-device proof covers ragged
  `B=2, Sq=17, Sk=19, D=128`, causal masking, f32/f16 storage with f32 softmax
  accumulation, positive command-buffer GPU time, and a shared NumPy oracle.
  The D=257 envelope negative returns status 0 and no device interval, so the
  legacy reference fallback cannot be mislabeled native. This is a landing
  rung, not closure: bias, softcap, window, MHA/GQA/MQA, long-context, resident
  command-buffer, cooperative-matrix, and MPSGraph candidate comparisons still
  need the full two-run measured corpus; APPLE-ATTN-BWD-1 is untouched.
- **2026-07-17 APPLE-ATTN-FWD-1 variant/selector rung:** one status-returning
  online-softmax ABI now composes additive bias, causal or sliding-window
  masking, logit soft-cap, and direct MHA/GQA/MQA KV-head indexing for native
  f32/f16 storage. It retains the actual threadgroup and pipeline limits and
  rejects invalid grouping, negative windows, and D>256 before submission. The
  exact-device matrix covers MHA, GQA, and MQA, ragged `Sq=5/Sk=37`, the
  combined bias+window+softcap contract, and MQA `Sk=1025`; every row matches
  the shared f32-accumulation oracle. The MPSGraph BSMM candidate now owns and
  labels its command buffer and returns native status. Two independent Apple7
  runs, each using seven alternating trials of 20 repetitions, compare f32/f16
  aligned `B1/H4/S64/D64`, ragged `B1/H4/Sq65/Sk67/D64`, and throughput
  `B1/H8/S128/D64` plain MHA. The retained
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all six end-to-end rows; production selection is exact-device,
  exact-shape, dtype, and timing-domain keyed. Device timing retains MSL for
  rows without a stable 5% MPSGraph win. The resident command-buffer candidate
  is measured separately in its device-resident input domain and retains live
  resources, but its shared-session command buffer exposes no complete device
  interval. No cooperative-matrix attention ABI exists, so that candidate is
  explicitly unavailable rather than assigned synthetic timing. This is not
  full APPLE-ATTN-FWD-1 closure: wider B/head/D and long-context matrices,
  variant-capable resident/cooperative candidates, and complete device timing
  remain open. bf16 continues to be labeled host-conversion plus f32 GPU
  compute, and APPLE-ATTN-BWD-1 remains separate. NVIDIA and ROCm are not
  applicable because the new ABI, selector, and physical schedule are
  Apple-only; shared attention semantics and numerical policy are unchanged.
- **2026-07-17 APPLE-ATTN-FWD-1 closure:** the forward lane now covers the
  remaining physical and evidence gaps without expanding into backward. The
  selector corpus spans `B=1/2`, 4/8/16 query heads, `D=64/128/256`, aligned
  and ragged lengths, and plain-MHA context through `Sk=1025`. The variant
  corpus adds MHA/GQA/MQA, bias+causal+window+softcap, `B=2`, ragged
  `Sq=65/Sk=67`, and decode-style MQA through `Sk=2049`. The resident scalar
  and one-SIMD-group-per-query-row candidates now accept the same variant ABI;
  the latter is named `cooperative_simdgroup` rather than being mislabeled a
  Metal cooperative-matrix route. No attention-specific cooperative-matrix ABI
  is available on this SDK/host, and that capability remains explicit rather
  than receiving synthetic measurements. f16 and bf16 keep native two-byte
  device storage; GPU-side casts surround f32 accumulation on the resident
  command buffer, with no host fp32 staging inside the attention ABI.
  `ts_enc_commit_wait` now publishes the completed owned-command-buffer Metal
  interval. Two independent Apple7 warm reports, each with five alternating
  trials of ten repetitions, retain 9 MSL variant rows and 18 resident versus
  cooperative rows; every row is native, matches the shared oracle, and every
  resident/cooperative row has 100% device-time coverage. Logical input/output
  bytes, residency, intermediate-storage policy, actual threadgroup/pipeline
  limits, GPU time, and end-to-end time are retained; unavailable occupancy,
  register, and spill counters remain null. The regenerated
  `benchmarks/baselines/apple7_attention_route_ledger.json` promotes MPSGraph
  for all eight plain-MHA end-to-end rows. In the distinct device-interval
  domain only f32 `B1/H16/Sq16/Sk1025/D256` has a stable two-run 5% win;
  all other device rows retain online MSL. `APPLE-ATTN-BWD-1` remains a
  separate open item and no backward implementation or policy changed.
  NVIDIA and ROCm are not applicable: this closes Apple-only runtime ABIs,
  storage handling, schedules, and evidence, with no shared IR, attention
  semantic, or numerical-policy change.
- **2026-07-17 APPLE-PAGED-KV-1 retained staged-gather rung:** the existing
  non-contiguous resident MPSGraph gather now encodes through an explicitly
  owned, labeled `MPSCommandBuffer`. `ResidentBlockPagedKVCache` retains
  `last_gather_execution` and the capture record for each gather; a framework
  pipeline that exposes no public PSO limits records the MPSGraph API and an
  explicit unavailability reason rather than synthetic resources. The
  exact-device proof interleaves two sequences to produce physical table
  `[0, 2, 4]`, gathers the correct non-identity values, and requires native
  status. Existing remap/reuse, concurrent-sequence, exhaustion, and teardown
  tests remain green. This closes provenance loss for the staged candidate but
  not APPLE-PAGED-KV-1: a direct resident page-table attention candidate,
  causal-offset/boundary stress, leak telemetry, and two-domain comparison are
  still required.
- **2026-07-17 APPLE-REPLAY-1 native block/timing landing rung:** output-only
  replay and fp32/f16 block decode now label their command buffers and retain
  live threadgroup/pipeline records. The block ABI returns native status, which
  propagates to `SSMStateHandle.last_block_execution`; N>256 returns an explicit
  reference provenance and common-oracle result. Focused rollback, forced
  binding-miss, f32/f16 block, resource, and ABI tests pass. Two independent
  Apple7 reports at 512 tokens, capacity 16, and 20 repetitions cover
  `1x128x128`, `1x256x128`, and `4x128x64`. The committed
  `benchmarks/baselines/apple7_replay_ssm_evidence.json` retains complete native,
  numerical, resource, end-to-end, and device-per-token evidence for all six
  output-only/block rows. End-to-end cross-run drift is 0.3--2.1%; device drift
  is 0.9--26.8%. The ledger deliberately makes no selector decision because the
  legacy benchmark does not interleave paired route blocks. Persistent resident
  inputs, forced flush/partial rejection/block-submit ordering, asynchronous
  ring backpressure, cleanup stress, and a paired selector corpus remain open.
  NVIDIA and ROCm are not applicable to these Apple-only runtime ABI changes;
  shared SSM state semantics and numerical policy are unchanged.
- **2026-07-18 APPLE-PAGED-KV-1 closure:** `ResidentBlockPagedKVCache` now owns
  one persistent int32 page table per live sequence. Its direct f32 MSL
  candidate forms rope-key scores and latent values by following that physical
  table in one dispatch; the staged peer performs two on-GPU non-contiguous
  gathers plus dense resident attention. Both share the same non-identity
  oracle, right-aligned or explicit causal offsets, and bounded windows. A
  failed multi-block reservation is transactional, lifecycle telemetry accounts
  for live pages/tables/calls, and teardown frees every table and pool. Thirteen
  focused tests pass on the Apple host, including exact-device direct/staged
  placement and equivalence. The committed two-run Apple7 corpus covers
  `127x64x32x1` and `512x128x64x1` with ten measured repetitions after three
  warmups. Direct wins both runs in both device and end-to-end domains and is
  promoted only for those exact f32 rows; unmeasured rows retain staged.
- **2026-07-18 APPLE-REPLAY-1 closure:** the Apple serving handle keeps scalar
  A, S0, and fixed-capacity delta/x/b/c rings in persistent `DeviceTensor`
  buffers. Block submissions encode against those buffers, commit without
  waiting, and rely on ordered Metal command-queue execution. Output slots stay
  leased until `wait()`, enforce explicit backpressure, reject flush/rollback
  while submissions are pending, and are drained during idempotent cleanup.
  Forced flush, ordered multi-block submission, rollback, partial speculative
  rejection, slot reuse, and cleanup match `SSMStateHandle` in seven new
  exact-device tests. The expanded Apple ReplaySSM and benchmark-contract
  regression set passes 52 tests. A narrow checkpoint-fold follow-up now gives
  one Metal lane to each `(batch, channel, state)` element, serially replays
  tokens without atomics, and writes resident `S0`; a second kernel clears all
  fixed-capacity rings in the same ordered command buffer. Native provenance,
  forced-boundary equivalence, repeated flush/cleanup, and a portable explicit
  fallback negative cover the
  lifecycle. The dedicated two-run Apple7 flush corpus records device and
  end-to-end timing separately at `1x128x64/T16` and `1x256x128/T16`: native
  device medians are 20.9--30.0 us and end-to-end medians are 298--318 us.
  The shared vectorized CPU fold remains faster end-to-end for these isolated
  flushes (44.6--146 us), so the native route is a residency/ordering closure,
  not a latency promotion. The paired serving two-run Apple7
  corpus compares `fused_block` with `resident_ring` at `1x128x64/T16` and
  `1x256x128/T16`, ten repetitions after three warmups. Fused block is the
  stable end-to-end winner; the smaller device-domain winner flips between
  runs and therefore earns no promotion, while the larger row stably retains
  fused block. NVIDIA and ROCm are not affected: their resident CUDA/HIP
  contexts and physical schedules remain independently proven.
- **2026-07-18 APPLE-RETUNE-1 paired-corpus foundation:**
  `benchmark_legacy_retune.py` now measures grouped GEMM, MoE SwiGLU,
  MPSGraph reduction, contiguous resident-KV reads, absorbed/explicit MLA, and
  ReplaySSM block/token-loop decode through one interleaved two-run schema.
  Every row shares a numerical oracle and records native/reference provenance,
  resource/API evidence, paired end-to-end medians, and a device interval only
  when it covers the complete route. The Apple7 corpus retains grouped fused
  GEMM and fused Replay decode, promotes single-dispatch MoE and absorbed MLA
  end-to-end on their exact small rows, and retains explicit MLA in the device
  domain. Reduction has end-to-end native evidence but no owned device interval;
  mapped KV and multi-dispatch peers remain explicitly ineligible for device
  selection. APPLE-RETUNE-1 stays active for wider shapes/dtypes, grouped
  SwiGLU/transport byte-bandwidth rows, and complete command-buffer intervals
  for the remaining composed routes.
- **2026-07-27 APPLE-RETUNE-1 ledger invalidated by a runtime-source edit:**
  `benchmarks/baselines/apple_strict_route_ledger.json` pins
  `context.runtime_fingerprint = sha256:74eb6e95…`, a whole-file hash of
  `apple_gpu_runtime.mm`. Adding the APPLE-PLACEMENT-ABI-1 status twins changed
  that hash, so `load_strict_route_ledger` now rejects every retained decision
  with `context_mismatch:runtime_fingerprint` and the exact-host admission test
  fails on a Metal host. **This is the guard working, not a regression** — the
  retained decisions were measured against a different runtime source, and the
  fingerprint cannot know that the new symbols touch no measured path. The
  ledger must be **re-measured**, not re-stamped: editing the fingerprint in
  place would assert that old measurements describe new code. (The test was
  already failing before this change for the same reason, so this is a renewal
  that was already owed.) The test is `hardware_apple_gpu`-gated, so CI is
  unaffected. **Design note for the renewal:** a whole-file hash over a
  ~24k-line runtime means any edit anywhere — including adding an unrelated
  symbol — invalidates every retained route decision. Consider fingerprinting
  the per-route kernel source instead, so evidence survives edits that provably
  cannot reach the measured path.
- **2026-07-21 APPLE-RETUNE-1 extended exact-host renewal:** the owned fresh
  dylib produced two committed-strength (`5 reps`, `3` interleaved trials) runs
  over both the original and 2x geometry rows: 48 route rows across grouped
  GEMM, MoE SwiGLU, reduction, resident-KV, MLA, and Replay decode. The fresh
  strict-v2 ledger admits 16 exact Apple7 f32 decisions and retains eight
  negative rows as explicit ineligible evidence: mapped resident-KV in both
  domains plus composed MoE and MPSGraph reduction in the device domain. The
  corpus does not claim f16/bf16 because the committed rows predate their
  owned same-storage ABI/oracle pair. Low-precision retuning remains separate
  until fresh committed-strength rows measure that pair; it must not convert
  inputs or borrow another route's proof.
- **2026-07-21 APPLE-RETUNE-1 transport/low-precision ratchet:** every renewed
  row now retains logical host-visible input/output bytes and its end-to-end
  logical bandwidth, explicitly labeled as distinct from device bandwidth. The
  grouped-SwiGLU small and 2x rows carry 102,432 and 409,632 logical bytes per
  call. The historic f16/bf16 rejection rows remain sealed evidence for the
  then-f32-only corpus. MoE now owns raw-storage f16/bf16 ABI symbols with an
  exact-device oracle and one complete command-buffer interval; it needs fresh
  committed-strength retune rows before selector admission. Grouped GEMM keeps
  its same-storage composed C-ABI route but has no fused low-precision package.
  Complete device intervals for composed SwiGLU, mapped resident-KV, and
  MPSGraph reduction remain absent: they need an owned complete-route ABI, not
  telemetry summation.
- **2026-07-21 APPLE-RETUNE-1 low-precision MoE admission:** two fresh exact
  Apple7 reports at five repetitions and three interleaved trials seal
  `apple7_lowp_moe_retune_two_run.json` and its strict-v2 sibling ledger. Raw
  f16/bf16 MoE storage at the base and 2x shapes is numerically valid, native,
  and has complete command-buffer intervals in every sample; all eight
  shape/dtype/timing-domain decisions retain `single_fused_lowp`. The remaining
  low-precision gap is grouped GEMM: its C ABI remains composed-per-expert, not
  an owned fused low-precision package.
- **2026-07-18 APPLE-ROUTE-1 strict-ingestion foundation:** production lookup
  no longer reads a literal exact-row table. The v2 ledger gate matches the live
  Apple family and physical-device model, OS, SDK, configured LLVM/compiler
  digest, runtime-source digest, expiry window, native provenance, correctness,
  and requested timing domain; admitted decisions expose their exact ledger-row
  citation. The fresh Apple7 retune ledger admits eight decisions in a clean
  host process. Older v1 GEMM/attention/backward/paged-KV/Replay ledgers lack
  this envelope and are rejected, so those operations conservatively retain
  their incumbents until fresh strict ledgers are recorded. APPLE-ROUTE-1 stays
  active until each completed family is migrated and package-subgraph selection
  is separated into its own strict ledger namespace. NVIDIA and ROCm are not
  applicable to this Apple-only corpus/selector change; their physical-device
  probes, retained ledgers, and production selectors are unchanged.
- **2026-07-21 APPLE-ROUTE-1 remeasurement rail:** every owning benchmark now
  captures the live exact-device context in its raw report, and
  `seal_strict_route_ledger.py` accepts only two independently produced reports
  to create a `runtime_route` v2 ledger. Sealing retains SHA-256 source-report
  digests and places rows without a selectable full-domain result in
  `ineligible_decisions`, outside selector-visible `decisions`. The renewed
  paired corpus plus GEMM/softmax, forward-attention, backward-attention, and
  epilogue owner lanes are migrated. Each historical schema-v1 file remains an
  inventory only; its sibling v2 ledger is the sole selector evidence.
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
- Attention backward now has an Apple-owned native proof and stable route
  ledger; its physical schedules remain independent of CUDA and ROCm.
- **2026-07-18 APPLE-ATTN-BWD-1 native-candidate foundation:** the Apple
  runtime now exposes a status-only f32 MHA backward ABI. Two MSL encoders on
  one labeled command buffer recompute the softmax and produce dQ, dK, and dV
  with f32 accumulation; each output element owns its reduction, so the route
  is deterministic and has zero workspace/atomic traffic. Exact-device ragged
  and causal oracle tests verify all three gradients and repeated launches are
  bit-identical. The same ABI now owns a zero-workspace atomic dK/dV candidate
  using relaxed compare/exchange f32 accumulation and a deterministic two-way
  split candidate using exactly one additional f32 dK+dV partial plus a
  fixed-order reduction. The policy rejects deterministic atomic requests and
  insufficient split workspace before dispatch. Exact-device tests cover all
  three routes on ragged, batched, causal, and noncausal shapes against the same
  oracle; serial and split repeats are bit-identical, while atomic repeats are
  validated numerically under its explicitly nondeterministic contract.
  `benchmark_attention_backward.py` produces paired route rows with warmup
  separation, per-trial GPU/end-to-end medians, resources, workspace policy,
  and per-gradient error. Two Apple7 smoke collections each have twelve native,
  numerically valid rows and complete device-time coverage. Atomic wins every
  end-to-end row on this small foundation matrix; device-interval winners vary
  by row and run, so no timing domain is collapsed into another and `auto`
  remains on serial recompute. This is not yet a selector corpus: GQA/MQA,
  bias, softcap/window, f16/bf16 storage, workspace caps, wider and long-context
  shapes, and a committed stable selection corpus remain active. NVIDIA
  and ROCm are not applicable: the shared derivative semantics are unchanged
  and no CUDA/ROCm schedule is transferred.
- **2026-07-18 APPLE-ATTN-BWD-1 closure:** all three candidates now use
  query-streaming softmax/dP work rather than recomputing one softmax per output
  element. Atomic work owns one query row and confines contention to final
  dK/dV updates; serial gives one deterministic owner each KV head; split gives
  two deterministic owners one exact additional f32 dK+dV footprint and then
  reduces in fixed order. The status ABI shares forward's flattened-head
  MHA/GQA/MQA mapping, right-aligned causal and sliding-window masks, additive
  bias, and correctly differentiated logit softcap. Legacy rectangular causal
  callers retain their original zero-offset triangle. Native f16 and bf16
  inputs are read directly from two-byte Metal storage; dQ/dK/dV accumulate and
  return f32. Exact-device tests cover every route and dtype, batched/ragged
  MHA, GQA, MQA, bias, causal/noncausal windows, softcap, invalid-route
  rejection, deterministic repeats, and workspace limits.
  Two independent Apple7 reports contain 18 native, numerically valid, fully
  device-timed rows each. The committed
  `benchmarks/baselines/apple7_attention_backward_route_ledger.json` contains
  twelve timing-domain decisions. End-to-end selection promotes split-reduce
  for four rows, including causal `Sk=1025`, and atomic for two rows; paired
  median wins range from 27.8% to 67.3%, with 100% trial wins in both reports.
  Every device-interval row retains serial recompute. The legacy ledger records
  exact-device/shape/dtype/domain decisions, but strict v2 production ingestion
  now retains serial until those rows are re-recorded with current context;
  determinism and split-workspace policy remain enforced independently.
  **APPLE-ATTN-BWD-1 is closed.** NVIDIA and
  ROCm are not applicable to the Apple ABI, storage readers, schedules, or
  selector rows; shared derivative semantics remain unchanged.
- FP8/FP4/MX execution remains gated by the macOS 27 SDK/runtime surface. The
  compiler-side scale-layout and multi-plane contracts already exist; do not
  claim hardware execution until the public Metal tensor path runs natively.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus now
  carries additive compiler/resource, cold/warm, cache, and two-run stability
  evidence. Existing v1/v2 rows migrate without changing Apple selection, and
  no CUDA schedule or selector is transferred to Metal. Apple follow-up is to
  populate the same logical evidence fields from Metal-native counters during
  its own performance work; current Apple plan state is otherwise unaffected.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: not applicable to Apple
  execution. The fixes are confined to Ubuntu apt.llvm.org discovery,
  CUDA/NVVM lowering, and Linux NVIDIA/ROCm lit shell selection. No Apple IR,
  ABI, Metal schedule, numerical policy, or exact-device evidence changed.

## Completion definition

This plan reaches `closed` only when all of the following are true:

1. Host-free and compiler-artifact tests remain portable lanes. Apple promotion
   is owned by one local Metal 4 exact-device gate with two fresh-process
   correctness runs, an isolated paired-performance corpus, and a sealed packet
   pushed to the coordinating PR. Registered GitHub self-hosted runners are not
   used. Metal 3 is a non-blocking compatibility lane.
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
is using the GPU. Metal 4 promotion requires a named Metal 4 host. Metal 3
coverage is compatibility-only and cannot promote a Metal 4 route; never
generalize a winner across Apple GPU families without a matching record.

### Use the dedicated LLVM/MLIR 23 prefix

The generic Homebrew `llvm` symlink is not this lane: it may resolve to a
different keg (currently LLVM 23) or be absent. Apple validation and
`build-apple` use the dedicated, pinned upstream `release/23.x` build at
`/opt/homebrew/llvm-23.1.0-rc1`; it must be built with
`LLVM_ENABLE_RTTI=ON`, or Tessera's pass and dialect typeinfo cannot link.
Before configuring or testing, set and validate this exact prefix:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
test -x "$TESSERA_LLVM23_PREFIX/bin/llvm-config"
test -d "$TESSERA_LLVM23_PREFIX/lib/cmake/mlir"
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

"$TESSERA_LLVM23_PREFIX/bin/llvm-config" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-opt" --version
"$TESSERA_LLVM23_PREFIX/bin/mlir-tblgen" --version
```

All three version commands must begin with `23.`. If either path check fails,
stop rather than falling back to `brew --prefix llvm` or AppleClang's system
libraries. To recreate the dedicated toolchain, install the Xcode Command Line
Tools first, then build it:

```bash
xcode-select --install                    # omit if already installed
brew update
brew install cmake ninja lit
git clone --depth 1 --branch release/23.x https://github.com/llvm/llvm-project.git /private/tmp/llvm-project-23
cmake -S /private/tmp/llvm-project-23/llvm -B /private/tmp/llvm-23-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/homebrew/llvm-23.1.0-rc1 \
  -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
  -DLLVM_TARGETS_TO_BUILD='AArch64;AMDGPU;NVPTX;X86' \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON
cmake --build /private/tmp/llvm-23-build --target install --parallel 8

export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
export PATH="$TESSERA_LLVM23_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$TESSERA_LLVM23_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
"$(brew --prefix lit)/bin/lit" --version
```

Do not use AppleClang's system LLVM libraries or mix the stable LLVM 23 keg
with this LLVM/MLIR 23 prefix. Record the upstream commit plus
`LLVM_ENABLE_RTTI=ON` in the build evidence.

For compiler artifacts, build the Apple backend and portable MLIR tools:

```bash
export TESSERA_LLVM23_PREFIX=/opt/homebrew/llvm-23.1.0-rc1
cmake -S . -B build-apple -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$TESSERA_LLVM23_PREFIX/bin/clang++" \
  -DLLVM_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/llvm" \
  -DMLIR_DIR="$TESSERA_LLVM23_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$(brew --prefix lit)/bin/lit" \
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

The 2026-07-16 shared compiler migration raises the project floor to matched
LLVM/MLIR 23 and removes the obsolete Apple dialect property switch. The
portable Apple sources are assessed by the shared-source migration, but the
current WSL host cannot build or execute the Darwin/Metal runtime; Apple
LLVM/MLIR 23 build and exact-device parity are **follow-up required** on the
named Apple hosts.

## Ordered work

This is a live queue, not a historical checklist. `closed` means the item's
stated gate is met; `landing` means the principal implementation and evidence
landed but a deliberately narrower follow-up remains; `active` is the next
implementation/proof work; `blocked` names an external prerequisite.

| Order | ID | Status | Current state and next action |
|---:|---|---|---|
| 1 | APPLE-TEST-1 | **closed** | The centralized hardware boundary collects 976 of 15,374 unit nodes, the structural scan finds zero inline Apple capability gates, and portable marker/provenance ratchets reject classification drift. |
| 2 | APPLE-CI-2 | **closed** | The host-free compiler ownership gate is executable and green for the declared Apple capability set, and now validates the exact LLVM/MLIR runner-utils path for every CMake cache type. |
| 3 | APPLE-TEST-2 | **closed** | Fresh-runtime correctness (**850/850**), fallback-injection negatives, ordering/stress, and the serial measured lane are complete. |
| 4 | APPLE-REG-1 | **closed** | ABI/target-map/exact-device/Tile drift gates are registered and passing. |
| 5 | APPLE-TILE-1 | **closed** | The selected f16/bf16 simdgroup fragment and its two-run corpus meet the completion gate. MPS retaining every measured end-to-end row is the valid production decision. |
| 6 | APPLE-GEMM-1 | **closed** | The Apple7 paired ledger has stable decisions for every measured row: three promotions and nineteen incumbent retentions. New devices/routes require a new corpus. |
| 7 | APPLE-EPILOGUE-1 | **closed** | Supported f32/f16/bf16 fusions have native-oracle/resource proof and stable end-to-end selection; MPSGraph now has an explicitly labeled Metal 4 whole-graph envelope, pending a fresh two-run device-domain corpus. |
| 8 | APPLE-ATTN-FWD-1 | **closed** | Native forward variants, resident/cooperative candidates, full stated corpus, two-run route ledger, and timing-domain selection are complete. Do not reopen it for backward work. |
| 9 | APPLE-ATTN-BWD-1 | **closed** | Native f32/f16/bf16 MHA/GQA/MQA serial, atomic, and split-reduce routes share one oracle and explicit workspace/determinism policy. The stable two-run Apple7 ledger selects end-to-end routes per exact row and retains serial for every device-domain row. |
| 10 | APPLE-PAGED-KV-1 | **closed** | Direct resident page-table MLA attention and the staged peer share a non-identity oracle, causal/window boundary proof, transactional exhaustion/leak telemetry, and a paired two-domain Apple7 corpus. The legacy corpus records direct wins; strict production ingestion retains staged until those rows are re-recorded with the v2 context envelope. |
| 11 | APPLE-REPLAY-1 | **closed** | Resident inputs, ordered asynchronous ring submissions, native deterministic checkpoint folding plus same-command-buffer ring clearing, forced flush/rollback/partial-rejection ordering, backpressure/cleanup stress, and paired selector evidence are complete. Unstable device-domain evidence retains the fused-block incumbent. |
| 12 | APPLE-RETUNE-1 | **active** | Fresh Apple7 f32 corpus has 16 selector-admissible decisions and eight explicit partial-domain negatives; the separate two-run f16/bf16 MoE corpus adds eight `single_fused_lowp` retain decisions with complete command-buffer proof. Grouped-SwiGLU logical byte/bandwidth accounting is retained. The remaining low-precision route gap is a fused grouped-GEMM package; complete-route ABIs remain required for composed/mapped device intervals. |
| 13 | APPLE-ROUTE-1 | **active** | Strict v2 sealing now binds producer context and source-report digests and retains only explicit negative rows outside selector decisions. The paired corpus and every legacy runtime-route owner (GEMM/softmax, forward/backward attention, epilogue) have fresh sibling v2 evidence; schema-v1 files are inventories only. Package subgraphs remain a separate namespace. |
| 14 | APPLE-DTYPE-1 | **blocked — SDK** | FP8/FP4/MX native execution awaits the public macOS 27 Metal tensor path. Keep older-host int4/int8/f16/bf16 regression coverage. |
| 15 | APPLE-CI-1 | **closed** | The local Metal 4 release gate serializes the physical Mac without registering a GitHub runner, builds fresh LLVM/MLIR 23 compiler/JIT/runtime artifacts, records power/thermal/GPU-contention availability, rejects incomplete or skipped evidence, runs correctness twice, and seals paired device/end-to-end evidence. The retained `docs/audit/evidence/apple/metal4/20260718-b1ee875/` packet proves two clean 11-test Apple7 runs under Xcode 26.6, two 8-row route reports with four Metal 4 rows each, and an 8-decision two-domain ledger against commit `b1ee87591ec701dd06a156cad8449f6498ae0891`. Portable CI validates its hashes and contents. Metal 3 remains non-blocking compatibility coverage. |
| 16 | APPLE-E2E-1 | **closed / bounded Level C** | Static, exact-device-oracle-backed GPU ABI families are closed: rank-2 f32 softmax/transpose; f32/f16/bf16 rank-3 batched-GEMM; strict and side-tensor PPO; EBM energy/Langevin/refinement/partition; cl30 Clifford geometric product; and static/batched Cholesky, Cholesky-solve, and triangular-solve. Every family has package, owned-fresh-dylib execute/compare, and repeated-launch cleanup proof on the exact device. Composite/package-subgraph, dynamic, stateful, unsupported, and multi-result GPU contracts, plus fleet/second-device proof, are retained follow-on work in APPLE-NATIVE-E2E-2. Metal-owned schedules, placement policy, and selectors are unchanged. NVIDIA and ROCm are not applicable: this changed no shared IR, ABI, schedule, or evidence claim. |
| 17 | APPLE-CPU-E2E-1 | **closed / bounded Level C** | Static f32 rank-2 matmul/gemm and rank-3 BMM; single-result Cholesky, triangular-solve, and Cholesky-solve; and tuple-output LU/QR/SVD Accelerate/LAPACK descriptors have exact-host execute/compare and repeated-launch cleanup proof through the owned rebuilt dylib. Dynamic shapes, other dtypes, and non-linalg contracts remain retained/reference and belong to APPLE-NATIVE-E2E-2. |
| 18 | APPLE-NATIVE-E2E-2 | **landing / fleet packets sealed; second-device proof hardware-gated** | The bounded local descriptor program is complete on the exact Apple7 Metal-4 device. Existing CPU f16/bf16 matmul/gemm and descriptor-state-registered, exact-host-replayed f32 row-softmax are complete. GPU static/dynamic f32/f16/bf16 GELU descriptors carry explicit fp32-accumulation provenance, two-byte low-precision bindings, storage-rounding oracles, and rank/dtype/result-shape/scalar rejection ratchets. Ordered reduced SVD and ReplaySSM lifecycle packages are joined by dynamic rank-1 i32 popcount (`Elements`), rank-2 f32 last-axis count-nonzero (`Outer/AxisExtent`), and rank-2 f32 row-softmax (`Rows/Columns`). Ordered top-k now uses a dedicated status-returning Metal ABI with numeric-descending, NaN-last, lower-index-tie semantics, ordered `(values,indices)` output bindings, `Rows/Columns/K` verification, and exact-device execute/compare/replay/rejection proof. The Metal-4 composite package remains sealed by tree digest, reflected positional externals, a private intermediates heap, and replay-safe cache identity. Its paired strict-v2 `package_subgraph` evidence comprises two independent 50-repetition by 5-trial reports: package promoted at `64x64x64`, live retained at `256x256x256`, and device-domain rows explicitly ineligible because the lane exposes only comparable complete-call timing. The local CPU ABI audit still finds no further owned static non-linalg candidate; speculative wrappers remain forbidden. **Both Apple fleet packets are now recorded and sealed (2026-07-27).** `benchmarks/e2e_spine/record_apple_packet.py --lane {apple_gpu,apple_cpu}` produces the two independent identities the registry keys on — `apple_gpu`/`apple7` and `apple_cpu`/`apple_m1_max` — into `docs/audit/evidence/e2e_spine/apple_gpu/apple7/` and `.../apple_cpu/apple_m1_max/`. Evidence never transfers between them, so each lane seals its own `report`/`resources`/`manifest` triple. Both validate independently with **max absolute error 0.0** against the shared oracle: `apple_gpu` proves `matmul` + `softmax` (Level A/B/C, 4 benchmark rows), `apple_cpu` proves `matmul` (2 rows). `matmul` on the GPU is proven as a batch-1 BMM because the Apple GPU GEMM contract is batched-only; the route name `apple_gpu_bmm_f32_batch1` keeps that visible rather than implying a 2-D kernel. `apple_cpu` scope is `matmul` alone because the registration declares only `("matmul","linalg")` — the lane executes softmax, but claiming it would be an undeclared family. **Both lanes report `kernel_wall`, not `device_event`**, because the MPSGraph matmul route has no device timer while the MSL softmax route does (row 29 — the first diagnosis of this was wrong and is corrected there). **GPU placement is proven independently of the oracle**: the void `..._f32` ABIs fall through to a numerically-identical CPU reference, so the recorder requires a positive `..._f32_status` result at both the fixture and timing shapes before sealing (row 30). Host identity is pinned per lane — CPU brand for `apple_cpu`, live Metal GPU family for `apple_gpu`, so an M3/M4 host cannot seal an `apple7` packet. `linalg`/`ppo`/`ebm`/`clifford` (GPU) and `linalg` (CPU) remain `packet_pending` — "Family absent from active packet" — and need corpus fixtures before they can be claimed. Only *second-device* proof (a non-Apple7 family) remains genuinely hardware-gated.  Any additional retained/reference family requires a separately owned future ABI item with a shape/dtype contract and exact-device oracle. NVIDIA and ROCm are not applicable: these Apple-private descriptor metadata and proof ratchets change no shared dtype, Graph-IR spelling, sibling ABI, or schedule. |

### APPLE-PIPE-1 landing evidence (2026-07-26)

`src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/
ThreadgroupPipelineToApple.cpp` is Apple's architecture-owned consumer of the
shared Tile physical-allocation and staged-pipeline SSA contract — the Metal
sibling of `ROCMWaveLdsPipeline.cpp`. It follows the AMD precedent of claiming
the shared ops in place with architecture-owned physical decisions rather than
translating them into a private handle type.

What Apple now decides, and where the decision is enforced:

- **Placement.** Each `smem` `tile.alloc` gets `tessera_apple.address_space`,
  `_threadgroup_offset` (16-byte aligned, packed in declaration order),
  `_threadgroup_bytes`, and `_threadgroup_capacity_bytes`; the enclosing
  function carries `tessera_apple.threadgroup_arena_bytes`. A function with no
  Tile allocation carries no arena attribute — the pass states a demand it
  measured, never a zero it assumed.
- **Buffering mode.** A depth-2 ring is named `ping_pong` and depth-1 `single`,
  so the emitter reads the mode instead of re-deriving it from `depth`.
- **Capability boundary.** Nine registered diagnostics
  (`APPLE_THREADGROUP_*`, `APPLE_STAGE_*`, `APPLE_TILE_UNSUPPORTED_VOCABULARY`)
  reject `tmem`/`gmem` placement, over-capacity arenas, rings deeper than the
  Metal ping-pong pair, name-based `#tile.buffer_ref` identity, and the
  NVIDIA-only TMA / mbarrier / TMEM / TCGen05 vocabulary. Per Decision #21
  each names the operation and the target; none silently no-ops. Silently
  narrowing a depth-4 ring to 2 was explicitly rejected as a design option —
  it would change the program's synchronization structure behind the author.

Evidence, all host-free on this Mac against the LLVM/MLIR 23 `build-apple`
compiler:

- `tests/tessera-ir/phase8/apple_threadgroup_pipeline.mlir` — SSA identity end
  to end, checked twice: driven directly *and* through the real
  `tessera-lower-to-apple_gpu` pipeline, so placement is decided once and
  survives to Target IR.
- `tests/tessera-ir/phase8/apple_threadgroup_pipeline_invalid.mlir` — eight
  split-input rejection cases. The harness was sanity-checked by mutating one
  expected code and confirming the fixture then fails.
- `tests/unit/test_apple_threadgroup_pipeline.py` (5 tests,
  `compiler_tool` + `compiler_apple`) — the cross-owner drift gate binding the
  C++ placer to `msl_gemm_emit.materialize_apple_simdgroup_tile_msl`: same
  capacity as the `AppleGPUTargetProfile`, same ping-pong rule as the emitted
  `As[2]`/`Bs[2]` staging, and an arena that reproduces the artifact's
  `total_threadgroup_bytes` at the expected offsets. Both owners reject the
  same over-capacity tile, each with its own code.
- Lanes: phase8 lit **72 passed / 2 unsupported / 0 failed**; the Apple
  host-free compiler ownership gate selects the new tests and passes
  (**6 passed, 46 foreign-compiler skips**); diagnostic-registry and
  pass-metadata drift gates green; `mypy` clean.

**PR #467 review fixes (2026-07-26).** Three real defects were found in review
and are now regression-locked in `tests/unit/test_apple_threadgroup_pipeline.py`:

1. **Loop-carried state was reported unrooted.** A ring threaded by `scf.for`
   reaches its advance as a *region iter_arg*, not the `pipeline_init` result —
   which is exactly how the canonical GEMM and streaming attention emit it. The
   membership test rejected that with `APPLE_STAGE_UNROOTED_ADVANCE`, so the
   pass refused the very shared contract it exists to consume. Block arguments
   are now resolved back to the matching loop init operand.
2. **The MMA gate could be bypassed.** It read only `numeric_policy.storage`,
   but the op verifier requires only `mma`, so an int4/FP8 descriptor with no
   policy — or with a laundering fp16 one — passed. The descriptor's declared
   A/B/accumulator types are now authoritative, and an unreadable descriptor is
   treated as "cannot prove a route exists" rather than as permission.
3. **Advances contradicted their own ring.** Every advance was stamped
   `ping_pong`, so a depth-1 pipeline handed the emitter two different physical
   schedules. Advances now carry the rooted initializer's mode.

NVIDIA and ROCm are not applicable to the Apple pass, attributes, or
diagnostics. The one shared-ground change is the `LseSaveOp` `Pure` correction
recorded below.

### APPLE-TILE-2 landing evidence (2026-07-26)

`CanonicalGemmToAppleGPU.cpp` recognizes the shared canonical reduction and
re-forms it as one Apple dispatch. The architectural choice, stated so it is
reviewable rather than implicit: **the canonical loop is a semantic contract,
not a schedule Metal must reproduce statement by statement.** It says "reduce
over K in FP32, zero-padding the ragged tail"; the Apple steel GEMM already
implements exactly that reduction with its own cooperative staging and
edge-masked stores. Emitting a literal three-loop MSL nest would be slower and
no more correct. NVIDIA and ROCm consume the same loop differently — per-K-step
Tensor Core / MFMA fragment issue — which is the intended per-target freedom.

- **Recognition is guarded, not greedy.** The claim requires the
  `tessera.canonical_k_step` marker, the three-deep `scf.for` structure, a K
  loop carrying `!tile.pipeline_state`, and two loop-invariant sliced operands.
  A user-written loop containing a matmul is left untouched.
- **Four registered diagnostics** (`APPLE_CANONICAL_GEMM_{UNRECOGNIZED,
  SHAPE,DTYPE,ACCUM}_*`) own the envelope. The f32 rejection is the one that
  keeps the incumbent honest: f32 GEMM stays on Accelerate/MPS instead of being
  quietly rerouted to a `simdgroup_matrix` path that has no f32 operand form.
- **Logical tile vs physical block.** The loop's `tessera.tile_*` values are
  logical steps the shared tiler clamps to the extent (a `13x16x11` GEMM yields
  steps of 13/11/16). An Apple threadgroup block must be a multiple of the
  8x8x8 fragment, so `msl_gemm_emit.apple_block_for_canonical_tile` rounds *up*
  — which is precisely what the contract's `ragged_zero_pad` guarantee licenses.
  This distinction is now explicit instead of an accidental shape agreement.

Evidence:

- `tests/tessera-ir/phase8/apple_canonical_gemm.mlir` — a plain Graph-IR matmul
  driven through the **shared** `--tessera-tiling` and then the Apple pass in
  one run, so the fixture consumes the real canonical form rather than a
  hand-written imitation. Checks the nest is consumed (`CHECK-NOT: scf.for`),
  the dispatch is singular, and fp16/bf16 select storage-matched symbols.
- `tests/unit/test_apple_canonical_gemm.py` — 7 tests. Four host-free
  (descriptor contents, bf16 symbol, f32 rejection, no-misclaim), and three
  `hardware_apple_gpu` **execute-and-compare rows that ran on this Apple7
  Metal device**: `16x16x16`, `32x16x32`, and the ragged `13x16x11`, each
  driven by the compiler-produced descriptor, each asserting `native is True`
  with a positive device time and matching the fp32 NumPy oracle.
- Lanes: phase8 lit **74 passed / 2 unsupported / 0 failed**; Apple unit sweep
  **3068 passed / 10 skipped / 1 failed**, the single failure being the
  pre-existing `test_strict_retune_ledger_admits_on_its_exact_live_apple_host`
  (see the note below); registry/metadata drift gates and `mypy` green.

**Pre-existing failure worth its own fix (not caused by this work).**
`test_strict_retune_ledger_admits_on_its_exact_live_apple_host` fails with
`context_mismatch:runtime_fingerprint`, and reproduces on a clean `HEAD` with
all of this work stashed. `apple_route_selector._runtime_source_fingerprint`
hashes `apple_gpu_runtime.mm`, whose current content no longer matches the
digest the committed strict-v2 retune ledger was sealed against. The practical
consequence is that the strict ledger currently admits **no** decisions on this
host, so APPLE-RETUNE-1's selector evidence is inert until the corpus is
re-recorded and re-sealed against the current runtime source.

### APPLE-ATTN-STREAM-1: LSE checkpoint migration (updated 2026-07-27)

The 2026-07-26 investigation below is retained as historical diagnosis and is
**superseded**. Cross-backend sync
`LSE-CHECKPOINT-CONTRACT-2026-07-27` removed destination-less emission and
replaced the declarations with explicit memref source/destination, SSA row
offset, identity, global-memory space, lifetime scope, cache policy, and
`MemWrite`/`MemRead` effects. The Apple pass no longer erases `lse.save`; any
live LSE remains a real unsupported ABI request and is rejected. Inference-only
forward sees no save. ROCm measured and selected its own gfx1151 128+ policy;
that threshold and AMD schedule do not transfer. Apple follow-up is required
only if a Metal training package elects to persist LSE.

#### Historical diagnosis (superseded)

This item first landed **blocked**, and the investigation is worth keeping
because the blocker turned out to be a bug in the shared contract rather than a
capability gap.

**The apparent blocker.** The shared lowering always terminates the recurrence
with `tessera_attn.lse_accumulate` -> `tessera_attn.lse.save`. `LseSaveOp`
carried no `Pure` trait, so MLIR had to treat it as side-effecting, while
Apple's fused ABI (`tessera_apple_gpu_flash_attn_*`) returns the attention
output only. Re-forming would either leave the whole recurrence alive to
recompute the LSE — attention computed twice — or silently drop a checkpoint
backward appeared to depend on. The pass refused.

**What the code actually shows.** There is no checkpoint to protect:

- the emission site (`TileIRLoweringPass.cpp:374`) **discards the result**;
- the result type is scalar `f32`, not the per-row `[tile_q]` LSE it names;
- `LseLoadOp` takes **no operands**, so no SSA edge, symbol, or handle links a
  load to a save — a backward lowering could not express *which* save its load
  reads, even if one wanted to;
- the only `lse.load` in the tree is a v1.3 example fixture.

This is the same name-free global-state modeling the 2026-07-26 wave already
ruled against twice: `#tile.buffer_ref` became `!tile.buffer`, annotation-only
`#tile.pipeline_state` became threaded SSA, and `TilePipelineLegality` now
*rejects* the annotation-only form. The LSE pair is the next unmigrated
instance, not an unlucky edge case.

**No backend consumes it, and all three that have an attention backward chose
recompute** — for the same reason, since a saved LSE reintroduces the workspace
their determinism contracts exist to eliminate:

| Backend | Attention backward | LSE source |
|---|---|---|
| ROCm gfx1151 | `GenerateWMMAFlashAttnBwdKernel.cpp` | recomputes `L[q] = logsumexp_k(scale*QK^T)` in a `_pre` pass; header states the backward "needs nothing saved from the forward" |
| NVIDIA sm_120 | `sm120_attention_backward_kernel.mlir` | `workspace_bytes = 0`, `workspace_owner = "output_element"` |
| Apple Apple7 | `flash_attn_bwd_*` | `bwd_query_stats` recomputes m/l per query; ABI takes no LSE buffer |
| x86 AVX-512 | none (forward only) | n/a |

**Resolution — and a rejected first attempt.** Marking `LseSaveOp` `Pure` was
tried first and **backed out**. Testing showed it changes emitted IR on every
backend (`tests/tessera-ir/phase3/flash_attn_full.mlir` asserts the op's
presence), and it would leave a trap for whoever implements the real FA-2
checkpoint, since a store must be non-`Pure` with `MemWrite`. An earlier claim
here that the trait "changes nothing emitted on any target" was wrong.

What landed instead touches no shared declaration: the Apple pass erases only a
`lse.save` whose own result is unused, as part of re-forming the recurrence it
is already rewriting. Both the Apple fixture and `flash_attn_full.mlir` pass
with the shared op unchanged. A `retain_lse` flag was also considered and
rejected — it would gate an op that cannot perform its stated function.

The vocabulary is deliberately **kept**, and the save-versus-recompute question
is now owned by the backends with the memory systems to settle it:
[`NVIDIA-LSE-1`](../nvidia/todo.md) and [`ROCM-LSE-1`](../rocm/todo.md), with
the contract, the FA-2 design, and the preferred source-level fix documented in
[`../../compiler/LSE_CHECKPOINT_CONTRACT.md`](../../compiler/LSE_CHECKPOINT_CONTRACT.md).

The Apple consumer erases only a `lse.save` whose own result is unused; anything
genuinely reading the LSE still refuses with
`APPLE_STREAMING_ATTN_LSE_UNSUPPORTED`. It also erases the now-dead staging
(Q copy, ring init/advance), because a leftover depth-3 `tile.pipeline_init`
would otherwise fail APPLE-PIPE-1 for a schedule the program no longer has.

NVIDIA and ROCm review note: nothing in this slice changes a shared op,
verifier, or emitted lowering. The one shared-ground item is the *documented*
source-level fix — stop emitting a destination-less save — which is filed in
both queues rather than landed from here.

### Attention-backward and stateful-transport rows (opened 2026-07-27)

A second wave of shared contracts landed while rows 19-23 were in flight, all
clustered on attention backward and stateful transport. They arrived with
Apple follow-ups recorded in the sync notes above but no owning rows, which is
the same gap rows 19-23 were opened to close for the previous wave. Rows 24-28
own them.

Two of these are close to work Apple already has. APPLE-ATTN-BWD-1 is closed
with proven serial / atomic / split-reduce Metal routes, and the ReplaySSM
lifecycle is closed with session-private ring and ordering semantics — so rows
25 and 28 are mostly *contract adoption*: deciding whether the shared phase
loops and generalized resident schema describe what Apple already runs, and
saying so explicitly either way.

### Shared-Tile-contract consumer rows (opened 2026-07-26)

The 2026-07-25/26 core wave (`CORE-GEMM-KLOOP`, `CORE-STREAMING-ATTN`,
`ROCM-SSA-LDS-PIPELINE`, `PACKED-LEGALIZE-CAPABILITY`) landed shared Tile
contracts whose Apple follow-ups were recorded in the sync notes above but had
no owning row. These rows own them. The audited starting position, verified in
source rather than prose:

- The Apple GPU pipeline runs `createTilingPass(valueMode=true)`, which
  deliberately does **not** tile to `scf.for` (`TilingPass.cpp:952`), so the
  canonical M/N/K K-loop has no Apple consumer.
- `TileToApple` matches whole-tensor `tile.matmul` / `tile.gemm` /
  `tile.batched_gemm` by name (`TileToApple.cpp:677`).
- `FlashAttnToAppleGPU` rewrites `tessera.flash_attn` **directly from Graph
  IR** to a monolithic runtime ABI call (`FlashAttnToAppleGPU.cpp:77`); Apple
  attention never crosses the Tile layer.
- Nothing under the Apple backend references `!tile.buffer`,
  `!tile.async_token`, or `!tile.pipeline_state`; the only consumers are
  `ROCMWaveLdsPipeline.cpp` and `WarpSpecializationPass.cpp`.

Apple is therefore the only actively developed backend whose GEMM *and*
attention paths bypass the shared Tile contracts. That is defensible on
measured grounds — the committed ledgers retain MPS/Accelerate for every
measured row — but it must be a *declared* architecture decision with a
capability-rejection or consumer proof, not undeclared divergence.

| Order | ID | Status | Current state and next action |
|---:|---|---|---|
| 19 | APPLE-PIPE-1 | **landing** | The `tessera-apple-threadgroup-pipeline` pass is a real consumer of the shared SSA vocabulary and runs first in `tessera-lower-to-apple_gpu`: `!tile.buffer` allocations are placed 16-byte-aligned into one capacity-bounded per-function threadgroup arena, and `!tile.pipeline_state` rings are claimed as `ping_pong` / `single` Metal staging. Nine registered diagnostics own the capability boundary. Evidence below. **Narrower follow-up:** the emitted steel MSL still computes its own staged bytes — the two owners are proven *equal*, not yet *sourced from one place* — and this rung is host-free by design, so it carries no exact-device execution proof. |
| 20 | APPLE-TILE-2 | **landing** | `tessera-apple-canonical-gemm` recognizes the shared three-deep M/N/K nest and re-forms it as one `simdgroup_matrix` dispatch carrying the loop's tile decision, `accumulate = "fp32"`, and the `ragged_zero_pad` guarantee. Exact-device execute-and-compare passes on Apple7 Metal for aligned and ragged rows, driven by the compiler-produced descriptor. The **incumbent rule is recorded in the pass itself**: recognition is not promotion. **Narrower follow-up:** no strict-v2 paired route-ledger row yet, so no timing-domain comparison exists and value-mode Accelerate/MPS remains the production route by default rather than by measurement. |
| 21 | APPLE-ATTN-STREAM-1 | **landing** | `tessera-apple-streaming-attention` recognizes the shared KV-block recurrence and re-forms it as one Apple flash-attention dispatch, carrying `causal` / `logical_sk` / `window_left/right` / `kv_block` **read off `tessera_attn.boundary_mask`** instead of re-derived — the ownership fix this row exists for. It runs first in `tessera-lower-to-apple_gpu`, ahead of APPLE-PIPE-1, because the shared depth-3 KV ring must be re-formed before the threadgroup pass judges a schedule the program is about to stop having. Unblocked without changing the shared `LseSaveOp` declaration (see below). **Narrower follow-up:** the descriptor targets the same ABI family as the incumbent, so numerical parity is proven structurally plus an on-device oracle check — not yet a full APPLE-ATTN-FWD-1 corpus re-run, and no selector changed. |
| 22 | APPLE-DTYPE-1-REJECT | **closed** | The macOS-27 SDK gate is enforced, not incidental. `tests/tessera-ir/phase8/apple_lowprecision_capability_gate.mlir` runs the same module through `--tessera-storage-legalize` twice: the `apple_gpu` target stamps no `tessera.storage_packed`/`_container` on either a block-scaled NVFP4 decode or a packed int4 contraction, while the `nvidia_sm120` contrast run stamps both — so the negative cannot pass merely because the pass did nothing. A block-scaled or otherwise unrouted cooperative-matrix descriptor is separately rejected with `APPLE_MMA_STORAGE_UNSUPPORTED`, and `tests/unit/test_apple_threadgroup_pipeline.py` binds that gate to `select_apple_simdgroup_fragment`: fp16/bf16 accepted by both owners, nvfp4/fp4/fp6/fp8/int4 refused by both. APPLE-DTYPE-1 itself stays **blocked — SDK**; this row proves the block, it does not lift it. |
| 23 | APPLE-COUNTER-1 | **landing** | `compiler/apple_counter_evidence.py` maps Metal telemetry onto the shared autotune-evidence fields with an explicit four-state reason on every field: `measured`, `not_measured` (device can, this run did not), `unsupported_by_device` (this GPU family cannot), `no_public_api` (Metal exposes no query — register count, scratch bytes, spill count, achieved occupancy). Supplying a value the capability bits do not support raises rather than silently downgrading, so a corpus cannot claim evidence the device cannot produce. Bit positions are drift-gated against the runtime's own documented matrix. **Narrower follow-up:** the benchmark writers do not yet emit these fields into a committed corpus, so this is the vocabulary and its guards, not a recorded two-run corpus. |
| 24 | APPLE-ATTN-STREAM-2 | **active** | Rank-4 batch/head streaming attention. `CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` added shared rank-4 distribution and a direct ROCm consumer; APPLE-ATTN-STREAM-1 re-forms rank-2 only and its `APPLE_STREAMING_ATTN_SHAPE_UNSUPPORTED` diagnostic refuses anything else. Extend the consumer to the rank-4 form, or record rank-4 as an explicit Apple non-goal with the reason. Gate: the same parity bar as rank-2 — boundary semantics read off the shared ops, plus an exact-device oracle row. |
| 25 | APPLE-ATTN-BWD-2 | **active** | Consume the shared tensor-valued attention **backward** phase loops. `ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` made gfx1151 the first direct physical consumer; Apple must validate the same dQ / split-dK/dV / fixed-reduction contract and map it to a Metal-owned package. APPLE-ATTN-BWD-1 already owns proven serial / atomic / split-reduce Metal routes, so this is contract adoption, not new kernels — the question is whether the shared phase loops describe the schedules Apple already runs. The AMD WMMA schedule, five-entry HSACO, HIP workspace, and host-wall timing do not transfer. |
| 26 | APPLE-ATTN-BWD-3 | **active** | `CORE-ATTENTION-BACKWARD-CONTRACT-2026-07-26` adds verified shared backward contracts; confirm Apple's backward satisfies them or record the divergence. The shared LSE checkpoint contract is now real and conditional; Apple retains recompute until an exact Metal package and benchmark justify a saved checkpoint. |
| 27 | APPLE-ATTN-MODIFIERS-1 | **active** | `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26` lands shared tensor-valued attention loop modifiers. Apple owns validating that its causal / sliding-window / softcap / bias / GQA-MQA envelope still expresses every admitted modifier after the shared change, and rejecting the rest by name rather than silently narrowing. |
| 28 | APPLE-STATEFUL-TRANSPORT-1 | **active** | `SSA-STATEFUL-TRANSPORT-2026-07-26` retired the `#tile.buffer_ref` compatibility reader and generalized the proven Apple ReplaySSM lifecycle schema to target-keyed resident ABIs, adding MoE launch-workspace ownership and optional rank/device topology binding. Apple keeps its session-private ring, flush/rollback, ordered submission, and drain-before-release semantics; the open item is Metal threadgroup scheduling against the generalized schema. APPLE-PIPE-1 already rejects name-based `#tile.buffer_ref` identity, so Apple is aligned with the retirement. |
| 29 | APPLE-DEVICE-EVENT-1 | **open — scoped to the MPSGraph route** | *Corrected 2026-07-27: the first diagnosis of this row was wrong.* The device timer is **not** broken on the descriptor lane. `tessera_apple_gpu_last_dispatch_device_time_ns()` reads `-1` only because dispatch telemetry is **opt-in and off by default**; after `tessera_apple_gpu_dispatch_telemetry_set_enabled(1)` the MSL softmax route reports a real `device_time_ns` with `timing_source=1` and a full threadgroup/execution-width resource record. The genuine gap is narrower: the **matmul route runs through MPSGraph**, which populates neither the command-buffer device timer nor the MSL dispatch record. Because `required_timing_domains` is report-wide and every family in scope must supply both domains, one family without a device timer forces the whole `apple_gpu` packet onto `kernel_wall`. Closing this means giving the MPSGraph route a device timer (or moving matmul to an MSL/`simdgroup_matrix` route that already has one), then re-recording with `required_timing_domains = ["device_event", "end_to_end"]`. Independent of `CAP_DISPATCH_BOUNDARY_SAMPLING` (bit 4), which this M1 Max does not report — the command-buffer interval needs no counter sampling. |
| 30 | APPLE-PLACEMENT-ABI-1 | **landed (2026-07-27), extension open** | `tessera_apple_gpu_softmax_f32` and `tessera_apple_gpu_bmm_f32` are `void` ABIs that fall through to a numerically-identical CPU reference when Metal is unavailable or a pipeline/allocation/command fails. Nothing in a numerical proof distinguishes the two paths, so an oracle-matching fixture could have been sealed as Level-C GPU evidence while running on the host. Both now have status-bearing twins (`..._f32_status`) following the documented TILE-1 precedent at `tessera_apple_gpu_mps_matmul_f16_status`, and the fleet recorder refuses to seal a fixture or benchmark whose placement is not positively proven — at the fixture shape *and* the timing shape, since a dispatch can succeed at one and fail at the other. The MSL dispatch record is captured where the route populates it (softmax) and reported absent, never inferred as CPU, where it does not (MPSGraph matmul). **Open:** the other ~130 `void` Apple GPU entry points have the same latent hazard; any that a benchmark or packet records must gain a status twin before its result is admitted as GPU evidence. |

## Canonical validation lanes

After APPLE-TEST-1 establishes complete marker coverage, the Apple host should
run these as independent commands:

```bash
# Host-free compiler, selector, validation, rejection, and fallback contracts.
python3 -m pytest tests/unit -q \
  -m "not hardware_apple_gpu and not performance"

# Apple compiler artifacts; this lane does not claim device execution. It
# reports foreign compiler proofs as explicit per-platform skips.
python3 scripts/run_apple_host_free_compiler_tests.py \
  --build-dir build-apple \
  --tool build-apple/tools/tessera-opt/tessera-opt

# Native Metal correctness, twice from the same fresh build/runtime image.
python3 -m pytest tests/unit -q \
  -m "hardware_apple_gpu and not performance" --durations=100 \
  --junitxml=/tmp/apple-device-correctness.xml

# Measured lane: serial execution only.
python3 -m pytest tests/unit -q -n 0 \
  -m "hardware_apple_gpu and performance" --durations=0 \
  --junitxml=/tmp/apple-performance.xml

# Metal 4 promotion runs locally on the named Mac, never through a registered
# GitHub runner. Push the sealed packet into the coordinating PR; portable CI
# rejects zero selected tests, skips, hash drift, unknown GPU families,
# reference rows, missing device intervals, and incomplete two-domain ledgers.
bash scripts/run_apple_metal4_release_gate.sh \
  --publish-dir docs/audit/evidence/apple/metal4/<run-id>

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

Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16`: shared typed Tile IR now
permits logical `scale_a`/`scale_b` fragments only on NVFP4 MMA descriptors.
Apple has no enabled NVFP4 cooperative-matrix route, so this is follow-up
required at capability rejection only; no NVIDIA nibble, lane, scale-selector,
or OMMA mapping applies to Metal.

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16`: the shared `FusedRegion`
oracle now names bias/activation/residual order and emits registered
`E_FUSED_EPILOGUE_*` rejection diagnostics. Apple retains its architecture-owned
MSL/Metal 4 schedules. NVIDIA validates the complete 43-case supported
execution matrix; Apple independently validated its supported semantic order,
dtype matrix, residual guards, and diagnostics on the exact Metal host before
closing APPLE-EPILOGUE-1. The schedules and exact-device claims remain
architecture-specific.

Cross-backend sync `PR420-REVIEW-2026-07-17`: not applicable to Apple compiler
or runtime behavior. The scale-origin repair and canonical `fp16` alias are
confined to the SM120 NVIDIA fragment materializer/selector, and the bootstrap
ordering repair is confined to Ubuntu apt.llvm.org setup. No Apple IR, Metal
layout, dtype support, ABI, schedule, or exact-device claim changes.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18`: not applicable to Apple
runtime execution. The change adds a CUDA-owned packed NVFP4 ABI, SM120
HMMA/QMMA/OMMA kernels, CUDA-event evidence, and device-keyed NVIDIA selector
rows. It changes no portable dtype spelling, ScaleLayout, epilogue order, or
autotune schema. Apple remains SDK-gated for FP8/FP4 tensor execution and does
not inherit CUDA fragments, resource values, timings, or promotions.

Cross-backend sync `E2E-SPINE-2026-07-18`: Apple participates in the shared
native-image and launch-descriptor contract through **APPLE-E2E-1** and
**APPLE-CPU-E2E-1**. The shared work may select and package an existing typed
Apple pipeline, but it does not transfer CUDA/ROCm schedules, change Metal
placement, promote a route, or convert host-free compilation into exact-device
proof. Existing runtime and artifact routes remain available until their
canonical replacements meet all four proof layers on the named Apple host.
The behavior-neutral E2E-SPINE-0 foundation is complete: Apple CPU/GPU exact
targets now have total declared-pipeline ownership and truthful partial-B/
absent-C inventory rows; Apple runtime selection is unchanged. E2E-SPINE-1 is
also complete: Apple will consume the shared image/descriptor identity,
bindings, generic geometry, workspace, ordering, and diagnostics, while Metal
threadgroup schedules and placement remain Apple-owned. No Apple route or
exact-device status changes until APPLE-E2E-1.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no Metal hook and
does not change value-mode classification, MPSGraph/Metal placement, pipeline
cache policy, or selectors; APPLE-E2E-1 still owns native package production,
Apple registration/submission, comparison, cleanup, and Level-C proof.
E2E-SPINE-3 is applicable as a family-granular proof envelope around bounded
Apple GPU/CPU Level-C scope. It standardizes shared fixture identity, cache
replay fields, benchmark metadata, sealed attachment hashes, and generated
fleet truth without changing Metal/Accelerate ABIs, schedules, placement, or
selectors. The existing Apple7 packet remains exact-device evidence for its
declared scope only; second-device/fleet proof remains APPLE-NATIVE-E2E-2 and
cannot be inferred from Apple7.
Fleet packet identity is now `(target, architecture)`, and Apple CPU plus
Apple7 GPU packets remain assigned to the M1 Max host. The NR2 WSL
`x86_64_base`/`sm_120a` slice and Strix Halo `x86_64_avx512`/`gfx1151` slice
transfer no Metal/Accelerate ABI, schedule, resource, timing, or readiness
claim. Apple packet recording remains a Mac-host follow-up.
The post-merge NR2 WSL packets now hash-seal base-x86 and bounded SM120
softmax/reduction evidence against source commit
`9f3757ef2dda2dd61ff94f1aefe0244f1b80f064`. Their generated-dashboard rows
do not change the Apple disposition: Apple CPU and Apple7 remain
`packet_pending` until independently recorded on the assigned M1 Max.
The NVIDIA-E2E-1 f16 landing slice was assessed as NVIDIA-only: it adds an
SM120 PTX package producer and exact CUDA submission hook, with no Metal hook,
Apple ABI, dtype registration, schedule, placement, or selector change.
The completed NVIDIA-E2E-1 NVFP4 slice extends the shared `tile.matmul_kernel`
verifier with an explicit packed-A/packed-B/scale-A/scale-B/output/M/N/K form.
Apple has no enabled NVFP4 cooperative-matrix execution route, so this is not
applicable to Metal lowering and requires no Apple ABI or selector change.
Apple inherits only the shared verifier rejection contract, not CUDA scale-word
packing, warp geometry, resource values, timings, or exact-device claims.

The first NVIDIA-E2E-2 slice changes the shared Graph→Tile async contract so a
copy produces `!tile.async_token`, its wait retires that token, and a matrix
consumer carries the dependency. Apple has no consumer for CUDA TMA/WGMMA
physical scheduling; its Metal and CPU pipelines, ABI, placement, selectors,
and execution claims are unchanged. The additive pipeline-registry
driver-source field and `tessera_nvidia` dialect manifest row are NVIDIA
bookkeeping. Exact SM builders are not applicable to Apple and transfer no
CUDA layout or schedule.

The NVIDIA-E2E-2 softmax slice adds the shared semantic
`tile.softmax_kernel(X, O, Rows, K)` envelope with explicit storage,
accumulation, and last-axis fields; the envelope now accepts f16/f32 storage
with f32 accumulation. It is not applicable to the current Apple
value/Metal/MPS compilation path, which already owns different typed calls and
physical reduction schedules. Apple does not inherit the SM120 thread-per-row
schedule, `nvvm.ex2`, PTX ABI, resources, timings, placement, or selector; no
Apple execution state changes.

The NVIDIA-E2E-2 dtype-totality slice changes the shared MMA selector contract
so fp32 Tensor Core selection requires an explicit TF32 math mode and bare
`fp4_e2m1` cannot alias NVIDIA NVFP4. Apple has no TF32 or NVFP4 cooperative
matrix route, so this is semantic parity only: it receives no CUDA scalar type,
fragment packing, MX/NV scale layout, PTX ABI, execution, or selector claim.
APPLE-DTYPE-1 remains SDK-gated for its own FP8/FP4 tensor formats.

The follow-on SM120 dtype slice adds a backend-private
`tessera_nvidia.mx_block_scale_mma` Target IR op and ptxas-backed FP6/MXFP4
register contracts. This is not applicable to Apple code generation: it adds
no shared storage dtype, Metal op, SIMD-group layout, scale ABI, runtime route,
or selector state. Apple FP8/FP4 proof remains owned by APPLE-DTYPE-1.

The NVIDIA-E2E-2 reduction slice adds a shared launch-level
`tile.reduce_kernel` semantic carrier. It is not applicable to Apple's current
value/Metal/MPS compilation path, which owns different typed reduction calls,
placement, and SIMD-group schedules. Apple inherits no SM120 launch ABI,
resources, timings, execution state, or selector change.

The NVIDIA-E2E-2 epilogue slice tightens only the shared Tile launch verifier
for explicit residual operands and order. Apple's existing typed Metal/MPS
epilogue contracts remain architecture-owned and inherit no CUDA ABI, layout,
resources, timings, execution state, or selector change.

The NVIDIA-E2E-2 attention slice adds a shared launch-level semantic carrier
for Q/K/V/O dimensions, storage/accumulation, scale, and causal behavior. It is
not applicable to Apple's existing MPSGraph/Metal attention executors and
transfers no CUDA schedule, pointer ABI, resources, timing, readiness, or
selector state. Any Apple adoption requires its own Metal materializer and
exact-device proof.

The NVIDIA paged-KV slice adds a shared logical-page gather carrier with
explicit f32 page storage, i32 page table, dimensions, range, and direct-route
semantics. Apple's resident Metal page-table attention remains architecture
owned and inherits no PTX ABI, CUDA schedule, evidence, or selector state.

The NVIDIA backward-attention slice adds a shared launch-level VJP carrier with
explicit determinism, mask/softcap, route, and workspace semantics. It is not
applicable to Apple's existing Metal/MPSGraph backward executor without an
Apple-owned materializer; no CUDA single-owner schedule, pointer ABI,
atomic/split resources, timing, readiness, or selector state transfers.

Cross-backend sync `E2E-DEVICE-LIBS-2026-07-19` adds logical name, content
digest, and link mode for LLVM-stage device libraries to the shared native-image
schema. It is not applicable to the current Metal/MSL/metallib path, which does
not link CUDA libdevice or ROCm OCML/OCKL/OCLC bitcode. Apple records no device
library and inherits no CUDA/ROCm discovery paths, symbols, cache keys, or
linker choices.

Cross-backend sync `CUDA-MATH-CONTRACT-2026-07-19` adds backend-neutral
`exp_mode` and `ftz` semantics to the shared Tile softmax envelope. The current
Apple paths do not consume that launch-level op, so the SM120 mapping to PTX
`ex2.approx.f32`, its 2-ULP bound, and CUDA cache-policy version are not
applicable. A future Apple lowering must select and prove its own Metal precise
or fast-math exponential route rather than inherit the CUDA approximation.

Cross-backend sync `CUDA-INTRINSIC-SURFACE-2026-07-19` adds shared canonical
toward-positive and toward-negative rounding names without changing the
existing default tuning sweep. CUDA's RN/RD/RU/RZ cast suffixes, integer packed
dots, and 2x16/4x8 SIMD functions are not Metal execution evidence. Apple must
map directed conversions and packed operations to its own MSL/Metal semantics
and device proof; no Apple ABI, route, or selector changes in this landing.

Cross-backend sync `PTX-TYPE-MEMORY-TRUTH-2026-07-19` is NVIDIA-private
physical truth: PTX bit registers, fragment operands, scopes, proxies, and
packed-access ordering do not transfer to Metal. Apple retains its own MSL
storage, SIMD-group/cooperative-tensor formats, address spaces, barriers, and
memory-order proof. The shared architectural conclusion is only that a language
dtype wrapper is not evidence of a native register or matrix execution route.

Cross-backend sync `NVIDIA-E2E-DTYPE-EXEC-2026-07-19` adds f64 to the portable
Tile epilogue output vocabulary for NVIDIA's compiler-owned DMMA path. This is
not applicable to Metal GPU execution: Apple GPU profiles expose no native
fp64, so no MSL type, SIMD-group matrix route, ABI, timing, or selector state
changes. Apple CPU fp64 remains independently owned.

Cross-backend sync `ROCM-E2E1-SOFTMAX-2026-07-19` is ROCm-owned. It maps the
already-shared `tile.softmax_kernel` envelope to `tessera_rocm.softmax`, adds an
HSACO package producer, and registers a gfx1151 HIP descriptor consumer. Apple
inherits no AMD exponential implementation, wave/LDS schedule, HSACO ABI,
resource value, timing, execution state, or selector change. The only shared
surface remains the previously assessed semantic Tile envelope and portable
native-image/launch schema. ROCm's content-addressed OCML/OCKL/OCLC population
is not applicable to Metal; Apple inherits no device-library record or cache
change.

Cross-backend sync `ROCM-DTYPE-TOTALITY-2026-07-19` is ROCm-owned and not
applicable to Apple target state. It introduces no shared dtype spelling or
alias and transfers no RDNA scalar, packed-dot, WMMA, accumulator, storage,
runtime, or selector claim to Metal or Apple CPU.

Cross-backend sync `ROCM-DTYPE1-CLOSE-2026-07-21` promotes signed `int4` and
alias `i4` into the shared canonical/Graph-IR vocabulary and adds signedness to
the shared packed-storage descriptor. Apple parity is validated at the logical
signed-int4 boundary; existing Metal packed-weight ABIs remain backend-owned.
No Apple target capability, physical schedule, runtime route, or selector is
promoted by the gfx1151 proof, and unsigned packed-4 remains unregistered.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: ROCM-E2E-1 memoizes
deterministic hashes for frozen runtime artifacts, native images, and launch
descriptors. Serialized identity values and required launch validation are
unchanged, so Metal schema parity is validated; no Apple ABI, schedule,
runtime route, performance claim, or selector changes.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19` is ROCm-owned. It consumes the
already-shared `tile.reduce_kernel` carrier and widens only its portable storage
verifier to admit bf16; the op registry and `Outer/AxisExtent/Inner` schema are
unchanged. Apple mappings are unchanged, and Metal/MPS reduction ABIs,
threadgroup schedules, exact-device evidence, runtime routes, and selectors are
unchanged; the ROCm five-argument HSACO ABI transfers no Apple claim.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19` is ROCm-owned. It consumes
the existing shared paged-KV carrier without changing its verifier or public op
schema. The ROCm directive, 256-thread gather, HSACO ABI, page-table validation,
and gfx1151 evidence transfer no Metal/MPS schedule, ABI, readiness, timing, or
selector claim; Apple's paged-cache routes remain independently owned.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19` is ROCm-owned. It
consumes the existing shared MoE dispatch carrier and public operation without
changing their verifier or dtype registry. The AMD direct-gather schedule,
HSACO ABI, index validation, and gfx1151 evidence are not applicable to Metal;
Apple retains its independent MoE transport implementation and selector.

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for host x86 targets as `native_cpu` with CPU-wall timing.
Apple GPU remains `native_gpu`, Apple CPU retains its independently owned
runtime routes, and no Metal/MPS ABI, schedule, device evidence, timing,
readiness, or selector state transfers. The x86 pilot consumes existing Tile
softmax/reduction carriers without changing their shared dtype or operation
registration.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the existing shared
matmul and attention carriers for f32 AVX-512 descriptors. Apple inherits no
x86 ABI, host vector schedule, timing, readiness, or selector state. Metal/MPS
matmul and attention remain independently selected, and x86's equal-head and
zero-dropout descriptor restrictions change no Apple capability or verifier.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. Apple parity is not applicable; no Apple pipeline, ABI,
schedule, capability, or selector changes. X86-E2E-2 subsequently closed the
remaining inventory and reassessed Apple at each shared-contract boundary.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. Apple parity is assessed at the carrier boundary only;
the AVX-512 ABIs, host-vector schedule, CPU-wall evidence, 16K binary selector
threshold, and runtime readiness do not transfer to Metal or Accelerate. No
Apple target, dtype, operation, ABI, execution, or selector row changes.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The only capability change is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. Apple inherits no host-vector ABI, null-operand
convention, 32K selector threshold, CPU timing, or execution claim; Metal and
Accelerate rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
Apple parity is assessed at the carrier boundary: no AVX-512 polynomial,
CPU-wall threshold, C ABI, Metal schedule, execution row, or selector transfers.
Existing Apple operations and routes remain independently owned.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. Apple parity is not
applicable: no Accelerate/Metal dtype capability, ABI, schedule, evidence, or
selector changes, and future ACE planning transfers no Apple execution claim.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, Apple target capabilities, Metal/MPS ABIs, schedules, and
selector state are unchanged; the Apple outcome is parity validated by the
shared attention lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations. The Apple Metal
evidence lane continues to use the pinned
`/opt/homebrew/llvm-23.1.0-rc1` prefix described above; a versioned Homebrew
`llvm@23` keg does not substitute for that evidence-producing toolchain.
Apple target semantics and Metal/MPS runtime contracts are unchanged, and the
LLVM 23 compiler/lit build validates parity.

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
now consumes items **8, 9, 10, 11, 13, 14** as its Apple execution vehicle — it
adds candidates/state-types under existing items rather than opening new ones,
and **inherits this plan's evidence contract unchanged** (native `native_gpu`
placement, separate GPU/end-to-end timing-domain keys, two-run + ≥5% promotion,
forced binding-miss → `reference_cpu`). Concretely: channel-wise KDA/GDN decode →
**APPLE-REPLAY-1** (extend ReplaySSM / `SSMStateHandle` / `DeltaNetStateHandle`);
`sliding_window`/full mixer forward has closed its current **APPLE-ATTN-FWD-1**
scope; any new Sequence Mixer forward candidate requires a separately scoped
follow-up rather than silently reopening that item. `windowed_kv` +
uniform-block planner → **APPLE-PAGED-KV-1**;
chunkwise-scan inner GEMMs → **APPLE-RETUNE-1**; mixer arbiter → **APPLE-ROUTE-1**;
low precision → **APPLE-DTYPE-1** (stays SDK-gated — no NVFP4 cooperative-matrix on
Apple, so the executing FP4 proof is on NR2 Pro sm_120); mixer backward →
**APPLE-ATTN-BWD-1**. This is a direction pointer; it changes no Apple gate,
route, or exact-device claim here.

On a subsequent Apple-host collection, refresh the recorded marker totals and
append any new failure table by execution family and device generation without
discarding the current exact-device evidence. This plan is already in
`landing`; move it to the Apple archive only after every completion gate is
met.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. Apple parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no Metal/MPS implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, Apple ABI, Metal/MPS schedule, dtype capability,
execution row, or selector. Apple parity is therefore not applicable; public
composite semantics continue to be assessed by Apple-owned typed routes.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the Apple not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` repairs the host build and
sanitizer lanes after the LLVM/MLIR 23 migration. The shared capability audit
also corrects the existing Apple GPU matmul row to admit the already-shipped
f32/f16/bf16 MPS and Tile-simdgroup value ABIs. This is parity repair for an
existing Apple contract, not a new Metal schedule or exact-device claim.

Cross-backend sync `E2E-SPINE-2026-07-18` extends the shared launch-level Tile
carrier inventory with deterministic f16/f32 attention-backward dropout replay,
an explicit fused paged-attention causal-offset descriptor, and typed
f16/bf16/f32 MoE storage. The NVIDIA materializers, PTX ABIs, SM120 schedules,
and exact-device evidence do not transfer. Apple already owns separate Metal
attention/paged-cache and low-precision dispatch contracts, so no Apple runtime
mapping is required by this NVIDIA slice; future use of the new portable
carrier spellings must be proven through an Apple-owned lowering and exact
Apple device evidence.

Cross-backend sync `ROCM-E2E-SPINE3-TEST1-2026-07-21` adds shared paged-KV and
MoE fixture identities to the E2E-SPINE-3 corpus and correctly marks nine
Metal-only compiler nodes as `compiler_apple`. Apple fixture-schema parity and
compiler ownership are validated; the gfx1151 HSACO, HIP launch contract,
resource fingerprints, timing, and exact-device packet do not transfer to
Metal. No Apple capability, schedule, execution row, or selector changes.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` lands the Apple-owned
declarative fusion table/generic rewrite and declarative value-envelope shape
constraints, and closes 11 shared dialect verifier holes. The shared MMA
selection is now recorded in Apple manifest rows and is available as an
equal-tier arbiter cost tie-break. Existing Metal/MPS ABIs and physical
schedules are unchanged; the LLVM 23 build validates compiler parity, while
exact-device performance evidence remains Apple-owned.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` adds an executable physical
layout contract to the generic emitter/cache and lands the first guarded
dynamic-shape execution route on x86. Apple is **follow-up required**: its
shape-materialized MSL candidates remain bucketed and no x86 row-major
materializer, CPU guard, or dtype default transfers to Metal/MPS. Apple keeps
its existing dtype and physical-layout ownership until an Apple-specific
materializer and exact-device evidence land.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. Apple remains **follow-up required** for an architecture-owned
Graph-cast materializer; the pass stays opt-in and transfers no Metal layout,
schedule, selector, or device proof. The x86 dynamic last-axis reduction guard
is not applicable to Apple’s bucketed MSL routes. Shared add/multiply/static-
broadcast adjoints change Graph IR only; no Apple backward runtime or exact-
device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only and are not applicable to Apple's
bucketed MSL routes; no Metal ABI, schedule, selector, backward runtime, or
exact-device claim transfers. Apple's architecture-owned Graph-cast consumer
is host-validated: it accepts row-major/BHSD/NHWC before runtime fusion/per-op
lowering and rejects unsupported column-major bindings. This changes binding
metadata only and claims no exact-device proof.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. Apple is **follow-up required** for backward execution:
no Metal/MPS ABI, affine gamma/beta contract, schedule, selector, runtime
binding, performance result, or exact-device proof is added here. Dynamic
statistics remain Graph IR until an Apple-owned materializer is implemented.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. Apple is **follow-up
required** for an architecture-owned dynamic affine normalization materializer
and backward runtime: the gfx1151 HSACO and AVX-512 ABIs, schedules, timing,
and exact-device evidence do not transfer to Metal/MPS. Shared static/dynamic
linalg and CPU-oracle proof validate the Graph contract only; no Apple
selector, execution row, or device claim changes.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` changes only
the ROCm architecture-owned backward schedule and temporary-buffer ABI. The
shared affine adjoint and f32 accumulation contract are unchanged. Apple still
requires its own Metal/MPS backward materializer and exact-device proof; the
gfx1151 two-kernel schedule, bitwise evidence, and timing do not transfer.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` adds family-specific
RMSNorm/LayerNorm backward execution rows and public JIT binding for ROCm and
x86. Apple remains **follow-up required**: neither the gfx1151 HSACO ABI nor
the AVX-512 f32 ABI, schedule, dtype-accumulation contract, timing, or device
evidence transfers to Metal/MPS. The shared Graph adjoint and dynamic Linalg
contract remain parity validated; no Apple execution row or selector changes.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. Apple parity is host-validated at Graph/linalg level, and
function-budgeted liveness-aware rematerialization is shared. The
address-space-3 Tile arena pipeline change is not applicable to Metal, whose
threadgroup allocation is architecture-owned; no MSL allocation, backward
runtime, performance, selector, or exact-device claim changes. An Apple
threadgroup-arena materializer remains follow-up required.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. Shape-preserving MSE participates in shared layout
propagation, and post-autodiff rematerialization now distinguishes saved
forward activations from backward temporaries. Apple parity is validated at
the shared IR boundary only. The gfx1151 HIP composition, module cache,
timings, and AVX-512 execution do not transfer to Metal/MPS; an Apple-owned
compiled MSE backward launch and exact-device evidence remain follow-up
required.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` adds shared
runtime-sized arena planning, cost-aware rematerialization metadata, and the
x86-owned default Graph-layout bridge. The new MSE backward execution proof is
ROCm gfx1151-only. Apple still requires its own compiled MSE VJP and
threadgroup-arena materializer; no HIP schedule, address-space-3 allocation,
x86 binding default, selector, or device claim transfers to Metal/MPS.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` adds shared
Graph-native MAE, Huber, SmoothL1, and SGD adjoints with dynamic Linalg and CPU
oracle proof. Apple is **follow-up required** for an architecture-owned
Metal/MPS backward materializer and exact-device evidence. The gfx1151 HIP
kernel, AVX-512 C ABI, module cache, timing, and selector state do not transfer.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds shared
Graph-native stable BCE-with-logits, class-index/label-smoothed cross entropy,
KL/JS, explicit Momentum/Nesterov state, and explicit Adam/AdamW moment-state
adjoints. Dynamic shared Linalg contracts are live for BCE, Momentum/Nesterov,
and Adam/AdamW. Apple is **follow-up required** for Metal/MPS backward
materializers and exact-device evidence; the gfx1151 and AVX-512 loss and
optimizer ABIs do not transfer. No Apple selector or support claim changes.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` adds shared
single-use loss-backward to SGD/AdamW fusion carriers and one-loop dynamic
Linalg lowering for MSE, MAE, Huber, SmoothL1, and BCE-with-logits. Apple parity
is validated only at the shared Graph/Linalg contract. Apple remains
**follow-up required** for an architecture-owned Metal/MPS fused training
materializer and exact-device evidence; gfx1151 HIP and AVX-512 ABIs, cache
identities, timings, and selector decisions do not transfer.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` replaces
the shared static address-space-3 alloca with a workgroup global and supports
dominance-scoped dynamic arena cohorts. This is not an Apple Metal allocation
claim: Apple still needs its architecture-owned threadgroup materializer and
exact-device evidence. The measured rematerialization corpus has gfx1151 and
AVX-512 rows only; no cost, selector, layout-default, or occupancy result
transfers to Apple.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` broadens the
shared measured-rematerialization schema to exact consumer chains and
64/128/192 matmul shapes with ReLU/GELU/SiLU. Apple remains **follow-up
required** for Metal measurements and policy selection. ROCm dynamic
normalization epilogues, HIP launch-sized LDS materialization, and packed IU4
WMMA are architecture-owned and transfer no MSL threadgroup-allocation,
packed consumer, performance, or selector claim. Apple's threadgroup arena
and physical packed consumers remain architecture-owned follow-ups.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24` extends the
shared rematerialization corpus schema with softmax, RMSNorm, and MSE producer
families plus measured workload-budget decisions. Apple remains **follow-up
required** for Metal measurements and policy selection. ROCm's packed
multi-arena LDS ABI, GELU normalization epilogue, and terminal-pack
dequant-GEMM consumer are architecture-owned; no MSL threadgroup allocation,
packed consumer, timing, selector, or support claim transfers. Apple's
threadgroup path-max contract and physical packed consumers remain open.

Cross-backend sync `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24` extends the
shared rematerialization evidence schema to a measured four-layer workload with
softmax, RMSNorm, MSE, Huber, SmoothL1, and BCE instances. Apple remains
**follow-up required** for Metal measurements and policy selection. ROCm's
branch-path dynamic-LDS expression, binary normalization epilogues, and packed
elementwise/sparse/cache ABIs are architecture-owned; no MSL threadgroup
expression, packed ABI, timing, selector, or support claim transfers.

Cross-backend sync `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24` adds a shared
model/device-derived rematerialization budget contract with explicit override
precedence and bounded dynamic parameters. Apple is **follow-up required** to
inject exact device capacity/reserve policy and validate model-level selection
with Metal measurements. ROCm's alias-aware nested/loop LDS slots and
40,208-byte gfx1151 packet are architecture-owned; no MSL threadgroup-memory
expression, occupancy, execution, or selector claim transfers.

Cross-backend sync `E2E-SPINE3-SM120-MEMORY-2026-07-24` extends the shared
fleet fixture corpus with bounded epilogue, attention, and ReplaySSM identities
and seals the six formerly pending NVIDIA SM120 family rows. Apple can reuse
only fixture identity and proof-schema structure. CUDA image descriptors,
NVPTX address-space-3 materialization, ptxas accounting, SM120 resources,
timings, and release readiness do not transfer to Metal. Apple CPU/Apple7
packet scope and the architecture-owned threadgroup-arena follow-up are
unchanged.

Cross-backend sync `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` is
NVIDIA-owned. It changes no shared Graph/Linalg mathematics and no Apple
execution row, selector, Metal ABI, or threadgroup-memory policy. The CUDA PTX
image/descriptor, CUDA-driver launch-v2 entry points, NVPTX external shared
symbol, ptxas/driver resources, and SM120 timings do not transfer to Apple
CPU/GPU. Apple retains its architecture-owned training and dynamic
threadgroup-arena follow-ups.

Cross-backend sync `CUDA-TRAINING-MEMORY-BREADTH-2026-07-24` adds only the
portable Graph IR carriers for model-parameter marking and bounded dynamic
parameter storage. NVIDIA owns the CUDA-context capacity/free-memory query,
FP16/BF16 PTX training ABIs, serialized dynamic-shared launch expressions, and
SM120 measurements. None transfers to Metal or closes Apple training,
threadgroup-memory, capacity-policy, or exact-device evidence.

Cross-backend sync `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` widens the
shared Tile softmax and attention verifier envelope to BF16 storage with FP32
accumulation and preserves the already-shared BF16 reduction contract,
including min. Apple parity is validated at that semantic boundary: the
existing Apple capability and execution records already own independent BF16
softmax, attention, and reduction routes. NVIDIA's typed PTX descriptors,
serial/cooperative-128 schedules, CUDA-driver ABI, ptxas resources, SM120
numerics, and WSL timings do not transfer to Metal or Apple CPU. No Apple
manifest, schedule, execution state, or selector changes are required.

The NVIDIA continuation adds an architecture-owned compiler/PTX normalization
image and consumes the already-shared `tessera.storage_pack` descriptor in
scale-bearing NVFP4/MXFP4/FP6 CUDA materializers. Apple semantic parity is
unchanged: its independent BF16 normalization routes remain authoritative,
and CUDA packing factors, scale ABIs, PTX byte loads, rejection fixtures,
resources, and SM120 evidence do not transfer to Metal. No Apple storage
legalization default, packed dtype, capability, or selector changes.

Cross-backend sync `NVIDIA-PACKED-MATH-2026-07-25` adds a CUDA-owned signed
INT4 descriptor consumer and a typed internal Tile carrier for a bounded CUDA
integer/cast/packed-SIMD subset. The shared `tessera.storage_pack` schema is
unchanged. NVIDIA's nibble layout, PTX instructions, CUDA launch ABI,
resources, cache evidence, and SM120 numerics do not transfer to Metal or
Apple CPU. Apple retains its own packed INT4 and math execution contracts; no
Apple capability, storage-legalization default, schedule, or selector changes.

Cross-backend sync `NVIDIA-PACKED-SSA-FOUNDATION-2026-07-25` changes the shared
pack descriptor from an unstructured dictionary to portable
`#tile.packed_format`/`#tile.packed_view`/`#tile.scale_layout` attributes and
adds generic packed load/store plus SSA buffer/pipeline vocabulary. Apple is
**follow-up required** for architecture-owned Metal packed physical consumers
and threadgroup allocation/pipeline threading. NVIDIA scale indexing, PTX,
CUDA Math target operations, SM120 resources, and device evidence do not
transfer; no Apple support or selector state changes.

The same synchronization point now adds shared SSA TMA descriptor, mbarrier,
mbarrier-token, TMEM, and TCGen05 vocabulary and makes NVIDIA WarpSpec consume
the shared allocation/pipeline identity. These operation definitions are
portable compiler structure, not Apple execution support: Metal has no TMA,
TMEM, or TCGen05 consumer, so those operations are **not applicable** to Apple
with that architecture-specific reason. Apple threadgroup allocation and
pipeline-state threading remain **follow-up required** on its own lowering;
no NVIDIA resource, runtime, or exact-device claim transfers.

Cross-backend sync `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds ROCm-owned
Adam/AdamW and KL/JS physical backward execution plus a ROCm normalization
softcap epilogue; no HIP kernel, gfx1151 timing, or selector evidence
transfers to Metal/MPS. Apple remains follow-up required for its
architecture-owned training backward materializers. The shared change is the
target-neutral, serializable dynamic-local-memory expression field on
`LaunchDescriptor`; Apple has no threadgroup-memory consumer of that field in
this change and retains its separately owned threadgroup materialization gap.

Cross-backend sync `ROCM-LION-BACKWARD-2026-07-27` adds only the ROCm-owned
physical consumer of the already-shared Lion stop-sign VJP policy and extends
the ROCm operation-total benchmark packet. HIP code objects, gfx1151 numerics,
and WSL timings do not transfer to Metal. Apple remains follow-up required for
an architecture-owned compiled Lion backward materializer; no Apple
capability, execution row, selector, or threadgroup-memory contract changes.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` emits a shared
unique-clock warmup/steady/cooldown dependency order after pipeline legality.
Metal/runtime consumption and collective overlap remain Apple-owned follow-up;
the structural carrier changes no Apple capability, selector, or exact-device
claim.

Cross-backend sync `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` adds a shared
runtime consumer for emitted 1F1B steps, including an independent collective
transport executor; measured schedule records now alter physical Schedule/Tile
attributes after target and evidence validation; and DeltaNet-family reverse
mode is an analytic carried-state recurrence with explicit forward/backward
schedule metadata. Apple can consume these shared contracts, but this change
contains no multi-rank Metal transport packet, Apple capacity injection,
measured Metal selector result, or sequence-mixer backward kernel. Those remain
Apple-owned exact-device follow-ups.

NVIDIA layout assignment now defaults on only because its named pipeline has an
immediate physical Graph-cast consumer. Apple already owns a separate
row-major/BHSD/NHWC Graph-layout materializer; no NVIDIA layout or execution
claim transfers. ROCm's factored Adafactor HSACO and gfx1151 timing likewise do
not transfer to Metal. An Apple factored optimizer implementation remains
follow-up required.

Cross-backend sync `CORE-PRODUCTION-EVIDENCE-2026-07-27` makes emitted pipeline
steps own serializable collective descriptors and adds a shared
replicated/rank-local OptimizerShard state machine. The runtime integration is
portable, but this continuation has no Metal multi-device transport
implementation or Apple exact-device packet. ROCm's physical Adafactor adjoint
and reverse-chunk DeltaNet HSACO are AMD-specific and do not transfer. Apple
sequence-mixer backward packaging and refreshed measured selector evidence
remain architecture-owned follow-ups.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` adds the
exact modified-Delta normalization VJP to physical ROCm and AVX-512 backward
paths and proves affine parallel chunk composition for `erase=false`. This
changes shared algorithm evidence, not Metal execution. The five-entry gfx1151
HSACO, AVX-512 ABI, and their resident timing packets do not transfer to Apple.
Metal sequence-mixer backward packaging, nonlinear/erase chunk scheduling, and
a refreshed exact-Apple-host selector packet remain architecture-owned.
