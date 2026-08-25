---
last_updated: 2026-08-24
audit_role: plan
plan_state: open
owner: x86 backend
target: x86_avx512
scope: x86 AVX-512 implementation/proof and AMX access planning
---

# x86 backend TODO

Cross-backend sync `SO3-INFER-EDGES-2026-08-24` — **shared W2.1/W5.2e
dependence-inference semantics + MegaMoE R3 producer; x86 outcome: parity
validated (AVX-512 host).** Same shared-analysis correction as the rocm
entry. x86 hosts the R3/composition analysis itself, so this backend is the
regression gate for the change: the composition-cost, graph-dataflow, and
MegaMoE suites pass on this host against the corrected semantics, with no
generated code or numerical change (analysis is prune/rank-only).


Cross-backend sync `NUMPOL-CARRIER-1-2026-08-24` — **shared Schedule→Tile
`numeric_policy` carrier contract (integrated-plan queue row 3b); x86
outcome: follow-up required.** Newly owned row, nothing implemented yet.
x86 has no fragment type, so it has NO carrier today — the accumulator
contract stated at Graph IR does not exist by the time the AVX-512 emitters
pick an instruction (Decision #32's original defect, on this backend in its
purest form). The row's acceptance names bit-identical existing x86 outputs,
so this backend is both a consumer of the contract and a regression gate for
it. Clean Zen 5 evidence required for any realizability verdict (FORGE §1.3).


Cross-backend sync `LAYOUT-ALG-APPLE-PHYSICAL-2026-08-24` — **shared ABI
assessed; no x86 physical change.** Apple now exports the existing C++ rank-2
plan through the native layout ABI for its MSL emitters and owns fresh M1 Max
proof. The four x86 core GEMMs continue to include the header authority
directly; Metal source templates, simdgroup scheduling, and Apple evidence
transfer no CPU ISA, host-admission, or performance claim.

Cross-backend sync `LAYOUT-ALG-L5-X86-2026-08-24` — **x86 composed-layout
consumer and exact CPU proof closed for the shared materializable set.**
`tessera-tile-to-x86` now re-runs the native C++ layout proof, expands tuple
codomains only as products of independently proven scalar maps, and emits exact
i64 mixed-radix `div/rem` arithmetic for static, bounded-dynamic, nested, and
static tuple-product layouts. CPU-owned guards reject negative or out-of-range
coordinates, nonpositive dynamic extents, and negative dynamic strides before address arithmetic. The generated
LLVM returns the independent expected values on both the local AVX2
Threadripper and the AVX-512 Ryzen AI Max+ 395 host; four malformed-runtime
controls fail closed on each. Non-affine and non-separable tuple codomains stay
outside the shared admissible set rather than being flattened.

Cross-backend sync `LAYOUT-ALG-L4-X86-2026-08-24` — **core AVX-512 GEMM index
authority and exact-host proof closed.** The f32, f64, bf16, and u8s8/VNNI
microkernels now derive every A/B/C rank-2 offset through the same header-only
C++ authority used by Tile materialization; their architecture-owned loop
nests and intrinsics are unchanged. Odd dimensions and vector tails pass an
independent scalar oracle on the AVX-512 Ryzen AI Max+ 395 host. The local
Threadripper AVX2 host now rejects AVX-512 images before `dlopen` instead of
risking `SIGILL`; deterministic admission tests cover both outcomes. This
closes the x86 core-matmul part of L4, not a composed-layout Target consumer or
an AMX claim.

Cross-backend sync `LAYOUT-ALG-L3-L5-DYNAMIC-2026-08-24` — **shared proof,
carrier parity, and x86 physical materialization are closed for the shared
admissible set.**
Factorization/residency and Schedule Object v2 proof fields are portable. The
new tuple-product Tile carrier is registered and the x86 target pass now
materializes it through exact scalar CPU arithmetic. CUDA/ROCm device results
and GPU schedules do not transfer. The core AVX-512 GEMM templates consume the
shared rank-2 authority independently.

Cross-backend sync `DYNAMIC-COMPOSED-SM120-2026-08-24` — **shared carrier
parity and x86 scalar-affine materialization are closed.** Nested outer
shape/stride trees with dynamic scalar-affine leaves now verify in Tile IR and
the x86 target pass consumes that canonical operand order with runtime guards.
The CUDA strided ABI and RTX proof still transfer no AVX-512 schedule or host
evidence. Non-affine carriers remain fail closed.

Cross-backend sync `SCHEDULED-MATMUL-TAIL-EPILOGUE-LDS-2026-08-24` — **shared
Graph lineage assessed; GPU physical work not applicable.** Optional matmul
bias and residual are now ordered Graph SSA operands, but the scheduled x86
consumer rejects the fused form until an AVX-512-owned descriptor/epilogue is
implemented. CUDA K-tail staging, reduced f16 stores, ROCm LDS ownership, and
RTX/Radeon evidence change no CPU loop nest or selector. Dynamic/nested
composed-layout materialization remains carrier-only for x86 too.

Cross-backend sync `SM120-MACRO-CTA-2026-08-24` — **not applicable to x86
physical lowering.** CUDA CTA/warp ownership, f16/bf16 two-stage `cp.async`
shared A/B staging, M/N zero-fill tails, barriers, launch geometry, and RTX
5070 profiling change no CPU loop nest, cache blocking, AVX-512/AMX ABI, or
exact-host evidence. The shared composed-layout carrier itself is unchanged.

Cross-backend sync `SM120-SCHEDULED-LICM-2026-08-24` — **not applicable to x86
codegen.** This is a private NVIDIA MLIR-to-PTX pipeline optimization with RTX
5070 evidence. x86 retains its CPU tiling/vectorization authority; CUDA warp,
CTA, shared-staging, and selector conclusions do not transfer.

Cross-backend sync `SM120-BLOCK-COORDINATE-2026-08-24` — **not applicable.**
The target-owned CUDA CTA coordinate operation changes no x86 loop nest,
AVX-512/AMX address calculation, runtime ABI, or exact-host evidence.


Cross-backend sync `CUTE-LAYOUT-MATERIALIZE-1-2026-08-23` — **shared static
affine coordinate ABI; x86 outcome: not applicable to physical lowering.**
The linear-base producer changes no AVX-512 address form, host ABI, or Zen 5
evidence. NVIDIA's SM120 view mapping and RTX 5070 proof are not an x86 layout
consumer or host proof.

Cross-backend sync `ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **ROCm-owned host-free
CI serialization lane; x86 outcome: not applicable.**
x86 emits host code through the LLVM/JIT path rather than a separate GPU code
object, so there is no hsaco-equivalent artifact to serialize and no analogous
blind spot: the x86 Target IR is already exercised in CI by the lit lane
(`TESSERA_BUILD_X86_BACKEND=ON`). No AMX/AVX-512 evidence is involved.

Cross-backend sync `CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned
pytest capability gate; x86 outcome: not applicable by design.**
x86 deliberately cannot use a `--help` probe: the x86 executable pass is
**always** registered so it fails closed with a rebuild diagnostic, which is
exactly why `tests/tessera-ir/lit.cfg.py:137` probes the dialect directly with a
`!tessera_x86.tile` load-time fixture instead. That mechanism also guards the
assertions-enabled registration regression behind Decision #19, so it is
intentionally kept, not replaced. No AMX/AVX-512 evidence is involved.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
fatbin/cubin runtime admission; x86 outcome: not applicable.** CUDA image
metadata, driver admission, and NVRTC fallback transfer no x86 shared object,
cache identity, physical schedule, AVX-512/AMX ABI, or exact-host evidence.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **NVIDIA cuFFT
foundation landed; x86 outcome: parity validated/no physical change.** The CUDA
package's reusable-plan and explicit-workspace contract transfers no cuFFT
handle, device workspace, schedule, or RTX evidence to AVX-512. x86 retains its
architecture-owned FFT package and bounded digest-keyed workspace.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **NVIDIA parity
exact-device validated; x86 outcome: parity validated/no physical change.** The CUDA backend's
new typed four-mode directive consumes the shared explicit Philox4x32-10
key/counter identity. x86 retains its architecture-owned distribution package,
runtime ABI, and exact-host evidence; no AVX-512 image, schedule, or evidence
transfers to sm_120.

Cross-backend sync `JIT-CACHE-BLOCK-2026-08-23` — **vectorize-lane
cache blocking landed: flat ~139-161 GFLOP/s across n=512-2048 (was
77/45/43 decaying), plus two structural fixes the loop surfaced.**
Sweep result (Strix Halo/Zen 5, env `TESSERA_JIT_CACHE_TILES`): tiling
M or N at the cache level made things WORSE (18-35 GFLOP/s — strided
cache-tile views defeat the inner kernel's vector loads); K-only
chunking won, and shrinking KC to the register k-tile (16) won
outright — 161/144/139 at 512/1024/2048. That config is really a loop
interchange: the k-chunk loop hoists outermost so a (16 x N) B
row-panel stays cache-resident across the whole (i,j) sweep. Pinned as
the default `{0,0,16}`. Structural fixes: **(1)**
`ConvertVectorToSCF` staged transfers through `memref.alloca`s INSIDE
the loop nest (scf.for is no AutomaticAllocationScope) — per-iteration
`llvm.alloca` overflowed the stack at invoke under two-level tiling
(~65k inner iterations at n=512) and was silently costing the
single-level lane too; switched to `full-unroll`, which alone lifted
single-level n=512 from 69 to 94.5 GFLOP/s. **(2)** The vectorizer
unconditionally rewrites the cache-level `tensor.insert_slice` into a
whole-tile transfer pair (64 KB vector SSA values); a
`demoteLargeFullTileTransfers` rewrite restores insert_slice form for
pairs ≥8 KB so bufferization makes them in-place. **(3)** Acceptance
check: `vectorize_children` "succeeds" having vectorized nothing (a
non-dividing KC left the matmul as scalar loops at ~1 GFLOP/s); a
candidate now counts only if no `linalg.matmul` survived, else the
chain falls back (two-level → single-level → scalar). Correctness:
odd/rectangular shapes (130x257x64, 257³, 1000³) verified; full packet
green with the lane on and off; HIP dlopen coexistence retained.


Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` — **the
`TESSERA_JIT_VECTORIZE` GEMM lane now actually vectorizes on LLVM 23:
3.4 → 106.6 GFLOP/s at n=256 (AVX-512 host; 69.0 at 512, 44.4 at 1024,
correctness within the reassoc GEMM tolerance).** Deep-loop findings,
each with a control: **(1) the LLVM-23 scalar carve-out was closed on a
wrong root cause and had never been re-checked** — every fleet box is
LLVM 23 and the compat test was Darwin-gated, so the gate itself kept
the vectorized path unexercised everywhere (the Decision #19
standing-lesson pattern); the recorded bufferization abort no longer
reproduces (verified with the registration absent — the gate removal,
not the registration, unblocked the lane). The tensor
SubsetOpInterface external models are now registered at engine setup
regardless (their absence is MLIR's abort-not-failure path when
bufferization queries an insert_slice).

> **Corrected 2026-08-23 (M1 Max, apple plan `APPLE-VECTORIZE-1`).** The
> abort *does* still reproduce, and the tensor-only registration was
> incomplete. The lane emits `vector.transfer_write` into a tensor, and
> the `vector` dialect promises `SubsetInsertionOpInterface` for it;
> `linalg` carries the same promise. On Darwin the first vectorized
> compile died with `LLVM ERROR: ... promised by dialect 'vector' but
> never implemented`. **This host could not have falsified the claim:**
> the promise check is `#ifndef NDEBUG` (`mlir/IR/OpDefinition.h`,
> `getInterfaceFor`) and the apt.llvm.org LLVM 23 used here is an NDEBUG
> build, so it silently loses the interface (conservative extra copies,
> still correct) instead of aborting. The Mac's LLVM 23.1.0-rc1 has
> assertions on and is currently the only box in the fleet that can
> falsify an MLIR contract claim of this kind — treat any "this MLIR
> abort no longer reproduces" conclusion reached here as provisional
> until it is re-run there. All three registrations (`tensor`, `linalg`,
> `vector`) are now in `tools/tessera-jit`.
>
> **Re-taken on this host the same day, so this is not a Mac-only claim.**
> `llvm-config --assertion-mode` here reports **OFF**, confirming the
> mechanism rather than assuming it; and the *control* — same commit, clean
> tree, unmodified tensor-only registration — passes
> `test_native_cpu_jit.py` **27/27**, i.e. the defect really is invisible
> here. With the patch applied, `tessera_jit` rebuilds clean and the
> native-CPU-JIT / signature-guard / totality / boundary-discovery /
> native-required packet is **73/73**. An alternating A/B of the two
> shared libraries at n=512, three reps each, is **indistinguishable**
> (60.0 / 67.6 / 74.2 pre-fix vs 75.7 / 86.9 / 55.3 GFLOP/s post-fix):
> run-to-run variance on this box swamps the effect, so **no performance
> change is claimed in either direction** — an earlier single-shot pair
> looked like a 24% regression and did not survive repetition. **(2) With the env var set,
every non-matmul module failed to compile** (the transform's empty
matmul match hard-failed stage 1b — measured 114 packet failures). The
lane now engages only for matmul-bearing modules, transforms a CLONE,
and swaps it in only on full success — any transform failure falls
back to the always-correct scalar pipeline. Pinned by
`test_vectorize_env_does_not_break_non_matmul_modules`. **(3) The
lane's `libmlir_c_runner_utils` dlopen was process-fatal beside HIP:**
that library links the dynamic `libLLVM.so`, and loading it alongside
libtessera_jit's static LLVM made a later `dlopen("libamdhip64.so")`
(comgr embeds a third LLVM) segfault — the ROCm state-machine tests
died the moment they loaded HIP after a vectorized compile (staged
probe: compile/invoke/heap all clean, dlopen fatal; runner-utils + HIP
without tessera_jit coexist fine). Fixed by implementing `memrefCopy`
in-process and registering it via ORC `registerSymbols`; the engine no
longer dlopens anything, which also deletes the hardcoded Homebrew
path. Full packet green with the lane ON and OFF (incl. the ROCm
state-machine suite + HIP loads). Mac note: the un-gating affects
Darwin too (also LLVM 23) — the vectorized path there is UNPROVEN
until the Darwin compat test is re-run on the M1 Max (recorded in the
apple plan). Follow-on opportunity: throughput falls with size (106 →
44 GFLOP/s at 1024) — no cache-level blocking above the 8x16x16
register tiles yet.


Cross-backend sync `JIT-MATH-AUDIT-2026-08-23` — **x86 host-JIT boundary
hardening + math-correctness pass: CLOSED (AVX-512 host).** Two defects
fixed, both measured on this host. **(1) `jb.invoke` boundary now fails
closed on signature mismatch.** The identity-layout ABI bakes static
extents into the generated indexing math, so a wrong-shape buffer was
guaranteed out-of-bounds (reproduced: a 2-element array against a
`tensor<7xf32>` module corrupted the heap and aborted in
`malloc_consolidate`). `tessera_jit` now records each function's
tensor-level signature at compile time (new `tessera_jit_signature` ABI),
`tessera_jit_invoke` validates arity + static extents from the
descriptors (memory-safety backstop for any caller), and Python
`invoke()` validates arity/rank/extents/dtype with actionable errors;
dynamic extents stay unconstrained.
`tests/unit/test_jit_invoke_signature_guard.py` (12 tests) pins both
layers, including the heap-corruption repro. **(2) The pipeline's
blanket `fastmath<fast>` stamp is narrowed to `reassoc|contract`.**
`fast` (nnan|ninf|nsz|arcp|afn) applied to every float add/sub/mul/div/neg
in the module: `arcp` measurably rewrote elementwise x/y into x*(1/y)
(1-ulp divergence vs correctly-rounded division), and nnan/ninf made
NaN/Inf inputs (e.g. -inf attention-mask biases) poison — latent on this
toolchain, legal to break. With reassoc|contract, division is bit-exact
vs numpy, NaN/Inf propagate bit-exactly through all four ops, and GEMM
throughput is retained (n=256 scalar lane: fast 3.34, narrowed 3.45,
no-fastmath control 2.7-2.8 GFLOP/s; n=512 was not comparable — WSL2
process-level timing variance swamped it). New regression tests pin
NaN/Inf propagation and correctly-rounded division in
`test_jit_tensor_elementwise_totality.py`. Full packet after both
changes: 219 passed (Darwin-only lanes skipped). Cross-target note: the
gfx1151 binary lane's signed-zero tie semantics deliberately differ —
see the rocm entry under this key.


Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; x86 outcome: parity validated, widened
family included (AVX-512 host, 2026-08-23).** `tessera_jit` rebuilt on the
Strix Halo box against the module-scope `convert-elementwise-to-linalg` +
residual legality gate; the widened packet passed:
`test_jit_tensor_elementwise_totality.py` **22/22**, native CPU JIT +
production phase 1/3 lanes **182 passed** (all 160 skips are Darwin-only
Apple/MTL4 lanes, correctly unevaluable on this host), and the paired
state machines re-ran green (x86 3/3 native; the sibling gfx1151 rows
5/5 exact-device, recorded in the rocm plan).  The rerun surfaced one
defect — in the test, not the lane: the totality suite's signed-zero
min/max assertions used numpy as oracle, whose ±0 tie resolution is
host-dependent (SSE returns the second operand; NEON orders the zeros),
so the assertion encoded the test host's ISA and failed 4 cases here
while passing on M1 Max.  Probed on this host, the JIT is
contract-correct per the arith ODS: `maximumf`/`minimumf` order signed
zeros (ties → +0.0 / −0.0), and `maxnumf`/`minnumf` ties are "either of
them".  The test now keys signed-zero expectations off the arith
contract (num-variant tie signs unasserted, as the contract specifies
neither); NaN propagation and finite-value checks keep the numpy oracle,
which is host-stable for those.


Cross-backend sync `APPLE-MINMAX-1-2026-08-23` — **Apple closed the
contract on its own hardware; x86 outcome: no change required, one open
question inherited.** The M1 Max audit found MPSGraph's max/min
non-conforming on both NaN and the ±0 tie and fixed it with the *same*
bitwise AND/OR tie blend this backend's AVX-512 vector body uses, so the
two agree by construction rather than coincidence. Apple has no
Adafactor device kernel and no `max(stat, eps)` floor anywhere, so the
NaN-laundering half of `JIT-MATH-AUDIT-2026-08-23` had no Apple sibling.
Metal evidence transfers nothing here. **Inherited open item:** the Apple
audit measured `relu(NaN) = 0` on device against a `np.maximum(0.0, x)`
reference that returns NaN, and deliberately did not fix it — the x86
relu lane is equally unaudited for this. Decide the `relu` NaN contract
fleet-wide before fixing any single backend.

Cross-backend sync `IEEE-MINMAX-CONTRACT-2026-08-23` — **x86 outcome:
VALIDATED (AVX-512 host).** The fleet-wide IEEE-754-2019 ±0-tie
decision (rocm plan owns the key) lands here in three routes: the
AVX-512 binary shim (`avx512_binary_f32.cpp` — scalar tail was
`a > b ? a : b`, vector body `vmaxps`, both second-operand-on-tie; now
signbit-select / bitwise AND-OR tie blend), and the two
backend-neutral numpy routes the codex review caught as still
route-dependent: the eager op namespace (`__init__.py` maximum/minimum)
and the Apple-lane numpy fallback (`_apple_gpu_binary_numpy`) now share
one reference implementation (`tessera/_ieee_minmax.py`, Decision #31)
instead of delegating to np.maximum/np.minimum, whose tie sign is
host-ISA-dependent (SSE second operand, NEON IEEE). Pinned by
`test_x86_binary_max_min_signed_zero_ties_are_ieee_ordered` (n=19:
vector body + scalar tail) and the route-consistency tests in
`test_ieee_minmax_reference.py`.


Cross-backend sync `JIT-MATH-AUDIT-FIXES-2026-08-23` — **x86 outcome:
VALIDATED (AVX-512 host).** The Adafactor NaN-floor fix (rocm plan owns
the key): the AVX-512 optimizer shim floored second-moment statistics
with `std::fmax` (NaN-suppressing), laundering a NaN statistic into eps
— identical to the ROCm maxnumf defect. All 13 floors now go through a
NaN-propagating helper; exact-host test
(`test_adafactor_factored_nan_gradient_propagates_like_reference`)
proves a NaN gradient poisons the full row+col exactly as the optim.py
reference does. The softmax maxnumf change is ROCm-kernel-only — no
x86 surface compiles through it.


Cross-backend sync `W4-SM-ROCM-2026-08-21` — **W4-PRODUCT-1 x86 outcome:
VALIDATED (AVX-512 host, 2026-08-21).** The sibling row landed: the same
paired `bounded_state_machine_v1` functions (forward + recompute_all
backward, both entry paths of the two-entry irreducible SCC, plus the
per-element cmpf→select machine) compile through `tessera_jit`'s
MLIR→LLVM→ORC chain and execute natively — digest/residual-policy bound,
native `cf.assert` bound trap (stronger than ROCm's host-checked STATUS),
proof-of-execution counter (`tests/unit/test_x86_state_machine_exec.py`).
One shared fix rode along: `convert-elementwise-to-linalg` joined the
tessera_jit pipeline (tensor-typed arith from the paired cotangent
accumulation had no bufferization interface). Correctness-only rows;
clean Zen 5 timing remains open per W4.3. Original follow-up text: gfx1151 landed the
first irreducible-state-machine execution rows (see the rocm entry). The
W4-PRODUCT-1 acceptance names native x86 rows too: an AVX-512-host
consumer for the same paired `bounded_state_machine_v1` functions (the
per-thread scalarization model ports directly; the x86 lane can execute
the structurized SCF via the existing compiled-kernel path or an
LLVM-lowered CPU kernel) plus the same digest-binding and host-enforced
bound check. Same box as ROCm — schedulable in a follow-up session.


Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; x86 outcome: parity validated (AVX-512 host).**
Same contract change as the rocm entry (lgamma/digamma datum-derived over
the polygamma tower; rmsnorm γ operand). Validated on the AVX-512 primary
box: all six `test_x86_*_loss_compiled.py` backward lanes green against
the switched registry (part of the 95/96 device-lane run recorded in the
rocm entry).


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; x86 outcome: parity validated (AVX-512 host).** Same
contract change as the rocm entry (structured trio jet-derived; six ops
datum-grown and switched). Validated on the AVX-512 primary box: 169 tests
across the six `test_x86_*_loss_compiled.py` backward lanes green against
the switched registry. AMX half not applicable (no fleet hardware).


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; x86 outcome: parity validated (AVX-512 host).** PR #600
retires the ODE-family pointwise hand rules behind the `DerivativeContract`
datum (dtype-preserving reference rules; unified log/sqrt boundary guard —
see the rocm entry for the full contract statement). x86 impact: the
compiled-loss backward lanes compare against this reference; validated on the
AVX-512 primary box — 169 tests across
`test_x86_{loss,binary_loss,class_loss,metric_loss,ebm_loss,rl_loss}_compiled.py`
green against the retired registry. The AMX half remains not applicable (no
fleet hardware; capability-gated as before).


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; x86 outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
`_apple_gpu_dispatch._prebuilt_candidate` decided whether a prebuilt dylib was
current by `ctypes.CDLL`-ing it and probing sentinel symbols. Loading is not a
read-only probe: it registers the runtime's Objective-C classes process-wide,
and skipping the candidate afterwards does not unregister them (the ObjC
runtime pins an image that has defined classes). A stale candidate therefore
stayed resident and the from-source dylib compiled next registered the same
classes again -- two images of one runtime, each with its own copy of every
file-static, including the thread_local last-error channel that
`_apple_gpu_run_checked` reads to decide whether a kernel failed. Staleness is
now read from the file's symbol table via `nm`, so a stale candidate is never
loaded; an undecidable probe (no `nm`) keeps the previous load-and-probe
behaviour rather than rejecting a library it cannot fault.
x86 impact: none. The AVX-512 lane loads `libtessera_x86_elementwise.so`
through its own path, not through `_apple_gpu_dispatch`. No Zen 5 retest
required and no device evidence is produced or claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; x86 outcome: not applicable today, but one follow-up recorded.**
The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
GPU elementwise-binary lane. `apple_gpu_runtime_stub.cpp` — compiled on every
NON-Darwin host so the C symbol exists — implemented opcodes 0-8 and its
`default:` arm assigned `out[i] = x`, so `mod`(9), `floor_div`(10), the six
comparisons(11-16), and the logical/bitwise ops(17-22) returned the LEFT operand
instead of computing. Because the symbol exists,
`_apple_gpu_dispatch_mpsgraph_binary` takes the kernel branch rather than its
numpy fallback, so those values came back as if computed, with no diagnostic
(Decision #21). Fixed by implementing opcodes 9-22 to match
`mpsg_binary_node` and the declared host reference
`runtime._apple_gpu_binary_numpy`, and by rejecting an unknown opcode through
the stub's last-error channel (new kind 3) so it routes to the host fallback
instead of returning a plausible buffer.
x86 impact: none from this change. The AVX-512 elementwise binary lane is
separate code (`_execute_x86_compiled_binary` →
`tessera_x86_avx512_binary_f32` / `tessera_x86_reference_binary_f32` in
`src/compiler/codegen/tessera_x86_backend/src/kernels/avx512_binary_f32.cpp`)
and does not call the Apple symbol.

**Follow-up recorded (not fixed here):** that kernel carries the SAME
silent-fallthrough shape — `scalar_binary` ends `default: return a;` and the
vector loop ends `default: y = a; break;`, both returning operand A for an
unrecognised kind. It is **latent, not live**: the kernel implements kinds 0-7
(`kSub`/`kDiv`/`kMax`/`kMin`/`kAdd`/`kMul`/`kMod`/`kFloorDiv`) and
`_X86_BINARY_OPS` maps exactly onto that range, so no reachable call hits the
default today. It becomes live the moment an opcode is added on one side only —
which is precisely how the Apple stub defect arose. Fixing it belongs in an x86
change with AVX-512 evidence from the Zen 5 box; no Zen 5 retest is required for
the present change and no device evidence is produced or claimed.
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; x86 outcome: not applicable today; parity by construction.**
The zero-function-candidate slice (PR #590) changes `JitFn`'s call ABI recovery
and adds one diagnostic code. A `@jit` function whose AST lowering produced no
function raised a bare `IndexError` from `_establish_tracer_authority`, and the
same absence left `_call_arg_names`/`_constraint_ir_args` empty — silently
mis-binding keyword calls and skipping call-time constraint re-checking. The ABI
is now derived from the Python signature via the shared
`graph_ir.ir_args_from_signature` (Decisions #30/#31), and the apple_gpu tracer
lane lifts foreign interpreter exceptions into the new registered
`JIT_APPLE_GPU_TRACE_FAILED` code (Decision #21). `TesseraTraceError` passes
through unwrapped.
x86 impact: none today, for the same reason as ROCm — a non-apple_gpu target
re-raises at decoration instead of deferring to the tracer. No Zen 5 retest
required and no device evidence is produced or claimed.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; x86 outcome: not applicable today, fails closed by
construction.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
x86 impact: none. `runtime._execute_x86_compiled_binary` binds both operands
positionally and raises `"binary math requires two operands (a, b)"` when fewer
are present, so the lifted-scalar form cannot reach the AVX-512 elementwise
kernel. No Zen 5 retest required and no device evidence is produced or claimed.
Follow-up: `_X86_BINARY_OPS` covers sub/div/mod/floor_div, so if that lane later
accepts a scalar kwarg form it must honor `scalar_side` first.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; X86 outcome: parity validated, no AVX-512 evidence changed.** The AD-LAW series (PR #588)
closes the swallowed-kwarg class registry-wide and adds the AD-WEIL-1 algebra
substrate plus all six executable laws. Reference-rule changes in this slice:
`stft`/`spectral_conv` JVPs **deleted** in favour of derivation from the
forward (both are bilinear, so every configuration key is honored by
construction); `jvp_istft` rewritten to honor axis/center/length/onesided/norm
while preserving its window quotient; `dequantize_nvfp4` fixed to accept the
per-block scale array its canonical forward takes (both modes previously
crashed); `jvp_lgamma`/`jvp_digamma` replaced (a dead zero-returning stub and
an identity placeholder); `jvp_cast` fixed for canonical dtype strings; the
shared polygamma helpers given reflection formulas (previously an O(n) loop
that hung on valid negative input — a live defect in the REVERSE path too).
x86 impact: the AVX-512 backward packages (incl. the selective_ssm `bwd_hardware_proven` row) compare against these oracles. Same split as ROCm — forward-mode references moved, VJP-side values did not. No Zen 5 retest required; no device evidence produced or claimed. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; x86 outcome: parity validated, no AVX-512 evidence
changed.** AD-LAW-1 (PR #584) adds law oracles (adjoint `⟨Jv,u⟩ = ⟨v,Jᵀu⟩` +
canonical-forward chain check) over the shared numpy reference JVP/VJP
registries — the oracle lane the AVX-512 backward packages (incl. the
selective_ssm `bwd_hardware_proven` row) differentially compare against —
plus the byte-gated `autodiff_law_audit` dashboard. Reference-rule fixes in
the same PR: `jvp_rmsnorm` eps default (1e-6 → the forward's 1e-5) and
`jvp_clamp` swallowing canonical `min`/`max`. x86 impact: norm bindings pass
`eps` explicitly and VJP-side defaults did not move, so no Zen 5 retest is
required; no device evidence is produced or claimed by this gate. Open
shared follow-up: 20 pinned swallowed-kwarg findings
(`test_autodiff_laws.py`) await triage. *Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for `clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft `norm` handling (√n-wrong under `norm="ortho"`); five entries benign-classified with recorded reasons; the open set is now the stft/istft/spectral_conv and quantize families only, riding their owning family reviews. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan..

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; AVX-512 evidence unchanged.** Dynamic region state now uses
bounded per-slot data tapes plus recorded logical-shape tapes. Polynomial shape
guards require complete concrete witnesses and do not enter Presburger proofs.
Variadic branch state reaches the structured CFG carrier. Only compiler-owned
extent assertions are replay-safe; mutation/RNG/I/O/ordered collectives remain
fail closed. A clean Zen 5 irreducible/dynamic-state packet is still required.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared parity
validated; AVX-512 evidence unchanged.** Relational Tile legality now has one
staged production authority, while old CLI passes are wrappers. Pure static
annotations abstract-trace before AST compatibility capture. NVIDIA
Lion/DeltaNet dispatch moved into family plugins; x86 keeps its independent
AVX-512 state-lineage packages and inherits no CUDA evidence.

Cross-backend sync `W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **shared and x86
plugin parity validated; no Zen 5 timing claim.** Bare Tile fragments and unknown
semantic selectors fail closed, whole-program memory activity is conservative,
and regression/BCE/class-loss VJPs use explicit x86 plugins. The x86 no-async
path is unchanged. Remaining public-module/frontend decomposition and clean
AVX-512 evidence are independent follow-ups.

Cross-backend sync `W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared
carrier and bounded AVX-512 correctness consumer landed.** SAVE/HYBRID selection now
retains its exact checkpoint, CFG, and residual identity through Graph→SCF.
Shared paired AD now consumes dynamic branch-local residual extents through
zero-extent inactive sentinels, bounded SAVE and sparse HYBRID `while` state
tapes, and bounded-dynamic counted-loop tapes. Source-CFG SCC analysis now
distinguishes acyclic, reducible, and true multi-entry irreducible graphs.
Bounded pure native graphs lower to a typed program-counter state machine with
nested canonical structured bodies and mixed control/tensor state. Saved
dynamic slots now require total data/shape-tape envelopes; unbounded,
unsupported-region, and unrecorded effectful forms remain fail closed. The committed Zen 5
WSL packet binds paired-IR/CFG/residual digests and executes AVX-512 children
without Graph re-entry or predicate replay. It is correctness evidence only;
clean bare-metal timing remains required for performance promotion. An exact
Zen 5 irreducible-state-machine row remains required before physical x86
execution of that new form is claimed.

Cross-backend sync `E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **bounded
AVX-512 SGD and Momentum/Nesterov reverse authority complete.** The plugins
bind one-execution tracer proof and functional state lineage through typed
`schedule.optimizer_vjp` → `tile.training_kernel` packages before the existing
AVX-512 calls. Adam/AdamW remain fail closed on x86 because no physical reverse
consumer exists; clean Zen 5 timing remains a separate evidence gate.

Cross-backend sync `E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **bounded AVX-512
Adafactor and sequence-mixer reverse authority complete.** Explicit plugins
now own full/factored Adafactor and causal gated/Kimi/modified DeltaNet
backward; the compiler route binds tracer identity, one-execution proof,
state/workspace lineage, Schedule, and exact Tile identity without passing
Graph metadata to the runtime. Existing AVX-512 numerical tests remain valid;
clean Zen 5 selector-grade timing is a separate evidence gate and no AMX claim
is implied.

Cross-backend sync `E2E-REAL-6D-LION-VJP-2026-08-17` — **bounded AVX-512
Lion reverse authority closed.** The flat functional Lion Graph ABI now has
exactly two results, `(new_param, new_moment)`, and concrete tracing executes
that source once. A structural non-reexecuting certificate binds the retained
AST candidate to tracer Graph IR; the family plugin then builds the existing
`schedule.lion_vjp` → `tile.training_kernel` state-lineage package. Runtime
receives no Graph operation metadata and validates the frontend certificate,
state lineage, Schedule identity, and exact Tile digest before the AVX-512
call. Public numerical, one-execution, and tamper tests pass. Clean Zen 5
performance evidence remains independent.

Cross-backend sync `E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **bounded public
AVX-512 canonical rank-4 attention reverse authority closed.** `flash_attn`,
GQA, and MQA now enter one native-VJP family plugin from tracer-produced Graph
IR. The plugin builds the exact `schedule.attention_backward` →
`tile.attention_backward_kernel` package, preserves x86’s saved-LSE identity,
and binds parent Graph, Schedule, Tile, native-image, and aggregate digests
before runtime. `JitFn` no longer constructs the x86 attention package. Public
ragged GQA numerical proof and lineage/tamper negatives pass for the explicit
zero-dropout differential envelope; active dropout remains fail-closed until
keyed replay can be certified without duplicate effects. The rank-3
`multi_head_attention` wrapper remains a compatibility path until its
reshape/transpose product is explicit; clean Zen 5 performance evidence is
independent and remains open.

Cross-backend sync `E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **bounded AVX-512
compound spectral reverse execution closed.** `spectral_filter` and
unbroadcast full `spectral_conv` now enter one native-VJP family plugin from
tracer-produced Graph IR. Source, Schedule, and Tile digests are validated
before the existing AVX-512 adjoint ABI executes; public numerical and
fail-closed tests cover filter, convolution, unsupported broadcasting, and
tampered lineage. Broader axes/dtypes and STFT/ISTFT backward remain separate
work, and clean Zen 5 performance evidence remains open.

Cross-backend sync `GFX1151-CALIB-BAREMETAL-2026-08-16` — **shared calibration
authority parity validated; no x86 evidence transfers.** `target_perf` now
rejects explicitly provisional and WSL-hosted corpora from its measured
selector registry while exposing a non-mutating pruning reader. AVX-512 code
and selectors are unchanged; clean Zen 5 perf/IBS evidence remains the x86
promotion authority.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared layout and
SO-1 parity validated; SO-2 no-async proof explicit.** The versioned C++
layout ABI is host-free and the GQA-fold consumer is common. Schedule Object
digests now bind R3 actions, edges, roles, residency, and resource vectors. x86
has no asynchronous-copy/mbarrier lane, so every typed family plugin declares
`async_role_policy = no_async_noop`; the SO-2 role carrier remains a verified
no-op for AVX-512 lowering. No x86 index template or raster output is
changed until L4, and no Zen 5/AMX evidence transfers.
Nested Schedule resource metadata is now frozen before hashing, and dynamic
rearrange/GQA-fold inference preserves ranked `?` dimensions. Neither change
selects a new x86 layout or raster policy.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **deterministic parallel VJP
implemented; bare-metal selector packet remains open.** AVX-512 attention
backward partitions query rows into deterministic contiguous worker ranges,
uses bounded private dK/dV partials, and reduces workers in fixed order. MHA,
GQA/MQA, saved/recomputed LSE, bias, window, softcap, ragged shapes, and repeated
bit-determinism pass. An equal-flags Zen 5 WSL comparison measured 1.85x median
speedup at B2/Hq8/Hkv2/S128/D64 with 1.67e-6 maximum absolute delta. This is
regression evidence only; clean bare-metal timing remains required for a
selector-grade claim.
Allocation and thread-construction failures are contained inside the C ABI:
started workers are joined, outputs reset, and the deterministic serial path
replayed, so no exception or joinable-thread destructor can terminate a caller.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared exact semantic
authority landed; AVX-512 physical follow-up required.** The compiler now owns
a typed constant-coefficient PDE carrier, exact-rational principal-symbol
classification, and a fail-closed centered-FTCS diagonal-diffusion certificate
with non-unit spacing. This adds no x86 Target record, vector package,
transport, or clean Zen 5 packet.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared SSA/product
foundation landed; native process transport remains open.** Planned bulk
reshards now become real Graph→Schedule→Tile SSA carrying digest, subgroup,
region, and deterministic all-to-all matching-round identity. Exact compiler
forward-over-reverse HVP is also live. This adds no MPI/OFI/SHMEM launcher,
native AVX-512 HVP package, or multi-rank/Zen 5 performance packet. Local-shard
result typing and process-transport evidence remain architecture-owned gates.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **AVX-512
normalization ownership migrated.** RMSNorm and LayerNorm reverse execution
now enters one native-VJP family plugin with explicit Graph/Schedule/Tile/x86
Target consumers. `JitFn` binds inputs and records the result but no longer
constructs the normalization backward package. Existing Zen 5 correctness
evidence remains authoritative; this synchronization adds no performance or
AMX claim. Other x86 backward families remain compatibility paths.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **parity assessed; no x86
physical change required.** The shared change is confined to AMD architecture,
datatype-role, and matrix-instruction selection. It changes no Graph dtype
spelling, public operation, Schedule contract, AVX-512/VNNI ABI, or AMX gate;
x86 evidence remains independent.

The shared `OP-DTYPE-FLOW-1` generator now audits every x86 operator/storage
pair from frontend through physical manifests. Accumulator-compatible dtypes
derived from target-wide AVX-512 legality remain `legal_only`; only explicit
x86 per-operation rows can claim a physical consumer or execution evidence.

## P0 root-caused — x86 dialect load was a build-flag leak, not an IR defect

`X86-DIALECT-LOAD-CRASH-2026-08-12` was **reopened by CI on 2026-08-15** (run
31893492411) after being closed on the Strix Halo host, and is now root-caused.
Both observations were correct; the host is what differed.

The dialect and its `TileType` registration were never at fault. The x86 kernel
project applied its detected AVX-512/AMX flags with **`add_compile_options`**,
which is *directory* scoped, and `add_subdirectory(lib/IR)` sat below that call
— so the hardware-free Target IR dialect was compiled with `-mavx512f
-mavx512bw … -mamx-tile …`. The compiler was then entitled to emit an AVX-512
encoding into dialect registration itself (confirmed by disassembly:
`vpbroadcastq %rgpr, %xmm`, a GPR-source broadcast that exists only under
AVX-512). On Zen 5 that instruction is legal, so the Strix Halo build ran
clean and the P0 read as closed. On the GitHub runner it is not, and
`tessera-opt` died the first time it touched the dialect.

The tell was in the exit status all along: **all 14 fixtures failed with signal
4 (SIGILL) at one identical address**, not SIGSEGV. An illegal instruction is a
build-configuration fact, not a memory-safety one — a crash *inside*
`Dialect::addType<TileType>()` was the first code from that translation unit to
execute, not the code that was wrong.

Fix: the flags are collected into `TESSERA_X86_ARCH_FLAGS` and applied with
`target_compile_options` to this directory's kernel targets only, enumerated via
`BUILDSYSTEM_TARGETS` so a later kernel target is covered automatically while
subdirectories are structurally excluded. `lib/IR/CMakeLists.txt` additionally
**fails configure** if any host-specific ISA flag is in scope, so the next
occurrence is a build error naming the cause rather than a runtime SIGILL that
reads as a corrupt MLIR install (Decision #21a, fail closed).

Standing lesson for this backend: a host that *has* the ISA cannot falsify a
host-portability claim. Decision #19's "lit-testable on any host" is only
evidenced by a host without AVX-512 — which, in the current fleet, means CI.

The named `x86_dialect_load.mlir` regression isolates dialect initialization
from operation lowering, so a future `TileType` registration failure cannot hide
behind the larger lit suite. Acceptance evidence for this fix is the green
`Validate / lit` lane on a non-AVX-512 runner, not a local rebuild.

## P0 fixed 2026-08-16 — `TileToX86Pass` declares its dialect dependency

Found while verifying the fix above, on an **assertions-enabled** LLVM/MLIR 23.
Distinct defect, same code path, and **CI structurally cannot see it**.

`src/transforms/lib/TileToX86Pass.cpp:1045` calls
`getContext().getOrLoadDialect("tessera_x86")` from inside `runOnOperation()`,
by name, to avoid linking the optional backend library. MLIR forbids loading a
dialect during pass execution:

```
LLVM ERROR: Loading a dialect (tessera_x86) while in a multi-threaded execution
context (maybe the PassManager): this can indicate a missing `dependentDialects`
in a pass for example.
```

That guard is `#ifndef NDEBUG`. The CI lit lane builds Release against apt
LLVM/MLIR 23 (assertions off), so the check is compiled out and the load becomes
silent undefined behaviour that happens to survive. On an assertions-enabled
build it is a hard error that fails **12 of the 15** x86 fixtures — i.e. the
whole x86 lit suite is currently unrunnable on a normal assert-enabled developer
toolchain, which is the second reason this backend's Decision #19 claim has been
easy to overstate.

The layering decision is now explicit. The hardware-free `TesseraX86IR` target
is created before `TesseraPasses` when x86 is enabled; the pass links that small
IR library, declares `TesseraX86Dialect` from `getDependentDialects()`, and is
compiled with `TESSERA_HAS_X86_TARGET_IR`. Native AVX-512/AMX kernel targets
remain later and do not enter the shared pass dependency. Builds without x86
retain the pass registration but fail before rewriting with an instruction to
enable `TESSERA_BUILD_X86_BACKEND`; there is no permissive unregistered marker
path. The forbidden `getOrLoadDialect("tessera_x86")` call is deleted.

The final composite-pipeline failure was a second namespace-authority defect:
`DistributionLoweringPass` defined a permissive private `schedule` dialect in
parallel with the ODS Schedule dialect. Composite x86 pipelines attempted to
register both classes and aborted. The canonical dialect is now isolated in
`TesseraScheduleIR`, both the programming-model and transform libraries link
that one target, and the private dialect is deleted.

Host LLVM/MLIR 23 validation rebuilt both x86-present and x86-absent drivers.
The positive Target fixture, negative verifier fixture, direct dialect-load
fixture, and executable family pipeline pass 4/4; the absent driver returns the
named fail-closed diagnostic before mutation. The installed WSL LLVM reports
assertions `OFF`, so an assertions-enabled toolchain rerun remains required as
the final external evidence packet; the source regression test independently
forbids reintroducing a runtime dialect load.

The production HIP build was also rebuilt with both
`TESSERA_BUILD_ROCM_BACKEND=ON` and `TESSERA_BUILD_X86_BACKEND=ON`. Its feature
ledger reports both `rocm-backend` and `x86-target-ir`; the complete shared lit
suite reports 329 enabled passes, 52 configuration-gated tests, and zero
failures across 381 discovered tests; the x86 verifier pair plus all twelve explicitly
gated ROCm fixtures execute 14/14 rather than becoming unsupported. This is
host-free compiler proof, not clean Zen 5 performance or AMX evidence.

---

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **x86 and combined
host proof closed; host-free lit coverage restored, no Zen 5 timing evidence
added.** The
`Validate / lit` lane was dead from 2026-08-11 to 2026-08-12 (pytest collection
aborted on a missing `ml_dtypes`, fixed in #554). The first green-collection run
discovered 367 fixtures, passed 318, and failed 27 — all from one cause: the
lane's `tessera-opt` was configured with `TESSERA_BUILD_APPLE_BACKEND=ON` only,
so `TesseraX86IR` never existed and `tessera_x86` went unregistered
(``error: Dialect `tessera_x86' not found for custom op
'tessera_x86.amx_tile_zero'``). x86 Target IR has therefore had **zero CI lit
coverage since the W0.10 dialect landed (2026-08-02)** — Decision #19's
"lit-testable hardware-free layer" was asserted but never exercised on CI.

This PR adds `-DTESSERA_BUILD_X86_BACKEND=ON` to that configure, recovering:
`phase2/x86_target_ir{,_invalid}.mlir` (including the negative fixture Decision
#19 requires), `phase2/tile_to_x86.mlir`, `phase2/tile_x86{,_base}_e2e.mlir`,
`phase2/x86_executable_pipeline.mlir`, `phase2/x86_kv_cache_lowering.mlir`,
`phase2/e2e_matmul_scheduled_x86_consumer.mlir`, and the x86 half of
`phase_f4/{es_low_rank_correction,spectral_backward}_x86_native.mlir`.

Scope limits, stated explicitly: this is **host-free IR/registration coverage
only**. It adds no AVX-512 physical consumer, no Zen 5 timing packet, and no
AMX evidence (no AMX hardware exists in the fleet; that lane stays
capability-gated). The flag does not gate on `CMAKE_SYSTEM_PROCESSOR`
(`src/CMakeLists.txt:44`), so it builds the dialect on the runner without
requiring AMX/AVX-512 at build time. Green-on-CI is the acceptance evidence;
if any recovered fixture fails, it is a real x86 defect that this lane was
previously hiding and it outranks the CI change.

**Sequencing note — x86 and ROCm share one host, so schedule them together.**
Strix Halo is Zen 5 + gfx1151 in a single box: it is the fleet's x86 host and
its ROCm host at once. Because `TESSERA_ENABLE_HIP=ON` avoids the `:95` lean
condition, one configure there carries core + ROCm + x86 in a single
`tessera-opt` — the only fleet configuration that runs both fixture families in
one `lit` invocation. Whoever picks up the ROCm side of
`CI-LIT-BACKEND-DIALECTS-2026-08-12` should add `-DTESSERA_BUILD_X86_BACKEND=ON`
and close the x86 host-side verification in the same session rather than
scheduling a second visit to the same machine; the procedure lives in
`docs/audit/backend/rocm/todo.md` under this sync key. Two caveats that do not
change: the box runs under WSL2, so host-wall timing there stays
**regression-only** and does not satisfy the clean AVX-512 timing packet still
open above; and Zen 5 has AVX-512 but **no AMX**, so nothing on this host can
retire the AMX lane.
Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no x86 package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **CPU gradient
correctness landed; broader AVX-512 follow-up required.** The executable CPU
gradient ABI consumes explicit axis spacing and passes non-unit-grid numerical
tests. General stencil apply, boundary, and halo primitives remain
artifact-only; x86 still owns their typed consumers and a clean Zen 5 packet.
No gfx1151 evidence transfers.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **follow-up required.**
The shared Block AttnRes numerical-policy, balanced-partition, VJP, and
softmax-merge oracle contracts apply to x86. The exact static all-f32 Graph
contract now lowers through the same content-addressed
`schedule.depth_attention` → `tile.depth_attention_kernel` boundary, with an
x86-owned workgroup policy included in its digest. This adds no AVX-512 Target
consumer or Zen 5 evidence. After the gfx1151 contract is proven, x86 must own
its vectorized stats-attention/merge package and independent correctness and
performance packet; ROCm schedules and evidence do not transfer.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **MiniMax MSA now owns a
digest-bound AVX-512 package; clean Zen 5 evidence remains open.** The package
binds Graph, Schedule, Tile, and Target identities and executes the declared
`x86_msa_compiled` lane without legacy Graph `ops` metadata or runtime
resynthesis. Structural and differential execution tests pass in WSL, but this
host cannot provide the required clean AVX-512 timing packet. DeepSeek MLA/DSA
remain separate family migrations.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **shared physical-byte
weight ABI landed; AVX-512/VNNI consumption remains follow-up required.** The
content-addressed carrier preserves INT4/FP8 checkpoint bytes and separate fp32
scales while prohibiting full-weight materialization. This slice adds no x86
packed-byte GEMM package and transfers no gfx1151 evidence. x86 still needs an
architecture-owned AVX-512/VNNI physical consumer and a clean Zen 5 packet.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared typed shape and
placement analyses landed; AVX-512/MPI physical follow-up remains open.** The
C++ compiler consumes coefficient-vector integer-affine and exact
modular/divisibility constraints with MLIR Presburger analysis. Shared placement
propagation now distinguishes replicated, tiled, partial-reduction, and unknown
states; catalog pointwise/reduction/collective rules feed an explicit fail-closed
reshard planner. Lowered `control_scan` has shared JVP/VJP products under
`recompute_all`; saved checkpoint policies remain rejected. No MPI/OFI/SHMEM
reshard materialization, native region product, or clean Zen 5 packet was added.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; AVX-512 physical follow-up required.**
The tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No native region-product package,
inferred-edge producer wiring, clean Zen 5 calibration, or selection claim was
added.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **shared reduction and
normalization product authority landed; exact AVX-512 rerun is blocked.**
Reduction now requires exact `schedule.reduce → tile.reduce_kernel → native
descriptor` children, and normalization owns a content-addressed composite
Schedule/Tile action program. Mandatory tracer/AST differential gates cover
pure native JVPs and VJPs. This WSL process cannot load the production AVX-512
shared image, so it cannot refresh the Zen 5 packet; no ROCm result transfers.
A clean bare-metal Zen 5 run with that image and calibrated timing remains the
architecture-owned gate.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; AVX-512 physical evidence is
unchanged.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare Graph/Schedule/Tile/AVX-512 disposition and own parent
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. The existing JVP children remain the
physical implementation; this slice adds no selector promotion. Carrying the
generated edges through x86 family pipelines and a clean Zen 5 calibrated
packet remain follow-up; the no-async path stays valid.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **bounded general-
residual and exact ISTFT-window products execute on AVX-512.** A content-
addressed solver parent binds five native child packages and performs restarted
GMRES with a true-residual check. The compiler now derives all five packages
from a verified typed residual Graph. Pointwise, sum/mean, rank-2 matmul,
transpose, distinct parameter/solution spaces, bounded-dynamic dimensions,
mixed f16/bf16 storage with explicit f32 widening, and statically bounded
`control_for` are represented in the content-addressed child program. Pure
scalar predicates also lower data-dependent `if` and bounded `while` to
digest-bound compare/select SSA with deterministic recomputation in every
primal and product child; the nonlinear
`R=x*x+sin(x)-theta` proof executes without a dense Jacobian. The new
`tessera.istft_jvp` carrier reaches one AVX-512 package
that reuses packed inverse frames and differentiates both overlap-add numerator
and quadratic window-energy normalization. The committed 30-sample WSL packet
records cold/warm state, resources, device/toolchain identity, artifact
digests, and numerical error for nonlinear, reduction, reduced-storage matmul,
bounded-dynamic mixed-storage, data-dependent `if`, data-dependent `while`, and
ISTFT-window products (maximum error 2.33e-6). Timing remains
regression-only because this is not a clean bare-metal run and has no
independent device clock. Odd/low-precision ISTFT products and the clean Zen 5
selector packet remain open.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **the first frontend/
family-plugin cohort is landing.** Native forward-product specialization now
binds tracer-produced canonical Graph IR and dispatches through explicit family
plugins outside `JitFn`. General solver contracts bind exact residual/JVP/VJP
identities and support exact matrix-free reference actions without numerical
re-entry. The six-case AVX-512 native forward-product cohort, including DCT-IV
with corrected logical-length normalization, passes on this Zen 5 WSL host.
General residual child-composite packages now execute and the typed residual
compiler derives all five children automatically for pointwise, reduction,
rank-2 matmul, bounded-dynamic/mixed-storage, and counted-region programs.
Architecture-owned packets for that expanded envelope and a clean bare-metal packet remain open. The AST lane stays available
for unmigrated families until their differential gates close.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP and
structured-region products landed; CPU process transport remains open.** Public
JVP/jacfwd no longer substitutes finite differences, and compiler forward mode
carries primal/tangent state through bounded SCF. The typed
`collective_permute` contract executes in the portable runtime, but it does not
constitute MPI/OFI/SHMEM execution. x86 still needs subgroup-aware process
launch plus a clean multi-rank Zen 5 correctness/performance packet.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; AVX-512 now has a
bounded child-composite proof.** Portable tracing now emits bounded SCF, and the paired compiler
differentiates effect-safe single-block `if`, counted `for`, and canonical
bounded `while` with implicit captures. General residual execution uses
restarted GMRES/CG policy and exposes convergence work; counted-region evidence
executes SAVE/RECOMPUTE/HYBRID cohorts. The monolithic diagonal-sqrt pilot is
retained, while the general parent now consumes compiler-derived, digest-bound
nonlinear product children. Typed reduction/matmul and statically counted-region
lowering is now shared software; pure scalar predicate-bearing residuals and
the expanded-family WSL correctness packet are now closed. Selected-checkpoint
lowering and a clean Zen 5 selector packet remain open; AMX is
unaffected.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; AVX-512 remains the proven no-async path.** Graph IR
now has one fail-closed, invalidatable shape/alias/liveness/memory-dependence/
activity analysis with C++ and Python query surfaces. Reverse AD and await
sinking consume it. No physical x86 schedule changes, and clean Zen 5 overlap
or performance evidence remains architecture-owned.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **affine normalization,
compound spectral, and matrix-free solver products execute on AVX-512.** The
multi-active product ABI carries named child outputs. Affine LayerNorm binds
data/scale/bias tangents to AVX-512 norm and binary lanes; spectral-filter JVP
executes both bilinear terms through the typed TSOL artifact; and the solver
artifact carries residual-JVP/non-transposed-solve lineage distinct from VJP.
All three pass numerical tests on the WSL-visible Zen 5 CPU. The native
multi-rank collective product requires a live NCCL hardware adapter and world
size >= 2; clean bare-metal Zen 5 timing and native process-transport evidence
remain open.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **the first native AVX-512
JVP package is executable; clean-host promotion remains open.** The
content-addressed parent binds compiler-emitted paired JVP IR to ordered native
child packages without Graph redispatch. Sum, non-affine RMSNorm, and packed
RFFT primal/tangent products pass independent formulas on this Zen 5 WSL host
(**3/3**). The packet records architecture and artifact lineage, but it is not
a clean bare-metal timing packet. Broader products and process-collective
hardware evidence remain architecture-owned.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; AVX-512 remains a no-async path.** Canonical Graph records now
carry effect, alias, mutation, and stochastic identity; Python and C++ consume
the same fail-closed facts and internal calls reach a fixed point. Await
sinking remains a proven x86 no-op. Native process transport and Zen 5 overlap
evidence remain x86-owned.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; native CPU transport remains open.** The
content-addressed plan binds chunk slices, per-expert capacity, two-live-frame
workspace limits, true-use dependencies, ordered collectives, and deterministic
combine order. R3 only prunes complete measured records; scalar clean-host
latency selects. Mock multi-rank execution is not MPI/OFI/SHMEM proof. x86
still needs native transport integration plus Zen 5 correctness/performance
evidence with stable affinity.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; no AVX-512 selection claim.** R3 validates
explicit dependencies and calibration identity, uses deterministic
critical-path/list scheduling, and composes compute/memory/communication lanes
with queue serialization. Exact small DAGs and proven lower-bound losers may be
pruned; every estimate is promotion-ineligible
and scalar measured latency remains authoritative. WSL timing and blocked PMU
access do not become calibration evidence; clean Zen 5 vectors remain required
before architecture-owned composition analysis or R4 transport overlap.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; Zen 5 evidence remains architecture-owned.**
Successful measured autotune rows may record compute time, dtype-correct bytes
moved, communication bytes, queue/resource identity, timing provenance, and
the measured-candidate digest. Analytical rows cannot claim the vector, and
scalar measured latency remains selector authority. The contract does not
upgrade WSL wall-clock or blocked PMU observations; clean-host AVX-512 timing,
affinity, and resource identity remain required before R3 composition use.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
async lineage landed; x86 is a proven no-op path.** Python Schedule→Tile no
longer emits internal `tessera.queue.*` markers, and the registered collective
await pass sinks only across operations proven memory-effect-free. Mutation,
RNG, aliases/casts, regions, and ordered collectives remain barriers. A direct
no-async function test records zero moved awaits; no AVX-512 package, ABI, or
performance claim changes. Future MPI/OFI/SHMEM overlap remains separately
architecture-owned.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **native x86 base RNG
transforms landed; clean Zen 5 packet remains required.** Explicit key/counter
Graph ops, estimator provenance, dropout replay, fixed-key EGGROLL JVP, and the
shared derivative proof matrix are compiler contracts. The x86 library now owns
uniform scaling, Box–Muller normal, and dropout masking instead of applying the
transforms in Python. Uniform words are bit exact and normal is one-f32-ULP
bounded. This local WSL host is not a clean Zen 5 evidence environment, so no
new AVX-512 performance or target-JVP promotion is claimed.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
AVX-512 execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No native
JVP package or Zen 5 evidence is claimed; target promotion remains fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; x86 physical consumption remains architecture-owned.** The
Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no AVX-512 package or Zen 5 evidence; x86
must lower and prove any native JVP package independently.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **the production
AVX-512 native packager now crosses one registered, closed-family executable
pipeline.** `tessera-x86-executable` validates the semantic family against
exactly one primary Tile carrier, binds `x86_64_avx512`, records the Tile producer,
registered `tessera_x86` Target consumer, and prebuilt native-image boundary,
then rejects surviving Tile operations or packages without both the typed
Target marker and native ABI call. Python package producers use
`X86ExecutablePipeline`; the former direct construction of
`--tessera-tile-to-x86=...` is gone from production packaging. The base x86
profile remains limited to its proven softmax/reduction subset, and AMX remains
access-gated. Spectral backward, solver IFT, scheduled matmul, attention, and
the broader AVX-512 ABI families all use the same boundary. Standalone legacy
pass fixtures remain only to test the underlying lowering pass, not as a
production configuration escape hatch.
Attention backward explicitly admits one forward-recompute companion carrier;
all other families remain single-carrier. This keeps LSE recompute identity in
the same content-addressed program without weakening family discrimination.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **the shared Graph,
Schedule, Tile, lineage, member-RNG-v1, and fp32 numeric-policy contract has
landed; rank-1 fp32 Zen 5 AVX-512 consumption is complete.** The typed x86
family lowers the exact architecture-bound Schedule→Tile artifact to
`tessera_x86_avx512_es_low_rank_correction_f32`; runtime execution preserves
the shared SplitMix64/Philox4x32/Box–Muller member identity, caches only
O(in+out) factors, and never materializes the perturbation matrix. The durable
packet `benchmarks/baselines/x86_zen5_es_low_rank_2026_08_09.json` records
correct aligned, ragged, and 16×32×1024→1024 cases (0.844 ms median for the
largest case) on Ryzen AI MAX+ 395. The separate integer lane remains
fail-closed: VNNI u8×s8→s32 cannot implement the Gaussian fp32 correction until
Graph/Schedule carry explicit quantization scales and saturating requantization.
W4 scalar-gather/member reconstruction passes mock-mesh proof; native x86
transport integration remains open.
Contract: `docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **shared
fail-closed artifact vocabulary adopted; not applicable to AVX-512.** Zero-CU
Copy Engine, GIN/RMA, and gfx1250 DDA are distinct RCCL lanes and cannot be
selected by the x86 runtime. The eventual MPI/OFI/SHMEM transport keeps its
own communicator, one-sided-memory, and evidence contracts.
The registered window and put/signal/wait Target operations are RCCL-gated;
they do not create an x86 RMA implementation or replace the future OFI/SHMEM
lane.
The launcher-neutral RCCL GIN harness is not an x86 transport. Only its explicit
rank/rendezvous and dual-clock evidence discipline is reusable by the future
MPI/OFI/SHMEM lane.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **shared
artifact vocabulary adopted; no x86 accelerator transport claim.** Target
collective artifacts now bind initiation, registration, ordering, capture
policy, backend/source identity, and capability evidence. NCCL/RCCL host calls
and AMD LSA/Copy Engine/DDA paths do not apply to AVX-512. The portable
CPU/software adapter remains the x86 path until a separately owned
MPI/OFI/SHMEM transport and multi-rank evidence are introduced. Shared
runtime/artifact capability-digest rejection applies to that future transport.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; process transport evidence open.** Active transform producers
now use the registered async `tessera_collective` dialect, explicit awaits, and
real SSA rewiring instead of unregistered marker strings. Runtime mesh-axis
validation fails closed when the adapter cannot implement a subgroup. x86
still needs a production multi-process transport and exact Zen 5 multi-rank
packet; AVX-512/AMX selectors and performance claims are unchanged.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **portable alias bridge
landed; native x86 transport unchanged.** Three public entries are compiler
placement/region contracts; `psum`/`pmean`/`pmax`/`pmin` and
`broadcast_to_axis` now resolve to the registered collective runtime instead
of remaining nine undifferentiated backend rows. `collective_permute` remains
an ordered point-to-point gap and fails closed. MPI/OFI/SHMEM binding and exact
multi-rank Zen 5 evidence remain open; AVX-512/AMX selectors are unchanged.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded AVX-512
physical pilot and packet landed.** The diagonal-sqrt residual lowers as one
content-addressed `schedule.solver_ift` → `tile.solver_ift_kernel` package and
executes residual, transposed diagonal matrix-free solve, and parameter adjoint
in the native AVX-512 image. The committed [Zen 5 3×257 f32 packet](../../../../benchmarks/baselines/x86_zen5_solver_ift_evidence.json)
reports 4.58e-7 maximum residual error, exact linear/adjoint outputs, 30 complete-
backward samples, and 3,084 retained bytes. General residuals and
iterative/Krylov solvers remain open and fail closed; AMX is unaffected.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
effect/control parity and native x86 consumption validated; multi-rank
transport open.** The native x86 spine passes 31 tests on the AVX-512 host;
three availability-gated cases skip. The loader now preserves
content-addressed base/AVX-512 image identity when the host Python lacks
`memfd_create` by using a unique unlinked temporary image. The four typed Tile
collectives now lower into a content-addressed portable Target queue and run
through the deterministic two-rank adapter. A production process transport
and exact multi-rank x86 proof remain open. No new performance or selector
promotion is made; AMX is unaffected.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; AVX-512/AMX packages unchanged.** Graph and canonical-
attention verifiers now inspect signed `IntegerAttr` values before enforcing
positive and non-negative bounds, so MLIR 23 unsigned value accessors cannot
admit negative schedules, cache windows, seeds, or control bounds. Direct
negative IR cases cover both dialects. No native ABI, schedule, selector,
package, or Zen 5/AMX evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **bounded AVX-512
backward packages and exact-host correctness landed; performance open.** The
content-addressed multi-output carrier now lowers complex-f32 spectral-filter
and unbroadcast last-axis full-f32 spectral-convolution adjoints to native x86
ABI calls without returning to Graph IR. Both gradient outputs pass direct
numerical checks on the local Zen 5 AVX-512 host; convolution preserves the
artifact-selected `backward`, `forward`, and `ortho` scale. Native STFT/ISTFT
backward, broader axes/dtypes/broadcasting, and a clean timing/resource packet
remain open and fail closed. Existing forward AVX-512 TSOL packages are
unchanged; AMX is not implicated.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR parity
validated; AVX-512 package unchanged.** Both compiler autodiff passes now
consume compiler-owned transposes for structural views, broadcast, and
operand-wise matmul. Paired CPU execution proves the emitted inverse view chain
and all matmul transpose-flag combinations. This transfers no AVX-512/AMX
schedule, selector, performance, or exact-device claim.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **portable
CPU and native AVX-512 proof separated; no x86 physical change.** The x86 row
now uses the `x86` manifest grain instead of accidentally reporting only five
portable-CPU runtime rows. Implementation-present entries remain distinct
from exact Zen 5 proof; AMX access state and selectors are unchanged.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **runtime and
packager discovery closed.** Both x86 entry points now honor the explicit
library override, then the common fail-closed `TESSERA_BUILD_DIR`, before
canonical defaults. The current clean LLVM/MLIR 23 tree loads the production
AVX-512 image and compiler together: the exact-host FFT suite passes 41/41.
The repaired comparison runner measures both normalized forward and inverse
C2C paths; a 64/256/1024, batch-4 smoke passes numerically. Its short WSL
host-wall samples are validation only, not a new performance promotion packet.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **Zen 5 package
evidence is now represented without changing a selector.** The standalone
dashboard generates its registry and compiler-layer counts, exact-target
manifest summary, and open queues. The audit no longer hides the verified
Adafactor Schedule→Tile/native package behind a single-GPU terminal override,
and the benchmark inventory now names the physical TSOL and Adafactor
harnesses. This is an audit correction only; AMX and clean bare-metal timing
gates remain unchanged.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **Zen 5 packed-real
lane retained.** The v3 FFT artifact distinguishes logical and physical
lengths and hashes its Hermitian policy. Supported even RFFT/IRFFT shapes now
enter architecture-owned native C ABI functions using an N/2 complex FFT;
odd or unevidenced factorisations retain an explicit full-complex fallback.
Cached Hermitian twiddles were required: recomputing sin/cos made the sum of
the parts slower than the old lane. After caching, WSL synchronized-host-wall
measurements at batch=32 show 1.33x/2.23x RFFT/IRFFT at N=256 and
1.76x/2.53x at N=1024. All 66 Schedule/Tile and x86 numerical tests pass.
Bare-metal timing remains required for a performance-promotion packet.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **parity
validated; no x86 physical change.** Shared compiler-test discovery now accepts
fail-closed `TESSERA_BUILD_DIR` selection while retaining explicit
`TESSERA_OPT` precedence. The migrated runtime-library and backend-tool users
are ROCm-owned; AVX-512/AMX packages, evidence, and selectors are unchanged.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; x86 physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no AVX-512
lowering or exact Zen 5 evidence. They remain explicitly reference-only until
an x86-owned physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **Zen 5 scan selector and
dtype boundary retained.** The f32 scan ABI now selects a 16-lane AVX-512
Hillis--Steele prefix for `cumsum` and `cumprod`; paired interleaved measurement
against the scalar reference records 1.48x and 1.47x speedups on Ryzen AI Max+
395. The same implementation regressed extrema, so `cummax` and `cummin`
deliberately retain the scalar recurrence. NaN propagation and signed-zero
behavior are exact-tested. The complete x86 physical math cohort passes 167
tests, and the benchmark packet covers unary, transcendental, binary,
reduction, and scan families. General x86 math remains an explicit f32 ABI;
target-wide bf16 capability does not imply bf16 support for these packages.
Binary packages now reject mixed input dtypes. Evidence:
`benchmarks/baselines/math_physical_zen5_2026_08_06.json`.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **Zen 5 physical
policy expansion implemented and retained.** The v3 contract specializes
bounded dynamic shapes into exact content-addressed packages, packs arbitrary
axes inside the native AVX-512 package, admits fp32/fp16/bf16 real storage with f32
accumulation, and carries backward/forward/ortho scaling into the native
package. ABI v4 and 36 focused contract/package/evidence tests pass on AMD Ryzen AI
Max+ 395. The policy
packet `benchmarks/baselines/tsol_physical_policies_zen5_2026_08_06.json`
now contains 30 numerical-and-performance rows across all five compound
operations, including seven digest-changing bounded specializations and
combined dynamic-axis-reduced-storage-ortho cases. Warm medians span
0.058--0.177 ms; every row meets its recorded error limit. Reduced storage
conversion and arbitrary-axis pack/unpack are package-owned host-side work
around f32 native FFT accumulation, not claims of reduced-arithmetic FFT
instructions or device-side packing.

Cross-backend sync `X86-WELFORD-PARITY-2026-08-06` — **native implementation
and exact Zen 5 validation complete.** `var`/`std` now call a dedicated
`tessera_x86_avx512_welford_f32` ABI. It accumulates independent SIMD-lane
f64 Welford states and merges them deterministically after the existing
arbitrary-axis fold, replacing the cancellation-prone `mean(x²)-mean(x)²`
composition. On the AMD Ryzen AI Max+ 395 Zen 5 host, the native image rebuilt
with LLVM/MLIR 23 and all 17 focused tests passed, including
large-offset/low-variance data, ragged extents, multiple axes, and tuple axes.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **shared schema and native
`TPROF-X86-TIME-1` timing provider landed; exact-host evidence open.** The ROCm
work establishes the shared
rule that every applicable clock is an independent provenance/validity record;
it does not require HIP-specific slots in x86 packets. The existing `cpu`
provider already exposes runtime callback spans and `steady_clock` timestamps,
but that is not yet an AVX-512 PMU or sampling profiler.

`TPROF-X86-TIME-1` now extends the same tprof artifact/CLI with four independently
validated x86 records:

1. `host_wall_ns`: raw `steady_clock` batch time with one terminal
   synchronization plus separate empty-call/dispatch calibration.
2. `monotonic_raw_ns`: Linux `CLOCK_MONOTONIC_RAW` for OS-adjustment and clock-
   agreement checks where available.
3. `tsc_cycles`: fenced `RDTSCP` entry/exit with invariant-TSC capability,
   calibrated frequency, start/end logical CPU, and migration invalidation.
4. `perf_task_clock_ns`: `perf_event_open` task-clock timing with explicit
   permission, multiplexing, enabled-time, and running-time fields.

The first portable PMU group is cycles, reference cycles, instructions,
branches, branch misses, cache references, and cache misses. Model-specific Zen
5 events, IBS samples, PEBS, raw event encodings, and derived cache-level claims
remain capability-gated by vendor/family/model, kernel support, and an exact-
machine event-map digest. Unsupported or permission-denied events are explicit
unavailable records, never zero counters.

The x86 provider runs the canonical typed AVX-512 package in a fresh subprocess
with recorded affinity, NUMA node, SMT state, governor/frequency context,
microcode, kernel, compiler, image/artifact digests, warm/cold state, and sample
distribution. Sampling uses Linux perf IP/callchain/branch-stack or AMD IBS only
when advertised; it must correlate samples to the package image and symbol
range. Instrumented and uninstrumented images stay paired so probe overhead is
visible.

WSL host timing may support same-host regression and retain/reject decisions.
PMU- or sampling-dependent promotion requires a non-virtualized exact Zen 5
host, non-multiplexed or correctly scaled counters, stable affinity, and an
independent clock-agreement packet. Intel AMX remains a separate named-host
evidence obligation. HIP `wall_clock64()`, `rtg_hsa_dispatch`, ROCprofiler, and
gfx1151 evidence do not transfer to x86.

Execution order:

1. **Complete:** generalize the TPROF clock-record validator and CLI away from
   GPU-only names.
2. **Complete:** add x86 host/raw/TSC timing with migration and invariant-TSC
   checks.
3. **Complete:** add the `perf_event_open` provider, permission diagnostics,
   scaling, portable PMU group, `tprof x86 timing-status`, and optional native
   proof snapshot workflow.
4. **Landing:** `tprof_x86_event_map.py` now digests the exact CPU identity,
   microcode, sysfs PMU encodings, and `perf` catalog. `tprof_x86_sample.py`
   embeds that map, pins the command with `taskset`, and records perf samples against an image
   build ID, DSO-aware symbolization, and the matching static ELF symbol range,
   avoiding invalid ASLR-relative comparisons. AMD IBS is admitted only for
   family 26 with AVX-512 and an advertised `ibs_op`/`ibs_fetch` event. Exact
   perf/IBS samples remain open because this WSL host has no `perf` executable
   and denies `perf_event_open`. The truthful host packet
   [`../../../../benchmarks/baselines/x86_zen5_event_map_2026_08_07.json`](../../../../benchmarks/baselines/x86_zen5_event_map_2026_08_07.json)
   identifies AMD family 26/model 112 and captures the visible sysfs event
   sources, but marks the catalog and promotion gates unavailable under WSL.
5. **Landing:** the exact-host runner now binds the existing aligned/ragged
   E2E-REAL-4 comparison to clock, sampling, source-commit, dirty-worktree, and
   artifact provenance. The current packet is
   [`../../../../benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json`](../../../../benchmarks/baselines/x86_zen5_profiler_packet_2026_08_06.json):
   scheduled/production is 1.018x aligned and 1.008x ragged with exact route
   parity. Verdict `retain`; WSL clocks, denied PMU access, unavailable symbol
   samples, and the development worktree block promotion. A clean bare-metal
   rerun with real samples remains open; AMX stays access-gated.

The next exact-host action is one clean bare-metal Zen 5 run with a fixed CPU,
stable governor/NUMA placement, working `perf_event_open`, the model-specific
event-map digest, and symbol-correlated cycles plus advertised IBS samples.
That packet must be regenerated from the same commit as the aligned/ragged
benchmark; WSL timing and the event-source inventory cannot satisfy it.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **shared typed carrier and
x86/Zen 5 physical consumer complete.** `tessera.scheduled_spectral.v5`
materializes one verified `schedule.spectral_program` →
`tile.spectral_program_kernel` edge and binds child FFT digests plus the full
compound policy. The native AVX-512 package now owns DCT mirroring,
padding/cropping, framing/windowing, half-spectrum expansion/compaction,
complex multiplication, deterministic overlap-add, and a bounded thread-local
digest-keyed workspace; runtime no longer reconstructs these programs with
host NumPy. Exact Zen 5 aligned, ragged, and prime/Bluestein cases agree with
NumPy. This is architecture-owned evidence and does not inherit gfx1151
performance or scheduling choices.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **shared artifact
truth assessed; no x86 physical change.** The FFT artifact can now identify
gfx1151's batched fused-LDS residency and v4 native family, but Zen 5 retains
its independently selected AVX-512 FFT/package policy. The HIP image build
dependency, LDS schedule, and WSL timings do not transfer to x86; no x86
correctness or performance claim changes.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **DCT physical coverage
expanded on Zen 5; clean-host promotion remains open.** DCT-I/II/III/IV now
carry distinct public, autodiff, Graph, Schedule, and Tile identities.
`tessera.x86.spectral_composite.v6` phase-corrects the FFT-backed type-II path
and provides separately hashed direct cosine kernels for types I/III/IV; exact
Zen 5 correctness coverage passes. The causal chunked-STFT state now binds its
policy digest and overlap lineage; centred streaming fails closed pending
explicit lookahead. The new DCT direct paths have no performance promotion.
The same boundary audit preserves scalar convolution and one-sample
STFT/ISTFT through explicit scalar and odd full-complex paths; native AVX-512
regressions pass on the Zen 5 host.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **shared atan2 semantic
fix and x86 Welford parity apply; ROCm physical kernels are not applicable.** Shared quadrant logic
now preserves signed-zero origins, infinity diagonals, and NaN propagation.
The x86 atan kernel remains the magnitude consumer and requires its existing
Zen 5 differential gate. The x86 statistical path now has an independently
implemented and Zen 5-tested native Welford ABI under
`X86-WELFORD-PARITY-2026-08-06`; it does not inherit gfx1151 evidence.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **not applicable; x86
parity assessed.** The opaque device-plan ABI, persistent HIP allocations, and
prebuilt ROCm shared image are architecture-owned. x86 retains its existing
content-addressed native package and thread-local cached Bluestein plan; no x86
schedule or numerical policy changed.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **Bluestein cache and work-gated
mixed-radix Stockham promoted; Rader conditional; Bailey rejected.** Immutable
chirp/kernel FFT plans plus thread-local padded workspaces improve warm
Bluestein by 1.57x--1.76x at N=127--1009. The native mixed-radix ABI caches
stage matrices/twiddles and executes 16 contiguous butterflies per AVX-512
codelet. It beat Bluestein on 12/13 measured composite shapes; representative
speedups are 1.91x at N=68, 3.75x at N=289, and 2.82x--4.90x at
N=768--5120. A factor-work gate promotes those wins through the exact
Schedule→Tile artifact while rejecting N=255 to cached Bluestein.

Rader is retained as a named candidate only: it wins at N=257 because its
convolution and workspace are half Bluestein's, but loses at N=127/509/1009.
The native Bailey candidate fuses its middle transpose with twiddle
multiplication yet remains 1.66x--2.23x slower at N=64K--1M, so it is rejected
without coefficient tuning. The sweep also fixed a correctness defect where
the out-of-place runtime helper mutated contiguous caller inputs.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **AVX-512 plan/twiddle
cache, generated permutation, and every planner-admitted mixed-radix codelet
are retained.** The production C2C ABI now keeps a
bounded thread-local immutable plan containing gather offsets and all
per-stage twiddles, removing allocation and transcendental work from warm
execution. Same-host ratios against `scipy.fft(workers=1)` improved from
8.63x/22.75x/17.20x to 3.46x/3.85x/3.10x at N=1024/8192/65536, with the
existing numerical gates passing. The former scalar random-swap bit reversal
is now a cached AVX-512 gather into reusable thread-local workspace; the gap
improved again to 2.92x/1.86x/2.73x with the C++ correctness corpus passing.
This is a retain, not library parity. The runtime-radix DFT loop is now a
compile-time specialization for radices 2/3/4/5/7/9/11/13/15/17, including
AVX-512 butterfly batches and unrolled scalar tails. Native tests cover every
radix plus a multistage N=68 round trip. The 2026-08-09 packet records correct
N=68/225/289/768/3072/5120 transforms: native is 2.79× and 2.65× faster than
NumPy at N=3072 and N=5120, while smaller rows remain slower. Therefore the
existing work gate is retained and no global selector promotion is made.
Remaining expansion is nontrivial stride, dtype, and larger-transform coverage.

The v2 content-addressed artifact records `cooley_tukey_dit`, host-inplace
residency, cached-f32 twiddles, workspace policy, radix sequence, and the
complex64/f32 numeric policy. Evidence is in
`benchmarks/baselines/fft_plan_cache_radix17_2026_08_05.json`.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **typed artifact consumption
implemented; physical package parity validated on Zen 5.**
ROCm's public runtime now consumes its proven Stockham/Bluestein package rather
than a duplicate O(N²) DFT. x86 keeps its existing AVX-512 radix-2/Bluestein
package unchanged and remains numerically covered. It now consumes the exact
content-addressed `schedule.fft`→`tile.fft_kernel` identity without Graph
metadata or a second planner decision. The contract preserves x86's radix-2,
tiny-DFT, and Bluestein choices rather than transferring gfx1151's physical
stage sequence. Focused Schedule/Tile tamper tests and exact Zen 5 FFT tests
pass. The remaining shared packaging action is ROCm-specific runtime-`hipcc`
removal; x86 keeps its existing prebuilt native ABI.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **parity validated on host; the reference lane for the family.**
Tessera's own FFT (Stockham, `TargetHooks/`) extends from powers of two to
every length: a generic radix-r stage for the odd small primes and Bluestein
for the rest. Shared contracts changed, so all four backends are affected:

* **Planning is now one implementation** (`TargetHooks/Common/FFTPlan.h`).
  CPU, AMD and NVIDIA each carried their own `while (n%4) ... while (n%2)`
  driver loop, and all three silently returned a HALF-FINISHED transform for
  any other N while reporting success. `LegalizeSpectral::pickRadixSequence`
  was a fourth copy, factoring over radices 7/5/3/4/2 and pushing a residual
  prime as a "stage" of that radix -- a stage nothing could execute.
* **Compiler routing was wrong independently of the kernels.**
  `LowerToTargetIR::stageSymbolFor` mapped every radix other than 4 to
  `ts_stockham_r2_*`, so a static N = 12 = 4x3 emitted a radix-2 call for a
  radix-3 stage. The runtime driver was correct; the compiler path was not, and
  direct driver tests could not see the difference.
* **New C ABI surface:** `ts_stockham_rn_<backend>(in, out, N, L, r, sign)`
  (note the extra radix argument, which r4/r2 do not take), plus
  `tessera.target_ir.stage_radices` carrying it, and a
  `tessera.target_ir.bluestein` marker routing those lengths to the driver.

The CPU hook is the F4 reference every other lane is checked against, so its
correctness gates the others. Verified against a naive fp64 DFT across 63 sizes
(51 mixed-radix, 12 Bluestein), zero failures, round trips to ~3e-6.

Its generic radix-r stage precomputes the r-point DFT matrix once per stage and
reuses it across every butterfly -- the opposite of the GPU choice, and the
clearest evidence the shared/per-target split is drawn in the right place.

No AVX-512 specialisation: the stages are scalar C++. Vectorising the butterfly
is open work, and this change neither helps nor blocks it.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **parity validated at the capability level; device evidence missing.**
PR #493 closed the Graph IR shape-rule registry: **303 declared / 6 deliberately
undeclared / 0 unexamined**, with the `MAX_UNCLASSIFIED` ratchet dropped 106 -> 0.
Shared contracts changed; all four backends are affected equally at the
reference level:

* **Result contracts.** Multi-result ops now emit every SSA result
  (`kv_cache.read -> (K, V)`, `top_k`, `qr`/`svd`/`lu`/`nonzero`), and tuple
  destructuring (`v, i = ...`) lowers. The emitter previously called the
  single-result `_infer_result_type`, so a declared multi-result contract
  stopped at Graph IR.
* **Stateful handles.** `!tessera.kv_cache` is now reachable from Python; the
  emitter had been printing `tensor<*x?>` for a type the ODS has always
  declared.
* **dtype policy.** An integer input to a float-producing op promotes to the
  declared `COMPUTE_FLOAT_DTYPE` (fp32) instead of NumPy's width-derived float
  (`cos(int8) -> f16`, `cos(int32) -> f64`); index/count results use a declared
  `INDEX_DTYPE`; complex is a LOGICAL dtype carried in an interleaved real pair,
  not a storage format.
* **Diagnostics.** The whole `GRAPH_IR_*` family (17 codes) is registered - the
  drift gate's scanner did not know the prefix, so it reported green while the
  family accumulated unregistered.

**This is the Python reference lane, not generated device code.** Complex FFT is SUPPORTED on `x86` - one of three targets (with `cpu` and
`apple_cpu`) declaring an `fft` capability entry, and complex maps onto the
interleaved fp32 pair AVX-512 already handles. No storage-contract change:
appending complex to any target's `supported_dtypes` was tried and correctly
broke `test_x86_dtype_contract`, because that tuple answers "what STORAGE has
this backend proven" and complex is not a storage format the ISA has. Complex is
declared on the transform ops that carry it, not on the target.

`x86_ready_storage_dtypes()` is unchanged. The reduced-precision compute contract
(promote -> compute at f32 -> store back) is reference-level; **the AVX-512 lane
has no exact-device proof that generated kernels honour it.**


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **follow-up required, emulated path.**
The x86 dtype contract records `fp8_e4m3` as `emulated`: packed-byte storage
with software conversion and fp32 compute, since Zen 5 has no native FP8
arithmetic. So x86 CAN carry real sub-byte STORAGE even though it cannot
compute in it — which makes it a useful place to prove the storage path
independently of native arithmetic. x86 owns deciding whether to materialize
packed fp8 storage or keep the f32 fake-quant reference.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **follow-up required, reference-level only.**
The reference lane now computes reduced-precision ops at f32 and stores back.
x86's AVX-512 kernels are the executable lane on this fleet and were not
re-verified against the corrected reference; the fp16/bf16 accumulate contract
is now stated explicitly, so a mismatch would be a real divergence rather than
an ambiguity. x86 owns an AVX-512 execute-and-compare — obtainable on the
Strix Halo box, unlike the AMX lane.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **not applicable, with a reason.**
x86 has no `tile.mma` consumer: `TileToX86Pass` lowers through the C-ABI shim
and the `tessera_x86` Target IR models the boundary with `abi_call`. The shared
`tessera::tile::dataOperands` helper is available to it but currently unused,
so there is nothing to migrate. Re-assess when the `x86vector.*` (AVX-512)
lowering lands, since that is where x86 would gain a matrix-op consumer.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **parity validated, new dialect landed.**
W0.10 closed x86's Decision #19 gap: it was the one backend with no Target IR
dialect at all (`TileToX86Pass` lowered to 21 `func::CallOp`s into a
hand-written C shim, and the Python emitter named a `tessera_x86.func` op no
dialect defined). No carve-out was granted. `tessera_x86` now exists with a real
`!tessera_x86.tile` type, is registered in `tessera-opt`, and separates
value-carrying AMX ops from directives; `abi_call` models the C-shim boundary
instead of hiding it. Positive and negative lit fixtures ship — the negative one
proves the verifier rejects a dot-product whose operands never came from a tile
load. The Python x86 emitter now parses, loads the dialect, and verifies.
**Scope limits:** AMX *lowering* is optional per project direction (expected
supersession by ACE), so the AMX ops are the IR-level contract only. The live
follow-up is `x86vector.*` (AVX-512) lowering instead of terminating in
`func.call` — that changes generated code and needs AVX-512
execute-and-compare, which IS obtainable on the Strix Halo box. No AMX
execution evidence is claimed: no machine in the fleet reports AMX.

## X86 attention and training closeout

Cross-backend sync `CORE-ATTENTION-TRAINING-X86-2026-07-30` — **closed for
Zen 5 AVX-512; no AMX claim.**

The pre-existing inventory is now explicit:

- AVX-512 rank-4 attention forward, Lion forward, SGD and
  Momentum/Nesterov VJPs, loss-to-SGD/AdamW fusion, and physical DeltaNet
  backward were already complete.
- `X86-ATTN-CANON-1` is complete. Canonical x86 packaging begins from
  `tessera.flash_attn`, runs the shared rank-4 batch/query-head/KV recurrence,
  fails closed unless the streaming `scf.for` structure is present, and only
  then selects the established typed AVX-512 attention ABI. The package source
  no longer presents a freshly synthesized `tile.attention_kernel` as its
  semantic authority. Existing f32 MHA/GQA, bias, causal/window, and softcap
  numerical behavior remains covered on the Ryzen AI MAX+ 395.
- `X86-ATTN-BWD-1` is complete. The x86 package structurally consumes the
  canonical tensor-valued dQ, split-dK/dV, and ascending fixed-order reduction
  loops. Its AVX-512 ABI executes MHA/GQA/MQA gradients with optional bias,
  causal/window, and softcap modifiers.
- `X86-LSE-1` is complete for this Zen 5 target. A 21-sample resident packet
  compares the established forward plus recomputed-LSE backward with the
  forward-with-LSE plus saved-LSE backward at sequence lengths 32/64/128.
  Saved LSE wins by 1.45x/1.23x/1.06x, so x86 selects `save_lse`. Evidence:
  [`../../../../benchmarks/baselines/x86_avx512_attention_lse_2026_07_30.json`](../../../../benchmarks/baselines/x86_avx512_attention_lse_2026_07_30.json).
- `X86-LION-BWD-1` is complete. One AVX-512 call implements the canonical
  stop-gradient-through-sign VJP for parameter, gradient, and carried moment.
- `X86-ADAFACTOR-1` is complete. AVX-512 factored row/column and lower-rank
  full-moment forward execution and analytic physical adjoints match the shared
  optimizer/VJP oracles.

The x86 work does not transfer physical schedules or evidence to sibling
backends. ROCm parity was already complete; Apple and NVIDIA retain their
architecture-owned canonical attention/backward and training materializer
items. Validation for this closeout is recorded by the owning PR.

## X86-SPINE-1: reconcile C synthesis with the MLIR/LLVM lane

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **AVX-512 lane landing; AMX
remains planned/access-gated.** Canonical target `x86` now has one meaning:
Graph/Tile IR lowered by `TileToX86Pass`, packaged with the C++ backend shared
image and typed launch descriptor. Vendor-family selection moved out of the
shared driver into `x86_native.native_package_kind()` / `package_native()`.
Apple CPU/GPU now use the same backend-owned admission shape while retaining an
explicit Value Target-IR compatibility/probe opt-out; this does not transfer any
AVX-512 ABI, schedule, or exact-device evidence.

APPLE-RASTER-1 subsequently consumed the shared mapping in emitted MSL and
retained row-major after mixed Apple7 timing. X86 remains not applicable because
CPU work partitioning is not GPU workgroup rasterization.

The former `emit/x86_llvm.py` implementation never emitted LLVM IR. It is now
`emit/x86_c.py`, registered under source target `x86_c`, and remains a measured
fused-region candidate under the canonical x86 arbiter. A compatibility import
preserves old module imports without reclaiming target `x86`. Its artifact
profile is explicit `x86-64-v4` rather than build-host-dependent
`-march=native`; the source carries that profile into the content-addressed
cache identity and the runner declines on hosts lacking the required AVX-512
feature set. Its execution tag is `x86_c_native`, distinct from the canonical
descriptor lane.

The native loader also keeps each memfd alive with its image. Previously Linux
could reuse an fd number and glibc could return the base-x86 handle for a later
AVX-512 `/proc/self/fd/N` load, making valid descriptor symbols appear absent.
A base-then-AVX-512 regression test guards distinct handles and symbols.

**Zen 5 proof.** The Ryzen AI Max+ 395 WSL host reports the complete
`x86-64-v4` AVX-512 feature set. The broader 2026-07-30 cleanup run passed 261
focused Python candidate/canonical/native/audit tests with one expected
capability skip. A fresh spine verification on the same host passes **63/63**
canonical-x86/native-descriptor plus explicit-`x86_c` source-candidate tests.
The current x86 dtype/ISA/capability and manifest gate is **26/26** (superseding
the earlier recorded count of 24), and all **18/18** rebuilt C++ backend
executables pass.

The seven x86-owned Tile-to-x86 MLIR fixtures pass **7/7**. The expanded
cross-target set now discovers 12 fixtures: **11 pass and 1 is expected
unsupported**. The unsupported `layout_target_materializers.mlir` fixture
requires the Apple backend; it is not an x86 failure. The native GEMM executable
reports `AMX not available; skipping` and runs the AVX-512 path. No AMX
readiness, numerical, or performance claim is inferred.

## X86-CALIB-1: split verdict on the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **retired step-distance;
bank-conflict not applicable; T1 hierarchy rejected for latency ranking.** Owning host
Zen 5 (Ryzen AI Max+ 395 CPU complex, AVX-512, no AMX).

The original two static device-free scores were assessed against measured latency
([`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8; motivation in
[`TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md) §2). They do
not get the same verdict here, and reporting one blended state would hide that.

**Bank-conflict analyzer — not applicable, architecture-specific reason.** It
counts N-way conflicts across a fixed number of software-managed scratchpad banks
under a wave's phase-grouped access. The x86 lane has no software-managed
scratchpad and no wave phases: AVX-512 loads go through a hardware-managed
L1/L2/L3 hierarchy where the analogous hazards are 4 KiB aliasing, cache-set
associativity conflicts, and store-forwarding stalls. Those are real, but they
are a different model with different inputs — not this analyzer with different
constants.

**Step-distance locality — rejected fleet-wide.** It failed on the ROCm
architecture from which it was derived, so x86 will not retune or revive it.

**T1 reuse/cache model — rejected for ranking, retained as a pruning
diagnostic.** The measured model now represents independent L1D/L2/L3 LRU
levels plus DRAM, using sysfs capacities, CPU-pinned resident-copy bandwidths,
and an architecture-derived single-core peak. Ten physically distinct blocked
AVX-512 GEMM candidates were measured on three workloads. The durable packet
`benchmarks/baselines/x86_zen5_t1_cache_model_2026_08_09.json` records median
Spearman rho -0.4062 and 0/3 measured-winner matches. This fails the
predeclared rho>0.25 promotion threshold. No coefficients were tuned, the
production selector is unchanged, and measured latency remains authoritative.

**Fleet outcome (2026-07-29).** ROCM-CALIB-1 reproduced 0/6 measured winners on
the AMD home architecture (median rho -0.1381, 0% positive), triggering the
agreed no-retuning stop rule. x86 no longer owes a calibration run for adoption
of this score. CPU cache-blocking or reuse-distance research remains valid as a
different model; it must not be presented as a resurrection of the rejected
step-distance latency ranker.

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **not applicable, with an
architecture-specific reason.** Schedule IR gained `raster_order` /
`raster_group` on `schedule.tile` / `schedule.knob` (arch-neutral definition in
`compiler/tile_rasterization.py`; rationale in
[`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2). The knob permutes *block ids across a 2-D launch grid* so that the set of
workgroups resident at one instant shares operand panels in a hardware-managed
L2. The x86 AMX/AVX-512 lane has no launch grid: `tessera_x86_backend` emits
loop nests over tiles executed by OpenMP threads, so there is no block id to
permute and no equivalent of a wave of co-resident workgroups contending for a
shared last-level cache in that pattern. The analogous x86 lever — loop order and
cache blocking in the C/LLVM emitter — already exists as a separate mechanism and
is not expressible as this permutation.

**This is a not-applicable for the *contract*, not for the underlying idea.**
Tile-granular reuse-distance analysis, the T1 item the same assessment proposes,
*does* port to AMX/AVX-512 cache blocking — that literature is a CPU literature
(Lam/Rothberg/Wolf 1991 on blocked algorithms). T1 v1 is now built; revisit x86
through `X86-CALIB-1` when the hierarchy inputs and Zen 5 corpus correlation are
ready, not by consuming `raster_order`. No exact-device evidence is owed for the
raster contract itself.

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **not applicable**. Apple
added `apple_gpu_air`, a precompiled-artifact lane behind the shared
`register_compiler(target, compile_fn)` seam, measured against its compile-on-
launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x; host-wall
timing on Apple M1 Max, not device-event evidence). x86 needs nothing here:
`_x86_compile_fn` already returns a real `.so` from clang, and the x86 lane has
no runtime-compilation path to compare it against. Recorded so the fast-path
shape is documented fleet-wide; the measurement method transfers when X86-1 is
proven on a Zen 5 host. No shared IR, ABI, dtype/op registration, or numerical
contract changed.

The fourth architecture queue, alongside
[`apple/todo.md`](../apple/todo.md), [`nvidia/todo.md`](../nvidia/todo.md), and
[`rocm/todo.md`](../rocm/todo.md). Opened 2026-07-28 because x86 work was being
discovered on an Apple Silicon host, where none of it can be proven.

**Owning-host rule.** x86 lanes are proven on an x86 host — the Zen 5 box for
AVX-512 (Core Ultra 7 265F has neither AVX-512 nor AMX; see the fleet notes).
Nothing in this queue may be marked complete from a Mac. An arm64 host can
author and structurally gate; it cannot produce device evidence.

**Scope split.** The exact-device target of this queue is `x86_avx512`, proven
on the Zen 5 host. AMX is **planned, access-gated**, and cannot inherit AVX-512
evidence: no AMX-capable owning host is currently named. X86-3 may reconcile
the compiler-lane architecture, but its AMX half remains open until a separate
AMX host, target identity, numerical packet, and performance evidence are
recorded. The access-gated correctness command now exists at
`scripts/run_x86_amx_release_gate.sh`; its existence is not device evidence.

## X86-1: close the portable-C plugin provenance and host gate

**Status: closed for the Zen 5 AVX-512 host.**

Historically `tests/unit/test_x86_plugin.py` asserted
`execution == "x86_native"` in 15 places. On the original Apple Silicon audit
host every one returned `"reference"`. The historical cause was not a defect:

* the former `emit/x86_llvm.py::_x86_compile_fn` compiled emitted C with
  `clang -O3 -march=native -fPIC -shared`;
* `platform.machine()` was `arm64`, so `-march=native` targeted ARM. The
  produced `.so` was not an x86 kernel, the runner declined, and it reported
  `reference` — the honest answer.

These failures were invisible until 2026-07-28 because `clang` was not on
`PATH`; the lane skipped for the wrong reason. Putting LLVM 23 on `PATH`
un-gated them. They fail identically on `main`.

Required work, on an x86 host:

1. **Complete.** `X86CEmitter` → `_x86_compile_fn` → `X86CRunner` returns
   `x86_c_native` with numerics matched against the F4 numpy oracle.
2. **Complete.** `native` baked in the build host, which made
   a cached artifact non-portable across the fleet and interacts badly with the
   content-addressed `kernel_cache` key (the key hashes source + dtype + target,
   *not* the host ISA — two hosts would collide on one entry). The selected
   profile is explicit `-march=x86-64-v4`, recorded in emitted source/cache
   identity and guarded before execution.
3. **Corrected.** AOCL-DLP registers only an unavailable hand-tuned candidate;
   it never registered a compiler. The C compiler now registers for `x86_c`,
   leaving canonical target `x86` unambiguous.

**Interim (landed 2026-07-28):** the assertions are host-gated so an arm64 host
skips instead of failing. The gate is `platform.machine()`, not a capability
probe — it says "this host cannot prove an x86 kernel", which is exactly the
claim. Removing the gate is not the fix; proving the lane on x86 is.

## X86-2: `_LANG = "c"` — the file name says LLVM, the emitter says C

**Status: closed.**

`emit/x86_c.py` sets `_LANG = "c"` and emits C for `clang`, matching its name.
`emit/x86_llvm.py` is a compatibility-only re-export.
That is the selected design: source-synthesis modules emit vendor source text
(CUDA C, HIP C++, MSL, C), while canonical `x86` reaches LLVM through the typed
C++ compiler spine. The compatibility shim preserves imports without restoring
the misleading compiler authority.

## X86-3: reconcile the two x86 lanes

**Status: AVX-512 half closed on Zen 5; AMX correctness lane landed but remains
planned/access-gated. A separately named AMX-capable host is still required.**

x86 reaches hardware two ways, and nothing arbitrates between them:

* **C++ MLIR** — `src/compiler/codegen/tessera_x86_backend/`, AMX BF16 +
  AVX-512 GEMM. Decision #1 records the existing end-to-end architecture;
  this plan may revalidate AVX-512 on Zen 5 but cannot refresh the AMX claim
  without an AMX-capable host.
* **Python C candidate** — `emit/x86_c.py` + optional `emit/x86_aocl_dlp.py`
  behind the arbiter; it no longer owns canonical target `x86`.

This is the same two-compiler split documented for Apple in
[`apple/todo.md`](../apple/todo.md); x86 has it too, and the resolution should
be consistent across the fleet rather than decided per backend. Blocked on the
spine decision in
[`../../compiler/COMPILER_THEORY_OF_OPERATION.md`](../../compiler/COMPILER_THEORY_OF_OPERATION.md).
The two required terminal outcomes are now explicit: AVX-512 is selected and
proven on Zen 5; AMX is planned/access-gated until a named capable host supplies
its own packet. Neither architecture promotes the other.

The AMX regression is now owned by
`tests/device/x86/test_amx_int8_gemm.py` and selected by
`scripts/run_x86_amx_release_gate.sh`. The gate fails closed on missing
AMX-TILE/AMX-INT8, runs native execution in a crash-isolated child, repeats the
K>64 numerical comparison twice without xdist, and retains identity, JUnit,
collection, and status artifacts. This closes the validation-ownership gap; it
does **not** close X86-3. A named Intel AMX host must still produce the packet,
and a separate measured-performance gate and baseline remain open.

Cross-backend sync `X86-AMX-DEVICE-2026-08-02` — **not applicable to Apple,
NVIDIA, and ROCm.** This change moves one x86-native regression and adds an
x86-owned local proof command; shared IR, runtime ABI, marker policy, and peer
backend device commands are unchanged.

## Cross-backend sync

`TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` — **parity validated**. x86 tests that
drive `tessera-opt` route through the shared capability-aware helper, so a build
without the owning backend skips with the missing pass named. No x86 pass body,
ABI, or numerical contract changed; no exact-device evidence claimed.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: not applicable — architecture-specific reason.** Zero files under `tessera_x86_backend/` reference `FragmentType` or `!tile.fragment` (measured 2026-08-03), and x86 has no cooperative-matrix fragment to model: it carries its own `!tessera_x86.tile` value type over AMX/AVX-512 ops (Decision #19, built typed from the start in W0.10).

That backend is in fact the reference shape for where W1.1 is heading — 0 `AnyType` / 0 `Variadic<AnyType>`, with a negative fixture proving the verifier rejects a dot-product whose operands never came from a tile load. Per project direction the AMX half stays an IR-level contract with no `amx.*` lowering, so no follow-up is created here.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: not applicable — architecture-specific reason.** Unchanged from `TILE-FRAGMENT-TYPE-PARAM-2026-08-03`: no cooperative-matrix fragment on this backend (it carries `!tessera_x86.tile`), so neither the typed `tile.mma` contract nor the accumulator-threading follow-up applies. AVX-512 K-loop accumulation is expressed in its own ops and is unaffected.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: not applicable — architecture-specific reason.** Probed: `--tessera-lower-to-x86` leaves `tile.mma` unlowered. x86 has no cooperative-matrix MMA path; AVX-512 K-loop accumulation is expressed in its own ops and never routes through this lowering.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: not applicable — architecture-specific reason.** x86 elementwise lanes raise `_RocmCompiledUnavailable` only for `lib is None` / missing-symbol conditions — envelope limits by construction, since there is no compile step whose output could be malformed. No x86 site was reclassified.

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: not applicable — architecture-specific reason.** x86 carries `!tessera_x86.tile` and has no cooperative-matrix `tile.mma` path; its pipelines are untouched.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: not applicable — architecture-specific reason.** x86 carries `!tessera_x86.tile` and has no `tile.view`-backed fragment path.

## Cross-backend sync `TILE-VIEW-LINEAR-BASE-2026-08-05` — should `tile.view` carry a precomputed linear base?

ROCm W1.1 step 3 (`W1_1_TYPING_DESIGN.md` §4.7) established that isolated
fragment address derivation could not express the direct lane's shared row
offset. Measurement selected an optional precomputed `linear_base` operand on
`tile.view`; logical row/column origins remain present for bounds.

ROCm implemented explicit `tile.view` linear-base sharing. Its new same-run
final rebuilt measurement improves typed/direct from 0.685x to 0.711x, but does not close the
gap; load scheduling/wait overhead remains the ROCm-owned follow-up.

**Outcome for x86: NOT APPLICABLE.** This backend consumes neither `tile.view`
nor `tile.fragment_pack` (0 files). AMX/AVX-512 operands come from
`tessera_x86.amx_tile_load` over the `!tessera_x86.tile` type (Decision #19),
which addresses its own source directly; there is no `tile.view`-backed fragment
path whose base could be hoisted. If a future x86 path adopts Tile fragments,
re-open under this key.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

Shared `tile.view` / `tile.store` can now carry an SSA leading dimension when
`#tile.memory_layout` states zero. **Outcome for x86: NOT APPLICABLE.** AVX-512
and access-gated AMX consume `!tessera_x86.tile`, not Tile fragments or
pointer-backed `tile.view`; no x86 lowering changed. Host Zen 5 validation:
x86 dtype + matmul-family suites, 21 passed.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records explicit artifact ancestry and
production `tessera-opt` registers the generated Schedule dialect. **x86
outcome: follow-up required under E2E-REAL-3.** Canonical x86 packaging still
accepts `GraphIRModule` and re-derives its launch Tile program, so the recorded
Graph→package-Tile edge exposes the fork and `lineage_complete` remains false.
No AVX-512 ABI, generated code, selector, or AMX gate changed. The consumer PR
must accept the canonical launch-Tile artifact and rerun Zen 5 exact execution;
this does not supply the separately access-gated Intel AMX packet.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

The shared C++ spine now preserves a bounded static Graph matmul behind a
content-addressed `schedule.matmul` SSA edge and lowers it exactly once to the
portable A/B/D/M/N/K `tile.matmul_kernel` contract. The x86 instance is f32
storage/accumulation/output with m16n16k16 row/col layout and explicit
pipeline/raster fields. **x86 outcome: structural parity validated; physical
follow-up required under E2E-REAL-3.** No AVX-512 execution or performance is
claimed by this host-free conversion. Canonical x86 packaging must accept this
exact Tile artifact, run TileToX86, and repeat the Zen 5 numerical/performance
ratchet without reconstructing the launch contract from Graph IR. Intel AMX
evidence remains separately access-gated.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The bounded f32 matmul package now accepts `ScheduledMatmulArtifact` and
consumes its exact launch-level Tile text through TileToX86. The compile bundle
records adjacent Graph→Schedule→Tile→Target→backend digests rather than a
Graph-owned package fork. **x86 outcome: parity validated for E2E-REAL-3.**
Exact Zen 5 descriptor execution agrees numerically on the established
`1x1x1`, `5x17x9`, and `16x31x19` corpus, and the physical lit fixture proves
no Graph, Schedule, or launch-level matmul op survives. E2E-REAL-4 still owns
the AVX-512 performance ratchet and promotion decision. This is not Intel AMX
evidence; that named-host packet remains access-gated.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

The scheduled artifact now separates the physical 16x16 instruction tile from
an architecture-owned macro tile; x86 selects 16x16 for both. **x86 outcome:
promote.** On the exact Ryzen AI MAX+ 395 Zen 5 host, the established aligned
`64x128x96` and ragged `127x65x79` rows are bit-identical to the production
AVX-512 package. Scheduled/production median ratios are 1.031x and 0.988x,
inside the existing 10% ratchet. The report records compiler/toolchain and all
Graph/Schedule/Tile/Target/image digests, compile state, image size, CPU
features, and host-wall operation-total timing:
[`../../../../benchmarks/baselines/x86_avx512_e2e_real4_matmul_2026_08_05.json`](../../../../benchmarks/baselines/x86_avx512_e2e_real4_matmul_2026_08_05.json).
This is AVX-512 evidence only; Intel AMX remains access-gated.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The bounded canonical f32 softmax/reduction route now crosses real
Graph→Schedule→Tile boundaries. `schedule.softmax` and `schedule.reduce` bind
architecture, numeric policy, axis/kind, launch width, and durable SHA-256
identity; `ScheduledKernelArtifact` feeds the exact Tile text to TileToX86
without Graph re-entry. Static last-axis softmax and last-axis rank-reducing
sum/mean/max are lineage-complete, and tampered policy fails closed. Exact Zen
5 AVX-512 descriptor launches for scheduled softmax and reduction agree with
NumPy. **x86 outcome: parity validated for the bounded E2E-REAL-5 slice; no new
selector or performance promotion.** `keepdims=true` remains on the explicit
Graph-owned descriptor route because canonical `tessera.reduce` is presently
rank-reducing. This is AVX-512 evidence only; the named Intel AMX lane remains
access-gated and unchanged.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

`schedule.attention` now binds the shared static rank-4 online-softmax
recurrence, modifiers, launch contract, and architecture-owned backward-LSE
policy into one SHA-256 identity. The x86 package consumes the exact emitted
`tile.attention_kernel` through TileToX86 without returning to Graph IR and
preserves `save_lse/saved`. **x86 outcome: parity validated for E2E-REAL-5A.**
On the exact Ryzen AI MAX+ 395 Zen 5 host, the scheduled AVX-512 descriptor
launch agrees with the NumPy oracle for ragged `Sq=5/Sk=7` f32 attention.
This changes no selector and supplies no Intel AMX evidence. Canonical
attention backward was the next x86 family boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

`schedule.attention_backward` now carries the canonical tensor-valued dQ,
split-dK/dV, and ascending-reduction loops as one content-addressed three-result
program artifact. The exact Tile program lowers to
`tessera_x86_flash_attn_bwd_f32`; its descriptor requires the forward-owned
`row_lse` buffer, so `save_lse/saved` is explicit data identity rather than an
untracked policy string. **x86 outcome: parity validated for E2E-REAL-5B.**
Exact Zen 5 tests pass for MHA, GQA, and MQA; aligned and ragged shapes; and the
combined causal, symmetric-window, bias, and softcap envelope while preserving
the established AVX-512 modifier contract. No AMX evidence or selector
promotion is inferred.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The Zen 5 Lion VJP, factored/full Adafactor VJP, and sequence-mixer backward
launchers now enforce the shared content-addressed logical-buffer lineage and
consume exact typed Schedule→Tile artifacts before native launch. Runtime
consumers no longer retain or reconstruct Graph-op metadata. **x86 outcome:
parity validated for the bounded E2E-REAL-5C slice.** Exact Zen 5 Lion,
factored/full Adafactor, and gated/modified DeltaNet backward tests pass. No AMX
evidence is inferred.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

The configuration schema makes the Tile producer, Target-IR consumer, and
backend image boundary explicit instead of accepting Python-composed pass
strings. **x86 outcome: parity validated.** The architecture-owned
`tessera-x86-executable` registry now applies the same closed-family rule while
preserving Schedule→Tile→`tessera_x86` identity and the stable AVX-512 shared
image. AMD async-copy/waitcnt semantics remain not applicable; Intel AMX stays
separately access-gated. No ROCm schedule or evidence transfers.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

`tessera.x86.spectral_composite.v5` consumes the new hashed compound-fusion
topology directly. Even-length convolution uses two packed N/2 RFFTs, a
Hermitian-bin multiply, and packed IRFFT; STFT frames into real storage before
the N/2 transform; ISTFT sends the half spectrum directly through packed IRFFT
before deterministic overlap-add. Odd windows retain a separately identified
full-complex fallback so the public shape envelope is unchanged.

**x86 verdict: retain.** All 16 native AVX-512 composite tests pass on the
Ryzen AI MAX+ 395 Zen 5 host. The 20-sample synchronized-host-wall packet is
[`../../../../benchmarks/baselines/tsol_packed_fusion_zen5_2026_08_08.json`](../../../../benchmarks/baselines/tsol_packed_fusion_zen5_2026_08_08.json);
baseline f32 maximum errors are `2.87e-6`, `9.71e-7`, and `8.77e-6` for
convolution/STFT/ISTFT. This changes no AMX claim. A same-run comparison with
the retired full-complex composite is still required before calling the v5
package a performance promotion rather than a correctness/workspace win.

That same-process comparison is now runnable through the comparison-only
full-complex symbols in the v6 native image; production artifacts cannot select
them. A pinned WSL Zen 5 20-sample run at batch 32 measured packed speedups of
1.470x, 1.525x, and 1.557x for `(input,kernel)` lengths `(256,65)`,
`(1024,257)`, and `(4096,513)`; maximum packed error was 3.05e-5. The runner
records cold and warm samples,
Schedule/Tile/child digests, CPU identity, affinity, governor visibility,
worktree state, and timing provenance. **Retain, not promote:** this checkout
was dirty, the prior model-specific event map remains promotion-ineligible,
and WSL exposes neither a frequency governor nor valid PMU provenance. A clean
pinned bare-metal Zen 5 `tprof` timing/symbol packet is
still required; the historical v5 selector packet is not relabeled as v6.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`); shared diagnostic `TILE_ASYNC_STAGE_NEGATIVE` added.
**x86 outcome: core parity validated; the async-copy model is not applicable
to physical x86 execution.** The x86 lane lowers tiled
matmul through `TileToX86Pass` `func.call`s and emits no
`tile.async_copy`/`tile.wait_async` — CPU lanes have no async-copy
double-buffering stage model to reconcile. PR #544's required build and unit
lanes validated the shared Tile dialect on an x86 CI host. This transfers no
AVX-512 schedule, Zen 5 performance result, or AMX evidence.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Not applicable.** x86 has no asynchronous-copy or mbarrier lane: the
x86/no-async path is a recorded no-op in the await-sinking contract (W5.2a),
and no x86 lowering consumes `tile.mbarrier.*`, `tile.tma.*`, or the new
registered sync ops. The shared Tile dialect still builds and verifies on
this configuration (combined-driver lit run green in the same 324/0 suite);
no AVX-512/AMX evidence transfers or is claimed.

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**2026-08-16 physical follow-up (`REF-TIER-PHYS-2026-08-16`): implemented for
the solver and four lattice transforms.** `schedule.tridiagonal_solve` selects
`batch_vector_thomas_v1`; the native kernel packs recurrence rows and executes
eight independent systems per AVX-512 fp64 vector before the final fp32 store.
`schedule.coalition_butterfly` carries `(half, sign, stage_order)` and one
AVX-512 Yates consumer implements all subset/superset zeta/Mobius variants in
fp64 working storage. Direct numerical tests cover scalar/tail/full SIMD
batches, non-power-of-two rejection, zero-pivot rejection, and all four
butterfly modes. This is a correctness retain, not a performance promotion;
a clean Zen 5 benchmark packet remains open. Coalition marginal, semivalue,
Boltzmann value, excess, and MEX remain reference/composition follow-ups rather
than five new one-off emitters. No AMX evidence is claimed.
Native solver and butterfly calls now assert their returned status in Target
IR, and the manifest records the shipped C symbols as `device_verified_abi`
rather than overclaiming generated-JIT proof. This is ABI/package evidence;
the existing direct fixtures remain the numerical authority.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**x86 outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions consumed only by `compiler/emit/apple_msl.py`; the Zen 5
AVX-512 scheduled reduction consumes its own `tile.reduce_kernel` lowering and is
unaffected. No x86 schedule, ABI, selector, or evidence changes. Same caution as
NVIDIA for any future `emit/x86_llvm.py` reduce emitter: `maxps`/`maxss` also
suppress NaN, so propagation must be explicit to honour `nan_mode = "propagate"`.

Also recorded for coordination: PR #571 admits Apple GPU into the shared
scheduled reduce contract (`scheduled_kernel.py`, last axis only) and closes
APPLE-DEVICE-EVENT-1 by giving the Apple MPSGraph BMM route an owned command
buffer. Both are Apple-guarded — the shared `scheduled_kernel` gate adds an
`apple_gpu` branch beside the existing x86/ROCm ones and changes neither — and
the runtime edit is in `apple_gpu_runtime.mm`, which no sibling links.

## APPLE-ATTN-BWD-PERF-1-2026-08-16 — backward row-prepass assessment (PR #572 merged)

**Outcome: not applicable to this architecture — no shared contract changed.**
Apple's attention-backward dK/dV split kernel became key-parallel by finally
implementing the `row_prepass` stage that
`attention_contract.plan_attention_backward_workspace` **already declares**
(`row_lse`, `row_delta`, consumed by `dkdv_split` and `dq`). The change is
confined to Apple-private MSL in `apple_gpu_runtime.mm` and its dispatch; no
shared IR, Schedule contract, workspace plan, dtype, ABI, or sibling schedule
was modified. The declared schedule is unchanged — still `split_count = 2` with
ascending `reduction_order = (0, 1)`; only Apple's thread mapping changed, which
is architecture-owned.

**Worth reading anyway, because the shape of the bug is portable.** The declared
workspace stage existed and had no consumer, so every kernel recomputed the row
statistics inline. That is not merely duplicated work: because the statistics
reduce over the whole key axis, an inline-recompute kernel must own an entire
query stream, which capped the dK/dV split at one thread per (partial, KV batch)
— 4 threads at `B1 Hq4/Hkv2 S64`. Implementing the declared prepass raised it to
`2 * kv_outer * Sk` and moved backward from 195 ms to 6.0 ms (~32x), while
keeping single-owner determinism. Any backend whose backward recomputes row
statistics inside its dK/dV kernel should check whether it has the same
structural cap before attributing slowness to memory or instruction mix.

## APPLE-NORM-VJP-1-2026-08-16 — Apple native-VJP registration assessment

**Outcome: not applicable — no shared contract changed.** Apple registered as a
Target consumer in the existing native-VJP normalization plugin and added its own
Metal MSL VJP kernels. The shared registry schema, the `_parse_compiled_norm_backward`
contract, the Graph/Schedule contracts, and every sibling executor are unchanged;
the new code is Apple-private (`apple_gpu_runtime.mm`, one runtime executor, two
`execution_matrix` rows behind one new Apple executor id). This target's own
normalization VJP rows, evidence and selectors are untouched.

Recorded because the *scoping* lesson is portable: the item was scoped as
"registry wiring", but the target had no normalization-backward ABI at all, so
the entry would have been an unconsumed declaration (Decision #29). Any backend
asked to join this boundary should confirm it owns an executable VJP before
declaring a consumer.

## APPLE-NORM-VJP-2-2026-08-16 — reduced-precision normalization VJP assessment

**Outcome: not applicable — no shared contract changed.** Apple's normalization
VJP gained f16/bf16 storage: four new Apple-private C ABI exports
(`tessera_apple_gpu_{rmsnorm,layer_norm}_bwd_{f16,bf16}`), their MSL, the
non-Darwin stub returning 0, and Apple's own ABI registry rows. The shared
native-VJP registry schema, `_parse_compiled_norm_backward`, the Graph/Schedule
contracts, the execution-matrix executor id, and every sibling executor are
unchanged. This architecture's normalization VJP dtype support, evidence and
selectors are untouched.

Two portable notes, recorded because both are cheap to get wrong:

1. **Store rounding is part of the numeric contract.** The kernel accumulates in
   f32 and stores back at the operand's own dtype, and bf16 stores use
   round-to-nearest-even. Truncating instead would bias every gradient in one
   direction — a systematic error, not noise. Any sibling adding reduced-precision
   gradients should match its own runtime's established rounding convention
   rather than defaulting to a shift.
2. **A tolerance derived from storage epsilon is what proves the accumulator.**
   Measured error is ~0.5-0.65 ulp of each format's epsilon, i.e. one rounding on
   store; storage-format accumulation would land near `sqrt(cols)` ulps. A test
   asserting against that second level catches a silently dropped f32
   accumulator, where a hand-picked loose tolerance would absorb it.

A third note is Apple-specific but the shape recurs: adding an export obliges
updating the dylib **freshness gates**, or a runtime built between two landings
exports the older names, passes the staleness check, and then fails at the call
site — which reads as a broken consumer rather than the stale build it is.

---

## Cross-backend record — MC1 matrix-function family (PR #596, 2026-08-20)

**Owning item:** MATRIX-CALCULUS-MC1 · **synchronization key:** `MC1-LINALG-FAMILY`

**Shared contracts changed.** Ten public op registrations (`det`, `logdet`,
`inv`, `solve`, `trace`, `eigh`, `kron`, `vec`, `matrix_power`, `norm`); two
new Graph IR lowering kinds (`linalg_function`, `linalg_multilinear`) with four
new shape rules (`matrix_scalar`, `vec`, `kron`, `eigh`); numeric-policy
entries for both kinds; two diagnostic codes (`E_LINALG_CONTRACT`,
`E_METRIC_CONTRACT`); the `degeneracy_policy` semantic key extended to
eigenvalue gaps and to the nuclear norm's rank condition.

**Outcome for this backend: `not applicable — no lane attempted or implied`.**

The family landed as a *derivative contract* — closed-form VJP+JVP pairs under
the AD law sweep — because that was the gap in the AD stack. No Tile IR
lowering, Target IR op, or kernel was added for any backend, and none is
implied: every one of the ten is registered `backend_kernel: partial` /
reference tier, and each is listed in the Apple no-lane golden and the
single-GPU closeout classifier as an *intentional* reference-only decision with
a stated rationale, so none of them appears on this backend's promote queue as
a phantom blocker.

**What a sibling picking this up would need to decide, per op.** The reference
implementations delegate to LAPACK through numpy, so a native lane is a
vendor-library question rather than a codegen one for `eigh`/`inv`/`solve`
(rocSOLVER / cuSOLVER / Accelerate / MKL), and a "probably never worth it"
question for `det`/`logdet`/`trace`/`matrix_power`, which reduce a small matrix
to one number. `kron` and `vec` are layout/contraction shaped and would ride
existing lanes if anything.

**Numeric policy this backend must match if it does build a lane.**
`linalg_function` and `linalg_solver` declare **f64 accumulate with f32 storage
admitted** — the same conditioning-sensitive policy `linalg_decomposition`
already carries, because `det`/`logdet`/`inv`/`matrix_power` carry a factor of
the condition number. `linalg_multilinear` (`trace`, `kron`) declares the
ordinary f32 accumulator, since neither carries one. A lane that accumulates in
storage precision would silently lose the digits these rules exist to keep.

**Degeneracy contract that travels with the rules.** `eigh`'s eigenvector
coupling `1/(w_j - w_i)` and `svd`'s `1/(s_j^2 - s_i^2)` have no limit at a
crossing; both fail closed under the declared `degeneracy_policy` rather than
emitting `inf`/`NaN`. Any native lane must reproduce that refusal, not paper
over it with an epsilon — a damped coupling returns a finite, plausible, wrong
gradient, which is the failure mode the key exists to prevent.

**Exact-device evidence: none, and none claimed.** Everything in PR #596 was
validated on the host-independent Python reference lane on an M1 Max, where no
AMD GPU, no CUDA device and no AVX-512 exist. No parity claim is made for any
backend.

**x86-specific note.** This backend already models a delegated boundary
explicitly (`tessera_x86.abi_call`, Decision #19), which makes it the natural
place to prototype how a LAPACK-backed matrix-function lane should be
represented in Target IR without hiding the delegation.

## Cross-backend sync `NVIDIA-BARRIER-AT-BIRTH-2026-08-21`

**Not applicable to x86 physical lowering.** The shared role-bearing Tile
contract remains compatible, while barrier binding and local TMA slots are
CUDA-owned. No CPU schedule, runtime ABI, or x86 evidence state changes.

## Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-21`

**NVIDIA-only conformance closure.** NVIDIA's Target-value restriction and
registered-dialect test leave the x86 ABI/call vocabulary and AVX-512 evidence
unchanged.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Outcome: parity already validated; no x86 physical change.** NVIDIA joined the
existing shared spectral and native-product contracts with CUDA-owned cuFFT and
Philox consumers. x86 retains its AVX-512 spectral/Philox packages and evidence;
no CUDA workspace or physical schedule is shared.
