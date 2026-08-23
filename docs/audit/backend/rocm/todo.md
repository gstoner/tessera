---
last_updated: 2026-08-23
audit_role: plan
plan_state: open
scope: ROCm backend implementation and exact-device proof
---

# ROCm backend TODO

`ROCM-CI-HSACO-SERIALIZE-2026-08-23` — **host-free ROCm serialization lane in
CI. Landed.**
PR #619 fixed a total outage of the compiled ROCm lanes (every one died at
hsaco serialization with `lld invocation failed`) that nothing caught, because
the only automated ROCm coverage is `check-tessera-rocm`, which CI does not run.
The natural conclusion — "GitHub has no AMD GPU, so this is uncatchable" — is
wrong. Emitting an hsaco is **compile-time** work: MLIR's ROCDL
`gpu-module-to-binary` shells out to `ld.lld` on the host and never touches a
device.

Measured 2026-08-23 on a HIP-less `tessera-rocm-opt` under `env -i` with no AMD
toolkit installed: with `ROCM_PATH` pointing at a shim holding only a symlink to
stock `lld-23`, the pipeline returns rc=0 and a real
`ELF 64-bit LSB shared object, AMD GPU architecture version 1`
(`e_machine 0xE0` = `EM_AMDGPU`); with `ROCM_PATH` unset it fails exactly as
#619 saw it. The negative control ships as a test, so the lane is known to be
able to observe the regression rather than passing vacuously. Also measured:
MLIR does **not** fall back to `PATH` for `ld.lld`, which is why a
non-interactive shell broke every compiled lane.

Buildable on a hosted runner because `tessera-rocm-opt` is a **separate
executable** — it sidesteps the lean artifact driver that
`TESSERA_BUILD_ROCM_BACKEND=ON` forces onto `tessera-opt`
(`tools/tessera-opt/CMakeLists.txt:95`) — and the ROCm backend's CMake has no
HIP dependency. It configured and built clean with ENABLE_HIP/ENABLE_CUDA off
(212 targets). `lld-23` is required and was not in the lit lane's apt set.

Lane: `rocm-serialize` in `.github/workflows/validate.yml`, gated like `lit`
(label / dispatch / push to main), registered in `OPTIONAL_LANES`
(`tests/unit/test_ci_workflow.py`) and `.github/BRANCH_PROTECTION.md`.
**Verified green on a real GitHub hosted runner** (workflow_dispatch run
32660815243).

**Coverage boundary — this closes a serialization blind spot, nothing more.**
It proves the lane still PRODUCES a code object; it does not prove the object
runs or computes the right answer. Execution evidence needs the real gfx1151
device and stays on that box (Decision #26). Kernels calling AMD's OCML need
`<ROCM_PATH>/amdgcn/bitcode`, absent on a stock runner: measured per kind,
`relu`/`silu` serialize against bare lld while `gelu` (→ `__ocml_tanh_f32`)
does not, so the always-on proof runs the bitcode-free path and the OCML case
is an explicitly skipped test — a stated gap, not a silent pass.

**Relationship to `CI-LIT-BACKEND-DIALECTS-2026-08-12` — that item stays OPEN.**
A review suggested this completes it; it does not. That item covers the 12
ROCm-driven fixtures under `tests/tessera-ir/`, which run through **`tessera-opt`**
and, as its own text records, are "not covered by the existing
`check-tessera-rocm` / `tessera-rocm-opt` lane" — the very driver this lane
uses. Its blocker (the lean-driver conflict in `tessera-opt` on a HIP-less host)
is untouched here. What this item does change is the premise that ROCm CI
coverage requires a HIP host: it does not, for the compile-time half.

Cross-backend sync `CI-BACKEND-CAPABILITY-SKIP-2026-08-23` — **Apple-owned
pytest capability gate; ROCm outcome: parity validated / no change required.**
ROCm already has both halves of this: `tests/tessera-ir/lit.cfg.py:126` derives
`tessera-rocm-backend` from the same `tessera-opt --help` probe, and the ROCm
pytest fixtures gate on `_tessera_opt_path()` / `_rocm_wmma_runtime_available()`.
No HSACO, HIP loader, schedule, or gfx1151 evidence transfers; ROCm test
behaviour is unchanged.


Cross-backend sync `NVIDIA-AOT-PACKAGE-V1-HARDEN-2026-08-22` — **NVIDIA-owned
fatbin/cubin runtime admission; ROCm outcome: not applicable.** The embedded
CUDA artifact/ABI/source identity and NVRTC fallback transfer no HSACO, HIP
loader, cache key, physical schedule, or gfx1151 evidence. ROCm packaging is
unchanged.

Cross-backend sync `NVIDIA-FFT-WORKSPACE-1-2026-08-22` — **NVIDIA cuFFT
foundation landed; ROCm outcome: parity validated/no physical change.** The
versioned explicit-workspace discipline matches ROCm's package ownership model,
but no cuFFT plan, CUDA workspace, schedule, or SuperBear evidence transfers to
gfx1151. ROCm retains its digest-bound Stockham/Bluestein plans and exact-device
proof.

Cross-backend sync `NVIDIA-RNG-PHILOX-CORE-2026-08-21` — **NVIDIA parity
exact-device validated; ROCm outcome: parity validated/no physical change.** NVIDIA now has a
typed four-mode directive and compiler generator using the same explicit
Philox4x32-10 key/counter identity. ROCm retains its independent
`tessera_rocm.philox` distribution generator, HIP runtime route, gfx1151
packages, and exact-device evidence; none of those artifacts or schedules
transfer to sm_120.

Cross-backend sync `APPLE-MINMAX-1-2026-08-23` — **Apple now conforms on
its own hardware; ROCm outcome: no change required, one open question
inherited.** The M1 Max audit measured MPSGraph's `maximum`/`minimum` as
maxNum/minNum — NaN suppressed, ±0 tie to the second operand, 7 of 12
special-value rows wrong — and wrapped it to IEEE. Two further
NaN-suppressing-`max` sites were found and fixed there (the `scatter_f32`
min/max reduce, and a `sqrt(max(0, s))` clamp in the Cl(3,0) norm); the
scatter fix follows *this* plan's rule that a reduce whose result is the
output must propagate NaN. Apple has no `max(stat, eps)` floor and no
Adafactor device kernel, so the eps-floor half of
`JIT-MATH-AUDIT-2026-08-23` had no Apple sibling. Metal evidence transfers
no gfx1151 claim. **Inherited open item:** Apple measured `relu(NaN) = 0`
on device against a NaN-returning reference and deliberately left it — the
ROCm relu lane is equally unaudited. Decide the `relu` NaN contract
fleet-wide before fixing any one backend.

Cross-backend sync `IEEE-MINMAX-CONTRACT-2026-08-23` — **the ±0 tie
contract decision is CLOSED: IEEE-754-2019 fleet-wide (owner decision
2026-08-23), exact-device validated on gfx1151 + AVX-512 host — and, since
2026-08-23, on Apple GPU (M1 Max) too; see `APPLE-MINMAX-1` in the apple
plan.**
`tessera.maximum`/`minimum` now order signed zeros everywhere (max tie
→ +0.0, min tie → −0.0), NaN propagating — the semantics
`arith.maximumf`/`minimumf` already carry. gfx1151: the numpy-emulating
`select(a==b, b, …)` wrapper in `GenerateROCMBinaryKernel.cpp` is
DELETED — probing bare maximumf/minimumf on device showed LLVM's
AMDGPU expansion is already fully IEEE (its ±0 fixup had been dead code
under the wrapper), so the fix removes a cmp+select per element. x86
shim (`avx512_binary_f32.cpp`): both the scalar tail (`a > b ? a : b`)
and the vector body (`vmaxps`) returned the second operand on ties;
fixed via signbit select (scalar) and a bitwise AND/OR tie blend
(vector). Rationale recorded in the tests: numpy is NOT a valid tie
oracle (SSE second-operand vs NEON IEEE — numpy disagrees with itself
across hosts). Pinned by
`test_binary_max_min_signed_zero_ties_are_ieee_ordered` +
`test_binary_minmax_signed_zero_contract` (gfx1151, explicit
expectations) and the x86 sibling
(`test_x86_binary_max_min_signed_zero_ties_are_ieee_ordered`, n=19
covering vector body + scalar tail); the CPU JIT lane was already
IEEE (totality suite). Apple MSL binary max/min is unaudited for the
same contract — recorded in the apple plan.


Cross-backend sync `JIT-MATH-AUDIT-FIXES-2026-08-23` — **both audit
notes from the entry below are now CLOSED as code changes, exact-device
validated on gfx1151.** **(a) Softmax running max switched
`maximumf` → `maxnumf`** (`GenerateROCMSoftmaxKernel.cpp`, both the
local strided loop and the warp-shuffle combine): end-to-end
unobservable — proven by bit-comparing device outputs before/after
across a 7-row special-value matrix (NaN rows, all-NaN, ±inf, -inf
masks, fully-masked, ±0 ties) for `softmax` and `softmax_safe`: NaN
patterns identical, finite values bit-exact — while the ISA drops all
13 `v_cmp_o_f32` + 13 of 15 `v_cndmask` NaN-fixups from the row-max
tree (~26 VALU ops, hsaco 5512 → 5256 B). The reduce kernel KEEPS
`maximumf` (its reduction result is the output; NaN must propagate).
**(b) Adafactor eps/tiny floors switched `maxnumf` → `maximumf`** (18
sites in `GenerateROCMOptimizerKernel.cpp`): the reference floors are
`np.maximum` (optim.py), so a NaN second-moment statistic must surface
as NaN in every update it feeds — maxnumf silently laundered it into
eps, giving finite-but-wrong updates to every parameter sharing the
poisoned row/col. New exact-device test
(`test_adafactor_factored_nan_gradient_propagates_like_reference`)
proves a NaN gradient poisons the full row+col exactly as the
reference does; the identical defect existed in the x86 AVX-512 shim
(`std::fmax` → NaN-propagating floor helper, same test added there;
the x86 rerun also caught a pre-existing crash-shaped hazard: see the
x86 plan). The Philox `maxnumf(u1, floor)` clamp is intentionally kept
— Philox uniforms cannot be NaN, and maxnumf is a bare v_max. Apple
siblings (MSL optimizer/softmax kernels) are unaudited for the same
patterns — recorded in the apple plan. Gates: optimizer/softmax/
binary/reduce compiled suites + `check-tessera-rocm` 63/63 green.


Cross-backend sync `JIT-MATH-AUDIT-2026-08-23` — **gfx1151 exact-device
math-correctness probes: special-value semantics verified; one deliberate
cross-target divergence recorded (no code change).** Probed on this box
through the compiled lanes: **reduce max/min** propagate NaN and handle
-inf bit-exactly vs numpy; **softmax** handles -inf mask entries (masked
→ 0, fully-masked row → NaN, matching the naive reference); **binary
maximum/minimum** propagate NaN (pre-existing test) and, newly pinned
(`test_binary_max_min_signed_zero_tie_returns_second_operand`), return
the SECOND operand on a ±0 ordered tie — a deliberate emitter choice
(`GenerateROCMBinaryKernel.cpp` wraps maximumf/minimumf in
`select(a==b, b, …)` to emulate numpy-on-x86/SSE). That diverges from the
arith.maximumf/minimumf IEEE tie ordering the x86 CPU JIT lane executes
(max tie → +0, min tie → −0; pinned in the totality suite). **Open
decision:** `tessera.maximum/minimum` ±0-tie semantics should be one
recorded contract across targets (Decision #21a); both pins name each
other so either resolution is a visible two-sided change. Two audit
notes, no action taken: (a) flash-attn fwd/bwd running max uses
`maxnumf` (NaN-suppressing) while the softmax kernel uses `maximumf`
(NaN-propagating) — end-to-end results agree because exp(NaN) still
propagates, and maxnumf is cheaper pre-gfx12 (bare v_max, no NaN fixup);
switching softmax's running max to maxnumf is a small verified-equivalent
optimization candidate. (b) `GenerateROCMOptimizerKernel.cpp` eps-clamps
via `maxnumf(x, eps)`, which launders a NaN input into eps rather than
propagating it — if NaN gradients should surface, that wants
`maximumf` or an explicit isnan gate.


Cross-backend sync `JIT-ELEMENTWISE-LINALG-2026-08-21` — **shared
`tessera_jit` pipeline change; ROCm outcome: not applicable (backend);
host lane validated (2026-08-23).** The ROCm device paths
(`tessera-rocm-executable` / hsaco lanes) do not consume `tessera_jit`; their
state-machine path independently scalarizes tensor slots before emitting GPU
arith and retains its gfx1151 exact-device proof below — re-confirmed in the
same session: `test_rocm_state_machine_exec.py` **5/5** on gfx1151.  The
host-CPU JIT lane on this box is the x86 lane; its widened
arithmetic/math totality rerun closed on this host (22/22 + 182-test
packet) — details, and the host-dependent numpy signed-zero oracle
defect the rerun caught, are recorded in the x86 plan's entry for this
key.


Cross-backend sync `JIT-VECTORIZE-UNGATED-2026-08-23` /
`JIT-CACHE-BLOCK-2026-08-23` / `JIT-MATH-AUDIT-2026-08-23` — **shared
`tessera_jit` boundary/pipeline changes; ROCm outcome: not applicable
(backend); HIP-coexistence hazard fixed and pinned on this box.** The
x86 plan owns these keys; the ROCm device paths do not consume
`tessera_jit`. The directly ROCm-relevant part: the vectorize lane's
old `libmlir_c_runner_utils` dlopen linked a second (dynamic) libLLVM
into the process and made a later `dlopen("libamdhip64.so")` segfault —
the gfx1151 state-machine tests died the moment they loaded HIP after
a vectorized compile. `memrefCopy` is now registered in-process; the
full packet including `test_rocm_state_machine_exec.py` (5/5
exact-device) passes with the vectorize lane both on and off, and a
probe pins vectorized-compile → heap-stress → HIP dlopen coexistence.

Cross-backend sync `W4-SM-ROCM-2026-08-21` — **W4-PRODUCT-1 exact-device
irreducible-state-machine row: VALIDATED (gfx1151).** New
`--generate-rocm-state-machine-kernel` (ROCm backend) lowers any
`--tessera-autodiff-paired` function carrying a `bounded_state_machine_v1`
loop — forward and generated backward alike — to one per-thread gpu.func:
each thread runs the whole PC machine on its element (SIMT divergence
carries per-element control flow), tensor slots scalarize to f32, the
structured-CFG digest and residual policy are stamped on the kernel, and
`cf.assert` becomes a per-thread STATUS conjunction the HOST enforces
(all-ones or the launch is rejected — max_steps exhaustion cannot pass
silently). Exact-device rows: both entry paths of a two-entry irreducible
SCC, forward (tanh∘tanh / tanh) and recompute_all backward vs the analytic
oracle, n=300 (non-multiple of the 256 block dim), rtol 1e-5
(`tests/unit/test_rocm_state_machine_exec.py`; lit:
`tests/tessera-ir/control_flow/w4_state_machine_kernel{,_reject}.mlir`).
Correctness-only — WSL; bare-metal timing remains open per W4.3.
Review follow-up (PR #605 → follow-up PR): the family is now wired into the
CANONICAL executable pipeline — `control_state_machine` in the C++
`tessera-rocm-executable` family list + generator chain, `FAMILY_PLUGINS`,
and the pipeline-registry spec (cross-registry totality test green); the
exec test compiles through that registered pipeline, not a hand-assembled
pass list. Digest binding is total (a machine with no digest fails closed;
multiple distinct digests stamp an ordered `structured_cfg.digests` array),
and interior rank-1 i1/int tensors (cmpf→select per-element selection)
scalarize to their element types — proven by a fifth exact-device row
(data-dependent select machine) plus lit positives/negatives.


Cross-backend sync `AD-DATUM-POLYGAMMA-2026-08-21` — **autodiff reference
numerical policy, wave 3; ROCm outcome: parity validated (exact-device,
gfx1151).** lgamma/digamma switch to datum-derived pairs over the new
polygamma tower (k=1 slopes are the displaced hand VJPs' series verbatim;
values mirror the canonical forwards bit-for-bit; n ≥ 2 rungs are new
machine-precision capability), and the rmsnorm derived pair gains the
optional γ operand (tape-reverse with γ was broken before retirement).
Validated on this box: the four ROCm autodiff/spectral device lanes +
six x86-shared loss lanes, 95/96 green against the switched registry —
the 1 failure is the pre-existing `test_autodiff_rocm_matmul_composed`
provenance-metadata assert (stale `implementation` name from the
E2E-REAL-6 tracer rename; numerics pass and the lane executes
`native_gpu`/`hip_runtime`).


Cross-backend sync `AD-RETIRE-2-2026-08-20` — **autodiff reference numerical
policy, wave 2; ROCm outcome: parity validated (exact-device, gfx1151).**
The retirement wave extends: softmax/logsumexp/rmsnorm-core production pairs
are jet-derived first-order specializations, and six new datum entries
(tan/asin/acos/erf/erfc/rsqrt) switch those ops to datum-derived pairs —
all dtype-preserving, with the displaced VJP guard conventions carried (two
more jvp/vjp boundary-guard inconsistencies fixed: rsqrt and the asin/acos
guarded-slope form). Validated on this box: native-JVP-compiled + ROCm
spectral-backward-exec + native-CPU vertical + spectral target binding
(24 tests) and the full law sweep, all green against the switched registry.


Cross-backend sync `AD-RETIRE-1-POINTWISE-2026-08-20` — **autodiff reference
numerical policy; ROCm outcome: parity validated (exact-device, gfx1151).**
PR #600 retires the 13 holonomic-ODE pointwise hand JVP/VJP pairs behind the
`DerivativeContract` datum. Two shared-contract changes: forward/reverse
reference rules now preserve the input dtype (a float32 trace stays float32;
the old factory promoted to float64), and log/sqrt carry ONE declared domain
guard for both modes (|x| < 1e-12 — previously the mode pair disagreed at the
boundary). ROCm impact: the native lanes compare against this reference and
both sides of every parity comparison read the same updated rules; validated
on this box (gfx1151): `test_native_jvp_compiled.py` +
`test_rocm_spectral_backward_exec.py` + native-CPU vertical, 24 tests green
against the retired registry. No kernel change; boundary inputs below 1e-12
are outside every sampled parity envelope.


Cross-backend sync `APPLE-RUNTIME-SINGLE-IMAGE-2026-08-19` — **Apple runtime
loading; ROCm outcome: not applicable.** The single-image slice fixes duplicate loading of the Apple GPU runtime.
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
ROCm impact: none. gfx1151 loads HIP/hsaco through its own path, not through
`_apple_gpu_dispatch`. Worth noting for symmetry: the duplicate-image class of
defect needs an ObjC/Mach-O runtime that registers classes at load, which the
ROCm lane does not have. No gfx1151 retest required and no device evidence is
produced or claimed.


Cross-backend sync `APPLE-STUB-BINARY-OPCODES-2026-08-19` — **shared runtime
contract; ROCm outcome: not applicable.** The portable-stub opcode slice fixes a silent wrong-answer class in the Apple
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
ROCm impact: none. The gfx1151 elementwise binary lane is separate code
(`_execute_rocm_compiled_binary` over a compiler-generated hsaco, dispatched by
`_ROCM_BINARY_OPS`) and does not call the Apple symbol. Checked for the same
defect shape: that lane binds both operands positionally and raises
`"binary requires two operands (a, b)"` rather than falling through, so it has
no silent-default arm. No gfx1151 retest required and no device evidence is
Cross-backend sync `ZERO-FUNCTION-CANDIDATE-2026-08-19` — **shared frontend ABI
and diagnostics; ROCm outcome: not applicable today; parity by construction.**
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
ROCm impact: none today. On a non-apple_gpu target an AST emission failure
re-raises `TesseraJitError` at decoration rather than deferring to the tracer,
so gfx1151 programs never reach the new code path or the new diagnostic. The ABI
recovery only replaces an empty tuple with the signature-derived one, so no
gfx1151 behaviour changes. No gfx1151 retest required and no device evidence is
produced or claimed.


Cross-backend sync `SCALAR-SIDE-ORDERING-2026-08-19` — **shared Graph IR
runtime contract; ROCm outcome: not applicable today, fails closed by
construction.** The `scalar_side` slice (PR #589) makes the Graph IR lifted-scalar form carry
operand order. `graph_ir._OpExtractor._try_map_binop` lifts a literal out of
either side of a `BinOp` into the `scalar` attribute and records the side; until
now no code in `python/`, `src/`, or `tools/` read that record (Decision #29), so
`2.0 - x` and `x - 2.0` emitted indistinguishable IR and any consumer binding
`scalar` as the right operand computed `x - 2.0` for both — sign-flipped for
`sub`, reciprocal for `div`, with no diagnostic. Shared contract changed: a lone
`scalar` means the RIGHT operand, `scalar_side="left"` requests the mirrored
binding, and any other value is rejected rather than guessed (Decision #21).
ROCm impact: none. `runtime._execute_rocm_compiled_binary` binds both operands
positionally from the op's `operands` list and raises
`"binary requires two operands (a, b)"` when fewer are present, so the
lifted-scalar form cannot reach the gfx1151 binary lane at all — an absent side
cannot be misread there. No gfx1151 retest required and no device evidence is
produced or claimed. Follow-up: if the ROCm binary lane later accepts a scalar
kwarg form, it must honor `scalar_side` before doing so, since `_ROCM_BINARY_OPS`
covers sub/div/pow/mod/floor_div.


Cross-backend sync `AD-LAW-SERIES-2026-08-19` — **shared reference rules and
test infrastructure; Rocm outcome: parity validated, no gfx1151 evidence changed.** The AD-LAW series (PR #588)
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
ROCm impact: the gfx1151 `bwd_hardware_proven` rows (flash_attn, selective_ssm) and the spectral backward packages differentially compare against these oracles. The spectral JVP changes affect FORWARD-mode reference values only; the hand-written spectral VJPs those packages are compared against are untouched. `test_rocm_norm_compiled.py` pins explicit eps and is unaffected. No gfx1151 retest required; no device evidence produced or claimed. Follow-up: if the spectral family later adopts derived forward references on device, re-run the AVX-512/gfx1151 spectral packets. The previously recorded open spectral/quantize swallow findings are
therefore CLOSED; `_OPEN_FORWARD_KEY_SWALLOWS` (42 entries from the tape
positional-routing scan) remains the open set.


Cross-backend sync `AD-LAW-1-SHARED-ORACLE-2026-08-18` — **shared test
infrastructure; ROCm outcome: parity validated, no gfx1151 evidence
changed.** AD-LAW-1 (PR #584) adds law oracles (adjoint `⟨Jv,u⟩ = ⟨v,Jᵀu⟩` +
canonical-forward chain check) over the shared numpy reference JVP/VJP
registries — the oracle lane the gfx1151 `bwd_hardware_proven` rows
(flash_attn, selective_ssm) and norm/attention backward packages
differentially compare against — plus the byte-gated `autodiff_law_audit`
dashboard. Reference-rule fixes in the same PR: `jvp_rmsnorm` eps default
(1e-6 → the forward's 1e-5) and `jvp_clamp` swallowing canonical
`min`/`max`. ROCm impact: `test_rocm_norm_compiled.py` pins `rmsnorm_safe`
at explicit eps=1e-6 (unchanged), and VJP-side defaults did not move, so no
gfx1151 retest is required; no device evidence is produced or claimed by
this gate. Open shared follow-up: 20 pinned swallowed-kwarg findings
(`test_autodiff_laws.py`) await triage. *Triage update, same key (AD-LAW-1b):* reference JVP fixes landed for `clip` alias deafness, `add`/`mul` unary-`scalar`, and fft/ifft/rfft/irfft `norm` handling (√n-wrong under `norm="ortho"`); five entries benign-classified with recorded reasons; the open set is now the stft/istft/spectral_conv and quantize families only, riding their owning family reviews. *Spec-growth update, same key (AD-LAW-1c):* law coverage roughly doubled (109 adjoint / 87 chain rows green incl. attention, spectral-complex, structural, and loss families); two more silent reference JVP defects found and fixed — `lgamma` (derivative was a dead stub returning 0) and `digamma` (whole JVP was an identity placeholder) — plus `jvp_cast` crashing on canonical dtype strings. Forward-mode reference oracles for those three ops changed; reverse-mode VJPs are untouched, so no backend backward package is affected. Reflection formulas added to the shared polygamma helpers (`_digamma_positive`/`_trigamma_positive`): the upward recurrence advanced by 1 per step, so a valid input like -1e9+0.5 spun for ~10^9 iterations — a live defect in the **reverse** path too, since `vjp_lgamma`/`vjp_digamma` already call these. Now O(1) on the whole real line, exact against the canonical forward, poles -> nan. — the
fft/stft `norm` swallowing is the family most relevant to the ROCm spectral
backward packages if its triage changes reference outputs.

Cross-backend sync `W4-DYNAMIC-EFFECT-NONLINEAR-CFG-2026-08-18` — **shared
contract parity; gfx1151 evidence unchanged.** Dynamic region state now uses
bounded per-slot data tapes plus recorded logical-shape tapes. Polynomial shape
guards require complete concrete witnesses and do not enter Presburger proofs.
Variadic branch state reaches the structured CFG carrier. Only compiler-owned
extent assertions are replay-safe; mutation/RNG/I/O/ordered collectives remain
fail closed. A bare-metal gfx1151 irreducible/dynamic-state packet is still
required.

Cross-backend sync `W2.4-E2E6-SYMBOLIC-2026-08-18` — **shared parity
validated; gfx1151 evidence unchanged.** Relational Tile legality now has one
staged production authority, while old CLI passes are wrappers. Pure static
annotations abstract-trace before AST compatibility capture. NVIDIA
Lion/DeltaNet dispatch moved into family plugins; ROCm keeps its independent
gfx1151 state-lineage packages and inherits no CUDA evidence.

`W1-W3-AUTHORITY-CLOSEOUT-2026-08-18` — **gfx1151 shared-contract parity and
current reverse-package ownership closed; no new device evidence.** ROCm
semantic selectors are ODS-owned, bare Tile fragments are rejected, WarpSpec
uses typed Schedule ancestry, and whole-program memory activity is conservative.
All currently admitted single-op ROCm VJPs—including losses, matmul composition,
and selective SSM—resolve through explicit plugins; unmatched families fail
closed. Forward producer ownership and gfx1200/gfx1250 physical proof remain
open and architecture-gated.

`W4-PRODUCT-1-RESIDUAL-CONTRACT-2026-08-17` — **shared carrier and bounded
gfx1151 correctness consumer landed.** SAVE/HYBRID checkpoint sets, CFG identity,
and residual identity now survive Graph→SCF validation. Shared paired AD now
consumes dynamic branch-local residual extents through zero-extent inactive
sentinels, bounded SAVE and sparse HYBRID `while` state tapes, and
bounded-dynamic counted-loop tapes. Source-CFG SCC analysis now distinguishes
acyclic, reducible, and true multi-entry irreducible graphs. Bounded pure
native graphs lower to a typed program-counter state machine with nested
canonical structured bodies and mixed control/tensor state. Saved dynamic slots
now require total data/shape-tape envelopes; unbounded, unsupported-region, and
unrecorded effectful forms remain fail closed. The existing gfx1151 WSL packet binds
paired-IR/CFG/residual digests and executes native HIP children without Graph
re-entry or predicate replay. It is correctness evidence only; bare-metal
device-clock/profiler timing remains required for selector promotion. An
exact-device irreducible-state-machine row remains required before physical
ROCm execution of that new form is claimed.

`E2E-REAL-6F-OPTIMIZER-VJP-2026-08-17` — **bounded gfx1151 optimizer reverse
authority complete.** Explicit plugins now own SGD, Momentum/Nesterov, and
Adam/AdamW. Each package binds the one-execution tracer certificate, functional
state/cotangent lineage, typed `schedule.optimizer_vjp`, exact
`tile.training_kernel`, and existing HIP physical consumer. Runtime receives
no Graph operation dictionary. gfx1200/gfx1250 remain fail closed and
bare-metal selector timing remains independent.

`E2E-REAL-6E-STATEFUL-VJP-2026-08-17` — **bounded gfx1151 Adafactor and
sequence-mixer reverse authority complete.** Explicit plugins now own
factored/full Adafactor and causal gated/Kimi/modified DeltaNet backward; the
compiler route binds tracer identity, one-execution proof, state/workspace
lineage, Schedule, and exact Tile identity without passing Graph metadata to
the runtime. Existing WSL-visible gfx1151 physical tests remain correctness
evidence, not bare-metal selector-grade timing. gfx1200/gfx1250 remain
fail-closed for these packages.

`E2E-REAL-6D-LION-VJP-2026-08-17` — **bounded gfx1151 Lion reverse authority
closed.** Lion now enters the shared non-reexecuting family plugin from a
tracer-owned two-result flat Graph operation. The plugin binds its structural
frontend certificate, source Graph digest, functional/no-alias state lineage,
typed `schedule.lion_vjp`, exact `tile.training_kernel`, and aggregate artifact
identity before runtime. The HIP executor receives no Graph metadata and the
existing WSL-visible gfx1151 numerical proof passes. This does not promote a
selector or transfer evidence to gfx1200/gfx1250; those targets remain
fail-closed pending architecture-owned packages and device packets.

`E2E-REAL-6C-ATTENTION-VJP-2026-08-17` — **public gfx1151 canonical rank-4
attention reverse authority closed for the admitted envelope.**
`flash_attn`, GQA, and MQA now enter one native-VJP family plugin from
tracer-produced Graph IR. The plugin lowers the shared tensor-valued
dQ/split-dK-dV/fixed-reduction program, prebuilds its exact five-entry gfx1151
image, and binds parent Graph, Schedule, Tile, image, and aggregate digests
before runtime. The runtime receives only that typed package; the former
`JitFn` ROCm attention artifact constructor is deleted. The work also repaired
a real ragged-causal defect: shared references, generated forward/backward
kernels, the direct lowering, and the shipped HIP ABI now use bottom-right
query alignment. The affected 67-test cohort passes on the WSL-visible
gfx1151. The public differential migration admits zero dropout only; the
existing physical dropout carrier is retained, but active dropout remains
fail-closed at this frontend boundary until keyed replay is non-reexecuting.
This is correctness/authority evidence, not bare-metal timing;
gfx1200/gfx1250 remain fail-closed and rank-3 MHA remains a compatibility path.

`E2E-REAL-6B-SPECTRAL-VJP-2026-08-17` — **public gfx1151 compound spectral
reverse execution closed for the bounded physical envelope.**
`spectral_filter` and unbroadcast full `spectral_conv` now enter one native-VJP
family plugin from tracer-produced Graph IR. The package binds source,
Schedule, and Tile digests, builds exact gfx1151 HSACO before runtime, and the
runtime consumes only that serialized image. Public filter and convolution
numerical tests pass on the WSL-visible gfx1151. The exact-target capability
registry now also reflects the already-proven compound spectral forward rows,
including logical complex64 carried as interleaved fp32. This is correctness
and authority evidence, not selector-grade timing; broader broadcasting,
axes/dtypes, STFT/ISTFT backward, and gfx1200/gfx1250 remain open.

`GFX1151-CALIB-BAREMETAL-2026-08-16` — **producer hardened; bare-metal packet
still hardware-gated.** The 2026-08-15 WSL figures (186.8 GB/s, 47.27 fp16
WMMA TFLOP/s, 50.22 bf16 WMMA TFLOP/s) are now explicitly
`provisional_pruning_only`; `target_perf.load_corpus()` rejects them instead of
silently assigning measured selector authority. The calibration runner records
independent host-wall and HIP-event samples and finalizes only a clean-source,
exact-gfx1151, bare-metal measurement paired with fresh-process ROCprofiler
dispatch/activity proof. This WSL host exposes `/dev/dxg`, not `/dev/kfd`, so it
cannot replace the packet. Run the documented two-step capture/finalize command
on bare-metal gfx1151, then review and commit the resulting selector-grade
corpus before changing any selector.
Finalization also binds each result to its expected kernel name, image digest,
raw timing samples, work count, and recomputed metric; unrelated profiler
dispatches or a requested/self-reported architecture cannot promote evidence.

Cross-backend sync `LAYOUT-SCHEDULE-OBJECT-2026-08-16` — **shared carrier and
gfx1151 physical role proof validated.** The first
native layout ABI and executable GQA fold transfer no AMD physical layout or
raster choice. SO-1 now owns the content-addressed action/edge/role/residency
value; SO-2 registers loop-carry-safe symbolic producer/consumer roles on Tile
pipelines and mbarriers. `rocm-wave-lds-pipeline` emits the role-bearing state
and `rocm-wave-lds-legality` consumes it. The exact CAKE §5.5 cohort passed 8/8
on gfx1151 with no skips; NVIDIA barrier-at-birth work does not block this AMD
proof.
Existing raster output remains bit-identical and row-major remains selected.
Nested Schedule resource metadata is now frozen before hashing, and dynamic
rearrange/GQA-fold inference preserves ranked `?` dimensions. Neither change
selects a new AMD layout or raster policy.

Cross-backend sync `ATTN-BWD-ARCH-2026-08-16` — **ROCm decomposition retained.**
The canonical forward/recompute → prepass → split dK/dV → fixed-order reduction
→ dQ program passed 23 host/device/package checks on gfx1151; five Apple-only
cases skipped for their declared Darwin gate. No unused workspace or missing
stage was found, so this slice makes no ROCm kernel change or performance
promotion.

Cross-backend sync `X86-PASS-DIALECT-DEPENDENCY-2026-08-16` — **shared build
parity validated; no ROCm physical follow-up.** The shared pass library now
models the optional hardware-free x86 Target dialect as a declared MLIR pass
dependency and fails closed when it is absent. ROCm-only compilation is
unchanged. The rebuilt combined LLVM 23 driver advertises both `rocm-backend`
and `x86-target-ir`; its shared lit suite passes 329 enabled tests with 52
configuration-gated Apple/NVIDIA fixtures, and the standalone ROCm suite
passes 61/61. The transform library now consumes the sole ODS Schedule dialect
from `TesseraScheduleIR`, eliminating the duplicate namespace that crashed five
composite x86 pipelines. No ROCm Target ABI, schedule, selector, or gfx1151
evidence changed.

Cross-backend sync `PDE-EXACT-CONTRACT-2026-08-14` — **shared exact semantic
authority landed; gfx1151 physical follow-up required.** The compiler now owns
a typed constant-coefficient PDE carrier, exact-rational principal-symbol
classification, and a fail-closed centered-FTCS diagonal-diffusion certificate
with non-unit spacing. The Graph ODS/passes, broader certificates, typed
stencil/boundary/halo package, transport, and gfx1151 packet remain open.

Cross-backend sync `DIST-SHARD-HVP-2026-08-14` — **shared SSA/product
foundation landed; gfx1151 transport evidence remains open.** Planned
all-gather/all-reduce/reduce-scatter/all-to-all conversions now become real
Graph→Schedule→Tile SSA carrying digest, subgroup, region, and deterministic
matching-round identity. Exact compiler forward-over-reverse HVP is also live.
This slice does not claim RCCL execution, a native gfx1151 HVP package, or a
multi-rank packet. Replicated-to-tiled local-shard typing and RCCL launcher
binding remain required before physical promotion; gfx1200/gfx1250 stay
fail-closed.

Cross-backend sync `E2E-REAL-6-NATIVE-VJP-2026-08-14` — **gfx1151
normalization ownership migrated; no selector change.** RMSNorm and LayerNorm
reverse execution now enters one declared native-VJP family plugin whose
Graph/Schedule/Tile/Target ownership names the gfx1151 normalization consumer.
`JitFn` binds arguments and records the plugin result but no longer constructs
this ROCm package. Existing exact-device numerics remain the evidence; other
ROCm backward families and package producers remain compatibility paths.

Cross-backend sync `AMD-ISA-DTYPE-2026-08-14` — **owning ROCm foundation
landing; physical expansion required.** The compiler now distinguishes RDNA3.5
gfx1151, RDNA4 gfx1200, and CDNA5 gfx1250/gfx1251 by architecture family,
wave size, matrix pipeline, dense/sparse legality, operand and accumulator
formats, scale lineage, exact instruction mnemonic, and implementation state.
MI455X is gfx1250 and MI430X is gfx1251; they share the CDNA5 low-precision
instruction ABI, gfx1251 adds FP64 WMMA, and they retain distinct cost-model
identities. The existing f16/bf16 K32 gfx125x
fragment reaches ROCDL under the new CDNA5 family. Remaining ROCm work is typed
FP8/BF8 and IU8 materialization, sparse SWMMAC, F8F6F4/FP4/MX scale operands,
gfx1251 FP64 WMMA, and independent gfx1200/MI455X/MI430X exact-device packets.
No gfx1151 schedule or evidence transfers.

The shared `OP-DTYPE-FLOW-1` generated matrix is the synchronization surface
for frontend → Graph → Schedule → Tile → ROCm datatype flow and TSOL.
Generic ROCm manifest rows apply only to gfx1151; gfx1200, gfx1250, and gfx1251
remain architecture-specific and fail closed unless they have their own row.
Policy-derived legality is labelled `legal_only`, not executable support.

> **CDNA 5 breaks the `family ⇒ (matrixOp, waveSize)` model (assessed 2026-08-13).**
> AMD's CDNA5 ISA guide contains **zero** MFMA/SMFMAC mnemonics — it uses
> `V_WMMA_*` / `V_SWMMAC_*` — and states in Chapter 1 that the device
> **supports only wave32**. `ROCMFragmentLayout.h:192` hardwires the CDNA branch
> to `"mfma", 64`. Not a live bug (CDNA 5 is not in the arch list, so it fails
> closed), but adding CDNA 5 is a change of SHAPE, not an arch append. See
> [`docs/reference/isa/PRIMARY_SOURCES.md`](../../../reference/isa/PRIMARY_SOURCES.md).

Cross-backend sync `CI-LIT-BACKEND-DIALECTS-2026-08-12` — **combined-host
validation closed; a dedicated ROCm CI lane remains open.**
The `Validate / lit` lane was dead from 2026-08-11 to 2026-08-12 (pytest
collection aborted on a missing `ml_dtypes`, fixed in #554), and the first
green-collection run exposed that the lane's `tessera-opt` registers neither
`tessera_x86` nor `tessera_rocm`: 27 of 367 fixtures failed with
``Dialect `tessera_rocm'/`tessera_x86' not found``. This PR restores the x86
half only. **ROCm cannot be added to that binary on a HIP-less runner**:
`tools/tessera-opt/CMakeLists.txt:95` forces `TESSERA_OPT_LEAN_ARTIFACT_DRIVER`
when `TESSERA_BUILD_ROCM_BACKEND=ON` with `ENABLE_HIP`/`ENABLE_CUDA` off, and
the lean arm drops core `TesseraIR`/`TesseraPasses` plus the x86 Target IR
(`:133` gates on `NOT lean`). Enabling it would trade ~12 ROCm fixtures for a
larger core+x86 regression. This is a CI coverage gap, **not** a ROCm compiler
defect, and it transfers no gfx1151 evidence.

**These 12 fixtures are now `// REQUIRES: tessera-rocm-backend`.** They were the
only ROCm-driven fixtures in `tests/tessera-ir/` missing the gate their siblings
already carry (`phase3/streaming_attention_backward_rocm{,_invalid,_nobias}.mlir`
have had it all along), so in a ROCm-less driver they hard-failed instead of
reporting UNSUPPORTED. The gate is derived, not asserted: `lit.cfg.py:126`
probes `tessera-opt --help` for `tessera-lower-to-rocm`, which is present
exactly when `TesseraROCMIR`/`TesseraROCMConversion` are linked. In a
ROCm-enabled build they run as before — **the gate hides nothing on a host that
can execute them**, which is why the coverage obligation below still stands.

Gated fixtures (all under `tests/tessera-ir/`, driven by `tessera-opt`, so
**not** covered by the existing `check-tessera-rocm` / `tessera-rocm-opt` lane):
`phase2/rocm_fragment_{ragged_bounds,strided_k}.mlir`,
`phase2/rocm_kind_enums_invalid.mlir`,
`phase2/rocm_semantic_attrs{,_reduction}_invalid.mlir`,
`phase2/rocm_typed_fragment_{composition,composition_invalid,rdna4_int4}.mlir`,
`phase2/e2e_matmul_scheduled_rocm_consumer.mlir`,
`phase3/streaming_attention_modifiers_rocm.mlir`,
`phase_f4/{es_low_rank_correction,spectral_backward}_rocm_native.mlir`.

**Reproduction — ROCm host (Strix Halo / gfx1151), host-free, no device needed:**

0. **Treat the `REQUIRES` gate as a debt marker, not a resolution.** On the
   Strix Halo box these fixtures must actually RUN — a lit summary showing
   them UNSUPPORTED there means the ROCm build is misconfigured, not that the
   work is done. Assert the feature is live before trusting a green run, from
   the build ledger rather than the pipeline list (see the probe table in step
   2 for why the distinction bites):
   `./build/tools/tessera-opt/tessera-opt --tessera-build-info | grep -qw rocm-backend`.

1. Reproduce the CI binary's blindness locally. A stock `tessera-opt` built
   without the ROCm flag must now report these fixtures UNSUPPORTED (before the
   gate they hard-failed); confirm the driver genuinely lacks the dialect
   rather than the fixtures having rotted:
   ```
   cmake -S . -B build-noro -G Ninja -DTESSERA_BUILD_APPLE_BACKEND=ON \
     -DTESSERA_BUILD_X86_BACKEND=ON \
     -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm
   ninja -C build-noro tessera-opt
   ```
2. Confirm the fixtures pass in a full ROCm build (HIP present ⇒ lean arm not
   taken ⇒ core + ROCm co-registered), which is the canonical configure in
   `CLAUDE.md`. **Add the x86 backend to the same build**: Strix Halo is the
   fleet's x86 host as well as its ROCm host (Zen 5 + gfx1151 in one box), and
   with `ENABLE_HIP=ON` the `:95` lean condition is not met, so one driver can
   carry core + ROCm + x86 together. That makes this the only configuration in
   the fleet that runs **both** fixture families in a single `lit` invocation —
   the CI lane structurally cannot, which is the whole reason this item exists:
   ```
   cmake -S . -B build -G Ninja -DTESSERA_ENABLE_HIP=ON \
     -DTESSERA_BUILD_ROCM_BACKEND=ON -DTESSERA_BUILD_X86_BACKEND=ON \
     -DCMAKE_PREFIX_PATH=/opt/rocm/core \
     -DMLIR_DIR=/usr/lib/llvm-23/lib/cmake/mlir -DLLVM_DIR=/usr/lib/llvm-23/lib/cmake/llvm
   ninja -C build tessera-opt
   # Preflight: assert BOTH dialects are linked, independently. Two required
   # checks, not one alternation — `grep -E 'a|b'` exits 0 on either match.
   INFO=$(./build/tools/tessera-opt/tessera-opt --tessera-build-info)
   grep -qw rocm-backend  <<<"$INFO" || { echo "ROCm dialect NOT linked"; exit 1; }
   grep -qw x86-target-ir <<<"$INFO" || { echo "x86 dialect NOT linked";  exit 1; }
   lit tests/tessera-ir/ -v
   ```
   **Probe the feature ledger, never the pipeline list.** `--tessera-build-info`
   prints `TESSERA_OPT_BUILD_FEATURES`, which is generated from the CMake
   feature ledger and so cannot drift from what was linked
   (`tools/tessera-opt/tessera-opt.cpp:309-328`). Pipeline names are **not** a
   valid proxy, and the two backends differ in a way that will mislead anyone
   who assumes symmetry:

   | Probe | Registered by | Valid dialect proxy? |
   |---|---|---|
   | `tessera-lower-to-rocm` | `Tessera_ROCM_Backend/lib/Conversion/Passes.cpp:501` — the backend library | **yes**, it is linked only with the backend |
   | `tessera-lower-to-x86` | `src/transforms/lib/Passes.cpp:416` — **core `TesseraPasses`** | **no**, present with or without `TesseraX86IR` |

   `X86-DIALECT-LOAD-CRASH-2026-08-12` is the proof this matters: in run
   31648897366 `tessera-lower-to-x86` was registered while the x86 dialect
   crashed on load. A pipeline-name probe would have reported the driver
   healthy. (This does **not** impeach `lit.cfg.py:126`, whose ROCm feature
   uses the left-hand row and is sound — but any future *x86* lit feature must
   use the ledger, not `tessera-lower-to-x86`.)
   Expect **zero** UNSUPPORTED among the 12 gated fixtures here. If any fixture
   fails **there**, it is a real ROCm (or x86) defect and outranks the CI gap —
   file it as its own item before touching CI.
3. Choose and own the CI recovery. Preferred: a second configure+build in the
   `lit` job producing a lean ROCm `tessera-opt` that runs only the ROCm-tagged
   fixtures (mirrors how `rocm compiler (host-free LLVM/MLIR 23)` already builds
   a separate driver). Alternative: relax the `:95` lean condition so a
   HIP-less ROCm build keeps core+x86 registration — larger blast radius,
   needs its own Decision-#19/#31 review, do not attempt as a drive-by.
4. **Completed 2026-08-14:** the production HIP build was rebuilt with both
   backend flags. Its ledger reports `rocm-backend` and `x86-target-ir`; the
   shared suite reports 321 enabled passes, 52 configuration-gated tests, and
   zero failures across 373 discovered tests; the x86 verifier pair plus all twelve gated
   ROCm fixtures execute 14/14. This closes host ownership, not the separate CI
   lane proposed in step 3.

Current host-free compiler validation: **57/57 ROCm backend lit tests pass**
after adding the Block AttnRes Target-record/generator fixture. Older counts in
historical evidence entries below describe their named packet, not the current
suite total.

Cross-backend sync `CI-LIT-DEPS-2026-08-12` — **parity validated; no physical
follow-up required.** PR 554 made the shared opt-in MLIR lit lane install the
workflow-owned Python dependency set before `lit`/FileCheck collection. This is
backend-neutral test infrastructure, changes no compiler/runtime contract, and
requires no ROCm package or exact-device evidence.

Cross-backend sync `PDE-STENCIL-FOUNDATION-1-2026-08-12` — **owning physical
follow-up required.** Shared IR now carries explicit finite tap coefficients
and required scheme/order/per-axis spacing, and unavailable AMD Target symbols
are truthfully artifact-only. ROCm owns the first typed stencil+halo consumer
and exact gfx1151 correctness/performance packet. gfx1200/gfx1250 remain
fail-closed and no CPU evidence transfers.

Cross-backend sync `BLOCK-ATTNRES-ROCM-2026-08-12` — **gfx1151 Phase 5
physical package retained; selector promotion remains open.** The shared Block AttnRes plan fixes a quotient/remainder block
partition, epsilon-qualified RMS key semantics, fp32 logit/softmax/accumulator
policy, and a dependency-free numerical/VJP/softmax-merge oracle. Phase 1 now
ships public fp32 stats/merge/finalize references, and Phase 2 ships the faithful
direct/two-phase stdlib recurrence. Phase 3 now owns typed statistics and
`depth_attn` Graph contracts plus compiler-native VJP/JVP product nodes. Phase 4
now owns one content-addressed `schedule.depth_attention` →
`tile.depth_attention_kernel` artifact; the hash binds its static shape,
epsilon, all-f32 numerical policy, source tile, workgroup, and exact
statistics/merge recurrences. Phase 5 now lowers that artifact into the typed
`tessera_rocm.depth_attention` record and a fused gfx1151 statistics-attention,
associative merge, and finalize kernel. Compiler-owned packaging and the
runtime descriptor consume the exact artifact without Graph re-entry. Three
exact-device shapes pass the independent fp32 oracle with maximum absolute
error `1.96e-5`; the packet records content digests and synchronized WSL
operation-total samples. Those samples include allocation, module load, copies,
and synchronization, so the physical package is retained but remains
selector-ineligible until bare-metal HIP-event or ROCprofiler timing is
recorded. gfx1200/gfx1250 remain fail-closed. No AVX-512 claim transfers.

Cross-backend sync `MODEL-FUSED-PHYS-1-2026-08-12` — **MiniMax MSA now
consumes an exact digest-bound package on gfx1151; DeepSeek remains open.** The
package binds Graph, Schedule, Tile, and Target digests and its runtime launch
contract; launch validates that lineage and executes `rocm_sparse_attn_compiled`
without legacy Graph `ops` metadata or Graph-to-backend reconstruction. An
exact gfx1151 execution matches the independent MSA oracle. This does not close
DeepSeek MLA/DSA, prefill/decode performance packets, or selector promotion;
gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `MODEL-WEIGHT-PHYS-1-2026-08-12` — **persistent packed
INT4 model operands execute on gfx1151; FP8 remains correctly unavailable.**
The shared model-weight ABI now carries immutable physical checkpoint bytes and
separate fp32 scales under one content digest and prohibits full-weight
materialization. gfx1151 uploads codes/scales once into two typed device leases;
the fused dequant-GEMM consumes those pointers across launches. The local
gfx1151 packet passes the groupwise fp32 oracle with no unpack/repack and records
weight/kernel digests. WSL host-wall timing is regression-only. gfx1151 rejects
FP8/BF8 by ISA contract; gfx1200/gfx1250 remain fail-closed pending independent
profiles, packages, and evidence.

Cross-backend sync `W4-PRESBURGER-SHARD-2026-08-12` — **shared typed shape and
placement analyses landed; gfx1151 physical follow-up remains open.** The C++
compiler consumes coefficient-vector integer-affine and exact modular/divisibility
constraints with MLIR Presburger analysis. Shared placement propagation now
distinguishes replicated, tiled, partial-reduction, and unknown states; catalog
pointwise/reduction/collective rules feed an explicit fail-closed reshard planner.
Lowered `control_scan` has shared JVP/VJP products under `recompute_all`; saved
checkpoint policies remain rejected. No RCCL reshard materialization, native
region product, or new bare-metal packet was added; gfx1200/gfx1250 remain
fail-closed.

Cross-backend sync `W4-CFG-RESIDUAL-W5.2G-2026-08-14` — **shared compiler
carriers and scalable scheduler landed; gfx1151 physical follow-up required.**
The tracer-owned structured CFG, block-wide Presburger identity, and executable
SAVE/HYBRID residual ABI change shared lineage only. The action-DAG model now
uses deterministic critical-path/list scheduling with safe lower-bound pruning
and a small-DAG exhaustive oracle. No ROCm region-product package, inferred-edge
producer wiring, bare-metal calibration, or selection claim was added;
gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `E2E-AUTH-DAG-2026-08-12` — **reduction and normalization
forward-product authority closed on gfx1151; calibrated promotion remains
open.** Reduction now launches two exact `schedule.reduce → tile.reduce_kernel
→ gfx1151 descriptor` children. Normalization consumes a content-addressed
composite Schedule/Tile action program and reconstructs no Graph operation at
runtime. Mandatory tracer/AST differential gates cover all pure native JVPs
and VJPs; the beta/gamma AST operand-order defect was fixed. The committed v2
packet records ten samples plus source/paired/Schedule/Tile/parent digests and
passes exact-device correctness. WSL supplies synchronized host wall only:
device-clock calibration and ROCprofiler activity are absent, so timing is
regression-only and selector promotion remains open. gfx1200/gfx1250 remain
fail-closed.

Cross-backend sync `E2E-AUTH-DAG-2026-08-11` — **shared frontend authority and
automatic dependence-edge contracts landed; gfx1151 physical evidence is
unchanged.** Pure straight-line tensor signatures now cache tracer-owned Graph
IR and can be differentially certified against the retained AST candidate.
Native-JVP plugins declare Graph/Schedule/Tile/gfx1151 disposition and own parent
package construction; compatibility gaps are explicit. W2.1 facts now generate conservative Tile action-DAG
edges with reason and analysis digests. The existing JVP children remain the
physical implementation; this slice adds no selector promotion. Carrying the
generated edges through ROCm family pipelines and a bare-metal calibrated
packet remain follow-up. gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-SOLVER-ISTFT-PHYSICAL-2026-08-11` — **bounded general-
residual and exact ISTFT-window products execute on gfx1151.** The general
solver parent binds five immutable native children (residual, solution JVP/VJP,
and parameter JVP/VJP) and runs restarted GMRES with a true-residual check.
The compiler now derives all five children from a verified typed residual Graph.
Pointwise, sum/mean, rank-2 matmul, transpose, distinct parameter/solution
spaces, bounded-dynamic dimensions, mixed f16/bf16 storage with explicit f32
widening, and statically bounded `control_for` are represented in the
content-addressed child program. Pure scalar predicates now also lower
data-dependent `if` and bounded `while` to digest-bound compare/select SSA;
each primal and product child recomputes the same predicate and never depends
on a host callback or an untracked tape. Exact gfx1151 proof covers nonlinear
`R=x*x+sin(x)-theta` without materializing a Jacobian.
`tessera.istft_jvp` now carries the nonlinear
window product explicitly, and the packed-real gfx1151 package differentiates
both overlap-add numerator and window-energy denominator. The committed WSL
packet records 30 warmed samples, cold start, child resources, artifact
digests, and numerical error for nonlinear, reduction, f16-storage matmul,
bounded-dynamic mixed-storage, data-dependent `if`, data-dependent `while`, and
ISTFT-window products (maximum error 1.28e-4). Its
host-wall medians are regression-only: device-clock calibration and a clean
bare-metal packet remain required. A fresh native gfx1151 clock probe confirms
`wall_clock64` is valid on this gfx11.5 device, while HIP events remain zero and
ROCprofiler activity is unavailable under WSL; the sample is retain/regression
evidence, not selector evidence. Odd-window/low-precision ISTFT products,
gfx1200, and gfx1250 remain fail-closed.

Cross-backend sync `E2E-REAL-6-JVP-SOLVER-2026-08-11` — **the first duplicate-
authority cohort is landing.** Native forward-product specialization now binds
tracer-produced canonical Graph IR and dispatches through explicit family
plugins outside `JitFn`. Scheduled gfx1151 attention backward now consumes its
content-addressed Tile artifact directly; it no longer reconstructs Graph IR to
re-enter the Graph-owned packager. General solver contracts now bind and
execute exact residual/JVP/VJP child packages. The typed residual compiler now
derives those children automatically for pointwise, reduction, rank-2 matmul,
bounded-dynamic/mixed-storage, and counted-region programs; architecture-owned
packets for that expanded envelope remain open. Existing diagonal-sqrt and attention
packets do not transfer to new solver families. The six-case native forward-product cohort,
including DCT-IV with corrected logical-length normalization, passes on the
WSL-visible gfx1151; this is correctness, not selector-grade timing.
gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-FWD-DIST-3-2026-08-11` — **shared exact JVP,
structured-region products, and typed point-to-point RCCL execution landed;
gfx1151 multi-rank evidence remains open.** Public JVP/jacfwd use registered
tangent products without finite differences, and compiler forward mode carries
primal/tangent state through bounded SCF. `collective_permute` reaches the
one-process/multi-device RCCL launcher as grouped send/receive with explicit
peer-map validation. A second visible GPU, subgroup communicator construction,
and exact RCCL correctness/performance packets remain required. LSA, GIN/RMA,
Copy Engine, and gfx1250 DDA retain their independent gates.

Cross-backend sync `W4-SOLVER-REGION-2026-08-11` — **shared bounded-region
adjoints and general matrix-free solver policy landed; gfx1151 physical proof
now has a bounded child-composite proof.** Portable tracing now emits bounded SCF, and the paired compiler
differentiates effect-safe single-block `if`, counted `for`, and canonical
bounded `while` with implicit captures. General residual execution uses
restarted GMRES/CG policy and exposes convergence work; counted-region evidence
executes SAVE/RECOMPUTE/HYBRID cohorts. gfx1151 retains the monolithic
diagonal-sqrt pilot and now executes digest-bound general residual children
through the matrix-free parent. Typed reduction/matmul and statically
counted-region lowering is now shared software; pure scalar predicate-bearing
residuals and the expanded-family WSL correctness packet are now closed.
Selected-checkpoint lowering and selector-grade bare-metal packets remain open;
gfx1200/gfx1250 stay fail-closed.

Cross-backend sync `COMP-GRAPH-DATAFLOW-W2.1-2026-08-11` — **shared
analysis substrate landed; existing gfx1151 evidence remains valid.** Graph IR
now has one fail-closed, invalidatable shape/alias/liveness/memory-dependence/
activity analysis with C++ and Python query surfaces. Reverse AD and await
sinking consume it and recompute after mutation. This changes legality
infrastructure only: it adds no gfx1151 performance claim, transfers no
schedule to gfx1200/gfx1250, and leaves both architectures fail-closed.

Cross-backend sync `AD-FWD-FAMILY-2-2026-08-11` — **affine normalization,
compound spectral, and matrix-free solver products execute on exact gfx1151.**
The product ABI now accepts multiple active operands and named child outputs.
Affine LayerNorm carries data/scale/bias tangents through generated norm and
binary HIP lanes; the spectral-filter product executes both bilinear terms
through the typed TSOL artifact; and `schedule.solver_ift` distinguishes the
compiler-emitted residual JVP plus non-transposed solve from the VJP chain.
All three exact-device numerical tests pass. The native multi-rank collective
product contract is also software-complete and requires an available RCCL
hardware adapter with world size >= 2; an actual multi-GPU RCCL packet remains
the evidence gate. gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-FWD-NATIVE-1-2026-08-11` — **the first native gfx1151
JVP package is executable and exact-device proven.** A content-addressed parent
binds the compiler-emitted paired-JVP digest to ordered physical child
packages; launch never returns to Graph IR. gfx1151 sum, non-affine RMSNorm,
and packed RFFT primal/tangent products pass independent NumPy formulas on the
WSL-visible device (**3/3**). Child substitution and parent tampering fail
closed. This is a correctness packet, not timing promotion. Additional
normalization/spectral/solver families and the native RCCL evidence packet
remain follow-ups. gfx1200/gfx1250 are rejected by the package
contract and remain fail-closed pending their own profiles and evidence.

Cross-backend sync `COMP-EFFECTS-W2.2-2026-08-10` — **shared registered-effect
analysis closed; no gfx1151 performance claim.** Canonical Graph records now
carry effect, alias, mutation, and stochastic identity; Python and C++ consume
the same fail-closed facts and internal calls reach a fixed point. Await
sinking uses that shared query. Existing gfx1151 async correctness remains
valid; RCCL overlap performance and exact-device packets remain ROCm-owned.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R4-2026-08-10` — **shared functional
MegaMoE plan consumption landed; RCCL/gfx1151 performance remains open.** The
runtime consumes a content-addressed plan carrying chunk slices, per-expert
capacity, a two-live-frame workspace limit, true-use dependencies, ordered
collectives, and deterministic combines. R3 only prunes complete measured
records; valid device-event/activity latency must select. Mock multi-rank proof
does not establish RCCL overlap, and WSL timing is promotion-ineligible.
gfx1151 needs native RCCL stream/event binding and a packet; gfx1200/gfx1250
remain fail-closed.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R3-2026-08-10` — **shared prune-only
Tile action-DAG model landed; gfx1151 promotion policy is unchanged.** The
model validates explicit dependencies and calibration identity, uses
deterministic critical-path/list scheduling, and composes compute/memory/
communication lanes with queue serialization. Exact small DAGs and proven
lower-bound losers may be pruned; every estimate is
promotion-ineligible and scalar measured latency remains authoritative. WSL
rows lacking valid HIP/activity calibration cannot become promotion evidence;
gfx1200/gfx1250 remain fail-closed. R4 must bind the model to an executable
ROCm overlap consumer and obtain architecture-owned measurements.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R2-2026-08-10` — **shared measured
resource-vector schema landed; gfx1151 providers can populate it without
changing selector authority.** Successful measured rows carry compute time,
dtype-correct bytes moved, communication bytes, queue/resource identity,
timing provenance, and the measured-candidate digest; the tuning cache retains
provenance across warm starts. Analytical rows cannot claim the vector and
scalar measured latency remains authoritative. Existing HIP/device-clock and
ROCprofiler eligibility rules still govern promotion; gfx1200/gfx1250 remain
fail-closed and need independent providers and packets.

Cross-backend sync `COMP-SCHED-OVERLAP-1-R1-2026-08-10` — **shared explicit
lineage is executable and gfx1151 parity is retained.** Python Schedule→Tile
async copies now produce named tokens consumed by waits; internal
`tessera.queue.*` compatibility markers are rejected. Registered collective
await sinking crosses only operations proven memory-effect-free and stops at
mutation, RNG, alias/cast, region, or ordered-collective barriers. Host LLVM /
MLIR 23 builds and the new/legacy collective lit tests pass. The WSL-visible
gfx1151 cohort remains **16/16**: four structural checks plus global→LDS,
five LDS-WMMA, five two-stage pipeline, and bit-identical via-Tile/production
execution. This is correctness retention, not a new performance promotion;
gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-STOCHASTIC-RNG-1-2026-08-10` — **gfx1151 base RNG
transforms are native and exact-device proven.** Explicit key/counter Graph ops,
estimator provenance, dropout replay, fixed-key EGGROLL JVP, and the shared
derivative proof matrix are compiler contracts. The gfx1151 generator now owns
uniform scaling, Box–Muller normal, and dropout masking rather than copying
uniforms to the host; the exact-device suite passes all six cases. Uniform
words are bit exact and normal is one-f32-ULP bounded. Native target JVP package
consumption remains a separate promotion gate; gfx1200/gfx1250 stay fail-closed.

Cross-backend sync `AD-FWD-PRODUCT-2-2026-08-10` — **public JVP ABI landed;
gfx1151 execution remains follow-up required.** Forward/JVP requests now carry
mode-neutral provenance and stable `wrt_indices`, and the compiler emits only
requested tangent terms. Tanh/sigmoid add direct CPU-oracle proof. No HIP/HSACO
package or gfx1151 evidence transfers; gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-FWD-CORE-1-2026-08-09` — **shared compiler JVP
foundation landed; ROCm physical consumption remains architecture-owned.**
The Graph dialect now exposes compiler-owned tangent rules and a paired
`--tessera-autodiff-forward` function contract. Matmul/mul has independent CPU
IR numerical proof, while unsupported active operations and regions fail
closed. The generated ledger distinguishes compiler `ir_tangent` evidence from
Python JVP registration. This changes no gfx1151 package or device evidence;
ROCm must lower and prove any native JVP package independently, and gfx1200 /
gfx1250 remain fail-closed.

Cross-backend sync `X86-TYPED-FAMILY-PLUGIN-2026-08-09` — **x86 parity landed;
ROCm Target-boundary regression repaired.** The shared executable-pipeline schema remains v1,
and x86 now owns a closed semantic-family registry plus registered Target
marker before its prebuilt AVX-512 image. ROCm retains its distinct
Graph/Tile→`tessera_rocm`→ROCDL/HSACO pipeline, gfx1151 selectors, async-copy
ordering, and exact-device evidence. `output=target` now lowers the scheduled
matmul carrier to one `tessera_rocm.wmma_gemm` directive without creating a
`gpu.module`; only `output=binary` runs the architecture generator. Aligned
gfx1151 exact-device scheduled matmul passes across both boundaries. No x86
schedule, ABI, or Zen 5 result transfers to gfx1151/gfx1200/gfx1250.

Cross-backend sync `EGGROLL-ES-LOWRANK-2026-08-09` — **functional rank-1 fp32
Graph → Schedule → Tile → ROCm consumption is exact-device verified on
gfx1151.** The content-addressed artifact binds shape, epoch, sigma,
antithetic/score policy, and the versioned SplitMix64/Philox4x32/Box–Muller
member RNG. `tests/unit/test_rocm_es_low_rank_exec.py` compiles the artifact to
HSACO, launches it through HIP, and compares it with the portable oracle. This
contract now also admits an architecture-bound Zen 5 AVX-512 artifact, but its
ABI, schedule, and timing packet do not transfer to gfx1151. ROCm's exact
architecture check and selector evidence remain unchanged. This
now uses a cooperative Wave32/LDS SGMV kernel: seed derivation occurs once per
row and `x@B` is reduced once before output scaling. It is exact-device correct,
with aligned-small and ragged 513×277 launch tests. The durable WSL packet
`benchmarks/baselines/rocm_gfx1151_es_low_rank.json` records a 16×32×1024→1024
case at 127,434 ns synchronized host-wall median and max absolute error
1.91e-6. HIP events remain zero/unavailable, so the packet correctly retains
the kernel for correctness but is selector-ineligible pending scalar/direct and
packed-WMMA comparisons on a valid device clock. Rank>1 and the `s32` WMMA-IU8
lane remain open. gfx1151 has no FP8 WMMA. Contract:
`docs/audit/compiler/EGGROLL_SUPPORT_PLAN.md`.
W4 scalar-gather/member reconstruction passes four-rank mock-mesh proof;
native RCCL multi-rank execution and timing remain open.

Cross-backend sync `COLLECTIVE-RCCL-ADVANCED-LANES-2026-08-09` — **three
independent executable boundaries landed; exact-device packets open.** Zero-CU Copy
Engine now has an explicit `copy_engine` artifact lane and a communicator
created with `NCCL_CTA_POLICY_ZERO`; legality requires ROCm 7.12+, a single
node, registered symmetric buffers, device-API support, no graph capture, and
an RCCL-supported operation. GIN/RMA is a distinct strict-ordering lane backed
by public `ncclPutSignal`/`ncclSignal`/`ncclWaitSignal` adapter calls; it
requires both host-RMA and nonzero GIN properties on a multi-node communicator,
so current WSL's `host_rma_support=true, gin_type=0` is correctly insufficient.
The registered Target dialect owns a typed window resource plus
`window.register`, `put_signal`, `signal`, `wait_signal`, and
`window.deregister`. Content-addressed packages validate window lifetimes and
the rank-local runtime consumes ordered records through an explicit adapter;
ordinary collective records cannot select GIN.
gfx1250 DDA remains selector-owned: the artifact requires `target_arch=gfx1250`
and a separate selector-evidence digest, and never calls RCCL internal DDA
symbols. The source authority is ROCm Systems/RCCL develop at
`5bc651a82683`; its CE and DDA routing predicates are modeled as eligibility
guards, not copied selectors. Current gfx1151 WSL can provide declaration
evidence only. Open hardware packets are: two-rank single-node CE correctness
plus RCCL route proof, multi-node GIN put/signal/wait ordering, and gfx1250
DDA correctness/route/performance. None promotes another lane.
The native GIN launcher binding now accepts explicit Tessera, OpenMPI, PMI, or
Slurm rank metadata without taking an MPI ABI dependency, exchanges the RCCL
unique ID through a run-scoped shared rendezvous, and aggregates exact
put/signal/wait readback plus HIP-event and host-wall timing across ranks. The
packet remains open because this WSL host has one gfx1151 and `gin_type=0`;
closure requires running the
[exact-device runbook](GIN_EXACT_DEVICE_RUNBOOK.md) on at least two GIN-connected
gfx1151 nodes with matching artifact and communicator digests. No single-node
functional result can close it.

Cross-backend sync `COLLECTIVE-NATIVE-FOUNDATION-2026-08-09` — **communicator
and window ownership landed; two-rank device evidence blocked by access.** The
C++ RCCL adapter issues all four collectives, queries each initialized
communicator, and owns symmetric strict-ordering registration through a
move-only RAII window. The Python runtime seals rank/device, device-API, LSA
team, multimem, GIN, and host-RMA properties into a topology-specific digest;
device artifacts reject mismatches. A compiled gfx1151 LSA harness requires
two distinct gfx1151 ordinals, one device-API-capable LSA team, peer writes,
and exact readback. On current WSL, communicator-property discovery passes but
reports `device_api_support=false`, `lsa_team_count=0`, `gin_type=0`, and
`host_rma_support=true`; symmetric registration is blocked and only one
gfx1151 is visible. The opt-in CTest therefore exits with the structured
hardware-skip code 77 and no correctness promotion. gfx1200 remains
fail-closed; gfx1250 DDA, zero-CU Copy Engine, and GIN/RMA remain independent.

Cross-backend sync `COLLECTIVE-ASYNC-UNIFY-2026-08-09` — **shared software
contract closed; native RCCL evidence open.** Active forward/autodiff producers
no longer emit unregistered `tessera.collective.*` markers: they create typed
`tessera_collective` futures, explicit awaits, and rewired SSA consumers.
All-to-all divisibility, QoS/chunk bounds, mesh-axis topology, and native fp32
storage now fail closed. gfx1151 still needs a real multi-GPU RCCL packet;
single-GPU WSL cannot supply it. gfx1200/gfx1250 and selectors are unchanged.

Cross-backend sync `DIST-SHARD-ALIAS-1-2026-08-09` — **portable alias bridge
landed; RCCL proof unchanged.** The public sharding inventory is no longer
treated as nine ROCm kernels: three entries are compile-time placement/region
contracts, four reductions lower to registered all-reduce records, and
`broadcast_to_axis` lowers to all-gather. Those five aliases execute through
the portable multi-rank runtime. `collective_permute` remains an explicit
point-to-point gap and fails closed. Native RCCL launch/evidence, frontend
capture, and gfx1151 multi-rank packets remain open; gfx1200/gfx1250 stay
fail-closed.

Cross-backend sync `AD-SOLVER-RESIDUAL-EVAL-2026-08-08` — **bounded gfx1151
physical pilot and correctness packet landed.** The diagonal-sqrt residual now
lowers as one content-addressed `schedule.solver_ift` →
`tile.solver_ift_kernel` → `tessera_rocm.solver_ift` package. Its generated HIP
kernel executes residual, transposed diagonal matrix-free solve, and parameter
adjoint. The committed [3×257 f32 packet](../../../../benchmarks/baselines/rocm_gfx1151_solver_ift_evidence.json)
reports zero maximum error and 30
complete-backward samples with 3,084 retained bytes. WSL synchronized-host
timing is regression-only, not a performance promotion. General residuals,
iterative/Krylov solvers, bare-metal timing, and gfx1200/gfx1250 remain open and
fail closed.

Cross-backend sync `AD-CORE-EFFECT-CONTROL-COLLECTIVE-2026-08-08` — **shared
effect/control parity and gfx1151 consumption validated; multi-rank transport
open.** The rebuilt LLVM/MLIR 23 ROCm compiler consumes canonical Graph
attention operations into physical GPU modules. The Target-IR artifact retains
its typed `tessera_rocm` directive after consuming Tile IR; the binary pipeline
then consumes that directive before its strict ROCDL boundary. The exact-device
gfx1151 spine passes 65/65.
The four typed Tile collective contracts now lower into the portable
`tessera_collective` Target dialect and content-addressed runtime-adapter
package; deterministic two-rank software tests execute all four operations.
This does not constitute native RCCL execution or multi-rank performance
proof. No new selector or performance promotion is made; gfx1200/gfx1250
remain fail-closed.

Cross-backend sync `GRAPH-VERIFY-SIGNED-1-2026-08-08` — **shared legality
parity validated; gfx1151 packages unchanged.** Graph and canonical-attention
verifiers now read signed `IntegerAttr` values before checking positive and
non-negative bounds, preventing MLIR 23 unsigned accessors from admitting
negative schedules, cache windows, seeds, or control bounds. Direct negative
IR cases cover both dialects. No HIP ABI, gfx1151 schedule, selector, image, or
exact-device evidence changes.

Cross-backend sync `AD-TSOL-SPECTRAL-NATIVE-2026-08-09` — **bounded gfx1151
backward packages and exact-device correctness landed; performance open.** The
content-addressed multi-output carrier now lowers complex-f32 spectral-filter
and unbroadcast last-axis full-f32 spectral-convolution adjoints into generated
ROCm GPU modules. Both packages compile through ROCDL to hsaco and pass direct
gfx1151 HIP-launch comparison for both gradient outputs; convolution preserves
`backward`, `forward`, and `ortho` scaling selected before Tile IR. Native
STFT/ISTFT backward, broader axes/dtypes/broadcasting, and a device-event
performance packet remain open and fail closed. Existing forward TSOL packages
are unchanged; gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `AD-CORE-LINEAR-1-2026-08-08` — **shared Graph-IR parity
validated; existing gfx1151 packages unchanged.** Both compiler autodiff passes
now consume a single linear-transposition interface for structural views,
broadcast, and operand-wise matmul, with paired CPU numerical proof and
fail-closed rejection. Existing gfx1151 matmul backward execution remains its
architecture-owned physical evidence; no HIP schedule, image, selector, or
timing result changes.

Cross-backend sync `COMPILER-DASHBOARD-PROOF-TRUTH-2026-08-08` — **gfx1151
exact proof retained; no ROCm physical change.** The codegen dashboard now
uses one manifest op×target denominator and reports runtime paths separately.
The two reference rows remain visible rather than being silently promoted; no
HIP image, selector, timing, or architecture support changes.

Cross-backend sync `X86-BUILD-ARTIFACT-DISCOVERY-2026-08-08` — **shared build
contract parity validated; no ROCm physical change.** The x86 runtime and
packager now join ROCm in honoring the fail-closed `TESSERA_BUILD_DIR`
selection. No HIP image, gfx1151 schedule, device evidence, or selector is
changed; a missing selected x86 tree cannot fall through to ROCm's build
artifacts accidentally.

Cross-backend sync `STANDALONE-COVERAGE-TRUTH-2026-08-08` — **gfx1151 package
evidence is now represented without changing a selector.** The standalone
dashboard generates its registry and compiler-layer counts, exact-target
manifest summary, and open queues. The audit no longer hides the verified
Adafactor Schedule→Tile/native package behind a single-GPU terminal override,
and the benchmark inventory now names the physical TSOL and Adafactor
harnesses. This is an audit correction only; no new timing eligibility,
gfx1200/gfx1250 support, or architecture transfer is claimed.

Cross-backend sync `TSOL-NATIVE-REAL-FFT-2026-08-08` — **gfx1151 correctness
retained; performance promotion remains open.** The v3 FFT artifact now binds
logical length separately from physical length, an explicit Hermitian layout,
and either `packed_even_n2_hermitian_v1` or a named full-complex fallback.
Even RFFT/IRFFT use a persistent N/2 complex plan plus gfx1151 device kernels
for Hermitian pre/post processing; odd lengths remain full-complex. The focused
gfx1151 corpus passes 42 tests. WSL synchronized-host-wall comparisons at
batch=32 show 1.11x RFFT and 1.18x IRFFT at N=1024, while N=256 RFFT is noisy
around parity. Retain the implementation, but do not issue a selector-grade
promotion until bare-metal device events confirm the envelope. The next ROCm
optimization is folding Hermitian pre/post processing into the fused-LDS
kernel for small transforms, eliminating its extra launch.

Cross-backend sync `ROCM-BUILD-ARTIFACT-DISCOVERY-2026-08-07` — **ROCm-owned
test-infrastructure closure.** `TESSERA_BUILD_DIR` now selects one CMake tree
fail-closed, an explicit `TESSERA_OPT` can infer its owning CMake root, and
artifact-specific overrides remain authoritative. Legacy ROCm tests no longer
hardcode `build/` or `build-rocm/` for `tessera-opt`, `tessera-rocm-opt`, lit
site configuration, the GEMM/attention runtime libraries, or the static runtime
bridge. Apple, NVIDIA, and x86 assessed the additive shared resolver as parity
validated with no physical schedule or exact-device claim transfer. On the
gfx1151 WSL host, the formerly path-gated cohort passes **89/89** using only
`TESSERA_BUILD_DIR`; the complete non-performance `test_rocm_*.py` corpus
passes **2340/2340** with no skips, and the shared discovery/audit/lit slice
passes **205/205**.

Cross-backend sync `AUTODIFF-RELAXATION-1-2026-08-07` — **shared
Python-reference contract; ROCm physical follow-up required.** `sparsemax`,
`entmax15`, `soft_top_k`, `gumbel_softmax`, and `perturbed_argmax` now have
storage-preserving reference semantics and autodiff rules, but no HIP lowering
or gfx1151 evidence. They remain explicitly reference-only until a ROCm-owned
physical package is selected and proven.

Cross-backend sync `MATH-PHYSICAL-2-2026-08-06` — **gfx1151 module reuse and
reduced-storage coverage retained.** Compiler-generated unary, binary,
reduction, and scan images now load each `(family, chip, operation, dtype)` HIP
module once per process instead of once per launch. On Radeon 8060S/gfx1151,
the seven-operation f32 packet improves synchronized WSL host-wall medians by
1.46x--3.58x, with identical numerical results. The same packet records 21
physical rows across fp32/fp16/bf16 storage; elementwise errors stay within
0.05, while length-1024 fp16/bf16 reductions and scans use explicit
storage-quantization limits of 0.6/5.0. The exact-device math suite passes 579
tests against the branch-built LLVM/MLIR 23 compiler. `lgamma` and `digamma`
are promoted to fp16/bf16 storage with f32 compute, matching their existing
exact-device tests. Binary packages now reject mixed input dtypes rather than
silently converting the second operand. Evidence:
`benchmarks/baselines/math_physical_gfx1151_2026_08_06.json`. WSL host-wall
timing is selector-ineligible; no kernel schedule is promoted from this packet.

Cross-backend sync `TSOL-CONTRACT-GENERALIZE-2026-08-06` — **gfx1151 physical
policy expansion implemented; selector timing remains gated.** The v3 contract
specializes bounded dynamic shapes into exact content-addressed packages,
packs arbitrary axes inside the native package, admits fp32/fp16/bf16 real
storage with f32 accumulation, and carries backward/forward/ortho scaling into
native HIP kernels. ABI v4
embeds the compiled architecture, so a stale image for another GPU fails
closed. Fifteen exact-device tests pass on Radeon 8060S/gfx1151. The packet
`benchmarks/baselines/tsol_physical_policies_gfx1151_2026_08_06.json` records
30 numerical-and-performance rows across all five operations, including seven
digest-changing bounded specializations and combined policy cases. Warm
medians span 0.462--6.129 ms in synchronized WSL host-wall time, and every row
meets its recorded error limit, but the packet is selector-ineligible until
bare-metal HIP device-event evidence exists. Reduced storage is an explicit
native-package host conversion into f32 device arithmetic; arbitrary-axis
pack/unpack is likewise package-owned host work. Separately stamped gfx1200 and
gfx1250 ABI-v4 images now cross-build with LLVM/ROCm 23 and have distinct image
digests recorded in
`benchmarks/baselines/tsol_rocm_arch_packages_2026_08_06.json`, but their
profiles are `build_only` and execution remains fail-closed until each target
owns an architecture schedule and exact-device packet.
Cross-backend sync `TPROF-MULTICLOCK-2026-08-06` — **ROCm owner; native
gfx1151 clock collector landed, bare-metal calibration open.**
`TPROF-ROCM-TIME-1` requires every benchmark sample to
preserve independent synchronized host-wall, HIP-event, instrumented
`wall_clock64()`, and profiler-activity records with source, validity,
calibration, instrumentation, and verdict eligibility. Missing clocks remain
explicit and no wall measurement may be relabeled as device time. The owning
spec is [`../../../spec/CITL_ROCM_TRACE_PROFILER_SPEC.md`](../../../spec/CITL_ROCM_TRACE_PROFILER_SPEC.md).

The gfx1151 rule is exact: it is RDNA 3.5 / GFXIP 11.5, so the HIP warning for
RDNA 3 / GFX11 does not justify a categorical `clock()`/`clock64()` ban on this
target. `wall_clock64()` is the default constant-frequency grid clock;
`clock()`/`clock64()` remain diagnostic candidates until an exact-device probe
establishes monotonicity, effective rate, wrap behavior, and cross-CU/WGP
comparability.

`TPROF-ROCM-RTG-1` adds an optional `rtg_hsa_dispatch` experiment after the
multi-clock contract. It may recover HSA dispatch timestamps when WSL
ROCprofiler device-activity tables are empty, but it intercepts AQL queues and
completion signals, must run in a fresh subprocess, and must record overhead,
callback completion, exit, and teardown state. It supplies no PMC, PC-sampling,
cache-counter, or stall claim. WSL evidence may rank and retain/reject on the
same host; promotion and counter-dependent decisions remain bare-metal-only.

Execution order:

1. **Complete:** implement the four-clock evidence schema and fail-closed
   validator.
2. **Complete:** add synchronized host batch/empty-launch calibration and
   HIP-event validation.
3. **Landing:** the optional `TPROF_WITH_HIP` provider now queries
   `hipDeviceAttributeWallClockRate`, instruments a multi-workgroup batch with
   `wall_clock64()`, records empty-launch overhead, and rejects every target
   except exact gfx1151. The WSL packet at
   [`../../../../benchmarks/baselines/rocm_gfx1151_multiclock_probe_2026_08_06.json`](../../../../benchmarks/baselines/rocm_gfx1151_multiclock_probe_2026_08_06.json)
   measured a valid device envelope while both HIP event intervals remained
   exactly zero. It is regression-eligible and promotion-ineligible. Paired
   application-kernel ISA/resource overhead and the diagnostic
   `clock()`/`clock64()` qualification probe remain open.
4. **Landing:** the fresh-process `rtg_hsa_dispatch` runner now sets the
   official child-only environment, records process/teardown state and output
   digests, and admits no counter or PC-sampling claim. Building the official
   tracer on this WSL host is blocked by the absent `libunwind` development
   headers; no RTG runtime evidence is claimed.
5. **Landing:** `tprof_rocm_native_capture.py` now invokes `rocprofv3` around
   the application, requests runtime/kernel/memory activity, PMCs, and PC
   samples, normalizes actual output into the provider trace, and feeds the
   dispatch envelope into the multi-clock artifact. The packet builder pairs
   clean/instrumented application images and rejects excessive timing or
   resource overhead. The checked-in WSL capability packet
   [`../../../../benchmarks/baselines/rocm_gfx1151_native_capture_2026_08_07.json`](../../../../benchmarks/baselines/rocm_gfx1151_native_capture_2026_08_07.json)
   records `/dev/dxg`, no `/dev/kfd`, and no activity, counters, or PC samples;
   paired with the existing zero-HIP-event multi-clock packet, it is
   promotion-ineligible. It deliberately stops
   before the WSL-incompatible profiler can abort.
6. **Open:** record a bare-metal gfx1151 calibration packet before promoting any timing
   or counter-dependent selector. The exact-host packet must include valid HIP
   event or ROCprofiler activity calibration, requested PMC/PC records, and a
   clean-versus-instrumented application-kernel ISA/resource/timing comparison.

Cross-backend sync `TSOL-ROCM-E2E-1-2026-08-06` — **typed Schedule→Tile and
bounded gfx1151 physical package complete.** The five TSOL
composites now execute in the prebuilt `libtessera_spectral_rocm.so` image in
the requested order: `spectral_filter`, `dct`, `spectral_conv`, `stft`, and
`istft`. `tessera.scheduled_spectral.v5` materializes one verified
`schedule.spectral_program` → `tile.spectral_program_kernel` edge and
content-addresses each compound
program's child FFT Schedule/Tile digests, interleaved-complex layout, axis,
padding/crop, window/hop/frame policy, normalization, exact persistent-workspace
bytes, f32 accumulation, native entry, and immutable-input/fresh-output
lineage. Framing, padding, complex multiply, half-spectrum packing, and
deterministic ascending-frame overlap-add remain on device; the public
host-pointer ABI stages only inputs and the final output. Runtime consumption
requires that artifact and no longer re-enters Graph metadata or host NumPy
composition. A bounded digest-keyed composite-plan cache owns one persistent
device allocation partitioned according to the artifact workspace contract;
child FFT plans remain independently digest-bound. Exact WSL-visible gfx1151
tests pass for aligned, batched, ragged, and prime-length Bluestein children.
`gfx1200`/`gfx1250` fail closed. Apple/NVIDIA physical adoption remains a
sibling-owned follow-up and inherits no gfx1151 result.

Cross-backend sync `TSOL-GFX1151-FUSED-BATCH-2026-08-08` — **missing-image
blocker closed and batched LDS execution retained on gfx1151.** A HIP-enabled
production compiler build now closes over `TesseraSpectralHIP`: building
`tessera-opt`, `tessera-rocm-opt`, or `TesseraCompilerFoundation` also builds
the exact-architecture `libtessera_spectral_rocm.so` image. The 15 formerly
blocked composite tests now pass, and the combined FFT/TSOL and artifact gates
pass **58/58**. The later `gfx1151_stockham_bluestein_v5` contract records
`persistent_device_plan_fused_lds_batch` for non-Bluestein power-of-two
lengths through 1024. The executor launches one workgroup per transform over
the whole batch, keeping every radix-4/radix-2 stage in two LDS slabs instead
of launching every stage separately for every row.

A 50-iteration same-host comparison against the committed 2026-08-06 packet
retained the implementation: median DCT improved from 1.442 to 0.513 ms
(2.81x), spectral convolution from 3.560 to 0.865 ms (4.12x), STFT from 5.574
to 0.627 ms (8.89x), and ISTFT from 6.129 to 0.661 ms (9.27x). Spectral filter,
which has no FFT child, remained effectively flat at 0.445 ms. Aligned,
bounded-dynamic, arbitrary-axis, fp16/bf16-storage, forward-normalized, and
ortho-normalized rows remained within their numerical limits. These are
synchronized WSL host-wall measurements: they justify retaining the launch
topology but remain ineligible for a performance-selector promotion claim
until bare-metal HIP-event/ROCprofiler calibration is available.

Cross-backend sync `TSOL-SPECTRAL-POLICY-2026-08-08` — **DCT physical coverage
expanded on gfx1151; performance promotion remains open.** DCT-I/II/III/IV now
carry distinct public, autodiff, Graph, Schedule, and Tile identities.
`tessera.rocm.spectral_composite.v6` keeps phase-corrected type II on its FFT
child and executes types I/III/IV through separately hashed direct cosine
kernels; exact-device smoke and compiled-suite coverage pass. The shared
chunked-STFT state binds policy and overlap lineage and fails closed for
centred streaming until lookahead is explicit. No direct-kernel selector or
performance claim is made for the newly added DCT types.
The v6 boundary audit also routes the standalone convolution export through
the packed R2C/C2R ABI, gives scalar convolution an explicit device kernel,
and preserves the odd full-complex fallback for one-sample STFT/ISTFT. Direct
standalone-ABI and canonical-package tests pass on gfx1151.

Cross-backend sync `ROCM-MATH-EVIDENCE-2026-08-06` — **correctness defects
fixed and boundary envelope expanded on gfx1151.** Var/std now use one
compiler-generated, centered parallel Welford reduction instead of
`mean(x²)-mean(x)²`; exact-device tests cover large-offset/low-variance data,
arbitrary axes, ragged extents, and fp16/bf16 storage. Boundary probes also
fixed `expm1(-0)`, NumPy-ordered signed-zero `maximum`/`minimum`, infinity and
signed-zero `mod`/`floor_div`, and all signed-zero/infinity/NaN `atan2`
quadrants. Trig large arguments, log/exp boundary domains, lgamma/digamma pole
neighborhoods, 4097-element scans, and reduced-precision scans now have
exact-device evidence. This changes no physical selector or performance claim.
The backend capability manifest now advertises `fp16`/`bf16`/`fp32` storage for
`var`/`std`, with f32 accumulation, matching that tested ABI. `count_nonzero`
remains a separate fp32 sum-composition claim.

Cross-backend sync `ROCM-FFT-PREBUILT-2026-08-05` — **prebuilt package and
persistent Bluestein plan boundary complete on gfx1151.** CMake now produces
and installs `libtessera_spectral_rocm.so`; canonical runtime loading never
invokes `hipcc`. Its versioned opaque ABI owns bounded reusable device plans:
N scratch for mixed radix, or four M buffers for Bluestein (three mutable plus
an immutable pretransformed chirp). Plans are keyed by N, direction, and the
validated Schedule→Tile SHA-256 digest, which the native image exposes.

The v3 Tile contract records `persistent_device_plan`,
`persistent_plan_n`/`persistent_plan_4m`, and
`persistent_device_chirp_fft`. Repeated N=257 execution retained one plan,
reported 4096 complex workspace elements, preserved its content digest, and
matched NumPy. Against per-call allocation, synchronized host-wall medians
improved 1.24x/1.45x/1.31x at N=257/509/1009 with identical error. Evidence:
`benchmarks/baselines/rocm_fft_plan_cache_gfx1151_2026_08_05.json`. Verdict:
retain. WSL timing remains selector-ineligible and does not promote fused LDS.

Cross-backend sync `FFT-PERF-2-2026-08-05` — **x86 parity assessed; gfx1151
promotion remains hardware-blocked.** The x86 cache, Rader, mixed-radix, and
Bailey decisions do not transfer physical schedules to RDNA. The fused-LDS
gfx1151 harness now records the raw HIP interval: all five N=64--1024 samples
returned exactly zero event milliseconds. `rocprofv3 --kernel-trace` also
initialized but emitted no trace files or dispatch timestamps under WSL DXG.
Correctness still passes, but selector-grade confirmation requires bare-metal
gfx1151. At that evidence point production still executed global-memory
ping-pong; `TSOL-GFX1151-FUSED-BATCH-2026-08-08` subsequently retained the
batched small-power-of-two LDS topology with explicit artifact identity while
leaving selector-grade promotion evidence open.

Cross-backend sync `FFT-PERF-FOUNDATION-2026-08-05` — **radix-17 retained and
the first fused-LDS envelope validated on gfx1151.** The optional C++ benchmark now runs
real rocFFT and verifies a round trip rather than timing a synthetic loop. A
same-host radix-17 comparison retained the direct generic stage over Bluestein:
5.66x/5.53x/3.11x faster at N=17/68/289 with lower forward and round-trip
error. Timing is synchronized host wall because WSL HIP events are invalid, so
the result selects arithmetic routing but does not promote a performance
selector. Evidence is in
`benchmarks/baselines/fft_plan_cache_radix17_2026_08_05.json`.

The content-addressed FFT contract continues to state the production truth:
`stockham_autosort`, a persistent device plan, N scratch or four-M Bluestein
workspace, a cached chirp FFT for Bluestein, complex64/f32, and the radix
sequence. A separate single-workgroup candidate now keeps every radix-4
and radix-2 stage in two LDS slabs for N=64--1024. It passed forward and
round-trip comparison on the WSL-visible gfx1151 and measured 1.84x--2.76x
faster than the shipping global-ping-pong lane in the same synchronized
host-wall domain. It was experimental at this checkpoint; the later
`TSOL-GFX1151-FUSED-BATCH-2026-08-08` batch-level validation retained it for
the gfx1151 v4 package, while valid device-event evidence is still required
for a performance-promotion claim. gfx1200/gfx1250 remain fail-closed.

Cross-backend sync `E2E-REAL-FFT-2026-08-05` — **gfx1151 typed
Schedule→Tile consumption and prebuilt packaging implemented.**
The public `rocm_fft_compiled` package had been disconnected from the proven
Stockham implementation: it executed `generate-rocm-dft-kernel`, an O(N²)
one-thread-per-bin diagnostic DFT, while the mixed-radix/Bluestein target hook
below existed only as an arbiter candidate. Same-session gfx1151 evidence made
the consequence concrete: at batch=1/N=16384, direct DFT was 6.49 ms with
1.255 max absolute error; batched Stockham was 1.03 ms with 2.73e-4 error
(host-wall synchronized timing in WSL). The runtime now fails closed onto the
shipping Stockham package, batches rows through one allocation, and records
the target hook's actual mixed-radix plan. The direct DFT remains buildable only
as a named diagnostic/oracle. Exact-device FFT/candidate gates pass (121 tests).

The production Graph→Schedule pass now emits one content-addressed
`schedule.fft` plus one durable artifact, and Schedule→Tile re-derives every
decision before emitting one verified `tile.fft_kernel`. The public runtime
validates and consumes that exact Tile digest, mixed-radix/Bluestein strategy,
shape, direction, and normalization without Graph metadata or a second Python
planner decision. Exact gfx1151 fft/ifft/rfft/irfft tests pass for mixed-radix,
Bluestein, aligned, batched, and inner-axis cases. `gfx1200` and `gfx1250` fail
closed until they own profiles and evidence.

The selected target hook is now a prebuilt native package; public execution no
longer invokes `hipcc`, and the exact Tile digest keys the opaque device plan.
The WSL HIP event clock returned zero/negative
values, so the harness now labels its synchronized host-wall fallback; a
bare-metal device-event performance packet is still required for promotion.

Cross-backend sync `FFT-MIXED-RADIX-BLUESTEIN-2026-08-03` — **parity validated on device; one follow-up.**
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

Verified on live gfx1151: 48 sizes against numpy with zero failures, round
trips to ~5.7e-6, covering mixed-radix (12, 15, 45, 255) and Bluestein (101,
257, 509, 1009) paths. The generic radix-r kernel evaluates its r-point DFT
inline rather than from a table -- the opposite of the CPU choice, because the
extra global traffic a table would cost outweighs the arithmetic at these
radices. That divergence is deliberate: the PLAN is shared, the EXECUTION is
not.

**Follow-up (fixed in review):** `bluestein_amd` allocates five M-sized buffers
where M can be nearly 4N, and its failure status was discarded by the driver.
The host wrapper would then synchronise, copy an UNINITIALISED `d_out`, and
return 0 -- the Python candidate labelling garbage as `rocm_stockham`. The
status now reaches the wrapper via `ts_fft_last_error_amd()`. No exact-device
evidence exists for the failure path itself (it needs induced allocation
pressure); the success paths are covered.


Cross-backend sync `SHAPE-RULE-REGISTRY-2026-08-03` — **ROCm FFT follow-up complete on gfx1151.**
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

The capability table now matches the implementation and backend manifest:
`fft`, `ifft`, `rfft`, and `irfft` are `ready` on `rocm_gfx1151` for logical
complex64 carried as interleaved fp32 pairs. They execute through the
strict mixed-radix Stockham/Bluestein `rocm_fft_compiled` package, not the
Python numerical fallback or the diagnostic direct DFT. Exact gfx1151
transform, round-trip, inner-axis, r2c/c2r, and codegen tests pass. This does
not add complex to the target storage-dtype tuple and does not create an FFT
claim on NVIDIA.

No ROCm-specific regression: the `tile.mma` data-operand rule this registry work
builds on was already ROCm's own - a file-local `static` in `TileToROCM.cpp`
before being promoted to `tessera::tile::dataOperands`.


Cross-backend sync `SUBBYTE-STORAGE-PATH-2026-08-03` — **not applicable on gfx1151, by hardware.**
Same multi-result quantize declaration as NVIDIA. The sub-byte STORAGE path
does not apply to the current ROCm target: the dtype contract already records
`fp8_e4m3` and `fp4_e2m1` as `unsupported` on gfx1151, which matches RDNA 3.5
having no FP8 WMMA (RDNA 4 / CDNA 4 add it). That contract is doing its job —
it is why declaring a real fp8 lowering here would be wrong rather than merely
unimplemented. Re-assess for gfx950 / gfx1201, which are MASTER_AUDIT P2.

Cross-backend sync `REDUCED-PRECISION-COMPUTE-2026-08-03` — **parity validated on gfx1151.**
Same shared policy change as NVIDIA. The reference lane now computes at f32 and
stores narrow, which fixed `rmsnorm` / `rmsnorm_safe` (the latter returned 0.0
instead of ~1.0 at fp16: sum(x**2) overflowed to inf, then x/inf underflowed).
The ROCm `generate-rocm-norm-kernel` lane performs f32 reductions regardless of
storage dtype. Exact gfx1151 fp16 and bf16 tests now compare `rmsnorm` and
`rmsnorm_safe` at magnitude 1e4 against the corrected public f32-compute,
narrow-store reference. All four cases are finite, nonzero, and within the
storage tolerance, closing the historical fp16 all-zero regression.

Cross-backend sync `TILE-MMA-DATA-OPERANDS-2026-08-03` — **parity validated, no ROCm behavior change.**
The `dataOperands` / `isTileControlType` rule ROCm already applied privately in
`TileToROCM.cpp` is now a shared Tile helper. ROCm **composes** rather than
forks: it ORs the shared Tile rule with its own `tessera_rocm::TokenType`,
because Tile IR must not depend on a backend dialect. Behavior is identical;
what changed is that the ODS verifier finally learned the rule ROCm had, which
is what made the typed form reachable at all. Verified on the gfx1151 box: full
lit and the typed-mma fixtures green.

Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-02` — **parity validated, contract changed.**
W0.9 made Decision #19 checkable by a real MLIR parse + dialect load + verifier
instead of substring matching, and ROCm's emitted Target IR did not survive it.
Fixed here: `tessera_rocm.mfma` / `async_copy` / `wait` now emit their real ODS
signatures (the async-copy token is threaded into the wait rather than dropped),
and `elementwise`, `kv_cache_read`, `msa_block_sparse`, and `profiler_probe`
are declared in `TesseraROCMOps.td` — they were emitted without being declared,
so the artifact named the dialect without being it. Validated on the gfx1151
box: plain and probe-annotated modules both parse and verify.
**No generated kernel changed** — this is Target-IR artifact text only, so no
new exact-device evidence is required and none is claimed.

## ROCM-SPINE-1: promote the gfx1151 package through canonical compilation

Cross-backend sync `EXECUTION-SPINE-2026-07-29` — **landing.**
NVIDIA removed its duplicate shared-driver family selector and made its
completed native package the toolchain-gated canonical default. ROCm already
has the closest architectural shape—compiler-owned Tile lowering, HSACO image,
typed descriptor, and runtime submission—and now has the same single
`native_package_kind` / `package_native` backend entry point.
`canonical_compile()` auto-promotes accepted modules only when
`native_packaging_available()` proves both `tessera-opt` and AMD clang are
present; the ROCm root must also resolve the OCML/OCKL/OCLC device libraries.
Explicit opt-out and unsupported-module fallback remain stable. No CUDA
schedule or evidence transfers.

Fresh validation on the owning Strix Halo WSL host (2026-07-30) saw `gfx1151`,
the rebuilt LLVM/MLIR 23 `tessera-opt`, AMD clang 23, and the 7.14 device
libraries, then passed the complete ROCm E2E spine **65/65**, including all
**24/24** exact-gfx1151 package, descriptor, cold/warm identity, and
launch-oracle nodes. The selector refactor does not change emitted
Tile/ROCDL/HSACO or a measured schedule.

**WSL ROCm-root requirement.** This packaged ROCm installation exposes its
compiler and device bitcode under `/opt/rocm/core`, so the exact-device command
must use `ROCM_PATH=/opt/rocm/core`. Setting `ROCM_PATH=/opt/rocm` lets the host
find AMD clang but makes MLIR's binary serialization look for the nonexistent
`/opt/rocm/amdgcn/bitcode`; that run fails before device launch and is not valid
gfx1151 evidence. This is an environment/path-propagation follow-up, not a
kernel correctness failure.

The x86 sibling subsequently reserved canonical `x86` for typed MLIR/native
packaging and renamed its portable-source candidate `x86_c`. ROCm already keeps
canonical HSACO packaging separate from the `rocm_hip` source candidate, so
this is parity validated with no HIP/ROCDL or gfx1151 evidence change. Apple
likewise owns its package-family admission locally and retains only an explicit
Value Target-IR compatibility/probe opt-out; no ROCm behavior changes.

APPLE-RASTER-1 subsequently consumed the shared map in emitted MSL and retained
row-major after mixed Apple7 timing. This neither changes the gfx1151 raster
implementation nor supplies its missing profiler evidence.

## ROCM-CALIB-1: supply gfx1151 evidence to the hardware-free score calibration

Cross-backend sync `COSTMODEL-CALIB-2026-07-29` — **step-distance rejected;
T1 reuse-distance follow-up required, owning host Strix Halo (Radeon 8060S,
gfx1151).**

The independent Zen 5 hierarchical T1 packet rejects ranking on x86 (median
rho -0.4062, 0/3 winner matches). That result reinforces the no-retuning policy
but does not close ROCm's separate gate: x86 L1D/L2/L3 inputs and blocked-loop
candidates do not transfer to gfx1151 L0/L1/L2 or Wave32 schedules.

**Correction that created this item.** `APPLE_AUDIT.md` originally scoped this
calibration to Apple alone, stating that ROCm and NVIDIA kernels "cannot be
measured". False here: this backend holds the measured size-adaptive grouped-GEMM
tile selector, the hot-path perf ratchet, a measured resident-GPU crossover for
large-block sparse attention, and the committed
`rocm_gfx1151_compiler_retune_2026_07_15.json` retune corpus.

**ROCm's special standing on this item.** The original step-distance locality
histogram and N-way bank-conflict analyzer were extracted from production AMD
code
([`../../compiler/AMD_KERNEL_COMPILER_SURVEY.md`](../../compiler/AMD_KERNEL_COMPILER_SURVEY.md)
§3.7–3.8). The gfx1151 correlation rejected that step-distance line on its home
architecture. That verdict is final: do not retune its coefficients or promote
it through a sibling backend.

The new shared T1 implementation is a different hypothesis, not a renamed
step-distance score: it materializes symbolic GEMM A/B tile identities, simulates
a capacity-bounded LRU in raster order, derives DRAM traffic from hits/misses,
and combines it with explicit gfx1151 compute/bandwidth inputs. It has no
preferred-tile, warp, or stage coefficients. It is pruning-only until it earns a
separate retain verdict.

**Bank-model constant re-derivation remains mandatory for any future
bank-conflict analysis.** The published phase table
is GFX950/wave64 with 64 banks (survey §5.1, now including all four phases —
`phase 1 = phase 0 + 32`, `phase 3 = phase 2 + 32`, the four disjoint and
covering lanes 0–63). gfx1151 is **wave32** with a different bank count, so every
constant must be re-derived before a conflict count means anything on this part.
ISA truth stays [`docs/reference/isa/rdna/`](../../../reference/isa/rdna/)
(gfx1151 = RDNA 3.5).

**Result — complete, terminal reject (2026-07-29).** gfx1151 uses wave32, one
32-lane issue phase, a 32-bank SIMD32-affiliated conflict domain, and 4-byte
banks. These constants are locked in
`compiler/hardware_free_scores.py` from RDNA3.5 ISA §§2.1, 12.1, and 12.5; the
GFX950 wave64/four-phase table was not reused.

The frozen analysis
[`benchmarks/baselines/rocm_gfx1151_hardware_free_calibration_2026_07_29.json`](../../../../benchmarks/baselines/rocm_gfx1151_hardware_free_calibration_2026_07_29.json)
scores all nine f32 retune tiles within each of the six committed shapes. It
reproduces **0/6** measured winners, always ranks 1x1 first, has median Spearman
**-0.1381** against the identifiable committed winner partial order, and has
**0%** positive shapes (gate: at least four shapes, median rho >=0.50, >=75%
positive). The hot-path file has three f32 rows but only one production latency
per shape, so it supplies corpus identity, not a second within-shape rank; that
limitation is recorded rather than filled with cross-shape correlation.

Verdict: **reject the step-distance locality score as a kernel latency/ranking
signal and end this line without retuning it.** The bank-conflict score is also
rejected as a latency ranker for this corpus: the measured f32 register-blocked
kernel has no LDS address trace. Its exact analyzer remains useful only as a
structural diagnostic for an explicit LDS layout. A fresh raw rerun was
attempted, but this WSL runtime returned a zero HIP-event interval; synchronized
host time was not relabeled as device time.

**Missing exact-device evidence.** Correlate the T1 cache/reuse score with
recorded gfx1151 latencies over the committed retune corpus and hot-path ratchet
rows, including the size-adaptive grouped-GEMM selector. Report per-family and
per-shape rank correlation and a retain/reject verdict. Failure on gfx1151 ends
this T1 line too; it does not authorize coefficient tuning.

## ROCM-RASTER-1: consume the shared block-rasterization contract

Cross-backend sync `RASTER-CONTRACT-2026-07-28` — **follow-up required, owning
host Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, gfx1151).**

**Shared contract changed.** Schedule IR gained `raster_order` (`row_major` |
`column_major` | `grouped_m` | `grouped_n`) and `raster_group` on
`schedule.tile` / `schedule.knob`, mirrored by `TuningConfig` and persisted in
the SQLite tuning cache. The order is a *permutation of block ids onto the tile
grid*, defined arch-neutrally in `compiler/tile_rasterization.py`; `emit_c()`
produces plain integer arithmetic valid identically under HIP and CUDA, so one
emission serves both leads. Rationale:
[`compiler/TILESIGHT_ASSESSMENT.md`](../../compiler/TILESIGHT_ASSESSMENT.md)
§3.2.

**Do not confuse this with the existing LDS swizzle.** `compiler/rocm_lds.py`
implements an XOR bank-conflict swizzle on *shared-memory addresses* — a
different mechanism at a different level. The new knob orders *workgroups across
the grid*. They compose; neither replaces the other.

**Implementation — complete (2026-07-29).** The Tier-3 WMMA candidate in
`python/tessera/compiler/emit/rocm_hip.py` now carries `raster_order` and
`raster_group` through `ROCmScheduleDescriptor`, the runtime HSACO cache and
directive, and `GenerateWMMAGemmKernel.cpp`. The production generator remaps the
2-D launch coordinates before computing the macro-tile origin. `row_major` is
the default and retains the direct `block_id.y` / `block_id.x` identity path;
the selector does not sweep or promote another order.

**Why this matters more on gfx1151 than anywhere else in the fleet.** Strix Halo
is a unified-memory APU: 256 GB/s of LPDDR5X shared with 16 Zen 5 cores, against
a 32 MB MALL. Grid traversal order decides the concurrent working set, so the
lever that keeps traffic inside MALL is worth more here than on a discrete part
with HBM headroom. Note the counter-evidence in the source paper, though: its
weakest result is CDNA2 (MI210, 23.4% MAPE) precisely because Composable Kernel
exposes no rasterization control — that caveat does not bind us, since we own the
emitter.

**Validation performed (host-free).** `tests/unit/test_tile_rasterization.py`
proves the permutation property over ragged grids and compiles the emitted C with
host clang, checking it against the Python reference for every block id under
`-Wall -Wshadow -Werror`.

**Exact-device result — implementation retained, promotion blocked.** The
gfx1151 artifact
[`benchmarks/baselines/rocm_gfx1151_raster_2026_07_29.json`](../../../../benchmarks/baselines/rocm_gfx1151_raster_2026_07_29.json)
executes square, wide, tall, and ragged buckets for row-major, column-major,
grouped-M {4,8}, and grouped-N {4,8}: **24/24** exact-device correctness rows
pass. HIP events return `0.0 ms` on this WSL stack, and
`rocprofv3-avail list --pmc` reports `No pmc counters supported` /
`Agent HW architecture is not supported`, so MALL/L2 evidence cannot be
collected here. Per the promotion rule, **row-major remains selected**. The
non-default choices are carried and executable, not promoted; a bare-metal or
profiler-supported gfx1151 follow-up is required for performance selection.

**Remaining exact-device evidence.** Valid gfx1151 latency and MALL/L2 hit-rate deltas
per `raster_order` × `raster_group` across the GEMM shape buckets, via
`rocprofv3` plus nonzero device-event timing. Until then the axis is **carried,
not swept**. The T1 reuse-distance model can distinguish raster orders, but it is
not exact-device promotion evidence and ROCM-CALIB-1 has not retained it. RDNA
ISA truth for anything emitted here stays
[`docs/reference/isa/rdna/`](../../../reference/isa/rdna/).

Cross-backend sync `APPLE-AOT-METALLIB-2026-07-28` — **parity validated — ROCm
is ahead**. Apple added `apple_gpu_air`, a precompiled-artifact lane behind the
shared `register_compiler(target, compile_fn)` seam, measured against its
compile-on-launch lane (cold pipeline creation 29.7 ms -> 15.2 ms, ~1.95x;
host-wall timing on Apple M1 Max, not device-event evidence). ROCm already has
the mature precompiled lane Apple just started: hsaco built by `tessera-opt`
itself (convert-gpu-to-rocdl -> rocdl-attach-target -> gpu-module-to-binary)
and loaded with hipModuleLoadData, ~601 references in runtime.py, alongside a
smaller HIPRTC-at-load WMMA lane. Apple's equivalent is one kernel old and is
produced by a Python shell-out to `xcrun` rather than by the compiler, so on
this axis Apple is behind ROCm and ROCm is the model to copy. An earlier
version of this note said ROCm 'has always been precompiled' with nothing to do
— the precompiled half was right, the claim that ROCm has no JIT lane was not,
and is withdrawn. No ROCm work is implied; recorded so the fleet picture is
accurate. No shared IR, ABI, dtype/op registration, or numerical contract
changed.

Cross-backend sync `TESSERA-OPT-CAPABILITY-SKIP-2026-07-27` completes the
migration `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` began. The remaining 43
test files that resolved `build/tools/tessera-opt/tessera-opt` themselves now
route through `tests/_support/compiler_tool.py`, joining the nine already
there, so a driver built without `TESSERA_BUILD_ROCM_BACKEND` produces a named
skip instead of `Unknown command line argument '--generate-rocm-*'`. The shared
check now also reads the passes named *inside* a `--pass-pipeline=` value,
where the equivalent gap surfaces as `does not refer to a registered pass or
pass pipeline`; that spelling is what most of these ROCm codegen gates actually
hit. Resolution prefers an in-repo build but takes the first candidate that
registers the requested passes, so a lean local build cannot mask a capable
one. ROCm is **parity validated** at the host-free boundary: against a
non-ROCm driver the migrated files move from 25 failures to 0 failures /
936 skips, and the full `-m "not slow"` unit sweep carries no
`Unknown command line argument` line. This is shared test infrastructure only —
no ROCm pass body, HSACO packaging, HIP launch ABI, selector, or numerical
contract changed, and **no exact-device evidence is claimed or required**; the
gfx1151 packets retained under `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` remain
the standing device proof.

Cross-backend sync `ROCM-BF16-ATTENTION-2026-07-27` closes the remaining
gfx1151 storage-parity evidence for optimized attention. The canonical rank-4
forward recurrence and tensor-valued deterministic backward loops now have
host-free package tests that select the BF16 ABI and prove BF16 provenance
through every stage of the compiler-owned five-entry HSACO. Exact Strix Halo
`gfx1151` execution covers ragged GQA (`Hq/Hkv=4/2`, `Sq/Sk=17/19`),
per-head bias, softcap, causal/window masks, and deterministic 25% dropout
replay. Forward matches the shared oracle at `0.002176881` maximum absolute
error; its 21-sample resident wall median is `0.099540 ms` versus the dedicated
`0.097814 ms` BF16 baseline and passes the 10% ratchet. Backward maximum
absolute errors are dQ `0.000198936`, dK `0.000280969`, and dV `0.002723813`;
the resident five-launch program median is `0.366238 ms` versus its dedicated
`0.362481 ms` BF16 baseline and passes the 10% ratchet. Both packets use HIP
7.14.60850, AMD clang 23, `hipModuleLaunchKernel`, `hipDeviceSynchronize`, and
separate host-wall timing domains; neither is selector-eligible device-event
evidence. The retained packets are
[`../../../../benchmarks/baselines/rocm_gfx1151_bf16_attention_forward.json`](../../../../benchmarks/baselines/rocm_gfx1151_bf16_attention_forward.json)
and
[`../../../../benchmarks/baselines/rocm_gfx1151_bf16_attention_backward.json`](../../../../benchmarks/baselines/rocm_gfx1151_bf16_attention_backward.json).

Cross-backend sync `LSE-CHECKPOINT-CONTRACT-2026-07-27` replaces the
destination-less LSE markers with explicit memref source/destination, SSA row
offset, identity, memory-space, lifetime-scope, cache-policy, and
`MemWrite`/`MemRead` effects. Default shared forward lowering no longer emits
a store to nowhere. The gfx1151 five-entry training package now supports
explicit saved and recompute modes; saved forward writes finalized
`m + log(l)` into launch-owned `row_lse`, while its `_pre` is D-only. An exact
FP16/BF16 17/64/128/256 sequence sweep originally motivated the provisional
`auto` policy that saves at `max(Sq,Sk) >= 128`. The newer resident dual-clock
packet records positive HIP events but is fail-closed on WSL, and FP16 at 256
does not reproduce a stable saved win. Gradient errors remain identical
between modes. Bare-metal gfx1151 timing must therefore confirm or replace the
threshold before it is production selector evidence; sibling physical
schedules and thresholds do not transfer. Contract and retained packet:
[`../../compiler/LSE_CHECKPOINT_CONTRACT.md`](../../compiler/LSE_CHECKPOINT_CONTRACT.md).

Cross-backend sync
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26` closes the remaining
gfx1151 direct-consumption action under `ROCM-E2E-ATTENTION`. The native
packager now emits one shared tensor program containing the canonical rank-4
forward recompute and `tessera_attn.backward`; shared lowering materializes
the dQ, split dK/dV, and ascending-reduction `scf.for` bodies before ROCm
selects a physical schedule. The gfx1151 adapter fails closed unless all three
phase bodies have the verified nesting, common dO/Q/K/V/bias SSA values,
launch-owned two-split workspace, explicit `[2,B,Hkv,Sk,D]` FP32 partials, and
ascending reduction order. Only then does it produce the compiler-owned
five-entry HSACO contract. The runtime descriptor requires semantic route
`canonical_tensor_backward_scf_for`, so the legacy
`tile.attention_backward_kernel` carrier can no longer silently source this
optimized package.

Host compiler fixtures cover bias-bearing, canonical zero-bias, and incomplete
phase rejection. Exact gfx1151 combined bias+softcap+window+dropout gradients
remain within maximum absolute errors dQ `0.000024833`, dK `0.000035211`, and
dV `0.000329971`. With the HSACO, workspace, and user buffers resident, five
warmups plus 21 synchronized five-launch samples measure a `0.367367 ms`
median versus the `0.368203 ms` baseline and `0.405023 ms` 10% cap. The image
is 86,232 bytes and launch workspace remains 37,888 bytes. This WSL host-wall
timing is a regression gate, not selector-eligible device-event evidence.

Cross-backend sync `CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26`
materializes the verified deterministic backward contract as tensor-valued
shared `scf.for` bodies. dQ carries one FP32 result tensor through explicit
batch/head/query/KV blocks; dK/dV carry launch-owned
`[split,B,Hkv,Sk,D]` partial tensors through the canonical
batch/KV-head/split/query-block/key-block order and reduce them in ascending
split order. The direct shared forward recurrence now owns registered
score-bias and softcap operations in `softcap(scale*QK^T + bias)` order,
including per-head rank-4 bias, and the gfx1151 adapter consumes those
operations without reconstructing semantics. Exact gfx1151 combined
bias+softcap+dropout execution matches the shared oracle at max error
`0.000271678`; resident `hipModuleLaunchKernel` +
`hipDeviceSynchronize` median is `0.098631 ms` versus the `0.097763 ms`
base-feature baseline and passes the 10% ratchet. The test also repaired the
HIP launch ABI to pass dropout probability and seed before the trailing bias
memref. At this synchronization point, direct ROCm consumption of the new
backward phase operations remained follow-up and the launch carrier was still
the physical packaging boundary; the 2026-07-27 sync above supersedes that
temporary boundary. That retained carrier package revalidates combined
dropout replay at max errors dQ `0.000024833`, dK `0.000035211`, and dV
`0.000329971`; its five-launch resident median is `0.364209 ms` versus the
`0.368203 ms` baseline and passes the 10% ratchet.

Cross-backend sync `SSA-STATEFUL-TRANSPORT-2026-07-26` completes
`ROCM-SSA-LDS`: every active structured LDS copy reaches target lowering with
`!tile.buffer` allocation identity, a `!tile.async_token`, and threaded
`!tile.pipeline_state`. ROCm planning, legality, and target lowering no longer
read `#tile.buffer_ref`; shared barrier/WarpSpec and ROCm overlap fixtures are
SSA-owned, and direct structured lowering fails closed when any ownership edge
is missing. The deprecated attribute remains parser-visible only for migration
diagnostics and archived IR. Host WSL passes 49/49 ROCm lit and 223/223
supported shared IR lit tests. Exact gfx1151 ReplaySSM/MoE validation passes
14/14. Seven-run compiler medians are 14.19–14.34 ms for planning and
15.35–15.74 ms for full lowering across one, two, four, and eight stages, with
zero legacy references in every result.

The same sync maps the proven gfx1151 ReplaySSM handle to the shared
session-persistent workspace and target-keyed lifecycle descriptor. It records
exclusive ring leases through wait/release, flush-to-checkpoint and ring-clear,
cursor rollback/tail invalidation, ordered submission, and drain-before-release
teardown. ROCm MoE dispatch/combine now validate the canonical launch-owned
metadata descriptor, grouped GEMM consumes its canonical int32 partition, and
the descriptor can bind an RCCL rank/device topology fingerprint without
claiming unmeasured multi-rank execution.

Cross-backend sync `ROCM-E2E-ATTENTION-CARRIERS-2026-07-26` is **landing**
under ROCM-E2E-2. ROCm now consumes canonical `tile.attention_kernel` and
`tile.attention_backward_kernel` launch carriers instead of rejecting every
survivor. A dynamic correctness-first FP32 recurrence covers runtime
B/Hq/Hkv/Sq/Sk/D/Dv, GQA, causal/window masks, additive bias, softcap, dropout
replay, ragged tails, and deterministic output-element-owned backward
reductions. Static equal head/value buckets divisible by 16, f16/bf16 storage,
and the supported causal-window envelope select the existing gfx1151 WMMA
physical schedule; Sq/Sk remain runtime values. The optimized schedule now
composes nonzero dropout with deterministic counter replay and combined
additive bias plus softcap; ragged GQA/MQA and causal/window masks remain on
that path. Compiler-owned
native packaging and the launch descriptor now join that forward carrier to an
HSACO and HIP submission. Exact gfx1151 evidence for B=1, Hq/Hkv=4/2,
Sq/Sk=17/19, D=64 records a 20,440-byte image, 8.44e-05 maximum absolute error,
and a seven-sample 2.21 ms host-wall operation-total median after warmup. The
v2 resident benchmark now loads the HSACO and allocates/copies Q/K/V/O once,
warms up five launches, then measures 21 individual
`hipModuleLaunchKernel` + `hipDeviceSynchronize` intervals with
`perf_counter_ns`. It records a 0.097763 ms kernel-wall median separately from
the 2.312196 ms operation-total median; the cold operation-total sample is
164.79 ms and compiler packaging is 566.60 ms in that run. Both are host-wall
domains under WSL, and the resident domain explicitly excludes module load,
allocation, transfers, and cleanup. Neither is selector-eligible device-event
evidence. The combined ROCm lit lane is 50/50 and the focused package/audit
gates remain green. The optimized feature slice adds exact-device coverage
for ragged GQA with bias, softcap, windowing, and dropout together; all six
compiled forward cases pass. The resident performance ratchet is now encoded
against the 0.097763 ms baseline with a 10% regression limit. A 21-sample run
after five warmups measured 0.095341 ms (0.9752x baseline), 8.44e-05 maximum
absolute error, and passed the ratchet. Backward native packaging now lowers
the same canonical carrier to one 83,800-byte HSACO containing forward
recompute, prepass, deterministic two-split dK/dV, reduction, and dQ entries.
One compiler-owned, 256-byte-aligned 37,888-byte launch workspace holds
forward O, row LSE/delta, and the two partial-gradient slices without overlap.
Exact gfx1151 B=1, Hq/Hkv=4/2, Sq/Sk=17/19, D=64 bias+softcap+window execution
records maximum absolute dQ/dK/dV errors of 1.55e-05, 2.20e-05, and 1.65e-04.
After five warmups, 21 synchronized resident five-kernel samples record a
0.368203 ms program-wall median; operation-total is 155.237686 ms and cold
compiler packaging is 961.303488 ms. The resident value is now the WSL
host-wall regression baseline with a 10% cap, not selector-eligible
device-event evidence. A fresh 21-sample ratchet run records 0.391975 ms,
below the 0.405023 ms cap, with the same gradient errors. Direct LDS-pipelined
canonical forward consumption lands in the follow-up below. Backward now
replays the forward `lcg32_counter_v1` mask in dP and dV without storing it.
Exact ragged-GQA bias+softcap+window+dropout errors are 2.48e-05, 3.52e-05,
and 3.30e-04 for dQ/dK/dV. A 5-warmup/21-sample resident run records
0.377553 ms against the 0.368203 ms baseline and 0.405023 ms cap; its
86,232-byte HSACO retains the 37,888-byte workspace. WSL host-wall timing
remains selector-ineligible.

Cross-backend sync `CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` is **landing**
under `CORE-STREAMING-ATTN-2026-07-26`. Shared lowering now distributes static
rank-4 attention through explicit batch and query-head `scf.for` loops, maps
GQA query heads to KV heads, and rank-reduces each head into the single
canonical KV-block recurrence. ROCm consumes that recurrence directly after
SSA LDS planning: target-stamped physical pipeline advances refine, rather
than replace, the two semantic producer/consumer pipeline values. The native
forward package uses this route for bias-free, non-softcap MHA/GQA including
causal left windows, ragged zero fill, and deterministic dropout metadata;
bias and softcap remain explicitly on the compatibility carrier until their
shared recurrence semantics land.

Historical landing state, superseded by
`CORE-ATTENTION-TENSOR-LOOPS-MODIFIERS-2026-07-26` and
`ROCM-ATTENTION-SHARED-BACKWARD-CONSUMER-2026-07-26`: exact gfx1151 execution
for B=1, Hq/Hkv=4/2, Sq/Sk=17/19, D=64 produced a
17,624-byte HSACO and 8.47e-05 maximum absolute error. After five warmups, 21
resident `hipModuleLaunchKernel` + `hipDeviceSynchronize` wall samples record a
0.095145 ms median, 0.9732x the existing 0.097763 ms baseline and below its
0.1075393 ms gate. Operation-total median is 2.322677 ms, cold operation-total
is 167.533902 ms, and packaging is 328.384463 ms. These WSL host-wall values
remain non-selector evidence. The checked-in packet is
`benchmarks/baselines/rocm_gfx1151_canonical_streaming_attention.json`.
The backward carrier verified launch-owned workspace, split count, 16-row
query/KV blocks, and ascending reduction order. Shared numerical semantics and
ROCm WMMA forward/LSE/gradient recomputation agreed on
`softcap(scale*QK^T + bias)`. The later synchronization points materialize the
tensor-valued shared backward `scf.for` bodies and consume shared forward
bias/softcap directly; those are closed and must not re-enter the active queue.

Cross-backend sync `ROCM-SSA-LDS-PIPELINE-2026-07-26` is **complete** under the
ROCm follow-up to `NVIDIA-PACKED-SSA-FOUNDATION-2026-07-25`. The
`rocm-wave-lds-pipeline` planner now makes `!tile.buffer` allocation roots,
`!tile.async_token` completion, and producer/consumer
`!tile.pipeline_state` def-use chains the physical ownership proof for AMD
global-to-LDS copies, waitcnt retirement, and WMMA/MFMA consumers. The
gfx1151 target lowering consumes the proof,
records SSA allocation identity and LDS bytes/layout on the target copy, and
removes dead portable lifetime operations at the target boundary. Structural
lit coverage, SSA overlap rejection, branch-local dominance, and a
repeated-median host-compiler benchmark are included. Post-rebase host WSL
validation is 49/49 ROCm lit and 257 passed with 19 intentionally skipped
focused unit/exact-device checks, including runnable
global-to-LDS and LDS-staged WMMA execute/compare on visible gfx1151. Seven-run
compiler medians span 14.19–14.34 ms for planning and 15.35–15.74 ms for full
ROCm lowering across one, two, four, and eight independent stages. Those are
compiler timings, not device-kernel latency or selector evidence.
Apple/NVIDIA schedules and evidence do not transfer.

Cross-backend sync `PACKED-LEGALIZE-CAPABILITY-2026-07-26` makes the shared
terminal storage pass inspect an operation's structured physical descriptor
and complete def-use consumer rather than treating a low-precision dtype as
execution evidence. NVIDIA SM120 now enables only its proven packed
load/unpack, unscaled load/store round trip, matching packed matmul, and
explicit conversion paths. ROCm's existing architecture-owned signed-INT4
WMMA gate and legacy fallback remain unchanged; scale-bearing FP4/FP6 generic
HIP consumers are still follow-up required on the ROCm box. The deprecated
`#tile.buffer_ref` attribute remains parser-only for archived IR; active ROCm
passes do not consume it. CUDA descriptors, schedules, SM120 evidence, and
selector state do not transfer.

Cross-backend sync `CORE-STREAMING-ATTN-2026-07-26` replaces the shared
rank-2 FlashAttention whole-KV lowering with an explicit KV-block `scf.for`
carrying the FP32 output accumulator, running maximum, normalization sum,
producer/consumer `!tile.pipeline_state` values, and absolute boundary offset.
The shared async seam now carries typed block coordinates and logical source
extents for ragged zero fill. NVIDIA WarpSpecialization also retires its
name-based `#tile.buffer_ref` and annotation-only `#tile.pipeline_state`
emission. Follow-up sync
`CORE-STREAMING-ATTN-RANK4-ROCM-2026-07-26` now maps rank-4 distribution and
the recurrence to gfx1151 LDS/waitcnt/barrier plus WMMA using AMD-owned SSA
allocation identity, with exact-device numerical and resident wall proof.
CUDA TMA/mbarrier mechanics, SM120 evidence, resources, timings, and selectors
do not transfer. Canonical deterministic backward workspace materialization
remains open shared work; no selector changes in this synchronization slice.

Cross-backend sync `CORE-GEMM-KLOOP-2026-07-25` changes the shared
Graph/Schedule→Tile GEMM contract to explicit M/N/K `scf.for`, FP32/INT32
loop-carried accumulation, zero-pad ragged guards, structured layouts, and SSA
pipeline dependencies. ROCm is **follow-up required** to consume that loop
through its architecture-owned WMMA/MFMA and LDS/waitcnt/barrier lowering.
NVIDIA MMA fragment sizes, PTX, SM120 resources, CUDA cache identity, timings,
and selector evidence do not transfer to gfx1151 or another AMD target. No ROCm
capability, execution state, schedule, or selector changes in this
synchronization slice.

Cross-backend sync `ROCM-CORE-GEMM-KLOOP-2026-07-27` closes that gfx1151
follow-up. The direct consumer now accepts the canonical loop only after shared
Tile lowering and the ROCm planner have placed real `!tile.buffer`,
`!tile.async_token`, and `!tile.pipeline_state` edges on the matrix consumer;
malformed or proof-free marked loops fail closed. The compiler owns both the
register incumbent and an explicit address-space-3 LDS comparison schedule
with cooperative ragged zero fill, K-loop-carried FP32/INT32 accumulation,
barriers, and gfx1151 WMMA. Exact aligned/ragged FP16, BF16, and INT8 rows all
execute and compare. The six register medians are 0.086620–0.115159 ms and pass
their 10% wall-clock ratchets with zero spills. The LDS lane uses 512–1024
bytes, 39–43 VGPRs, 31 SGPRs, and zero spills, but is 1.0684–1.55094x slower,
so it is retained as a correctness/comparison route and the register schedule
remains selected. The durable packet is
[`../../../../benchmarks/baselines/rocm_gfx1151_canonical_gemm_kloop.json`](../../../../benchmarks/baselines/rocm_gfx1151_canonical_gemm_kloop.json).
Apple simdgroup and NVIDIA Tensor Core schedules/evidence do not transfer.

Cross-backend sync `COMPILER-LIT-BACKEND-GATING-2026-07-24`: retired seven
never-runnable ROCm 7.2 pseudo-IR fixtures whose undefined
`tessera_opt_built` feature masked a nonexistent global target flag,
unregistered operations, and obsolete MFMA/WMMA arity. Canonical
architecture-keyed Tile→ROCm→ROCDL coverage remains in the ROCm backend lit
suite and the 17 correctly gated shared-suite ROCm fixtures pass against the
ROCm-enabled compiler. `validate_hipcc_compile.py` now labels and runs its
handwritten intrinsic catalog strictly as a HIP-toolchain probe, not Tessera
emission evidence. No exact-device status or selector changed.

Cross-backend sync `COMPILER-PYTEST-PLATFORM-SKIPS-2026-07-24`: shared
compiler-owner markers now report foreign compiler proofs as skipped with the
required Apple, CUDA, ROCm, X86, or AVX512 system and a per-system count. This
is test-harness observability only; ROCm compiler ownership, HIP evidence, and
selector state are unchanged.

Cross-backend sync `STATEFUL-TRANSPORT-FOUNDATION-2026-07-19`: the shared launch
workspace schema now distinguishes per-launch scratch from session-persistent,
preserved state. ReplaySSM and MoE metadata contracts are portable, but this
NVIDIA slice changes no HIP allocation, wave schedule, event/ring protocol,
resource claim, timing row, or selector. ROCm's proven gfx1151 resident handle
must map its lifecycle to the shared descriptor in a ROCm-owned follow-up;
CUDA local-device bandwidth supplies no gfx1151 or multi-rank evidence.

Cross-backend sync `NVIDIA-E2E2-STATEFUL-REDUCE-2026-07-19` extends the shared
Tile surface with explicit ReplaySSM decode/flush, MoE dispatch/combine/grouped
GEMM, and `Outer/AxisExtent/Inner` reduction carriers, plus a backend-neutral
rank/device topology fingerprint. ROCM-E2E-2 must assess mapping these carriers
to the existing HIP generators and RCCL execution; this is follow-up required,
not CUDA parity evidence. ROCm inherits no warp schedule, PTX ABI, NCCL call,
resources, timing, or selector, and gfx1151 remains unsupported for FP8 WMMA.

This is the working ROCm implementation queue. It consolidates the open actions
from [`ROCM_AUDIT.md`](ROCM_AUDIT.md), the portable Tile fragment work in
[`tile_fragment_abi.md`](../../../architecture/proposals/tile_fragment_abi.md),
the serving work in
[`REPLAYSSM_PLAN.md`](../../roadmap/archive/REPLAYSSM_PLAN.md), and the generated exact-
target status in [`rocm_target_map.md`](../../generated/rocm_target_map.md).

The generated target map and runtime/conformance dashboards remain the status
authorities. This file owns sequencing and completion gates; it must not promote
an artifact-only row by prose.

## Rules of completion

A ROCm item is complete only when all applicable evidence refers to the same
exact target:

1. the compiler emits a target-valid artifact;
2. the artifact assembles for the named gfx architecture;
3. the production runtime launches it;
4. device output matches a numerical oracle, including required ragged cases;
5. performance work compares against the retained production path with device
   timing and an explicit promotion threshold;
6. evidence records the actual `evidence_arch` and updates the generated target,
   runtime, and conformance views.

The generic `rocm` target is a family rollup. Evidence from gfx1151 must never
promote gfx1200, gfx1201, gfx1250, gfx942, or gfx950.

## Current baseline—not TODO

The following foundations already execute on `rocm_gfx1151` and should be
preserved while completing this queue:

- general f16/bf16/int8/int4 WMMA GEMM with tiled K loops, ragged boundaries,
  fused epilogues, runtime launch, and size-aware macro-tiles;
- compiler-generated flash-attention forward/backward, GQA/MQA/MHA, sliding
  windows, bias, logit soft-capping, and causal/ragged handling;
- linear, sparse, recurrent, DeltaNet, normalization, activation, reduction,
  positional, MoE, grouped-GEMM, and selective-SSM lanes listed as verified in
  the generated exact-target map;
- runnable global-to-LDS asynchronous copy and structured layout consumption;
- a versioned PLHD paged-KV ABI with an i32 logical-to-physical page table;
- cooperative sparse attention and resident top-k with committed comparative
  ratchets.
- Cross-backend sync `NVIDIA-TEST5-2026-07-16`: the shared autotune corpus v3
  adds compiler/resource, cold/warm, cache, and two-run stability evidence while
  retaining v1/v2 loading. ROCm corpus round-trip and warm-start behavior are
  parity validated by `test_rocm_measured_autotune.py`; no NVIDIA schedule,
  resource claim, or selector decision applies to gfx1151 or other AMD targets.
- Cross-backend sync `LLVM23-NVIDIA-2026-07-16`: ROCm's lit configuration uses
  LLVM 23's supported internal shell, matching its already recorded 32/32
  gfx1151 WSL proof. The Ubuntu bootstrap now probes the apt.llvm.org suite so
  current LLVM 23 snapshot packages work on Resolute as well as the documented
  Noble host. No NVIDIA lowering, route, timing, or resource evidence is
  transferred to ROCm; no new AMD exact-device claim is made here.
- Cross-backend sync `NVFP4-TILE-SCALES-2026-07-16`: shared typed Tile IR now
  permits logical `scale_a`/`scale_b` fragments only on NVFP4 MMA descriptors.
  This is not applicable to enabled gfx1151 WMMA matrix forms: gfx1151 has no
  NVFP4 block-scaled matrix instruction. ROCm retains its named unsupported
  capability result; NVIDIA nibble packing and scale-selector lane maps are not
  transferred.
- Cross-backend sync `PR420-REVIEW-2026-07-17`: the NVFP4 scale-origin repair
  and canonical `fp16` selector alias are NVIDIA-only and do not change ROCm
  fragment layouts, dtype support, runtime ABI, or exact-device evidence. The
  shared Ubuntu bootstrap now installs `ca-certificates`, `wget`, and `gnupg`
  before probing apt.llvm.org and removes only its version-owned stale source
  file before the prerequisite update. This is parity validated as setup
  infrastructure; it does not transfer CUDA schedules or make a new gfx claim.

## LLVM/MLIR 23 and ROCm 7.14 transition evidence

**Status: host build and gfx1151 correctness ratchets complete in WSL
(2026-07-16); bare-metal-only gates remain open.**

The project-wide compiler floor is now a matched LLVM/MLIR 23 toolchain. On the
gfx1151 WSL host, the validated configuration uses upstream Ubuntu LLVM/MLIR
23.0.0 for Tessera's C++ compiler and TheRock Core SDK 7.14 for HIP, HIPRTC,
device libraries, and the HIP compiler. Mixing the former LLVM/MLIR 23 build
with TheRock's LLVM 23 `ocml.bc` was rejected after the reader reported an
LLVM-bitcode attribute-version mismatch.

The clean `build-rocm-7.14-llvm23-clean` configuration and full Ninja build
pass.
The migration includes MLIR 23's removed dialect property switch, Queue
TableGen name collision, greedy-rewrite API split, tiling-interface alignment
overload, vector multi-reduction API split, and MFMA control operands becoming
attributes. Validation on the visible `gfx1151` device records:

- initial ROCm Target IR transition slice: **32/32 pass**; the current expanded
  suite is **53/53** after typed composition, strict executable-boundary, and
  dtype-totality additions;
- compiled ROCm correctness corpus on gfx1151: **1280/1280 pass**;
- valid baseline/performance ratchets: **21/21 pass**;
- combined paged-KV, ReplaySSM, portable Tile, grouped GEMM/SwiGLU, and
  architecture sweep: **86/90**, with only four source-confirmed invalid
  zero-event assertions remaining and no gfx1250/gfx1251 LLVM 23 failures;
- HIP version **7.14.60850**, TheRock HIP clang **23.0.0git**, and upstream
  LLVM/MLIR **23.0.0**.

This is WSL exact-gfx1151 correctness evidence, not bare-metal transport
evidence and not evidence for any sibling architecture.

## Status ledger

| ID | State | Current outcome |
|---|---|---|
| LLVM23/ROCm 7.14 | complete on gfx1151 WSL | Clean build, 53/53 ROCm lit, 1280/1280 compiled correctness, and 21/21 valid performance ratchets pass; the combined sweep is 86/90 with four zero-event-only failures. |
| ROCM-TILE-1 | complete on gfx1151 | Portable f16/bf16/int8/int4 fragments execute and compare on gfx1151. Other architectures are owned by ROCM-1 through ROCM-5. |
| ROCM-9 | complete on gfx1151 | Non-identity paged-KV direct and gather routes execute, compare, and have a measured serving decision. |
| ROCM-REPLAY-1 | complete on gfx1151 | Persistent state, flush/rollback, block submission, asynchronous ring, lifetime proof, and the wider performance matrix are committed. |
| ROCM-6 | open revalidation; timing blocked | LLVM/MLIR 23 + ROCm 7.14 correctness is green for G6-A/B/C, but WSL HIP events return invalid zero durations. Existing production choices stay in force pending valid paired device timing. |
| ROCM-8 | blocked | Bare-metal gfx1151 access is required; WSL characterization cannot close it. |
| ROCM-1/2/3 | open, access-gated | P0 exact-device execution on gfx950, gfx1201, and gfx1250 is the active release frontier. |
| ROCM-4a/4b | open, access-gated | P1 compatibility execution on gfx1200 and gfx942 follows the P0 packet. |
| ROCM-5 | landing, exact-device closure open | Architecture-owned descriptors and cross-assembly exist; numerical and performance closure depends on ROCM-1 through ROCM-4b. |
| ROCM-E2E-1 | complete on gfx1151 | The f16/f32 pilot lowers typed Tile IR to `tessera_rocm.softmax`, packages an ELF HSACO and descriptor, executes across the exact-device boundary/aligned/ragged matrix, rejects invalid contracts, retains driver-selected device-library plus cold/warm identity, and passes isolated device and end-to-end non-regression. |
| ROCM-E2E-2 | complete on gfx1151 | Reduction covers f16/bf16/f32 input with f32 output and passes all paired gates. Direct f32/i32 paged-KV and MoE dispatch have typed artifact/descriptor, negative, exact-device, and retained-route evidence; both movement descriptors are measured non-winners, so production routes remain selected. |
| E2E-SPINE-3 | complete on gfx1151 | The hash-sealed four-family release packet records exact-device Level-C results, cold/warm identity, resources, and stable kernel-wall/end-to-end timing for softmax, reduction, paged-KV, and MoE. All four fleet rows are `release_ready`; no production selector changed. |
| ROCM-TEST-1 | complete | The ROCm-only LLVM/MLIR 23 build owns a 35-node host-free compiler lane; 35/35 pass with Apple/NVIDIA/CPU ownership excluded and foreign pipeline absence retained in the report. |
| ROCM-DTYPE-1 | complete on gfx1151 | FP64 and integer widths have per-operation Target-IR/runtime assessments; unsigned LLVM probes pass without inventing unsigned storage ABIs; signed int4 is canonical and physically packed; gfx1151 FP8/BF8 is rejected by name. |
| ROCM-SSA-LDS | complete | AMD async-copy, waitcnt, and matrix consumers use shared SSA allocation, token, and pipeline-state identity; compatibility readers are retired, shared/ROCm fixtures are SSA-only, host structural and compiler-benchmark gates pass, while exact-device performance remains intentionally unclaimed. |
| ROCM-E2E-ATTENTION | complete on gfx1151 | FP16 and BF16 forward consume the canonical rank-4 recurrence with per-head bias, softcap, dropout, GQA/window/ragged policy, and exact numerical/timing proof. FP16 and BF16 backward consume the tensor-valued shared dQ/split-partial/ascending-reduction loops directly into the compiler-owned five-entry package; exact combined gradients and dtype-specific resident program-wall ratchets pass. |
| ROCM-CORE-GEMM-KLOOP | complete on gfx1151 | Canonical M/N/K loops reach the compiler-owned register and LDS WMMA schedules only through SSA allocation/token/pipeline proof. Six aligned/ragged FP16/BF16/INT8 rows pass; measured LDS staging is retained but not selected because it loses all six wall-clock comparisons. |
| ROCM-CALIB-1 | complete; terminal reject | Re-derived gfx1151 wave32/32-bank constants; the locality score reproduced 0/6 committed winners (median rho -0.1381, 0% positive), so the latency-ranking line ends without retuning. Bank conflict remains structural-only for explicit LDS traces. |
| ROCM-RASTER-1 | implementation complete; promotion blocked | Shared raster fields reach the generated WMMA kernel and 24/24 live gfx1151 correctness rows pass. WSL HIP events are zero and rocprofv3 exposes no PMCs, so row-major remains selected pending valid device timing plus MALL/L2 evidence. |

## Recommended open-work order

Completed and measured-non-winning gfx1151 work is intentionally absent from
this queue. The local WSL host may prepare artifacts and harnesses, but only the
named exact device can satisfy an execution gate.

| Order | ID | Work | Access state | Completion gate |
|---:|---|---|---|---|
| 1 | ROCM-COSTMODEL-T1 | Validate T1 reuse/cache pruning on gfx1151; keep step-distance rejected | Strix Halo `gfx1151`; committed retune and hot-path corpus | Rank correlations over the retune corpus and hot-path ratchet establish a retain/reject verdict; failure ends T1 rather than triggering coefficient tuning. |
| 2 | ROCM-RASTER-1B | Collect promotion evidence for the implemented shared raster contract | bare-metal or profiler-supported Strix Halo `gfx1151`; 24/24 WSL correctness rows already pass | Valid device-event timing and `rocprofv3` MALL/L2 counters establish any architecture-owned raster-order/group decision; row-major remains selected until then. |
| 3 | ROCM-2 | Run the common P0 packet on Radeon AI PRO R9700 `gfx1201` | owner and reservation required | RDNA 4 WMMA-v2 f16/bf16 plus enabled FP8/integer forms assemble, launch, match aligned/ragged oracles, and record resources and timing. |
| 4 | ROCM-1 | Run the common P0 packet on MI350-series `gfx950` | owner and reservation required | CDNA 4 matmul, flash attention, softmax, and GELU launch and compare; low-precision breadth advances only with physical-layout proof. |
| 5 | ROCM-3 | Run the common P0 packet on MI455X `gfx1250` | owner and reservation required | The upstream-LLVM artifact joins to a launch/numerical proof; WMMA-v2 properties and fragment layout match the device. |
| 6 | ROCM-6 | Revalidate G6-A/B/C with valid paired device timing | bare-metal gfx1151 or repaired event timing required | Original correctness, resource, aligned/ragged, dtype, device-time, and E2E gates are rerun under LLVM/MLIR 23 + ROCm 7.14 before reaffirming or changing production. |
| 7 | ROCM-8 | Measure copy versus mapped-host memory on bare-metal `gfx1151` | bare-metal owner and reservation required | Repeated kernel-only and end-to-end measurements establish a stable crossover without using WSL evidence. |
| 8 | ROCM-4b | Retain compatibility proof on MI300X/MI325X `gfx942` | owner and reservation required | f16/bf16 MFMA plus retained matmul/attention/softmax/GELU paths launch and compare. |
| 9 | ROCM-4a | Add Radeon RX 9000 `gfx1200` exact-device proof | owner and reservation required | Matmul launches and compares; unsupported forms reject stably. |
| 10 | ROCM-5 | Close the architecture-owned fragment umbrella | depends on ROCM-1 through ROCM-4b | Every enabled family/dtype has exact-device packing, numerical, resource, and timing evidence, or an explicit unsupported/deferred state. |
| 11 | ROCM-LSE-1 | Extend the landed gfx1151 LSE selector beyond WSL | **gfx1151 implementation and initial decision complete; bare-metal/CDNA follow-up gated** | The real shared checkpoint contract, selectable physical save/recompute paths, exact FP16/BF16 17/64/128/256 correctness sweep, retained host-wall evidence, and 128+ saved selector are landed. Revalidate on bare-metal gfx1151 and measure architecture-owned thresholds on gfx950/gfx1250; do not transfer the WSL threshold. |

### `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` verification result

| Sync key | ROCm host and build identity | Lean guard | Unit result versus Apple baseline | Streaming-attention lit | Remaining gates and disposition |
|---|---|---|---|---|---|
| `TESSERA-OPT-BUILD-CAPABILITY-2026-07-27` | **Strix Halo WSL `gfx1151`**, HIP **7.14.60850**, LLVM/Clang **23.0.0**. The full build configured and built cleanly. `--tessera-build-info` reported `build profile: full` and `features: core-tessera-ir fa4-attn fa4-queue neighbors rocm-backend scaling-resilience solvers tpp`. | **2a passed:** lean ROCm + Apple failed configuration nonzero and named only `apple-backend`, with both resolutions. **2b passed after repair:** the legitimate lean ROCm driver builds and reports `build profile: lean-artifact-driver`, `features: rocm-backend`; ambient default FA4 targets no longer masquerade as explicitly requested driver features. | Repository-venv run: **59 failed, 13,485 passed, 2,037 skipped, 857 deselected**, versus the Apple **178 failed / 12,761 passed** baseline. None of the 35 enumerated ROCm pass-registration files failed and no `Unknown command line argument '--generate-rocm-*'` survived. The nine migrated helper files passed **610/610** with no “does not register” skip. The remaining broad-suite failures were foreign/backend-local or host-state failures, not ROCm pass-registration failures. | The documented `TESSERA_OPT_BIN` invocation now selects `build-rocm` directly: the two named fixtures pass **2/2**, the complete `streaming_attention` filter passes **7/7**, and the rebuilt separate `tessera-rocm-opt` backend suite passes **50/50** through `test_rocm_lit_suite.py`. | `tessera-emit-rocdl` was listed; bogus options emitted `error: this pipeline takes no options` and exited nonzero; MLIR→HSACO passed **3/3**. The sealed gfx1151 packet validated (4 families, 4 Level-C fixtures, max absolute error 0.0), and **24 generated docs** were in sync. The verification item is complete. |

## ROCM-TEST-1: host-free compiler ownership

**Status: complete (2026-07-21).**

The clean ROCm-only capability set declares ROCm and HIP
enabled with Apple, NVIDIA, and CUDA disabled. The owned marker expression
explicitly excludes Apple, CPU, NVIDIA, performance, and every hardware lane;
it collects 35 ROCm compiler nodes and passes **35/35** with 17,675 tests
deselected. The ROCm pipeline probe is available, while the Apple and NVIDIA
pipeline probes retain their expected unregistered-pipeline diagnostics. The
report also retains the build command, CMake flags, LLVM/MLIR 23 tool and runner
paths, exact node IDs, collection diagnostics, and final pytest output at
`docs/audit/evidence/compiler/rocm_gfx1151/llvm23-host-free/report.json`.

Nine parametrized Metal-only compiler nodes discovered during ownership
collection now carry the existing `compiler_apple` owner marker and are listed
in the Apple inventory. This is an ownership correction, not a compiler
semantic or backend capability change.

## ROCM-E2E-1: typed softmax compilation spine

**Status: complete on `gfx1151` (2026-07-19).**

The first slice consumes the shared `tile.softmax_kernel(X, O, Rows, K)`
semantic envelope and emits `tessera_rocm.softmax` from the architecture-owned
`lower-tile-to-rocm` pass. The canonical ROCm pipeline invokes the existing
softmax generator after that adaptation; no other standalone ROCm generator
was appended. Python supplies the typed Tile request and packages the resulting
Target IR, ROCDL-produced ELF HSACO, ordered f16/f32 ABI, shape guards, gfx1151
workgroup policy, and launch descriptor. The generic runtime validates that
descriptor before the exact `rocm_gfx1151` hook allocates, submits, copies back,
and cleans up the HIP resources.

Host WSL validation used LLVM/MLIR 23 with the ROCm 7.14 build and a visible
Radeon 8060S `gfx1151`. The focused lit fixture passed, the Python/registry
slice passed 154 tests, real packaging produced a 5,808-byte ELF HSACO, and the
expanded focused slice passed 24 tests. Eight exact descriptor launches cover
f16/f32 at boundary `(1,1)`, aligned `(4,256)`, ragged `(3,17)`, and
multi-stride `(2,257)` shapes and match the stable-softmax oracle. Host-free
negatives reject unsupported dtype, dynamic shape, mismatched result, runtime
dtype/shape binding, and scalar contracts before submission.

AMD clang is authoritative for the `gfx1151` `--rocm-path` selection. The
native image records content digests for OCML, OCKL,
`oclc_unsafe_math_off`, `oclc_finite_only_off`,
`oclc_wavefrontsize64_off`, `oclc_isa_version_1151`, and
`oclc_abi_version_600` with `compiler_driver` link mode and no installation
paths. Those records enter both cache and toolchain identity. An exact cold then
warm package retained identical image, payload, and library identities. The
legacy runtime-authored `rocm_softmax_compiled` path remains intact and no
selector changed.

The isolated serial performance gate is recorded in
`benchmarks/baselines/rocm_gfx1151_e2e_softmax_comparison.json`. Nine
alternating paired trials per row keep 100-launch HIP-event timing separate
from allocation/copy-inclusive `runtime.launch` wall time. All eight f16/f32
aligned, ragged, and multi-stride rows remain numerically correct with exact
route parity. Compiler/retained resources match at 16 VGPR, 14 SGPR, 32 bytes
LDS, zero private segment, and zero spills. Device speedups span
0.981--1.008x and pass the 10% non-regression gate on every row.

The first A/B run exposed that the retained `rocm_softmax_compiled` executor
freed device buffers but did not unload its HIP module and leaked resources on
several failures. That lifecycle defect is fixed before accepting comparison
evidence. With both routes cleanup-complete, the first A/B run isolated the two
fp16 misses to repeated deterministic identity work on the descriptor route:
`RuntimeArtifact.artifact_hash` cost about 84.8 us, the image and descriptor
digests about 22 us together, and required per-launch validation about 24 us.
The immutable artifact, image, and descriptor identities are now memoized;
contract validation still runs on every launch. The unchanged serial gate then
passes all eight rows: device speedups span 0.981--1.008x and end-to-end
speedups span 0.979--1.022x, including fp16 `(128,256)` at 1.009x and
`(64,1024)` at 0.997x. Resources and numerical results remain identical.
ROCM-E2E-1 is complete. The incumbent route and production selector remain
unchanged; retiring runtime text synthesis is an explicit ROCM-E2E-2 route
decision, not an implication of this measurement.

Cross-backend sync `E2E-FROZEN-IDENTITY-CACHE-2026-07-19`: deterministic hashes
for frozen runtime artifacts, native images, and launch descriptors are cached
without changing their serialized values or validation rules. CUDA and Metal
schema parity is validated; no sibling ABI, schedule, runtime route, timing
claim, or selector changes.

## ROCM-E2E-2: typed directive and generator breadth

**Status: complete on `gfx1151` (2026-07-19).**

The reduction breadth slices consume the already-shared
`tile.reduce_kernel(X, O, Outer, AxisExtent, Inner)` carrier. ROCm lowering
requires explicit f16/bf16/f32 storage, f32 accumulation/output, normalized
axis, keepdims, sum/mean/max semantics, NaN propagation, and the portable
serial schedule, then selects its own 256-thread workgroup-per-output
implementation. The existing legacy four-argument row-reduction directive
remains valid; only the typed carrier selects the five-argument
`outer_axis_inner` ABI.

Canonical packaging retains Tile IR and typed `tessera_rocm.reduce` Target IR,
builds an ELF HSACO, records the driver-selected device-library identities, and
emits shape guards plus `Outer/AxisExtent/Inner` scalars. The exact gfx1151 WSL
host passes f16 sum on axis 0, bf16 mean on a middle axis with `keepdims`, and
f32 max on the last axis against NumPy. The shared Tile verifier now admits
bf16 reduction storage; NVIDIA keeps its backend-specific f16/f32 boundary and
Apple mappings are unchanged.

The isolated serial comparison is recorded in
`benchmarks/baselines/rocm_gfx1151_e2e_reduce_comparison.json`. Nine alternating
paired trials per row separate resident HIP-event timing from full
`runtime.launch` wall time across f16/bf16/f32 sum/mean/max and axis-0/middle/
last layouts. The kernel hoists arbitrary-axis base/stride calculations and
specializes the last-axis case. All nine comparison rows pass: end-to-end
speedups span 0.934--1.020x and the layout-equivalent last-axis device rows span
0.935--1.011x. Device-event values for axis-0/middle rows remain diagnostic,
not a promotion gate, because the incumbent performs an untimed host transpose
before launching a contiguous row kernel while the typed route directly reads
the original strided layout. Descriptor-minus-retained host overhead spans
-0.041 through +0.155 ms. The selector and retained route remain unchanged.

The next family consumes shared
`tile.paged_kv_read_kernel(Pages, Table, O, P, LP, PageSize, H, D, Start,
Tokens)`, lowers to typed `tessera_rocm.paged_kv_read`, emits a direct 256-thread
f32 gather with an i32 page table, and packages the three-buffer/seven-scalar
descriptor. The runtime validates shapes, range, dtypes, and every physical
page index before submission. Exact gfx1151 execution matches a non-identity
permuted-page oracle bit-for-bit at single-token, page-crossing, short-ragged,
and full-capacity ranges. Static contract and pre-HIP negatives cover bounds,
result shape, table dtype, and invalid physical pages. This adds no selector
promotion and does not replace the existing ROCM-9 HIP movement route.

The third family consumes shared `tile.moe_dispatch_kernel(X, Token, O, T, S,
H)`, lowers to typed `tessera_rocm.moe_dispatch`, and generates a direct
256-thread f32 gather from an i32 token-of-slot vector. Its three-buffer/
three-scalar descriptor rejects out-of-range indices before HIP. Exact gfx1151
execution is bit-exact at tiny `(1,1,1)`, ragged `(7,9,13)`, and wide
`(17,5,257)` shapes.

The final paired movement record is
`benchmarks/baselines/rocm_gfx1151_e2e_movement_comparison.json`. Typed
paged-KV is device-competitive at 0.960x but only 0.282x end-to-end versus the
retained HIP route; typed MoE dispatch is 0.826x end-to-end versus the retained
row gather. Both remain numerically exact. These measured non-winning results
close the route disposition without weakening the 10% promotion threshold:
ROCM-9 paged movement and the retained MoE transport stay selected, and no
runtime-authored route is retired. Future attention, backward, ReplaySSM, or
additional transport carriers are separately scoped breadth work rather than
silently extending ROCM-E2E-2. The item is complete for its reduction plus two
movement-family scope.

Cross-backend sync `ROCM-E2E2-REDUCE-2026-07-19`: this slice consumes the
existing shared reduction carrier and widens its storage verifier to bf16; the
scalar schema and public op registry are unchanged. NVIDIA retains an explicit
backend f16/f32 boundary, and Apple has no mapping change. The ROCm lowering,
five-argument ABI, HSACO, measurements, and selector state transfer no sibling
backend claim.

Cross-backend sync `ROCM-E2E2-PAGED-KV-2026-07-19`: ROCm consumes the existing
shared paged-KV carrier and public operation without changing either verifier
or schema. The new ROCm target directive, gather schedule, HSACO ABI, runtime
validation, and exact-device evidence are architecture-owned. NVIDIA's PTX
mapping remains parity validated at the carrier boundary; Apple retains its
independent Metal/MPS paged-cache routes.

Cross-backend sync `ROCM-E2E2-MOE-DISPATCH-2026-07-19`: ROCm consumes the
existing shared MoE dispatch carrier and public operation without changing
their verifier, dtype registry, or scalar ABI. The ROCm directive, direct
gather schedule, HSACO descriptor, pre-launch index validation, and gfx1151
evidence are architecture-owned. NVIDIA and Apple retain their independently
scheduled transport routes; no AMD timing, readiness, or selector claim
transfers.

## ROCM-DTYPE-1: gfx1151 datatype totality

**Status: complete on `gfx1151` (2026-07-21).**

`python/tessera/compiler/rocm_dtype_contract.py` separates ISA support from
Tessera execution readiness. Every canonical and planned/gated dtype has one
row; every positive architecture claim names an opcode present in the
AMD-PDF-derived `rdna35/instructions.json`. The community RDNA3.5 Markdown is
retained as a human-readable cross-check, not substituted for the checked-in
JSON source and hash.

| Format group | RDNA3.5 scalar/vector role | gfx1151 WMMA role | ROCm/LLVM state | Tessera target state |
|---|---|---|---|---|
| fp64 | native vector arithmetic and conversion | unsupported | validated | assessed unavailable: no numeric Target-IR generator/runtime ABI |
| fp32 | native | accumulator only | validated | ready |
| fp16 | native | input; fp16/fp32 accumulator | validated | ready |
| bf16 | packed dot/WMMA | input; bf16/fp32 accumulator | validated | ready |
| int8 / uint8 | packed dot | IU8 input; int32 accumulator | validated | int8 ready; uint8 compiler-validated but planned-gated |
| int4 / uint4 physical IU4 | packed dot | IU4 input; int32 accumulator | validated | signed int4 ready for matmul; no first-class uint4 spelling |
| int16 / uint16 | packed/native vector arithmetic | unsupported | validated | int16 assessed unavailable; uint16 planned-gated |
| int32 / uint32 | native vector arithmetic | int32 accumulator only | validated | int32 ABI-only for indices/results/WMMA accumulator; uint32 planned-gated |
| int64 / uint64 | expanded instruction sequences | unsupported | validated | int64 ABI-only for shape scalars; uint64 planned-gated |
| bool | compare/mask logic | unsupported | available, focused validation open | unregistered |
| FP8/BF8, FP6, FP4, MX formats, NVFP4 | unsupported | unsupported | unsupported | not applicable or planned-gated negative |
| complex formats | no native complex datatype | unsupported | no native datatype | planned-gated |

The exact `gfx1151` target capability is `{fp32, fp16, bf16, int8, int4}`.
Canonical signed `int4` maps to Graph IR `i4`; its physical runtime ABI packs
two two's-complement nibbles per int8 byte, lower logical index in the low
nibble. The storage descriptor carries `signedness`, and the ROCm consumer
rejects contradictory unsigned metadata with
`DTYPE_PACK_SIGNEDNESS_MISMATCH`. Unsigned LLVM/ROCm instructions were probed
for u8/u16/u32/u64, but no unsigned Graph storage or runtime ABI was inferred.

FP64 and signed integer widths are closed by per-operation assessment rather
than blanket promotion: fp64 and int16 have no numeric tensor route; int32 is
ABI-only for indices, control results, and WMMA accumulation; int64 is ABI-only
for launch shape scalars. Both FP8 encodings are explicitly rejected on
gfx1151 by `ROCM_TILE_UNSUPPORTED_DTYPE` in selection, Target IR, and runtime.

Closure evidence on the gfx1151 host: LLVM 23 compile probes cover fp64,
signed i16/i32/i64, and unsigned u8/u16/u32/u64; ROCm backend lit is 53/53;
the focused registry/target/audit suite is 512/512; and four aligned/ragged packed-int4
launch comparisons pass exactly against NumPy int32 accumulation. Cross-backend
sync `ROCM-DTYPE1-CLOSE-2026-07-21` owns the additive canonical-int4 and storage
descriptor changes; no production selector changes.

## ROCM-TILE-1: portable Tile fragments

**Status: complete on `gfx1151` (2026-07-14).**

The portable dialect defines `!tile.fragment`, `#tile.mma_desc`, `tile.view`,
`tile.fragment_pack`, `tile.mma`, `tile.fragment_unpack`, and `tile.store`.
ROCm now consumes the typed form for gfx1151 Wave32 f16, bf16, signed int8, and
signed int4 WMMA. The same logical fixture owns only descriptors and layouts;
the backend owns the physical VGPR fragments, packing, accumulator map, and
stores.

The checked-in proof is `gfx1151_tile_fragment_store.mlir` plus
`test_rocm_wmma_gemm_generated.py`. It covers parser/verifier materialization,
real ROCDL WMMA intrinsics, gfx1151 hsaco assembly, HIP module launch, and
exact-device comparison with aligned and ragged launch-level shapes. Negative
fixtures reject a contradictory B storage order and FP8/BF8 with named
diagnostics. The portable launch contract also reuses the production multi-tile
K-loop, fused bias/ReLU/GELU/SiLU, output conversion, and ragged-store generator
while preserving the portable operand ABI. The portable adapter feeds that
generator through an in-memory request rather than a temporary target-IR
directive, and per-column bias is loaded once per fast/edge output tile and
reused across its accumulator elements.

### Build steps

1. Resolve the portable MMA descriptor through ROCm's existing
   `MmaDescriptor` selector.
2. Implement pointer/layout-bearing A and B `tile.fragment_pack` for the
   gfx1151 Wave32 `16x16x16` WMMA lane.
3. Lower the accumulator fragment through `tessera_rocm.wmma` to the matching
   ROCDL/LLVM intrinsic.
4. Implement accumulator `fragment_unpack` and masked row-major store.
5. Reuse the existing epilogue contract for bias, ReLU/GELU/SiLU, output
   conversion, and ragged stores.
6. Add launch-level multi-tile grid and K-loop fixtures using the same logical
   Tile program as the NVIDIA path.

### Dtype order

| Target family | First contracts | Explicit guard |
|---|---|---|
| gfx1151 RDNA 3.5 | f16, then bf16, int8, int4 | FP8/BF8 WMMA must fail with a named capability diagnostic; gfx1151 does not have those matrix forms. |
| gfx1200/gfx1201 RDNA 4 | f16/bf16, then E4M3/E5M2 and integer forms supported by WMMA v2 | No promotion without matching RDNA 4 silicon. |
| gfx942/gfx950 CDNA | f16/bf16 MFMA first; add target-supported FP8/FP6/FP4 only from the CDNA descriptor table | Never route an RDNA WMMA fragment map into MFMA. |

### Required proof

- positive parser/verifier and negative layout/descriptor fixtures;
- structural ROCDL/LLVM intrinsic check;
- object/hsaco assembly for the exact target;
- aligned and ragged execute-and-compare through the production HIP bridge;
- a named rejection for every unsupported dtype/architecture pairing.

## ROCM-5: architecture-owned fragment layouts

**Status: compiler and exact-target assembly complete; remote exact-device
closure remains open (2026-07-15).**

The portable Tile program no longer inherits gfx1151's physical register map.
`rocm_fragment.py` and `ROCMFragmentLayout.h` select a data-only physical
descriptor after the exact gfx architecture is known. The descriptor owns the
matrix family, Wave32/Wave64 width, per-lane elements and registers, gfx11 input
replication, accumulator map, intrinsic ABI, and materialization readiness.
Python/C++ name-consistency tests and named family/dtype/shape errors prevent a
prefix fallback from silently selecting the wrong ABI.

The same logical pack/MMA/unpack/store fixture now lowers as follows:

| Architecture family | Physical contract | Enabled forms | Cross-assembly resources |
|---|---|---|---|
| gfx1100/gfx1151 | duplicated gfx11 Wave32 WMMA, padded accumulator map | f16/bf16/int8/int4 | f16: 25 VGPR, 6 SGPR |
| gfx1200/gfx1201 | dense SOA Wave32 RDNA 4 WMMA | f16/bf16/E4M3/E5M2/int8, K32 int4 | 18–35 VGPR, 8 SGPR |
| gfx1250/gfx1251 | K32 Wave32 WMMA-v2 with typed `modC` and reuse properties (`signA`/`signB` are not properties of the LLVM 23 f16/bf16 ops) | f16/bf16 | 28 VGPR, 6 SGPR |
| gfx90a | Wave64 CDNA2 MFMA | f16/bf16 | 12 VGPR, 12 SGPR |
| gfx940/gfx942 | Wave64 CDNA3 MFMA | f16/bf16 | gfx942: 14 VGPR, 14 SGPR |
| gfx950 | Wave64 CDNA4 MFMA | f16/bf16 | 14 VGPR, 14 SGPR |

All serialized rows use zero LDS and scratch and report zero VGPR/SGPR spills.
LLVM 23 still cannot serialize `gfx940` in the installed Ubuntu package;
gfx942 supplies the same-family object proof. The
repeated-median compiler/serializer harness is
`benchmark_rocm_arch_fragments.py`; the stable resource baseline is
`rocm_arch_fragment_resources.json`.

Exact-device execution remains deliberately narrower than cross-assembly. The
available gfx1151 host passes f16, bf16, signed int8, and signed int4 numerical
oracles. No RDNA 4, gfx125x, or CDNA performance or numerical claim is promoted
without matching silicon. The remaining ROCM-5 completion work is therefore:

1. run the shared fixture on gfx1200/gfx1201 and compare every enabled dtype;
2. run f16/bf16 on at least one gfx942 and one gfx950 device;
3. record kernel-only latency and measured occupancy on each exact device;
4. enable gfx125x FP8 and additional CDNA low-precision forms only after their
   physical packing map and numerical oracle are proven on matching hardware.

## ROCM-9: paged-KV serving

**Status: complete on `gfx1151` (2026-07-14).**

The ABI is fixed and portable:

- K/V pages: physical `[P, L, H, D]`;
- page table: i32 logical-page to physical-page mapping;
- token indices: i64 gather order;
- attention must not assume identity or contiguous page placement.

The production selector now mirrors the CUDA reference design while preserving
the portable ABI: it verifies both the retained gather→FA path and a direct
page-table consumer against one oracle, records HIP-event and full-call timing,
and warm-starts from D2. The exact-device fixture combines a non-identity table,
arbitrary token order, page crossings, causal decode offset, and MQA/GQA head
mapping. It also exposed and fixed a baseline bug: dense FA's query-zero causal
triangle is not the right-aligned `T-Q` decode mask, so gather→FA now supplies
that offset explicitly as additive bias.

### Completed work

1. `test_live_rocm_paged_gather_handles_permuted_pages` executes on gfx1151.
2. The combined direct fixture covers non-identity placement, arbitrary order,
   page crossings, MQA grouping, and a multi-query causal offset.
3. Gather→dense-FA remains the named baseline and shares the same oracle.
4. Direct HIP attention consumes PLHD K/V, the i32 page table, and i64 order in
   its K/V traversal without materializing dense K/V.
5. Both routes are HIP-event timed and full-call timed at 128, 512, 2,048, and
   8,192 cached tokens.
6. Both timing modes are committed to D2; full-call winners warm-start serving.

### Measured decision

Shape is `Q=1, Hq=Hkv=4, D=32, L=16`, causal decode, f32 PLHD storage. Times are
milliseconds on the exact `gfx1151` host. The direct candidate wins the current
host-pointer ABI end to end at every measured length, so it is the production
selection for these buckets. Gather→FA wins device-only time decisively; if the
cache ABI becomes device-resident, that evidence requires re-evaluating the
selection rather than carrying the host-pointer verdict forward.

| Cached tokens | gather→FA device | direct device | device winner | gather→FA E2E | direct E2E | serving winner |
|---:|---:|---:|---|---:|---:|---|
| 128 | 0.0232 | 0.1157 | gather→FA | 4.158 | 1.158 | direct |
| 512 | 0.0671 | 0.4510 | gather→FA | 4.350 | 2.389 | direct |
| 2,048 | 0.1721 | 1.7967 | gather→FA | 6.614 | 5.741 | direct |
| 8,192 | 0.6444 | 7.2107 | gather→FA | 22.674 | 19.466 | direct |

Evidence starts in:

- `tests/unit/test_paged_kv_rocm_abi.py`;
- `tests/unit/test_paged_kv_rocm_native.py`;
- `python/tessera/compiler/emit/rocm_hip.py`;
- `python/tessera/cache/paged_kv.py`;
- `benchmarks/rocm/record_paged_kv_corpus.py`;
- `benchmarks/baselines/rocm_gfx1151_paged_kv.json`;
- `benchmarks/baselines/autotune_corpus.json`.

## ROCM-REPLAY-1: ReplaySSM serving parity

**Status: complete on `gfx1151` (2026-07-14).**

The reference state ABI, flush policy, speculative rollback, and CUDA serving
implementation already define the semantics. The ROCm work must preserve those
semantics rather than introduce a backend-specific cache layout.

### Architecture plan

The implementation is a handle-side serving runtime, not a new Graph-IR state
type. `SSMStateHandle` remains the semantic authority for shape validation,
flush policy, checkpoint/restore, cloning, and the host reference mirror. A
scalar-A-only ROCm context is attached by `rocm_ssm_replay_state_handle`; if the
HIP toolchain or exact device is unavailable, the factory retains the same
handle and honestly falls back to the reference path.

The ROCm context owns these allocations for its entire lifetime:

- resident checkpoint `S0[B,D,N]` and scalar decay `A[D]`;
- fixed-capacity replay inputs `delta[L,B,D]`, `x[L,B,D]`, and `b[L,B,N]`;
- one scratch `c[B,N]` and `y[B,D]` for synchronous decode;
- an ordered ring of at least two asynchronous slots, each with pinned host
  staging for `(delta,x,b,c,y)`, device output `[L,B,D]`, and begin/completion
  HIP events.

One nonblocking producer HIP stream owns all device mutation. Append, block
submit, output-only reconstruction, and flush are ordered on this stream. The
output-only kernel reads resident `S0` plus replay inputs and writes only
`y[B,D]`. The flush kernel is the sole writer of `S0`; after it completes, both
host and device cursors return to zero. General `(D,N)` A continues through the
reference handle because the scalar-A factorization is the ReplaySSM contract
implemented by this first serving slice.

Rollback never launches a kernel: the host cursor rewinds and future appends
overwrite rejected positions. A speculative block must fit wholly before the
reserved flush boundary; mid-block flush is rejected. Synchronous block submit
and asynchronous submit use identical validation and token ordering. The host
mirror advances only after a successful enqueue, so failed submissions cannot
create a split-brain cursor.

An async slot has three states: free, leased to a result, and retired pending a
consumer event. Submission copies into pinned staging, enqueues H2D + ordered
decode kernels, retains device outputs, and records completion. `wait()` performs
the explicit D2H handoff and frees the lease. Device consumers receive an opaque
HIP buffer and producer-stream handle; `event.wait_on(stream)` establishes
cross-stream order, and `release(stream=...)` retires the lease only after that
consumer. Reusing a slot before its completion event is forbidden and reports
backpressure rather than silently synchronizing.

Correctness gates are layered:

1. source/ABI and shape guards run without a GPU;
2. output-only, flush, long decode, reset, rollback, speculative rejection, and
   block submission compare against `SSMStateHandle` on gfx1151;
3. multi-slot ordering, backpressure, wait/download, device lease, and release
   lifetime run on the live HIP runtime;
4. device-event and wall-time benchmarks compare replay against eager summary
   traffic and commit exact-architecture evidence.

### Completed work

1. The HIP context owns persistent S0 and fixed-capacity replay inputs.
2. Scalar-A output-only decode writes only `[B,D]`; shared history Gram scalars
   are computed once per `(token,batch)` rather than once per channel.
3. Only the flush kernel materializes and writes the full state.
4. Ordered block submission covers prefill and speculative verification.
5. The multi-slot ring uses pinned staging, a nonblocking producer stream,
   begin/completion events, backpressure, and opaque output leases.
6. Device consumers can wait on the producer event and retire a lease on their
   own HIP stream without a host download.
7. Summary, sequential output-only, and four-slot async modes have committed
   device/wall timing and analytical traffic rows.

### Required proof

- long decode across multiple flushes versus `SSMStateHandle`;
- rollback and speculative rejection equivalence;
- output-only and state-and-output route equivalence;
- ring ordering, backpressure, event waits, and device-buffer lifetime tests;
- representative serving rows committed to D2 with exact gfx1151 provenance.

All required proof cases execute in `test_ssm_rocm_replay.py`: 43-token decode
across repeated flushes, rollback and reset, speculative suffix rejection,
ordered block submission, two-slot backpressure, event timing, host wait, HIP
device-buffer exposure, cross-stream wait, and lease retirement. The broader
SSM/ReplaySSM regression sweep is 95 passed, 4 skipped.

### Measured decision

The table reports milliseconds per token for 64-token scalar-A decode on exact
gfx1151 (five repetitions). `summary` reconstructs output and writes resident
S0 every token. `output-only` is true sequential decode. `async` submits four
16-token blocks through the ordered slot ring.

| Shape `(B,D,N)` | Mode | Device ms/token | Wall ms/token | Speedup vs summary wall | State-traffic reduction |
|---|---|---:|---:|---:|---:|
| `1,64,64` | summary | 0.0770 | 0.1822 | 1.00× | 1.0× |
| `1,64,64` | output-only | 0.0400 | 0.1583 | 1.15× | 32.1× |
| `1,64,64` | async, chunk 16 | 0.0295 | 0.0337 | 5.41× | 32.1× |
| `1,128,128` | summary | 0.0769 | 0.1871 | 1.00× | 1.0× |
| `1,128,128` | output-only | 0.0410 | 0.1462 | 1.28× | 51.5× |
| `1,128,128` | async, chunk 16 | 0.0358 | 0.0403 | 4.64× | 51.5× |

#### Wider compiler matrix

The follow-up compiler matrix expands this to five geometries
(`1x32x16`, `1x64x64`, `1x128x64`, `1x128x128`, and batched `4x64x64`), token
lengths 16/64/256, capacities 16/64, and async schedules `(chunk=4,slots=2)`
and `(chunk=16,slots=4)`. It contains 75 exact-device rows, including forced
flush cases. Every row is checked against `SSMStateHandle`; maximum absolute
error is `5.07e-8`, and no ReplaySSM row loses to its matching summary row.

| Mode | Wall speedup min / median / max | Device speedup min / median / max |
|---|---|---|
| sequential output-only | 1.07× / 1.24× / 1.42× | 1.86× / 2.26× / 3.02× |
| async chunk 4, two slots | 3.11× / 3.69× / 5.06× | 1.96× / 2.58× / 3.66× |
| async chunk 16, four slots | 4.11× / 5.04× / 6.31× | 2.12× / 2.81× / 4.21× |

At `T=256, capacity=64` (three real flushes), chunk-16 throughput ranges from
22,998 tokens/s for `1x128x128` to 31,021 tokens/s for `1x32x16`. The narrowest
sequential win is 1.07× (`4x64x64`, `T=64`, `capacity=16`), so it is the first
candidate for a future performance ratchet rather than being hidden by the
matrix aggregate.

Evidence is in:

- `python/tessera/compiler/emit/rocm_hip.py`;
- `python/tessera/runtime.py`;
- `tests/unit/test_ssm_rocm_replay.py`;
- `tests/unit/test_ssm_rocm_replay_benchmark.py`;
- `benchmarks/rocm/benchmark_ssm_replay.py`;
- `benchmarks/baselines/rocm_gfx1151_ssm_replay.json`;
- `benchmarks/baselines/rocm_gfx1151_ssm_replay_matrix.json`.

## ROCM-6: performance redesign experiments

**Status: open revalidation under LLVM/MLIR 23 + ROCm 7.14; performance
blocked by invalid WSL HIP event timing (2026-07-16).**

The refreshed build passes the full required correctness matrices: G6-A passes
20/20 schedule rows over its four aligned/ragged/dtype cases; G6-B passes all
four cases with maximum difference `8.36e-6` versus one wave; G6-C passes all
six cases with maximum difference `3.13e-7` versus serial dK/dV. G6-B retains
its D=128 resource advantage: 121 VGPR with zero scratch/spills versus 218 VGPR
for one wave.

All three paired performance harnesses are blocked because ROCm 7.14 WSL HIP
event calls return success but report `0.0 ms`. The harnesses now reject
zero/non-finite samples and expose correctness-only mode. Until valid paired
device and E2E timing is collected, existing production choices stay in force
without claiming the old performance decisions were reaffirmed:

- G6-A remains non-production and is reopened for measurement;
- G6-B remains the current production route, with correctness/resources
  reaffirmed but performance revalidation open;
- G6-C remains non-production, with correctness reaffirmed but its prior
  performance rejection awaiting revalidation.

Evidence is recorded in
`benchmarks/baselines/rocm_gfx1151_rocm6_llvm23_rocm714_revalidation.json`.

### Phase 0: rebaseline older kernels with the current compiler

The 2026-07-14 exact-device survey reran the older compiler-generated GEMM and
flash-attention ladders with ROCm 7.2 and LLVM 23. This changes the premise of
G6-A but not G6-B or G6-C.

Generated f16 GEMM now reaches 12.23, 12.99, 23.51, and 26.29 TFLOP/s at
512³, 1024³, 2048³, and 4096³. The best tiles are respectively 2x4, 4x4, 2x4,
and 4x4. In particular, 4x4 is no longer a universal compiler-path VGPR cliff:
it wins at 1024³ and 4096³. At 4096³ the same 4x4 schedule reaches 25.90
TFLOP/s bf16, 29.53 TOP/s int8, and 31.32 TOP/s int4. G6-A must therefore begin
by rebuilding the size/dtype schedule ratchet with repeated medians and compiler
resource evidence; the two-wave/LDS reduction candidate is implemented only if
that renewed evidence still exposes an occupancy gap.

The older pipeline route is still specialized. It is 2.53x faster at 512³ but
only 1.002x, 1.007x, and 1.006x the direct route at 1024³, 2048³, and 4096³.
It must not become a general default.

Attention did not receive the same automatic compiler uplift. Forward remains
6.57--6.89 TFLOP/s at D=64 and 3.04 TFLOP/s at D=128; backward remains
3.92--4.15 TFLOP/s at D=64 and 2.13 TFLOP/s at D=128. Those numbers reproduce
the previous ceiling closely enough that the two-wave D=128 forward experiment
and split/reduced dK/dV backward remain the first kernel redesigns.

The survey is committed in
`benchmarks/baselines/rocm_gfx1151_legacy_compiler_rebaseline.json`. It is
exploratory evidence, not yet a promotion ratchet: the next measurement pass
must use repeated medians, preserve numerical oracles, and record code-object
VGPR/LDS occupancy before changing production selection.

That promotion pass is now complete. The 37-case, 185-row matrix uses nine
interleaved trials per tile so APU clock movement is paired rather than mistaken
for a schedule win. It covers square sizes 512 through 4096, transition sizes
1536/3072, model-shaped rectangular rows through `4096x11008x4096`, three
ragged rungs, f16/bf16/int8/int4, and bias/ReLU/GELU/SiLU epilogues. Every row
matches its numerical oracle and every tile is bitwise equal to the common
device result.

Assembler metadata explains why selection remains shape-dependent. Plain f16
1x1 and 2x2 use 51 and 136 VGPRs with no spills. The 2x4, 3x4, and 4x4 kernels
reach 256 VGPRs and report respectively 41, 257, and 392 spills with 108, 424,
and 736 bytes of scratch per work-item. Large aligned and wide shapes can
amortize that cost; small and ragged shapes cannot.

Production now uses the rows that clear both gates (at least 3% paired-median
gain and a win in at least 75% of interleaved rounds). Three near-ties retain
the previous selector: 3072-cube 4x4 over 3x4 (2.6%), the required large ragged
2x4 over 3x4 (2.1% after a 21-trial tie-breaker), and skinny-M=128 2x2 over
2x4 (2.8%). Evidence is committed in
`benchmarks/baselines/rocm_gfx1151_gemm_schedule_matrix.json`.

### G6-A: VGPR-bounded multi-wave GEMM

**Status: reopened for performance revalidation (2026-07-16); non-production
until the original gate passes.**

The ROCm 7.14 production correctness and performance ratchets pass, and the
existing repeated-median schedule baseline remains shape-dependent. A renewed
G6-A matrix was attempted at both required aligned f16 sizes plus the ragged
and int8 rungs. Under WSL, ROCm 7.14's HIP event API returned success but a
zero elapsed time for module-launch batches; the harness now rejects zero,
non-finite, and failed timing samples rather than emitting fabricated
throughput. Because the renewed measurement is invalid rather than negative,
it cannot reject or promote the two-wave/LDS-reduction design. Valid repeated
device timing must show whether the existing selector misses the stated 10%
opportunity before implementation proceeds.

- Split an output macro-tile across two Wave32 groups.
- Reduce bounded partial f32 accumulators through LDS.
- Keep per-wave accumulator pressure below the measured 4x4 VGPR cliff.
- Measure f16/bf16 at 2048³ and 4096³, ragged
  `2049x4093x2051`, and int8 at 2048³.
- Promote only with at least 10% median gain on both aligned f16 rungs, no rung
  slower by more than 3%, and all dtype/ragged oracles green.

### G6-B: two-wave online-softmax forward attention

**Status: current production route from ROCm 7.2 evidence; LLVM/MLIR 23 + ROCm
7.14 correctness/resources reaffirmed, performance revalidation open.**

- Give two waves one query tile and share K/V traversal.
- Merge per-wave online `(m,l,O)` state once per K/V tile.
- Measure `(1,8,512,64)`, `(1,8,1024,64)`, `(1,16,1024,128)`, and causal
  sequence 1009 at D=128.
- Promote only with at least 10% gain on both D=128 rungs and no D=64
  regression beyond 3%.

Two Wave32 groups now own one query tile at D=128. They split the QK head-
dimension chunks, reduce a bounded 2x16x16 partial score tile through 2 KiB of
additional LDS, share the online-softmax state, and split the PV output chunks
without a second reduction. The assembler result moves from 256 to 121 VGPRs,
removes 82 VGPR spills and 332 scratch bytes, and raises modeled occupancy from
6 to 12 waves/SIMD. Nine interleaved trials measure 2.045x at noncausal
`(1,16,1024,128)` and 2.106x at causal sequence 1009, with 100% win rates and
maximum differences of 5.6e-7 and 8.4e-6. D=64 and advanced GQA/window/bias/
soft-cap variants retain the one-wave kernel pending their own matrix.

### G6-C: split/reduced dK/dV backward attention

**Status: implemented and non-production; LLVM/MLIR 23 + ROCm 7.14
correctness reaffirmed, performance rejection revalidation open.**

- Separate dQ and dK/dV wave ownership.
- Reduce bounded partial dK/dV tiles in a second generated kernel.
- Measure `(1,8,512,64)` and `(1,16,1024,128)`, causal/noncausal and GQA.
- Promote only with at least 15% at D=128 and 10% at D=64; temporary storage
  must remain below one extra K+V gradient footprint.

The opt-in compiler candidate partitions query tiles across two dK/dV blocks,
writes the second split into exactly one extra dK+dV footprint, and launches a
generated reduction kernel. It matches the serial kernel across MHA, causal,
ragged, and GQA with maximum absolute error 3.13e-7. It does not clear the
performance gate: nine-trial device speedups span 0.908--1.013x and end-to-end
speedups span 0.904--1.070x, with neither required D=64 nor D=128 rung meeting
the gain threshold. The existing key-tile/head grid already exposes enough
parallelism, so the extra global partial traffic is not amortized. Production
therefore remains on serial-per-key-tile dK/dV; the candidate, correctness tests,
and rejection benchmark stay available for later architectures.

### Older-kernel retune closeout

The scalar f32 GEMM compiler now accepts compile-time output tiles. Production
uses 2x2 only for square sizes through 256 (0.0483 ms versus 0.0508 ms for the
old 4x4 at 256 cube) and conservatively retains 4x4 elsewhere where device and
end-to-end winners disagree. Grouped GEMM uses `tn=1` below 64k output elements,
`tn=2` at 64k, and `tn=4` from 131k, with exact-divisibility fallback; promoted
model rows are 1.92--3.42x faster in the resident kernel.

Grouped SwiGLU now collapses `3E` per-expert GEMM launches into three grouped
launches. E8 model rows improve resident GEMM time by 4.64--7.64x and end-to-end
time by 3.58--4.72x with 100% paired win rates. By contrast, compiler-generated
row-gather and weighted-scatter candidates for KV/MoE transport did not clear
both resident and end-to-end gates and remain non-production experiments.

The consolidated evidence is
`benchmarks/baselines/rocm_gfx1151_compiler_retune_2026_07_15.json`; the
reproducers are the `benchmark_rocm_{f32,grouped_gemm,swiglu,transport}_retune.py`
and `benchmark_rocm_g6{b_two_wave,c_split_reduced}.py` scripts.

All winners update
`benchmarks/baselines/rocm_gfx1151_hot_paths.json`; native counter collection
uses `benchmarks/rocm/collect_rocm6_counters.py` only on bare metal.

## Exact-device expansion queue

These retain the release priorities from `ROCM_AUDIT.md`, but execution is
hardware-gated. Compiler-only work may proceed locally; promotion waits for the
named device.

| ID | Priority | State | Target | Required first proof |
|---|---|---|---|---|
| ROCM-2 | P0 | open, access-gated | gfx1201, Radeon AI PRO R9700 | RDNA 4 matmul assembles, launches, and matches; establish WMMA-v2 fragment layout before adding FP8. |
| ROCM-1 | P0 | open, access-gated | gfx950, MI350 series | Compile, launch, and numerical proof for matmul, flash attention, softmax, and GELU; then CDNA 4 FP8/FP6/FP4 breadth. |
| ROCM-3 | P0 | open, access-gated | gfx1250, MI455X | Join the upstream-LLVM artifact to an exact-device matmul launch and numerical fixture. |
| ROCM-4b | P1 | open, access-gated | gfx942, MI300X/MI325X | Retain explicit compatibility proof for matmul, flash attention, softmax, and GELU. |
| ROCM-4a | P1 | open, access-gated | gfx1200, Radeon RX 9000 | Exact-device matmul proof plus stable rejection of unsupported feature forms. |
| ROCM-5 | P1 | landing; depends on rows above | all above | Close RDNA 4 WMMA-v2, gfx125x WMMA-v2, and CDNA MFMA descriptors with exact-device layouts, dtype guards, resources, and numerical proof. |

### P0 exact-device access coordination

The three P0 queues require externally scheduled hardware; no P0 device is
reachable from the gfx1151 WSL host. The access handoff is ready with one
common synchronization key, `ROCM-P0-LLVM23-2026-07`, and must retain the
configure cache, compiler versions, device identity, JUnit, emitted object,
and numerical outputs for each run.

| Queue | Required host | Access state | First scheduled command/result |
|---|---|---|---|
| ROCM-1 | MI350-series `gfx950` | owner and reservation required | LLVM/MLIR 23 clean build; matmul, flash attention, softmax, and GELU compile/launch/oracle packet |
| ROCM-2 | Radeon AI PRO R9700 `gfx1201` | owner and reservation required | LLVM/MLIR 23 clean build; WMMA-v2 fragment layout plus aligned/ragged matmul packet |
| ROCM-3 | MI455X `gfx1250` | owner and reservation required | LLVM/MLIR 23 clean build; upstream artifact joined to launch and numerical packet |

Access coordination is not complete until a named operator and reservation are
recorded for each host. Compiler-only artifacts may be prepared locally, but
no queue status advances from that evidence.

The LLVM 23 compiler-only handoff is prepared with
`benchmark_rocm_arch_fragments.py --artifact-directory`. The 2026-07-16 packet
contains input MLIR, target-lowered ROCDL, embedded code-object MLIR, resource
metadata, and median/MAD compiler timings for gfx1201, gfx950, gfx1250, gfx942,
and gfx1200. Every requested row assembled with a real target intrinsic, zero
scratch, and zero VGPR/SGPR spills. Recreate the transferable packet from the
clean build with:

```bash
TESSERA_OPT="$PWD/build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt" \
MLIR_OPT=/usr/lib/llvm-23/bin/mlir-opt \
.venv/bin/python benchmarks/rocm/benchmark_rocm_arch_fragments.py \
  --repetitions 3 --arch gfx1201 --arch gfx950 --arch gfx1250 \
  --arch gfx942 --arch gfx1200 \
  --artifact-directory /tmp/tessera-rocm714-remote-packets \
  --output /tmp/tessera-rocm714-remote-packets.json
```

These packets are compiler-only. Remote operators must append device identity,
module load/launch, aligned and ragged numerical output, device and end-to-end
timing, and measured occupancy before any exact-target row advances.
The local bundle contains 40 files at
`/tmp/tessera-rocm714-remote-packets.tar.gz` with SHA-256
`3d569f1de9c837fefef5a84c435c5508f2f8d1c691c38620e59dd6a6a015ee4e`.

## ROCM-8: bare-metal copy versus zero-copy

**Status: blocked on access to a bare-metal gfx1151 host (2026-07-16). WSL
results are characterization only and cannot close ROCM-8.**

WSL measurements show an environment-specific crossover, but Windows driver
round trips affect registration and allocation. Before automatic selection:

1. collect copy and mapped-host measurements on bare-metal gfx1151;
2. report both kernel-only and end-to-end latency;
3. cover at least 256³ through 2,048³ GEMM plus representative serving buffers;
4. repeat enough samples to establish a stable crossover;
5. keep `TESSERA_ROCM_ZEROCOPY=1` opt-in unless the bare-metal evidence is
   reproducible and guarded by a ratchet.

## Accepted-deferred work

Consumer plan `SEQUENCE-MIXER-2026-07-17`: the compiler-direction Sequence Mixer
track ([`../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md`](../../compiler/SEQUENCE_MIXER_ENGINEERING_PLAN.md))
consumes ROCm as a **lead performance target** (Decision #28 — its WMMA/MFMA
candidates set the ceiling, never capped by the shared mixer framework). It
**extends already-complete vehicles** rather than opening new items: channel-wise
KDA/GDN decode → **ROCM-REPLAY-1** (add the channel-diagonal transition to the
proven persistent/flush/rollback/async-ring path); `windowed_kv` mixer state →
**ROCM-9** (window ring on the proven direct + gather paged-KV routes);
chunkwise-scan inner GEMMs → **ROCM-TILE-1** WMMA f16/bf16/int8/int4 fragments;
`sliding_window`/full mixer forward → **ROCM-6 G6-B**; mixer backward → **ROCM-6
G6-C** (split/reduced dK/dV). Low-precision mixer GEMMs stay bf16/f16 on gfx1151
per the standing **FP8/FP4 WMMA guard**; FP8/FP4 forms are the access-gated
CDNA4/RDNA4 packet (ROCM-1/2/3). Valid paired mixer device timing needs bare-metal
gfx1151 (ROCM-8). Inherits the exact-device native-provenance + aligned/ragged +
RDNA↛MFMA-guard contract unchanged. Direction pointer only; no ROCm gate or
exact-device claim changes here.

Cross-backend sync `EPILOGUE-CONTRACT-2026-07-16` updates only the shared
`FusedRegion` bias/activation/residual order and registered rejection
diagnostics. Existing gfx1151 epilogue fixtures already consume this oracle;
NVIDIA now validates the complete 43-case CUDA execution matrix. No CUDA warp,
register, dtype-support, or schedule result transfers to ROCm. Re-run the shared
contract and architecture-supported execution matrix on the gfx1151 host when
this coordinating change lands; retain explicit not-applicable results for
CUDA-only storage forms.

Cross-backend sync `NVIDIA-SM120-LOWP-2026-07-18`: not applicable to gfx1151
execution. The CUDA work changes no shared dtype/ScaleLayout, epilogue, or
autotune schema. gfx1151 supports neither FP8/BF8 WMMA nor NVIDIA NVFP4 OMMA;
ROCm therefore inherits no CUDA nibble packing, scale selector, wave schedule,
resource value, timing, or selector promotion. RDNA4/CDNA4 low-precision work
remains behind its architecture-specific exact-device queues.

Cross-backend sync `E2E-SPINE-2026-07-18`: ROCm owns **ROCM-E2E-1** and
**ROCM-E2E-2**. The shared contract standardizes native-image metadata and
launch descriptors only; ROCm continues to own directives, generators, AMDGPU
ISA selection, wave/LDS schedules, HSACO production, and selectors. The softmax
pilot must match the existing gfx1151 route before its runtime-authored text
bridge is removed. NVIDIA physical schedules do not transfer, and later
architecture breadth still requires the exact-device queues above. The
completed E2E-SPINE-0 foundation gives every registered ROCm family/exact target
a total family-shared pipeline row, including gfx1250, while preserving the
exact-target artifact fallback and generic runtime route. E2E-SPINE-1 adds the
portable HSACO image/descriptor envelope and registered validation diagnostics
without encoding AMD wave/LDS schedules or changing any HIP route. ROCM-E2E-1
remains responsible for the first typed softmax producer and exact gfx1151 join.
E2E-SPINE-2 completes the shared typed carriers, stage ledger, cache join, and
descriptor-first exact-target launcher registry. It registers no HIP hook and
does not replace runtime-authored directives or existing gfx1151 executors;
ROCM-E2E-1 still owns HSACO production, `gfx1151` registration/submission,
softmax comparison, cleanup, and the first ROCm Level-C row.
The NVIDIA-E2E-1 f16 landing slice was assessed as NVIDIA-only: it adds an
SM120 PTX package producer and exact CUDA submission hook, with no HIP hook,
ROCm directive/ABI, wave/LDS schedule, dtype registration, or selector change.
The completed NVIDIA-E2E-1 NVFP4 slice extends the shared `tile.matmul_kernel`
verifier with an explicit packed-A/packed-B/scale-A/scale-B/output/M/N/K form.
This is not applicable to the gfx1151 WMMA lowering because that ISA has no
NVFP4 block-scaled matrix instruction. ROCm inherits only shared verifier
rejection behavior; it does not inherit CUDA scale-word packing, warp geometry,
resources, timings, ABI registration, or selector evidence. RDNA4/CDNA4 work
remains in its architecture-owned exact-device queue.

The first NVIDIA-E2E-2 slice changes the shared Graph→Tile async contract so a
copy produces `!tile.async_token`, its wait retires that token, and a matrix
consumer carries the dependency. This is parity validated with ROCm's existing
token/retirement legality model and adds no ROCm directive, AMDGPU instruction,
wave/LDS schedule, HSACO ABI, selector, or execution claim. The additive
pipeline-registry driver-source field and `tessera_nvidia` dialect manifest row
are NVIDIA bookkeeping; ROCm pipeline ownership and runtime routing are
unchanged. Exact SM builders and CUDA TMA/WGMMA behavior are not applicable to
ROCm.

**Superseded/closed (2026-07-26):** the NVIDIA-E2E-2 softmax slice added the shared semantic
`tile.softmax_kernel(X, O, Rows, K)` envelope with explicit storage,
accumulation, and last-axis fields; the envelope now accepts f16/f32 storage
with f32 accumulation. ROCM-E2E-1 has since landed the ROCm carrier consumer
and exact gfx1151 softmax slice; this historical assessment request is closed
and must not be reopened as active work. ROCm does not inherit the SM120
thread-per-row schedule, `nvvm.ex2`, PTX ABI, resources, timings, or selector.
The cooperative HIP softmax execution path remains architecture-owned.

The NVIDIA-E2E-2 dtype-totality slice changes the shared MMA selector contract:
fp32 Tensor Core selection now requires an explicit TF32 math mode, and bare
`fp4_e2m1` no longer aliases NVIDIA NVFP4. This semantic separation is parity
validated for ROCm: AMD xf32 continues to require its architecture-owned math
mode, and RDNA/CDNA FP4/MX scale formats remain distinct from UE4M3-scaled
NVFP4. The new SM120 scalar/vector and fragment table transfers no AMD ISA,
wave/VGPR layout, scale encoding, HSACO ABI, runtime readiness, or selector.

The follow-on SM120 dtype slice adds a backend-private
`tessera_nvidia.mx_block_scale_mma` Target IR op and ptxas-backed FP6/MXFP4
register contracts. This is not applicable to ROCm code generation and changes
no AMD dtype registry, MFMA/WMMA descriptor, scale encoding, HSACO ABI, runtime
route, or selector state. Exact CDNA/RDNA low-precision evidence remains owned
by the corresponding ROCm items.

**Superseded/closed (2026-07-26):** the NVIDIA-E2E-2 reduction slice added the shared semantic
`tile.reduce_kernel(X, O, Outer, AxisExtent, Inner)` envelope with explicit
kind, storage, accumulation, normalized axis/keepdims, schedule, and NaN
policy. ROCM-E2E-2 has since landed the ROCm carrier consumer and exact gfx1151
reduction slice; this historical assessment request is closed and must not be
reopened as active work. ROCm inherits neither the CUDA serial nor
cooperative-128 schedule, PTX ABI, resource/timing evidence, nor selector
change.

The NVIDIA-E2E-2 epilogue slice tightens the shared `tile.matmul_kernel`
verifier around optional residual operands and the portable
matmul/bias/activation/residual order. ROCm semantic parity is preserved, but
its materializer must opt into that launch form explicitly; no CUDA buffer
layout, warp schedule, PTX ABI, resources, timing, or readiness transfers.

The NVIDIA-E2E-2 attention slice adds the shared semantic
`tile.attention_kernel(Q,K,V,O,B,Hq,Hkv,Sq,Sk,D,Dv)` carrier with explicit
storage, f32 accumulation/output, scale, and causal policy. ROCM-E2E-2 must
assess adapting it to the existing HIP/MFMA forward routes; this is follow-up
required, not parity inferred. ROCm inherits no CUDA thread-per-output
schedule, PTX ABI, resource/timing evidence, readiness, or selector change.

The NVIDIA paged-KV slice adds a shared logical-page gather carrier with f32
pages, i32 table, explicit dimensions/range, and a direct-route semantic tag.
ROCM-E2E-2 must map it to the existing HIP paged movement lane under its own
ISA and exact-device proof; no PTX ABI, CUDA schedule, timing, or selector
state transfers.

The NVIDIA backward-attention slice adds a shared
`tile.attention_backward_kernel` semantic carrier with explicit mask, softcap,
determinism, route, and workspace fields. ROCM-E2E-2 must assess mapping it to
the existing compiler-generated HIP/WMMA backward sequence. The CUDA
single-owner schedule, zero-workspace reference ABI, atomic/split resources,
timings, and selector evidence do not transfer to gfx1151.

Cross-backend sync `E2E-DEVICE-LIBS-2026-07-19` extends the shared native-image
contract with content-addressed LLVM-stage device-library provenance. ROCm must
populate it from the matching clang/ROCm-driver-selected OCML, OCKL, and OCLC
set under `--rocm-path`; it must not copy NVIDIA's explicit single-libdevice
link rule or hand-assemble an OCLC set independently of architecture, wavefront,
and math-mode flags. ROCM-E2E-1 now closes that follow-up for the gfx1151
softmax pilot: AMD clang selects the seven records, their content digests enter
cache/toolchain identity, and absolute paths stay out of the artifact. The
existing LLVM 23/TheRock 7.14 compatibility rule remains mandatory, and no gfx
execution or selector state changes in the NVIDIA-owned landing.

Cross-backend sync `CUDA-MATH-CONTRACT-2026-07-19` makes the shared Tile softmax
envelope state its exponential mode and FTZ behavior instead of deriving either
from compiler optimization flags. ROCm must map that semantic choice to an
architecture-owned OCML/intrinsic route and validate its own accuracy and
denormal behavior under ROCM-E2E-1; PTX's `ex2.approx.f32` bound is not AMD
evidence and no gfx selector changes here.

Cross-backend sync `CUDA-INTRINSIC-SURFACE-2026-07-19` extends the shared
rounding vocabulary with toward-positive and toward-negative while preserving
the existing default RTNE/RTNA/RTZ tuning sweep. This is shared semantic parity;
ROCm must map directed conversions to AMDGPU/OCML behavior under its own typed
lowering and exact-device proof. CUDA integer, DP2A/DP4A, cast-suffix, and packed
SIMD symbols transfer no AMD instruction, wave layout, runtime route, or
selector. The NVIDIA inventory marks all Tessera Target-IR/runtime rows planned.

Cross-backend sync `PTX-TYPE-MEMORY-TRUTH-2026-07-19` adds physical PTX register
and format-kind fields to the NVIDIA dtype contract and a backend-private PTX
memory-model guard. This is not AMD ISA evidence. ROCm must continue deriving
VGPR packing, alternate formats, scopes, atomics, and cache/coherence behavior
from the applicable RDNA/CDNA ISA and ROCm device-library contract. The only
shared outcome is the architectural rule that language dtype availability does
not imply a native register type or executable matrix route.

Cross-backend sync `NVIDIA-E2E-DTYPE-EXEC-2026-07-19` extends the shared Tile
epilogue output vocabulary with f64 for NVIDIA's architecture-owned m8n8k4
DMMA route. ROCm records parity at the shared semantic layer only: AMD
MFMA/WMMA lane maps, packed formats, code-object ABI, timings, and selectors do
not inherit CUDA evidence. Existing ROCm f64 states still require exact-gfx
proof.

Cross-backend sync `X86-E2E1-NATIVE-CPU-2026-07-19` classifies shared native
descriptor results for `cpu`, `x86`, `x86_amx`, and `x86_avx512` as
`native_cpu` with CPU-wall timing rather than GPU-event timing. ROCm artifact,
ABI, stream, event, HSACO, device-library, exact-gfx, and selector contracts are
unchanged. The x86 pilot consumes existing Tile softmax/reduction carriers and
adds no shared dtype, operation, or verifier state that ROCm must implement.

Cross-backend sync `X86-E2E1-BREADTH-2026-07-19` consumes the already-shared
matmul and attention launch carriers for x86 f32 GEMM and MHA descriptors.
ROCm inherits no AVX-512 ABI, host schedule, timing, readiness, or selector
state; its WMMA/MFMA and attention routes remain architecture-owned. The x86
restriction to equal query/KV head counts and zero dropout is not a shared
semantic restriction and changes no ROCm verifier or capability row.

Do not schedule these without new evidence:

- flash-attention K/V double buffering, until a measured workload or target is
  staging-bound rather than occupancy/LDS-bound;
- packed-memory int4 GEMM on gfx1151 as a compute optimization—its value there
  is footprint/bandwidth, not higher matrix issue rate;
- automatic zero-copy selection before ROCM-8 bare-metal measurements;
- FP8/BF8 WMMA on gfx1151, which is an unsupported instruction claim rather
  than an optimization opportunity.

## Validation and update checklist

Run tests on the host ROCm environment with the intended GPU and toolchain
visible. A sandbox fallback cannot count as device proof.

- Tile/WMMA/MFMA:
  `test_rocm_target_wmma_lowering.py`, `test_rocm_wmma_gemm_via_mlir.py`, and the
  new portable fragment fixtures.
- Paged serving: `test_paged_kv_rocm_abi.py` and
  `test_paged_kv_rocm_native.py`.
- GEMM/attention ratchets: `test_rocm_perf_ratchet.py` and the corresponding
  benchmark recorder.
- Exact-target dashboards:
  `python -m tessera.cli.gpu_target_map --target=rocm --render` and
  `python -m tessera.cli.conformance_matrix --render`.
- Static gates: mypy ratchet, Ruff, and `git diff --check`.
- After changes: `graphify update .`.

## Definition of ROCm roadmap closure

The ROCm roadmap is not closed merely because gfx1151 has broad operator
coverage. Closure requires:

- portable Tile fragments executing on at least one RDNA WMMA and one CDNA MFMA
  target;
- stable paged-KV serving with exact-device proof and measured route selection;
- ReplaySSM persistent/asynchronous serving parity;
- ROCM-6 candidates revalidated under the current toolchain and either promoted
  through their ratchets or explicitly retained as measured non-winners;
- exact-device evidence for the priority RDNA 4/CDNA targets, without inherited
  gfx1151 proof;
- generated target, runtime, and conformance dashboards agreeing with the
  checked-in evidence.

Cross-backend sync `E2E-SPINE-2026-07-18` records the 2026-07-20 scoped x86
selector retirement: eligible static X86-E2E-1 modules now use their canonical
descriptor by default. ROCm parity is not applicable; no ROCm pipeline, HSACO
ABI, schedule, dtype capability, or selector changes. X86-E2E-2 subsequently
closed the remaining inventory and reassessed ROCm at each shared-contract
boundary.

E2E-SPINE-3 is complete on the assigned Strix Halo host under synchronization
key `ROCM-E2E-SPINE3-TEST1-2026-07-21`. The hash-sealed packet at
`docs/audit/evidence/e2e_spine/rocm_gfx1151/gfx1151` packages shared f32
softmax, reduction, non-identity paged-KV, and permuted MoE fixtures against
source commit `a5243d14a4039a3fa6ddcdb0276eb803ff567d19`. All four fixtures have
exact-target Level-C provenance and zero maximum absolute error; four
cold/warm proofs retain identical image, payload, descriptor, and toolchain
identity.

The packet contains eight selected benchmark rows: kernel-wall and
allocation/copy-inclusive end-to-end timing for each family. Each row uses two
interleaved 101-sample cohorts, passes the 5% stability limit, and records its
resource fingerprint. Kernel-wall timing batches resident HIP launches and a
single synchronization because WSL HIP event intervals are invalid on this
host; the timing domain is named honestly and is not presented as a device
event. The packet is release-evidence selection only. ROCM-E2E-1/-2 retained
production routes remain selected and `production_selector_changed` is false.
Other gfx targets remain explicit unavailable/deferred rows and cannot inherit
gfx1151 evidence.

Fleet packet identity remains `(target, architecture)`. The assigned Strix Halo
host now satisfies the `rocm_gfx1151/gfx1151` obligation independently of the
NR2 WSL host's `x86_64_base` and `sm_120a` evidence. The four gfx1151 dashboard
rows are therefore `release_ready`; no sibling packet or architecture inherits
that proof.

Cross-backend sync `X86-E2E2-ELEMENTWISE-2026-07-20` adds the internal shared
`tile.elementwise_kernel` semantic carrier for f32 unary/binary and f32-to-bool
predicate requests. ROCm parity is assessed at the carrier boundary only;
AVX-512 ABIs, host schedules, CPU-wall evidence, and the 16K binary selector
threshold transfer no RDNA instruction, HSACO ABI, exact-gfx evidence, runtime
readiness, or selector claim. Existing ROCm elementwise routes remain
architecture-owned and unchanged.

Cross-backend sync `X86-E2E2-TYPED-LOGIC-2026-07-20` widens that internal
carrier with compare, logical, and bitwise semantics plus explicit f32/i8/i32
physical storage. The capability repair is x86-owned bool/int32 truth for
already-shipped AVX-512 ABIs. ROCm inherits no x86 C ABI, null-operand
convention, 32K selector threshold, CPU timing, RDNA instruction, HSACO route,
or selector claim; ROCm target and execution rows remain unchanged.

Cross-backend sync `X86-E2E2-FLAT-FOLLOWON-2026-07-20` extends the shared
elementwise carrier with where, transcendental, and binary-math semantics.
ROCm parity is assessed at the carrier boundary only; AVX-512 approximations,
C ABIs, CPU-wall thresholds, exact-host evidence, RDNA instructions, HSACO
routes, and ROCm selectors do not transfer.

Cross-backend sync `X86-E2E2-DTYPE-2026-07-20` adds an x86-only datatype/CPUID
contract and BF16, VNNI U8/S8, and FP64 descriptor ABIs. ROCM-DTYPE-1 remains
the independent AMD GPU authority; no gfx capability, packing, accumulator,
execution, or selector row changes.

Cross-backend sync `ATTN-DIALECT-MLIR23-2026-07-20` corrects the internal MLIR
attention dialect namespace from the nested `tessera.attn` spelling to the
MLIR-23-compatible `tessera_attn` spelling. Public Graph IR operation names,
attention semantics, ROCm target capabilities, HSACO ABIs, schedules, and
selector state are unchanged; ROCm parity is validated by the shared attention
lit coverage.

Cross-backend sync `LLVM23-BACKBONE-2026-07-20` makes LLVM/MLIR 23.x the sole
accepted compiler build environment. Top-level and standalone CMake entry
points reject every other major and mixed installations; ROCm uses the
versioned apt LLVM 23 packages with ROCm 7.14. ROCm target semantics, HSACO
ABIs, and selectors are unchanged; host-free compiler/lit and gfx1151 unit
proofs validate parity without transferring evidence to another AMD target.

Cross-backend sync `X86-E2E2-COHORT2-2026-07-20` adds shared typed Tile
carriers for argreduce, inclusive scan, unweighted row normalization,
interleaved-pair RoPE, and ALiBi. ROCm parity is assessed at the semantic
carrier boundary only. AVX-512 ABIs, CPU schedules, Ryzen timing, and route
disposition transfer no RDNA/HSACO implementation, device evidence, or selector.

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` adds an explicitly x86-owned
`tile.x86_abi_kernel` and cohort-3/4 C-ABI registry. It changes no portable
semantic Tile carrier, RDNA/HSACO ABI, GPU schedule, dtype capability,
execution row, or selector. ROCm parity is therefore not applicable.
X86-E2E-2 is now closed with measured x86-only selector thresholds; this does
not change the ROCm not-applicable disposition or transfer device proof.

Cross-backend sync `LLVM23-LOCAL-CLEANUP-2026-07-20` rebuilds the generic host
compiler against LLVM/MLIR 23 and makes Linux TSAN use the LLVM 23 Clang
runtime. The gfx1151 paged-attention recorder now represents an invalid HIP
event interval as unavailable instead of fabricating positive device evidence;
end-to-end route timing and selection remain valid. No RDNA schedule, HSACO ABI,
dtype capability, or selector threshold changes.

Cross-backend sync `E2E-SPINE-2026-07-18` extends shared launch-level Tile
carriers for deterministic f16/f32 attention-backward dropout replay, fused
paged-attention with an explicit causal offset, and f16/bf16/f32 MoE storage.
This is **follow-up required** only if ROCm selects those portable carriers:
CUDA materializers, PTX ABIs, SM120 schedules, resources, and evidence do not
transfer to RDNA. Existing ROCm attention, paged-KV, and MoE capability rows
remain unchanged until gfx-owned lowering, HSACO launch, numerical, and exact
device proof land.

Cross-backend sync `CORE-COMPILER-1-2026-07-22` closes shared Graph/Neighbors
verifier gaps and makes ROCm's live MMA/WMMA tables the source for manifest MMA
metadata and equal-tier arbiter footprint cost. The existing ROCm descriptor
field remains for compatibility, gfx1151 FP8/BF8 rejection stays inherited
from the ISA tables, and no RDNA schedule, HSACO ABI, selector promotion, or
exact-device claim changes. LLVM 23 ROCm compiler parity is validated.

Cross-backend sync `CORE-COMPILER-2-2026-07-22` makes the ROCm-owned backend
pipeline the first complete default dtype chain: compute legalization, storage
legalization, then descriptor consumption before WMMA generation. This is safe
because ROCm already owns the executable packed INT4/INT8 consumer. Structured
`#tile.layout` remains executable in ROCm Tile lowering; the new generic
row-major materializer and guarded dynamic launch are x86-only and transfer no
RDNA schedule, HSACO ABI, selector, or exact-device claim.

Cross-backend sync `CORE-COMPILER-NEXT-2026-07-22` tightens shared Graph layout
propagation through agreed-layout pointwise chains and last-axis reductions,
preserves packed-storage attributes, and records source-layout provenance on
inserted casts. ROCm continues to consume its independent structured
`#tile.layout`; no Graph-cast materializer, RDNA schedule, HSACO ABI, selector,
or device proof transfers. The x86 dynamic last-axis reduction guard is not a
ROCm execution claim. Shared add/multiply/static-broadcast adjoints change Graph
IR only; no gfx backward runtime or exact-device promotion is claimed.

Cross-backend sync `CORE-COMPILER-FOLLOWON-2026-07-22` adds shared kind-aware
sum/mean, GELU/SiLU, and softmax Graph adjoints with host CPU oracle proof.
Dynamic mean, max/min, ReLU, and normalization remain explicit fallbacks for
the documented Graph-contract reasons. Guarded dynamic softmax, attention, and
growing KV-cache execution are x86-only; no RDNA layout, HSACO ABI, schedule,
selector, backward runtime, or exact-device claim transfers. Apple/NVIDIA
Graph-cast materializers likewise do not transfer: ROCm's independent
structured `#tile.layout` consumer remains unchanged.

Cross-backend sync `CORE-COMPILER-ADJOINTS-2026-07-22` registers shared
tensor-to-i1 comparison contracts plus internal scalar-threshold,
rank-reduced normalization-statistics, and explicit broadcast-in-dimension
Graph carriers. ReLU and unweighted RMSNorm/LayerNorm paired adjoints are
static/dynamic Graph-native and CPU-IR oracle-proven; the static shared path
lowers through linalg. ROCm is **follow-up required** for backward execution:
no HSACO/HIP ABI, affine gamma/beta contract, RDNA schedule, selector, runtime
binding, performance result, or gfx1151 device proof is added here. Dynamic
statistics remain Graph IR until a ROCm-owned materializer lands; existing
forward norm kernels transfer no backward claim.

Cross-backend sync `CORE-COMPILER-NORM-AFFINE-2026-07-22` makes integer
comparison signedness explicit in shared Graph IR and adds dynamic-dimension
carriers plus channel-affine RMSNorm/LayerNorm adjoints. ROCm owns the first
runtime-shaped affine forward materializer: one shape-independent gfx1151
HSACO ABI executes unary/affine RMSNorm and LayerNorm for f32/f16/bf16, while
signed-i32 and unsigned-u32 comparisons use distinct semantic routes over
signless physical storage. Exact gfx1151 numerical tests and operation-total
host-wall measurements are recorded. Backward HSACO launch is **follow-up
required**; Graph-native `dx`/`dgamma`/`dbeta` and forward device execution do
not imply compiled GPU backward execution or selector promotion.

Cross-backend sync `CORE-COMPILER-NORM-BWD-2026-07-22` closes the ROCm
normalization-backward follow-up. Family-specific RMSNorm and LayerNorm rows
now bind the public paired `native_backward` seam to compiler-generated,
runtime-shaped gfx HSACO. f32/f16/bf16 X/dY/dX storage is exact-device proven
on gfx1151; dGamma/dBeta accumulate and return in f32, and stable statistics
are recomputed from X under the recorded `recompute_all` residual policy.
Unary/affine, ragged/rank-3, large-offset, cache-identity, ROCDL, and
operation-total evidence are green. The atomic affine accumulation is not
claimed bitwise deterministic, and no selector promotion or sibling-device
proof transfers.

Cross-backend sync `CORE-COMPILER-NORM-BWD-DETERMINISM-2026-07-22` replaces
that atomic boundary with a deterministic two-kernel gfx1151 route. The row
kernel writes private f32 dGamma contributions and the channel kernel folds
those plus dY-for-dBeta in ascending row order, eliminating global atomics and
cross-workgroup write races. Repeated f32/f16/bf16 RMSNorm and LayerNorm launches are bitwise
identical on the LLVM/MLIR 23 + ROCm 7.14 gfx1151 host; exact numerical and
operation-total affine medians are 2.28--2.40 ms for the retained 32x128 and
7x300 packet shapes (maximum f32 error 3.82e-6), improving on the former
2.37--2.61 ms atomic-route range. Affine calls temporarily consume one
`M*K*sizeof(f32)` dGamma-partial buffer; dBeta needs no extra partial storage. This is a ROCm
schedule/temporary-storage change only; no selector promotion or sibling proof
transfers.

Cross-backend sync `CORE-COMPILER-LAYOUT-AUTODIFF-MEMORY-2026-07-23` completes
the shared transpose/packed epilogue/reduction layout envelope and adds native
guarded-dynamic broadcast, runtime-extent mean, and equal-share max/min Graph
adjoints. ROCm parity is host-validated through the shared linalg contract.
The ROCm backend pipeline now executes Tile buffer reuse and materializes one
address-space-3 LDS arena with typed planned-offset views before wave/Tile
lowering. Function-budgeted liveness-aware rematerialization also runs in the
shared production post-autodiff pipeline. Exact gfx1151 arena occupancy,
backward reduction launch, and performance evidence remain follow-up required;
no selector or device claim is transferred from host-free validation.

Cross-backend sync `CORE-COMPILER-TRAINING-SPINE-2026-07-23` registers
`tessera.loss.mse` and its paired backward carrier as verifier-checked shared
Graph IR, with dynamic none/sum/mean Linalg lowering and FP32 compute for
FP16/BF16 storage. The canonical Graph spelling now executes on gfx1151 for
multiple ranks through one shape-independent HSACO identity. Scalar sum/mean
retain the pointwise device allocation and feed it directly to the generated
reduction kernel; HIP module/function caching removes the former roughly
2 ms reload penalty. On this unified-memory gfx1151 host, measured medians over
263--262144 elements put the device and retained host epilogues within roughly
10%; the device path avoids the full-tensor D2H transfer and is retained as the
memory-safe default, without a selector claim. Exact tests cover f32/f16/bf16
and dynamic shapes. Compiled ROCm MSE backward remains follow-up required; the
shared Linalg adjoint is not an HSACO launch claim.

Cross-backend sync `CORE-COMPILER-DEEPENING-2026-07-23` closes that gfx1151 MSE
backward follow-up. One compiler-generated runtime-sized HIP kernel consumes
prediction, target, and reduction-aware cotangent and writes dPrediction plus
dTarget for none/sum/mean. Ragged 263-element and rank-2 exact-device rows pass
the numerical oracle, and the public `native_backward` seam records the
`save_inputs` residual policy. A 30-sample warmed operation-total mean-reduction
row at 65,536 f32 elements measures 0.875 ms median / 1.013 ms p95 on this
gfx1151 host. Shared dynamic address-space-3 arenas and the
measured rematerialization-cost contract also land, but no LDS occupancy,
selector, or sibling-device performance claim is inferred.

Cross-backend sync `CORE-COMPILER-TRAINING-BREADTH-2026-07-23` extends the
shared compiler-owned training graph with verifier-checked MAE, Huber,
SmoothL1, and SGD forward/adjoint ops plus dynamic Linalg lowering. ROCm owns
exact gfx1151 backward execution for all four regression losses and SGD.
Losses use one generated runtime-sized HIP kernel for both input gradients;
SGD uses a dedicated one-launch VJP after operation-total measurement exposed
the former two-launch composition at 5.83 ms. The retained 65,536-element
packet records 0.84--0.88 ms regression-loss medians and 0.82 ms SGD median
after module/function caching. Boundary semantics are explicit: MAE ties at
zero, Huber uses the closed `abs(error) <= delta` branch, and SmoothL1 uses the
open `abs(error) < beta` branch. f32 is exact-device proven; no selector,
other optimizer adjoint, or sibling AMD-device claim transfers.

Cross-backend sync `CORE-COMPILER-TRAINING-SERIES-2026-07-23` adds stable
BCE-with-logits and class-index/label-smoothed cross-entropy paired Graph
contracts plus exact one-launch gfx1151 backward execution. It also registers
KL/JS paired carriers and upgrades Momentum/Nesterov and Adam/AdamW from
opaque state to explicit tensor-state adjoints. Momentum/Nesterov have exact
single-launch gfx1151 VJPs and no-residual cache identities. The
`ROCM-TRAINING-MEMORY-FUSION-2026-07-27` continuation below closes the
architecture-owned Adam/AdamW and KL/JS physical backward gaps. Lion already
has explicit moment state and exact forward execution; Adafactor factored
state/execution, additional training-step fusion, and selector closeout remain
open.

Cross-backend sync `CORE-COMPILER-TRAINING-FUSION-2026-07-23` closes the first
training-step fusion envelope for MSE, MAE, Huber, SmoothL1, and stable
BCE-with-logits feeding SGD or AdamW. A shared post-autodiff rewrite requires
the prediction gradient to have exactly one optimizer use, preserves the target
gradient, and internalizes dPrediction. The fused Graph carriers lower through
one dynamic Linalg loop. On gfx1151, one compiler-generated runtime-sized HIP
launch writes updated parameter state (plus both AdamW moments) and dTarget.
All five families pass exact f32 execution for none/mean cotangents, including
wave and tail paths; different runtime shapes reuse one
`chip,dtype,kind,parameter,reduction` HSACO identity. At 65,536 elements, the
retained operation-total packet records 3.45--3.72x SGD and 1.72--1.80x AdamW
median speedups over loss backward plus optimizer launches. This does not close
the standalone Adam/AdamW optimizer-adjoint ABI. Series-2 dynamic
reduction/softmax/attention/paged-KV execution and exact LDS occupancy remain
open; no sibling-device selector or schedule claim transfers.

ROCm-owned slice `CORE-COMPILER-ROCM-DYNAMIC-2026-07-23` connects the existing
runtime-sized gfx1151 reduction, softmax, flash-attention, and KV movement
kernels to the shared prelaunch dynamic-shape guards. Exact execution covers
ragged last-axis reductions/softmax, unequal query/key sequence lengths
(including 33x17 and 33x79 rows), and capacity/logical-length-varying KV
append/read/prune. Reduction and softmax reuse one HSACO across rank/extent
changes. The retained 30-sample host-wall packet at 129x511 records 2.17 ms
reduction and 2.12 ms softmax medians; 33-query/79-key f16 attention records
2.03 ms, and a 17-row append into a 256-row cache records 1.98 ms. Policy:
reduction/softmax and KV capacity/logical lengths remain fully dynamic;
attention sequence lengths remain dynamic while head dimension, dtype, GQA,
window, softcap, bias, and two-wave features remain compile/cache buckets.
Dynamic normalization/loss fusion into additional downstream consumers remains
open; no selector claim transfers to other AMD devices.

Cross-backend sync `CORE-COMPILER-MEMORY-LAYOUT-CLOSEOUT-2026-07-23` corrects
the shared static arena ABI from an address-space-3 LLVM alloca (which the live
driver reported as zero LDS) to a real workgroup global. Exact gfx1151 HIP
queries now report 16,384/32,768 LDS bytes for the corresponding arenas; reuse
of two 16 KiB buffers halves planner peak and raises active blocks/CU from 2 to
4. Non-entry dynamic descriptors materialize in dominance-scoped cohorts.
The measured rematerialization corpus records a 128³ matmul+ReLU recompute at
1.65 ms operation-total while removing a 65,536-byte saved activation. x86
column-major ABI evidence is sibling-only. Dynamic arena launch sizing and
broader cost rows remain follow-up required; no other AMD-device claim
transfers.

ROCm-owned continuation `CORE-COMPILER-HONEST-BOUNDARIES-2026-07-23` closes
four previously named boundaries on the local gfx1151 host. Runtime-shaped
RMSNorm/LayerNorm fuse ReLU or SiLU into their final write pass and reuse one
HSACO per chip/kind/dtype/epilogue across extents. The post-ROCDL
`rocm-materialize-dynamic-lds` pass maps one runtime-sized address-space-3
arena to HIP launch-provided dynamic LDS; exact 16/32 KiB launches execute and
report 4/2 active blocks per CU. Multiple dynamic arenas are explicitly
rejected until a packed-offset launch ABI exists. Rematerialization evidence
now covers 64/128/192 matmul shapes with ReLU/GELU/SiLU and consumer-aware
lookup. Finally, compiled signed INT4 pack/unpack feeds the packed-memory IU4
WMMA route without host repacking; aligned and ragged results are exact and
physical inputs are approximately half the logical i8-container size. This
does not default-enable terminal packing for operations without a physical
consumer or transfer evidence to another AMD architecture.

ROCm-owned continuation `CORE-COMPILER-HONEST-BOUNDARIES-2-2026-07-24`
replaces the straight-line multi-arena rejection with an executable packed
launch ABI. Runtime byte arenas share one HIP dynamic-LDS symbol and use
16-byte-aligned prefix-sum offsets; the host supplies the matching aligned
total. Exact 8+8 KiB and ragged 12,289+4,111 byte launches are non-aliasing and
report 4/3 active blocks per CU. Control-flow-disjoint arenas remain honestly
rejected until a path-max launch expression exists. The measured
rematerialization corpus adds softmax, RMSNorm, and MSE producer families plus
a 512 KiB workload policy; gfx1151 selects MSE+softmax for the retained
three-producer workload. Dynamic RMSNorm/LayerNorm add canonical tanh-GELU as
a one-launch consumer. Finally, compiled terminal INT4 packing feeds the
existing group-scaled dequant-GEMM physical ABI with no host unpack/repack;
the exact 33x64x29 row halves code bytes and measures 2.15 ms operation-total.
No selector/default changes or evidence transfer to another AMD architecture.

ROCm-owned continuation `CORE-COMPILER-HONEST-BOUNDARIES-3-2026-07-24`
closes the four remaining packet boundaries on gfx1151. Direct mutually
exclusive branch successors reuse dynamic-LDS offset zero under a recorded
`max_of_aligned_sums` launch expression; the exact ragged 12,289/32,001-byte
case launches with 32,016 bytes, verifies both paths, avoids 12,304 bytes versus
summation, and reports two active blocks/CU. Sequential, nested, looping, and
escaping lifetimes remain deliberately rejected until a general lifetime
expression exists. The rematerialization packet is now four layers and six
producer families (softmax, RMSNorm, MSE, Huber, SmoothL1, BCE), with every
instance measured on gfx1151 and AVX-512; both reduce a 17,301,504-byte
activation set below a 2 MiB budget using target-specific costs. Dynamic
RMSNorm/LayerNorm now fuse same-shape residual add/multiply; the retained
7x300 LayerNorm+add row is 2.23 ms operation-total. Packed signed INT4 now has
three additional physical gfx1151 consumers—nibblewise ReLU, indexed sparse
gather, and packed cache append—with 30-sample medians of 3.88, 2.31, and
2.26 ms and no host unpack/repack. These are gfx1151-owned ABI/evidence claims;
no selector default or sibling AMD claim transfers.

ROCm-owned continuation `CORE-COMPILER-CFG-MEMORY-BUDGETS-2026-07-24`
replaces direct-branch path enumeration with alias-aware SSA-liveness
interference slots. Nested branch and loop-local runtime LDS arenas reuse one
slot; an outer arena live across the loop receives a distinct aligned slot.
CFG-forwarded size arguments resolve to host-visible kernel-argument leaves,
and arbitrary local size arithmetic remains a named rejection until a
serializable expression ABI exists. Exact gfx1151 execution verifies both
branches across two loop iterations with slots `[8192]` and
`max[4097,12289,32001]`: the launch uses 40,208 bytes, produces exact results,
and reports one active block/CU. The shared rematerialization pass now derives
activation budgets from a device/model envelope when no explicit override is
present. This changes no ROCm selector and transfers no LDS evidence to another
AMD architecture.

Cross-backend sync `E2E-SPINE3-SM120-MEMORY-2026-07-24` extends the shared
fleet fixture corpus with bounded epilogue, attention, and ReplaySSM identities
and seals the six formerly pending NVIDIA SM120 family rows. ROCm can reuse
only fixture identity and proof-schema structure. CUDA image descriptors,
NVPTX address-space-3 materialization, ptxas accounting, SM120 resources,
timings, and release readiness do not transfer to AMDGPU. The existing
gfx1151 packet and architecture-owned LDS/rematerialization evidence remain
unchanged.

Cross-backend sync `CUDA-TRAINING-MEMORY-FOUNDATION-2026-07-24` is
NVIDIA-owned. It changes no shared Graph/Linalg mathematics and no ROCm
execution row, selector, HIP ABI, or LDS policy. The CUDA PTX image/descriptor,
CUDA-driver launch-v2 entry points, NVPTX external shared symbol, ptxas/driver
resources, and SM120 timings do not transfer to gfx1151. ROCm's existing
training kernels and `aligned_sum_of_slot_maxima` LDS proof remain the
architecture-owned sibling implementation.

Cross-backend sync `CUDA-TRAINING-MEMORY-BREADTH-2026-07-24` adds only the
portable Graph IR carriers for model-parameter marking and bounded dynamic
parameter storage. NVIDIA owns the CUDA-context capacity/free-memory query,
FP16/BF16 PTX training ABIs, serialized dynamic-shared launch expressions, and
SM120 measurements. None transfers to HIP/AMDGPU or changes ROCm execution,
LDS sizing, rematerialization policy, selectors, or exact-device claims.

Cross-backend sync `NVIDIA-BF16-CANONICAL-BREADTH-2026-07-25` widens the
shared Tile softmax and attention verifier envelope to BF16 storage with FP32
accumulation and preserves the shared BF16 reduction/min contract. ROCm parity
is validated at the semantic boundary: gfx1151 already owns exact-device
f16/bf16/f32 reduction proof, while any use of the widened softmax or attention
carrier remains architecture-owned and must pass through ROCDL/HSACO with
gfx1151 evidence. NVIDIA's PTX descriptors, serial/cooperative-128 schedules,
CUDA launch ABI, SM120 resource packet, timing, and selector disposition do
not transfer. No ROCm capability, execution row, schedule, or selector changes.

The NVIDIA continuation adds its architecture-owned compiler/PTX BF16
normalization image and a physical CUDA consumer of the shared
`tessera.storage_pack` descriptor for scale-bearing NVFP4/MXFP4/FP6. ROCm
descriptor-schema parity remains validated by its existing WMMA int4/int8
consumer; NVIDIA's format-defined signedness, scale ABI, packing loads,
resources, and SM120 evidence do not transfer to RDNA. No ROCm pipeline,
capability, execution row, schedule, or selector changes.

Cross-backend sync `NVIDIA-PACKED-MATH-2026-07-25` consumes the existing shared
signed-INT4 `tessera.storage_pack` schema in a CUDA-owned correctness schedule
and adds a typed internal Tile carrier for a bounded CUDA Math subset. Shared
descriptor semantics are parity validated, but NVIDIA's packed A/B layout,
PTX instructions, CUDA launch ABI, resources, cache proof, and SM120 execution
do not transfer to HIP/AMDGPU. ROCm's independent WMMA INT4 consumer remains
authoritative; no ROCm pipeline, capability, schedule, or selector changes.

Cross-backend sync `NVIDIA-PACKED-SSA-FOUNDATION-2026-07-25` migrates the
existing ROCm WMMA pack reader to the structured shared
`#tile.packed_format` contract and introduces portable
`#tile.packed_view`/`#tile.scale_layout`, generic packed load/store, and SSA
buffer/pipeline vocabulary. ROCm's signed INT4 WMMA physical consumer remains
authoritative. Generic HIP FP4/FP6 scale consumers and WarpSpec/LDS SSA
threading are **follow-up required** if selected; CUDA lane order, PTX Math
operations, SM120 resources, and exact-device evidence do not transfer.

The synchronization point now also registers shared SSA TMA descriptor,
mbarrier, mbarrier-token, TMEM, and TCGen05 vocabulary and makes the NVIDIA
WarpSpec path consume shared allocation/pipeline identity. TMA/TMEM/TCGen05 are
**not applicable** to AMDGPU with an architecture-specific reason; ROCm must
map its own LDS allocations, async-copy/waitcnt/s_barrier dependencies, and
WMMA/MFMA consumers onto the shared buffer/pipeline model. That remains
**follow-up required** and no NVIDIA structural or device evidence closes it.

ROCm-owned continuation `ROCM-TRAINING-MEMORY-FUSION-2026-07-27` adds a
compiler-generated one-launch paired Adam/AdamW VJP over explicit parameter,
gradient, first-moment, and second-moment state. Exact ragged gfx1151 tests
cover both weight-decay variants. KL and Jensen-Shannon divergence now lower
to one paired-gradient HIP kernel with arbitrary class-axis addressing,
scalar/tensor cotangents, strict epsilon-clamp derivatives, and exact rank-3
and mean-reduction proof. A warmed 30-sample operation-total packet records
4.059 ms AdamW at 257x255, 1.169 ms KL at 17x31x13, and 1.135 ms JS at
527x13; allocation, copies, cache lookup, launch, synchronization, and result
copies are included, so these are not kernel-only selector evidence.

Continuation `ROCM-LION-BACKWARD-2026-07-27` closes the remaining elementwise
Lion adjoint on gfx1151. The compiler now emits one paired HIP kernel for
parameter, gradient, and carried-moment VJPs under the canonical
stop-gradient-through-sign policy; the affine VJP requires no saved forward
residuals. Exact ragged `5x19` device proof covers nonzero decay and both
output cotangents. The refreshed 30-sample operation-total packet records
2.908 ms AdamW, 1.383 ms Lion, 1.186 ms KL, and 1.116 ms JS medians; these
include allocation, copies, cache lookup, launch, synchronization, and result
copies and remain selector-ineligible. Adafactor's factored row/column state
topology remains the next optimizer execution gap.

The same continuation lifts dynamic local-memory arithmetic into the shared
`LaunchDescriptor`: serialized argument, constant, add, multiply, max,
path-max, and alignment expressions are validated, signed-i64 checked,
content-addressed, and resolved from launch scalars. Legacy descriptors retain
their digest when no expression is present, and NVIDIA's existing serialized
dynamic-shared probe now consumes the shared field rather than backend
provenance. ROCm interference-slot allocation already handles sequential,
nested, looping, forwarded-alias, and conservative escaping lifetimes;
  compiler emission of arbitrary local arithmetic into the new descriptor and
  production device-capacity injection remained open at this synchronization
  point. The later `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record closes
  gfx1151 production capacity injection; arbitrary IR-expression emission and
  sibling-launcher injection remain open.

Dynamic RMSNorm and LayerNorm gain a canonical one-launch softcap consumer,
`cap * tanh(value / cap)`, alongside ReLU, SiLU, GELU, add, and multiply.
Positive finite cap validation, ragged/rank-3 exact gfx1151 tests, and
host-free ROCDL structural checks land without changing a production selector.
Adafactor's factored row/column state and broader loss-to-optimizer fusion
remain the next training/fusion work.

`ROCM-LSE-1` revalidation now records 21-sample resident FP16 and BF16
wall-clock plus HIP-event distributions at 17/64/128/256. HIP 7.14 returns
positive events under this WSL host, but the packet is fail-closed as
`blocked_wsl_device_event_not_transferable`; the 128+ saved-LSE policy still
awaits bare-metal gfx1151 confirmation. The threshold is therefore
**provisional, not production selector evidence**: a bare-metal owner must
repeat valid device-event measurements and retain or replace 128+ before this
item can close. The 2026-07-30 x86 attention/training closeout changes no ROCm
kernel, ABI, schedule, or completion state.

Cross-backend sync `CORE-SCHEDULE-1F1B-MATERIALIZE-2026-07-27` makes the shared
pipeline legality pass emit an explicit, unique-clock warmup/steady/cooldown
dependency order after proving stage and transport legality. This changes no
AMDGPU schedule or HIP execution row yet. At this synchronization point ROCm
runtime consumption and collective overlap were follow-up required; the next
`CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` record supersedes that structural
gap with the shared runtime consumer. No CUDA/Apple physical evidence transfers.

ROCm-owned continuation `CORE-COMPILER-RUNTIME-CLOSEOUT-2026-07-27` lands the
physical Adafactor program on gfx1151. One compiler-owned HSACO contains
deterministic row-moment, column-moment, ordered row-mean, factored parameter
update, and lower-rank full-moment update entries with explicit state ABI.
Three-step exact-device fixtures cover both factored and full-moment state and
match `optim.adafactor`; a 30-sample `257x255` WSL
operation-total packet records a 1.632 ms median and 1.775 ms p95 including
allocation, copies, cache lookup, four launches, synchronization, and result
  copies. The packet is selector-ineligible. A physical Adafactor adjoint
  remained an explicit follow-up at this point; the later
  `CORE-PRODUCTION-EVIDENCE-2026-07-27` record closes it on gfx1151.

The same continuation queries total/free bytes from the retained HIP context
and injects the effective capacity, reserve, bounded dynamic parameters,
gradient/optimizer copies, and persistent state into the shared
rematerialization contract. Model-level static/dynamic validation is covered;
full-graph measured rematerialization selection remains open.

The shared unique-clock 1F1B carrier now has a runtime consumer: compute follows
the emitted order while selected backward collectives execute on an independent
transport executor and are joined before completion. This closes the missing
runtime-consumption seam structurally; a real multi-rank ROCm transport packet
and optimizer-shard collective mapping remain exact-device follow-ups.

Measured schedule records can now override tile M/N/K, warp count, and pipeline
depth in the actual Schedule IR and Tile IR, with target/evidence/latency
validation. This removes the prior metadata-only autotune result path. ROCm
kernel-specific candidate measurements and selector ratchets remain owned by
their individual work items.

The DeltaNet/Kimi/modified-delta family replaces finite-difference reverse mode
with an analytic O(S) reverse recurrence for Q/K/V, gate, beta, decay, erase,
modified normalization, and carried state. Schedule IR records FP32 carried
state, chunked/recurrent forward, and reverse-token backward ordering.
Directional-derivative fixtures prove the shared math. The physical-packaging
follow-up in the next synchronization record supersedes this earlier open
statement; the later `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28`
record also supersedes the earlier parallel-chunk open statement.

Continuation `CORE-PRODUCTION-EVIDENCE-2026-07-27` closes the physical
factored/full Adafactor adjoint follow-up on gfx1151. The compiler-owned
ten-entry HSACO now includes deterministic checkpoint/recompute, mean,
row/column, finalize, and full-moment reverse entries; exact factored and
full-moment fixtures match the shared analytic VJP. The operation-total packet
records a 14.422 ms median and 15.038 ms p95 for `257x255`, including
allocation, copies, cache lookup, seven launches, synchronization, and result
copies. It is deliberately selector-ineligible.

Emitted 1F1B steps now own serializable collective descriptors. The shared
OptimizerShard runtime enforces replicated-to-rank-local reduce-scatter and
rank-local-to-replicated all-gather transitions while transport overlaps
compute. NCCL and RCCL use the same collective ABI with CUDA and HIP runtimes
selected independently. Deterministic two-rank integration uses the exact
descriptor path. This host exposes one gfx1151 and no loadable RCCL library, so
real multi-rank RCCL evidence remains blocked rather than inferred.

ROCm DeltaNet backward is physically packaged as a five-entry FP32 program:
checkpoint, affine chunk summary, deterministic prefix, parallel chunk fill,
and unique-`(batch,head)` reverse. Exact gfx1151 proof crosses chunk boundaries
and matches dQ/dK/dV and gate/beta/decay derivatives for gated and modified
recurrences, including the modified-normalization VJP. For `erase=false`, the
compiler composes exact `state_out = scale * state_in + update` chunk summaries;
state-dependent erase retains its exact serial checkpoint dependency.

Cross-backend sync `CORE-SEQUENCE-MIXER-PHYSICAL-BACKWARD-2026-07-28` records a
resident two-cohort gfx1151 packet for modified backward at `[2,8,128,16]`.
Chunk 16 wins both cohorts (12.982 and 13.008 ms median) over the serial
chunk-64 baseline (13.689 and 13.710 ms), with `2.21e-7` maximum error and
0.20% cross-cohort variation. The packet is intentionally selector-ineligible:
this exact gfx1151 host is WSL, so bare-metal timing remains the production
selector gate. Apple/CUDA schedules are not inferred from this AMD package.

## Cross-backend sync `TILE-FRAGMENT-TYPE-PARAM-2026-08-03` — `!tile.fragment` parameterized (W1.1 step 1)

Shared Tile IR type changed: `!tile.fragment` gained `(m, n, k, elem, acc, role, layout, family)` and a domain verifier. **No behaviour changes in this PR** — the bare `!tile.fragment` still parses AND still prints bare, so every existing producer and fixture is unaffected. All 7 C++ `FragmentType` uses are `isa<>` checks, so there were no construction sites to migrate.

**Outcome: follow-up required.** 5 files under this backend reference `FragmentType` / `!tile.fragment`. Same two obligations as NVIDIA:

* **Step 2b (blocking).** `TileToROCM.cpp` requires a `FragmentZeroOp` accumulator (3 sites), so a K-loop iter-arg accumulator cannot lower here either.
* **Step 3.** `GenerateWMMA{Gemm,LinearAttn,FlashAttn}Kernel.cpp` are 3 of the 5 `tile.mma` construction sites to migrate.

This backend is the reason `family` lives in the TYPE rather than the attribute. `ROCMFragmentLayout.h` resolves a family to a `FragmentLayoutDescriptor` whose **wave size differs** — 32 for RDNA3/RDNA4/gfx125x WMMA, 64 for CDNA MFMA — with different element counts and formats, and `TileToROCM.cpp` states outright that those descriptors "are intentionally non-interchangeable". The fragment type now encodes that, so a mismatch is a type error rather than a lowering-time one.

No gfx1151 device evidence in this PR; no generated code changed, so none is due.

## Cross-backend sync `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03` — typed `tile.mma` K-loop (W1.1 step 2)

Shared Tile IR contract changed: `MMAOp::verify()` (and the `fragment_pack` / `fragment_zero` producers) now read the operand contract from the fragment TYPE when it is parameterized, falling back to producer-chasing for the bare form. `#tile.mma_desc` is optional on the typed path and cross-checked when present. **The canonical K-loop now verifies.** No lowering changed in this PR, and no existing IR is affected — the bare form keeps its old path.

**Outcome: follow-up required — and larger than previously recorded.**

Step 2b was scoped as "teach the lowering to accept a region iter-arg". Measuring `TileToROCM.cpp` while implementing step 2 shows that is wrong and would have been a correctness bug: the typed path **materializes a zero** and passes it as the MMA's C operand —

```cpp
Value zero = arith::ConstantOp::create(builder, loc, accTy,
                                       builder.getZeroAttr(accTy));
state.addOperands({*a, *b, zero});
```

— and never reads `mmaData[2]` as a value (`cZero` is used only for the null check and a dead-op erase). `FragmentZeroOp` is therefore a **precondition the lowering relies on**, not a pattern it matches. Accepting a block-argument accumulator without threading the value would make every K-loop iteration recompute from zero: the GEMM returns the last K-step's partial product, with no diagnostic.

Real 2b here: thread the accumulator into the MMA, which means **converting the `scf.for` region signature** (Tile `!tile.fragment` -> lowered `vector<N x f32>`), not swapping an operand.

**Gate must include numerics.** This box executes gfx1151, so the ROCm half is verifiable here: run a K-loop GEMM and compare against a reference. A lowering fixture that only checks emitted ops would pass while the kernel returned a wrong answer — exactly this failure mode. No device evidence in this PR; none due, since no generated code changed.

## Cross-backend sync `NVWGMMA-ACCUMULATOR-GUARD-2026-08-03` — WGMMA accumulator drop (W1.1 step 2b guard)

A `tile.mma` carrying an accumulator was lowered by `NVWGMMALoweringPass` to a **two-operand** WGMMA call: the accumulator was discarded, the shape hardcoded `m64n64k16`, and the dtype inferred through `dyn_cast<ShapedType>` (which a `!tile.fragment` is not, so it defaulted to bf16) — with **rc=0 and no diagnostic**. A K-loop recomputed A×B from nothing each step and returned the last partial product.

Measured on merged main, this was **not** specific to the typed fragment form: a legacy bare `tile.mma(A, B, C)` — what `LowerKReductionAddToTileMMA` emits for the canonical K-step — was dropped identically. **No fixture in the tree covered either case**, which is how it survived. The guard therefore keys on *has an accumulator*, not *is typed*.

**Outcome: parity validated — no change required.**

Probed with the same legacy 3-operand `tile.mma`: `--tessera-lower-to-rocm` fails closed with a named error and leaves `tile.mma` in place. `TileToROCM` already requires a `FragmentZeroOp` accumulator and materialises its own zero, so it never silently discards a caller's accumulator.

ROCm remains the reference behaviour for this seam, and its own step-2b work (real accumulator threading, gated on gfx1151 numerics) is tracked under `TILE-FRAGMENT-KLOOP-ACCUM-2026-08-03`.

## Cross-backend sync `ROCM-COMPILED-STRICT-DISPATCH-2026-08-04` — compiled-lane failures stop masquerading

Runtime dispatch contract changed. A compiled-ROCm **failure** (tessera-opt ran and serialized no kernel, or emitted a non-ELF blob) now routes through the existing `_note_dispatch_fallback` funnel, so `TESSERA_STRICT_DISPATCH=1` raises instead of degrading. **Envelope limits** (no libamdhip64, hipInit failed, tessera-opt not built, dtype/rank/arch out of range) are unchanged and still degrade silently — making those raise would break strict runs on every CPU-only host.

Measured before the fix: a deliberately broken pass pipeline returned `ok=True, compiler_path="rocm_compiled", execution_kind="native_gpu"` with correct numbers. Strict-mode suite results are identical before and after (18 fail both ways, all pre-existing), so this adds no new failures.

**Outcome: follow-up required — this backend owns the change.**

18 failure-class raise sites across the gemm, canonical gemm, flash_attn fwd/bwd, linear_attn, softmax and norm fwd/bwd lanes now go through `_rocm_compiled_failed`. Two were missed on the first pass because the conversion matched MESSAGE TEXT; the structural gate `test_every_non_elf_check_routes_through_the_funnel` now enumerates by the `!= b"\x7fELF"` guard instead, so wording cannot hide one again.

**Why it mattered here specifically:** W1.1 step 2b is gated on gfx1151 numerics, and that gate could not distinguish a working lowering from a silent fallback. With it fixed, the 2b measurement ran and inverted the plan (see `W1_1_TYPING_DESIGN.md` §4.3).

## Cross-backend sync `ROCM-PIPELINE-TILE-LOWERING-2026-08-04` — the compiled pipeline can lower `tile.mma`

Both ROCm compiled pipelines (plain and canonical) now run `lower-tile-to-rocm{arch=<chip>}` after `generate-wmma-gemm-kernel`. Verified byte-identical hsaco with and without the pass on the default path, so the production lane is unchanged.

**Outcome: follow-up required — this backend owns the change.**

`GenerateWMMAGemmKernel{via-tile=true}` emits `tile.mma %a, %b, %acc`, but no runtime pipeline lowered it: the op reached LLVM translation and died on *"missing LLVMTranslationDialectInterface registration ... for op: tile.mma"*. W1.1's Tile-IR seam was therefore unreachable from the lane that actually executes.

With the pass in place, via-tile compiles and runs **bit-identical** to the production `tessera_rocm.wmma` lane on gfx1151 at 64^3, 256^3 and 128x96x64 — so the accumulator survives the round trip. Gated by `test_rocm_pipeline_tile_lowering.py`: a structural count (every wmma-gemm lane must have a lowering) plus a hardware numeric comparison carrying the bogus-option control.

`arch=` is mandatory and gated separately: the pass defaults to a CDNA part and emits `llvm.amdgcn.mfma.contract`, an MFMA intrinsic wrong for RDNA 3.5 that does not resolve.

**Closed 2026-08-04 (W1.1 step 0).** `TileToROCM`'s typed branch no longer
requires a `FragmentZeroOp` accumulator: the typed path is now a dialect
conversion (`!tile.fragment<...>` -> `vector<N x T>`), so a K-loop iter-arg
accumulator, chained MMAs, and any non-zero accumulator lower by
composition. Fixture: `rocm_typed_fragment_composition.mlir`.

**Strided-K landed 2026-08-04.** The prerequisite named here — that
`materializeFragmentPack` addressed only fragments whose K axis is CONTIGUOUS
in memory, while the producer stores B row-major (`k * N + col`, stride N) — is
closed. The fragment always walks K; the memory order now only decides whether
that walk is contiguous (`vector.load` / `maskedload`) or strided by the leading
dimension (element-by-element gather with the same `inb ? value : zero` shape
the generator uses by hand). All four (role x order) combinations are
addressable; two of them were previously rejected as an "unsupported source
layout". The contiguous path is **byte-identical** — verified by diffing lowered
output for the composition, ragged-bounds, and RDNA4-int4 fixtures. Fixture:
`rocm_fragment_strided_k.mlir`.

**Stack context (measured 2026-08-05):** see
[`ROCM_LANE_MAP.md`](ROCM_LANE_MAP.md). The executing ROCm GEMM lane starts
from a hand-built **Target-IR** directive and traverses no Graph IR, no
Schedule IR, and no Tile IR; `lower-tile-to-rocm` is in its pipeline but is a
verified no-op on the default path. So W1.1's typed contract affects no
executing kernel until step 3 lands. Keep two costs apart: closing the Tile
fragment contract (steps 3-5) is **5 C++ `tile.mma` creation sites + the Python
emitters**, while making the backend traverse Tile IR is **58 expanders** and is
unpriced. Only the second scales with the expander population.

**Step 3 pilot landed:** `GenerateWMMAGemmKernel{via-tile=true}` is the first
C++ producer of the full typed `tile.view` -> `fragment_pack` -> `tile.mma` ->
`fragment_unpack` -> `tile.store` chain. At production `mt=2, nt=4`, the
structural gate sees 24 views/packs, 8 typed zeros, 32 MMAs, and 16
unpack/stores; no Tile op or unrealized cast survives ROCDL lowering. The
default production lane remains direct. The ROCm flash- and linear-attention
sites now use typed register-owned fragment bridges for computed values that
cannot be reconstructed from pointer-backed views; their final ROCDL operation
multisets match the direct lanes. The only remaining C++ sites are the two
NVIDIA-owned tensor producers in `TileIRLoweringPass`, and Python has no direct
`tile.mma` emitter after converging on `tile.matmul_kernel` launch envelopes.
The migration is scoped by measurement in
`W1_1_TYPING_DESIGN.md` §4.7:

* **Bit-identity is reachable** — the producer's B addressing `(k0+j)*N + col`
  and the strided gather's `(k0*N + col) + j*N` are the same integer, and the
  producer already assembles B as `memref.load` + `vector.insert`, the same
  shape the gather emits. That was step 3's main risk, and it does not depend
  on any instruction count.
* **Address FORM differs structurally.** Read off the emitted IR: the producer
  computes `(k0+j)*N` once per `j` and reuses that one SSA value across all `nt`
  B fragments, and hoists `arK[mi]` out of the K loop entirely.
  `fragment_pack` derives its address from `(base, rowOrigin, colOrigin)` in
  isolation — it cannot see its siblings and has no operand for a precomputed
  base. **How much this costs is NOT quantified**: the affected multiplies are
  largely loop-invariant, so LICM/CSE may recover them, and the sharing that
  does not survive scales with `nt`, which is 4 in production, not 16. An
  earlier version of this entry carried a 32-vs-288 comparison — both numbers
  were wrong (a truncated `awk` range and a config that is not the default);
  see `W1_1_TYPING_DESIGN.md` §4.7.
* **Resolved design question:** measurement selected an optional precomputed
  `linear_base` on `tile.view`; logical origins remain for bounds. This exposes
  the direct lane's A-base hoisting and sibling-B sharing, although the result
  below shows that address form was not the dominant remaining cost.
  Cross-backend sync key `TILE-VIEW-LINEAR-BASE-2026-08-05`; NVIDIA is
  follow-up required, x86 and Apple not applicable (§4.7 table).

Gate: `test_via_tile_matches_the_production_lane_on_hardware` (bit-identical vs
the production lane, with a control that must fail and a fallback-fatal guard),
then throughput against `GEMM_PERF_LADDER.md`'s 8.02 TFLOP/s row at the same
`timer_source`.

**Exact gfx1151 outcome, updated 2026-08-05.** The hardware test passes with
zero-difference output at both aligned `64x64x64` and ragged `65x67x31`; the
second case makes the M/N edge masks, K-tail masks, dynamic K/N leading
dimensions, and bounded stores live. The corrected, order-balanced 2048^3
host-wall run (100 launches x 3 interleaved trials) measures **12.53 TFLOP/s**
for the typed lane, **1.562x** the committed 8.02 baseline. The unchanged
shared-library baseline harness remeasures **8.07 TFLOP/s**, confirming the
reference row. The direct compiler-generated lane measures **18.28 TFLOP/s**,
so typed/direct is only **0.685x**. Verdict: **retain the typed lane as an
experimental correctness path, do not promote it**; pursue the precomputed-base
or equivalent address-hoisting work under
`TILE-VIEW-LINEAR-BASE-2026-08-05` before another promotion measurement.

That follow-up is now implemented: `tile.view` accepts an optional
`linear_base` SSA operand while retaining logical row/column origins for bounds,
and the 2x4 producer supplies the hoisted A and shared B bases. The exact
aligned/ragged bit-identity gate remains green. A fresh 2048^3, 100-launch x 3
interleaved host-wall packet measures **15.45 TFLOP/s typed versus 21.74 direct
(0.711x)**. This is a modest improvement over 0.685x, not performance recovery.
The typed image uses 161 VGPR/90 SGPR with no spills, but contains 281 waits
versus 85 and more fragmented scalar half-loads than the direct lane. An
experimental sibling-fragment load-grouping contract measured only 0.709x and
was removed. Verdict remains **retain experimental, reject promotion**; the
next performance design must reduce fragment-load scheduling/wait overhead,
not retune coefficients or hide the gap.

Binary decomposition narrows that next design further. In the aligned 2x4
steady-state K loop, both images select exactly 64 scalar B half-loads, four
128-bit A loads, eight `v_wmma_f32_16x16x16_f16` instructions, and progressive
`vmcnt(3/2/1/0)` waits. The typed loop is 134 instructions versus 142 direct,
uses 161 VGPR/82 SGPR with no spills, and therefore is not losing to extra
arithmetic or occupancy. The material difference is cross-fragment VMEM
scheduling: typed forms two 32-load B clauses (all low halves, then all high
halves), while direct interleaves address generation with shorter clauses and
adjacent K-row halves. A size sweep supports a cache-latency diagnosis:
`1024^3` is 10.35/10.47 TFLOP/s (0.988x), `2048^3` is approximately 0.71x, and
`4096^3` is 14.86/19.66 TFLOP/s (0.756x). Forced scheduler barriers, encoded
clause-break spans, adjacent-pair IR construction, and LLVM's small-GEMM IGLP
hint were all measured and rejected; the first two regressed to 0.622x and
0.596x, while the latter two produced no useful ISA or throughput change.
LLVM's variant-0 small-GEMM strategy is intrinsically a repeated
`DS(2) -> WMMA(1)` pipeline, so it cannot classify this register-owned kernel's
VMEM gathers. A Tessera-specific `VMEM_READ(8) -> VALU(4)` IGroupLP experiment
did shorten the hot loop from two 32-load clauses to twelve direct-like clauses,
but still measured only 15.15/21.26 TFLOP/s (0.712x). Clause shape alone is
therefore not the missing contract: the scheduler must retain which loads form
one sibling fragment and adjacent K-row register pair.
The whole-sibling hypothesis was then implemented rather than left as a plan:
TileToROCM materialized all four B packs K-major, fenced adjacent K-row pairs,
and retained the same progressive waits. It passed the aligned/ragged
bit-identity gate and produced the intended direct-like load order, but measured
only **14.79/21.05 TFLOP/s (0.703x)**. Combining it with an explicit
latency-oriented two-waves/EU request measured **14.89/21.12 (0.705x)**. Both
experiments were removed. The retained evidence now rejects instruction count,
occupancy, clause length, and fragment/K-pair order as isolated explanations.
The next credible implementation is to factor the direct lane's complete
physical panel emitter (address evolution, fragment construction, WMMA issue,
and resource policy) behind a target-owned interface and make both the direct
and typed Tile consumers call that same implementation. This preserves the
typed artifact boundary while eliminating another approximate reconstruction.

**Closed on gfx1151, 2026-08-05.** The target-owned physical-panel emitter now
backs the direct generator and the atomic typed-panel fallback. More
importantly, the production 2x4 f16/bf16 typed contract stamps a SHA-256 over
the ordered semantic Tile body (views, packs, accumulator-threaded MMAs,
unpacks, stores, types, physical layouts, and producer relationships).
`lower-tile-to-rocm{arch=gfx1151}` verifies that identity and the exact
24-view/24-pack/32-MMA/16-unpack/16-store topology before the shared target
emitter materializes the complete physical function. It does not re-enter
Graph IR, and valid scheduler annotations between the two passes do not alter
the semantic digest. Same-typed operand substitution and panel-attribute
tampering both fail closed.

The resulting direct and typed Target IR is text-identical and the serialized
HSACO is byte-identical (SHA-256
`bfb1d40b7c9938f4a6c9ccc19b365611d9d0c2b7d97eb9c54ad97752c67f1a97`),
including 256 VGPR, 107 SGPR, and the same cold-path spill record. Exact-device
aligned/ragged output remains bit-identical. Interleaved 100-launch host-wall
packets measure typed/direct **10.68/10.67 TFLOP/s (1.001x) at 1024^3**,
**21.27/21.42 (0.993x) at 2048^3**, and **20.54/20.58 (0.998x) at 4096^3**.
The former 0.711x typed performance gap is closed. The canonical scheduled
gfx1151 package now selects `via-tile=true`; the generic direct generator stays
as the byte-identity oracle/candidate until duplicate-authority deletion.
gfx1200/gfx1250 remain fail-closed and receive no transferred schedule claim.

The pre-existing WSL teardown abort is resolved rather than waived. ROCm
runtime and benchmark loaders no longer preload HIP/HIPRTC or promote their
symbols process-wide; directly linked plugins and `libamdhip64.so` are opened
with `RTLD_LOCAL`. The combined aligned+ragged pytest exits normally, and the
original `benchmark_rocm_wmma_gemm.py --iters 100` baseline run now exits zero
after printing all three sizes.

## Cross-backend sync `TILE-DYNAMIC-LEADING-DIM-2026-08-04` — generic typed fragment addresses

`#tile.memory_layout<leading_dim = 0>` now means `tile.view` / `tile.store`
carry the runtime leading dimension as their final SSA operand. Bounded forms
retain row/column bounds immediately before it. This is required by the
problem-size-generic ROCm GEMM: hard-coding the 64x64 hardware-gate shape would
have made the gate green while misaddressing every other N/K.

**Outcome: owning backend; parity validated on gfx1151.** ROCm consumes bounded
and unbounded dynamic forms, masks ragged loads/stores, and lowers the full
chain. The default direct lane is unchanged.

## Cross-backend sync `TILE-VIEW-BOUNDED-CONTRACT-2026-08-04` — bounded `tile.view` is a shared contract

`ViewOp::verify` now defines the pointer-backed operand contract: exactly 3 `(base, rowOrigin, colOrigin)` or 5 with `(rowBound, colBound)`. It previously accepted any count >= 3, so a 4-operand view was legal and meaningless and the bounded form's validity was decided by whichever backend looked.

**Outcome: parity validated — supports the bounded form.** `materializeFragmentPack` masks loads against the bounds with `vector.create_mask` + `maskedload` (PR #510). Fixture: `rocm_fragment_ragged_bounds.mlir`.

## Cross-backend sync `E2E-REAL-LINEAGE-SCHEDULE-2026-08-05`

Shared compiler orchestration now records producer, parent/output digests,
representation, and contract version per artifact, and production
`tessera-opt` registers the generated Schedule dialect. **ROCm outcome:
follow-up required under E2E-REAL-3.** Existing gfx1151 packages still consume
`GraphIRModule`; their rebuilt Tile artifact therefore records Graph—not the
shared Schedule artifact—as its parent and correctly reports
`lineage_complete = false`. No HSACO, descriptor, selector, or gfx1151 schedule
changed. The later ROCm consumer PR must accept the canonical launch-Tile
artifact and rerun aligned/ragged exact-device gates.

## Cross-backend sync `E2E-REAL-SCHEDULED-MATMUL-2026-08-05`

The shared C++ spine now preserves a bounded static Graph matmul behind a
content-addressed `schedule.matmul` SSA edge and lowers it exactly once to the
portable A/B/D/M/N/K `tile.matmul_kernel` contract. The gfx1151 instance is
f16 storage with f32 accumulation/output, m16n16k16 row/col WMMA, explicit
pipeline/raster fields, and a retained schedule digest. Dynamic shapes and
ROCm f32 storage fail closed. **ROCm outcome: structural parity validated;
physical follow-up required under E2E-REAL-3.** No HSACO or device execution is
claimed here. Canonical packaging must next accept this exact Tile artifact,
run `GenerateWMMAGemmKernel` then Tile→ROCM/LLVM, and repeat aligned/ragged
gfx1151 numerical and performance gates without Graph re-entry.

## Cross-backend sync `E2E-REAL-PHYSICAL-CONSUMERS-2026-08-05`

The bounded f16/f32 matmul package now accepts `ScheduledMatmulArtifact` and
consumes its exact launch-level Tile text. Production lowering runs
`GenerateWMMAGemmKernel` before Tile→ROCm/ROCDL, produces a gfx1151 HSACO, and
records adjacent Graph→Schedule→Tile→Target→backend digests; the runtime admits
the new typed matmul ABI. **ROCm outcome: parity validated for E2E-REAL-3.**
Host WSL exact-device descriptor launches agree numerically for aligned
`32x32x32` and ragged `17x19x23` cases, and the physical lit fixture proves no
Graph, Schedule, or launch-level matmul op survives. This does not promote the
route: E2E-REAL-4 must still record device-event throughput against the 8.02
TFLOP/s floor and same-run direct lane. NVIDIA and Apple schedules are not
inferred from this gfx1151 evidence.

## Cross-backend sync `E2E-REAL-PERFORMANCE-2026-08-05`

The first canonical run exposed a real schedule-loss defect: carrying only the
16x16 WMMA instruction tile produced 4.14 TFLOP/s. The shared Schedule/Tile and
descriptor contracts now carry a separate architecture-owned macro tile;
gfx1151 selects its already committed 32x64 shape, and the ROCm adapter derives
2x4 without embedding that spelling in shared IR. Runtime launch geometry
consumes the same descriptor value. **ROCm outcome: retain, not promote.** The
corrected 2048-cubed run reaches 18.29 TFLOP/s versus 18.34 direct (0.997x) and
2.281x the 8.02 floor. Aligned `64x64x64` and ragged `65x67x31` outputs are
bit-identical between routes. The WSL `/dev/dxg` packet records synchronized
resident host-wall timing, so it is valid comparison evidence but not
selector-grade device-event evidence:
[`../../../../benchmarks/baselines/rocm_gfx1151_e2e_real4_matmul_2026_08_05.json`](../../../../benchmarks/baselines/rocm_gfx1151_e2e_real4_matmul_2026_08_05.json).
Bare-metal timing can promote the route without another schedule redesign.
gfx1200/gfx1250 must provide their own macro tile and instruction-family
profiles and fail closed until their exact-device packets land.

## Cross-backend sync `E2E-REAL-SEMANTIC-KERNELS-2026-08-05`

The bounded canonical f32 softmax/reduction route now crosses real
Graph→Schedule→Tile boundaries. `schedule.softmax` and `schedule.reduce` bind
architecture, numeric policy, axis/kind, launch width, and durable SHA-256
identity; `ScheduledKernelArtifact` then feeds the exact Tile text to the
existing gfx1151 physical compiler without Graph re-entry. Static last-axis
softmax and arbitrary-axis rank-reducing sum/mean/max are lineage-complete.
Tampered policy fails closed. On the exact WSL-visible Radeon 8060S/gfx1151,
both scheduled descriptor launches agree numerically with NumPy. **ROCm
outcome: parity validated for the bounded E2E-REAL-5 slice; no selector or
performance promotion.** Existing f16/bf16→f32 reductions, f16 softmax, and
`keepdims=true` descriptors remain explicit Graph-owned routes because the
canonical `tessera.reduce` op currently requires same-element-type,
rank-reduced output. gfx1200/gfx1250 remain fail-closed and require their own
Schedule policy and exact-device evidence.

## Cross-backend sync `E2E-REAL-ATTENTION-2026-08-05`

`schedule.attention` now binds the shared static rank-4 online-softmax
recurrence, modifiers, launch contract, and architecture-owned backward-LSE
policy into one SHA-256 identity. The gfx1151 package consumes the exact
emitted `tile.attention_kernel` through TileToROCm/ROCDL/HSACO without Graph
re-entry and preserves `gfx1151_auto_128` with the per-shape saved/recompute
selection. **ROCm outcome: parity validated for E2E-REAL-5A.** On the exact
WSL-visible Radeon 8060S/gfx1151, the scheduled f16→f32 descriptor loads and
launches, and GQA plus causal-window ragged `Sq=17/Sk=19` output agrees with the
shared streaming-attention oracle. gfx1200/gfx1250 explicitly fail closed;
they require architecture-owned profiles and exact-device packets. Canonical
attention backward was the next shared family boundary.

## Cross-backend sync `E2E-REAL-ATTENTION-BACKWARD-2026-08-05`

`schedule.attention_backward` now binds dQ/dK/dV identity, the canonical
tensor loops, fixed two-split dK/dV reduction, launch workspace, modifiers, and
the gfx1151 LSE selection into one SHA-256 artifact. The gfx1151 consumer
compiles that exact Schedule→Tile program into the existing ordered five-entry
HSACO package without Graph re-entry. **ROCm outcome: parity validated for
E2E-REAL-5B.** On the WSL-visible Radeon 8060S/gfx1151, exact tests pass for
MHA/GQA/MQA, aligned and ragged shapes, causal window, bias, softcap, and all
three gradients. Saved-LSE operand indexing is now explicit in the direct Tile
materializer. gfx1200/gfx1250 remain fail-closed pending their own profiles and
device evidence; no schedule transfers from gfx1151.

## Cross-backend sync `E2E-REAL-5C-STATE-LINEAGE-2026-08-05`

The gfx1151 Lion VJP, factored/full Adafactor VJP, and sequence-mixer backward
launchers now enforce shared content-addressed logical-buffer lineage and
consume exact typed Schedule→Tile artifacts before HIP compilation. Lion and
Adafactor declare functional/no-alias versioned state transitions;
`schedule.sequence_mixer_backward` binds the checkpoint, chunk summary/prefix/
fill, reverse phases, workspace, and six fresh gradient outputs. Runtime
consumers no longer retain or reconstruct Graph-op metadata. **ROCm outcome:
parity validated for the bounded E2E-REAL-5C slice.** Exact gfx1151 Lion,
factored/full Adafactor, and gated/modified DeltaNet backward tests pass.
gfx1200/gfx1250 stay fail-closed pending profiles and exact-device evidence.

## Cross-backend sync `ROCM-TYPED-EXECUTABLE-PIPELINE-2026-08-07`

`ROCM-EXEC-PIPELINE-1` is **complete** for its bounded gfx1151 family set.
Production matmul, softmax, reduction,
paged-KV, attention forward/backward, and MoE dispatch now select typed
semantic-family plugins through `tessera-rocm-executable`; Python no longer
assembles those generators, conversions, or packaging passes as comma-separated
strings. The pipeline stamps its Tile producer, `tessera_rocm` Target-IR
consumer, `rocdl_hsaco` code generator, architecture, family, and terminal
artifact level as module attributes. Binary construction runs runnable
`tessera_rocm.async_copy` lowering before Target-IR-to-ROCDL conversion and
lowers its counter-class threshold to a real `rocdl.s.waitcnt` before the workgroup
barrier. The executable boundary now rejects every surviving `tile.*` or
`tessera_rocm.*` op, `llvm.mlir.undef`, and `.contract` symbol. Compatibility
aliases stop at typed Target IR: they no longer fabricate marker calls or
replace live results with undefined values. Target-only `buffer_load` and
`ds_read_tr` contracts remain inspectable but cannot be mislabeled executable.
gfx1151 passed the 53-test ROCm lit suite and 151 focused runtime/family tests.
gfx1200/gfx1250 remain fail-closed.
The ROCm-only compiler driver now registers the same ROCDL translation and
target interfaces as production `tessera-opt`; the executable async-copy
fixture reaches `gpu.binary` through both drivers instead of advertising a
pipeline that only the production driver could serialize.
The follow-on family expansion is also complete for the existing physical
scalar activation/unary/binary/where/compare/predicate/logical/bitwise lanes;
Cholesky/triangular-solve/LU/QR/SVD; block-sparse attention/top-k and
SpMM/SDDMM; and recurrent-cell, linear-attention, selective-SSM forward/
backward, and DeltaNet. Each caller now selects a named semantic plugin and the
C++ pipeline owns its generator and binary packaging order. Exact gfx1151
validation passed **607/607** scalar tests and **188/188** solver/sparse/
sequence tests.

The final runtime-family migration is complete. Normalization, arg-reduction,
scan, gather/scatter, f32 GEMM/batched GEMM, sort, RNG, losses, optimizers,
quantization/packing, EBM, DFT, ALiBi/RoPE, Clifford, DSpark, and MLA decode now
also select closed semantic plugins. The generic
`_build_rocm_elementwise_hsaco(pass_name, ...)` helper and the last manually
composed `gpu-module-to-binary` strings have been deleted. A structural unit
gate forbids either escape hatch from returning. The complete migrated
`test_rocm_*compiled.py` selection passed **1534** tests. The temporary-build
spectral-image omission that separately blocked 15 TSOL composite tests is now
closed by `TSOL-GFX1151-FUSED-BATCH-2026-08-08`; all 15 execute against the
selected build's prebuilt image. The 53-test ROCm lit suite passed **53/53**,
and the dedicated DFT/no-escape-hatch plus audit/registry selection passed
**185/185**.
`build.hidden/` was not used or modified.

## Cross-backend sync `TSOL-PACKED-FUSION-2026-08-08`

`tessera.scheduled_spectral.v5` now hashes the compound fusion topology in
addition to child FFT digests. On gfx1151, even-length spectral convolution,
STFT, and ISTFT bind N/2 child plans and execute device-resident
RFFT→complex-multiply→IRFFT, frame/window→RFFT, and IRFFT→window/overlap-add
programs through `tessera.rocm.spectral_composite.v5`. Full-complex N-point
intermediates and their workspace allocations are removed from these paths.
Odd windows retain support through a separately named and hashed full-complex
fallback; they cannot be mistaken for packed execution. gfx1200/gfx1250
remain fail-closed and inherit no gfx1151 schedule.

**ROCm verdict: retain, not promote.** The package rebuilds for gfx1151 and all
15 exact-device composite tests pass on the WSL-visible Radeon 8060S. The
20-sample host-wall packet records numerical error, artifact/child digests,
physical lengths, workspace, dtype, axis, and normalization policy in
[`../../../../benchmarks/baselines/tsol_packed_fusion_gfx1151_2026_08_08.json`](../../../../benchmarks/baselines/tsol_packed_fusion_gfx1151_2026_08_08.json).
Baseline f32 maxima are `2.86e-6` for convolution, `1.23e-6` for STFT, and
`8.77e-6` for ISTFT. WSL host-wall medians are regression evidence only;
promotion still requires bare-metal device-event timing and a same-run
full-complex/rocFFT comparison.

The follow-on `gfx1151_stockham_bluestein_v6` path now folds real-pair packing,
all N/2 Stockham stages, and Hermitian post-processing into one fused-LDS RFFT
dispatch. IRFFT likewise folds Hermitian preprocessing, inverse Stockham, N/2
scaling, and paired-real stores into one dispatch, removing both the separate
pre-kernel and device-to-device copy. The residency identity is
`persistent_device_plan_fused_lds_hermitian_batch`; **58/58** focused
Schedule→Tile and exact-device FFT/TSOL tests pass. This is a correctness and
launch-count retain verdict only. Fresh bare-metal gfx1151 device-event timing
and application-kernel instrumentation comparison remain required before
performance promotion. gfx1200/gfx1250 stay fail-closed.

## Cross-backend sync `TILE-SYNC-RECONCILE-2026-08-10`

`tile.async_copy`/`tile.wait_async` now have one declared contract (ODS dual
form, `TileOps.td`): typed `!tile.async_token` SSA is production; the legacy
`tile.barrier_id`/`tile.depends_on`/`stage` grouping keys are the declared
compatibility envelope, optional and conservative on absence. New shared
diagnostic `TILE_ASYNC_STAGE_NEGATIVE` (a present `stage` must be >= 0).
**ROCm outcome: host-free compiler parity and exact gfx1151 correctness
validated; performance policy unchanged.** The legacy attr
envelope is load-bearing exactly on this backend's lanes (`TileToROCM.cpp`,
`ROCMWaveLdsPipeline.cpp`, `GenerateWMMAGemmKernel.cpp`), and the new check
now runs on every `tile.async_copy` those pipelines emit. PR #544's required
host-free LLVM/MLIR 23 ROCm compiler lane passed, closing dialect, verifier,
and lowering parity. The follow-up WSL-visible gfx1151 run then passed the
global→LDS round trip, five LDS-staged WMMA cases, five two-stage pipelined
WMMA cases, and the bit-identical via-Tile/production comparison (**16/16**
focused tests including four structural checks). The accompanying compiler
benchmark remains explicitly `host_compiler_only`; no selector or device-
performance promotion is inferred. The attributed Mac-only ROCm cases also
passed in a **25/25** owning-WSL cohort. Full commands and provenance are
recorded in the completed
`docs/audit/compiler/archive/STRIX_HALO_WORKLIST_2026-08-10.md`.

## TILE-SYNC-TYPED-2026-08-15 — shared Tile sync ABI assessment (PR #566)

**Parity validated on-device.** The changed operand-segment ABI
(`tile.mbarrier.wait`) is NVIDIA vocabulary with no ROCm consumer; ROCm's
token lane (`tessera_rocm.async_copy` → `!tessera_rocm.token` → `wait`) and
the shared `tile.async_copy`/`tile.wait_async` dual-form contract are
unchanged. Evidence on the changed tree: the full compiled gfx1151 device
sweep — 1,569 tests (every `_compiled` lane + staged global→LDS/LDS-WMMA +
canonical GEMM matrix) — green; `check-tessera-rocm` green modulo the
pre-existing `gfx1151_philox_distributions.mlir` failure (fails on clean
main; independent of this change).

## REF-TIER-OPS-2026-08-15 — reference-tier op registration assessment (PR #568)

PR #568 registered ten new public operations through the canonical op catalog
and the primitive coverage registry — `tridiagonal_solve` (Thomas recurrence,
PDE plan §III.1 / TSOL-A1) and the nine-op coalition-lattice family
(`game_subset_zeta`, `game_subset_mobius`, `game_superset_zeta`,
`game_superset_mobius`, `game_coalition_marginal`, `game_semivalue`,
`game_boltzmann_value`, `game_coalition_excess`, `game_mex`). Op registration
is a shared contract, so this queue records the outcome per AGENTS.md
"Cross-backend work coordination"; PR #568 itself landed without these records.

**2026-08-16 physical follow-up (`REF-TIER-PHYS-2026-08-16`): host-free lane
implemented for the solver and four lattice transforms.** The solver selects
`cooperative_lds_pcr_v1`: one workgroup per system, logarithmic parallel cyclic
reduction, fp64 LDS recurrence state, and one final fp32 store. It deliberately
does not map Thomas' serial row sweep onto a wave. The four subset/superset
zeta/Mobius operations share one parameterized, ascending-bit Yates
Schedule→Tile→`tessera_rocm` consumer and one fp64-LDS generator. gfx1200 and
gfx1250 remain fail-closed. Host-free Target/generator fixtures are the current
evidence; exact-device gfx1151 correctness and performance packets remain
required before promotion. Coalition marginal, semivalue, Boltzmann value,
excess, and MEX remain reference/composition follow-ups rather than five new
one-off emitters.
The PCR artifact now has an explicit per-equation i32 status ABI for invalid or
non-finite pivots. The coalition consumer enforces matching workgroup identity
and the 64 KiB fp64-LDS envelope before generation; these strengthen host-free
correctness but do not constitute a device packet.

## APPLE-SCHEDULED-REDUCE-NAN-2026-08-16 — shared reduce NaN semantics (PR #571)

**Shared contract changed; assess before relying on extrema reductions.** The
synthesizer's reduce vocabulary (`compiler/fusion_core.py::_PW_REDUCE_KINDS`)
emitted `max(acc, v)` / `min(acc, v)` for `amax`/`amin`. Metal's `max`/`min` are
IEEE maxNum/minNum-style and **suppress** a NaN operand, so the emitted kernel
disagreed with the table's own numpy reference (`a.max(-1)`, which propagates)
and with the `nan_mode = "propagate"` the reduce Schedule artifact declares.
With the `-INFINITY` seed an all-NaN row reduced to **`-inf`** — missing data
silently becoming a finite extreme. The accumulators now propagate explicitly.

**ROCm outcome: not applicable — no consumer.** `_PW_REDUCE_KINDS` supplies MSL
accumulate expressions and its only kernel consumer is
`compiler/emit/apple_msl.py`; the gfx1151 reduction path is the HIP
`tile.reduce_kernel` lowering, which is unaffected. gfx1151's arbitrary-axis
scheduled reduction is explicitly retained (Apple admits last-axis only) and a
test asserts the Apple bound does not narrow it. No ROCm dtype, schedule, ABI,
or evidence changes. If a future ROCm reduce emitter reuses this table, it
inherits the corrected semantics rather than the bug.

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

**ROCm-specific note.** rocSOLVER covers `eigh`/`inv`/`solve`; the same
delegated-vs-generated question as NVIDIA applies. gfx1151 is the box that could
actually evaluate a lane, so any first proof belongs there.

## Cross-backend sync `NVIDIA-BARRIER-AT-BIRTH-2026-08-21`

**Parity assessed; NVIDIA-only physical follow-up.** The shared Schedule/Tile
roles remain valid, but the new binding consumer is NVIDIA TMA/mbarrier. ROCm
keeps its architecture-owned LDS/token synchronization and no gfx schedule or
exact-device evidence transfers.

## Cross-backend sync `TARGET-IR-CONFORMANCE-2026-08-21`

**NVIDIA-only conformance closure.** NVIDIA's Target-value restriction and
registered-dialect test do not alter ROCm's MFMA/ABI surface or gfx1151
evidence. ROCm remains independently validated on its own build/device lanes.


## Cross-backend sync `NVIDIA-SPECTRAL-PHILOX-JVP-2026-08-22`

**Outcome: parity already validated; no ROCm physical change.** NVIDIA joined the
existing shared spectral and native-product contracts with CUDA-owned cuFFT and
Philox consumers. gfx1151 retains its Stockham/Bluestein, compound spectral,
spectral backward, and Philox packages and exact-device evidence. No CUDA plan,
workspace, or schedule is transferred to ROCm.
