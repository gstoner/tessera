---
last_updated: 2026-09-16
audit_role: reference
scope: EBM sampling loops as native GPU packages; nonlinear and manifold energies
---

# The EBM sampling loop as a native GPU package

Owner: [W4-PRODUCT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#w4-product-1)
with [AD-SOLVER-IFT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1);
acceptance from the [GA/EBM review](GA_EBM_ARCHITECTURE_REVIEW.md) §"an
energy is a typed program". Sync keys `EBM-NATIVE-QUADRATIC-2026-09-16`
(CPU lane), `EBM-NATIVE-GPU-2026-09-16` (device lane) and
`EBM-NONLINEAR-MANIFOLD-2026-09-16` (N1 nonlinear energies, M1 sphere).
Written after the CPU-lane slice landed (`tests/unit/test_ebm_native_langevin.py`)
and after driving that same loop as far as the existing device routes take it;
every "today" claim below is a measured stop, not a reading of prose.

> **Updated 2026-09-16, same day — G2 and T1 landed, and G2 landed by a
> different mechanism than §3.3 scoped.** The cooperative kernel is produced
> by a new pass over the *already lowered* loop
> (`tessera-row-program-to-gpu`, §3.3), not by a `schedule.ebm_langevin` /
> `tile.langevin_kernel` contract: once the paired autodiff pass, the EBM
> lowering, `tessera-to-linalg` and inlining have run, the loop function *is*
> a `[rows, features]` row program in linalg, and that program is the
> scalar body the Tile route would otherwise have carried as a region. The
> row-program emitter maps it directly (one block per row, one lane per
> feature, the K-step `scf.for` and the Philox draw inside the kernel, state
> in registers, ordered shared-memory reductions). Measured on gfx1151,
> gfx1201 and sm_120: one launch per loop, bit-exact with the declared
> policy for every case in the packets. §3.3 below is rewritten to what
> shipped; §3.2's G1 (serial residency through the native-tape route) was
> **not** built — the cooperative route made it unnecessary for this loop
> and it stays scoped only as the fallback for programs outside the
> row-program envelope. `tessera-opt` now registers the EBM and Clifford
> dialects and passes when built with them (T1), so the whole chain is one
> driver invocation, and the assertions-enabled driver on Tajasarus
> falsified three dialect promises the NDEBUG fleet had run green through,
> and sm_120 exposed libdevice's approximate `sqrtf` default (see §6).

## 1. What the CPU slice established, and what it did not

The loop is now a program the compiler owns end to end on the CPU lane:

```
@quadratic_energy(y, x)   tessera.autodiff = "reverse"       (Graph IR)
   └─ tessera-autodiff-paired ──▶ @quadratic_energy__bwd(y, x, cot) -> (dy, dx)
scf.for K { tessera_ebm.langevin_step(y, key, x) }
   └─ tessera-ebm-lower-langevin ──▶ call @E__bwd; y - η·g + √(2ηT)·z; key+[0,1]
                                     z from Philox-4x32-10 / Box-Muller in a linalg.generic
   └─ tessera-to-linalg, one-shot bufferize, DPS, LLVM (libtessera_jit)
```

One native invocation executes all K steps; noise is generated inside the
compiled loop; the gradient is derived, not written. Three shared-compiler
gaps were closed to get there (`sub` adjoint; `unsqueeze`/`broadcast` linalg
lowering; the JIT's DPS rewrite now follows intra-module calls).

What it did **not** establish, and what this document plans:

1. **A device package for the loop.** The Clifford GPU route
   (`native_clifford_gpu.py`) works because a rank-1 Clifford op expands to
   *scalar* arithmetic that a hand-written per-thread skeleton can host. The
   Langevin body is not scalar: `@E__bwd` is tensor-level linalg with a
   row reduction (`0.5·Σ(x−y)²` reduces over features), and the loop carries
   a whole state tensor across steps. The skeleton route cannot host that.
2. **Nonlinear and manifold energies.** `manifold = "sphere"` and
   `"bivector"` fail closed in `tessera-ebm-lower-langevin`; only the
   Euclidean integrator lowers. The reference integrators exist in
   `python/tessera/ebm/geo_sampling.py` and the GA/EBM review asks for
   "explicit singularity/branch handling" before promotion.

## 2. Measured position of the existing device routes (2026-09-16)

Three routes were tried on the Mac with the EBM loop module produced by
`native_langevin.langevin_loop_module((4, 8), eta=0.1, temperature=0.7, steps=3)`.

| Route | Mechanism | Result on the EBM loop | Why |
|---|---|---|---|
| Arena skeleton (`native_clifford_gpu.py`) | Python writes a per-thread `gpu.func`; a dialect driver expands one op; the arena pipeline folds tensors to scalars | **Not applicable** | The body is tensor-level linalg with a reduction and an `scf.for` carrying a tensor; there is no single scalar op to expand |
| Serial native tape (`tessera-native-tape-to-gpu`, the SSD serial lane in `native_ssd._prepare`) | mlir-opt bufferizes the loop function (`one-shot-bufferize` + `buffer-results-to-out-params` + `convert-linalg-to-loops`); the Tessera pass wraps the single host function into a `gpu.module` kernel with device-pointer args at grid (1,1,1) block (1,1,1); `build_native_gpu_storage` packages it | **Rejected at the pass's admission check** — `native tape GPU requires an isolated static f32/f64 buffer product with bounded for/if control …` | Two precise reasons, both in `NativeTapeToGPUPass.cpp:46-51`: (a) the module must hold **exactly one** function, and the loop module holds four (`@quadratic_energy`, its `__bwd`, the energy entry, the loop); (b) the module must carry exactly one owner marker (`tessera.ssd.source`, `tessera.ann.source`, the AD product pair, `tessera.native_result_program`, or `tessera.source_state`), and no EBM marker exists |
| Cooperative Tile lowering (`--tessera-schedule-to-tile=ssd-gpu=<backend>`, `NativeSSD.h`) | The Schedule op carries the loop contract; the Tile lowering emits a block-per-(row, column) kernel with lanes owning state elements, LDS + barriers for the reduction | **No EBM Schedule op exists** | `tessera_ebm.langevin_step` is an EBM-dialect op with a symbolic `energy_fn`; nothing projects it into `schedule.*`, so `PMPasses` has nothing to lower |

Two tooling facts shaped the experiment and are gaps in their own right:

* The front of the chain needs **three drivers** (`tessera-opt` for the
  paired autodiff and `tessera-to-linalg`, `ts-ebm-opt` for the EBM lowering,
  `mlir-opt` for bufferization), and `tessera-opt` prints Tessera ops in
  custom syntax that `ts-ebm-opt` cannot parse — only `--mlir-print-op-generic`
  round-trips. `libtessera_jit` avoids this because it links every dialect
  in-process. `tessera-opt` registers neither the EBM nor the Clifford
  dialect even when they are built.
* `tessera-opt` already registers the upstream **linalg → scf.parallel →
  gpu.launch → outlining** spine (`kEmitSpinePrefix`, `tessera-opt.cpp:249`),
  labelled "EMISSION ONLY … launch stays hardware-gated" in June. Launch is no
  longer gated: `build_native_gpu_storage` + `NativeTensorCall` launch any
  `gpu.module` with the storage ABI on gfx1151, gfx1201 and sm_120. That
  spine is therefore a third candidate, not just an inspection aid.

## 3. Architecture

### 3.1 Principle: one lowering authority, three physical routes

Decision #31 (one production lowering per boundary) and #28 (measured
selection among implementations) settle the shape. The EBM lowering
(`tessera-ebm-lower-langevin`) stays the **only** authority for what a
Langevin step means — gradient call, integrator, noise policy, key advance —
and produces tensor-level IR. What differs per target is only how that IR
becomes a kernel:

```
                     tessera_ebm.langevin_step (+ energy_fn, captures)
                                       │  tessera-ebm-lower-langevin (one authority)
                                       ▼
            scf.for over tensors: call @E__bwd, arith, linalg.generic(noise)
          ┌──────────────────────┬──────────────────────┬────────────────────────┐
          ▼                      ▼                      ▼                        ▼
   CPU JIT (landed)     G1 serial device kernel   G2 cooperative kernel    Apple arena (MSL)
   libtessera_jit       native-tape-to-gpu        linalg→parallel→gpu      apple_native_arena
   one call, K steps    grid 1, one call, K steps rows→blocks, feats→lanes  (later; same IR)
```

Every route validates by **replay** (`replay_arena_ir` byte-compares the
packaged IR with a fresh lowering — the same discipline the heap, SSD and
Clifford packages use) and reports the route in its packet (Decision #12).

### 3.2 G1 — serial device residency (the acceptance's "no per-step host transfers" on a device)

Purpose: prove the loop is device-resident and bit-exact with the declared
policy on gfx1151/gfx1201/sm_120 with the smallest possible change, before
any parallel mapping. Serial means one thread executes the whole loop; it is
a correctness route (as the SSD serial lane is), never a performance claim.

Changes, each small and testable:

1. **Isolate the loop function.** After `tessera-ebm-lower-langevin`, inline
   `@E__bwd` (and `@E` if the energy entry is exported) into the loop function
   and drop the dead originals — `mlir::inlineCall` on the call sites the
   lowering just emitted, gated on a pass option `isolate-loop=true`. The
   native tape pass then sees exactly one function. The JIT keeps calls
   (its DPS rewrite now follows them); only the device route inlines.
2. **An EBM owner marker.** The lowering stamps
   `tessera.ebm.source = "<serialized Graph-level module>"` on the module
   when isolating, and `NativeTapeToGPUPass` admits it as a sixth owner kind
   (the `ssd` branch is the template: one function, static buffers, bounded
   `for/if`). The marker is what lets `validate()` re-derive the kernel from
   the source and byte-compare it — the replay contract.
3. **Admission audit of the body.** The pass's whitelist ("bounded for/if
   control, static f32/f64 buffers, ≤ 4096 temporary bytes") must be checked
   against what the loop actually contains after `convert-linalg-to-loops`:
   integer `arith` on i32/i64 (Philox), `arith.mului_extended`, `math.log`,
   `math.cos`, `math.sqrt`, `index_cast`, and the key tensor `memref<2xi64>`.
   Expected finding: the integer/`mului_extended` ops are new to that pass
   and its op whitelist needs them; the NVVM/ROCDL lowerings handle them
   (the Clifford packets already execute `mului_extended`-free scalar code;
   `math.*` lowers through `convert-math-to-llvm`, as the arena pipeline does).
4. **Python:** `native_langevin.package_ebm_langevin_native(shape, …, target)`
   mirroring `package_clifford_native`, a `run_*` through `NativeTensorCall`
   (host arrays in/out via the driver handle, as `CliffordDeviceProgram`
   does), execution-matrix rows `rocm` / `rocm_ebm_langevin_native_compiled`
   and `nvidia_sm120` / `nvidia_ebm_langevin_native_compiled`, and a recorder
   writing per-device packets.

Acceptance G1: fixed-key samples bit-exact with `reference_langevin_loop` for
K ∈ {1, 5, 12} on all three devices; the packet records that the loop ran as
one launch (the binding's launch count) and that no host buffer was touched
between steps (the kernel's only host contact is the argument copy-in and
result copy-out). Not a performance route; no promotion.

### 3.3 G2 — cooperative kernel (rows to blocks, features to lanes) — landed 2026-09-16

Purpose: the loop at real batch sizes, as one launch. What shipped is the
**row-program emitter**, `src/transforms/lib/RowProgramToGPUPass.cpp`
(`--tessera-row-program-to-gpu=backend={nvidia,rocm} entry=<fn>`), consuming
the lowered loop after
`--tessera-autodiff-paired --tessera-ebm-canonicalize --tessera-ebm-lower-langevin
--tessera-to-linalg --inline --convert-elementwise-to-linalg --canonicalize --cse`
in one `tessera-opt` invocation:

* **Input envelope (fails closed outside it).** One entry `func.func` whose
  operands and results are `[R, F]` f32 tensors (lane values), `[R, 1]` /
  `[R]` f32 tensors (row values) and small integer vectors of ≤ 8 elements
  (uniform values, e.g. the Philox key); a body of parallel
  `linalg.generic` (identity or row-broadcast maps), feature-axis
  `linalg.reduce`, `tensor.empty`/`linalg.fill`, `expand_shape`/`collapse_shape`
  between row shapes, `tensor.extract` on a uniform vector with a constant
  index, splat constants, and `scf.for` over any mix of those. `F ≤ 1024`
  (one lane each; lanes = next power of two ≥ F, spare lanes masked). Any
  call is refused with "inline every call before lowering"; anything else
  names the op it cannot classify.
* **Mapping.** One `gpu.func @row_program` in `gpu.module @native_row`,
  `known_block_size = [lanes, 1, 1]`, one block per row. Lane values live in
  registers across the whole program including every `scf.for` iteration
  (the loop is carried as `iter_args` of scalars, so a K-step Langevin loop
  is *one* kernel with the state, the gradient and the noise never leaving
  registers). Arguments are `!llvm.ptr<1>` per tensor plus the index scratch
  the arena pipeline expects; loads and stores are guarded by the lane mask
  (lane values) or the block leader (row / uniform values). The
  `tile.alloc_shared` marker is emitted so `build_native_gpu_storage`'s
  shared-memory sizer sees the kernel it already knows.
* **Reductions.** A feature-axis `linalg.reduce` becomes an **ordered**
  fold: every active lane stores its element into a
  `@row_reduction` array in address space 3, `gpu.barrier`, the leader folds
  the combiner over lanes **in index order** into a scalar and writes it
  back, `gpu.barrier`, every lane reads the broadcast, `gpu.barrier`. The
  declared reduction order is therefore "sequential over the feature index",
  the same order a host `for` loop in f32 produces — which is why the row
  normalization proof below is bit-exact, not tolerance-based.
* **Noise.** The EBM lowering's Philox-4x32-10 / Box–Muller `linalg.generic`
  is an ordinary lane body to the emitter: `linalg.index 0/1` become
  `(block_id, thread_id)`, the ten `arith.mului_extended` rounds and the
  f64 `math.log`/`math.cos` stay inside the loop, and the key words are
  uniform values carried through the loop (canonicalization hoists the
  invariant `key[0]`, so the packaged loop carries `(state f32, key[1] i64)`).
* **Provenance.** The module carries `tessera.row_program.{source, entry,
  rows, features, backend}`; the tensor contract producer merges into that
  attribute dictionary instead of requiring a bare module, so the package
  records the lowered program it was built from.

Python: `compiler/native_row_program.py` (`row_program_kernel` →
`row_program_device_source` → `bind_row_program` → `row_program_device`, a
cached `HostArrayProgram`) is the seam every row-shaped domain loop packages
through; `ebm/native_langevin.py` builds the Langevin module and its specs
on top of it (`langevin_device_source`, `native_langevin_loop_device`,
`package_ebm_langevin_native`); `runtime.launch` rows
`rocm` / `rocm_ebm_langevin_native_compiled` and
`nvidia_sm120` / `nvidia_ebm_langevin_native_compiled`.

Measured (packets in `benchmarks/baselines/ebm_langevin_native_gpu_20260916/`,
`tests/unit/test_ebm_native_langevin_gpu.py`): on gfx1151 (Princess-Luna),
gfx1201 (Tajasarus, `TESSERA_ROCM_CHIP=gfx1201`, assertions-ON driver) and
sm_120 (Super-Bear) the loop is one launch and **bit-exact** with
`reference_langevin_loop` for K ∈ {1, 4, 5, 8, 12}, F ∈ {5, 8, 33, 100, 1024},
T ∈ {0, 0.3, 0.5, 0.7} — worst absolute error 0 in every row, including the
f64 `log`/`cos` of the noise (the libm risk in §6 did not materialize on any
of the three devices). The T = 0 loop is the plain descent. The quadratic
gradient is elementwise (the sum-reduce adjoint is a broadcast), so the
Langevin kernel carries no reduction; the reduction path is proven
separately by a row-normalization program (`x / sqrt(Σ_f x²)`, F up to
1024) that matches the sequential-order f32 fold bit-for-bit on the same
three devices.

What this is not: a performance claim. The route's per-call host transfers
are correctness-only; the Python-emitted `rocm_ebm_langevin_compiled` /
`x86_ebm_langevin_compiled` / Apple kernels remain their lanes until the
dispatch/allocation/traffic/kernel-time comparison the GA/EBM review asks for
is recorded with `route` and `latency_source`. The envelope is also narrow
on purpose: rows wider than 1024 features (a second reduction level), row
programs with data-dependent control flow, and energies whose gradient is
not a row program (cross-row coupling) fail closed and would go through
§3.2's serial route or a real Tile contract.

Why not the Tile contract scoped before: `schedule.ebm_langevin` with a body
region would have carried exactly the program the emitter now reads from
linalg, and would have needed its own body lowering — a second scalar
emitter beside the one the Tile kernels use (Decision #31). The row-program
pass is the smaller authority: it adds one pass over upstream dialects, no
new ops, and the same pass serves any `[rows, features]` domain program (the
GA family's batched products qualify once their ExpandProductTable output is
inlined). If a future energy needs multi-level reductions or cross-row
coupling, the Tile contract is the next step and this pass is its
measurement baseline.

Acceptance G2, as met: bit-exactness (samples *and* the ordered reductions);
one launch per loop; the packets. As not met: the separate overhead
measurement against the Python-emitted lanes — promotion waits for it.

### 3.4 Apple

The Apple GPU route is the MSL arena (`apple_native_arena.py`,
`--tessera-tile-buffer-arena=emit-apple-msl=true`), not the NVVM/ROCDL
pipeline. G1's isolated loop function is the input it needs; the open work
is that the arena emitter takes a Tile-level buffer program, not a
bufferized scf loop. Route through G2b's `tile.langevin_kernel` once it
exists; do not write a third emitter (`CLAUDE.md` §"the real Apple gap").

## 4. Nonlinear and manifold energies

> **Updated 2026-09-16 — N1 and M1 landed; this section is now a record of
> what shipped, not only a plan.** Three energies (quadratic, Huber,
> softplus) run through the same integrator on the CPU JIT lane and as
> cooperative kernels on the device, and `manifold = "sphere"` has a native
> integrator with a per-row status word. §4.1's admission rule held: the
> lowering is energy-agnostic, so each new energy was a *differentiation*
> question, and the one gap it found was `softplus`, whose adjoint was a
> `custom_adjoint_call` placeholder (a host VJP, i.e. a per-step transfer).
> It now has a native adjoint, `dy · sigmoid(x)` — §4.1's own stability note,
> implemented — and a stable linalg lowering, `max(x,0) + log1p(exp(-|x|))`.
> Huber needed no new adjoint, as predicted, but its backward yields two
> results from one body, which the row-program emitter had to learn.
> §4.2's contract shipped intact: entry precondition on the squared norm,
> retraction underflow keeping the previous state, both reported per row and
> never silently repaired, with the dot products and norms declared as
> sequential f32 row sums — which the emitter's ordered fold reproduces, so
> the device agrees with the host sequential fold rather than merely within a
> tolerance. §4.3 (bivector) is unchanged and still refuses.


### 4.1 Nonlinear energies: nothing manifold-specific is needed

The Euclidean integrator is energy-agnostic: the lowering calls `@E__bwd`
whatever `E` is. A nonlinear energy is therefore a **differentiation
coverage** question, not a lowering one. The quadratic case already exposed
that coverage gap twice (`sub`; `unsqueeze`/`broadcast`). The next energies
to admit, chosen because each exercises one more adjoint on the Graph IR
path and each has an Apple or ROCm reference kernel to compare against:

| Energy | Graph ops on the gradient path | Adjoint status today | Branch / singularity handling |
|---|---|---|---|
| Huber / smooth-L1 on `x − y` | `tessera.loss.huber` | adjoint exists (`HuberLossOp::buildAdjoint`) | the kink at `|r| = δ` is a `compare_scalar` + `masked_fill`, already the pattern ReLU uses |
| Softplus / log-cosh energy | `softplus`, `log`, `cosh` | `tanh`/`sigmoid`/`gelu` have adjoints; `softplus`/`cosh` need checking | overflow guard: the adjoint must use the stable form `sigmoid(r)`, never `exp(r)/(1+exp(r))` |
| Softmax-normalized energy (contrastive rows) | `tessera.softmax`, `reduce` | adjoints exist | the row maximum subtraction is inside the softmax adjoint already |
| Energy with a learned `W`: `0.5·‖x − Wy‖²` | `tessera.matmul` | adjoint exists; this is the paired-matmul row | none |

Rule for admission (fails closed, as today): the paired autodiff pass must
produce `@E__bwd` with **no** `tessera.custom_adjoint_call` in it — a
placeholder means the gradient would round-trip through the Python VJP
registry, which is a host transfer per step. The EBM lowering checks for the
placeholder and refuses.

### 4.2 Sphere: tangent projection + retraction

Reference (`geo_sampling.sphere_langevin_step`): with `x` unit-norm per row,
`ξ ~ N(0, I)`,

```
g_t = g − ⟨g, x⟩ x            (project ∇E to the tangent plane)
ξ_t = ξ − ⟨ξ, x⟩ x            (project the noise likewise)
x'  = (x − η g_t + √(2ηT) ξ_t) / ‖x − η g_t + √(2ηT) ξ_t‖   (retraction = normalize)
```

Lowering: three per-row dot products and one norm, i.e. **row reductions
over the feature axis**, then elementwise work. That is precisely what G2b's
kernel provides and G1's serial kernel trivially provides; on the CPU lane
it is `linalg.reduce` + broadcast — every piece already lowers after this
slice's `unsqueeze`/`broadcast` fix.

Singularities and their contract (fail closed, Decision #21a):

* **Entry precondition** `|‖x‖ − 1| ≤ 1e-3` per row. The reference asserts
  it on the host. Natively: the kernel writes a per-row status word (the
  pass already has `status-buffer` machinery for structured assertions), and
  the package refuses the result when any row violated it; no silent
  renormalization on entry.
* **Retraction at the origin.** If `‖x − η g_t + …‖` underflows (a step that
  cancels `x` exactly, or a huge η), normalization divides by ~0. Contract:
  `‖·‖ < ε` (ε = 1e-12 in f32 squared-norm terms) sets the row status and
  keeps the previous `x`; the reference gets the same guard so both agree.
* **Projection when `⟨g, x⟩ x` dominates.** Numerically benign (a
  subtraction of two O(‖g‖) terms), but the *order* of the dot-product
  reduction must be declared: sequential over features in G1/CPU, the
  block-reduction order in G2b, recorded in `numeric_policy` and matched by
  the reference's `np.float32` accumulate; the parity tolerance is 1e-5, not
  bit-exactness, for the projected quantities (the noise itself stays
  bit-exact).
* **Temperature 0** must reduce to projected gradient descent with
  retraction; a T = 0 fixture checks `‖x'‖ = 1` and monotone energy decrease
  for the quadratic energy on the sphere.

`manifold = "sphere"` therefore lowers as: G1/CPU immediately (reductions
already lower), G2b when the Tile contract lands. The per-row status word is
the new piece.

### 4.3 Bivector (grade-restricted Lie-algebra sampling)

Reference (`geo_sampling.bivector_langevin_step`): state is a multivector
constrained to grade `k` (default 2, so(3) in Cl(3,0)); both the gradient
and the noise are grade-projected, then the Euclidean affine step applies:

```
g_k = grade_k(∇E)      ξ_k = grade_k(ξ)      s' = s − η g_k + √(2ηT) ξ_k
```

Lowering: the grade projection is exactly the `tessera_clifford.grade` op
the Clifford slice lowered (a compile-time keep-mask: `UnaryMapPattern`,
`Unary::Grade`). The EBM lowering for `manifold = "bivector"` emits
`clifford.grade(grade=k)` on the gradient and on the noise before the affine
update — the EBM dialect depends on the Clifford dialect for this integrator
only, which is a build-time dependency `TesseraEBM → TesseraClifford` gated
on `TESSERA_BUILD_CLIFFORD_BACKEND` (both are ON fleet-wide since this
stream). No new arithmetic, no singularity: the projection is linear and the
step stays in the subspace by construction (the reference asserts the input
grade; natively the input-grade check is a status word like the sphere's
norm check).

Two facts to keep honest: this integrator samples on the **Lie algebra**
(grade-k multivectors), not on the group — mapping to rotors is `exp`, which
has no native lowering yet (`GA-NATIVE-FAMILY-2026-09-16` listed `exp`/`log`
as open) — and the state layout is `[..., 2^n]` coefficients, so a batch of
rotors is the batched Clifford tensor the W6.4 lowering already handles.

### 4.4 What stays closed

* `manifold` values outside {euclidean, sphere, bivector} — the enum is
  closed by `EBM_ManifoldAttr`.
* Energies whose `__bwd` needs a `custom_adjoint_call` (host VJP).
* Dynamic shapes (the loop nest and Philox counter need static extents; the
  reference keys the counter on the flat index).
* `exp`/`log` of multivectors (rotor sampling on the group).

## 5. Slices and acceptance, in order

| Slice | Deliverable | Acceptance | Fleet |
|---|---|---|---|
| G1 | isolate-loop inlining + `tessera.ebm.source` owner + admission audit; `package_ebm_langevin_native`; rows for `rocm` and `nvidia_sm120`; recorder | bit-exact K ∈ {1,5,12} on gfx1151, gfx1201, sm_120; one launch per loop | Princess-Luna, Tajasarus, Super-Bear |
| M1 | **landed 2026-09-16**: `manifold = "sphere"` in `tessera-ebm-lower-langevin` (tangent projection, retraction, per-row i32 status word ORed across the loop) on the CPU JIT lane and as a cooperative kernel | ‖x'‖ = 1 per row, T = 0 monotone descent, parity with the reference for every energy, status words equal, entry violation flagged on the offending row only | Mac, gfx1151, gfx1201, sm_120 |
| M2 | `manifold = "bivector"` via `clifford.grade` on gradient and noise; EBM → Clifford build dependency | parity with `bivector_langevin_step` on Cl(3,0) grade 2; state stays grade-2 over 100 steps | same |
| N1 | **landed 2026-09-16**: Huber and softplus energies through the paired pass with no placeholder; `softplus` gained its native adjoint (`dy · sigmoid(x)`) and a stable lowering | each energy matches its own numpy oracle and descends on its own compiler-derived gradient; no `custom_adjoint_call` survives in any kernel | Mac, gfx1151, gfx1201, sm_120 |
| G2 | **landed 2026-09-16** as the row-program emitter over the lowered loop (§3.3), not a Schedule/Tile op; rows `rocm_ebm_langevin_native_compiled` / `nvidia_ebm_langevin_native_compiled`; recorder | bit-exact on gfx1151, gfx1201, sm_120 (packets); one launch per loop; **open:** the dispatch/alloc/traffic/kernel-time packets vs the Python-emitted lanes | all three GPUs |
| T1 | **landed 2026-09-16**: `tessera-opt` registers EBM + Clifford dialects and passes when built with them (`TESSERA_HAVE_EBM` / `TESSERA_HAVE_CLIFFORD`) plus `convert-elementwise-to-linalg`; the whole chain is one invocation, and lit gains a `tessera-ebm` feature | one `tessera-opt` invocation runs the G2 chain; fixtures `phase2_autodiff/row_program_to_gpu_langevin.mlir`, `phase_f5/row_program_to_gpu_reduce.mlir` | host-free |

N1 and M1 followed G2/T1 rather than preceding them, because the cooperative
route turned out to carry both without new device work: an energy is a
gradient the integrator calls, and the sphere's projections are exactly the
row reductions the emitter already fences. G2 and T1 landed first, and G1 was
skipped: the cooperative route covers the
quadratic loop outright, so serial residency is only the fallback for
programs outside the row-program envelope. M1/M2 now build on G2 (the
tangent projection and retraction are row programs: a feature-axis
reduction followed by lane arithmetic, exactly what the emitter maps); N1 is
independent and can run in parallel on the CPU lane.

## 6. Risks named now

* **Reduction order across lanes.** *Resolved by construction (2026-09-16):*
  the emitter's fold is sequential over the feature index, so it reproduces
  the host's sequential f32 sum bit-for-bit (row normalization proof, three
  devices). A tree or warp-shuffle reduction, if one is ever adopted for
  speed, changes the declared order and must re-derive the reference — never
  hide it behind a looser tolerance.
* **Philox on ROCDL.** `arith.mului_extended` lowers to `llvm.umul_with_overflow`
  or a 64-bit multiply; both targets have it, but the first G1 run on each
  device is the proof, not the ISA table. The `philox_msl_source` template
  already exists for Apple.
* **`math.log`/`math.cos` precision.** *Measured 2026-09-16:* the f64
  `log`/`cos` of the Box–Muller draw matched the host bit-for-bit on
  gfx1151, gfx1201 and sm_120 in every packet row (worst abs error 0 after
  the single f32 rounding). The risk stays named: a toolkit upgrade can
  change a libm ulp, and the recorder will show it as a non-zero
  `max_abs_error` on a noise row — amend the policy to a tolerance for `z`
  then, never silently.
* **libdevice's approximate f32 paths are the NVVM route's default.**
  *Measured 2026-09-16:* `math.sqrt` on the NVVM route lowers to
  `__nv_sqrtf`, whose precise branch is gated on the `__CUDA_PREC_SQRT`
  reflect value that MLIR's pipeline never sets (LLVM 23's NVVMReflect
  exposes only `nvvm-reflect-ftz`), so the kernel ran `MUFU.SQRT` and one
  row of the row-normalization proof was 1 ulp off on sm_120 — the
  division was `div.rn` and both RDNA parts were exact. `convert-gpu-to-nvvm`
  marks the LLVM math intrinsics illegal, so the emitter now calls
  libdevice's rounding-explicit `__nv_fsqrt_rn` on NVIDIA (correctly rounded
  whatever the reflect flags say) and pins `llvm.intr.sqrt` on ROCm (AMDGPU's
  correctly rounded expansion). Every other `math.*` f32 op that reaches libdevice
  on this route (`rsqrt`, `exp`, `log`, `tanh`…) is still subject to the
  same default and must be measured before its result is called exact; the
  f64 `log`/`cos` of the noise are bit-exact because libdevice's f64 paths
  carry no approximate branch.
* **A registered dialect withdraws `--allow-unregistered-dialect`.** Registering
  the EBM and Clifford dialects in `tessera-opt` (T1) turned three
  `tests/tessera-ir/phase7` fixtures red: they predated their dialects and
  named placeholder ops (`energy_quadratic`, `annealing_schedule`,
  `partition_exact`, `rotor_from_axis`, `grade_projection`, …) with f32
  attributes the real ops reject, and MLIR refuses an unknown or malformed op
  inside a *registered* dialect whatever that flag says. CI does not run this
  suite, so the reds sat on main. All three now use the real spellings and
  verify with no escape flag; the genuinely absent ops (rotor construction,
  an annealing schedule) are named as gaps rather than faked. Expect this
  whenever a dialect is first registered in a driver.
* **Silent pass failure and dangling slot references (found 2026-09-16).**
  The row-program emitter could fail with no diagnostic, and a failed slot
  lookup returned an empty slot that every caller then indexed; separately, a
  reshape copied a slot *reference* into the same `DenseMap`, so the insertion
  could rehash and the reshaped row arrived empty. The first two are
  fail-open defects, the third memory-unsafe. The pass now refuses to fail
  without naming an op, and every refusal names the value and its producer.
* **NDEBUG hides dialect-promise defects (Decision #19, third instance).**
  The chain ran green on every NDEBUG driver in the fleet and aborted twice
  on Tajasarus's assertions-ON driver: `--inline` needs the LLVM dialect's
  promised `DialectInlinerInterface`, which `registerAllExtensions` does not
  provide (`tessera-opt` now registers it), and the row-program pass emits
  `tile.alloc_shared` without declaring the tile dialect as a dependency
  (now declared), and parsing its `tessera.*`-prefixed provenance
  attributes loads the Tessera dialect, which an input without Tessera ops
  never loaded (now declared). All one-line fixes; none visible without that
  box. Route every new pass through it before recording "passes".
* **Register pressure in G2b.** Keeping a row's state and gradient in
  registers across K steps bounds the feature width per lane; wider rows
  need the second reduction level. Declare the admitted width in the
  contract and fail closed above it.
* **Owner-marker sprawl in `NativeTapeToGPUPass`.** Adding a sixth owner
  kind is the fifth time that pass grew a special case. Record it in the
  compiler audit as technical debt to fold into a generic "isolated buffer
  program" admission with a required `tessera.source_kind` attribute.
