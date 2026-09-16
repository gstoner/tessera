---
last_updated: 2026-09-16
audit_role: reference
scope: EBM sampling loops as native GPU packages; nonlinear and manifold energies
---

# The EBM sampling loop as a native GPU package

Owner: [W4-PRODUCT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#w4-product-1)
with [AD-SOLVER-IFT-1](../compiler/INTEGRATED_COMPILER_PLAN.md#ad-solver-ift-1);
acceptance from the [GA/EBM review](GA_EBM_ARCHITECTURE_REVIEW.md) §"an
energy is a typed program". Sync key `EBM-NATIVE-QUADRATIC-2026-09-16`.
Written after the CPU-lane slice landed (`tests/unit/test_ebm_native_langevin.py`)
and after driving that same loop as far as the existing device routes take it;
every "today" claim below is a measured stop, not a reading of prose.

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

### 3.3 G2 — cooperative kernel (rows to blocks, features to lanes)

Purpose: the loop at real batch sizes. Two candidate mechanisms, decided by
measurement per Decision #28, both fed by the same lowered IR:

* **G2a upstream spine.** `convert-linalg-to-parallel-loops` →
  `gpu-map-parallel-loops` → `convert-parallel-loops-to-gpu` →
  `gpu-kernel-outlining` on the bufferized loop function. Free of new
  Tessera code, but it produces **one kernel per linalg op** — the row
  reduction inside `@E__bwd`, the elementwise update and the noise generic
  each outline separately, so a K-step loop becomes K×(3–4) launches driven
  from the host. That is exactly the per-step transfer the acceptance forbids
  unless the `scf.for` itself is on the device, which the spine cannot do
  (the loop would have to be inside one kernel). G2a is therefore a
  measurement baseline and a fallback for energies whose gradient cannot be
  expressed as a single block-local program, not the target.
* **G2b Tile contract.** Project the lowered loop into a Schedule/Tile
  contract the way SSD is: a `schedule.ebm_langevin` op carrying (state
  shape, η, T, K, integrator, noise policy, and the energy's `__bwd` as a
  **region** rather than a symbol), lowered by `PMPasses` to a
  `tile.langevin_kernel` and materialized per target (`materializeSm120…` in
  `NVIDIALowering.cpp` beside the softmax/reduce/norm kernels; the
  `TileToROCM` streaming lane on ROCm). Lane mapping follows `NativeSSD.h`:
  one block owns one row (feature count ≤ block width; wider rows tile the
  feature axis with a second reduction level), lanes own feature elements
  and keep the state in registers across all K steps, the per-row reduction
  in the gradient runs in LDS/shared memory with a barrier, Philox is
  per-element from (flat index, key[1]), and the key advance is a scalar the
  block leader writes. The body region is lowered by the *same* scalar
  emitter the Tile kernels already use for elementwise arithmetic; a
  reduction inside the region maps to the kernel's row-reduction primitive.
  Decision #32 carries `numeric_policy` (f32 accumulation, ordered
  reduction, no fast-math) onto the contract so the reduction order is
  declared, not incidental.

G2b is the target because it is the only shape that keeps the K-step loop
inside one kernel with the state resident in registers, which is the whole
point of the acceptance clause. G2a exists to measure it against.

Acceptance G2: same bit-exactness as G1 (the noise policy is per-element, so
lane mapping cannot change samples; the row reduction's order is declared
and the reference sums in the same order), plus the separate
dispatch/allocation/memory-traffic/kernel-time measurement the GA/EBM review
asks for, recorded with `route` and `latency_source`. Promotion of any lane
over the Python-emitted `*_ebm_langevin_compiled` kernels waits for that
measurement.

### 3.4 Apple

The Apple GPU route is the MSL arena (`apple_native_arena.py`,
`--tessera-tile-buffer-arena=emit-apple-msl=true`), not the NVVM/ROCDL
pipeline. G1's isolated loop function is the input it needs; the open work
is that the arena emitter takes a Tile-level buffer program, not a
bufferized scf loop. Route through G2b's `tile.langevin_kernel` once it
exists; do not write a third emitter (`CLAUDE.md` §"the real Apple gap").

## 4. Nonlinear and manifold energies

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
| M1 | `manifold = "sphere"` on CPU + G1: tangent projection, retraction, per-row status word; reference guard mirrored | ‖x'‖ = 1 per row, T = 0 monotone descent, 1e-5 parity on projected quantities, bit-exact noise | Mac + x86 + G1 devices |
| M2 | `manifold = "bivector"` via `clifford.grade` on gradient and noise; EBM → Clifford build dependency | parity with `bivector_langevin_step` on Cl(3,0) grade 2; state stays grade-2 over 100 steps | same |
| N1 | Huber and softplus energies through the paired pass with no placeholder; the `softplus`/`cosh` adjoints if missing | gradient matches finite differences and the Apple/ROCm reference kernels | Mac + x86 |
| G2 | `schedule.ebm_langevin` / `tile.langevin_kernel` with a body region; sm_120 and ROCm materializers; G2a spine as baseline | bit-exact samples; separate dispatch/alloc/traffic/kernel-time packets vs the Python-emitted lanes | all three GPUs |
| T1 | register EBM + Clifford in `tessera-opt` when built, so one driver runs the whole chain (the JIT already does) | the G1 chain runs as one `tessera-opt` invocation | host-free |

G1 before M1/M2 because the manifold integrators need the same device
residency proof and the status-word machinery; N1 is independent and can run
in parallel on the CPU lane; G2 last because it is the only slice that
changes the Schedule/Tile dialects and it should land against a working G1
that already proves the numbers.

## 6. Risks named now

* **Reduction order across lanes.** G2b's block reduction cannot reproduce
  the sequential f32 sum the CPU/G1 routes use. The contract must state the
  order; parity for reduced quantities is tolerance-based while the noise
  stays bit-exact. Do not hide this behind a looser global tolerance.
* **Philox on ROCDL.** `arith.mului_extended` lowers to `llvm.umul_with_overflow`
  or a 64-bit multiply; both targets have it, but the first G1 run on each
  device is the proof, not the ISA table. The `philox_msl_source` template
  already exists for Apple.
* **`math.log`/`math.cos` precision.** `convert-math-to-llvm` maps to
  `llvm.intr.log`/`cos`; the device libm implementations differ from the
  host's by ulps. The reference computes in f64 and rounds once; the kernel
  does the same (f64 intermediates), which is why the CPU lane is bit-exact.
  Device libm f64 `cos` may still differ in the last ulp — if it does, the
  packet records the observed max ulp and the policy is amended to a
  tolerance for `z`, never silently.
* **Register pressure in G2b.** Keeping a row's state and gradient in
  registers across K steps bounds the feature width per lane; wider rows
  need the second reduction level. Declare the admitted width in the
  contract and fail closed above it.
* **Owner-marker sprawl in `NativeTapeToGPUPass`.** Adding a sixth owner
  kind is the fifth time that pass grew a special case. Record it in the
  compiler audit as technical debt to fold into a generic "isolated buffer
  program" admission with a required `tessera.source_kind` attribute.
