---
last_updated: 2026-09-15
audit_role: reference
scope: GA/EBM frontend, native lowering, differentiation and workload residency
---

# GA / EBM architecture: current gaps and native integration

This source-backed refresh supersedes the
[August review](archive/GA_EBM_ARCHITECTURE_REVIEW_2026-08-02.md). The original
findings remain useful history, but several fixes landed and the proposed
Python-emitter destination conflicts with the MLIR/LLVM native foundation.
The [domain audit](DOMAIN_AUDIT.md) owns domain narrative and the
[integrated compiler plan](../compiler/INTEGRATED_COMPILER_PLAN.md) owns order.

## Re-check — 2026-09-15

No disposition above changed. `ExpandProductTable.cpp` still restricts v1 to
rank-1 static tensors (**closed 2026-09-16**: see the batched update below);
`energy.py::langevin_step` still falls back to
`_numerical_grad` without `grad_fn`; `EBM_ManifoldAttr` still admits only
euclidean/sphere/bivector. What did move: native HVP and attention JVP products
landed (2026-09-13) under AD-HIGHER-1, so the W6.4 / W6.3 pairing below now
routes as **W6.4 with AD-HIGHER-1** (W6.3 is a `successor` row in the plan), and
W3.5 routes to AD-SOLVER-IFT-1. Per-target execution evidence for GA/EBM is
read from the generated [domain proof ladder](../generated/domain_proof_ladder.md).

## Findings reconciled with source

| Original finding | Current disposition and evidence |
|---|---|
| §1.1 invalid/missing manifold accepted | **Fixed at the semantic boundary.** `EBMOps.td::EBM_ManifoldAttr` admits euclidean/sphere/bivector; `Canonicalize.cpp` rejects omission/unknown values. `canonicalize_rejects_bad_manifold.mlir` is the negative fixture. This does not establish a device consumer for arbitrary manifold metadata. |
| §1.2 host energy/gradient boundary | **Still material.** Native arithmetic on precomputed gradients does not lower a callable energy. `geo_sampling.py::_tape_grad` and `_tape_grad_mv` reduce host differentiation cost for traceable functions, but do not compile the energy loop to the device. |
| §1.3 disconnected GA producers and unbatched native expansion | **Batched expansion closed 2026-09-16 (`GA-NATIVE-BATCHED-2026-09-16`).** `ExpandProductTable.cpp` lowers any static `[..., dim]` rank to an scf.for nest over the compile-time table, and the product executes through MLIR/LLVM inside `libtessera_jit` (`cpu` / `cpu_clifford_llvm_jit`; M1 Max, Zen 5, Zen 2 parity with the GA reference). Still open: `RotorSandwichFold` and the other Clifford ops (reverse, wedge, contractions) have no native lowering behind the JIT, the GPU route (arena pipeline) is not built, and the specialized Python/runtime kernels remain a second implementation until the native package displaces them; recognizing a pattern is not execution proof. |
| §1.4 pass-description drift | **Partially repaired, still inconsistent.** `CliffordPasses.td` now says annotation-only, yet `GradeFusion.cpp` and `ExpandProductTable.cpp` perform real rewrites. Some EBM headers still say stub while the bodies annotate. Reconcile each description with the registered body; do not copy either blanket label. |
| §1.5 EBM checkpoint policy | **Removed from the default pipeline; standalone marker remains.** `CheckpointInnerLoop.cpp` still sets syntactic recompute/budget attributes. Do not restore it as production rematerialization without a shared demand/effect analysis and native policy consumer. |
| §2.1 grade information discarded | **Fixed for bounded consumers.** `ga/ops.py::_product_grade_contract`, native `InputGradeFusionPattern`, and `ExpandProductTable` consume operand grades. The latter prunes emitted terms; this is more than an unused annotation. |
| §2.2 dense coefficient storage | **Still a design/performance question.** Packed grade storage needs a typed layout and ABI, alias/AD rules and workload measurement. A twofold reduction in logical coefficients is not automatically a twofold throughput gain. |
| §2.3 table-driven synthesis | **Keep the idea; change the owner.** Share typed finite-algebra lowering through native MLIR/linalg/vector/LLVM boundaries. Existing source emitters can be candidates/oracles, not the new canonical compiler. |
| §2.4 PGA/CGA expansion | **Proof-gated.** A generic Cayley table does not prove invertibility, exp/log branches, degenerate metrics, differentiation, physical storage or device performance. Do not remove the signature gate as a substitute for those contracts. |
| §2.5 serial scalar accumulation | **Measure under declared numerical policy.** A balanced reduction or FMA can change rounding. Compare native loop/vector/unrolled alternatives; permit reassociation only when policy allows it. |
| §2.6 numerical-gradient defaults | **Partly improved.** Geometric samplers have guarded tape gradients; generic `energy.py::langevin_step` still chooses finite differences without `grad_fn`. Consolidate the traceability contract without treating an unrecorded NumPy callback as a zero derivative. |
| §2.7 AIS/Monte-Carlo | **After resident energy/sampler integration.** Their loops need compiled energy, RNG, state and reduction semantics. A separate AIS template would duplicate the unresolved substrate. |

Source anchors: [EBM ODS](../../../src/solvers/ebm/lib/Dialect/EBM/EBMOps.td),
[canonicalization](../../../src/solvers/ebm/lib/Passes/Canonicalize.cpp),
[grade fusion](../../../src/solvers/clifford/lib/Passes/GradeFusion.cpp),
[product expansion](../../../src/solvers/clifford/lib/Passes/ExpandProductTable.cpp),
[geometric sampling](../../../python/tessera/ebm/geo_sampling.py), and
[energy reference](../../../python/tessera/ebm/energy.py).

## Native architecture and acceptance

### W6.4 (formerly W3.6) — batched Clifford products and fusion

Use `[..., coefficients]` tensors with explicit algebra and admitted grade sets.
Lower batch loops plus the compile-time sparse multiplication table through
native IR, preserving the layout and numeric-policy contract. No new domain
scheduler or backend-specific table interpreter is needed.

Accept after scalar/batched/ragged shapes and mixed consumers match the reference;
forbidden grade terms are absent from emitted IR; fused rotor products retain
all consumers' semantics; and a native package executes on its owning host.
Measure dispatch/allocation overhead, memory traffic and kernel time separately.
A fused norm identity must respect its metric/signature assumptions.

### W4-PRODUCT-1 / AD-SOLVER-IFT-1 (formerly W3.5 / W4; W6.1 → AD-HIGHER-1) — an energy is a typed program

Define admitted energy inputs/captures and a scalar result, then use the shared
native derivative interfaces. Opaque Python/NumPy callbacks retain an explicitly
reported reference path; a supported native request must never silently call
back into Python. An analytic `grad_fn` must agree with the energy's contract.

The sampler body must carry manifold identity, RNG key/counter, step schedule,
state mutation and differentiation boundary through the same IR. Reuse the
[effect plan](../compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md), shared IFT where
implicit differentiation is appropriate, and native loop/tape products where
trajectory derivatives are required. EBM, Riemannian OT and GA flows share these
mechanisms but do not have identical mathematical derivative contracts.

Acceptance begins with a quadratic energy: native forward/gradient agree with
independent formulas, fixed-key samples agree with the declared policy, and the
complete loop executes without per-step host gradient transfers. Extend to a
nonlinear energy and manifold case with explicit singularity/branch handling.

### AD-RESIDUAL-EVAL-1 (formerly W5.1) / W2.4a — residuals and device ownership

Saved trajectory demand comes from the derivative program, not the presence of
an EBM op inside a loop. Retain exact generations, valid extents and completion
ownership. The new static f32 split-tape consumer is a correctness baseline;
it does not yet supply arbitrary mixed-state stochastic sampler tapes.
Compare full backward work and retained bytes before promoting a policy.

### W6.4 / AD-HIGHER-1 (formerly W6.3) — one finite-algebra lowering, separate proofs

The Python finite-algebra/jet substrate already exists in `autodiff/algebra.py`;
its Clifford table is checked against the GA oracle. The remaining opportunity
is native, typed and batched lowering shared with jets, including sparse grade
layouts where worthwhile. Clifford multiplication is noncommutative; the Weil
jet algebra is commutative/nilpotent. Share representation and lowering, not
unproved identities. Native jet coefficients also require scaling, order-zero
control, numeric policy and conditioning limits.

PGA/CGA, high-order attention and packed-grade kernels are later consumers.
Promotion requires complete operation/signature registration and exact-device
proof, not an allow-list edit or a theoretical operation-count reduction.

## Backend scope

Apple's specialized kernels are valuable evidence for their admitted envelopes.
They are neither whole-loop residency evidence nor justification to declare
NumPy optimal on Apple CPU. x86 CPU, CUDA and ROCm need independent packages,
ISA admission and workload measurements. In particular, CUDA schedules and
private-memory representations must not be transferred to AMDGPU or Metal.

No new runtime, registry status or performance claim is introduced by this review.

## Batched native products — 2026-09-16

Sync `GA-NATIVE-BATCHED-2026-09-16` (W6.4). The first W6.4 acceptance clause is met for the geometric
product: scalar (rank 1) and batched (rank 2, rank 3) shapes match the
standalone GA reference through a native package; forbidden grade terms are
absent from the emitted IR (`expand_batched.mlir` checks the grade-2 pruning;
the runtime test checks the pruned coefficients are written as zero and never
computed); dynamic extents fail closed. The lowering is one path —
`GradeFusion` → `ExpandProductTable` → scf/tensor/arith → one-shot
bufferization → LLVM — shared by `ts-clifford-opt` and `libtessera_jit`; no
`emit/` source generator was added. Executed on M1 Max (arm64), Princess-Luna
(Zen 5) and Super-Bear (Zen 2) via `tests/unit/test_clifford_jit_native.py`;
the execution matrix now carries the `cpu` row and the domain proof ladder
counts it. Second slice the same day (`GA-NATIVE-FAMILY-2026-09-16`): wedge, left
contraction, inner, norm, the three involutions, Hodge star, grade projection
and rotor sandwich lower through the same table and execute behind the JIT on
all three CPU hosts (rotor sandwich expands to its product chain there; the
fused marker survives the standalone pipeline for backends with a kernel).
Third slice (`GA-NATIVE-GPU-2026-09-16`): the same lowering, expanded by
`ts-clifford-opt` inside a per-thread kernel skeleton and folded to scalar
device code by the arena pipeline, executes as a native storage package on
gfx1151, gfx1201 and sm_120 (ten ops × three shapes; the grade-2 pruning is
counted in the emitted IR: 24 of 64 products). Not yet met: `exp`/`log` and
the field ops, ragged batches (the loop nest needs static extents), an Apple
package route, and the separate dispatch/allocation/
memory/kernel-time measurements the acceptance asks for — no performance
claim is made, and the Python `x86_clifford_compiled` / `rocm_clifford_compiled`
kernels are unchanged and remain the device lanes until displaced by measured
evidence.

## The bivector integrator and the overhead measurement — 2026-09-16

Sync `EBM-BIVECTOR-OVERHEAD-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). The
manifold enum is now fully served: `manifold = "bivector"` grade-projects the
gradient and the noise through the Clifford dialect's own op, keeps the state
exactly in the subspace over a 100-step chain, and reports its entry-grade
precondition per row. The acceptance's last clause, the overhead comparison,
is measured: the native route is flat in K (one launch and one host round trip
for the whole loop) while the Python-emitted route costs about 1.8 ms per step
because its kernel takes the gradient from the caller — 1.6x at K = 1 and 53x
at K = 32 on gfx1151. That is a dispatch result and **promotes nothing**:
kernel time is unavailable on either WSL2 ROCm box, and on two of the three
devices the cooperative kernel is the only compiled Langevin lane, so only
gfx1151 can run the comparison. Still open: promotion (kernel-time attribution
and bare-metal calibration), `exp`/`log` of multivectors for rotor sampling on
the group, opaque callbacks, and Apple.

## Nonlinear energies and the sphere integrator — 2026-09-16

Sync `EBM-NONLINEAR-MANIFOLD-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). The
acceptance's remaining clauses: **an energy is a typed program whatever the
energy is** — quadratic, Huber and softplus all run through the one
integrator, each matching its own oracle, with no `custom_adjoint_call` left
in any kernel (softplus gained a native `dy · sigmoid(x)` adjoint and a stable
lowering); and **a manifold integrator fails closed** — `manifold = "sphere"`
projects the gradient and the noise to the tangent plane and retracts by
normalization, reporting its entry precondition and retraction underflow in a
per-row status word rather than repairing either silently (Decision #21a). The
declared reduction order is sequential over features and the device's ordered
fold reproduces it, so the projections agree with the host fold rather than
merely within a tolerance. Verified on the Mac, gfx1151, gfx1201 and sm_120.
Still not met: the bivector integrator (it needs the Clifford `grade` op inside
the EBM lowering), opaque callbacks, Apple, and the overhead measurement that
would let this lane displace the Python-emitted kernels.

## The Langevin loop as one cooperative kernel — 2026-09-16

Sync `EBM-NATIVE-GPU-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). The device
half of the first acceptance clause: the compiler-derived gradient is
lowered *inside* a kernel by the new row-program emitter
(`tessera-row-program-to-gpu`) over the lowered loop — one block per row,
one lane per feature, the K-step loop with its Philox draw carried in
registers, ordered shared-memory reductions — packaged by the native GPU
storage route and launched as one call. Bit-exact with the declared policy
on gfx1151, gfx1201 and sm_120 (packets, worst abs error 0), reductions
bit-exact with the sequential f32 fold, one launch per loop, one driver
invocation for the chain. Still not met: the overhead measurement that
would let this lane displace the Python-emitted `*_ebm_langevin_compiled`
kernels (no promotion is claimed); nonlinear and manifold energies (the
sphere and bivector integrators are row programs the same emitter maps and
are the next slices); opaque callbacks; Apple. Details and the reason the
scoped Tile contract was not needed: [EBM_NATIVE_LOOP_ARCHITECTURE.md](EBM_NATIVE_LOOP_ARCHITECTURE.md) §3.3.

## Quadratic energy through the backbone — 2026-09-16

Sync `EBM-NATIVE-QUADRATIC-2026-09-16` (W4-PRODUCT-1 / AD-SOLVER-IFT-1). The first acceptance clause
above is met on the CPU lane: the quadratic energy is a Graph IR function,
its gradient is the compiler's (paired reverse-mode), fixed-key samples agree
bit-for-bit with the declared Philox / Box-Muller policy, and a K-step loop
executes as one native call without per-step host gradient transfers, on the
M1 Max, Princess-Luna and Super-Bear. The EBM dialect gained its first
lowering pass and a `captures` operand on `langevin_step`; the shared
compiler gained the `sub` adjoint and `unsqueeze`/`broadcast` lowerings the
energy's gradient needed. Not yet met: the nonlinear and manifold cases
(sphere / bivector integrators fail closed), opaque callbacks (still the
reported reference path), a GPU package for the loop, and the "no per-step
transfers" claim on a device — this is the CPU lane. `energy.py::langevin_step`
still takes finite differences without `grad_fn`; the native lane is opt-in.
The device package, the sphere and bivector integrators and the nonlinear
energies are scoped as slices G1/M1/M2/N1/G2/T1 in
[EBM_NATIVE_LOOP_ARCHITECTURE.md](EBM_NATIVE_LOOP_ARCHITECTURE.md), which
records where each existing device route stops on this loop today.

