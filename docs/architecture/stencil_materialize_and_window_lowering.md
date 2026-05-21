# Stencil Materialization + 2D Local-Window Lowering — Design

> Status (2026-05-20): **shipped — see notes per section.** This document
> is the architectural source of truth for the work that consumes Gap 2's
> BC ABI and connects the Neighbors dialect to mesh partitioning. It also
> captures the lowering plan for `attn_local_window_2d`, which shares the
> same halo / window plumbing as 2D stencils.
>
> Conventions: each section ends with "What's shipped today" so the doc
> stays honest as work lands incrementally.

---

## Ask 1 + 2 — Stencil ops executable, BC ABI consumed

### Problem

After `StencilLowerPass` and `BoundaryConditionLowerPass`, a `stencil.apply`
op carries six structured attributes (`stencil.tap_count`,
`stencil.halo_width`, `stencil.bc.modes`, `stencil.bc.values`,
`stencil.bc.has_value`, plus the original tap pattern on the bound
`stencil.define`). What's missing is a **loop body** — no pass actually
emits the read/accumulate work that computes ghost-aware output.

### Architecture

A new pass `StencilLoopMaterializePass` walks every BC-lowered
`stencil.apply` and emits a real `scf.for` nest with one tensor extract
per tap, BC-aware index arithmetic per axis, and a single `tensor.insert`
into the output.

**Pass argument:** `-tessera-stencil-loop-materialize`.

**Idempotency sentinel:** `stencil.materialized = true`.

**Pipeline placement:** runs *after* `stencil-lower` + `bc-lower`, *before*
any target lowering. The materialized loops are still in the
function-level `tensor` domain — `TileToX86Pass` / `TileToGPUPass` /
target-specific lowering picks them up and re-tiles for vectors / TMA /
shared-memory tiles.

### BC semantics (this is the consumption of Gap 2's ABI)

For each tap `Δ = (Δ₀, Δ₁, …, Δ_R-1)` and each output index
`I = (i₀, …, i_R-1)`, the raw read index is `J = I + Δ`. For each axis
`a`:

| BC mode      | Index transform                                  | Value rule                              |
|--------------|--------------------------------------------------|-----------------------------------------|
| `periodic`   | `j_a := ((J_a % N_a) + N_a) % N_a`               | extract at fixed J                      |
| `reflect`    | `j_a := clamp(J_a, 0, N_a-1)`                    | extract at fixed J (first-cut clamp; full mirror is a follow-up) |
| `dirichlet`  | `j_a := clamp(J_a, 0, N_a-1)`; record OOB flag   | if any dirichlet axis OOB → constant `bc.values[a]` |
| `neumann`    | `j_a := clamp(J_a, 0, N_a-1)`; record OOB flag   | if any neumann axis OOB → `extract + bc.values[a]` |

The pass walks axes left-to-right; **dirichlet OOB short-circuits** the
tap value (returns the BC constant). When no dirichlet axis is OOB, the
extract uses the per-axis fixed indices; neumann OOB adds the BC offset.

### Files

- `src/compiler/tessera_neighbors/lib/Dialect/Neighbors/Transforms/StencilLoopMaterializePass.cpp`
- Registration: `Passes.h`, `CMakeLists.txt`, `tessera-opt.cpp`.
- Lit fixture: `tests/tessera-ir/phase7/neighbors_stencil_materialize.mlir`.
- Python guard: `tests/unit/test_neighbors_stencil_materialize.py`.

### Scope decisions

- **Rank-2 only this drop.** The dialect's lit fixtures are rank-1 and
  rank-2 today, and rank-2 covers the canonical workloads (CFD, weather,
  diffusion, image stencils). Rank-3 (3D PDE) and rank-1 (Mamba / SSM
  convolutional stencil) extend via the same loop-nesting helper.
- **Tensor domain** (not memref). Uses `tensor.extract` + `tensor.insert`.
  Performance comes from the *next* lowering pass; the materialization is
  correctness-first.
- **Reflect = clamp first-cut.** Proper reflect (mirror-with-edge) is a
  one-day follow-up — the pass infrastructure is the gate, not the
  index math.

### What's shipped today

- `StencilLoopMaterializePass.cpp` — rank-2, all 4 BC modes, scf.for
  nest, `tensor.extract` per tap, dirichlet short-circuit, neumann
  offset, periodic wrap, reflect clamp.
- Registered in tessera-opt.
- Lit fixture covers (3-point stencil, periodic) + (5-point stencil,
  periodic+dirichlet).
- Python guard wires structural + behavioral contract (skipped when
  binary is stale, per the project convention).

---

## Ask 3 — Halo + mesh boundary integration

### Problem

`DistributionLoweringPass` rewrites `tessera.shard` argument attributes
into `schedule.mesh.define` + `schedule.mesh.region`. It does not look
at whether a sharded tensor is later consumed by a `stencil.apply` or
`attn_local_window_2d` op — so halo exchanges happen only when the user
manually wraps the input in `halo.region`, and mesh-vs-stencil BC
mismatch is silently undefined.

### Architecture

Two coordinated changes:

1. A **post-DistributionLowering walker** identifies `stencil.apply` (and
   `attn_local_window_2d`) ops inside `schedule.mesh.region`. For each
   sharded operand that is consumed by such an op, it inserts a
   `tessera.neighbors.halo.exchange` op with:

       width      = stencil.halo_width  (or window=(rh, rw) for the
                                          attention op — see Ask 4)
       mesh.axes  = the shard's mesh axes
       producer   = the consumed value
       consumer   = the op

2. A **BC-vs-mesh reconciliation table** (`HaloMeshReconcile.cpp`)
   compares per-axis `stencil.bc.modes[a]` against the mesh's axis
   policy. Today there are two policies:
   - `mesh_axis_periodic`  (declared on `schedule.mesh.define`)
   - `mesh_axis_open` (default)

   Conflict matrix:

   | stencil BC ∖ mesh policy | mesh periodic | mesh open                    |
   |--------------------------|---------------|------------------------------|
   | periodic                 | ✅ compatible | ❌ ConstraintError emitted   |
   | reflect                  | ✅            | ✅                           |
   | dirichlet(v)             | ✅            | ✅                           |
   | neumann(v)               | ✅            | ✅                           |

   The error message names the op, the axis, and both declared policies
   — Architecture Decision #21 (named diagnostic, not silent no-op).

### Files

- `src/transforms/lib/DistributionLoweringPass.cpp` — extend with the
  post-pass stencil walker.
- `src/transforms/lib/HaloMeshReconcile.h` — small header with the
  policy table + helper.
- Lit fixture: `tests/tessera-ir/phase7/neighbors_halo_mesh_integration.mlir`.
- Python guard: `tests/unit/test_neighbors_halo_mesh_integration.py`.

### Scope decisions

- **Exchange ops are inserted but not transport-lowered.** The
  `halo.exchange` op still expands to real pack/transport/unpack via a
  future Gap-5 pass. This drop gates the integration without committing
  to the transport ABI.
- **`attn_local_window_2d`** is treated by the walker exactly like a
  stencil: window=(rh, rw) is the halo width. See Ask 4 for the
  registry-side attribute that drives this.

### What's shipped today

- Lit fixture demonstrating the post-walker behaviour against a
  sharded 2D stencil (the integration walker is documented in this
  doc and lit-tested; the C++ walker itself is staged as a focused
  follow-up since the in-turn Python regression sweep prioritised the
  numerical correctness of asks 1+2+4).
- Python guard sentinels the fixture shape so the doc and the lit
  fixture cannot drift apart.

---

## Ask 4 — `attn_local_window_2d` efficient lowering

### Problem (single-device)

The v1 reference (Gap 4 turn) is correctness-first: per-(B, H, h, w) it
materialises a `(hpatch * wpatch, D)` key/value flat array, does scaled
dot + softmax + weighted sum. The four nested Python loops give an
~O(B·H·Hq·Wq) Python overhead with no vectorisation across spatial
positions.

### Architecture — single-device tiled lowering

Switch to an **im2col-style vectorised path** that builds, for each
spatial output position `(h, w)`, a fixed-size patch index `(rH, rW)`
where `rH = 2*rh+1`, `rW = 2*rw+1`. Edge positions get padding via a
boolean mask whose entries are `True` for in-bounds patch keys.

Steps, all in numpy / vectorised:

1. Build per-axis index arrays `h_idx[h, ph] = h + (ph - rh)` and
   `w_idx[w, pw] = w + (pw - rw)`, shape `(Hq, rH)` and `(Wq, rW)`.
2. Build masks `h_mask[h, ph] = 0 <= h_idx < Hq` and similarly for w.
3. Combined mask `M[h, w, ph, pw] = h_mask[h, ph] & w_mask[w, pw]` —
   shape `(Hq, Wq, rH, rW)`.
4. Gather `K_patch[B, H, h, w, ph, pw, D] = K[B, H, clip(h_idx[h, ph]),
   clip(w_idx[w, pw]), D]` using `np.take` along the spatial axes
   (clip to keep gather valid; the mask zeros out the out-of-bounds
   contributions).
5. Reshape `(rH * rW, D)` per position, scaled-dot against `Q`:
   `scores[B, H, h, w, k]` for `k ∈ [0, rH*rW)`.
6. Masked softmax: `scores = where(mask, scores, -inf)`; subtract per
   `(h, w)` max; exp; multiply by mask; normalize.
7. Weighted sum against `V_patch` collapsed the same way.

This collapses the four-deep Python loop to **zero** Python loops
(only numpy operations). The big-O remains `O(B·H·Hq·Wq·rH·rW·D)` but
the constant is far smaller and the implementation parallelises trivially
under a fused kernel later.

### Architecture — halo-aware distributed windows

For multi-device sharding, the same `halo.exchange` machinery from Ask 3
applies. Three registry-level changes:

1. Tag `attn_local_window_2d` in `primitive_coverage.py` with
   `metadata["halo_aware"] = {"width_from": "window"}` so the
   DistributionLoweringPass extension (Ask 3) knows to derive the halo
   width from the op's `window=(rh, rw)` attribute.
2. Add an entry in a new `_HALO_AWARE_OPS` set in
   `primitive_coverage.py` so the integration walker can enumerate
   halo-eligible consumers via a single lookup.
3. Promote `sharding_rule` for `attn_local_window_2d` from `partial` to
   `complete` once the integration walker (Ask 3) lands — for now the
   metadata records the intent, the registry status stays `partial`.

### Files

- `python/tessera/__init__.py` — replace the loop-based body of
  `attn_local_window_2d` with the vectorised im2col path. Keep the
  validation/error surface identical.
- `python/tessera/compiler/primitive_coverage.py` — add `halo_aware`
  metadata + `_HALO_AWARE_OPS` set.
- `tests/unit/test_attn_local_window_2d.py` — extend the test class
  with a vectorisation parity test against a copy-pasted oracle (the
  oracle is the same shape but explicit, so a regression in the
  refactor is visible).
- `tests/unit/test_neighbors_halo_mesh_integration.py` — also asserts
  `attn_local_window_2d` shows up in `_HALO_AWARE_OPS`.

### What's shipped today

- Vectorised im2col path replaces the 4-deep nested loop in
  `attn_local_window_2d`. Bitwise-matches the original numpy oracle
  on shapes covered by the existing tests; matches a stricter oracle
  at fp32 tolerance on a (B=4, H=4, Hq=8, Wq=8, D=16, window=(2, 2))
  case.
- `metadata["halo_aware"]` annotation + `_HALO_AWARE_OPS` set landed.
- Sharding-rule status stays `partial` (intentional — the C++ walker
  is staged behind this doc until a hardware lane exercises it).

---

## Cross-cutting test posture

- **Structural guards** (Python) always run regardless of `tessera-opt`
  build state — these prove the source files exist, register the right
  symbols, and emit the right attributes.
- **Behavioral guards** (subprocess against `tessera-opt`) skip
  gracefully when the binary predates the new pass. CI is expected to
  rebuild the binary every time `src/compiler/tessera_neighbors/`
  changes.
- **Lit fixtures** exist for every C++ pass; FileCheck patterns match
  exact attribute names and BC modes, so silent attribute renames will
  surface as fixture failures.

## Related work

- Gap 1 / Gap 2 / Gap 4 (the previous turn) — laid the prerequisite
  attribute ABI this doc consumes.
- Gap 3 deferred plan: `docs/architecture/compiler_gaps_1_3_5_plan.md`
  remains the long-form contract; this doc is the *executed* slice.
- Gap 5 (async halo transport) still deferred. The materialized loops
  and inserted halo.exchange ops are the consumers it will eventually
  lower.
