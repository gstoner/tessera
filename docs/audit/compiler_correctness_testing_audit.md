# Compiler-Correctness Testing Coverage Audit

> Status (2026-05-21): **first survey.** Identifies every compiler area
> that benefits from the *six-layer test pyramid* introduced in the
> May-2026 testing-strategy review, and records current coverage status
> per area.  Updates land in lockstep with each compiler-correctness
> improvement.
>
> Legend for the **Coverage** column:
>   ✅  shipped today
>   ⚠️  partial — covered for one path, missing for peers
>   ⏳  planned, not yet shipped
>   —  not applicable

---

## Six layers of compiler-correctness testing

| Layer | What it catches | Cost |
|---|---|---|
| 1 — Structural guards     | File exists; registration wired; CMake builds | secs, no binary |
| 2 — Verifier negatives    | Type-system violations the compiler should reject | secs |
| 3 — Roundtrip lit         | Parse → print → parse stability (silent SSA corruption) | ms per fixture |
| 4 — FileCheck IR shape    | Pass emitted the structural primitives intended | ms per fixture |
| 5 — ABI / interface locks | Runtime symbol contracts (arg count, types) | ms |
| 6 — Differential / oracle | Semantic equivalence to a trusted reference | ms–seconds |

Two **cross-cutting** concerns that don't fit neatly into one layer but
matter for compiler-correctness:

| Cross-cutting | What it catches |
|---|---|
| **X-a — Parser fuzz** | String-eating functions that crash on malformed input |
| **X-b — Pass-order matrix** | Implicit dependencies between passes (attribute hand-off, idempotency, commutativity) |

---

## Coverage matrix — string parsers (X-a)

Every string-eating boundary between IR and pass options should have a
fuzz test that:
  1. Generates structured legal input, asserts no-crash + expected
     parse result.
  2. Generates malformed input, asserts no-crash (parser may emit a
     diagnostic; it must not segfault or hang).

| Parser | Source | Tokens / grammar | Coverage |
|---|---|---|---|
| `parseBcString` (BC ABI: periodic/reflect/dirichlet(v)/neumann(v)) | `BoundaryConditionLowerPass.cpp` | comma-separated, scalar payload in parens | ✅ **Gap-2 lands** — `test_string_parsers_fuzz.py::TestBCParserFuzz` |
| `splitComma` (mesh-axes + mesh-sizes pass options) | `DistributionLoweringPass.cpp` | comma-separated, trimmed | ✅ **Gap-2 lands** — `TestMeshAxesFuzz` |
| `featureMapToInt` (linear-attn feature_map enum) | `LinearAttnToAppleGPU.cpp` | exact-match: elu/relu/identity/polynomial_2 | ✅ **Gap-2 lands** — `TestFeatureMapEnumFuzz` |
| `canonicalize_dtype` (15-name canonical + alias set + planned/gated) | `python/tessera/dtype.py` | exact-name, with aliases | ⚠️ — partial via existing `test_canonical_dtype.py` (135 hand-written); fuzz lane planned |
| `_jit_target` string aliases (`apple_cpu` / `apple_gpu` / `rocm` / `metalium`) | `python/tessera/compiler/jit.py` | exact-set string | ⚠️ — exact-match cases tested; unknown-string negative path tested; no fuzz |
| Numeric-policy string fields (storage/accum/rounding/math_mode) | `python/tessera/compiler/primitive_coverage.py` | exact-set strings | ⚠️ — constructor validation tested; no malformed-input fuzz |
| Tap-list parser (`DeltaArrayAttr`) | `tessera_neighbors.td` ODS | MLIR attr syntax → parsed by MLIR core | — handled by MLIR's own attribute parser |

**Priority:** the four ✅ targets are the ones where a malformed input
crashes a real compiler pipeline.  The dtype + jit-target cases above
are not crash-prone (Python ValueError → caught early); they get fuzz
coverage if/when string acceptance broadens.

---

## Coverage matrix — pass-order matrices (X-b)

For every chained pipeline alias (`-tessera-lower-to-X`), the pass
sequence enforces *implicit* dependencies: an attribute set by pass A
must exist when pass B runs.  A pass-order matrix locks those
dependencies as named diagnostics rather than silent crashes.

| Pipeline | Passes (in order) | Dependencies | Coverage |
|---|---|---|---|
| **Halo / Neighbors** (Phase 7) | stencil-lower → bc-lower → halo-mesh-integration → halo-transport-lower | `stencil.bc`→`stencil.bc.modes`; bc.modes→materialize; halo.exchange→transport-triple | ✅ **Gap-3 lands** — `test_neighbors_pass_order_matrix.py` |
| **Spectral solver** | LegalizeSpectral → SpectralMXP → TransposePlan → Autotune → LowerToTargetIR → DistributedFFT | annotation-driven planning passes; each consumes the previous's attrs | ✅ **Gap-3 lands** — `test_spectral_pass_order_matrix.py` |
| **x86 lowering (Phase 2)** | DistributionLowering → EffectAnnotation → Tiling → TileToX86 | EffectAnnotation must precede CollectiveInsertion (Architecture Decision in CLAUDE.md) | ⚠️ — happy-path lit covered in `phase2/full_pipeline.mlir`; reorder matrix planned |
| **NVIDIA SM_90** | EffectAnnotation → Canonicalize → SwigluFusion → MLA/NSA/Hybrid/Lightning/Delta fusion → DistributionLowering → TileIRLowering → WarpSpec → AsyncCopy → WGMMA → TMA → NVFlashAttnEmitter | 13 passes; many implicit fusion-order deps | ⏳ — explicit plan: `docs/audit/nvidia_rocm_execute_and_compare_plan.md`; paired with Phase G hardware enablement |
| **Apple GPU runtime** | 14 lowerings: NSA / MLA / SwiGLU / matmul→softmax→matmul fusions → matmul→softmax / gelu / rmsnorm fusions → per-op (matmul mps, rope, flash_attn, linear-attn, attn_local_window_2d, softmax, gelu) | "longest fusion first" — a per-op pass running before its fusion stealing the op is the canonical failure | ✅ **shipped 2026-05-21** — `test_apple_gpu_pass_order_matrix.py` (18 tests; 11 dependency pairs locked) |
| **TPP space-time** | 7 passes via `tpp-space-time` | space-time decomposition order | ⏳ — happy-path covered; reorder matrix planned |
| **NV Rubin CPX** | 4 CPX passes | CPX-specific | ⏳ — separate driver `tessera-cpx-opt`; reorder matrix planned |
| **Phase 5 core solvers** | 11 solver passes (Sparse, RNG, Newton, Trig, Periodic, …) | Some pass-order requirements documented in CLAUDE.md | ⏳ — happy-path covered; reorder matrix planned |

**Priority order:**

1. **Halo + Spectral** — both ship in this audit's drop because they
   are the most recently-added pipelines and the order contracts are
   freshest in our minds.
2. **Apple GPU runtime** — "longest fusion first" is a real contract;
   a reorder-matrix protects against a future per-op pass that gets
   greedy.  The 9 passes are well-bounded.
3. **NVIDIA SM_90** — explicitly paired with NVIDIA backend enablement
   (per the user direction on Gap 1).  No point locking an order
   before the pipeline actually runs on hardware.
4. **TPP, CPX, Phase 5** — happy-path covered; not in current critical
   path.

---

## Coverage matrix — semantic / oracle (layer 6)

Layer 6 — differential testing against a trusted reference — is the
*single* layer that catches semantic bugs other layers can't see.
Today's coverage:

| Op family | Reference | Coverage |
|---|---|---|
| `attn_local_window_2d` | `_numpy_oracle` in `test_attn_local_window_2d.py` | ✅ |
| Halo transport (pack/transport/unpack) | `tessera.testing.halo_transport` mock-collective | ✅ |
| **diffusion grid core** (proving workload: `corrdiff_core`) | deterministic numpy reference baked into the model | ✅ |
| Linear attention | `_linear_attn_reference` (host fallback in same test as MSL kernel) | ✅ |
| FA-4 sparse attention | `_attention_vjp` numeric finite-diff fallback | ⚠️ — VJP only; forward pass implicit |
| Spectral (fft/ifft/rfft/irfft/stft/istft/dct) | `np.fft.*` references in `test_spectral_solver_passes.py` | ✅ |
| Stencil materialization | **no oracle yet** — pass emits IR but no execution lane runs it | ⏳ — **deferred to Gap 1** (NVIDIA/AMD GPU enablement) |
| 2D window attention native lowering | Python `attn_local_window_2d` is the oracle today; native MSL kernel doesn't execute yet | ⏳ — paired with Apple GPU kernel ship |
| MLA decode fused / NSA fused | Host reference matches fused path | ⚠️ — happy-path; full envelope sweep planned |
| **Halo transport (pack/transport/unpack triples)** | `test_halo_execution_lane.py` — IR parameters → mock-collective execution → numpy oracle, bitwise compare | ✅ **shipped 2026-05-21** — first end-to-end execute-and-compare lane; template for future hardware lanes |
| **diffusion grid core IR-visible** (stencil + window-attn + halo, via `corrdiff_core` workload) | `test_corrdiff_ir_visible.py` — proves three workstreams compose in one IR flow | ✅ **shipped 2026-05-21** — capstone integration test |

**The standing rule:** every new lowering pass that emits IR which
*can be executed* must ship with an oracle compare in the same PR.
Today three passes emit non-executable IR (StencilLoopMaterialize,
HaloTransportLower, AttnLocalWindow2DToAppleGPU) — all three are
explicitly tagged as waiting on Gap 1 / native kernel ship.

---

## Recommendations (prioritized)

1. ✅ **Shipped:** Halo + Spectral pass-order matrices (Gap 3).
2. ✅ **Shipped:** BC parser + splitComma + feature_map enum fuzz (Gap 2).
3. ✅ **Shipped 2026-05-21:** Apple GPU runtime pipeline reorder matrix
   (14 passes, 11 dependency pairs, longest-fusion-first contract locked).
4. ✅ **Shipped 2026-05-21:** HaloMeshIntegrationPass extension to
   recognise `attn_local_window_2d` (closes the loop on Sub-1's
   `_HALO_AWARE_OPS` registry).
5. ✅ **Shipped 2026-05-21:** Halo execute-and-compare lane — first
   Layer-6 oracle test in the project; template for future hardware
   lanes.
6. ✅ **Shipped 2026-05-21:** CorrDiff IR-visible fixture — proves the
   three Phase 7 workstreams compose in one IR flow.
7. ⏳ **Backlog (gated on Phase G):** NVIDIA SM_90 pass-order matrix
   + runtime ABI lock + H100 execute-and-compare lane.  Full plan:
   `docs/audit/nvidia_rocm_execute_and_compare_plan.md`.
8. ⏳ **Backlog (gated on Phase H):** ROCm CDNA/RDNA pass-order matrix
   + per-arch runtime ABI lock + MI300X/MI325X execute-and-compare
   lane.  Same plan doc.
9. ⏳ **Backlog (low priority):** TPP, CPX, Phase 5 reorder matrices;
   dtype + jit-target fuzz lanes (Python validation already crash-safe).

## Cost summary

| Item | Effort | Status |
|---|---|---|
| This audit document | ~1 hour | ✅ |
| Halo pass-order matrix | ~4 hours | ✅ |
| Spectral pass-order matrix | ~3 hours | ✅ |
| BC parser + peers fuzz | ~3 hours | ✅ |
| Apple GPU reorder matrix | ~3 hours | ✅ shipped 2026-05-21 |
| HaloMeshIntegrationPass `attn_local_window_2d` extension | ~3 hours | ✅ shipped 2026-05-21 |
| Halo execute-and-compare lane | ~3 hours | ✅ shipped 2026-05-21 |
| CorrDiff IR-visible fixture | ~2 hours | ✅ shipped 2026-05-21 |
| NVIDIA/ROCm plan doc | ~1 hour | ✅ shipped 2026-05-21 |
| NVIDIA SM_90 matrix + ABI lock + H100 oracle | ~7 days (gated on hardware) | ⏳ |
| ROCm matrix + ABI lock + MI300X/MI325X oracle | ~10-15 days (gated on hardware) | ⏳ |

Total shipped 2026-05-20/21: ~22 engineering hours.  Total backlog
through NVIDIA + ROCm hardware enablement: ~17-22 engineering days,
gated on hardware access per the deferral in
`docs/audit/nvidia_rocm_execute_and_compare_plan.md`.
