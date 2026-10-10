# gfx1201 continuous reverse residual dependency integration

Owner AD-RESIDUAL-EVAL-1; shared FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1.
Sync CONTINUOUS-SCALED-REVERSE-RESIDUALS-20261009; follows PR917.

Native AD already builds a recompute-all backward Graph. Native export now
takes the dependency closure of requested gradients: required primal products,
residual sums/permutations, adjoint reductions and cotangent sums/permutations.
Only selected dependencies execute. A-only differentiation of the two-product
chain needs an intermediate cotangent but no primal recomputation; C gradients
need the first product. Full seven-role reverse contains nine members.
All arithmetic, recomputation, buffer lifetimes, Schedule/Tile and LLVM lowering
are compiler-owned. Python validates/marshals the emitted ABI.

## Manifest extension

Schema 1 and the native C ABI remain compatible. New reverse exports carry
dependency_policy = native_recompute_prefix_v1. Every step has dependency_kind
(residual or cotangent). Every reduction has seed_input, an actual SSA buffer
binding derived from the AD region's cotangent capture attribute. Requested
gradient outputs/contribution sums retain gradient_argument; private intermediate
cotangents may omit it. The seed must trace to the final root output cotangent
through admitted adjoints, sums or permutations; coefficients cannot be
cotangents. Residual products cannot consume a cotangent. Prefix SSA, exact
storage/geometry, requested output roles and recomputed first-write/last-read
lifetimes are checked against each member's complete native program witness.
Legacy manifests without the policy retain their original stricter input-frame
validation. Unknown policies are rejected.

## Proof and measurements

105 matching native compiler/admission/manifest tests pass, including forged seed,
dependency kind, output role and lifetime rejection. 567 affected frontend and
projection tests pass with 42 explicit compiler skips. Ruff and zero-error mypy
pass. 70 exact gfx1201 device tests pass, including changed-input/seed replay,
requested subsets, role ordering, retained outputs, compiler/eager-free warm
calls and adjacent public floating reverse families.

RX 9070 XT identity, UUID, source/tool/image hashes are in gfx1201.json.
The timing frame is (M,K,N,P)=(2,9,5,3), K/N block size 4, ragged groups.
Device gradient checks additionally cover (3,17,7,6) and (1,4,4,4).
Independent scalar adjoint oracle max absolute error is 4.65661e-10.
Seven 128-repeat HIP windows: full nine-member native reverse median 22.486 us.
Grouped per-member times are recorded independently and include device graph
dispatch; they are not additive interleaved-program timing.
Public compilation-warm reverse median is 2.416 ms; preparation/allocation is
not excluded. This is functional/timing characterization, not a speedup claim.

Reproduce with matching core/ROCm tools and checked HIP provider:
python benchmarks/rocm/record_continuous_scaled_reverse.py --output packet.json

Mapped composed chains, aliases, dynamic shapes, residual-save policy selection,
generic scaled batching/transpose closure and backend performance obligations
remain open. No sibling architecture execution follows from this packet.

Final gates: 308 audit/diagnostic/pass/recorder checks and 32 generated documents
pass. Another 85 exact gfx1201 FP8 scale-gradient/composed/permutation cases pass
with the shared native capture metadata.
The final full host unit lane reports 20,448 passed, 9,592 skipped and two
failures: generic scaled_matmul batching and transpose closure. The recorder
naming failure seen in the initial run was fixed in its tracked owning plan;
the final full run proves it is resolved. No coverage/test state was weakened.
