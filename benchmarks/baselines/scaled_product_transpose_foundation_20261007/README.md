# Typed scale transpose foundation

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SCALED-PRODUCT-TRANSPOSE-2026-10-07.

The independent float64 diagnostic computes the scale cotangents in original
storage. For K group g, P_g = A_g @ B_g:
dSA[m,g] = sum_n(dY[m,n] * P_g[m,n] * SB[g,n//scaleN]).
dSB[g,c] = sum_m,sum_{n in c}(dY[m,n] * P_g[m,n] * SA[m,g]).
Shared storage sums over every logical batch prefix; mapped storage retains
the exact prefix. Final ragged K/N groups stop at the logical bounds.

Twelve host WSL tests check every scale coordinate against central finite
differences using an independent scalar contraction. Twelve owning gfx1201
tests compare the existing native public JVP against this transpose action:
<Jv,dY> = <v,J^T dY>. They cover two-axis batches, all three policies, KN/NK
storage and N129/K1536. Changed-seed warm runs forbid compiler and production
reference calls. This is native JVP evidence, not native VJP execution.

The native MLIR LinearTransposeInterface now constructs tensor.generate/scf
scale reductions for static typed E4M3FN matrices with f32 scales/output and
exact_per_block policy. Discrete matrices receive no implicit STE; E8M0 and
packed physical profiles retain their explicit derivative boundary.
Matching compiler build passes. Three native fixtures pass, including paired/in-place construction for three rank-four policies and scalar transposeA/B ragged groups. The initial fixture failures were visibility/missing-body expectations; logs are retained. 338 focused registry/oracle tests pass.

Remaining: derive these reductions into native Schedule/Tile kernels, extend
the immutable program member/export/ABI contract, preserve SSA lifetimes and
completion ownership, execute public reverse AD on gfx1201, compare all scale
gradients and record separate kernel/program/public timings. This foundation
does not close the generic transpose assertion or promote sibling backends.
