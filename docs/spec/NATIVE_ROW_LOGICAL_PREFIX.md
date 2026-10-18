---
last_updated: 2026-10-18
audit_role: reference
---

# Native row logical prefixes

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: NATIVE-ATTENTION-SOURCE-REVERSE-20261018.

A static compact row graph retains its logical tensor shape in Graph IR.
Normalization, statistics and reductions operate on the final axis. A
rank-reduced statistic has exactly the leading shape; its broadcast dimensions
are the ordered leading axes. Scalar constants retain empty broadcast axes.

The native physical view combines the leading extents into rows and retains
the last extent as columns. Both must be positive and fit signed 32-bit native
geometry; prefix multiplication is checked before multiplying. Dynamic shapes,
encodings, nonleading broadcasts and nonfinal reductions are not admitted.
The source/body hash retains the full logical Graph, so equal flattened shapes
do not substitute one source contract for another.

The SM120 schema-1 two-result and schema-6 one-result pointer ABIs are unchanged:
two readonly inputs, distinct outputs and explicit i64 row/column guards. Only
physical views are flattened; no Python derivative expression or Graph rewrite
implements this mapping. The source attention reverse owner must validate its
logical root/result shapes and retain normalized Q, saved output and FP32 LSE
through the complete pullback. O and LSE cotangents remain distinct.

ROCm schema-7 source admission remains rank two pending its owning follow-up.
Apple and x86 receive no physical row profile. The initial candidate has six
valid rank-three/four red fixtures and three negative-geometry/sibling checks.
Matching-build proof passes 91 unique compiler cases and 27 unique RTX5070
member/adjacent cases, with twenty unchanged rank-two stage identities. Eight
paired raw-member timing profiles pass before/after oracles. Portable
composed ownership and complete stage/program timing remain pending. Composed source reverse and the
five aggregate slices remain open.
