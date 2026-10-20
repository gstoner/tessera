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

## Bounded query-row integration candidate — 2026-10-19

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

The additive SM120 schema-8 row profile is the native prerequisite for bounded
original-source attention reverse. Its logical Graph has rank four with fixed
B/H/D and a dynamic query axis. The sealed producer dictionary has an explicit
positive i64 sequence_bound; its rows field equals B*H*sequence_bound with
checked signed-32-bit multiplication. Columns remain D.

Graph-to-Schedule and Schedule-to-Tile preserve the dynamic logical types.
Physical views retain the checked allocation capacity. A separate active-row
guard uses the runtime row scalar. The kernel checks positive rows,
rows<=capacity, rows divisible by B*H, and the fixed column count before
access. Per-thread row guards exclude inactive capacity rows. Source scalar
arithmetic and storage rounding are cloned from actual row SSA.

The three-pointer/two-i64 ABI retains two read-only inputs and one output.
Tile/Target metadata records schema 8 and row_multiple=B*H. Static schemas
1/6/7 keep their prior meanings, exact-row guards and artifact recipes. The
bounded profile is not admitted to ROCm, Apple or x86 physical lowering.

The private prerequisite passes 410 focused cases, twelve RTX5070 member
cases and the owning gfx1201 shared-materializer regression. Two twelve-profile
timing orders remove the initial consistent width257 regression while width35
remains mixed. Full receipts and source bindings are in
benchmarks/baselines/native_bounded_query_rows_20261019/README.md.
Composed normalization/AD/capacity-owner/public/portable integration remains
pending; no aggregate five-slice requirement is closed.

The shared Tile scalar-view prover follows true scalar conjunctions in an
actual enclosing then-region to find an unsigned coordinate bound. It does not
trust metadata or traverse disjunction, negation or selects. Together with
the existing constant-step SCF induction proof, the dominating physical row
bound removes redundant per-element guards. Both kernel geometry and active
query-row guards remain. The packet records owning numerical/negative-control and timing evidence
for this proof; full composed integration remains separate.
