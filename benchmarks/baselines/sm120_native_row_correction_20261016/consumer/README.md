# Native row-residual backward consumer

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

A distinct compiler-internal checkpoint_backward_rows tensor product takes
`dO,Q,K,V,[bias],LSE,row_correction`. Native paired AD derives it from the actual
original backward product, retaining source-pair witness, frontend roles,
requested activity and readonly argument properties. Rounded O and the already
applied dLSE seed are absent. Legacy backward Graph/Tile/package ABIs remain
unchanged. A sealed multi-result Schedule verifies complete SSA result order,
physical roles, storage, row capacity and compact-output launch policy before a
distinct Tile pointer ABI. Combined gradient launch capacity is checked at both
Schedule and authored Tile boundaries.

The production portable compiler lane passes 574 consumer/producer/legacy AD and
mandatory registry cases with zero skips. Owning RTX5070/SM120 proof passes 120
cases with zero skips, across two receipts: 96 full-gradient and 24 compact cases.
The full-gradient lane executes the same images on two changed root/seed sets,
compares to independent FP64 math and requires bitwise equality with original
inline gradients. Producer and consumer launch on the same stream without an
intermediate host synchronization. Perturbing the explicit row residual changes
Q/K/bias as predicted while dV remains bitwise unchanged. Guarded row/gradient
stores pass. Compact Q-only, dV-only and Q/bias outputs pass with64/128 threads;
dV-only intentionally omits the producer, receives NaN row storage and leaves it
untouched. FP16/BF16, half/F32 dO, GQA, causal variants, physical bias, LSE seeds,
Sq1, Sq>Sk and ragged Sk129 are covered.

The initial Schedule-swap fixture did not mutate real SSA names and its text
comparison was fooled by a trailing newline. The fixed fixture swaps actual
same-typed results and requires real content mutation. The failing discovery
receipt and successful final receipts are retained. record.json binds source
snapshots, production tools, unchanged provider identity and exact GPU identity.

Execution here uses diagnostic resident launchers. Additive prepared ownership,
private row lifetime, complete all-member portable provenance, original public
frontend stream/snapshot/generation/replay proof and separate stage/completed-call
benchmarks remain open. No public default, speedup or five-slice closure is claimed.
