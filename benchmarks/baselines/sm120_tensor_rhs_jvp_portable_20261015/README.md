# Portable paired RHS tensor JVP on RTX 5070

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-TENSOR-RHS-JVP-20261015.

## Contract and proof

Distinct portable schema tessera.nvidia.tensor_rhs_jvp.v1 retains the native
paired v7 plan, ordered Graph/Schedule/Tile witnesses, Target IR, PTX images
and checked descriptors. Restore validates SSA roots, both returned outputs,
private buffer bytes/lifetimes and member provenance/ABI before native prepare.
Existing one-output validators still reject v7. Digests check integrity, not
authenticity: these are trusted executable artifacts.

The explicit PreparedTensorJVP.manifest/from_manifest API executes through the
same sealed C++ owner. This is prepared replay, not a generic RuntimeArtifact
or public JVP artifact dispatch extension. Fresh-process FP16/BF16 tests disable
the compiler, packaging functions and Python member launcher. Outputs are
bitwise equal to compiled execution and remain independent across later calls.
Resealed lifetime, role, output, shape and Graph corruption rejects. The full
paired suite passes 31 cases on RTX 5070/sm120, LLVM/MLIR 23.1.1 assertions.

## Timings

Six correctness-gated RMSNorm -> ReLU -> matmul RHS-only JVP cases cover both
FP16 and BF16 at M,K,N 7/32/9, 17/48/19 and 129/256/33. Captured native program
medians are 6.512–52.125 us. Completed prepared replay is 0.272–0.330 ms;
completed public frontend calls are 0.751–0.815 ms. These API paths have different
host work; no device speedup or default promotion is claimed. Compile/prepare,
member/program graph intervals and completed host calls are separate fields.
Maximum absolute error is 4.77e-6 against independently stored-coefficient
matmul oracles. All inputs and retained outputs remain unchanged.

Preparing identical images acquires a new native lease, making an older handle
stale. The retained initial failed benchmark log demonstrates this guard.
The corrected benchmark completes original-owner timing before preparing replay.
The six final cases and raw samples/source/compiler/provider identities are in
timings.json. Timing schema v2 adds explicit prepared replay fields to v1.

## Remaining

Active-source/A+B derivatives, composed reverse, symbolic/resident roots,
encoded/fused envelopes and public generic paired artifact dispatch remain open.
No gfx1201/gfx1151/Apple/x86 evidence is inferred.

Final frozen regression gate: 338 tests pass, combining 31 paired tests with
audit/diagnostic/pass gates. Mypy passes both changed runtime/compiler modules.
source-manifest.json records final file hashes separately from benchmark-time
hashes; only a restore union annotation and test assertions changed afterward.
