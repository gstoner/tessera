# gfx1151 native replay cache — 2026-10-08

Owner E2E-REAL-6. Sync ROCM-NATIVE-REPLAY-CACHE-2026-10-08.
Isolated follow-up to draft PR895.

The package verifier retains native MLIR ownership: successful Schedule-to-Tile
and canonicalization outputs are memoized on exact input IR, pass, resolved
compiler, executable plus loaded-library contents, and complete environment.
Every package still validates the artifact, compares replayed Tile text and
projects descriptor fields. Failures and outputs over 16 MiB are not retained.
The cache is bounded to 256 entries and 16 MiB, with LRU eviction.

Princess-Luna's live runtime reports gfx1151. The recorder executes six supported
profiles against independent float64 numerical references before timing.
Four unary candidate profiles use seven alternating A/B pairs with warm native
images. Only ancestry replay caching changes. Images, entry symbols and compiler/
toolchain fingerprints match in every arm. Uncached replay calls make three
subprocesses; cached calls make one (AMD clang device-library discovery).
Version queries are zero in both arms.
Softmax medians: 62.656/63.021 ms uncached, 21.226/21.768 ms cached.
Mean reduction medians: 61.194/61.799 ms uncached, 20.849/20.712 ms cached.
The two fp16 matmul profiles are unaffected controls, including ragged K31:
59.239/58.075 ms versus 59.341/57.905 ms. Their three subprocesses are unchanged.
These are package wall measurements, not kernel speedups.

Focused tests: 229 passed, 196 skipped with production compiler on Super-Bear;
48 passed, 11 skipped with live gfx1151 on Princess-Luna. The native unary
migration lane includes stale storage/keepdims/output rejection after a cache
prime. No skipped row is counted as proof.

Compiler/runtime hashes and current Python provenance for Princess-Luna are in
the separate version-cache follow-up receipt at
/home/angstorms/scratch/pr895-followups/gfx1151-version-cache-20261008/.
The replay cache candidate adds rocm_pass_cache.py and changes
native_unary_contract.py; their hashes are recorded in the recorder packet
and follow-up receipt. The owning compiler is the existing LLVM23.1.1 snapshot;
this is not a fresh matching-source build claim.

Sibling assessment: gfx1201 uses this path but requires owning-device numerical
and timing follow-up. x86 retains its existing cache implementation. NVIDIA and
Apple do not enter this new ROCm cache path; no parity claim.
Open: matmul ancestry caching, discovery overhead, wider unary envelopes, fresh
gfx1151 matching-source build, generic closure and PR895 CI repair.
