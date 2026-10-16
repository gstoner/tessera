# PR958 producer thread ABI verification

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-NATIVE-ROW-CORRECTION-20261016.

Review identified that the shared Tile verifier accepted 64/128-thread row
producers, although the native owner always launches this producer with 32.
The target index stride follows the accepted attribute, so the mismatch can
skip rows. Native producer geometry is now exactly 32; consumers remain 64/128.

Twelve direct Tile cases remove the optional sealed parent dictionary so the
operation verifier is exercised directly. FP16/BF16 producer 32 passes,
producer 64/128 rejects, consumer 64/128 passes and consumer 32 rejects. The old
compiler accepts all four invalid producer cases (4 failed/8 passed red receipt).
The rebuilt three-driver focused lane passes 420 cases, including all 12 direct
cases, native target/consumer and portable/diagnostic/pass metadata controls.
116 owning RTX5070 public/prepared device tests pass; all green lanes have zero
skips. Audit lifecycle passes 11. Source/tool/provider identities are retained.

All four backend plans assess this shared verifier correction. No sibling
physical consumer, runtime ABI or schedule is added. Existing generated legal
geometry, arithmetic and runtime implementation remain unchanged; the earlier
benchmark packets retain their original measured-source scope.

The complete production compiler-route lane passes 1,237 tests with zero skips;
ci_require_executed confirms every selected proof executed. Exact-head CI on
the updated PR remains required before merge.

Affected audit/spec/ODS-consumer modules pass 791 cases with zero skips.
