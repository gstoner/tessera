# Checkpoint identity review follow-through — 2026-10-16

Review base: 3d6df4a5330ab8c988af46821037c5271bf60de9.
Synchronization key: SM120-SAVED-LSE-STORAGE-20261016.

Thirteen regressions reproduce missing storage propagation before the repair.
The generated forward-only wrapper now carries the native artifact's actual
storage into its checkpoint identity. Scheduled AD pairs explicitly require
their existing FP32 storage contract. Every production identity call supplies
source/output storage; a source-wide compiler call-site audit holds that rule.
The legacy resident AD tape validates its FP32 data/gradient roles separately
from the delivered typed public forward owner.

All 168 focused wrapper, identity and bounded-contract tests pass, and mypy
reports zero errors. Full unit, production-route and owning-device validation
for this changed Python contract are being completed before the new PR head.

The native compiler source tree and native provider are unchanged. Earlier
timings and receipts retain their original source scope. source-manifest.json
binds the changed source and exact pre-review snapshots, so earlier hashes
are checked against their retained bytes rather than rewritten as current proof.
The five-slice aggregate, half-storage AD and sibling physical integration
remain open.
