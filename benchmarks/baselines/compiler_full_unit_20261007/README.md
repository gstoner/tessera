# Current compiler non-slow unit lane — 2026-10-07

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Status: completed, red. Source was frozen for this run.

Super-Bear WSL, matching core/NVIDIA validation environment:
python -m pytest -m "not slow" -q tests/unit

unit.txt records 3 failed, 21694 passed, 7503 skipped, 874 deselected.
The failures are scaled_matmul batching closure, scaled_matmul transpose
closure, and the new packet directory's missing tracked-document citation.
The citation is corrected in all four tracked backend queues; focused gates
record its result separately. The generic closure assertions and coverage
states are retained. Skipped lanes do not establish owning-device parity.

The paged-KV address candidate was developed on Princess-Luna after its
reference snapshot was preserved. It is not part of this frozen-source sweep.
