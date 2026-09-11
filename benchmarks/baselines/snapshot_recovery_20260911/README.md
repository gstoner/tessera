# Snapshot cleanup during parent recovery

PR #744 follow-up: snapshot cleanup and the parent epoch wait must carry recovery
readiness while closing a poisoned incremental pool. The readiness override is
scoped to cleanup under the owner lock and reset on success or failure. Ordinary
reads still reject poisoned parents; active readers and failed completion proofs
still prevent reclamation. The gated owner does not admit live snapshots.

`record_snapshot_recovery.py` injects receipt-copy exceptions after native pin
and unpin submission, with an existing snapshot. Independent CUDA SM120 and ROCm
gfx1151 packets verify that ordinary snapshot reads reject, then parent close
proves completion and releases snapshot and pool storage. Source/compiler hashes
identify the tested implementations. This is not a driver-hang or performance claim.
