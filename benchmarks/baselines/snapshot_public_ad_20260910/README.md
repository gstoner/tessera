# Snapshot marking and public asynchronous SSD AD

CUDA RTX 5070 / SM120 and ROCm Radeon 8060S / gfx1151 executed
`benchmarks/record_snapshot_public_ad.py` on their owning WSL hosts. Packets
identify the compiler and allocation, marking, seeded collection, discovered
graph and forward/backward package bindings. Source hashes identify this dirty
working-tree increment; these are not clean revision-bound promotion packets.

Checks cover conservative snapshot survival while roots change, collection of
unreachable records, rejection of a dead marked seed, hook-free discovery of a
plain-instance/list cycle, and public `vjp` dispatch to resident SSD. Capture,
two composed adjoints, external result copies and whole-frame retirement use
three streams. Context synchronization is forbidden through the asynchronous
path. All five composed gradients equal the synchronous paired-program oracle.
This oracle is a composition regression, not an independent mathematical AD
proof; earlier SSD numerical derivative tests remain required.

Private snapshot marking can overlap mutations; final remark/sweep is exclusive.
These short correctness workloads do not measure simultaneous hardware activity
or speedup. Both hosts report WSL. No physical overlap, bare-metal calibration,
Apple/x86 parity, arbitrary object semantics or performance promotion is claimed.
