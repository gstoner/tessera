# Resident DAG and immutable pool snapshots — 2026-09-11

Correctness on Super-Bear RTX 5070 SM120 and Princess-Luna Radeon 8060S gfx1151,
both WSL. Per-backend packets retain compiler/recorder hashes, SSD/collector
identities and every native addition kernel identity. No performance claim.

The recorder holds an immutable snapshot read scope while the live pool retires
an unreachable cross-batch cycle, reuses a slot and updates roots. Snapshot
state remains unchanged. Copies are made under the live writer epoch; subsequent
snapshot reads do not hold the live-pool exclusion. This proves safe concurrent
admission through separate storage, not measured kernel overlap or unrestricted
concurrent access to the same memory. Snapshot close is completion-checked and
synchronous; parent close preserves snapshot ownership.

A three-call SSD DAG reuses four public inputs and fans its first output into
two downstream uses. Public VJP accumulates all paths through native f32 addition
kernels. All five public gradients are compared to central finite differences
of an independent float64 NumPy recurrence. The maximum absolute gradient error
is below 2.8e-9 on each device. These are small correctness fixtures, not tuned
workloads. Copyback explicitly waits on the borrowed view's stream. Whole-program
retirement includes addition outputs and off-thread module unloading.

Run `benchmarks/record_dag_snapshot.py --backend nvidia|rocm --compiler <tool>
--output <packet>` with the owning backend environment.

Separate CUDA pytest checks cover irregular additive negative-infinity masks
with ragged GQA, causal and left-window masking at Q/K=(3,5) and (5,3). Empty
rows refuse before launch. Boolean/broadcast masks are not admitted.

Still open: canonical whole-program composition IR, effectful/control-flow AD,
additional families, same-call input alias admission, unused-input zeros,
arbitrary extension discovery, unrestricted same-storage sweep barriers,
Apple/x86 integration and selector-grade performance evidence.
