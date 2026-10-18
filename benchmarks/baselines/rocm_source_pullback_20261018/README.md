# Native public ROCm source pullback

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: ROCM-SOURCE-PULLBACK-20261018.
Contract: [ROCM_SOURCE_PULLBACK](../../../docs/spec/ROCM_SOURCE_PULLBACK.md).

## Implemented route

The original static FP16/BF16 affine-free RMSNorm → ReLU → plain FP32-output
matmul source VJP follows frontend Graph → native paired AD → native SSA
member Graphs → Schedule → Tile → ROCm Target → ROCDL/LLVM → HSACO.
Native Schedule and Target exports own the scalar and row launch frames.
One checked C++ HIP owner retains readonly roots, private intermediate storage
and returned generations. Python validates those records and binds buffers;
it does not reconstruct derivative arithmetic or launch individual members.

Ordinary native_backward on exact rocm_gfx1151/rocm_gfx1201 JIT targets
retains a certified native owner. Common-runtime portable replay borrows that
owner, or validates and prepares a temporary owner after it is closed.
Parent Graph, exact target, named roots, result placement and image/ABI
lineage are checked before provider access. Returned host arrays own snapshots.
The generic rocm alias, resident roots, dynamic shapes and arbitrary layouts
are not admitted by this composite route.

## Evidence scope

[validation_summary.json](validation_summary.json) records the final gates:
30,706 CPU passes (3,724 declared skips), 1,983 compiler-route passes with zero
skips, 106 device passes on each AMD host, and 167 RTX5070 sibling passes.
Selections overlap and are not a sum of independent cases. Forty NVIDIA
native member-stage digests remain identical to the pre-slice baseline.
Ruff, the zero-error mypy ratchet, all five surface audits and claim lint pass.

[gfx1201.json](gfx1201.json) and [gfx1151.json](gfx1151.json) bind exact live
GPU identities, compiler/provider bytes, retained image hashes and separate
timing domains. The numerical receipts report zero per-element error-budget
violations across sixteen changed-input/cancellation cases on each GPU.
Eight compressed portable programs retain the full checked native lineage;
they restore without compiler calls and match the timed full-package hashes.
Root permutations intentionally reuse physical images; their full package
and SSA root-role identities remain distinct. Each owning host runs the same
candidate Python source and compiler executable bytes. Path-sensitive compiler
cache identities are recorded separately from raw executable and loaded-library
SHA256 values; matching executable bytes alone do not assert identical host
loader environments.

The numerical envelope includes half storage, FP32 cotangents, argument
permutations, changed roots, independent outputs, cancellation witnesses,
one-column reductions, row-block crossings and ragged contractions.
The independent FP64 oracle accounts for the AD Graph's explicit half
intermediate store and FP32 reduction error. Finite Gaussian regimes avoid
ReLU storage-underflow ambiguity; no exhaustive rounding/nonfinite AD
certificate is claimed.

The final oracle file removes one trailing blank line after measurement.
[source_format_reconciliation.json](source_format_reconciliation.json) verifies
identical Python AST and reconstructs the exact measured bytes; current and
measured source maps are both retained. No numerical expression changed.

## Timing regions

- Captured member events include device-graph member dispatch; capture and
  instantiation are outside the measured region.
- Native program events cover an ordinary native stream launch-loop window,
  already divided by repeats. They may include CPU enqueue gaps.
- Prepared host calls complete root update, invoke and readback.
- Portable host calls deserialize, validate, prepare, invoke, read and close.
- Warm public host calls execute ordinary native_backward with the retained
  owner and certificate/dispatch work. Warm tests prohibit compiler calls.
  First-call records follow image priming and are not cold-process costs.

Five windows and 128 event repetitions characterize each of four named
storage/root-order profiles per GPU. These regions are distinct, and no
isolated-kernel latency, comparative speedup or default-schedule promotion
follows from the packet.

## Recorded named-profile medians

| Host / exact GPU | Native program event window | Prepared update/invoke/read | Warm public native_backward | Portable completed call |
|---|---:|---:|---:|---:|
| Tajasaurus / RX9070XT gfx1201 | 14.81–17.71 us | 0.39–0.46 ms | 1.20–1.30 ms | 11.54–11.83 ms |
| Princess-Luna / Radeon8060S gfx1151 | 9.66–10.56 us | 0.38–0.41 ms | 1.29–1.58 ms | 13.00–13.41 ms |

## Remaining scope

Broader frontend/AD integration, numerical regimes, dynamic/resident/layout
envelopes, representative-model quality and the original five aggregate
programs remain open. This native source-pullback increment closes none of
those aggregate plans.
