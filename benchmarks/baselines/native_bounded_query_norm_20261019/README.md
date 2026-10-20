# Native bounded-query RMSNorm prerequisite

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

This private increment implements the forward normalization member needed by
bounded original-source attention reverse. It does not implement the composed
bounded reverse program or close an aggregate five-slice requirement.

## Compiler and exact-device evidence

The matching compiler is bound by DECISION.json and both device packets.
The member preserves dynamic Sq through native Graph/Schedule/Tile/NVIDIA
lowering, with fixed B/H/D capacities and guarded runtime row geometry.
No production Python kernel or derivative construction is added.

- 23 normalization compiler contract cases pass in the mixed cohort.
- 510 mixed compiler/registry/audit/device tests pass; 15 declared skips cover
  foreign gfx1151 and unavailable Apple hardware. Twelve executed owning
  gfx1201 source-reverse tests are included in the 510, not additional.
- Eight RTX 5070 FP16/BF16 serial/cooperative device cases pass independent
  numerical, exceptional-value, signed-zero, tail, root and invalid-geometry
  checks across shrink/grow reuse.
- Twelve static RMSNorm/LayerNorm profiles retain identical four-stage artifacts
  against the immutable row-prerequisite compiler. Artifact equality does not
  constitute a new device certificate.

The first focused run ended nonzero despite 498 passed/15 skipped: its selected
ROCm lane was hollow on this device. The corrected mixed run retained every
selected test and added owning gfx1201 execution; no gate was weakened.
Earlier target-boundary failures and initial logs are preserved as diagnostics.

## Timings

Both profile orders pass 36 profiles each, with 21 alternating trials and
32 compiled kernel nodes per captured graph. Inputs and all compiler-stage
hashes match across orders. Correctness and raw-byte parity precede and follow
timing. Repeated orders are repeatability evidence, not 72 distinct envelopes.

The recorder retains ordinary event dispatch (including Python enqueue gaps),
graph dispatch divided by its checked 32 kernel nodes, and completed raw host
launch/synchronization separately. Graph replay is a diagnostic timing method,
not a new production execution route or isolated hardware-counter kernel time.
Actual occupancy uses the real 128-thread block. See DECISION.json for both
orders and all domains; no broad performance promotion follows.

Original ordinary-only packets and recorder are preserved. Raw receipts and
the static parity reproduction script are compressed without discarding their
content. Full source/public/portable integration and full delivery gates remain
open.

## Cooperative short-window outlier

The reversed 32-node order measured bounded/static 1.2199 at FP16,
capacity [1,4,257,257], runtime Sq=1. The original samples remain in this packet.
Both functions report 12 active blocks per SM, 512 shared bytes and zero local
bytes. A targeted repeat retained the frozen compiler/images and increased
capture length to 256 nodes with 31 alternating trials in both orders, including
Sq=1,3,257. All six graph-dispatch ratios were 0.9991-1.0000. The initial
regression was not reproduced; this does not establish universal equivalence.
The ordinary timing domains from the repeat also remain available.
