# Native padded resident RHS, 2026-10-07

Owner W1.1 / E2E-REAL-6. Source is integrated into the unpublished aggregate;
matching aggregate runtime rebuild and integrated validation remain pending.
The recorded candidate runtime is built on the owning Super-Bear host; its
hash and compiler/source identities are in timing.json.

The existing dynamic strided consumer receives the RHS leading dimension from
its physical view. Native validation checks minor-axis density, positive major
pitch, overflow-safe physical span, actual CUDA allocation capacity and
output/intermediate disjointness against that entire span. Offset views retain
their allocation owner. Source, output, residual and intermediate retain their
existing compact ABI. Static consumer images do not acquire a new scalar ABI.

## Exact-device evidence

RTX 5070 SM12.0, CUDA 13.4.59. 249 candidate gates pass: 55 padded layout/
invalid-span cases plus 57 compact resident and 137 host-owner regressions.
24 additional changed-weight and active-shape cases replay through one owner
and update the same padded allocation. Independent decoded numerics, unchanged
backing bytes, allocation overflow/offset checks and recovery are tested.

48 alternating same-image rows cover three producers, FP16/BF16, both RHS
orientations, two shapes and fused/unfused consumers. Numerical checks precede
and follow each arm. Component CUDA event dispatch windows remain separate
from synchronized call walls. Control/native wall ratios span
0.979-3.011, median 2.514.
One row remains about 2.13% slower. No selector promotion
or isolated-kernel gain is claimed. Candidate source hashes are verified
after timing; relocated tests/recorder retain their original source snapshots.

## Cross-backend assessment and remaining work

This CUDA ABI is NVIDIA-specific. gfx1151/gfx1201 HIP owners, Apple Metal and
x86 CPU execution retain separate storage/physical evidence; no parity is
inferred from these packets. General producer composition/AD, padded source/
output/residual layouts, negative/minor-strided views and asynchronous result
lifetime remain open. Generic scaled-matmul closure and publication are open.

## Matching aggregate runtime proof

The aggregate native CMake rebuild succeeds. The matching-runtime focused
lane passes 581 tests across padded/compact resident calls, native host
ownership and required registry/lifecycle/recorder gates.
The canonical recorder passes all 48 same-image A/B rows; post-timing source
and runtime fingerprints are verified. Control/native resident wall ratios
span 0.965-3.035, median 2.440.
No isolated-kernel gain or universal selector promotion is claimed.
The worst matching-runtime row is 3.62% slower; this prevents a universal
performance claim. Raw receipts: integrated-gates.log, integrated-timing.json/log
and integrated-fingerprints.json. Generated documents are in sync (32 files),
and the fresh post-integration Graphify gate finished with exit 0. Full-unit generic closure and publication remain open.
