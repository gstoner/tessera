---
last_updated: 2026-10-19
audit_role: reference
---

# Bounded public original-source reverse evidence

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.
Contract: docs/spec/NATIVE_BOUNDED_SOURCE_PUBLIC.md.

## Exact-device and host-free proof

The selected production provider runs on NVIDIA GeForce RTX5070 / SM120,
UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver610.88 / CUDA13.3.
Frozen LLVM/MLIR23.1.1 compiler SHA256:
0e8161de4555ae0626b19d2d79f509fc42fd6ef9f1b5f8178541dded7744e64c.
Production provider SHA256:
df4552fafd31f19fe03c858fba2c7da3e9310abed3076a50fd7c8bf8906d25ac.

- bounded-prepared.xml: 24 prepared cases, both formats, explicit inline/row
  decompositions, normal/permuted/swapped roles and two capacity/width profiles.
- public-bounded.xml: 30 public cases: 12 bounded shrink/grow, four bounded
  fresh-process portable replay and 14 pre-existing static public regressions.
- public-prefix.xml: two additional B=2,Hq=4,Hkv=2 shrink/grow cases with selected gradients; total public coverage is 32 distinct cases (18 bounded and 14 static).
- bounded-python-regression.xml: 241 portable/frame cases. Earlier subsets
  and reruns are not additional distinct evidence.
- bounded-public-shared.txt: 346 plugin/portable-auto/diagnostic/pass cases.

Device tests use independent source-VJP oracles and check one retained owner/
images, trace at Sq=Sk=1, serial/cooperative norms, independent output lifetime,
readonly roots, five resident producer streams, invalid capacity refusal with
compilation forbidden and cleared failed-call export. Four fresh children
disable compiler paths and external subprocesses, restore serialized programs,
retain one owner and replay three changing sequence frames numerically.

The unit portable fixtures use signature-only PTX stubs. They prove offline
contract checking, not GPU execution. Actual compiled image execution belongs
only to the device cases above.

## Paired public measurements

Recorded by benchmarks/nvidia/benchmark_bounded_public_source_reverse.py.

Reproduce on the owning SM120 host with the matching compiler/provider selected:

    python benchmarks/nvidia/benchmark_bounded_public_source_reverse.py --output public-order-a.json
    python benchmarks/nvidia/benchmark_bounded_public_source_reverse.py --reverse --output public-order-b.json


public-order-a.json and public-order-b.json each contain 12 profiles with
21 alternating samples and eight completed calls per arm/sample. Profile,
initial arm and post-trace frame ordering are counterbalanced. Both orders
trace the bounded program at one. The inputs match between arms/orders.
Every arm is checked against an independent oracle before and after timing.

Bounded/static completed-public median ratios span 0.942-1.017 in order A
and 0.918-1.050 in order B. Bounded owners stay at one per capacity/profile;
static owners grow to three across three sequence frames. New static-frame
completed host windows include compilation/prepare (2.56-3.52 seconds);
bounded replay of new in-capacity frames is 0.92-2.31 milliseconds.
These are compilation-inclusive first-frame windows, not pure compile timings.

Raw warm completed host samples and separately last-call program/stage CUDA
event samples remain in the JSON. Native events can include enqueue gaps and
are not isolated profiler kernel timings. Capacity-based automatic selection
may differ from static selection on tiny frames. Mixed results, including the
5% warm regression, are retained; no universal scheduling promotion follows.

Benchmark source fingerprints bind the measured snapshot. Later edits only
rename portable validation locals to resolve mypy inference, add explicit
missing-role rejection and document the JIT argument; final-source numerical
and delivery gates are recorded separately. No compiler or provider changed.

## Failures and delivery gates

bounded-portable-red.txt preserves the original unsupported-schema refusal.
bounded-portable-first.txt exposed an incorrect validator assumption that Tile
rows carry sequence_bound; the native Tile contract instead carries row capacity
and row_multiple. bounded-portable-second.txt exposed an accepted rehashed
module-capacity tamper. The validator now checks each module/member boundary.
mypy-before-fix.txt retains four variable inference errors; the baseline remains
zero and the fix renames locals and explicitly narrows optional capacity roles.

Production-route proof passes 2,459 tests with no skips; require_executed passes.
Mypy remains at zero errors; Ruff, 22 document lifecycle/ownership tests and all
five surface audits plus claim lint pass. All 33 generated views regenerate.
The first full CPU run passed 30,770 and skipped 3,710, with one recorder naming
failure: this new recorder lacked an explicit tracked README citation. The
Recorded by line and reproduction commands above fix that ownership omission;
the empty-orphan ratchet is unchanged. Its focused rerun passes, followed by
a clean full CPU run: 30,771 passed, 3,710 skipped, zero failures (cpu-final.txt
and cpu-final-summary.json). The final generated drift check confirms all
33 views in sync. Two added owning batch/grouped-head cases pass separately;
no compiler, provider or production Python implementation changed after the
clean CPU run. The cumulative source review found no actionable issues.
This packet does not close generic source/layout/AD, mixed/quantized attention,
representative model/corpus quality or ROCm physical performance obligations.
