# Independent ANN rewrite tuning — 2026-09-08

Nine independent fixed-count runs per target used a frozen 16x8 affine/ReLU
workload. The incumbent keeps serial lowering; the transformed candidate uses
native elementwise fusion and one independent row per GPU thread. Source and
image replay bind both schedules separately.

| Target | Median package speedup | Median-order-statistic interval | Scoped selection |
|---|---:|---|---|
| SM120 / Super-Bear | 1.0753x | [1.0374, 1.0883] | Rewrite |
| gfx1151 / Princess-Luna | 0.9997x | [0.9657, 1.0170] | Incumbent |

Reproduce with `record_native_ann_execution.py --rows 16 --width 8
--tune-transformed`, seeds 1..9, then `select_native_ann_measurements.py`.
Each run checks native numerical results; scoped arbitration separately binds
source, image, domain, budget and recorder identity and verifies the selected
candidate. The registration is closed afterward.

The 2% lower-bound gate selects the SM120 rewrite for this registered region.
This is warm host-wall package timing including transfers, not kernel timing,
measured overlap or a global route promotion. Wider shapes and nonlinear
families need fresh proof. Earlier `deep_ann_20260908` packets are historical
snapshots and are not refreshed by changing their hashes.
