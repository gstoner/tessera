# Broader scoped ANN comparisons — 2026-09-08

Two frozen workloads extend the prior 16x8 affine/ReLU experiment:
8x4 affine/absolute value and 32x4 affine/ReLU. Each uses nine independent runs,
31 randomized alternating samples per run, and five warmups. The transformed
candidate alone uses elementwise fusion and row parallelism. Both candidates
retain native source/image replay and analytic domain/error-budget checks.

Run `record_native_ann_execution.py --rows 8 --width 4 --activation abs
--tune-transformed` or `--rows 32 --width 4 --activation relu --tune-transformed`,
seeds 1..9; feed those reports to `select_native_ann_measurements.py`. Per-target
`selection.json` records the confidence interval and scoped decision. Registrations
are retired after oracle-verifying the selected candidate.

Timing is warm host-wall package latency including transfers. It does not measure
kernel time, clean native-OS performance, overlap or global production promotion.
The lower bound must clear the fixed 2% margin; otherwise the incumbent remains.
Historical packets are not refreshed by editing their source hashes.

| Target | Workload | Median speedup | Median interval | Scoped selection |
|---|---|---:|---|---|
| nvidia | 32x4-relu | 1.0136x | [0.9906, 1.0412] | Incumbent |
| nvidia | 8x4-abs | 1.0086x | [0.9784, 1.0120] | Incumbent |
| rocm | 32x4-relu | 0.9942x | [0.9801, 1.0049] | Incumbent |
| rocm | 8x4-abs | 0.9934x | [0.9881, 1.0026] | Incumbent |
