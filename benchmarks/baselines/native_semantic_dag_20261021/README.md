# Native semantic DAG ownership — RTX5070

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-SEMANTIC-DAG-20261021.

The frozen cc3e73c8 compiler owns Graph → Schedule → Tile → NVIDIA Target →
NVVM/PTX members and a checked native frame for one-root semantic DAGs.
The runtime provider is unchanged. Exact GPU, tool, provider and source hashes
are in `semantic-dag-timings.json`; compiler source changes are preserved in
`semantic-dag-candidate.json`. The source checkout is merged PR1000 plus the
recorded compiler/Python/test overlay; this is candidate evidence.

All 48 owning RTX5070 tests pass. They cover RMSNorm→ReLU, softmax→RMSNorm,
LayerNorm→softmax and a ReLU/add fork, f16/bf16/f32, static and bounded rows,
strict source replay, output canaries, immutable inputs, retained outputs,
one arena across M=1/17/65/5, over-capacity refusal, and ordinary public JIT
with eager/tracing/compiler calls disabled on warm execution. The numerical
oracle uses float64 formulas and rounds each semantic intermediate to its
physical storage. This establishes the tested tolerance envelope, not bitwise
identity for transcendental/reduction arithmetic.

The merged control fails all 36 new compiler ownership cases. The candidate
passes 213 focused compiler/JIT cases, 23 recorder guards and 11 audit lifecycle
cases. The initial expanded compiler command collected incompatible standalone
numerical suites and duplicate device/unit module names; that invocation is
invalid as lane evidence. The actual CI tests/unit selector passes all 2,930 selected cases with zero skips (278.84s); its log is retained. The two later-added fused-owner controls pass separately in the 44-case semantic cohort.

## Timing

All 36 rows completed with independent numerical checks before/after timing.
Completed public medians range from 0.386439
to 0.496284 ms. Each row contains six
samples of sixteen completed calls, including binding/transfers/readback and
excluding compilation/prepare. One cold sample per case/storage is repeated
across its three active-M rows and is not three independent observations.

Only 334/432 device event windows meet the 10ms minimum across
all independently measured member/program regions. Short windows are retained
as diagnostic and must not support a performance claim. Captured graph and
uncaptured launch measurements are separate; member/program regions are not
additive. Images and arena capacity remain unchanged across M=1/17/65.
No baseline speedup, default promotion or aggregate program closure is claimed.

General dynamic axes/layouts, noncanonical accumulators, control flow, AD
composition, all four sibling physical routes and all five original programs
retain their independent follow-ups. Two fused-matmul epilogue controls additionally preserve their existing single-kernel owner. The initial control omitted final Target lowering; the corrected complete pipeline passes all 44 semantic ownership/control cases. Ruff passes. The full compiler lane, final changed-file Ruff and packet integrity pass. The broad CPU unit lane and local publication gates subsequently pass, as recorded below. Public exact-head CI/review and merge remain pending.

## Resident follow-up

Eighteen additional owning RTX5070 cases pass with deliberately delayed
resident uploads on explicit, legacy-default and per-thread-default streams.
RMSNorm/ReLU and fork/add run in f16/bf16/f32 without warm tracing/compilation;
results survive owner/session teardown and source buffers remain unchanged.
This is stream-ordering evidence, not timing evidence.

The recorder's source hashes describe the measured snapshot. The preserved
`semantic-dag-source.tgz` contains its device helper before the resident-only
test addition; oracle and timed functions are unchanged by that addition.
Both the original 48-case and separate 18-case result logs are retained.
The final audit/recorder cohort passes 34 cases after document regeneration.

## Final local delivery gates

The host WSL CPU lane passes **30,931 cases**, with **3,721 declared skips**,
zero failures/errors and eleven optimizer warnings (454.87s). Lossless compressed
stdout/JUnit evidence is retained; XML totals are independently checked.
The actual CI compiler selector passes 2,930 cases without skips; the later
fused-owner controls pass separately. All five surface audits, both claim-lint
gates, project-wide Ruff, zero-error mypy, 33 generated documents and the
42-case final routing/lifecycle/recorder cohort pass. The initial final-drift
command named a nonexistent test module; the actual routing module passes in
the separate final-routing log. No failed invocation is called validation.
Public exact-head CI/review and merge remain pending.
