# Native three-root MLP integration candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-MLP-20261020.
Base: merged PR993, 1b80c6db1ce2d43fbab27b60f4c4dd6df9aa6b31.
Status: locally validated implementation; exact-head CI and review remain open.

The actual frontend captures X @ Up, ReLU, and hidden @ Down as three stages.
Both matmuls request their output precision explicitly. FP16 intermediates use
two tensor-core members. FP32 intermediates retain FP32 activation storage and
use the existing ordered scalar-FMA consumer, with no half/TF32 conversion.
FP16 roots and BF16 roots are covered; the explicit-half BF16 profile uses an
FP16 Down operand to match its declared intermediate storage.

The initial four FP32 compiler cases fail admission. The first repair admits
them but exposes missing scalar-member launch metadata; the preserved initial
green log also catches a fixture that omitted explicit FP32 output precision.
The final 115-case compiler lane passes all MLP and adjacent program tests.
The shared diagnostic/pass/dtype/operator drift lane passes 470 cases.
A separate frame-capacity/JIT-contract/audit-lifecycle lane passes 40 cases.
Overlapping runs are not added into a total.

Matching RTX5070 device proof passes 26 cases: aligned/ragged two-matmul
programs, normal/reordered roots, both intermediate precisions, changed inputs,
retained outputs, stable owner allocation, close/rebind and named artifact
replay. Separate native FP32 ReLU leaf diagnostics preserve signed zero,
NaN payloads, infinities and subnormal bits in serial and flat schedules.
The leaf diagnostics are not a claim of standalone FP32 public JIT coverage.
Device identity and compiler source hashes accompany the timing packet.
Compiler SHA256: a2cc0401d531bc1599bb7ce917c2f676be5eb1b54235b646079d5304b59d824a.

Eight correctness-gated rows now record cold calls, seven completed-public
samples of eight calls, and independent 64-replay CUDA-event/captured-event
program and stage windows. Completed-public medians span 0.2743–0.4766 ms;
cold calls span 2.924–3.711 seconds. Captured program windows span
4.438–13.792 microseconds; enqueued program windows span 24.112–44.172
microseconds. These tiny-shape windows are distinct domains, not additive
components or claims of pure-kernel speedup. The FP32 scalar consumer is not
compared as the same numerical workload as the FP16 intermediate. Raw recorder
and eight-row source/compiler/device-bound packets are preserved alongside.
Broader regression, reproducible recorder integration and delivery gates
remain in progress.
No cross-precision speedup, generic W1.1 closure, sibling physical support,
dynamic/layout/AD/encoded coverage or five-slice completion is claimed.

## Wider finite-input numerical gate

The 64x256x128x64 source exposed four FP16-intermediate failures under the
initial fixed relative/absolute output comparison. Diagnostic prefixes retain
the original compiler-projected member images: 7–11 stored producer values
differ from exact-dot-then-half rounding, ReLU is bitwise equal to activation
of that native producer, and downstream error against the actual producer is
0.96–1.18e-6. Raw failures and native-prefix attribution are retained in
mlp-numerical-attribution.json and its frozen attribute_mlp.py recorder.

The corrected independent oracle encloses each FP32 dot with
gamma_k = k*2^-24/(1-k*2^-24) times the absolute dot, including gradual
underflow. Monotone rounding of its endpoints through the declared intermediate
store preserves half-midpoint ambiguity. ReLU and the RHS-sign-aware final
linear interval propagate that enclosure; the consumer includes its own
FP32 gamma_h bound. Inputs must be finite and intermediate stores nonoverflowing.
This is a per-element numerical bound, not a fitted global tolerance.
Negative controls reject a hidden half conversion in an FP32 source and
changed arithmetic. The oracle/compiler lane passes 25 cases.
The expanded exact-device lane passes all 87 cases (34 MLP/leaf cases,
45 native owner regressions and 8 source packages). Owning gfx1201 sibling
scalar packages pass 67 cases. Production compiler proof passes 2619 cases
with zero skips; 25 focused MLP/oracle cases cover its final two added negative
controls. The repeated full CPU lane passes 30785 cases with 3710 declared skips.
Exact-head CI and review remain open.

Reproduce from the owning checkout after sourcing the matching compiler and
native provider environment:

    python -m benchmarks.nvidia.record_native_tensor_mlp --output /tmp/mlp.json

The CLI validates recording counts before probing hardware, verifies actual
SM120 identity and matching compiler SHA, records all actual frame geometry,
IR/image digests, source hashes and separate timing domains. The initial
record_mlp.py remains a frozen raw recorder; use the repository CLI above for
new measurements.

## Final reproducible CLI packet

mlp-cli-timing.json is complete with 12 rows over all three shapes. Every
timed row passes the independent enclosure before/after public and captured
profiling. Completed-public medians span 0.2694–0.4583 ms, cold calls
2.959–3.539 s, captured program windows 4.559–27.696 us and enqueued program
windows 25.184–56.261 us. Per-stage timings and actual launch geometry are in
each row. These domains are not additive, and precision profiles are not
interchangeable performance controls. No schedule/default promotion is made.
The earlier eight-row packet is historical and is not merged with these samples.

All five surface audits and README claim checks pass. Ruff and zero-error
Mypy pass. Generated documents were regenerated by their owning generators.
The initial large-shape failures and numerical attribution are retained.

Raw pytest/stdout logs containing trailing spaces are preserved losslessly as
gzip files. lossless-raw-logs.json binds original uncompressed and gzip SHA256
values; decompression reproduces the original diagnostic bytes.

## Final local delivery gates

The first full CPU lane had two failures: the former FP32 activation refusal
fixture and unnamed frozen diagnostics. The fixture now proves FP32 lowering
while still rejecting F64 and sibling architectures; the recorders are indexed.
The complete repeated lane passes 30785 with 3710 declared skips.
The raw initial failure is preserved losslessly with its uncompressed SHA.
The final staged citation/audit/surface gate passes 50, the applicable
registry/MLP lane passes 495, and final activation/manifest checks pass 52
with 8 declared non-SM120 skips. Overlapping runs are not added.

Eight additional exact RTX5070 resident cases prove three independent
producer streams, no host coercion, no warm tools/tracing, stable retained
allocation and completed output lifetime for both root/storage orders and
intermediate precisions. These supplement the earlier 87-case device lane.
Compiler arithmetic and the native provider are unchanged between these runs.
Exact-head GitHub CI/review and aggregate five-slice closure remain open.

PR994 review follow-through completes standalone typed activation and its
public/runtime/registry boundary; see review/README.md. It preserves the
no-growth Python packager gate and target-specific census evidence.
