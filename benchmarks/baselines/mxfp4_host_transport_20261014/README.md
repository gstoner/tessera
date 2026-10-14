# Native gfx1201 host-traffic policy

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key ROCM-PROGRAM-TRAFFIC-20261014.

## Implementation and proof

The native C++ program provider now includes upload **and completed-output
readback** in its automatic pinned-staging decision, at the existing 256 KiB
threshold. Each slab remains capped at 8 MiB. Explicit overrides, checked
capacities, generation/completion, retained outputs and transport-specific
idle-cache identity are unchanged. gfx1151 keeps its prior default; no other
backend policy is transferred.

Eighteen compiled policy cases cover threshold equality, split traffic,
overflow, slab bounds, sibling architecture defaults and overrides. A fresh
HIP provider passes 82 gfx1201 FP8/MXFP8/MXFP4 public/numerical/cache/lifetime
cases, including an output-dominated MXFP8 case that fails the preserved
baseline's mode/cache assertion. 87 focused package/policy gates pass.
The required broad lane found two tracer element-spelling regressions; the
implementation is repaired while the original f32 assertion is retained.
209 cast/native-MXFP4/dtype gates and 13 post-fix device cases pass.

## Matched experiment

[Build manifest](build.json) records native compiler, identical optimization
and link settings, shared source/dependency hashes and both provider hashes.
The baseline runtime source is preserved from commit `690f403a7`.
Four isolated baseline/candidate/candidate/baseline packets query the active
RX9070XT/gfx1201 UUID and record the actual provider build sources:

- [baseline 1](baseline1.json)
- [candidate 1](candidate1.json)
- [candidate 2](candidate2.json)
- [baseline 2](baseline2.json)

Shared frontend/compiler source hashes and compiler/image-cache binaries
match across all four. Every primal, JVP and reverse program manifest and
member image group is identical. The provider policy/binary is the intentional
difference. Every timing domain rechecks independent numerical outputs and
forbids warm compiler subprocesses. Final packets include the corrected tracer.

Captured member windows and program-event windows remain distinct from
completed public calls. The table shows ranges of the two per-packet medians,
each based on seven completed public samples, in **milliseconds**.
These are matched-shape comparisons within each row; formats are not compared
as interchangeable workloads or quality levels.

| Format | M×K×N | Public call | Baseline ms | Traffic policy ms |
| --- | --- | --- | ---: | ---: |
| fp8_e4m3/fp32 | 17×256×19 | primal | 1.957–2.025 | 2.007–2.013 |
| fp8_e4m3/fp32 | 17×256×19 | JVP | 1.326–1.376 | 1.383–1.415 |
| fp8_e4m3/fp32 | 17×256×19 | reverse | 5.727–5.751 | 5.796–5.849 |
| fp8_e4m3/e8m0 | 17×256×19 | primal | 1.934–1.945 | 1.950–2.018 |
| fp8_e4m3/e8m0 | 17×256×19 | JVP | 1.172–1.188 | 1.297–1.345 |
| fp8_e4m3/e8m0 | 17×256×19 | reverse | 1.487–1.594 | 1.560–1.626 |
| mxfp4_w4a8_exact/e8m0 | 17×64×19 | primal | 1.860–1.882 | 1.882–1.944 |
| mxfp4_w4a8_exact/e8m0 | 17×64×19 | JVP | 1.144–1.193 | 1.154–1.208 |
| mxfp4_w4a8_exact/e8m0 | 17×64×19 | reverse | 1.515–1.563 | 1.524–1.528 |
| mxfp4_w4a8_exact/e8m0 | 256×128×80 | primal | 3.062–3.331 | 3.373–3.482 |
| mxfp4_w4a8_exact/e8m0 | 256×128×80 | JVP | 3.677–4.152 | 1.279–1.298 |
| mxfp4_w4a8_exact/e8m0 | 256×128×80 | reverse | 3.431–3.907 | 1.647–1.671 |
| mxfp4_w4a8_exact/e8m0 | 256×128×160 | primal | 3.430–3.467 | 1.984–2.093 |
| mxfp4_w4a8_exact/e8m0 | 256×128×160 | JVP | 1.283–1.407 | 1.309–1.410 |
| mxfp4_w4a8_exact/e8m0 | 256×128×160 | reverse | 1.652–1.684 | 1.626–1.630 |

N80 paired AD crosses the combined-traffic threshold and improves. N160
ordinary primal now crosses it too. N80 ordinary primal remains below the
threshold and pageable; further small-traffic evaluation remains open.
Unselected control public medians vary, including a higher MXFP8 JVP median
in the candidate packets. Separate 50-call host profiles have overlapping
1.64–1.68 ms baseline / 1.65–1.72 ms candidate diagnostic totals and locate
substantial repeated Python Graph deepcopy/admission work. Profiling overhead
is excluded from the table. No generic public speedup or zero-overhead claim.

The original Radiance per-column **kernel** gap, broader transfer thresholds,
warm metadata-admission cost, generic layouts/dynamic identity, scale/higher
AD and the five compiler slices remain open. No kernel schedule or numeric
default is promoted.

Reproduce both providers with the flags and shared dependencies in
`build.json`, then run the existing recorder in fresh processes. The baseline
command must pass `--provider-source <preserved-native_program_runtime.cpp>
--provider-source-revision 690f403a7`; the candidate uses the current source.
Both use `--formats fp32 e8m0 mxfp4`. Set
`TESSERA_ROCM_NATIVE_MOVEMENT_LIB` to the corresponding provider and leave
`TESSERA_ROCM_PROGRAM_PINNED` unset.
