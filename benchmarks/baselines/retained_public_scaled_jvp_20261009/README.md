# Retained native public scaled JVP — GFX1201

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key: GFX1201-RETAINED-PUBLIC-JVP-20261009.
Shared integration key: PUBLIC-NATIVE-JVP-20261009.

The control is the published public integration from PR934/c39aba390.
The candidate binds that exact compiled Graph/Schedule/Tile/ROCm/LLVM program
to one caller-owned checked C++ execution plan. It validates the copied artifact
and child binding at preparation, then retains private native storage. Warm
calls perform checked input movement, native dispatch and independent readback;
they do not reserialize/decode the artifact or prepare another native owner.
Native numerical algorithms and images are unchanged.

Ownership is bounded to eight prepared programs per native JIT owner.
Creation/cleanup is synchronized; each frame's update/invoke/read is one
transaction. PID checks precede inherited locks and driver access. Explicit
close and native guarded finalization retire handles. A sealed execution
snapshot cannot be rebound through mutation of an exported metadata dictionary.

## Exact-device validation

70 owning RX 9070 XT/gfx1201 tests pass, including 36 numerical/map/activity
profiles, warm validation/preparation refusal, independent callers, malformed
frames, concurrency, pre-driver fork refusal, resealed wrong bindings, snapshot
mutation, garbage collection and existing FP8 scale-JVP regressions.
50 RTX5070/SM120 public JVP/resident-forward regressions pass for the shared
JIT lifecycle changes; no AMD scaled schedule is transferred to CUDA.

## Matched measurement

One recorder executes fresh processes in A/B/B/A order. Each arm has 36
profiles and checks primal/tangent against an independent FP64 oracle before
timing and after every public/event/captured-member sample. Compilers and fresh
HIP providers are identical across arms. Controls pin 23 sources, candidates
24 including the owned execution-plan binding.

| Matched pair | Candidate/control public median range | Geometric mean |
| --- | ---: | ---: |
| control1 / candidate1 | 0.370–0.681 | 0.508 |
| control2 / candidate2 | 0.380–0.659 | 0.514 |

Completed public calls improve by 31.9–63.0% across these named profiles.
Candidate medians span 0.568–1.208 ms; controls span 1.098–2.600 ms.
All 36 native program and member-image digests match in each pair.
Maximum absolute oracle error remains 8.189881861575543e-8.

The interleaved native event ratios span 0.950–1.018 and 0.976–1.071;
no native kernel speedup is claimed. Captured member windows exceed 3.353 ms.
Timing domains retain the baseline definitions: public calls include binding,
uploads, dispatch and completed independent readback; native events exclude
preparation/update/readback and include enqueue gaps; captured members use a
separate grouped pure-SSA diagnostic schedule.

Recorder: benchmarks/rocm/record_retained_public_scaled_jvp.py.
Packets: control1.json, candidate1.json, candidate2.json, control2.json.
Tests: tests/device/rocm/test_public_native_scaled_jvp_transform.py and
tests/device/rocm/test_retained_scaled_jvp_admission.py.

Remaining public overhead, general/mixed-axis composition, generic scaled
batching/transpose closure and quantized AD remain open. This result does not
close the larger five-slice program or establish gfx1151/Apple/x86 parity.
