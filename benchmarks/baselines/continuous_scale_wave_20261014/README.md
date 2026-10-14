# Continuous scale-adjoint wave member scheduling

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: CONTINUOUS-SCALE-WAVE-20261014.
Based on merged PR944. Published in [PR945](https://github.com/gstoner/tessera/pull/945).

Native Graph-to-Schedule applies an explicit scale-wave request to lhs/rhs
scale adjoints only. Continuous products, matrix gradients and movement
carriers retain serial Schedule/Tile geometry. Continuous four-FP32 wave
members carry matching scale-role metadata on the Tile wrapper and the
sealed GPU body; Tile/Target verification and checked package validation
reject other roles. Existing FP8 geometry and numeric policies are retained.
Automatic selection and the serial default are unchanged.

## Numerical and lifetime proof

Matching LLVM/MLIR 23.1.1 compilers build on Tajasaurus and Super-Bear.
Four matrix-orientation regressions fail against the published compiler and
pass after the member classification and role-bound Tile contract changes.
The focused compiler lane passes 30 cases, including the existing 25 auto
selector cases and a forged-role rejection.
The owning RX9070XT/gfx1201 lane passes 12 public FP32/FP8/MXFP8 gate cases
and 40 additional continuous scale-wave cases: shared/independent/broadcast
roots, four logical matrix orientations, reordered requested gradients,
inverse output seed, changed inputs and retained generations.
Numerical references use independent FP64 products, analytic shared-gradient
sums or finite differences. Warm gate replay forbids compiler subprocesses.
These named cases do not establish generic dynamic, encoded-storage or
higher-order AD closure.

Host WSL focused scaled-AD/registry suite: 3,281 passed.
Audit/recorder lifecycle gates: 12 passed. Ruff and focused mypy pass; all 32
generated documents are in sync and compiler-plan ownership passes. These
are focused gates, not a fresh full non-slow suite for this candidate.

## Isolated balanced characterization

Four final packets run serial/wave/wave/serial, alone on the owning GPU.
Source and binary pins match across all four packets and current source.
Primal/JVP programs and images are identical across arms; reverse identities
match within each recipe. The E8M0 gate-only reverse images also match across
recipes because it has no continuous scale derivative.
Maximum absolute error across the packets is 3.2223833414946057e-06.

| Storage / scale | Serial reverse program (us) | Wave reverse program (us) | Serial public reverse (ms) | Wave public reverse (ms) |
| --- | ---: | ---: | ---: | ---: |
| FP32 / FP32, M7 K37 N19 | 36.564–36.685 | 25.009–25.152 | 1.812–1.852 | 1.767–1.802 |
| E4M3 / FP32, M17 K256 N19 | 3889.289–3900.627 | 167.629–167.998 | 5.832–5.972 | 1.670–1.892 |
| E4M3 / E8M0, M17 K256 N19 | 111.623–112.899 | 111.098–112.110 | 1.528–1.574 | 1.529–1.619 |

Program timings are HIP event windows around 128 native interleaved
repetitions and include host dispatch gaps. Public reverse includes binding,
upload, checked dispatch, readback and independent returned arrays.
The packets also retain separate 2048-repeat grouped member device windows;
those are a diagnostic schedule, not ordinary public execution.
All timing domains are checked numerically before and after each round.

This is named characterization, not a default policy or general public
speedup claim. The continuous public difference is small; E8M0 public
variation remains despite unchanged reverse images. FP8/MXFP8 are included
in the same evaluation. MXFP4 residual AD and its M256 attribution remain
separate open obligations.

Recorder: benchmarks/rocm/record_native_encoded_scaled_gate.py, with
--formats continuous fp32 e8m0 and an explicit --scale-vjp-schedule.
