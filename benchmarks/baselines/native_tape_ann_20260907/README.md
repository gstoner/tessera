# Native data-dependent tape and nonlinear ANN evidence — 2026-09-07

Owners: F3 / FA-1 / AD-RESIDUAL-EVAL-1 / W2.4a / IR-NATIVE-FOUNDATION-1.

## Correctness and ownership

Both owning WSL hosts passed six cases: four data-dependent while exits (0, 1,
2, and 3 steps), plus true/false standalone predicate branches. Each case submits
two derivative generations on distinct caller-owned streams, checks both against
independent derivatives, verifies that saved residuals remain unchanged, polls
completion owners, and explicitly releases only those derivative allocations.
The reports record 64 derivative bytes released per two-input while case and
32 per one-input predicate case. This is logical allocation accounting, not
allocator peak consumption. Predicate residuals use i8; counter checkpoints use
i64. Closing a frame invalidates exported views.

- `tape-next-nvidia.json`: Super-Bear, RTX 5070 / sm_120, CUDA 13.3.
- `tape-next-rocm.json`: Princess-Luna, Radeon 8060S / gfx1151.
- Recorder: `benchmarks/record_native_tape_extensions.py`.

This proves bounded static-shape native products, not arbitrary while/CFG
recovery, shape-varying allocation, asynchronous reclamation of arbitrary external
readers, kernel overlap, Metal tape binding, or x86 tape execution. Generation
release retains a context completion barrier; event polling alone does not free
exported allocations. Both LLVM builds have assertions disabled.

## Nonlinear ANN execution and package performance

Each target independently compiled the original two-affine-layer program and
its native one-layer rewrite, both with an unchanged terminal ReLU. Nonsplat
weights exercise native constant materialization. Four input probes, including
negative and mixed-sign values, pass independent rational-oracle error bounds
before and after timing. Admission binds the declared infinity-norm input domain
(1.0), absolute rewrite budget (0.001), native source replay, and target package.

Timing uses **nine independent processes**, each with 31 randomly ordered paired
samples after warmup. Each measured call includes stable-input copying, domain
checks, H2D, native dispatch/synchronization and D2H, using reusable device buffers
and compiled bindings. Compilation and initial ancestry verification are outside
the timed call. These are host wall-clock package measurements, not GPU kernel
clocks or counter attribution. No comparison crosses architectures.

| Target | Median speedup | Exact median bounds | Performance eligible | Production promoted |
|---|---:|---:|---|---|
| CUDA sm_120 | 1.00447× | [0.98833×, 1.02162×] | No | No |
| ROCm gfx1151 | 1.00243× | [0.99360×, 1.01223×] | No | No |

The existing exact order-statistic method provides at least 95% one-sided
coverage for each endpoint. Nine runs select the second-smallest/largest values,
so one extreme run does not alone set an endpoint. The lower bound must exceed
1.02×. Both candidates refuse performance admission; the incumbent remains.
The helper rejects edited summaries, booleans/nonfinite timings, unequal artifact
identities, duplicate process IDs, or a run count other than nine. Reports are
measurement evidence, not executable authorization. GPU arbiter registration and
tuned parallel schedules remain open even if a future measurement clears the gate.

Raw evidence: `ann-{nvidia,rocm}-{1..9}.json`; dispositions:
`ann-nvidia-summary.json` and `ann-rocm-summary.json`. Recorder:
`benchmarks/record_native_ann_execution.py`. All raw reports include source and
recorder SHA256 fingerprints, native pair/package identities and numerical bounds.
The two native compilers have separate identities; physical proof is not shared.
