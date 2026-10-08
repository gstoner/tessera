# Native SM120 score-bias JVP integration

Owner: AD-RESIDUAL-EVAL-1. Sync: NVIDIA-BIAS-JVP-2026-10-06.
Siblings: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.

## Verified route

Public traced compile/capture and ordinary native_jvp carry distinct static
f32 Q/K/V plus rank-four full or broadcast score bias through native
Graph/AD -> Schedule -> Tile -> LLVM/NVVM -> checked CUDA execution.
The internal JVP op carries bias/dbias explicitly. Scale applies to QK and
its direction before the unscaled dbias addition. Native Schedule seals
activity, physical bias shape and paired O/LSE identity, including the
broadcast reduction policy. Inactive tangent loads are removed natively.

Portable validation checks physical shapes, roles, image/sizer pins and the
complete manifest before allocation. Old unbiased exports/products are retained.
One declared export, tessera_nvidia_attention_jvp_prepare_bias, admits the
explicit biased product. C++ retains images, private arena, context generation
and synchronous completion; Python retains bounded registration handles and
frontend bindings.

## Exact-device evidence

RTX5070 / SM120, driver 610.88; matching LLVM/MLIR 23.1.1 compiler.
36 compile/capture and 36 ordinary public/matched common-runtime cases pass
independent fp64 analytic JVP and central differences. Cases include grouped
heads, causal/noncausal, K=5/129, full/key-varying/constant broadcast bias,
bias-only/value-only/all-role directions and reordered frontend arguments.
Maximum public error: 3.96953575982e-08. Caller mutation, retained results and
repeated directions pass. Three compiler-forbidden fresh-process replays pass.
38 device tests pass, including biased extents, closed/context/PID guards and
existing unbiased JVP/prepared reverse regressions. 543 host gates pass; 16 AVX-512/ROCm environment skips are explicitly reported
in host-tests.txt. They do not establish sibling execution proof.

## Timing scope

Five alternating rounds per case compare the same public compiled product.
Median host call across cases: prepared 1.208768 ms; captured replay
9.155271 ms. Median per-case prepared/replay ratio: 0.134275743.
Synchronous common-runtime walls include transfers. Separate native forward
and tangent CUDA events are in public-packet.json. packet.json records
preloaded tangent dispatch windows and allocating capture/JVP walls.
No GPU algorithm gain or default promotion is claimed. Packet fingerprints
retain the revision actually measured.

## Remaining obligations

General composed/dynamic/layout/aliasing/resident/async/higher AD, dropout and
wider dtypes remain open. This static bias-JVP increment does not close the
full five-slice program. All four queues assess shared op/AD/ABI changes.
Apple/ROCm/x86 bias-JVP consumers and exact-device parity require their own
implementation/evidence; SM120 schedules and measurements are not transferred.
FP8/MXFP8/MXFP4 remain independent evaluation gates.
Graphify refresh is unavailable: graphify is missing from this WSL checkout.
