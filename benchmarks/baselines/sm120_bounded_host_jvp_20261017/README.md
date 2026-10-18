# SM120 bounded prepared host attention JVP

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-BOUNDED-HOST-JVP-20261017.

## Compiler and runtime boundary

The public call fn.native_jvp(..., sequence_bounds=(Sq, Sk)) captures typed
Graph IR and uses verified native AD, Schedule, Tile, NVIDIA Target and
LLVM/PTX lowering. An AST/tracer structural certificate binds topology and
numerical policy with zero concrete frontend executions. Warm replay does
not invoke the concrete frontend or compiler. Python binds the verified
program; the C++ owner retains modules, a capacity arena and pinned staging.

Runtime Sq/Sk set checked byte lengths, grids, shared-memory size and kernel
indices. Batch, heads, feature dimensions and storage retain their compiled
contracts. Producer events order resident snapshots; results are independent
completed host arrays. Close retires the native owner. Schema 5 remains
saved-LSE plus bounded specifically.

The exact native forward parameter layout distinguishes static physical or
implicit logical bias (no extra shape parameters) from explicitly shaped
dynamic bias (four extra parameters). The additive native prepare flag is
checked against shape policy and actual CUDA parameter widths/counts before
exposing a handle. Tangent sizing/launch uses scratch/query_size/key_size.
Legacy static prepare exports retain their own ABI.

## Evidence

Source-identity binds the clean measured revision, selected core/target
compilers, CUDA driver/runtime and oracle sources. Selections overlap:

- 71 widened native/host cases pass, including FP16/BF16/F32 controls,
  authored FP32 output and dynamic query/key/logical bias.
- 73 static/public/producer-stream/direct-tape regressions pass.
- 24 native-ABI/preflight cases check refusal, untouched sentinel outputs
  and healthy reuse. Eleven source/target guards include trace-dependent
  Python numerical policy rejection before compiler access.
- Eleven synthetic-clock sampler checks prove actual adjacent arm reversal,
  common repetitions, valid floors and exclusion of validation/sample reads.
- Fresh-process replay passes 32 arm/profile cases with all subprocesses
  forbidden after the live device probe. Sixteen separate FP64 finite
  differences validate the unrounded model oracle. Native typed errors are
  retained separately in standalone-numerics.json.xz.
- The final combined selection passes 89 tests in 53.01 seconds.
- Focused registry/ABI/audit gates pass 331 tests; one Darwin Metal execution
  case skips on WSL. This is no Apple numerical proof.
- The type gate reports zero errors, baseline zero.

Paired forward/reverse runs cover 32 rows on RTX5070/sm_120, using identical
stored operands for bounded and static arms. All windows pass the 20 ms summed
completed-call floor. Public-call latency is 0.443-0.633 ms for the bounded route
and 1.588-2.042 ms for static, a 2.99-3.84x ratio across these observations.
Native forward/JVP CUDA event dispatch samples remain separate. Corresponding
forward images, tangent images and host sizers match across both run orders.
This is explicit bounded-route characterization; no default promotion.

The timing profiles are B1/Hq2/Hkv1/D16/Dv16, Sq/Sk 1/1, 7/9, 31/65, 127/131,
FP16/BF16, saved LSE, causal alignment, and absent/scalar physical bias.
Wider output/bias execution checks use their own smaller profiles.

## Red/green engineering controls

The original host preflight rejected symbolic extents. Initial C++ parameter
checks then exposed eight static-bias refusals, followed by three implicit
logical-bias refusals. Each was resolved by matching native lowering's actual
parameter contract; the parameter check remains. Source-identity records the
historical control revision and patch. Early compiler-build and first-event
records are engineering diagnostics; the paired packets carry the current
timing scope.

## Reproduce

Run from the Tessera root on the owning WSL host with matching LLVM/MLIR 23.1.1
tools and the new native runtime. Keep unrelated timing workloads stopped.

    python -m benchmarks.nvidia.benchmark_typed_attention_jvp --bounded --windows 10 --min-window-ms 20 --output /absolute/scratch/forward.json
    python -m benchmarks.nvidia.benchmark_typed_attention_jvp --bounded --reverse --windows 10 --min-window-ms 20 --output /absolute/scratch/reverse.json

Replay this packet without compiler calls:

    python benchmarks/baselines/sm120_bounded_host_jvp_20261017/replay_numerics.py --output /absolute/scratch/replay.json

Large raw receipts are losslessly compressed; raw-archives.json binds both
forms. Program archives are shared by immutable digest, preserving all
referenced artifacts once.

## Remaining gates and scope

Full CPU/production compiler-route and final documentation/public CI/review
are pending. Mixed/quantized attention, wider masks/layouts, general composed
AD/resident results, the other architectures and all five aggregate goal
requirements remain open. The resident-input snapshot path returns completed host results.
