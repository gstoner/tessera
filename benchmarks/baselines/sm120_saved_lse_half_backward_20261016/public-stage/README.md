# Static typed public SM120 attention reverse

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-SAVED-LSE-HALF-BACKWARD-20261016.

Frozen 68c7e59300 passes 26,022 non-slow unit cases (5,902 skipped), 1,061 required
production compiler-route cases and 339 owning RTX5070 cases. Both compiler-route and
device proof lanes have zero skips/failures; source/tool/provider witnesses remain
unchanged. All twelve audit lanes pass. Thirty-two matched stored-value characterization
rows and thirty-two fresh compiler-absent portable replays pass numerical checks.

The source archive preserves measured bytes independently of future checkout edits. The
earlier native-stage packet retains its separate 9b2159e655 compiler/kernel scope.
CI, review and delivery are tracked by the coordinating PR.

## Contract

Original half semantic Graph -> native reverse AD -> checkpoint Schedule -> Tile ->
NVIDIA Target IR -> LLVM/NVVM/PTX -> checked native owner. QKV and requested gradients
preserve FP16/BF16 storage; O/dO share half or FP32 storage. LSE, optional LSE seed,
bias and bias gradients remain FP32. Half correction is FP32 sum(P*dot(dO,V)); rounded
saved O is not substituted. Half portable VJP schema v2 is distinct; FP32 v1 and JVP
schemas 5/6 meanings remain unchanged.

Static synchronous native_backward accepts compact host or CUDA Array Interface roots,
honors producer streams, snapshots inputs and returns independent completed host
gradients. Explicit capture GPU generations, half JVP, dynamic public reverse ownership,
mixed/pitched storage, resident returned gradients and wider masks/composition remain
follow-ups. No sibling hardware execution evidence transfers from CUDA.

## Validation and characterization

Counts above come from the final raw JUnit and witnesses. The packet retains the
optional-dependency counterexample and its FP32/FP16 owner regression tests; ml_dtypes
is required only by BF16 roles. The native bias-gradient diagnostic preserves the
existing precise refusal.

Benchmark/index.json records 32 rows: 16 FP16/BF16 source configurations and 16 FP32
controls expanded from the identical stored quantized input values. Host/resident calls
alternate within each arm; storage/control arms execute sequentially, with their order
reversed across adjacent configurations. This is storage/residual-policy
characterization, not an interleaved old/new causal speedup experiment or default-route
promotion. Seven rounds of three completed calls retain separate native forward/backward
event and public wall domains. Event windows exclude snapshots/downloads and may include
dispatch gaps. Compiler/concrete-frontend subprocesses are forbidden during
timing/replay. A fresh process restores every portable artifact with compiler paths
absent, after device/library-only discovery and before any package preparation.

All ranges below are milliseconds across individual profile medians. They are not pooled
distributions, paired causal speedups or kernel-only timings.

| Storage | Rows | Host public call | Resident public call | Resident forward event | Resident backward event | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 | 8 | 1.2336–2.8690 | 1.5844–3.2775 | 0.0108–0.0828 | 0.0227–1.0242 | 0.000122 |
| bf16 | 8 | 1.2692–2.7556 | 1.6840–3.2713 | 0.0100–0.0922 | 0.0213–1.0181 | 0.000943 |
| fp32_matched_quantized_control | 16 | 1.1619–1.5823 | 1.5651–2.0687 | 0.0103–0.1062 | 0.0136–0.0743 | 4.69e-08 |

Typed row correction remains a native optimization obligation. The source-attribution
record identifies repeated per-output rowStats work; its reuse multiplicities are not
hardware counts or expected speedups. A shared native per-row producer remains a design
candidate requiring numerical and interleaved timing proof.

 ## Reproduction

On Super-Bear, source the matching typed runtime/tool environment, then run:

    python -m benchmarks.nvidia.record_typed_attention_backward --output SCRATCH/benchmark
    TESSERA_OPT=/missing/tessera-opt TESSERA_NVIDIA_OPT=/missing/tessera-nvidia-opt python -m benchmarks.nvidia.record_typed_attention_backward --replay SCRATCH/benchmark/index.json

The archived executing validation helpers retain exact test selectors and commands. All
tests ran in host WSL. Interrupted earlier source 95986 broad validation is not a final
gate. A stale CMake audit cache was removed after confirming it referenced the prior
scratch checkout; the failed attempt remains retained separately from the successful
audit receipt.

## Phase continuation

The source and tool/provider pins remain unchanged across phase 1 and its explicit
continuation. The three successful broad unit/route/executed lanes retain their original
runner identity and raw receipts. Coverage export stopped because its scratch output
directory held a prior revision. That output was preserved separately, then remaining
audits, isolated timing/replay and device proof resumed on the same source. The phase 1
witness, continuation helper hash and per-lane executing helpers are retained; no
earlier lane is relabeled as rerun.
