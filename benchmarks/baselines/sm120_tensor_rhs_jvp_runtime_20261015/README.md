# Public paired tensor JVP artifact dispatch

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-TENSOR-RHS-JVP-20261015.

The existing nvidia_sm120_tensor_jvp_compiled execution row now resolves its
declared nvidia_tensor_rhs_jvp_program executor. The last successful named
static JVP exposes native_jvp_runtime_artifact(); a failed new JVP clears that
persisted product. The ABI is primal_0, primal_1 and tangent_<RHS root index>.
An inspection artifact is a detached copy; edits cannot mutate retained JIT
metadata. The original v7 plan and distinct portable paired v1 are unchanged.

Runtime replay verifies native SSA/witness/image/descriptor contracts and
parent Graph/target/input names before provider access. It returns a tuple of
independent primal/tangent f32 arrays and one completion receipt per native
member. It owns its stream and rejects an external stream. A matching live
JIT owner is borrowed, avoiding a second prepare that invalidates its lease;
standalone replay prepares and retires a temporary owner. The weak registry
does not retain device allocations after caller retirement. Fresh child replay
disables compiler discovery, member packaging and Python member submission.

The numerical scope is the named static FP16/BF16 producer-prefix/RHS-only JVP.
Public controls include both root permutations, JSON replay, continuation on
the original JIT owner, detached inspection metadata and corruption rejection.
Generic active-source/A+B derivatives, reverse, resident roots, dynamic shapes,
encoded operands and fused consumers remain separate requirements. No sibling
physical support or overall five-slice closure is inferred.

## Exact-device tests

The final frozen RTX 5070 gate passes 392 cases (eight environment-specific
skips), including 34 paired tests plus runtime artifact/executor/report and
audit/diagnostic/pass/spec-sync/freshness checks. Imported-contract mypy passes
all four changed compiler/runtime modules. Initial dashboard drift is repaired
by its owning execution-matrix generator, with the red log retained. No
numerical/physical compiler source or provider changes.

## Host overhead attribution and A/B

A correctness-checked 100-call cProfile identifies restore at 0.467 s of the
instrumented 0.536 s total. This is diagnostic host attribution, not device
timing or an uninstrumented speed ratio. Most repeated work reconstructs and
validates member artifact objects.

An eight-entry LRU retains only fully validated native plans. Its key hashes
the entire typed JSON payload, including the declared digest; unchanged/resealed
display keys cannot bypass validation after a content edit. Changed certificates,
including resealed invalid lifetimes, reenter full admission and reject. Parent
Graph/target/argument names are checked on every call. The cache retains no
native handles or device allocations; owners remain weak references.

Six matched alternating cached/full-validation windows use identical images,
the same live owner and eight completed calls per arm. Every arm checks both
outputs against independent stored-coefficient matmul oracles. RuntimeArtifact
host medians are 0.898–1.066 ms cached and 3.358–3.582 ms with full validation
per call (3.36–3.75x host improvement in this named envelope). The control swaps
only the Python certificate validator; it still fully validates and runs the
same native owner. Capture/profile timing remains separate: 6.33–52.03 us
native program medians. Prepared replay is 0.271–0.422 ms. Maximum absolute
error remains 4.77e-6. No device speedup, serving throughput or default physical
schedule promotion is claimed. Further host validation/hash/copy costs remain.

matched.json uses benchmark schema v4, adding the full-validation control
and raw alternating order to earlier v3 replay timing fields. Initial v3 timing
and diagnostic profile are preserved separately. Source/compiler/provider/image
identities and raw member/program/host samples are retained; final benchmark
source hashes match the frozen runtime files.

## Frozen full non-slow regression

The clean fc712bfd6 source passes the complete tests/unit non-slow command on
Super-Bear WSL: 26,269 passed, 7,700 skipped, 876 deselected in 2017.89 seconds.
The source/tool/provider/device receipt and compressed raw log are retained.
This is a source-bound host/owning-RTX5070 regression certificate, not additional
ROCm/Apple or AVX-512 physical proof. It does not sum earlier focused counts.
ROCm compiler environment is unset in this lane; sibling owning receipts remain
separate. The isolated normalization accumulator-policy prototype is outside
this frozen source and is not certified by this result.
