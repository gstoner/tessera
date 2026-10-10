# SM120 prepared native matmul ownership

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-PREPARED-MATMUL-2026-10-06. Required host: Super-Bear RTX 5070.

## Native execution and ownership

Ordinary static FP16/BF16 matmul now prepares and invokes its verified
Graph -> Schedule -> Tile -> NVIDIA Target -> LLVM/NVVM image through a C++
owner. No Python kernel body or physical schedule is constructed. Native
compiler ancestry, exact static descriptor, scalar ABI, ordering, geometry
and image identity are checked before preparation.

C++ pins one module, function, context identity and stream per handle.
Synchronous calls share a grow-only aligned scratch arena in the same verified
context; the native mutex holds its lease through kernel completion and host
download. All inputs are copied before output download, preserving host
aliasing. Scratch never escapes as a device buffer. Idle storage may be retired
and allocation retried once on growth OOM; this branch is source-reviewed,
not a fault-injected GPU evidence claim.

Compile-report capture is preserved on warm native calls. Reports carry the
executed artifact hash and native Graph/Schedule/Tile/Target output digests;
69 focused report checks pass with one explicit skip.

Host shape, dtype, capacity, alignment and all non-singleton physical strides
are validated before allocation, copies or launch. Singleton-axis strides are
nonsemantic and accept equivalent NumPy C/F representations. Each returned
host result has its own storage. Fork/context changes and closed handles
are rejected; explicit close retires the modules and last context scratch.

Warm public calls select an already traced, sealed specialization by tensor
shape/dtype/physical strides. They reuse the checked native owner without
tracing, compilation, eager evaluation or portable descriptor restoration.
Changing tensor values rebinds the input views. A different unprepared signature
still enters the full canonical frontend and compiler.

TESSERA_NVIDIA_PREPARED_MATMUL=0 selects the existing canonical portable
descriptor control. Missing matching runtime exports retain that checked
route. Macro-CTA, quantized, AD, resident and asynchronous envelopes do not
enter this typed static host owner.

## Exact-device A/B evidence

Four fresh processes run control/prepared then prepared/control. Each contains
12 matched 17x35x19 rows: FP16/BF16, C/F RHS, plain, bias/ReLU/residual f32,
and bias/ReLU/residual final f16 output. Every matched pair has the same
compiled image digest, compiler binaries, native runtime binary and source
fingerprints. Independent fp64 comparison precedes timing; public output and
resident package output are bitwise equal after timing.

Five warm public wall samples and five 200-launch CUDA-event windows are
recorded per row. Resident event windows include driver dispatch gaps.
Cold public time includes compiler/context/module preparation. Warm public
wall includes host allocation, transport, binding and synchronization.
Those scopes cannot be subtracted as pure Python time.

| Scope | Final matched evidence |
| --- | --- |
| Warm control public wall | 0.658–0.968 ms |
| Warm prepared public wall | 0.219–0.395 ms |
| Median per-case prepared/control wall ratio, forward order | 0.405027 |
| Median per-case prepared/control wall ratio, reversed order | 0.395491 |
| All per-case wall ratios | 0.310–0.470 |

This is a host-runtime reduction for this envelope, not a GPU kernel speedup,
universal compiler latency claim or FP8/MXFP8/MXFP4 strategy promotion.
Raw controls remain in control_forward.json/control_reverse.json; candidate
records in prepared_forward.json/prepared_reverse.json; analysis.json retains
every ratio and identical-image witness.

## Validation and remaining scope

75 owning-device cases pass: the 12 ordinary public/portable tests, native
host-view guards, closed/rebind, context/fork, descriptor drift, concurrent
invocations, eight warmed signature handles with shared scratch reuse, and
48 singleton-layout cases (including explicit refusal of genuinely
column-major A when both A axes exceed one). The A contract remains row-major;
general A layout integration is still required. No test accepts a wrong result
or a portable/eager warm fallback.

651 focused native/JIT/ABI/diagnostic/pass-metadata checks pass; 66 explicit
skips are not exact-device evidence. The fork test exercises pre-mutex
rejection and emits the host Python fork deprecation warning.

All four architecture plans are assessed. ROCm movement/softmax/ingest and
Apple/x86 dispatch do not enter the SM120 cache or C ABI. No sibling-device
evidence transfers. The five-slice goal remains open for general producer/
composed AD/layout/dynamic integration and the remaining ROCm performance
obligations. Graphify refresh could not run because its CLI is absent on WSL.

## Subsequent runtime increment

The native producer-to-matmul owner in ../nvidia_prepared_lhs_20261006/README.md
also changes this shared single-matmul owner's transfers to retained pinned
staging on its own stream. Additional large first-frame/changed-value checks
pass for FP16/BF16 and C/F RHS. This packet's historical timings remain bound
to its original runtime/source hashes and are not current-runtime retimings.
