# Native automatic source-attention strategy

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-SOURCE-ATTENTION-ROWS-20261018.

Ordinary source VJP now requests automatic policy from native MLIR. The native
projector uses actual Q/K/V roles and selects five-member row correction when
logical normalization rows>=32, D>=32, Sk>=4 and Dv>=4; otherwise it retains
the four-member inline program. Python does not select physical strategy.
Explicit rows remains authoritative; no-auto v1 meanings remain unchanged.
Portable replay validates scoped module UnitAttr intent and the sealed selected
ABI without invoking a compiler or duplicating native scheduling policy.

The six-backend LLVM/MLIR23.1.1 compiler passes 120 native/portable cases and
the 81-case offline suite; overlapping reruns are not summed. Twenty saved
programs / ninety members retain byte-identical Graph, Schedule, Tile, Target
and native stages. The full production compiler-route lane passes 2,176 tests
with none skipped. Owning RTX5070 proof covers 91 unique source/public/primal/
stream/recovery cases, including four small/wide fresh-process replays with
compiler and subprocess access disabled.

Two public A/B orders compare against merged PR988's direct retained-owner
frontend with identical inputs and unchanged normalization/forward images.
Each order has ten format/profile rows, twenty-one samples, eight calls/arm.
Completed public control/candidate ratios across FP16/BF16 and both orders:
small 0.952-1.008x, tall 1.562-1.715x, D257 3.528-3.818x, medium 3.998-4.070x,
long 11.079-11.577x. Small images are identical; mixed host variation stays in
the raw packet. These are completed public calls, not isolated kernel speedups.
Stage/device windows may include enqueue gaps. Boundary/corner measurements
retain their earlier explicit compiler identity.

Evidence: benchmarks/baselines/native_source_attention_auto_20261018/README.md.
The explicit prepared packet remains a historical e63eb93f/B29 checkpoint.
Remaining: final CPU/documentation and public CI/review gates. Wider dynamic,
bias/dropout/layout/AD/encoded envelopes and all five aggregate programs remain
open. No quantized attention or sibling physical admission is implied.

## Public performance and provenance

| Profile | Completed public control/candidate ratio, both formats/orders |
| --- | --- |
| small | 0.952-1.008 |
| tall | 1.562-1.715 |
| wide | 3.528-3.818 |
| medium | 3.998-4.070 |
| long | 11.079-11.577 |

The control public_inline_control.py is extracted unchanged from merged PR988
(65b8fb663). It uses the same direct retained-owner execution architecture,
omitting only automatic request. It is not the older serialization-heavy
frontend. Cold warmup is sequential and shared; no cold-speedup claim is made.
Warm A/B forbids subprocesses and the original Python function. Independent
float64 source-gradient numerics and immutable inputs are checked outside
timing. Stage, native completed host and public-call samples remain separate.

Full compiler/provider hashes are in manifest.json. Every measured source in
auto-executed-source-identities.json matches the implementation candidate; only
the final execution-matrix reason description and recorder trailing blank-line
removal differ; both are recorded separately. Boundary
and corner controls retain B29 compiler identity. All twenty complete public
program archives match between orders; sixteen boundary and four corner
archives also match. Duplicate copies are removed only after byte equality.

Raw JSON is losslessly gzip-compressed. manifest.json records decompressed
SHA256 and original-path lookup to deduplicated archives. fresh/ preserves
four actual serialized runtime artifacts, roots and outputs. Arrays use FP32
transport; cast roots according to sealed buffer storage before launch.
XML/text proofs are compressed unchanged. Hashes establish integrity, not
authentication of arbitrary machine code.

## Reproduce

On RTX5070/CUDA13.3 with the matching compiler/provider, run:

    python -m pytest tests/device/nvidia/test_public_source_attention_reverse.py -q
    python benchmarks/nvidia/benchmark_public_source_attention_reverse.py --control ~/scratch/public_inline_control.py --output ~/scratch/source-auto-order0.json --samples 21 --repeats 8 --profiles small tall wide medium long --order-offset 0

Repeat with a fresh output path and --order-offset 1. Run orders sequentially
without competing GPU work. Full CPU/public CI gates remain separate from
exact-device proof and physical strategy measurements.

## Preserved recorders and outputs

- measure_policy_boundaries.py.gz recorded boundary-ab.json.gz with explicit B29 inline/rows programs.
- measure_policy_corner.py.gz recorded corner-ab.json.gz with the combined lower-bound geometry.
- public_inline_control.py is the unchanged merged PR988 direct-owner frontend used by both public-auto-order packets.

The first two files retain their measured contents byte for byte; only their
archival names use underscores and lossless gzip compression. Raw recorder hashes and original filenames
remain in the measurement provenance.
