# Prepared native row owner and matched timing

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

The additive C++ owner launches actual original-Graph/native-AD forward,
row-correction and backward images, with private row storage and same-stream
execution. Pending resident producers are ordered before private snapshots.
Value-only activity omits row image/workspace/launch. The new explicit timing
capacity protects four values; legacy invocation keeps its two-float ABI.
The CUDA driver verifies actual compiled parameter width/offset/count.

Direct RTX5070 owner proof passes 45 cases; existing public typed reverse passes
172 cases against the same new provider. Focused owner/artifact/registry proof
passes 466 host-free cases with zero skips. Two additional JVP/VJP stream-isolation cases pass with zero skips
and are recorded separately. Frozen record.json binds each successful receipt and
source/tool/provider identity; it distinguishes previous/public compatibility
from new direct native-owner evidence.

The eight-row A/B uses identical stored FP16/BF16 roots and FP32 dO/bias/seed,
bit-identical forward PTX, eight alternating AB/BA batches and five samples per
arm per batch. FP64 and bitwise old/new checks pass before timing; FP64 numerical
checks also run outside every measured completed call. The complete device
interval has the same event boundaries for both arms, including native inter-
stage gaps. Completed-call wall time includes diagnostic Python fixture packing
and allocation plus the synchronous native owner, not public @jit compilation,
program validation or frontend dispatch. Those domains are never mixed.

Sq7/Sk129 and Sq17/Sk129 improve complete device windows approximately 3.6x/10x
and diagnostic completed calls approximately 2.5x/7.6x across the two storages.
Tiny shapes are near parity; tiny FP16 completed-call time regresses about 4%.
No global default or shape-general performance claim is promoted. Complete
all-member portable provenance, public frontend replay/lifetime/stream checks
and public completed-call benchmarking remain open.

Specification: docs/spec/SM120_ATTENTION_ROW_OWNER_CONTRACT.md.

## Matched diagnostic results

All rows use FP32 dO, physical bias and LSE seed. Ratios are inline/prepass;
values above one favor the prepass. Wall ratios are diagnostic owner calls,
not public frontend latency.

| QKV | Sq / Sk / D / Dv | Inline device ms | Prepass device ms | Device ratio | Diagnostic wall ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| f16 | 3 / 5 / 4 / 3 | 0.081568 | 0.079600 | 1.025 | 0.961 |
| f16 | 1 / 129 / 8 / 6 | 0.283872 | 0.267296 | 1.062 | 1.015 |
| f16 | 7 / 129 / 8 / 6 | 0.909104 | 0.251136 | 3.620 | 2.502 |
| f16 | 17 / 129 / 32 / 24 | 6.380544 | 0.632912 | 10.081 | 7.654 |
| bf16 | 3 / 5 / 4 / 3 | 0.074560 | 0.072336 | 1.031 | 1.005 |
| bf16 | 1 / 129 / 8 / 6 | 0.278448 | 0.265936 | 1.047 | 1.022 |
| bf16 | 7 / 129 / 8 / 6 | 0.931216 | 0.260992 | 3.568 | 2.453 |
| bf16 | 17 / 129 / 32 / 24 | 6.341200 | 0.631440 | 10.042 | 7.581 |

The compressed recorder is included with its exact test-helper source snapshot.
To repeat, use the matching Super-Bear WSL source/tool/provider environment,
extract record_benchmark.py.gz under ~/scratch and supply a new --output path.
The recorder reports fresh hashes and refuses source/provider changes during the
run. Preserve this frozen packet instead of overwriting or relabeling it.

Both arms request their available timing outputs: two floats for inline and
four for the new owner. Completed-call timing therefore includes those native
timing queries; the separate device window remains the same boundary definition.
