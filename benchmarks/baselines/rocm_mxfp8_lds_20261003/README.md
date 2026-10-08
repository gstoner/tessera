# gfx1201 MXFP8 LDS package evaluation

Owner ROCM-FP8-BLOCKSCALE-1; sibling format evaluation ROCM-MXFP4-W4A8-1.
Synchronization key ROCM-MXFP8-LDS-2026-10-03. Implementation remains unpublished.

## Retained change

The native MLIR Graph-to-Schedule pass selects an eight-wave 128×64 or 128×128
LDS recipe for standard E4M3/E8M0 K32, per-column scales on gfx1201 NK storage.
Automatic admission requires M >= 128, K >= 1024 and at least 64 workgroups
with the 128×64 panel. The 128×128 panel is selected only when it also supplies
64 workgroups. Smaller grids and K < 1024 retain the one-wave seed.

The frontend can explicitly request auto, seed or LDS performance intent.
C++ owns schedule generation, Tile fragments and Target/LLVM lowering. Python
does not construct Tile bodies or backend kernels. Shared Schedule verification,
Target admission, native image identity and checked HIP launcher admission agree
on these named profiles. No new ABI, canonical dtype or scale numerical policy
is introduced. Each semantic K32 group starts a zero partial accumulator, scales
it once, and joins in ascending group order.

Image projection removes resolved scheduling intent and proven shape constants
while retaining the physical recipe and edge class. Package guards still describe
their own Graph shapes. Reusing the LDS image for smaller K requires a new verified
Graph and descriptor; the original K1024 descriptor correctly rejects that launch.
Tests prove both refusal and same-image reuse at K32/64/128/192/256/512.

## Exact-device evaluation

RX 9070 XT, gfx1201, Tajasaurus WSL2. All nine shapes compare FP8 K128/N128,
FP8 K32/N1 control, standard MXFP8 K32/N1 and explicitly approximate folded MXFP4
from identical f32 source operands. Each arm passes an independent f64 decoded
reference before timing. Source quantization, folding error and native arithmetic
error are separate. This is synthetic operand quality evidence, not model accuracy.

Two reference/candidate pairs reverse process order on the second pair. Each arm
has five rotated device windows with a device-clock/HIP-event agreement gate.
Device execution includes GPU graph dispatch. Checked host end-to-end timing
includes staging, transfers and synchronization and is recorded separately;
compile and graph capture are outside device timing.

| M×N×K | Candidate device µs, runs 1 / 2 | Reference/candidate speedup, runs 1 / 2 | Recipe |
| --- | ---: | ---: | --- |
| 200×256×128 | 8.49 / 8.49 | 1.000 / 0.942 | seed |
| 200×256×1024 | 33.12 / 33.14 | 0.990 / 0.991 | seed |
| 200×4096×1536 | 52.81 / 53.02 | 2.257 / 2.280 | LDS |
| 200×2048×2048 | 49.81 / 49.60 | 2.016 / 2.100 | LDS |
| 200×8192×1024 | 71.02 / 71.26 | 1.936 / 1.952 | LDS |
| 256×1024×1024 | 19.92 / 19.94 | 0.967 / 1.000 | seed |
| 256×4096×5120 | 163.52 / 164.10 | 2.996 / 2.990 | LDS |
| 128×4096×128 | 7.23 / 7.25 | 0.995 / 0.998 | seed |
| 512×4096×1024 | 66.40 / 66.48 | 1.949 / 1.974 | LDS |

The five changed wide-grid cases improve 1.94–3.00×. Seed images are unchanged;
the smallest row has a 6% cross-process spread, so no seed speedup is claimed.
FP8, its K32 control and folded MXFP4 native payloads are byte-identical to the
reference for every shape. MXFP8 seed payloads are also unchanged. The comparison
checks input digests, exact device, numerical gates and each retained ISA digest.

The historical unguarded candidate used LDS at 128×4096×128 and lost about 8%.
The K >= 1024 automatic guard excludes it. Its original JSON/timing is retained
as historical evidence; its candidate-prefixed ISA sidecars were superseded by
the final candidate run. Do not use that historical record for ISA attribution.
Only reference.json, candidate.json, reference-repeat.json and
candidate-repeat.json sidecars are admitted by comparison.json.

The recorder's existing selector_promotion=false field means no cross-format
default or canonical dtype promotion. This slice DOES retain the bounded native
MXFP8 schedule rule above. It does not select MXFP8 over FP8/MXFP4 globally.

## Validation and source identity

final-regression.txt: 496 tests pass on gfx1201, including the LDS numerical,
cache/guard tests, FP8 partial-copy regressions and diagnostic/pass/op registry
gates. SCHED, GEN and LOWER FileCheck prefixes pass on the matching final compiler.
existing-mxfp8-packages.txt: 28 earlier checked MXFP8 cases pass; existing-fp8-packages.txt: 99 W8A8 cases pass. audit-docs.txt: 11 audit tests pass; compiler-plan.txt and generated-docs.txt retain plan ownership and 32 generated-doc gates. Ruff and git diff --check pass. Graphify update is unavailable in the WSL checkout (exit 127, recorded in graphify-update.txt).

guarded-tests.txt: 126 focused package/identity tests pass before the broader run.
A matching Super-Bear build passes 345 host registry/package gates and 86 existing RTX 5070 scheduled matmul/attention cases (six existing oracle warnings). nvidia-host-gates.txt, nvidia-parity.txt and nvidia-device.txt retain the receipts. This is existing-route parity, not a CUDA E8M0 consumer.

Each packet records compiler SHA256 and Python/C++ source hashes. The reference
binary is a preserved pre-LDS compiler. reference_source_override binds its four
changed C++ files to snapshots; its Python recorder/runtime hashes describe the
actual harness at recording time. The first reference predates frontend intent
validation, while the repeated reference uses the final harness with default
auto (no new module attribute). Neither source fingerprint is relabeled.

## Remaining gates

FP8, MXFP8 and MXFP4 remain mandatory before another strategy decision.
MXFP8 LDS still trails FP8/folded MXFP4 on the wide cases; next test a native
multi-group K64 slab with separate K32 accumulator joins, rather than changing
scale semantics. Measure barrier/LDS/global movement and correctness first.
The M256 Radiance per-column gap is not attributed by this packet. Persistent
scheduling, broader shapes/layouts and source-model quality remain open.
gfx1151 lacks RDNA4 FP8 WMMA; this recipe does not establish its support.
Apple/x86 need their own E8M0 consumers. Existing NVIDIA parity does not establish
a CUDA E8M0 implementation or justify transferring AMD physical schedules.
