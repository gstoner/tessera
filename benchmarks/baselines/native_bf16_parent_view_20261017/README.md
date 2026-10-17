# Native BF16 parent views and resident NVFP4 activation

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-BF16-PARENT-VIEW-20261017.

## Compiled boundary

Original BF16 parent → complementary views → explicit FP32 widenings → SiLU
→ BF16 store follows native Graph → Schedule → typed Tile → ROCm Target →
GPU/ROCDL/LLVM → HSACO. Row schema 3 carries this storage/compute policy;
row schema 2 keeps its FP32 meaning and SM120 row schema 1 is unchanged.

The native textual NVFP4 program groups all six activation ops into one member
after expansion and the BF16 scaled product. Its nine physical buffers include
one private BF16 parent and one returned BF16 activation. There are no
allocations for slices or register casts. The named schema
tessera.native.nvfp4_program.v5 is static activation composition; v3/v4 keep
static/bounded two-member meanings. Attention schema numbering is independent.

The checked parent ABI has BF16 input/output storage and FP32 compute. Python
marshals metadata and bytes; it does not compute SiLU, construct Tile arithmetic,
materialize a FP32 parent or read the private matmul parent to the host.

## Evidence

- Matching assertions-enabled LLVM/MLIR 23.1.1 full and incremental tools build.
- 31 native stage/partition/existing-profile tests pass.
- 45 portable frame/stage/ABI/static-policy and captured-Boolean tests pass.
- 454 dtype/tensor-attribute/diagnostic/pass/audit checks pass at their recorded
  checkpoint; the later static-conditional frontend repair has focused tests.
- Final-source owning RX 9070 XT/gfx1201 selection: 71 passed, comprising
  thirteen existing f32 parent-view cases, seven BF16 public JIT/replay cases (including a
  fresh process with compilers and eager SiLU forbidden), two native resident
  lifetime cases, 22 existing prepared cases and 27 existing bounded cases pass.
- Owning RTX 5070/SM120: twelve source-active schema-1 producer regressions pass.
  They establish existing NVIDIA parity, not admission of the ROCm profile.
- Mypy reports no issues in 657 sources at the recorded resident checkpoint.
  Selections overlap and are not summed.

Source parity independently matches 51 source/dependency files and all three
compiler/library binaries. The source manifest also records the matching
NVIDIA tool, build-log hashes and actual LLVM version. Fresh validation
checkouts can have source timestamps later than copied tools; byte identity and
matching build receipts establish provenance without touching source timestamps.

## Timing method

record_bf16_resident_frozen.py is the frozen diagnostic recorder, originally
executed as benchmark-bf16-resident.py from the owning validation scratch
folder with env-bf16.sh. Import formatting and one final blank line are normalized;
recorder-byte-identity.json preserves both hashes and the exact executed bytes
as base64. No numerical or timing expression is changed.
It imports the named textual Graph and independent numerical fixtures; their
identities are part of the source packet. This is a synthetic characterization,
not checkpoint/model-quality acceptance or a speedup/default-route promotion.

The recorder checks independent E2M1 decode/product/BF16-parent and SiLU/BF16
results before and after every timing window. Captured member windows include
device graph dispatch and exclude capture/instantiation/host enqueue. Grouped
member timing is distinct from interleaved product+activation event timing,
which can include CPU enqueue gaps. Completed host calls include activation
update/upload, native invocation and output readback on a retained owner.

The initial bf16-resident-characterization-short-window.json has sub-1ms
windows and is retained only as a rejected measurement diagnostic. The final
recorder adapts repeats and rejects windows shorter than 10ms. Raw samples,
repeat counts and wall/device timing definitions are in
bf16-resident-characterization.json.

## Still required

The resident three-member proof consumes textual Graph input. Public Python
authoring of BF16 scaled-product storage still needs its frontend/storage
integration; the standalone BF16 activation JIT is separately proved here.
Three-member bounded runtime count/grid expressions, activation quantization,
down projection, full NVFP4 MLP/original-checkpoint quality, model-sized matched
performance and the other original five-slice goal obligations remain open.
Apple/x86 need their own consumers and device evidence. No sibling numerical
schedule, native FP4 WMMA instruction, or general cast/slice support is claimed.

Final-source captured windows span 19.99–22.77 ms. Completed resident host
medians are 0.544, 0.493 and 0.577 ms for the three recorded synthetic shapes.
No measured host improvement or matched baseline ratio is asserted.

Final focused gates: 711 passed, four skipped; all 33 generated documents and
compiler-plan links pass. Final mypy has zero errors in 657 sources, and Ruff
passes. The FP32-source cast capability remains Graph-only outside the named
consumer. initial-counterexamples.json preserves the observed frontend, native
partition and AST-assignment failure boundaries and raw-log hashes.
