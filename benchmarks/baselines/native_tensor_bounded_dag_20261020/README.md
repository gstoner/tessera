# Native bounded producer DAG proof and characterization

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NATIVE-BOUNDED-SSA-DAG-20261020; base PR999 merge
`7d83bf5a0db9492d5ea8752f1bcd37714d9a0f90`.

## Compiler and execution

LLVM/MLIR 23.1.1 assertion compiler:
`06e3712d8122e21b4ddb022acb8e01a08566ac87d044dbd61caedf830e87859c`.
Native provider remains
`06fcdb02b05e52f681d35427afc76a8b3a6c772d4fa88a281b69fb70006c9465`.
RTX5070 UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, compute 12.0,
driver 610.88. Packet identities bind the compiler and current source hashes;
the explicit post-build frontend certificate overlay is retained separately.
Compiler/provider bytes do not change with that overlay.

Original Python/textual Graph -> native SSA row-relation analysis ->
Schedule -> Tile views/typed fragments or FP32 scalar FMA -> NVIDIA Target ->
NVVM/PTX -> checked native frame-v2 owner. Source replay is exact. Only M
varies; K/N, compact layouts, numerical policy and intermediate storage remain
fixed. Static RHS producers keep constant row/scalar/grid roles. The compiler
does not infer variable axes from equal traced extents.

## Numerical and regression evidence

- New compiler controls: 34 failures/one refusal pass on da1c9468;
  29 passes/six static-RHS active-type failures on db37f8dd; all 35 pass
  on the repaired 06e3712d compiler. The focused cohort passes 107.
- Full production compiler route: 2,862 passed, zero skips.
- Exact device: 28 new textual/policy DAG cases plus eight existing bounded
  regressions pass. Changed M=17,1,65,5 retains one arena; independent numerical
  checks, canaries, immutable roots and over-capacity rejection pass.
- Ordinary public JIT: all 42 cases pass, covering seven families, two root
  storages and traces (M,K)=(17,32),(1,1),(17,17). Warm calls forbid eager
  execution, tracing and subprocesses, retain image/program/owner identity,
  and replay serialized artifacts at changed M.
- Initial public run: 36 failures/six passes. Eighteen failures were imports
  inside fixture bodies, twelve exposed missing ReLU certificate admission,
  and six exposed literal-policy dictionary admission. The initial fixture
  and failed receipts are retained. Corrected certificate controls pass 35;
  documentation/diagnostic/pass-metadata/certificate gate passes 350.
- Owning gfx1201 control: exact composed MXFP4 numerics pass, with five native
  format image controls unchanged against frozen 01922c4a. This establishes
  sibling noninterference for those controls, not ROCm bounded-DAG execution.

## Final explicit Schedule ODS refinement

The final compiler declares unary row_bound in Schedule ODS and is frozen at
`8903c7698b18164b6e430d7e549ddafb8a1119fcb3a3de696adfe167124b8817`.
All 2,883 compiler cases pass without skips. The focused gate passes 173,
with eight owning-GPU fixture cases excluded on Tajasaurus. On Super-Bear,
the final textual/public/static suite passes 102 cases in one run; the eight
fixture-gated unit cases pass in a separate run. Initial unit-fixture proof
had six stale legacy public-schema assertions and two passes; the correction
requires the native schema, compiler path and exact three-stage dependency
chain, and closes the public native owner. Numerical tolerances are unchanged.
All initial/corrected receipts and the test-only overlay identity are retained.

All forty member images, fourteen frames and complete Target IR modules match
the measured 06e3712d compiler byte-for-byte. Earlier eight static programs
retain eighteen images against frozen 73739736. Final gfx1201 exact numerics
and five image controls pass unchanged. This is scoped sibling noninterference.

The CPU selector records 30,922 passes / 3,721 declared skips and one recorder
inventory failure. Both new recorders are now named in the tracked central
output table; its correction plus baseline/routing/lifecycle controls pass 23.
This is a focused correction, not a claimed rerun of the whole CPU lane.
Final exact-head CI remains required. Ruff and the zero-error mypy ratchet pass.

## Explicit compiler ownership correction

Final provenance review found that the explicit Graph compiler argument was
not forwarded into member image lowering. The normal device/timing runs used
matching primary/backend environments and retain their original identities.
Packaging, immutable validation and restoration now forward the selected
compiler explicitly. Three conflicting-environment regressions fail before
the fix and pass afterward; the adjacent cohort passes 120. The corrected
source passes all 2,886 compiler-route cases without skips and the complete
102-case RTX5070 suite again. Ruff and the zero-error mypy ratchet pass.

Authoritative A/B proof is `evidence/native-dag-explicit-tool-image-controls.json`:
22 programs / 58 images (fourteen bounded programs / forty images plus eight
static programs / eighteen images). Every actual returned backend compiler
fingerprint must match its explicitly selected tool despite a conflicting
common environment. All images and frames match; all bounded Target IR matches.
Older comparison files retained their selected Graph/Target compilers but
used common da1c9468 image conversion. The final identity receipt annotates
that historical scope; the stronger explicit-tool checks replace it. Earlier
timings are not relabeled. See the source-only overlay and final identity.

## Timing domains

`sm120.json` contains 42 rows: seven families x FP16/BF16 x active M=1,17,65.
Each case/storage group retains the same member image hashes and one native
arena across all three M values. Independent numerical checks run before and
after timing. K=32, intermediate width=16 for MLPs, final N=11.

One cold public call per case/storage group includes tracing, compilation,
preparation and first execution. Its value is repeated on the three related
rows for context and must not be counted as three observations.
CUDA events measure each member and the whole program independently; regions
are not additive. Of 504 windows, 253 meet the 10ms minimum and 251 remain
diagnostic. Completed warm public calls have medians 0.441–0.860 ms, including
binding, transfers and readback, excluding compilation/preparation. Six
samples of sixteen calls are retained per row. These validation shapes do
not establish performance at application scale, a speedup or route promotion.
The earlier static full-pipeline packet has different measurement scopes and
its own da1c9468 identity; do not pool them.

## Reproduction and limits

In the matching RTX5070 WSL environment with frozen tool/provider:

```sh
python -m pytest tests/device/nvidia/test_native_tensor_bounded_dag.py -q
python -m benchmarks.nvidia.record_native_tensor_bounded_dag --output sm120.json
```

The proof is leading-M compact straight-line producer integration. General
axes/layouts, noncanonical accumulator reconstruction, control flow, AD,
all five aggregate programs and final exact-head publication gates remain open.
All 33 generated documents, all five surfaces, claim lint and Graphify pass.
No public PR is open at this checkpoint; exact-head CI/review remains required.
Compressed receipts preserve exact original contents; integrity.json binds
this packet and its evidence files.
