# Native independent shape ownership — 2026-10-21

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key NATIVE-INDEPENDENT-SHAPE-20261021.

## NATIVE-INDEPENDENT-SHAPE-20261021 — native execution proof

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-INDEPENDENT-SHAPE-20261021.

Native semantic dimension relations project independent M/N/K capacities
through compact rank-two producer DAGs. Graph projection clones its source;
verified Schedule/Tile views retain actual compact strides, capacity guards,
initialized storage and retirement after every reader. Member semantic axis
identities are checked even when capacities coincide. Selected leaf ABIs emit
typed MLIR shape arithmetic; cold LLVM lowering supplies a native CPU companion.
The retained C++ frame-v3 owner evaluates it transactionally before committing
launch state. Pure native descriptor queries precede output allocation. Python
transports the checked contract and native bytes without evaluating shapes.

The final frozen e8d55135 compiler passes 648 focused cases and all 3,076
production compiler-route cases with zero selected skips. Existing v1/v2
programs retain byte-identical 28 Target programs and 12 PTX images versus
merged cc3e73c8. The final identity check changes no measured GPU member image
or CPU shape image relative to the preceding candidate. Frame-v3 adds two
native exports; all 103 previous provider exports remain available. The code
image is retained through frame close and quarantined if close fails.

Owning RTX5070/sm120 proof covers four named families: RMSNorm LHS to matmul,
RHS add with LHS ReLU to matmul, add/ReLU to matmul, and matmul/ReLU/matmul
with FP32 middle storage. Roots use FP16 or BF16; independent dimensions vary
within M65/N47/K64 (the first two-matmul contraction remains fixed K32).
58 device cases and one CPU oracle case pass, including trace-at-one/equal
extents, changed dimensions, failed transactions, retained outputs, one arena,
artifact restoration and delayed explicit/legacy/PTDS producer streams.
144 existing native device controls pass with the new provider.

The final 24-row packet records six samples of 32 completed public calls.
Public medians are 0.437324-0.696365 ms; captured native-program medians are
0.002322-0.014476 ms. Only 144/288 whole event windows satisfy the duration
gate for every region; short regions remain diagnostic. Domains are separate
and nonadditive. These are characterization results, without a speedup claim
or default-route promotion. The packet preserves the initial norm-rounding
failure, the independently measured FP32 midpoint witness and the corrected
storage-rounding interval oracle; numerical kernels and policy are unchanged.
Evidence: benchmarks/baselines/native_independent_shapes_20261021/README.md.

All five original aggregate programs remain open. General layouts, control
flow, accumulator reconstruction, wider producers and AD composition remain
follow-up required. Tensor frame schema 3 is independent of attention schema
numbering; attention schema 5 remains saved-LSE plus bounded specifically.

## Evidence identities and reproduction

Final packet: `native-independent-shapes-20261021-final.json`.
Final GPU log: `native-shape-final-device.txt` (58 device plus one CPU oracle).
Final compiler log/XML: `native-shape-final-compiler-route.*`.
Frozen compiler SHA256: e8d55135ff96de7ecebf6e2c78e773e9d24b31bbb5bc6aafe4d6669e1e1c0f4a.
Native provider SHA256: 35e9a9b934a1b8020ceaa6c9259b6a7bc50e747ca124d8576b046ec311d35ee3.
GPU: NVIDIA GeForce RTX 5070, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545,
driver 610.88, compute capability 12.0. LLVM/MLIR 23.1.1, CUDA 13.3.

The recorder is `benchmarks/nvidia/record_native_independent_shapes.py`.
Use matching compiler and LLVM tools plus a rebuilt native provider with both
frame-v3 exports. Run from the checkout root on Super-Bear WSL, with
`TESSERA_OPT` and `TESSERA_NVIDIA_OPT` selecting the recorded compiler,
`TESSERA_NVIDIA_PTX_LAUNCH_LIB` and `TESSERA_NATIVE_FRAME_TEST_LIB` selecting
the same recorded provider, and the matching LLVM tools on the selected PATH:

```sh
python benchmarks/nvidia/record_native_independent_shapes.py \
  --output "$PWD/scratch/native-independent-shapes-fresh.json" \
  --rounds 6 --calls 32 --bounds 65 47 64 \
  --shapes 17x11x23 65x47x64 5x31x7
```

Create the scratch output parent first and choose a fresh path; the recorder
refuses an existing packet or artifact directory.
The packet records source/tool/provider identities and artifacts. Every group
reuses images, shape companion and arena across three active shapes. Warm
public calls forbid retracing, eager evaluation and compiler subprocesses.

The final artifact directory contains actual Graph/Schedule/Tile/Target IR,
member PTX and native host shape companions. Their recorded absolute scratch
paths identify the original run; copies here retain their original bytes.
`integrity.json` inventories retained files by SHA256.

## Preserved failed and corrected experiments

`native-independent-shapes-20261021.json` is the failed first experiment, not
completion evidence. Its rounding failure at RMSNorm M65/N47/K64 seed23403
is preserved with its log. `norm-rounding-diagnosis.json`, the witness NPZ and
two independently compiled native intermediate PTX controls establish the
FP32-to-FP16 midpoint. The corrected oracle admits storage rounding over a
conservative FP32 interval and rejects an incorrect norm; it changes no GPU
kernel or production numerical policy.

`native-independent-shapes-20261021-rounding-reviewed.json` is an intermediate
24-row run using the earlier 7aff1d3a compiler and 16 calls per sample. It is
historical, not the final e8d55135/32-call packet. The old equal-capacity
semantic-identity regression is also retained: the old compiler accepts an
invalid root-axis exchange; the final compiler rejects it without changing
measured member PTX or host shape images.

Final WSL publication gates: the CI-equivalent CPU selector passes 30,947
with 3,721 declared skips and no failures/errors. All 345 focused
registry/lifecycle checks pass, all five surface audits pass, all 33 generated
documents are synchronized, Ruff passes and mypy stays at zero errors.
Graphify is refreshed. Exact-head public CI/review and merge remain pending.

## Automatic-route prerequisite review — PR1002

The cold frame-v3 automatic route now checks all three executable host shape
tools (`mlir-opt`, `mlir-translate`, `clang`) before tracing/admission. Missing,
undiscovered or nonexecutable tools leave existing routes eligible; admitted
programs still propagate real compiler errors. Cached warm frames do not
rediscover tools. Six original regression failures are retained, followed by
30 passing prerequisite/frontend/package cases. The expanded ten-case tool
matrix covers three failure modes per tool and a positive admission control.

On RTX5070, the revised source passes 58 device cases, ten prerequisites and
one CPU oracle regression (69 total). A fresh six-sample/32-call benchmark
completes all 24 rows: public medians 0.439771-0.684228 ms. All copied compiler
artifacts, native member images, shape companions and frames are byte-identical
to the original final packet. The review packet references those retained
artifacts without duplicating them. Earlier timings retain their original
source fingerprints. No speedup or aggregate closure follows.
Evidence: benchmarks/baselines/native_independent_shapes_20261021/prerequisite-review/.
