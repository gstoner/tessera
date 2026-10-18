# Native source-attention row correction

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-SOURCE-ATTENTION-ROWS-20261018.
Base: merged PR988 / 65b8fb663. This packet records the explicit v2 candidate,
not public automatic-policy promotion or closure of the five aggregate slices.

## Native program and proof

A shared native projector derives the correction and compact consumer from
the actual generated checkpoint adjoint. The five typed members pass Graph,
Schedule, Tile, NVIDIA Target and NVVM/PTX. Thirteen checked buffers preserve
original roots/seeds, a private FP32 row, gradient order and source pullback.
Both correction and backward consume private normalized Q. Rounded saved O
is not a correction input. V1/default meanings remain unchanged.

The matching six-backend LLVM/MLIR23.1.1 assertions compiler passes 84 native/
portable cases inside the 388-case diagnostic/pass/contract gate. An additional
226 existing standalone/bounded row-route tests pass. The 53-case portable
rerun overlaps the 84-case gate and is not added as unique evidence. The
Python type ratchet remains at zero errors; naming/lifecycle gates pass twelve.

Exact RTX5070/SM120 passes 89 distinct cases: forty v1/v2 source-gradient,
ordering and native refusal/recovery checks; forty-nine v2 fresh-process,
existing public reverse and source-forward controls. Fresh v2 child replay
disables compiler paths and all external subprocesses. Actual child artifacts,
root arrays and outputs are preserved under device/. NPZ arrays use FP32
transport; restore half root storage according to the sealed buffer contract.

## Composed source A/B

measurements/order0.json.gz and order1.json.gz retain twenty-one alternating samples
of eight calls per arm for each of five profiles and two storage formats.
The initial arm order reverses between runs. Every sample is checked against
an independent float64 attention/RMSNorm source-gradient oracle with explicit
half storage boundaries and FP32 normalization statistics. Inputs are unchanged.
Normalization and forward image bytes match between strategies. All twenty
complete program/image identities match between orders; one copy of each
program archive serves both raw packets.

| Profile | Prepared end-to-end inline/rows ratio, both formats/orders |
| --- | --- |
| Small | 0.867–0.999 |
| Tall | 1.926–2.017 |
| D257 | 4.781–4.864 |
| Medium | 5.296–5.638 |
| Long | 12.599–14.362 |

The extra stage is flat to about fifteen percent slower in the small profiles.
Larger profiles improve, including the complete native source pullback. These
are prepared owner calls, not ordinary public JIT latency or an isolated kernel
speedup. Stage/program CUDA event windows may include enqueue gaps; completed
native host and prepared end-to-end regions stay separate. Compile and native
prepare costs are outside warm windows. Raw order differences are retained.

## Reproduce and remaining scope

Use the compiler/provider hashes in measurements/manifest.json and
device/device-identities.txt with CUDA13.3 on RTX5070:

python -m pytest tests/device/nvidia/test_source_attention_reverse.py tests/device/nvidia/test_portable_source_attention_reverse_rows.py -q
python benchmarks/nvidia/benchmark_source_attention_rows.py --output-dir ~/scratch/source-rows-ab --samples 21 --repeats 8 --order-offset 0

Automatic native selection, ordinary public integration for that selection,
fresh final full gates and one focused PR remain required. Dynamic/bias/dropout/
layout, general AD, encoded attention and all five aggregate programs stay open.
No Apple/x86/gfx1151/gfx1201 physical proof transfers from these CUDA results.
Content hashes establish integrity, not authentication of arbitrary PTX.

## Source identity scopes

manifest.json records the candidate implementation snapshot. The actual
measured Python/provider files are separately recorded in
device/executed-source-identities.json. Device execution_matrix.py is exactly
merged PR988's file; the candidate changes only its executor/reason descriptions.
The three modified native driver modules match the measured files byte for byte.
The six-backend compiler was built on Tajasaurus from the native projection
sources in this snapshot and transferred by its recorded binary SHA. Runtime
provider sources and the measured CUDA library retain their separate identities.

## Historical checkpoint

This explicit-v2 packet belongs to private checkpoint e63eb93f5c97cd9989898541c8a7b80d1bf8473c and compiler B29. Its pending-policy wording describes that checkpoint. Subsequent native-auto policy, public benchmarks and compiler proofs are recorded separately in ../native_source_attention_auto_20261018/. Old measured source identities are retained. Raw order JSON is losslessly gzip-compressed with decompressed SHA256 in the measurement manifest.
