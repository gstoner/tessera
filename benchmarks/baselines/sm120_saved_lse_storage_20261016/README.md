# SM120 saved-LSE storage — 2026-10-16

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-SAVED-LSE-STORAGE-20261016.
Recorded by: `benchmarks/nvidia/record_saved_lse_storage.py`.

## Delivered proof

Static compact FP16/BF16 Q/K/V execute through the real public frontend,
typed semantic Graph MLIR, native checkpoint projection, Schedule/Tile,
NVIDIA Target, LLVM/NVVM/PTX, and a checked retained native owner. Primary O
retains source storage; row-LSE stays FP32. Twelve independently authored
half-input/FP32-output cases also execute numerically. Bias stays FP32.

The first complete device run reproduced 24 missing half native routes while
all 12 FP32 controls passed. The production trace had inferred FP32 LSE and
then overwritten that auxiliary with Q storage. The result-specific trace
repair and native storage integration now pass all 36 host/resident/replay
cases, including causal/noncausal, dense/broadcast bias, ragged key extent,
permuted arguments, delayed cross-stream writes and O/LSE byte canaries.

All 960 production compiler-route cases and 94 adjacent RTX5070 device cases
pass without skips. The final required CPU unit lane passes 25,975 cases (5,902 skipped), with
unchanged source/tool witnesses before and after.
The 528 focused registry/dtype/diagnostic/pass tests pass. All 48 native
storage/descriptor-tamper cases and 59 repaired checkpoint/index cases pass.
The type-check ratchet reports zero errors.

## Source and exact device

Implementation freeze: de14978d8a94916c8438ddcd737c2289ee20b5e7.
Final recording freeze: 9566c7e61c39865e94d7253d4478a20e8cd783aa.
Only test fixtures, recorder indexing and documentation changed between them.
`validation/source-manifest.json` binds the 20 implementation files to their
actual bytes; packet consumers compare them with the published source.
Benchmark source and tool fingerprints match before/after the recording.

Device: NVIDIA GeForce RTX 5070, SM120 / compute 12.0,
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, CUDA 13.3.
Compiler: matching LLVM/MLIR 23.1.1 with assertions. Native provider and both
compiler hashes are in `benchmark/packet.json`.

## Timings

Fifteen alternating host/resident rounds follow independent FP64 numerical
checks from the exact stored inputs. Every timed result is checked again.
Kernel events exclude stream waits, snapshots and downloads. Completed public
wall calls include those costs and return independent host O/LSE arrays.
CPU-heavy validation finished before the final timing run began.

Ranges below span causal/noncausal and no-bias/broadcast-bias variants.
They are milliseconds; each value is a range of per-case medians.

| Storage | Q rows / K rows / D | Resident kernel | Completed host | Completed resident |
| --- | --- | --- | --- | --- |
| FP16 | 3 / 5 / 4 | 0.0086–0.0111 | 0.391–0.419 | 0.420–0.469 |
| BF16 | 3 / 5 / 4 | 0.0082–0.0098 | 0.386–0.426 | 0.405–0.481 |
| FP32 | 3 / 5 / 4 | 0.0096–0.0122 | 0.391–0.493 | 0.445–0.509 |
| FP16 | 32 / 129 / 32 | 0.1839–0.1937 | 0.573–0.614 | 0.582–0.637 |
| BF16 | 32 / 129 / 32 | 0.1834–0.1906 | 0.578–0.609 | 0.576–0.633 |
| FP32 | 32 / 129 / 32 | 0.1929–0.2014 | 0.563–0.657 | 0.609–0.656 |

This is a functional storage characterization, without a vendor comparison or
default promotion. Host overhead remains material. Different storage rows
use their own exactly quantized inputs and are checked against their own oracle.

## Compatibility, replay and historical red gates

Twelve old FP32 controls retain identical raw Tile IR, Target IR, PTX, entries
and ABI IDs; both artifact sets and the comparison are under `controls/`.
Wrapper image/artifact digests may change with compiler fingerprints and are
not used as a raw-code equality claim.

All 24 timed standalone packages pass a fresh-process numerical replay with
compiler subprocesses disabled. The CUDA driver may still JIT their native
PTX image. `benchmark/replay.json` records each package.
Dense-bias and explicitly authored FP32-output coverage has independent device
receipts rather than being inferred from the timed broadcast-only arms.

The first broad unit run reported seven failures: six old mocks omitted the
new explicit storage fields, and the recorder lacked an index entry. Those
fixtures and the benchmark README were repaired without changing production
code or weakening the derivative ABI. Historical red log/XML and the original
device red test are compressed under `history/`, with their original scope.

Exact executed helper bytes are retained as compressed provenance snapshots:
`helpers/capture_fp32_controls.py.gz`, `helpers/check_fp32_output.py.gz` and
`helpers/run_final_unit.py.gz`. Their scratch paths describe the original
invocations; they are provenance helpers, not additional product routes.

## Reproduce

Build the exact source with pinned LLVM/MLIR 23.1.1, CUDA 13.3 and the owning
RTX5070 visible in WSL. Set TESSERA_OPT, TESSERA_NVIDIA_OPT and the freshly
built TESSERA_NVIDIA_PTX_LAUNCH_LIB to those matching binaries. Then run:

```sh
python -m pytest tests/unit/test_saved_lse_storage_contract.py -q
python -m pytest tests/device/nvidia/test_saved_lse_storage.py -q
python benchmarks/nvidia/record_saved_lse_storage.py --output "$HOME/scratch/saved-lse" --repetitions 15
python benchmarks/nvidia/record_saved_lse_storage.py --output "$HOME/scratch/saved-lse" --replay-only
```

## Still open

Half-storage JVP/reverse integration, generic producers, dynamic/composed
resident execution, resident outputs, wider masks and representative
performance/model-quality closure. Saved-LSE paired AD remains FP32, with
schema 5 bounded and schema 6 static. Apple, gfx1151/gfx1201 and x86 require
their own native physical consumers and device proof. FP8/MXFP8/MXFP4 routes
and default strategy decisions are unchanged. The five-slice aggregate remains open.
