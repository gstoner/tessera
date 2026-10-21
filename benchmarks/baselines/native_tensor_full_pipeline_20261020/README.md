# Complete registered SM120 producer pipeline characterization

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync NATIVE-REGISTERED-PIPELINE-20261020. Candidate base PR999 merge
`7d83bf5a0db9492d5ea8752f1bcd37714d9a0f90`.

`sm120.json` records twelve static validation-shape rows on RTX5070,
UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, compute 12.0.
The assertion-enabled LLVM/MLIR 23.1.1 compiler is frozen at
`da1c946840304c4a456445e105e18d389927817e00779236c3c99ab846a8f7ec`.
The unchanged native provider is
`06fcdb02b05e52f681d35427afc76a8b3a6c772d4fa88a281b69fb70006c9465`.
The packet records current source hashes as well as the compiler build receipt.
The device helper has an explicit post-build harness correction; compiler bytes
are unchanged. Frozen build identity is retained rather than rewritten.

## Route and numerical gate

Original traced Graph -> registered SM120 Graph pipeline -> Schedule/Tile ->
NVIDIA Target -> NVVM -> PTX -> generic prepared native frame owner.
Each Target module passes strict native C++ source replay with byte-identical
output, contains NVVM MMA and no remaining Target MMA carriers. Cases cover
stored add, ReLU, RMSNorm, softmax and two-matmul MLPs with FP16 or FP32
intermediate storage, each with FP16 and BF16 roots. Independent stored-operation
oracles run before and after timing. Inputs are unchanged; one private arena
is retained. Compiler subprocesses are forbidden during completed owner samples.

## Measurement domains

- Compile/replay/image construction: one observed sample per row, 1.03–1.68 s.
  Source generation is outside this region; this is not cold process startup.
- Native owner preparation: first row 486.22 ms; later rows 3.95–5.22 ms.
  The first sample includes process/driver initialization and is not pooled
  with later samples as a steady-state preparation estimate.
- Independent member and whole-program CUDA event regions: six rounds with
  counterbalanced captured/enqueued order and the existing adaptive repeat
  limits. Regions are not additive. Only 72/144 windows meet the minimum 10ms
  duration; all shorter windows are explicitly diagnostic.
- Completed retained owner calls: six samples of sixteen calls, medians
  0.199–0.315 ms. Includes binding, transfers, output allocation and readback;
  excludes compilation/preparation. This is not an ordinary public JIT call.

These small shapes characterize integration overhead. There is no baseline
speedup, physical schedule promotion, broad performance or bounded-DAG claim.
The original five-program objective remains open.

## Reproduction

From the exact-device WSL checkout with the frozen compiler/provider environment:

```sh
python -m benchmarks.nvidia.record_native_tensor_full_pipeline --output sm120.json
```

Malformed rounds, call counts and duplicate selections reject before hardware
identity probing. The full compiler lane and broader documentation gates remain
pending at this checkpoint.
