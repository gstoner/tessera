# Typed bounded resident attention generations on RTX5070

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-TYPED-RESIDENT-CAPTURE-20261017.
Runtime source: 99e389916681f264de9afe976ca3df11a44b3bf4.
Recorder source: faf3a82c55436a4f49f980edd74ecf7b9ce74d8d. Delivery pending.

## Integrated boundary

Original half Python/textual Graph -> native reverse AD -> verified
checkpoint Schedule -> Tile -> NVIDIA Target -> LLVM/NVVM/PTX -> checked
resident CUDA generation. The existing compiler already emitted typed bounded
products; the old capture tape refused them as FP32-only. This slice carries
actual storage through allocations, input spans, private QKV/O/LSE snapshots,
reverse seeds and retained gradient generations. BF16 semantic dtype survives
its CUDA void byte spelling. No image, numerical policy, physical schedule,
selector or descriptor ABI change is introduced.

56 new exact-device cases cover FP16/BF16, saved primary/LSE seeds, compact
outputs, query/key broadcast traced at extent one, bounded replay, asynchronous
capture/consumer ordering, delayed producer writes, private snapshots,
forward-only products and independent FP32 O/dO with half QKV/gradients.
Packages restore and execute with compiler subprocesses forbidden. Repeated
backward preserves earlier gradients; closed generations reject array access.
317 existing owning-device/source regressions pass without skips. Broad WSL
CPU validation passes 22,240 (9,946 skipped), with 220 focused host tests,
zero-error mypy for three runtime files plus two recorders, Ruff and 319
audit/diagnostic/pass/storage gates. All 33 generated documents pass.

## Timing characterization

Each row uses matching stored half inputs and an independent FP64 oracle for
O/LSE and gradients before and after profiling. Kernels are the exact functions
and scalar ABI loaded by the checked captured program. Diagnostic CUDA graphs
repeat each native kernel 32 times, then average three event windows per
kernel. These windows include graph dispatch overhead and exclude copies,
module preparation and readback; they are not instruction-counter latency.

| Storage | Sq/Sk/D/Dv | Forward dispatch us | Backward dispatch ms | Completed forward capture ms | Completed reverse owner ms |
|---|---|---:|---:|---:|---:|
| bfloat16 | 3/5/4/3 | 2.92 | 0.0148 | 5.753 | 0.194 |
| bfloat16 | 16/33/16/8 | 26.73 | 0.6492 | 5.688 | 0.895 |
| bfloat16 | 32/129/32/16 | 180.03 | 10.4222 | 6.388 | 10.758 |
| float16 | 3/5/4/3 | 2.96 | 0.0148 | 5.787 | 0.221 |
| float16 | 16/33/16/8 | 26.74 | 0.6528 | 8.989 | 0.854 |
| float16 | 32/129/32/16 | 180.51 | 10.4403 | 6.221 | 10.705 |

Forward capture host scopes include module load, validation, allocations,
private snapshots and completion, but exclude compilation, root upload and
close. Reverse host scopes include fresh gradient allocations and seed copies,
but exclude seed upload and result download. GPU/CPU validation, documentation and Graphify jobs finished
before the final timing run. The packet does not claim a speedup against an earlier route or
promote a default. The large direct reverse kernel and forward-owner host cost
remain measured optimization obligations.

Raw samples, device UUID/driver, tool hash, compiled image hashes, native
Schedule identities and program hashes are recorded per row. Portable programs
are stored once as lossless .json.xz; decompressed hashes and promoted recorder
bytes were verified. No input checkpoint or hardware evidence is transferred
from a sibling backend.

## Reproduce

On Super-Bear, source the matching CUDA13.3 and LLVM/MLIR23.1.1 environment:

```sh
python -m benchmarks.nvidia.benchmark_typed_resident_attention_capture \
  --output /home/angstorms/scratch/typed-resident-capture/timings
```

## Remaining

Half JVP, mixed/pitched roots, composed/nested AD, wider masks, ordinary dynamic
reverse breadth and performance closure remain open. Explicit bounded resident
capture is the proved envelope. The original five-slice goal is not complete.
All four backend plans assess shared native buffer representation; ROCm keeps
its prior FP32 callers and Apple/x86 do not adopt this CUDA capture contract.
