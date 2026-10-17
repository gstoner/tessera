# Original Qwen gate/up source through resident BF16 activation

Pinned nvidia/Qwen3-8B-NVFP4 gate/up weights and Qwen/Qwen3-8B reference
tensor hashes are retained in original-store-source/gfx1201.json. The same
captured layer-zero MLP inputs used by the earlier source product control
drive M=19 and M=256, N=24576, K=4096.

The ordinary Python source compiles through native Graph partition,
Schedule/Tile and ROCm/LLVM images. Three members own expansion, product with
explicit BF16 store, and view/FP32-SiLU/BF16 output. There are nine physical
buffers; the BF16 parent remains private. Portable replay executes with
compiler subprocess calls forbidden.

## Numerical scope

Both rows have zero violations against independent decoded-weight FP64
projection intervals. The FP32 accumulation bound is propagated through BF16
parent rounding, the SiLU critical point, product extrema and final BF16
rounding, with one outward BF16 ULP for activation FP32 arithmetic. This
specific finite-input allowance is explicit; it is not a universal proof of
the device exponential library.

Relative RMS versus the decoded NVFP4/BF16-parent activation oracle is
approximately 0.00354% at M19 and 0.00268% at M256. Relative RMS versus the
original BF16-weight activation reference is 4.86% and 4.57%. These are
projection/activation characterization results, not full-model accuracy or
representative-corpus acceptance.

## Timing scope

Median captured expansion/product/activation windows per iteration:

| M | Expansion ms | Product ms | Activation ms | Completed prepared host ms |
|---|---:|---:|---:|---:|
| 19 | 0.691 | 2.472 | 0.00449 | 4.515 |
| 256 | 0.711 | 1.386 | 0.0339 | 4.997 |

Grouped member timings include graph dispatch, excluding capture,
instantiation and CPU enqueue. The ordinary resident product/activation event
range may include enqueue gaps. Host timing includes activation upload,
resident invocation and result readback, excluding compilation/preparation.
All windows are correctness-checked and last at least 10ms.

Input E4M3/K16 activation quantization remains benchmark preparation on the
host. Native activation quantization, down projection, full MLP, general
bounded three-member replay and end-model quality/performance remain open.
No matched speedup or strategy promotion is claimed.

The supplementary device-probe.json identifies the live RX9070XT/gfx1201 and
matching compiler/provider after the run; the execution packet itself binds
architecture and compiler/provider hashes during recording.
