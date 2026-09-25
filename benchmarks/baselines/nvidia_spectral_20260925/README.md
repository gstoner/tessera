# NVIDIA spectral: before/after packet — 2026-09-25

Owner `NVIDIA-FFT-WORKSPACE-1`; sync `NVIDIA-SPECTRAL-DEEPEN-2026-09-25`.

Host **The-Super-Bear** (RTX 5070, sm_120, WSL2), CUDA 13.4.59, driver 610.88,
schema `tessera.nvidia_spectral_benchmark.v1`, package ABI
`tessera.nvidia.cuda_fft_workspace.v4`. Produced by
`benchmarks/spectral/benchmark_nvidia_spectral.py`; every row is checked
against NumPy/SciPy (or the op's own contract) before it is timed and carries
its `route` and `latency_source` (Decision #12). Latencies are
`host_wall_synchronized` medians of warm calls through the public entry
point (`rt.launch` / `native_jvp` / `native_backward`), host buffers in and
out. WSL2 timings; no bare-metal calibration (not a promotion claim).

| File | Revision | What |
|---|---|---|
| `baseline_before.json` | `a58a09db` (#841 head) | forward rows, 21 repeats |
| `baseline_before_autodiff.json` | `a58a09db` spectral library + Python, from a worktree | autodiff rows, 5 repeats (the STFT VJP call is 1.7 s) |
| `after.json` | `41e008db` | every row, 30 repeats |

The autodiff before-packet loaded the base commit's `libtessera_nvidia_fft.so`
(built from that worktree with the build tree's own compile commands, via
`TESSERA_NVIDIA_FFT_LIB`) and the base commit's Python. It used the branch's
`tessera-opt` through `TESSERA_OPT`, because the worktree has no build and
this change touches no compiler source.

## Result

| Case | Before (ms) | After (ms) | Speedup | NumPy (ms) |
|---|---:|---:|---:|---:|
| fft_c2c_1x1024 | 0.301 | 0.215 | 1.4× | 0.014 |
| fft_c2c_64x1024 | 0.388 | 0.383 | 1.0× | 0.268 |
| fft_c2c_256x4096 | 2.559 | 2.039 | 1.3× | 12.868 |
| fft_c2c_16x65536 | 2.521 | 1.881 | 1.3× | 17.582 |
| fft_c2c_64x1009 | 0.457 | 0.358 | 1.3× | 1.211 |
| rfft_64x4096 | 1.037 | 0.461 | 2.2× | 0.668 |
| irfft_64x4096 | 1.033 | 0.467 | 2.2× | 0.398 |
| rfft_16x65536 | 2.231 | 0.996 | 2.2× | 5.878 |
| irfft_16x65536 | 2.172 | 1.018 | 2.1× | 2.094 |
| dct2_64x1024 | 11.952 | 0.337 | 35× | 0.087 |
| stft_8x16000_n512_h128 | 5.695 | 0.604 | 9.4× | 3.121 |
| stft_32x48000_n1024_h256 | 47.971 | 4.414 | 11× | 46.505 |
| spectral_conv_16384x257 | 0.932 | 0.450 | 2.1× | 0.275 |
| spectral_conv_262144x1025 | 6.580 | 0.615 | 11× | 14.166 |
| spectral_filter_64x4096 | 0.174 | 0.182 | 1.0× | 0.064 |
| fft_c2c_1x1024_device_resident | — | 0.038 | new | — |
| fft_c2c_64x1024_device_resident | — | 0.044 | new | — |
| fft_c2c_256x4096_device_resident | — | 0.047 | new | — |
| fft_c2c_16x65536_device_resident | — | 0.070 | new | — |
| stft_jvp_8x16000_n512_h128 | 12.607 | 1.653 | 7.6× | — |
| stft_vjp_8x16000_n512_h128 | 1658.511 | 1.144 | 1450× | — |
| istft_jvp_8x122x257 | 21.312 | 1.935 | 11× | — |
| istft_vjp_8x122x257 (strided spectrum) | 77.117 | 2.095 | 37× | — |
| istft_vjp_8x122x257_compact | 76.357 | 1.437 | 53× | — |

Where the GPU still loses to NumPy, it loses at small sizes. Examples are
`fft_c2c_1x1024`, `dct2_64x1024`, `spectral_conv_16384x257`, `irfft_64x4096`
and `spectral_filter`. The host-buffer entry point costs about 0.2 ms of copies
and synchronization per call. The device-resident rows show the same C2C
transforms without it, at 0.04–0.07 ms. `spectral_filter` stays a host complex
multiply: a GPU version would only add two copies to a 0.06 ms operation.

## Attribution (how each hotspot was found)

- **DCT-II** (`ncu`): 99% of the call was an O(N²) fp64 direct kernel (sm_120
  runs fp64 at 1/64 of fp32). It is now Makhoul's FFT-based DCT-II/III.
- **STFT forward** (`nsys` API trace): five `cudaMalloc`/`cudaFree` pairs plus
  a fresh cuFFT plan per call around ~15 µs of kernels. Fixed with a
  per-device plan LRU and a scratch pool.
- **FFT host entry** (`nsys`): per-call allocation plus a redundant
  synchronize. Fixed with per-plan staging. The new device-pointer entry
  points skip staging entirely.
- **spectral_conv**: three host-staged FFTs plus a NumPy multiply, replaced
  by one native batched R2C → multiply → C2R call.
- **STFT/ISTFT backward**: direct-DFT kernels, now inverse/forward cuFFT
  transforms plus deterministic gathers.
- **STFT/ISTFT JVP** (`nsys`): nine `cudaMalloc` per call at ~257 µs median,
  ~13 ms of a 14 ms call. Now pooled.
- **Host staging** (cProfile shows the time inside the ctypes call, while
  `ncu` shows ~30–70 µs of kernels): both build trees on this box configure
  with an empty `CMAKE_BUILD_TYPE`, so the library's host code compiles at
  `-O0`. The unconditional pack-then-fold, and `packHostLayout`'s per-element
  div/mod, cost milliseconds. Rebuilding the same sources with host `-O3`
  roughly halved every autodiff row. The code fix works under either build
  type: compact inputs are handed straight to the device copy, outputs land in
  place, and the strided pack is an odometer walk.
- **Window-gradient reductions** (`ncu`): one thread per window element (512
  threads) took 0.41 ms (STFT) and 1.59 ms (ISTFT). One block per element with
  a fixed-order tree sum takes 0.026 ms and 0.072 ms.

## Validation

At `41e008db` on this host: the FFT/spectral device set
(`test_fft_workspace`, `test_spectral_{autodiff,jvp,policy}`,
`test_native_vjp_execution_certificates`, plus the spectral/JVP/capability
unit files) has **143 passed, 4 skipped**. The skips are the x86 and gfx1151
spectral packages, which are not on this box.

## Correction: `cold_ms` in these packets is a second call (2026-09-25)

Each case was invoked once, untimed, for its correctness check before the
timer recorded `cold_ms`. So `cold_ms` in the JSON files here measured the
**second** call, after compilation, package images and plans were already
populated. Do not read it as cold-start cost. The warm medians (`latency_ms`,
`p10_ms`, `p90_ms`) are unaffected. The benchmark now times the first
invocation as `cold_ms`. These packets are left as recorded rather than
relabelled; a cold figure needs a new measurement.
