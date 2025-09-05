# Tessera Spectral Operators — Performance Tests (v1)

**Path:** `tessera/tools/perf/spectral/`

This package provides a reproducible, extensible harness to benchmark key spectral operators commonly used in ML and scientific computing:

- 1D/2D FFT (C2C, R2C, C2R)
- DCT-II / DCT-III (1D/2D) via FFT identities
- Spectral convolution (1D/2D) via FFT → pointwise multiply → IFFT
- Power spectrum / magnitude and simple low-pass “spectral pooling”

It measures wall time, estimated FLOPs (for FFT/DCT and spectral conv), device memory traffic (approx.), arithmetic intensity (AI), GFLOP/s, and GB/s.
Results are written to CSV and summarized into a small HTML report with PNG charts.

The harness **auto-detects** CPU/GPU availability and can run on:
- **NumPy** (CPU)
- **PyTorch** (CPU/GPU via `torch.fft.*`) — optional but recommended for GPU coverage

> If PyTorch with CUDA or ROCm is present, the harness will use it when `--device auto` or `--device cuda|rocm` is requested.

---

## Quick Start

```bash
cd tessera/tools/perf/spectral

# (Optional) Create & activate a venv, then:
pip install numpy matplotlib torch  # torch optional for GPU runs

# Run a small sweep on CPU:
python spectral_bench.py --ops fft1d,fft2d,dct2,conv1d_fft --sizes 1024,2048,4096 --device cpu --repeats 30 --warmup 5

# Auto-detect device (will use GPU if available):
python spectral_bench.py --ops fft1d,fft2d --sizes 1024,2048,4096,8192 --device auto

# Generate charts + HTML report from CSV:
python spectral_report.py --results results/results.csv --outdir results/report
```

A convenience script runs a broader default sweep:

```bash
bash run_all.sh
```

---

## What’s Inside

- `spectral_bench.py` — Main CLI benchmark runner.
- `spectral_math.py` — FFT/DCT utilities (NumPy + optional PyTorch), spectral conv helpers.
- `spectral_report.py` — Produces charts (GFLOPs vs size, GB/s vs size) and a compact HTML summary.
- `configs/default.yaml` — A sample config with typical sweeps (1D and 2D).
- `cpp/` — CMake skeleton for an optional C++ microbench that can target FFTW/cufft/rocFFT if available.
- `results/` — Output directory (CSV, PNGs, HTML).

---

## Metrics & Models

**FFT FLOP model (approx.):**
- 1D Complex-to-Complex (length N): `5 * N * log2(N)` **real** FLOPs
- 2D Complex (H×W): `5 * H * W * (log2(H) + log2(W))`
- Real transforms (R2C/C2R): approximately half the complex FLOPs (heuristic: ×0.5)

**Spectral Conv (valid sizes, frequency-domain):**
- Forward FFT of input + kernel, pointwise complex-multiply, inverse FFT.
- FLOPs ≈ FLOPs(FFT(input)) + FLOPs(FFT(kernel)) + `6 * HWB` (complex mul: 6 real FLOPs per element) + FLOPs(IFFT)

**Bytes model (approx.):**
- `bytes ≈ (#reads + #writes) * sizeof(dtype)`
- For in-place ops, we conservatively use ~`2 * tensor_nbytes`.

These models are **estimates** to support comparative analysis and roofline-style reasoning. For exact accounting, couple with hardware profilers.

---

## NVTX / tprof hooks

When running with CUDA, the harness will emit NVTX ranges around kernels if `torch.cuda.nvtx` is available. You can capture them with Nsight Systems/Compute or your `tprof` tooling.

---

## C++ Microbench (optional)

Under `cpp/`, a small CMake project is provided that can build CPU/GPU microbenches if the corresponding libraries are found:

- FFTW (CPU), cuFFT (CUDA), rocFFT (ROCm)

```bash
cd cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spectral_fft_bench --help
```

---

## CSV Schema

`results/results.csv` columns:

```
timestamp, op, device, backend, dtype, shape, batch, repeats, time_ms, gflops, gbs, ai, bytes, flops, err_rel
```

- `err_rel` is a relative correctness check against a NumPy reference.

---

## License

MIT, © 2025 Tessera project contributors.
