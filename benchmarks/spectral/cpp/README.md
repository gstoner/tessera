# C++ Spectral Microbench (Optional)

Builds a small executable that measures actual complex64 C2C transforms with:
- **FFTW** (CPU)
- **cuFFT** (CUDA)
- **rocFFT** (ROCm)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spectral_fft_bench --help
./build/spectral_fft_bench --backend fftw --N 65536 --batch 8
./build/spectral_fft_bench --backend rocfft --N 65536 --batch 8
```

The CMake script autodetects optional libraries. Selecting an unavailable
backend fails closed; there is no synthetic timing fallback. Each successful
run writes one `tessera.fft_benchmark.v1` JSON row containing plan time, warm
execution latency, workspace size, nominal GFLOP/s, timing domain, and a
forward/inverse round-trip error. GPU timing is synchronized host wall so WSL
results are not mislabeled as device-event evidence.

The production AVX-512 package has a same-process SciPy comparator:

```bash
python benchmarks/spectral/benchmark_x86_fft.py --batch 1
```

It measures the native Tessera C ABI and `scipy.fft(workers=1)` on identical
complex64 inputs and fails before emitting a row if numerical comparison fails.

The algorithm-candidate sweep keeps selector state explicit while comparing
cached/uncached Bluestein, Rader, native AVX-512 mixed radix, and native Bailey
six-step:

```bash
python benchmarks/spectral/benchmark_x86_fft_algorithms.py
```

It emits `tessera.fft_algorithm_comparison.v1` JSON rows with cold and warm
timing where applicable, factorization, workspace, numerical error, and
`production`, `historical_baseline`, or `candidate_only` selector state.
