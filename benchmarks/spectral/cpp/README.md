# C++ Spectral Microbench (Optional)

Builds a small executable that can time simple FFT runs with:
- **FFTW** (CPU)
- **cuFFT** (CUDA)
- **rocFFT** (ROCm)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spectral_fft_bench --help
```

> The CMake script autodetects available libraries and defines compile flags accordingly.
