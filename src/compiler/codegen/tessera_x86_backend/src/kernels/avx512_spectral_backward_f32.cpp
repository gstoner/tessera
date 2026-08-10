// Native AVX-512 compound-spectral adjoints.
//
// These kernels deliberately consume the dimensions and normalization chosen
// by the content-addressed Schedule -> Tile artifact.  They do not rediscover
// an FFT plan or re-enter Graph IR.

#include <cstdint>

extern "C" void tessera_x86_avx512_spectral_filter_bwd_c64(
    const float *dy, const float *input, const float *filter, float *dx,
    float *dfilter, int64_t elements) {
  for (int64_t i = 0; i < elements; ++i) {
    const float dyr = dy[2 * i], dyi = dy[2 * i + 1];
    const float xr = input[2 * i], xi = input[2 * i + 1];
    const float fr = filter[2 * i], fi = filter[2 * i + 1];
    // Wirtinger VJP for complex multiplication: dy * conj(other).
    dx[2 * i] = dyr * fr + dyi * fi;
    dx[2 * i + 1] = dyi * fr - dyr * fi;
    dfilter[2 * i] = dyr * xr + dyi * xi;
    dfilter[2 * i + 1] = dyi * xr - dyr * xi;
  }
}

extern "C" void tessera_x86_avx512_spectral_conv_bwd_f32(
    const float *dy, const float *input, const float *kernel, float *dx,
    float *dkernel, int64_t batch, int64_t outputLength,
    int64_t inputLength, int64_t kernelLength, float scale) {
  for (int64_t row = 0; row < batch; ++row) {
    const float *rowDy = dy + row * outputLength;
    const float *rowInput = input + row * inputLength;
    const float *rowKernel = kernel + row * kernelLength;
    float *rowDx = dx + row * inputLength;
    float *rowDkernel = dkernel + row * kernelLength;
    for (int64_t i = 0; i < inputLength; ++i) {
      float value = 0.0f;
      for (int64_t j = 0; j < kernelLength; ++j)
        value += rowDy[i + j] * rowKernel[j];
      rowDx[i] = value * scale;
    }
    for (int64_t j = 0; j < kernelLength; ++j) {
      float value = 0.0f;
      for (int64_t i = 0; i < inputLength; ++i)
        value += rowDy[i + j] * rowInput[i];
      rowDkernel[j] = value * scale;
    }
  }
}
