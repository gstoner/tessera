//===- SpectralComposite.hip - device-resident gfx1151 TSOL composites ----===//
//
// Physical helper package for the five public TSOL composites. FFT arithmetic
// remains owned by the content-addressed persistent plan ABI in
// StockhamRadix4.hip; this file owns only framing, packing, complex multiply,
// padding/cropping, and deterministic overlap-add around those child plans.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>

#include <cstddef>

extern "C" int ts_fft_plan_execute_device_batch_amd(
    void *, const float2 *, float2 *, int, hipStream_t);
extern "C" int ts_fft_plan_length_amd(void *);

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float2 multiply(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

__global__ void complexMultiply(const float2 *a, const float2 *b, float2 *out,
                                size_t elements) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) out[i] = multiply(a[i], b[i]);
}

__global__ void mirrorRealForDCT(const float *input, float2 *padded, int batch,
                                 int n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * size_t(2 * n);
  if (i >= total) return;
  int row = int(i / size_t(2 * n));
  int column = int(i % size_t(2 * n));
  int source = column < n ? column : 2 * n - 1 - column;
  padded[i] = make_float2(input[size_t(row) * n + source], 0.0f);
}

__global__ void realPrefix(const float2 *input, float *output, int batch,
                           int inputStride, int outputLength) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputLength;
  if (i >= total) return;
  int row = int(i / size_t(outputLength));
  int column = int(i % size_t(outputLength));
  output[i] = input[size_t(row) * inputStride + column].x;
}

__global__ void padReal(const float *input, float2 *padded, int batch,
                        int inputLength, int paddedLength) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * paddedLength;
  if (i >= total) return;
  int row = int(i / size_t(paddedLength));
  int column = int(i % size_t(paddedLength));
  float value = column < inputLength
                    ? input[size_t(row) * inputLength + column]
                    : 0.0f;
  padded[i] = make_float2(value, 0.0f);
}

__global__ void frameWindow(const float *input, const float *window,
                            float2 *framesOut, int batch, int samples, int win,
                            int hop, int frames) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * win;
  if (i >= total) return;
  int column = int(i % size_t(win));
  size_t row = i / size_t(win);
  int frame = int(row % size_t(frames));
  int batchIndex = int(row / size_t(frames));
  int source = frame * hop + column;
  float value = source < samples
                    ? input[size_t(batchIndex) * samples + source] * window[column]
                    : 0.0f;
  framesOut[i] = make_float2(value, 0.0f);
}

__global__ void compactHalfSpectrum(const float2 *full, float2 *half, int rows,
                                    int n) {
  int bins = n / 2 + 1;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(rows) * bins;
  if (i >= total) return;
  int row = int(i / size_t(bins));
  int column = int(i % size_t(bins));
  half[i] = full[size_t(row) * n + column];
}

__global__ void expandHalfSpectrum(const float2 *half, float2 *full, int rows,
                                   int n) {
  int bins = n / 2 + 1;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(rows) * n;
  if (i >= total) return;
  int row = int(i / size_t(n));
  int column = int(i % size_t(n));
  if (column < bins) {
    full[i] = half[size_t(row) * bins + column];
  } else {
    float2 value = half[size_t(row) * bins + (n - column)];
    full[i] = make_float2(value.x, -value.y);
  }
}

// One thread owns one output sample and visits contributing frames in ascending
// order. This avoids atomic-order drift and makes overlap-add deterministic.
__global__ void overlapAdd(const float2 *framesIn, const float *window,
                           float *output, int batch, int frames, int win,
                           int hop, int samples) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * samples;
  if (i >= total) return;
  int batchIndex = int(i / size_t(samples));
  int sample = int(i % size_t(samples));
  float sum = 0.0f, weight = 0.0f;
  for (int frame = 0; frame < frames; ++frame) {
    int local = sample - frame * hop;
    if (local < 0 || local >= win) continue;
    float w = window[local];
    sum += framesIn[(size_t(batchIndex) * frames + frame) * win + local].x * w;
    weight += w * w;
  }
  output[i] = sum / (weight > 1.0e-12f ? weight : 1.0e-12f);
}

template <typename T> bool allocate(T **pointer, size_t elements) {
  return hipMalloc(pointer, elements * sizeof(T)) == hipSuccess;
}

template <typename... Pointers> void release(Pointers... pointers) {
  ((pointers ? (void)hipFree(pointers) : (void)0), ...);
}

int syncAndCopy(void *host, const void *device, size_t bytes) {
  if (hipDeviceSynchronize() != hipSuccess) return 8;
  return hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost) == hipSuccess
             ? 0
             : 9;
}

} // namespace

extern "C" const char *ts_spectral_composite_package_abi_amd() {
  return "tessera.rocm.spectral_composite.v1";
}

extern "C" int ts_spectral_filter_hostptr_amd(
    const float2 *aHost, const float2 *bHost, float2 *outputHost,
    long long elements) {
  if (!aHost || !bHost || !outputHost || elements <= 0) return 10;
  float2 *a = nullptr, *b = nullptr, *output = nullptr;
  size_t count = size_t(elements), bytes = count * sizeof(float2);
  if (!allocate(&a, count) || !allocate(&b, count) ||
      !allocate(&output, count)) {
    release(a, b, output);
    return 11;
  }
  int rc = 0;
  if (hipMemcpy(a, aHost, bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(b, bHost, bytes, hipMemcpyHostToDevice) != hipSuccess)
    rc = 12;
  if (!rc) {
    int blocks = int((count + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(complexMultiply, dim3(blocks), dim3(kThreads), 0, 0, a,
                       b, output, count);
    rc = syncAndCopy(outputHost, output, bytes);
  }
  release(a, b, output);
  return rc;
}

extern "C" int ts_dct_hostptr_batch_amd(void *forwardPlan,
                                          const float *inputHost,
                                          float *outputHost, int batch, int n) {
  if (!forwardPlan || !inputHost || !outputHost || batch <= 0 || n <= 0 ||
      ts_fft_plan_length_amd(forwardPlan) != 2 * n)
    return 20;
  float *input = nullptr, *output = nullptr;
  float2 *padded = nullptr, *transformed = nullptr;
  size_t inCount = size_t(batch) * n, fftCount = 2 * inCount;
  if (!allocate(&input, inCount) || !allocate(&output, inCount) ||
      !allocate(&padded, fftCount) || !allocate(&transformed, fftCount)) {
    release(input, output, padded, transformed);
    return 21;
  }
  int rc = hipMemcpy(input, inputHost, inCount * sizeof(float),
                     hipMemcpyHostToDevice) == hipSuccess
               ? 0
               : 22;
  if (!rc) {
    hipLaunchKernelGGL(mirrorRealForDCT,
                       dim3((fftCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, padded, batch, n);
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, padded, transformed,
                                               batch, 0);
  }
  if (!rc) {
    hipLaunchKernelGGL(realPrefix, dim3((inCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, output, batch, 2 * n,
                       n);
    rc = syncAndCopy(outputHost, output, inCount * sizeof(float));
  }
  release(input, output, padded, transformed);
  return rc;
}

extern "C" int ts_spectral_conv_hostptr_batch_amd(
    void *forwardPlan, void *inversePlan, const float *inputHost, int inputN,
    const float *kernelHost, int kernelN, float *outputHost, int batch,
    int fftN) {
  int outputN = inputN + kernelN - 1;
  if (!forwardPlan || !inversePlan || !inputHost || !kernelHost ||
      !outputHost || inputN <= 0 || kernelN <= 0 || batch <= 0 ||
      fftN < outputN || ts_fft_plan_length_amd(forwardPlan) != fftN ||
      ts_fft_plan_length_amd(inversePlan) != fftN)
    return 30;
  float *input = nullptr, *kernel = nullptr, *output = nullptr;
  float2 *xPad = nullptr, *wPad = nullptr, *xFFT = nullptr, *wFFT = nullptr,
         *product = nullptr, *inverse = nullptr;
  size_t xCount = size_t(batch) * inputN;
  size_t wCount = size_t(batch) * kernelN;
  size_t fftCount = size_t(batch) * fftN;
  size_t outputCount = size_t(batch) * outputN;
  if (!allocate(&input, xCount) || !allocate(&kernel, wCount) ||
      !allocate(&output, outputCount) || !allocate(&xPad, fftCount) ||
      !allocate(&wPad, fftCount) || !allocate(&xFFT, fftCount) ||
      !allocate(&wFFT, fftCount) || !allocate(&product, fftCount) ||
      !allocate(&inverse, fftCount)) {
    release(input, kernel, output, xPad, wPad, xFFT, wFFT, product, inverse);
    return 31;
  }
  int rc = 0;
  if (hipMemcpy(input, inputHost, xCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(kernel, kernelHost, wCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    rc = 32;
  int blocks = int((fftCount + kThreads - 1) / kThreads);
  if (!rc) {
    hipLaunchKernelGGL(padReal, dim3(blocks), dim3(kThreads), 0, 0, input,
                       xPad, batch, inputN, fftN);
    hipLaunchKernelGGL(padReal, dim3(blocks), dim3(kThreads), 0, 0, kernel,
                       wPad, batch, kernelN, fftN);
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, xPad, xFFT, batch,
                                               0);
  }
  if (!rc)
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, wPad, wFFT, batch,
                                               0);
  if (!rc) {
    hipLaunchKernelGGL(complexMultiply, dim3(blocks), dim3(kThreads), 0, 0,
                       xFFT, wFFT, product, fftCount);
    rc = ts_fft_plan_execute_device_batch_amd(inversePlan, product, inverse,
                                               batch, 0);
  }
  if (!rc) {
    hipLaunchKernelGGL(realPrefix,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, inverse, output, batch, fftN,
                       outputN);
    rc = syncAndCopy(outputHost, output, outputCount * sizeof(float));
  }
  release(input, kernel, output, xPad, wPad, xFFT, wFFT, product, inverse);
  return rc;
}

extern "C" int ts_stft_hostptr_batch_amd(
    void *forwardPlan, const float *inputHost, const float *windowHost,
    float2 *outputHost, int batch, int samples, int win, int hop, int frames) {
  if (!forwardPlan || !inputHost || !windowHost || !outputHost || batch <= 0 ||
      samples <= 0 || win <= 0 || hop <= 0 || frames <= 0 ||
      ts_fft_plan_length_amd(forwardPlan) != win)
    return 40;
  int rows = batch * frames, bins = win / 2 + 1;
  float *input = nullptr, *window = nullptr;
  float2 *framed = nullptr, *full = nullptr, *half = nullptr;
  size_t inputCount = size_t(batch) * samples;
  size_t frameCount = size_t(rows) * win;
  size_t halfCount = size_t(rows) * bins;
  if (!allocate(&input, inputCount) || !allocate(&window, size_t(win)) ||
      !allocate(&framed, frameCount) || !allocate(&full, frameCount) ||
      !allocate(&half, halfCount)) {
    release(input, window, framed, full, half);
    return 41;
  }
  int rc = 0;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(window, windowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    rc = 42;
  if (!rc) {
    hipLaunchKernelGGL(frameWindow,
                       dim3((frameCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, window, framed, batch,
                       samples, win, hop, frames);
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, framed, full, rows,
                                               0);
  }
  if (!rc) {
    hipLaunchKernelGGL(compactHalfSpectrum,
                       dim3((halfCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, full, half, rows, win);
    rc = syncAndCopy(outputHost, half, halfCount * sizeof(float2));
  }
  release(input, window, framed, full, half);
  return rc;
}

extern "C" int ts_istft_hostptr_batch_amd(
    void *inversePlan, const float2 *inputHost, const float *windowHost,
    float *outputHost, int batch, int frames, int win, int hop) {
  if (!inversePlan || !inputHost || !windowHost || !outputHost || batch <= 0 ||
      frames <= 0 || win <= 0 || hop <= 0 ||
      ts_fft_plan_length_amd(inversePlan) != win)
    return 50;
  int rows = batch * frames, bins = win / 2 + 1;
  int samples = (frames - 1) * hop + win;
  float *window = nullptr, *output = nullptr;
  float2 *half = nullptr, *full = nullptr, *transformed = nullptr;
  size_t halfCount = size_t(rows) * bins;
  size_t fullCount = size_t(rows) * win;
  size_t outputCount = size_t(batch) * samples;
  if (!allocate(&window, size_t(win)) || !allocate(&output, outputCount) ||
      !allocate(&half, halfCount) || !allocate(&full, fullCount) ||
      !allocate(&transformed, fullCount)) {
    release(window, output, half, full, transformed);
    return 51;
  }
  int rc = 0;
  if (hipMemcpy(half, inputHost, halfCount * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(window, windowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    rc = 52;
  if (!rc) {
    hipLaunchKernelGGL(expandHalfSpectrum,
                       dim3((fullCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, half, full, rows, win);
    rc = ts_fft_plan_execute_device_batch_amd(inversePlan, full, transformed,
                                               rows, 0);
  }
  if (!rc) {
    hipLaunchKernelGGL(overlapAdd,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, window, output, batch,
                       frames, win, hop, samples);
    rc = syncAndCopy(outputHost, output, outputCount * sizeof(float));
  }
  release(window, output, half, full, transformed);
  return rc;
}
