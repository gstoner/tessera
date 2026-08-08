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
#include <cstring>
#include <new>
#include <vector>

extern "C" int ts_fft_plan_execute_device_batch_amd(
    void *, const float2 *, float2 *, int, hipStream_t);
extern "C" int ts_fft_plan_execute_r2c_packed_device_batch_amd(
    void *, const float *, float2 *, int, int, hipStream_t);
extern "C" int ts_fft_plan_execute_c2r_packed_device_batch_amd(
    void *, const float2 *, float *, int, int, hipStream_t);
extern "C" int ts_fft_plan_length_amd(void *);

namespace {

constexpr int kThreads = 256;

struct CompositePlan {
  void *workspace = nullptr;
  size_t bytes = 0;
  char digest[65] = {};
};

struct WorkspaceCursor {
  explicit WorkspaceCursor(CompositePlan *plan)
      : base(static_cast<unsigned char *>(plan->workspace)), capacity(plan->bytes) {}

  template <typename T> T *take(size_t elements) {
    size_t alignment = alignof(T);
    offset = (offset + alignment - 1) & ~(alignment - 1);
    size_t requested = elements * sizeof(T);
    if (offset > capacity || requested > capacity - offset) return nullptr;
    T *result = reinterpret_cast<T *>(base + offset);
    offset += requested;
    return result;
  }

  unsigned char *base;
  size_t capacity;
  size_t offset = 0;
};

bool validDigest(const char *digest) {
  if (!digest || std::strlen(digest) != 64) return false;
  for (const char *at = digest; *at; ++at)
    if (!((*at >= '0' && *at <= '9') || (*at >= 'a' && *at <= 'f')))
      return false;
  return true;
}

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
                           int inputStride, int outputLength, float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputLength;
  if (i >= total) return;
  int row = int(i / size_t(outputLength));
  int column = int(i % size_t(outputLength));
  output[i] = input[size_t(row) * inputStride + column].x * scale;
}

__global__ void realPrefixScalar(const float *input, float *output, int batch,
                                 int inputStride, int outputLength,
                                 float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputLength;
  if (i >= total) return;
  int row = int(i / size_t(outputLength));
  int column = int(i % size_t(outputLength));
  output[i] = input[size_t(row) * inputStride + column] * scale;
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

__global__ void padRealScalar(const float *input, float *padded, int batch,
                              int inputLength, int paddedLength) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * paddedLength;
  if (i >= total) return;
  int row = int(i / size_t(paddedLength));
  int column = int(i % size_t(paddedLength));
  padded[i] = column < inputLength
                  ? input[size_t(row) * inputLength + column]
                  : 0.0f;
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

__global__ void frameWindowReal(const float *input, const float *window,
                                float *framesOut, int batch, int samples,
                                int win, int hop, int frames) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * win;
  if (i >= total) return;
  int column = int(i % size_t(win));
  size_t row = i / size_t(win);
  int frame = int(row % size_t(frames));
  int batchIndex = int(row / size_t(frames));
  int source = frame * hop + column;
  framesOut[i] = source < samples
                     ? input[size_t(batchIndex) * samples + source] * window[column]
                     : 0.0f;
}

__global__ void scaleComplex(float2 *values, size_t elements, float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) {
    values[i].x *= scale;
    values[i].y *= scale;
  }
}

__global__ void compactHalfSpectrum(const float2 *full, float2 *half, int rows,
                                    int n, float scale) {
  int bins = n / 2 + 1;
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(rows) * bins;
  if (i >= total) return;
  int row = int(i / size_t(bins));
  int column = int(i % size_t(bins));
  float2 value = full[size_t(row) * n + column];
  half[i] = make_float2(value.x * scale, value.y * scale);
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
                           int hop, int samples, float scale) {
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
  output[i] = (sum / (weight > 1.0e-12f ? weight : 1.0e-12f)) * scale;
}

__global__ void overlapAddReal(const float *framesIn, const float *window,
                               float *output, int batch, int frames, int win,
                               int hop, int samples, float scale) {
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
    sum += framesIn[(size_t(batchIndex) * frames + frame) * win + local] * w;
    weight += w * w;
  }
  output[i] = (sum / (weight > 1.0e-12f ? weight : 1.0e-12f)) * scale;
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
  return "tessera.rocm.spectral_composite.v5";
}

#ifndef TESSERA_ROCM_SPECTRAL_ARCH
#define TESSERA_ROCM_SPECTRAL_ARCH "unknown"
#endif

extern "C" const char *ts_spectral_composite_arch_amd() {
  return TESSERA_ROCM_SPECTRAL_ARCH;
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
                                          float *outputHost, int batch, int n,
                                          float outputScale) {
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
                       n, outputScale);
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
      fftN < outputN || fftN < 2 || (fftN & 1) ||
      ts_fft_plan_length_amd(forwardPlan) != fftN / 2 ||
      ts_fft_plan_length_amd(inversePlan) != fftN / 2)
    return 30;
  float *input = nullptr, *kernel = nullptr, *output = nullptr;
  float2 *xPad = nullptr, *wPad = nullptr, *xFFT = nullptr, *wFFT = nullptr,
         *product = nullptr, *inverse = nullptr;
  size_t xCount = size_t(batch) * inputN;
  size_t wCount = size_t(batch) * kernelN;
  size_t fftCount = size_t(batch) * fftN;
  size_t halfCount = size_t(batch) * (fftN / 2 + 1);
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
                       outputN, 1.0f);
    rc = syncAndCopy(outputHost, output, outputCount * sizeof(float));
  }
  release(input, kernel, output, xPad, wPad, xFFT, wFFT, product, inverse);
  return rc;
}

extern "C" int ts_stft_hostptr_batch_amd(
    void *forwardPlan, const float *inputHost, const float *windowHost,
    float2 *outputHost, int batch, int samples, int win, int hop, int frames,
    float outputScale) {
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
                       dim3(kThreads), 0, 0, full, half, rows, win, outputScale);
    rc = syncAndCopy(outputHost, half, halfCount * sizeof(float2));
  }
  release(input, window, framed, full, half);
  return rc;
}

extern "C" int ts_istft_hostptr_batch_amd(
    void *inversePlan, const float2 *inputHost, const float *windowHost,
    float *outputHost, int batch, int frames, int win, int hop,
    float outputScale) {
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
                       frames, win, hop, samples, outputScale);
    rc = syncAndCopy(outputHost, output, outputCount * sizeof(float));
  }
  release(window, output, half, full, transformed);
  return rc;
}

extern "C" int ts_spectral_composite_plan_create_amd(
    const char *artifactDigest, long long workspaceBytes, void **outPlan) {
  if (!outPlan || !validDigest(artifactDigest) || workspaceBytes <= 0) return 60;
  *outPlan = nullptr;
  CompositePlan *plan = new (std::nothrow) CompositePlan();
  if (!plan) return 61;
  plan->bytes = size_t(workspaceBytes);
  std::memcpy(plan->digest, artifactDigest, 65);
  if (hipMalloc(&plan->workspace, plan->bytes) != hipSuccess) {
    delete plan;
    return 62;
  }
  *outPlan = plan;
  return 0;
}

extern "C" void ts_spectral_composite_plan_destroy_amd(void *opaque) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan) return;
  if (plan->workspace) (void)hipFree(plan->workspace);
  delete plan;
}

extern "C" const char *ts_spectral_composite_plan_digest_amd(void *opaque) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  return plan ? plan->digest : nullptr;
}

extern "C" long long ts_spectral_composite_plan_workspace_bytes_amd(void *opaque) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  return plan ? static_cast<long long>(plan->bytes) : -1;
}

extern "C" int ts_spectral_filter_plan_hostptr_amd(
    void *opaque, const float2 *aHost, const float2 *bHost, float2 *outputHost,
    long long elements) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !aHost || !bHost || !outputHost || elements <= 0) return 70;
  WorkspaceCursor cursor(plan);
  size_t count = size_t(elements), bytes = count * sizeof(float2);
  float2 *a = cursor.take<float2>(count);
  float2 *b = cursor.take<float2>(count);
  float2 *output = cursor.take<float2>(count);
  if (!a || !b || !output) return 71;
  if (hipMemcpy(a, aHost, bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(b, bHost, bytes, hipMemcpyHostToDevice) != hipSuccess)
    return 72;
  int blocks = int((count + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(complexMultiply, dim3(blocks), dim3(kThreads), 0, 0, a,
                     b, output, count);
  return syncAndCopy(outputHost, output, bytes);
}

extern "C" int ts_dct_plan_hostptr_batch_amd(
    void *opaque, void *forwardPlan, const float *inputHost,
    float *outputHost, int batch, int n, float outputScale) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !forwardPlan || !inputHost || !outputHost || batch <= 0 ||
      n <= 0 || ts_fft_plan_length_amd(forwardPlan) != 2 * n)
    return 80;
  size_t inputCount = size_t(batch) * n, fftCount = 2 * inputCount;
  WorkspaceCursor cursor(plan);
  float *input = cursor.take<float>(inputCount);
  float *output = cursor.take<float>(inputCount);
  float2 *padded = cursor.take<float2>(fftCount);
  float2 *transformed = cursor.take<float2>(fftCount);
  if (!input || !output || !padded || !transformed) return 81;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 82;
  hipLaunchKernelGGL(mirrorRealForDCT,
                     dim3((fftCount + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, 0, input, padded, batch, n);
  int rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, padded, transformed,
                                                 batch, 0);
  if (rc) return rc;
  hipLaunchKernelGGL(realPrefix,
                     dim3((inputCount + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, 0, transformed, output, batch, 2 * n, n,
                     outputScale);
  return syncAndCopy(outputHost, output, inputCount * sizeof(float));
}

extern "C" int ts_spectral_conv_plan_hostptr_batch_amd(
    void *opaque, void *forwardPlan, void *inversePlan, const float *inputHost,
    int inputN, const float *kernelHost, int kernelN, float *outputHost,
    int batch, int fftN) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  int outputN = inputN + kernelN - 1;
  if (!plan || !forwardPlan || !inversePlan || !inputHost || !kernelHost ||
      !outputHost || inputN <= 0 || kernelN <= 0 || batch <= 0 ||
      fftN < outputN || fftN < 2 || (fftN & 1) ||
      ts_fft_plan_length_amd(forwardPlan) != fftN / 2 ||
      ts_fft_plan_length_amd(inversePlan) != fftN / 2)
    return 90;
  size_t inputCount = size_t(batch) * inputN;
  size_t kernelCount = size_t(batch) * kernelN;
  size_t fftCount = size_t(batch) * fftN;
  size_t halfCount = size_t(batch) * (fftN / 2 + 1);
  size_t outputCount = size_t(batch) * outputN;
  WorkspaceCursor cursor(plan);
  float *input = cursor.take<float>(inputCount);
  float *kernel = cursor.take<float>(kernelCount);
  float *output = cursor.take<float>(outputCount);
  float *xPad = cursor.take<float>(fftCount);
  float *wPad = cursor.take<float>(fftCount);
  float2 *xFFT = cursor.take<float2>(halfCount);
  float2 *wFFT = cursor.take<float2>(halfCount);
  float2 *product = cursor.take<float2>(halfCount);
  float *inverse = cursor.take<float>(fftCount);
  if (!input || !kernel || !output || !xPad || !wPad || !xFFT || !wFFT ||
      !product || !inverse)
    return 91;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(kernel, kernelHost, kernelCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 92;
  int blocks = int((fftCount + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(padRealScalar, dim3(blocks), dim3(kThreads), 0, 0, input, xPad,
                     batch, inputN, fftN);
  hipLaunchKernelGGL(padRealScalar, dim3(blocks), dim3(kThreads), 0, 0, kernel, wPad,
                     batch, kernelN, fftN);
  int rc = ts_fft_plan_execute_r2c_packed_device_batch_amd(
      forwardPlan, xPad, xFFT, batch, fftN, 0);
  if (!rc) rc = ts_fft_plan_execute_r2c_packed_device_batch_amd(
      forwardPlan, wPad, wFFT, batch, fftN, 0);
  if (!rc) {
    int halfBlocks = int((halfCount + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(complexMultiply, dim3(halfBlocks), dim3(kThreads), 0, 0,
                       xFFT, wFFT, product, halfCount);
    rc = ts_fft_plan_execute_c2r_packed_device_batch_amd(
        inversePlan, product, inverse, batch, fftN, 0);
  }
  if (rc) return rc;
  hipLaunchKernelGGL(realPrefixScalar,
                     dim3((outputCount + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, 0, inverse, output, batch, fftN, outputN,
                     1.0f);
  return syncAndCopy(outputHost, output, outputCount * sizeof(float));
}

extern "C" int ts_stft_plan_hostptr_batch_amd(
    void *opaque, void *forwardPlan, const float *inputHost,
    const float *windowHost, float2 *outputHost, int batch, int samples,
    int win, int hop, int frames, float outputScale) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !forwardPlan || !inputHost || !windowHost || !outputHost ||
      batch <= 0 || samples <= 0 || win < 2 || hop <= 0 || frames <= 0 ||
      ts_fft_plan_length_amd(forwardPlan) != ((win & 1) ? win : win / 2))
    return 100;
  int rows = batch * frames, bins = win / 2 + 1;
  size_t inputCount = size_t(batch) * samples;
  size_t frameCount = size_t(rows) * win;
  size_t halfCount = size_t(rows) * bins;
  WorkspaceCursor cursor(plan);
  float *input = cursor.take<float>(inputCount);
  float *window = cursor.take<float>(size_t(win));
  if (!input || !window) return 101;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(window, windowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 102;
  if (win & 1) {
    float2 *framed = cursor.take<float2>(frameCount);
    float2 *full = cursor.take<float2>(frameCount);
    float2 *half = cursor.take<float2>(halfCount);
    if (!framed || !full || !half) return 101;
    hipLaunchKernelGGL(frameWindow,
                       dim3((frameCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, window, framed, batch,
                       samples, win, hop, frames);
    int rc = ts_fft_plan_execute_device_batch_amd(
        forwardPlan, framed, full, rows, 0);
    if (rc) return rc;
    hipLaunchKernelGGL(compactHalfSpectrum,
                       dim3((halfCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, full, half, rows, win,
                       outputScale);
    return syncAndCopy(outputHost, half, halfCount * sizeof(float2));
  }
  float *framed = cursor.take<float>(frameCount);
  float2 *half = cursor.take<float2>(halfCount);
  if (!framed || !half) return 101;
  hipLaunchKernelGGL(frameWindowReal,
                     dim3((frameCount + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, 0, input, window, framed, batch,
                     samples, win, hop, frames);
  int rc = ts_fft_plan_execute_r2c_packed_device_batch_amd(
      forwardPlan, framed, half, rows, win, 0);
  if (rc) return rc;
  if (outputScale != 1.0f)
    hipLaunchKernelGGL(scaleComplex,
                       dim3((halfCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, half, halfCount, outputScale);
  return syncAndCopy(outputHost, half, halfCount * sizeof(float2));
}

extern "C" int ts_istft_plan_hostptr_batch_amd(
    void *opaque, void *inversePlan, const float2 *inputHost,
    const float *windowHost, float *outputHost, int batch, int frames,
    int win, int hop, float outputScale) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !inversePlan || !inputHost || !windowHost || !outputHost ||
      batch <= 0 || frames <= 0 || win < 2 || hop <= 0 ||
      ts_fft_plan_length_amd(inversePlan) != ((win & 1) ? win : win / 2))
    return 110;
  int rows = batch * frames, bins = win / 2 + 1;
  int samples = (frames - 1) * hop + win;
  size_t halfCount = size_t(rows) * bins;
  size_t fullCount = size_t(rows) * win;
  size_t outputCount = size_t(batch) * samples;
  WorkspaceCursor cursor(plan);
  float *window = cursor.take<float>(size_t(win));
  float *output = cursor.take<float>(outputCount);
  float2 *half = cursor.take<float2>(halfCount);
  if (!window || !output || !half) return 111;
  if (hipMemcpy(half, inputHost, halfCount * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(window, windowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 112;
  if (win & 1) {
    float2 *full = cursor.take<float2>(fullCount);
    float2 *transformed = cursor.take<float2>(fullCount);
    if (!full || !transformed) return 111;
    hipLaunchKernelGGL(expandHalfSpectrum,
                       dim3((fullCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, half, full, rows, win);
    int rc = ts_fft_plan_execute_device_batch_amd(
        inversePlan, full, transformed, rows, 0);
    if (rc) return rc;
    hipLaunchKernelGGL(overlapAdd,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, window, output, batch,
                       frames, win, hop, samples, outputScale);
  } else {
    float *transformed = cursor.take<float>(fullCount);
    if (!transformed) return 111;
    int rc = ts_fft_plan_execute_c2r_packed_device_batch_amd(
        inversePlan, half, transformed, rows, win, 0);
    if (rc) return rc;
    hipLaunchKernelGGL(overlapAddReal,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, window, output, batch,
                       frames, win, hop, samples, outputScale);
  }
  return syncAndCopy(outputHost, output, outputCount * sizeof(float));
}

namespace {

bool validStorage(int storage) { return storage >= 0 && storage <= 2; }

float loadStorage(const void *input, size_t index, int storage) {
  if (storage == 0) return static_cast<const float *>(input)[index];
  uint16_t bits = static_cast<const uint16_t *>(input)[index];
  if (storage == 1) {
    _Float16 half;
    std::memcpy(&half, &bits, sizeof(bits));
    return static_cast<float>(half);
  }
  uint32_t wide = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

void storeStorage(void *output, size_t index, int storage, float value) {
  if (storage == 0) {
    static_cast<float *>(output)[index] = value;
    return;
  }
  if (storage == 1) {
    _Float16 half = static_cast<_Float16>(value);
    std::memcpy(static_cast<uint16_t *>(output) + index, &half, sizeof(half));
    return;
  }
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
  static_cast<uint16_t *>(output)[index] = uint16_t((bits + rounding) >> 16);
}

std::vector<float> unpackStorage(const void *input, size_t elements,
                                 int storage) {
  std::vector<float> result(elements);
  for (size_t i = 0; i < elements; ++i)
    result[i] = loadStorage(input, i, storage);
  return result;
}

void packStorage(const std::vector<float> &input, void *output, int storage) {
  for (size_t i = 0; i < input.size(); ++i)
    storeStorage(output, i, storage, input[i]);
}

} // namespace

extern "C" int ts_dct_plan_hostptr_batch_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost, void *outputHost,
    int batch, int n, int storage, float outputScale) {
  if (!validStorage(storage) || !inputHost || !outputHost) return 120;
  if (storage == 0)
    return ts_dct_plan_hostptr_batch_amd(
        opaque, forwardPlan, static_cast<const float *>(inputHost),
        static_cast<float *>(outputHost), batch, n, outputScale);
  size_t elements = size_t(batch) * n;
  std::vector<float> input = unpackStorage(inputHost, elements, storage);
  std::vector<float> output(elements);
  int rc = ts_dct_plan_hostptr_batch_amd(
      opaque, forwardPlan, input.data(), output.data(), batch, n, outputScale);
  if (!rc) packStorage(output, outputHost, storage);
  return rc;
}

extern "C" int ts_spectral_conv_plan_hostptr_batch_storage_amd(
    void *opaque, void *forwardPlan, void *inversePlan, const void *inputHost,
    int inputN, const void *kernelHost, int kernelN, void *outputHost,
    int batch, int fftN, int storage) {
  if (!validStorage(storage) || !inputHost || !kernelHost || !outputHost)
    return 130;
  if (storage == 0)
    return ts_spectral_conv_plan_hostptr_batch_amd(
        opaque, forwardPlan, inversePlan,
        static_cast<const float *>(inputHost), inputN,
        static_cast<const float *>(kernelHost), kernelN,
        static_cast<float *>(outputHost), batch, fftN);
  size_t inputElements = size_t(batch) * inputN;
  size_t kernelElements = size_t(batch) * kernelN;
  size_t outputElements = size_t(batch) * (inputN + kernelN - 1);
  std::vector<float> input = unpackStorage(inputHost, inputElements, storage);
  std::vector<float> kernel = unpackStorage(kernelHost, kernelElements, storage);
  std::vector<float> output(outputElements);
  int rc = ts_spectral_conv_plan_hostptr_batch_amd(
      opaque, forwardPlan, inversePlan, input.data(), inputN, kernel.data(),
      kernelN, output.data(), batch, fftN);
  if (!rc) packStorage(output, outputHost, storage);
  return rc;
}

extern "C" int ts_stft_plan_hostptr_batch_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost,
    const void *windowHost, float2 *outputHost, int batch, int samples,
    int win, int hop, int frames, int storage, float outputScale) {
  if (!validStorage(storage) || !inputHost || !windowHost || !outputHost)
    return 140;
  if (storage == 0)
    return ts_stft_plan_hostptr_batch_amd(
        opaque, forwardPlan, static_cast<const float *>(inputHost),
        static_cast<const float *>(windowHost), outputHost, batch, samples,
        win, hop, frames, outputScale);
  std::vector<float> input = unpackStorage(
      inputHost, size_t(batch) * samples, storage);
  std::vector<float> window = unpackStorage(windowHost, size_t(win), storage);
  return ts_stft_plan_hostptr_batch_amd(
      opaque, forwardPlan, input.data(), window.data(), outputHost, batch,
      samples, win, hop, frames, outputScale);
}

extern "C" int ts_istft_plan_hostptr_batch_storage_amd(
    void *opaque, void *inversePlan, const float2 *inputHost,
    const void *windowHost, void *outputHost, int batch, int frames,
    int win, int hop, int storage, float outputScale) {
  if (!validStorage(storage) || !inputHost || !windowHost || !outputHost)
    return 150;
  if (storage == 0)
    return ts_istft_plan_hostptr_batch_amd(
        opaque, inversePlan, inputHost,
        static_cast<const float *>(windowHost), static_cast<float *>(outputHost),
        batch, frames, win, hop, outputScale);
  int samples = (frames - 1) * hop + win;
  std::vector<float> window = unpackStorage(windowHost, size_t(win), storage);
  std::vector<float> output(size_t(batch) * samples);
  int rc = ts_istft_plan_hostptr_batch_amd(
      opaque, inversePlan, inputHost, window.data(), output.data(), batch,
      frames, win, hop, outputScale);
  if (!rc) packStorage(output, outputHost, storage);
  return rc;
}

namespace {

size_t storageBytes(int storage) { return storage == 0 ? 4 : 2; }

void packAxis(const void *input, void *output, int64_t outer,
              int64_t axisExtent, int64_t inner, size_t bytes) {
  const auto *source = static_cast<const unsigned char *>(input);
  auto *destination = static_cast<unsigned char *>(output);
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j)
      for (int64_t i = 0; i < axisExtent; ++i) {
        int64_t sourceIndex = (o * axisExtent + i) * inner + j;
        int64_t destinationIndex = (o * inner + j) * axisExtent + i;
        std::memcpy(destination + destinationIndex * bytes,
                    source + sourceIndex * bytes, bytes);
      }
}

void unpackAxis(const void *input, void *output, int64_t outer,
                int64_t axisExtent, int64_t inner, size_t bytes) {
  packAxis(input, output, outer, inner, axisExtent, bytes);
}

} // namespace

extern "C" int ts_dct_plan_hostptr_strided_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost, void *outputHost,
    int64_t outer, int64_t n, int64_t inner, int storage, float outputScale) {
  if (!validStorage(storage) || outer <= 0 || inner <= 0) return 160;
  size_t bytes = storageBytes(storage);
  size_t total = size_t(outer * n * inner) * bytes;
  std::vector<unsigned char> input(total), output(total);
  packAxis(inputHost, input.data(), outer, n, inner, bytes);
  int rc = ts_dct_plan_hostptr_batch_storage_amd(
      opaque, forwardPlan, input.data(), output.data(), int(outer * inner),
      int(n), storage, outputScale);
  if (!rc) unpackAxis(output.data(), outputHost, outer, n, inner, bytes);
  return rc;
}

extern "C" int ts_spectral_conv_plan_hostptr_strided_storage_amd(
    void *opaque, void *forwardPlan, void *inversePlan, const void *inputHost,
    int inputN, const void *kernelHost, int kernelN, void *outputHost,
    int64_t outer, int64_t inner, int fftN, int storage) {
  if (!validStorage(storage) || outer <= 0 || inner <= 0) return 170;
  int outputN = inputN + kernelN - 1;
  int64_t batch = outer * inner;
  size_t bytes = storageBytes(storage);
  std::vector<unsigned char> input(size_t(batch * inputN) * bytes);
  std::vector<unsigned char> kernel(size_t(batch * kernelN) * bytes);
  std::vector<unsigned char> output(size_t(batch * outputN) * bytes);
  packAxis(inputHost, input.data(), outer, inputN, inner, bytes);
  packAxis(kernelHost, kernel.data(), outer, kernelN, inner, bytes);
  int rc = ts_spectral_conv_plan_hostptr_batch_storage_amd(
      opaque, forwardPlan, inversePlan, input.data(), inputN, kernel.data(),
      kernelN, output.data(), int(batch), fftN, storage);
  if (!rc)
    unpackAxis(output.data(), outputHost, outer, outputN, inner, bytes);
  return rc;
}

extern "C" int ts_stft_plan_hostptr_strided_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost,
    const void *windowHost, float2 *outputHost, int64_t outer, int samples,
    int64_t inner, int win, int hop, int frames, int storage,
    float outputScale) {
  if (!validStorage(storage) || outer <= 0 || inner <= 0) return 180;
  int64_t batch = outer * inner;
  int bins = win / 2 + 1;
  size_t bytes = storageBytes(storage);
  std::vector<unsigned char> input(size_t(batch * samples) * bytes);
  std::vector<float2> output(size_t(batch * frames * bins));
  packAxis(inputHost, input.data(), outer, samples, inner, bytes);
  int rc = ts_stft_plan_hostptr_batch_storage_amd(
      opaque, forwardPlan, input.data(), windowHost, output.data(), int(batch),
      samples, win, hop, frames, storage, outputScale);
  if (!rc)
    unpackAxis(output.data(), outputHost, outer, int64_t(frames) * bins, inner,
               sizeof(float2));
  return rc;
}

extern "C" int ts_istft_plan_hostptr_strided_storage_amd(
    void *opaque, void *inversePlan, const float2 *inputHost,
    const void *windowHost, void *outputHost, int64_t outer, int frames,
    int bins, int64_t inner, int win, int hop, int storage,
    float outputScale) {
  if (!validStorage(storage) || outer <= 0 || inner <= 0 ||
      bins != win / 2 + 1)
    return 190;
  int64_t batch = outer * inner;
  int samples = (frames - 1) * hop + win;
  size_t bytes = storageBytes(storage);
  std::vector<float2> input(size_t(batch * frames * bins));
  std::vector<unsigned char> output(size_t(batch * samples) * bytes);
  packAxis(inputHost, input.data(), outer, int64_t(frames) * bins, inner,
           sizeof(float2));
  int rc = ts_istft_plan_hostptr_batch_storage_amd(
      opaque, inversePlan, input.data(), windowHost, output.data(), int(batch),
      frames, win, hop, storage, outputScale);
  if (!rc)
    unpackAxis(output.data(), outputHost, outer, samples, inner, bytes);
  return rc;
}
