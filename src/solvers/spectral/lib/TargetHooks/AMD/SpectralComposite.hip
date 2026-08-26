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
#include <limits>
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

__global__ void realMultiply(const float *a, const float *b, float *out,
                             size_t elements) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < elements) out[i] = a[i] * b[i];
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

__global__ void dctDirect(const float *input, float *output, int batch, int n,
                          int dctType, float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * n;
  if (i >= total) return;
  int row = int(i / size_t(n));
  int k = int(i % size_t(n));
  double value = 0.0;
  if (dctType == 1) {
    value = input[size_t(row) * n] +
            ((k & 1) ? -1.0 : 1.0) * input[size_t(row) * n + n - 1];
    for (int j = 1; j + 1 < n; ++j)
      value += 2.0 * input[size_t(row) * n + j] *
               cos(M_PI * double(j * k) / double(n - 1));
  } else if (dctType == 3) {
    value = input[size_t(row) * n];
    for (int j = 1; j < n; ++j)
      value += 2.0 * input[size_t(row) * n + j] *
               cos(M_PI * double((2 * k + 1) * j) / double(2 * n));
  } else {
    for (int j = 0; j < n; ++j)
      value += 2.0 * input[size_t(row) * n + j] *
               cos(M_PI * double((2 * k + 1) * (2 * j + 1)) /
                   double(4 * n));
  }
  output[i] = float(value * scale);
}

__global__ void dctType2Prefix(const float2 *input, float *output, int batch,
                               int n, float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * n;
  if (i >= total) return;
  int row = int(i / size_t(n));
  int k = int(i % size_t(n));
  float s, c;
  __sincosf(-float(M_PI) * float(k) / float(2 * n), &s, &c);
  float2 value = input[size_t(row) * size_t(2 * n) + k];
  output[i] = (value.x * c - value.y * s) * scale;
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

__global__ void stftDirectPolicy(const float *input, const float *windows,
                                 float2 *output, int batch, int samples,
                                 int fftN, int hop, int frames, int bins,
                                 float scale, int center, int padMode) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * bins;
  if (i >= total) return;
  int bin = int(i % bins);
  size_t rowFrame = i / bins;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int pad = center ? fftN / 2 : 0;
  double real = 0.0, imag = 0.0;
  for (int local = 0; local < fftN; ++local) {
    int source = frame * hop + local - pad;
    if ((source < 0 || source >= samples) && padMode == 1)
      while (source < 0 || source >= samples)
        source = source < 0 ? -source : 2 * samples - 2 - source;
    float value = source >= 0 && source < samples
                      ? input[size_t(row) * samples + source] *
                            windows[size_t(row) * fftN + local]
                      : 0.0f;
    double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
    real += double(value) * cos(angle);
    imag -= double(value) * sin(angle);
  }
  output[i] = make_float2(float(real * scale), float(imag * scale));
}

__global__ void istftDirectPolicy(const float2 *spectrum,
                                  const float *windows, float *output,
                                  int batch, int frames, int bins, int fftN,
                                  int hop, int outputSamples, float scale,
                                  int center, int onesided) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputSamples;
  if (i >= total) return;
  int sample = int(i % outputSamples) + (center ? fftN / 2 : 0);
  int row = int(i / outputSamples);
  double numerator = 0.0, denominator = 0.0;
  for (int frame = 0; frame < frames; ++frame) {
    int local = sample - frame * hop;
    if (local < 0 || local >= fftN) continue;
    double time = 0.0;
    for (int bin = 0; bin < bins; ++bin) {
      double weight = 1.0;
      if (onesided && bin > 0 && !(fftN % 2 == 0 && bin == bins - 1))
        weight = 2.0;
      float2 value = spectrum[(size_t(row) * frames + frame) * bins + bin];
      double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
      time += weight * (double(value.x) * cos(angle) -
                        double(value.y) * sin(angle));
    }
    double w = windows[size_t(row) * fftN + local];
    numerator += (time / double(fftN)) * w;
    denominator += w * w;
  }
  output[i] = float(numerator /
                    (denominator > 1.0e-12 ? denominator : 1.0e-12) * scale);
}

__global__ void stftBackwardInputPolicy(
    const float2 *dy, const float *windows, float *dx, int batch, int samples,
    int fftN, int hop, int frames, int bins, float scale, int center,
    int padMode) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * samples;
  if (i >= total) return;
  int sampleIndex = int(i % samples), row = int(i / samples);
  int pad = center ? fftN / 2 : 0;
  double result = 0.0;
  for (int frame = 0; frame < frames; ++frame)
    for (int local = 0; local < fftN; ++local) {
      int source = frame * hop + local - pad;
      if ((source < 0 || source >= samples) && padMode == 1)
        while (source < 0 || source >= samples)
          source = source < 0 ? -source : 2 * samples - 2 - source;
      if (source != sampleIndex) continue;
      double grad = 0.0;
      for (int bin = 0; bin < bins; ++bin) {
        float2 upstream = dy[(size_t(row) * frames + frame) * bins + bin];
        double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
        grad += double(upstream.x) * cos(angle) -
                double(upstream.y) * sin(angle);
      }
      result += grad * scale * windows[size_t(row) * fftN + local];
    }
  dx[i] = float(result);
}

__global__ void stftBackwardWindowPolicy(
    const float2 *dy, const float *input, const int *rowWindow,
    float *dwindow, int batch, int windowRows, int samples, int fftN,
    int win, int hop, int frames, int bins, float scale, int center,
    int padMode) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(windowRows) * win;
  if (i >= total) return;
  int localWindow = int(i % win), windowRow = int(i / win);
  int local = (fftN - win) / 2 + localWindow;
  int pad = center ? fftN / 2 : 0;
  double result = 0.0;
  for (int row = 0; row < batch; ++row) {
    if (rowWindow[row] != windowRow) continue;
    for (int frame = 0; frame < frames; ++frame) {
      int source = frame * hop + local - pad;
      bool present = source >= 0 && source < samples;
      if (!present && padMode == 1) {
        while (source < 0 || source >= samples)
          source = source < 0 ? -source : 2 * samples - 2 - source;
        present = true;
      }
      if (!present) continue;
      double grad = 0.0;
      for (int bin = 0; bin < bins; ++bin) {
        float2 upstream = dy[(size_t(row) * frames + frame) * bins + bin];
        double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
        grad += double(upstream.x) * cos(angle) -
                double(upstream.y) * sin(angle);
      }
      result += grad * scale * input[size_t(row) * samples + source];
    }
  }
  dwindow[i] = float(result);
}

__global__ void istftFrameValuesPolicy(const float2 *spectrum, float *framesOut,
                                       int batch, int frames, int bins,
                                       int fftN, float inverseScale,
                                       int onesided) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * fftN;
  if (i >= total) return;
  int local = int(i % fftN);
  size_t rowFrame = i / fftN;
  int frame = int(rowFrame % frames), row = int(rowFrame / frames);
  double value = 0.0;
  for (int bin = 0; bin < bins; ++bin) {
    double weight = 1.0;
    if (onesided && bin > 0 && !(fftN % 2 == 0 && bin == bins - 1))
      weight = 2.0;
    float2 spectral = spectrum[(size_t(row) * frames + frame) * bins + bin];
    double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
    value += weight * (double(spectral.x) * cos(angle) -
                       double(spectral.y) * sin(angle));
  }
  framesOut[i] = float(value * inverseScale);
}

__global__ void istftBackwardFramesPolicy(
    const float *dy, const float *frameValues, const float *windows,
    float *dframes, int batch, int frames, int fftN, int hop,
    int outputSamples, int center) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * fftN;
  if (i >= total) return;
  int local = int(i % fftN);
  size_t rowFrame = i / fftN;
  int frame = int(rowFrame % frames), row = int(rowFrame / frames);
  int output = frame * hop + local - (center ? fftN / 2 : 0);
  if (output < 0 || output >= outputSamples) {
    dframes[i] = 0.0f;
    return;
  }
  int rawSample = output + (center ? fftN / 2 : 0);
  double numerator = 0.0, denominator = 0.0;
  for (int other = 0; other < frames; ++other) {
    int otherLocal = rawSample - other * hop;
    if (otherLocal < 0 || otherLocal >= fftN) continue;
    double w = windows[size_t(row) * fftN + otherLocal];
    numerator += frameValues[(size_t(row) * frames + other) * fftN + otherLocal] * w;
    denominator += w * w;
  }
  double safe = denominator > 1.0e-12 ? denominator : 1.0e-12;
  dframes[i] = float(double(dy[size_t(row) * outputSamples + output]) / safe *
                     windows[size_t(row) * fftN + local]);
}

__global__ void istftBackwardSpectrumPolicy(
    const float *dframes, float2 *dspectrum, int batch, int frames, int bins,
    int fftN, float inverseScale, int onesided) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * bins;
  if (i >= total) return;
  int bin = int(i % bins);
  size_t rowFrame = i / bins;
  int frame = int(rowFrame % frames), row = int(rowFrame / frames);
  double real = 0.0, imag = 0.0;
  for (int local = 0; local < fftN; ++local) {
    double angle = 2.0 * M_PI * double(bin * local) / double(fftN);
    double value = dframes[(size_t(row) * frames + frame) * fftN + local];
    real += value * cos(angle);
    imag -= value * sin(angle);
  }
  double weight = inverseScale;
  if (onesided && bin > 0 && !(fftN % 2 == 0 && bin == bins - 1))
    weight *= 2.0;
  dspectrum[i] = make_float2(float(real * weight), float(imag * weight));
}

__global__ void istftBackwardWindowPolicy(
    const float *dy, const float *frameValues, const float *windows,
    const int *rowWindow, float *dwindow, int batch, int windowRows,
    int frames, int fftN, int win, int hop, int outputSamples, int center) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(windowRows) * win;
  if (i >= total) return;
  int localWindow = int(i % win), windowRow = int(i / win);
  int local = (fftN - win) / 2 + localWindow;
  int trim = center ? fftN / 2 : 0;
  double result = 0.0;
  for (int row = 0; row < batch; ++row) {
    if (rowWindow[row] != windowRow) continue;
    for (int frame = 0; frame < frames; ++frame) {
      int output = frame * hop + local - trim;
      if (output < 0 || output >= outputSamples) continue;
      int rawSample = output + trim;
      double numerator = 0.0, denominator = 0.0;
      for (int other = 0; other < frames; ++other) {
        int otherLocal = rawSample - other * hop;
        if (otherLocal < 0 || otherLocal >= fftN) continue;
        double w = windows[size_t(row) * fftN + otherLocal];
        numerator += frameValues[(size_t(row) * frames + other) * fftN + otherLocal] * w;
        denominator += w * w;
      }
      double safe = denominator > 1.0e-12 ? denominator : 1.0e-12;
      double upstream = dy[size_t(row) * outputSamples + output];
      double draw = upstream / safe;
      double dweight = denominator > 1.0e-12
                           ? -upstream * numerator / (safe * safe) : 0.0;
      double w = windows[size_t(row) * fftN + local];
      double frameValue = frameValues[(size_t(row) * frames + frame) * fftN + local];
      result += draw * frameValue + 2.0 * dweight * w;
    }
  }
  dwindow[i] = float(result);
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

__global__ void overlapAddJvpReal(
    const float *framesIn, const float *dframesIn, const float *window,
    const float *dwindow, float *primal, float *tangent, int batch,
    int frames, int win, int hop, int samples, float scale) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * samples;
  if (i >= total) return;
  int batchIndex = int(i / size_t(samples));
  int sample = int(i % size_t(samples));
  float numerator = 0.0f, denominator = 0.0f;
  float dnumerator = 0.0f, ddenominator = 0.0f;
  for (int frame = 0; frame < frames; ++frame) {
    int local = sample - frame * hop;
    if (local < 0 || local >= win) continue;
    size_t at = (size_t(batchIndex) * frames + frame) * win + local;
    float w = window[local], dw = dwindow ? dwindow[local] : 0.0f;
    numerator += framesIn[at] * w;
    denominator += w * w;
    dnumerator += (dframesIn ? dframesIn[at] * w : 0.0f) +
                  framesIn[at] * dw;
    ddenominator += 2.0f * w * dw;
  }
  float safe = denominator > 1.0e-12f ? denominator : 1.0e-12f;
  primal[i] = numerator / safe * scale;
  tangent[i] =
      (dnumerator / safe - numerator * ddenominator / (safe * safe)) * scale;
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
  return "tessera.rocm.spectral_composite.v7";
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
                                          int dctType, float outputScale) {
  if (!inputHost || !outputHost || batch <= 0 || n <= 0 || dctType < 1 ||
      dctType > 4 || (dctType == 1 && n < 2) ||
      (dctType == 2 &&
       (!forwardPlan || ts_fft_plan_length_amd(forwardPlan) != 2 * n)))
    return 20;
  float *input = nullptr, *output = nullptr;
  float2 *padded = nullptr, *transformed = nullptr;
  size_t inCount = size_t(batch) * n, fftCount = 2 * inCount;
  if (!allocate(&input, inCount) || !allocate(&output, inCount) ||
      (dctType == 2 &&
       (!allocate(&padded, fftCount) || !allocate(&transformed, fftCount)))) {
    release(input, output, padded, transformed);
    return 21;
  }
  int rc = hipMemcpy(input, inputHost, inCount * sizeof(float),
                     hipMemcpyHostToDevice) == hipSuccess
               ? 0
               : 22;
  if (!rc && dctType == 2) {
    hipLaunchKernelGGL(mirrorRealForDCT,
                       dim3((fftCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, padded, batch, n);
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, padded, transformed,
                                               batch, 0);
  }
  if (!rc && dctType == 2) {
    hipLaunchKernelGGL(dctType2Prefix,
                       dim3((inCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, output, batch, n,
                       outputScale);
  } else if (!rc) {
    hipLaunchKernelGGL(dctDirect,
                       dim3((inCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, output, batch, n, dctType,
                       outputScale);
  }
  if (!rc) {
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
  if (!inputHost || !kernelHost || !outputHost || inputN <= 0 ||
      kernelN <= 0 || batch <= 0 || fftN < outputN)
    return 30;
  bool scalar = fftN == 1;
  if (!scalar &&
      (!forwardPlan || !inversePlan || fftN < 2 || (fftN & 1) ||
       ts_fft_plan_length_amd(forwardPlan) != fftN / 2 ||
       ts_fft_plan_length_amd(inversePlan) != fftN / 2))
    return 30;
  float *input = nullptr, *kernel = nullptr, *output = nullptr;
  float *xPad = nullptr, *wPad = nullptr, *inverse = nullptr;
  float2 *xFFT = nullptr, *wFFT = nullptr, *product = nullptr;
  size_t xCount = size_t(batch) * inputN;
  size_t wCount = size_t(batch) * kernelN;
  size_t fftCount = size_t(batch) * fftN;
  size_t halfCount = size_t(batch) * (fftN / 2 + 1);
  size_t outputCount = size_t(batch) * outputN;
  if (!allocate(&input, xCount) || !allocate(&kernel, wCount) ||
      !allocate(&output, outputCount) ||
      (!scalar &&
       (!allocate(&xPad, fftCount) || !allocate(&wPad, fftCount) ||
        !allocate(&xFFT, halfCount) || !allocate(&wFFT, halfCount) ||
        !allocate(&product, halfCount) || !allocate(&inverse, fftCount)))) {
    release(input, kernel, output, xPad, wPad, xFFT, wFFT, product, inverse);
    return 31;
  }
  int rc = 0;
  if (hipMemcpy(input, inputHost, xCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(kernel, kernelHost, wCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    rc = 32;
  if (!rc && scalar) {
    hipLaunchKernelGGL(realMultiply,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, kernel, output,
                       outputCount);
  } else if (!rc) {
    int blocks = int((fftCount + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(padRealScalar, dim3(blocks), dim3(kThreads), 0, 0,
                       input, xPad, batch, inputN, fftN);
    hipLaunchKernelGGL(padRealScalar, dim3(blocks), dim3(kThreads), 0, 0,
                       kernel, wPad, batch, kernelN, fftN);
    rc = ts_fft_plan_execute_r2c_packed_device_batch_amd(
        forwardPlan, xPad, xFFT, batch, fftN, 0);
    if (!rc)
      rc = ts_fft_plan_execute_r2c_packed_device_batch_amd(
          forwardPlan, wPad, wFFT, batch, fftN, 0);
    if (!rc) {
      int halfBlocks = int((halfCount + kThreads - 1) / kThreads);
      hipLaunchKernelGGL(complexMultiply, dim3(halfBlocks), dim3(kThreads), 0,
                         0, xFFT, wFFT, product, halfCount);
      rc = ts_fft_plan_execute_c2r_packed_device_batch_amd(
          inversePlan, product, inverse, batch, fftN, 0);
    }
    if (!rc)
      hipLaunchKernelGGL(realPrefixScalar,
                         dim3((outputCount + kThreads - 1) / kThreads),
                         dim3(kThreads), 0, 0, inverse, output, batch, fftN,
                         outputN, 1.0f);
  }
  if (!rc)
    rc = syncAndCopy(outputHost, output, outputCount * sizeof(float));
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
    float *outputHost, int batch, int n, int dctType, float outputScale) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !inputHost || !outputHost || batch <= 0 || n <= 0 ||
      dctType < 1 || dctType > 4 || (dctType == 1 && n < 2) ||
      (dctType == 2 &&
       (!forwardPlan || ts_fft_plan_length_amd(forwardPlan) != 2 * n)))
    return 80;
  size_t inputCount = size_t(batch) * n, fftCount = 2 * inputCount;
  WorkspaceCursor cursor(plan);
  float *input = cursor.take<float>(inputCount);
  float *output = cursor.take<float>(inputCount);
  float2 *padded = dctType == 2 ? cursor.take<float2>(fftCount) : nullptr;
  float2 *transformed = dctType == 2 ? cursor.take<float2>(fftCount) : nullptr;
  if (!input || !output || (dctType == 2 && (!padded || !transformed))) return 81;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 82;
  int rc = 0;
  if (dctType == 2) {
    hipLaunchKernelGGL(mirrorRealForDCT,
                       dim3((fftCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, padded, batch, n);
    rc = ts_fft_plan_execute_device_batch_amd(forwardPlan, padded, transformed,
                                               batch, 0);
    if (rc) return rc;
    hipLaunchKernelGGL(dctType2Prefix,
                       dim3((inputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, transformed, output, batch, n,
                       outputScale);
  } else {
    hipLaunchKernelGGL(dctDirect,
                       dim3((inputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, output, batch, n, dctType,
                       outputScale);
  }
  return syncAndCopy(outputHost, output, inputCount * sizeof(float));
}

extern "C" int ts_spectral_conv_plan_hostptr_batch_amd(
    void *opaque, void *forwardPlan, void *inversePlan, const float *inputHost,
    int inputN, const float *kernelHost, int kernelN, float *outputHost,
    int batch, int fftN) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  int outputN = inputN + kernelN - 1;
  if (!plan || !inputHost || !kernelHost || !outputHost || inputN <= 0 ||
      kernelN <= 0 || batch <= 0 || fftN < outputN)
    return 90;
  bool scalar = fftN == 1;
  if (!scalar &&
      (!forwardPlan || !inversePlan || fftN < 2 || (fftN & 1) ||
       ts_fft_plan_length_amd(forwardPlan) != fftN / 2 ||
       ts_fft_plan_length_amd(inversePlan) != fftN / 2))
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
  if (!input || !kernel || !output) return 91;
  if (hipMemcpy(input, inputHost, inputCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(kernel, kernelHost, kernelCount * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 92;
  if (scalar) {
    hipLaunchKernelGGL(realMultiply,
                       dim3((outputCount + kThreads - 1) / kThreads),
                       dim3(kThreads), 0, 0, input, kernel, output,
                       outputCount);
    return syncAndCopy(outputHost, output, outputCount * sizeof(float));
  }
  float *xPad = cursor.take<float>(fftCount);
  float *wPad = cursor.take<float>(fftCount);
  float2 *xFFT = cursor.take<float2>(halfCount);
  float2 *wFFT = cursor.take<float2>(halfCount);
  float2 *product = cursor.take<float2>(halfCount);
  float *inverse = cursor.take<float>(fftCount);
  if (!xPad || !wPad || !xFFT || !wFFT || !product || !inverse)
    return 91;
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
      batch <= 0 || samples <= 0 || win <= 0 || hop <= 0 || frames <= 0 ||
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
      batch <= 0 || frames <= 0 || win <= 0 || hop <= 0 ||
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

extern "C" int ts_istft_jvp_plan_hostptr_batch_amd(
    void *opaque, void *inversePlan, const float2 *inputHost,
    const float *windowHost, const float2 *dinputHost,
    const float *dwindowHost, float *primalHost, float *tangentHost, int batch,
    int frames, int win, int hop, float outputScale) {
  CompositePlan *plan = static_cast<CompositePlan *>(opaque);
  if (!plan || !inversePlan || !inputHost || !windowHost || !dwindowHost ||
      !primalHost || !tangentHost || batch <= 0 || frames <= 0 || win <= 0 ||
      (win & 1) || hop <= 0 ||
      ts_fft_plan_length_amd(inversePlan) != win / 2)
    return 120;
  int rows = batch * frames, bins = win / 2 + 1;
  int samples = (frames - 1) * hop + win;
  size_t halfCount = size_t(rows) * bins;
  size_t frameCount = size_t(rows) * win;
  size_t outputCount = size_t(batch) * samples;
  float2 *input = nullptr, *dinput = nullptr;
  float *window = nullptr, *dwindow = nullptr;
  float *timeFrames = nullptr, *dtimeFrames = nullptr;
  float *primal = nullptr, *tangent = nullptr;
  if (!allocate(&input, halfCount) ||
      (dinputHost && !allocate(&dinput, halfCount)) ||
      !allocate(&window, size_t(win)) || !allocate(&dwindow, size_t(win)) ||
      !allocate(&timeFrames, frameCount) ||
      (dinputHost && !allocate(&dtimeFrames, frameCount)) ||
      !allocate(&primal, outputCount) || !allocate(&tangent, outputCount)) {
    release(input, dinput, window, dwindow, timeFrames, dtimeFrames, primal,
            tangent);
    return 121;
  }
  int rc = 0;
  if (hipMemcpy(input, inputHost, halfCount * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      (dinputHost &&
       hipMemcpy(dinput, dinputHost, halfCount * sizeof(float2),
                 hipMemcpyHostToDevice) != hipSuccess) ||
      hipMemcpy(window, windowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dwindow, dwindowHost, size_t(win) * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) {
    rc = 122;
  } else if ((rc = ts_fft_plan_execute_c2r_packed_device_batch_amd(
                  inversePlan, input, timeFrames, rows, win, 0)) != 0) {
  } else if (dinputHost &&
             (rc = ts_fft_plan_execute_c2r_packed_device_batch_amd(
                  inversePlan, dinput, dtimeFrames, rows, win, 0)) != 0) {
  } else {
    hipLaunchKernelGGL(
        overlapAddJvpReal,
        dim3((outputCount + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        timeFrames, dtimeFrames, window, dwindow, primal, tangent, batch,
        frames, win, hop, samples, outputScale);
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(primalHost, primal, outputCount * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(tangentHost, tangent, outputCount * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      rc = 123;
  }
  release(input, dinput, window, dwindow, timeFrames, dtimeFrames, primal,
          tangent);
  return rc;
}

namespace {

bool validStorage(int storage) { return storage >= 0 && storage <= 2; }

bool checkedHostBatch(int64_t outer, int64_t inner, int &batch) {
  if (outer <= 0 || inner <= 0 ||
      outer > std::numeric_limits<int>::max() / inner)
    return false;
  batch = int(outer * inner);
  return true;
}

bool checkedHostElements(int64_t first, int64_t second, int64_t third,
                         size_t elementBytes, size_t &elements) {
  if (first <= 0 || second <= 0 || third <= 0) return false;
  const uint64_t limit = std::numeric_limits<size_t>::max() / elementBytes;
  if (uint64_t(first) > limit / uint64_t(second))
    return false;
  const uint64_t firstSecond = uint64_t(first) * uint64_t(second);
  if (firstSecond > limit / uint64_t(third)) return false;
  elements = size_t(firstSecond * uint64_t(third));
  return true;
}

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

bool packHostLayout(const void *input, void *output, int rank,
                    const int64_t *shape, const int64_t *strides,
                    size_t bytes, size_t &elements) {
  if (!input || !output || !shape || !strides || rank <= 0 || rank > 8)
    return false;
  elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || (shape[dim] > 1 && strides[dim] == 0) ||
        size_t(shape[dim]) > std::numeric_limits<size_t>::max() / elements)
      return false;
    elements *= size_t(shape[dim]);
  }
  const auto *source = static_cast<const unsigned char *>(input);
  auto *target = static_cast<unsigned char *>(output);
  for (size_t logical = 0; logical < elements; ++logical) {
    size_t cursor = logical;
    int64_t offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t coordinate = int64_t(cursor % size_t(shape[dim]));
      cursor /= size_t(shape[dim]);
      offset += coordinate * strides[dim];
    }
    std::memcpy(target + logical * bytes,
                source + std::ptrdiff_t(offset) * std::ptrdiff_t(bytes), bytes);
  }
  return true;
}

bool expandHostWindows(const void *window, int storage, int windowRank,
                       const int64_t *windowShape,
                       const int64_t *windowStrides,
                       const std::vector<int64_t> &batchShape, int fftN,
                       std::vector<float> &expanded) {
  if (!window || !windowShape || !windowStrides || windowRank < 1 ||
      windowRank > 8 || windowRank - 1 > int(batchShape.size())) return false;
  int64_t win = windowShape[windowRank - 1];
  if (win <= 0 || win > fftN) return false;
  int leading = int(batchShape.size()) - (windowRank - 1);
  size_t batch = 1;
  for (int64_t extent : batchShape) {
    if (extent <= 0 || size_t(extent) > std::numeric_limits<size_t>::max() / batch)
      return false;
    batch *= size_t(extent);
  }
  for (int dim = 0; dim < windowRank - 1; ++dim)
    if (windowShape[dim] != 1 && windowShape[dim] != batchShape[leading + dim])
      return false;
  expanded.assign(batch * size_t(fftN), 0.0f);
  int offset = (fftN - int(win)) / 2;
  std::vector<int64_t> coordinate(batchShape.size());
  for (size_t row = 0; row < batch; ++row) {
    size_t cursor = row;
    for (int dim = int(batchShape.size()) - 1; dim >= 0; --dim) {
      coordinate[dim] = int64_t(cursor % size_t(batchShape[dim]));
      cursor /= size_t(batchShape[dim]);
    }
    int64_t base = 0;
    for (int dim = 0; dim < windowRank - 1; ++dim) {
      int64_t at = windowShape[dim] == 1 ? 0 : coordinate[leading + dim];
      base += at * windowStrides[dim];
    }
    for (int local = 0; local < win; ++local) {
      int64_t at = base + int64_t(local) * windowStrides[windowRank - 1];
      if (at < 0) return false;
      expanded[row * size_t(fftN) + offset + local] =
          loadStorage(window, size_t(at), storage);
    }
  }
  return true;
}

bool buildWindowRowMap(int windowRank, const int64_t *windowShape,
                       const std::vector<int64_t> &batchShape,
                       std::vector<int> &rowWindow, int &windowRows) {
  if (!windowShape || windowRank < 1 ||
      windowRank - 1 > int(batchShape.size())) return false;
  int leading = int(batchShape.size()) - (windowRank - 1);
  size_t batch = 1;
  for (int64_t extent : batchShape) batch *= size_t(extent);
  int64_t rows = 1;
  for (int dim = 0; dim < windowRank - 1; ++dim) {
    if (windowShape[dim] != 1 && windowShape[dim] != batchShape[leading + dim])
      return false;
    if (windowShape[dim] > std::numeric_limits<int>::max() / rows) return false;
    rows *= windowShape[dim];
  }
  if (rows <= 0 || rows > std::numeric_limits<int>::max()) return false;
  windowRows = int(rows);
  rowWindow.resize(batch);
  std::vector<int64_t> coordinate(batchShape.size());
  for (size_t row = 0; row < batch; ++row) {
    size_t cursor = row;
    for (int dim = int(batchShape.size()) - 1; dim >= 0; --dim) {
      coordinate[dim] = int64_t(cursor % size_t(batchShape[dim]));
      cursor /= size_t(batchShape[dim]);
    }
    int logical = 0;
    for (int dim = 0; dim < windowRank - 1; ++dim) {
      int selected = windowShape[dim] == 1 ? 0 : int(coordinate[leading + dim]);
      logical = logical * int(windowShape[dim]) + selected;
    }
    rowWindow[row] = logical;
  }
  return true;
}

} // namespace

extern "C" int ts_dct_plan_hostptr_batch_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost, void *outputHost,
    int batch, int n, int dctType, int storage, float outputScale) {
  if (!validStorage(storage) || !inputHost || !outputHost) return 120;
  if (storage == 0)
    return ts_dct_plan_hostptr_batch_amd(
        opaque, forwardPlan, static_cast<const float *>(inputHost),
        static_cast<float *>(outputHost), batch, n, dctType, outputScale);
  size_t elements = size_t(batch) * n;
  std::vector<float> input = unpackStorage(inputHost, elements, storage);
  std::vector<float> output(elements);
  int rc = ts_dct_plan_hostptr_batch_amd(
      opaque, forwardPlan, input.data(), output.data(), batch, n, dctType,
      outputScale);
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
    int64_t outer, int64_t n, int64_t inner, int dctType, int storage,
    float outputScale) {
  if (!validStorage(storage) || outer <= 0 || inner <= 0) return 160;
  size_t bytes = storageBytes(storage);
  size_t total = size_t(outer * n * inner) * bytes;
  std::vector<unsigned char> input(total), output(total);
  packAxis(inputHost, input.data(), outer, n, inner, bytes);
  int rc = ts_dct_plan_hostptr_batch_storage_amd(
      opaque, forwardPlan, input.data(), output.data(), int(outer * inner),
      int(n), dctType, storage, outputScale);
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

// TSOL-POLICY-PHYS-1: package-owned centered padding and ISTFT trimming.
// padMode is 0=constant, 1=reflect. Explicit output length is cropping-only.
extern "C" int ts_stft_plan_hostptr_policy_strided_storage_amd(
    void *opaque, void *forwardPlan, const void *inputHost,
    const void *windowHost, float2 *outputHost, int64_t outer, int samples,
    int64_t inner, int win, int hop, int frames, int storage,
    float outputScale, int center, int padMode) {
  if (!validStorage(storage) || !opaque || !forwardPlan || !inputHost ||
      !windowHost || !outputHost || outer <= 0 || inner <= 0 || samples <= 0 ||
      win <= 0 || hop <= 0 || (center != 0 && center != 1) ||
      (padMode != 0 && padMode != 1))
    return 200;
  int pad = center ? win / 2 : 0;
  if (padMode == 1 && samples <= pad) return 201;
  if (center && win != 16 && win != 18) return 202;
  int64_t paddedWide = int64_t(samples) + 2 * int64_t(pad);
  if (paddedWide > std::numeric_limits<int>::max()) return 202;
  int paddedSamples = int(paddedWide);
  if (frames != (paddedSamples - win) / hop + 1) return 202;
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 202;
  int bins = win / 2 + 1;
  size_t bytes = storageBytes(storage);
  size_t inputElements = 0, paddedElements = 0, outputElements = 0;
  if (!checkedHostElements(batch, samples, 1, bytes, inputElements) ||
      !checkedHostElements(batch, paddedSamples, 1, bytes, paddedElements) ||
      !checkedHostElements(batch, frames, bins, sizeof(float2), outputElements))
    return 202;
  std::vector<unsigned char> input(inputElements * bytes);
  std::vector<unsigned char> padded(paddedElements * bytes, 0);
  std::vector<float2> output(outputElements);
  packAxis(inputHost, input.data(), outer, samples, inner, bytes);
  for (int64_t row = 0; row < batch; ++row)
    for (int at = 0; at < paddedSamples; ++at) {
      int source = at - pad;
      bool present = source >= 0 && source < samples;
      if (!present && padMode == 1) {
        source = source < 0 ? -source : 2 * samples - 2 - source;
        present = source >= 0 && source < samples;
      }
      if (present)
        std::memcpy(padded.data() + size_t(row * paddedSamples + at) * bytes,
                    input.data() + size_t(row * samples + source) * bytes,
                    bytes);
    }
  int rc = ts_stft_plan_hostptr_batch_storage_amd(
      opaque, forwardPlan, padded.data(), windowHost, output.data(), batch,
      paddedSamples, win, hop, frames, storage, outputScale);
  if (!rc)
    unpackAxis(output.data(), outputHost, outer, int64_t(frames) * bins, inner,
               sizeof(float2));
  return rc;
}

extern "C" int ts_istft_plan_hostptr_policy_strided_storage_amd(
    void *opaque, void *inversePlan, const float2 *inputHost,
    const void *windowHost, void *outputHost, int64_t outer, int frames,
    int bins, int64_t inner, int win, int hop, int storage,
    float outputScale, int center, int outputSamples) {
  if (!validStorage(storage) || !opaque || !inversePlan || !inputHost ||
      !windowHost || !outputHost || outer <= 0 || inner <= 0 ||
      bins != win / 2 + 1 || frames <= 0 || hop <= 0 || win <= 0 ||
      (center != 0 && center != 1))
    return 210;
  int64_t rawWide = int64_t(frames - 1) * int64_t(hop) + win;
  if (rawWide > std::numeric_limits<int>::max()) return 211;
  int rawSamples = int(rawWide);
  int start = center ? win / 2 : 0;
  int available = rawSamples - 2 * start;
  bool expanded = center || outputSamples != rawSamples;
  if (outputSamples <= 0 || outputSamples > available ||
      (expanded && win != 16 && win != 18))
    return 211;
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 211;
  size_t bytes = storageBytes(storage);
  size_t inputElements = 0, rawElements = 0, croppedElements = 0;
  if (!checkedHostElements(batch, frames, bins, sizeof(float2), inputElements) ||
      !checkedHostElements(batch, rawSamples, 1, bytes, rawElements) ||
      !checkedHostElements(batch, outputSamples, 1, bytes, croppedElements))
    return 211;
  std::vector<float2> input(inputElements);
  std::vector<unsigned char> raw(rawElements * bytes);
  std::vector<unsigned char> cropped(croppedElements * bytes);
  packAxis(inputHost, input.data(), outer, int64_t(frames) * bins, inner,
           sizeof(float2));
  int rc = ts_istft_plan_hostptr_batch_storage_amd(
      opaque, inversePlan, input.data(), windowHost, raw.data(), batch,
      frames, win, hop, storage, outputScale);
  if (!rc) {
    for (int64_t row = 0; row < batch; ++row)
      std::memcpy(cropped.data() + size_t(row * outputSamples) * bytes,
                  raw.data() + size_t(row * rawSamples + start) * bytes,
                  size_t(outputSamples) * bytes);
    unpackAxis(cropped.data(), outputHost, outer, outputSamples, inner, bytes);
  }
  return rc;
}

extern "C" int ts_stft_plan_hostptr_broadcast_layout_storage_amd(
    void *opaque, const void *inputHost, const void *windowHost,
    float2 *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int fftN,
    int hop, int frames, int storage, float outputScale, int center,
    int padMode, int onesided) {
  if (!opaque || !inputHost || !windowHost || !outputHost ||
      !validStorage(storage) || rank <= 0 || rank > 8 || axis < 0 ||
      axis >= rank || fftN <= 0 || hop <= 0 || frames <= 0 ||
      (center != 0 && center != 1) || (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1)) return 220;
  size_t bytes = storageBytes(storage), elements = 1;
  std::vector<unsigned char> contiguous;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || size_t(shape[dim]) >
                               std::numeric_limits<size_t>::max() / elements)
      return 221;
    elements *= size_t(shape[dim]);
  }
  contiguous.resize(elements * bytes);
  size_t packedElements = 0;
  if (!packHostLayout(inputHost, contiguous.data(), rank, shape, strides,
                      bytes, packedElements) || packedElements != elements)
    return 221;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < rank; ++dim) {
    if (dim < axis) outer *= shape[dim];
    else if (dim > axis) inner *= shape[dim];
    if (dim != axis) batchShape.push_back(shape[dim]);
  }
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 221;
  int samples = int(shape[axis]);
  int pad = center ? fftN / 2 : 0;
  int paddedSamples = std::max(samples + 2 * pad, fftN);
  if (frames != (paddedSamples - fftN) / hop + 1 ||
      (padMode == 1 && center && samples <= pad)) return 222;
  std::vector<unsigned char> axisPacked(elements * bytes);
  packAxis(contiguous.data(), axisPacked.data(), outer, samples, inner, bytes);
  std::vector<float> input = unpackStorage(axisPacked.data(), elements, storage);
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, storage, windowRank, windowShape,
                         windowStrides, batchShape, fftN, windows)) return 223;
  int bins = onesided ? fftN / 2 + 1 : fftN;
  size_t outputElements = size_t(batch) * frames * bins;
  float *deviceInput = nullptr, *deviceWindows = nullptr;
  float2 *deviceOutput = nullptr;
  if (!allocate(&deviceInput, input.size()) ||
      !allocate(&deviceWindows, windows.size()) ||
      !allocate(&deviceOutput, outputElements)) {
    release(deviceInput, deviceWindows, deviceOutput);
    return 224;
  }
  int rc = 0;
  if (hipMemcpy(deviceInput, input.data(), input.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceWindows, windows.data(), windows.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) rc = 225;
  std::vector<float2> packed(outputElements);
  if (!rc) {
    hipLaunchKernelGGL(stftDirectPolicy,
        dim3((outputElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceInput, deviceWindows, deviceOutput, batch, samples, fftN, hop,
        frames, bins, outputScale, center, padMode);
    rc = syncAndCopy(packed.data(), deviceOutput,
                     outputElements * sizeof(float2));
  }
  if (!rc)
    unpackAxis(packed.data(), outputHost, outer, int64_t(frames) * bins, inner,
               sizeof(float2));
  release(deviceInput, deviceWindows, deviceOutput);
  return rc;
}

extern "C" int ts_istft_plan_hostptr_broadcast_layout_storage_amd(
    void *opaque, const float2 *inputHost, const void *windowHost,
    void *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int fftN,
    int hop, int storage, float outputScale, int center, int outputSamples,
    int onesided) {
  if (!opaque || !inputHost || !windowHost || !outputHost ||
      !validStorage(storage) || rank < 2 || rank > 8 || axis <= 0 ||
      axis >= rank || fftN <= 0 || hop <= 0 || outputSamples <= 0 ||
      (center != 0 && center != 1) || (onesided != 0 && onesided != 1))
    return 230;
  int frameAxis = axis - 1, frames = int(shape[frameAxis]);
  int bins = int(shape[axis]);
  if (bins != (onesided ? fftN / 2 + 1 : fftN)) return 231;
  size_t elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || size_t(shape[dim]) >
                               std::numeric_limits<size_t>::max() / elements)
      return 231;
    elements *= size_t(shape[dim]);
  }
  std::vector<float2> contiguous(elements), spectra(elements);
  size_t packedElements = 0;
  if (!packHostLayout(inputHost, contiguous.data(), rank, shape, strides,
                      sizeof(float2), packedElements) || packedElements != elements)
    return 231;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < rank; ++dim) {
    if (dim < frameAxis) outer *= shape[dim];
    else if (dim > axis) inner *= shape[dim];
    if (dim != frameAxis && dim != axis) batchShape.push_back(shape[dim]);
  }
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 231;
  packAxis(contiguous.data(), spectra.data(), outer, int64_t(frames) * bins,
           inner, sizeof(float2));
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, storage, windowRank, windowShape,
                         windowStrides, batchShape, fftN, windows)) return 232;
  int rawSamples = (frames - 1) * hop + fftN;
  int available = rawSamples - (center ? fftN : 0);
  if (outputSamples > available) return 233;
  size_t outputElements = size_t(batch) * outputSamples;
  float2 *deviceSpectrum = nullptr;
  float *deviceWindows = nullptr, *deviceOutput = nullptr;
  if (!allocate(&deviceSpectrum, spectra.size()) ||
      !allocate(&deviceWindows, windows.size()) ||
      !allocate(&deviceOutput, outputElements)) {
    release(deviceSpectrum, deviceWindows, deviceOutput);
    return 234;
  }
  int rc = 0;
  if (hipMemcpy(deviceSpectrum, spectra.data(), spectra.size() * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceWindows, windows.data(), windows.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) rc = 235;
  std::vector<float> output(outputElements);
  if (!rc) {
    hipLaunchKernelGGL(istftDirectPolicy,
        dim3((outputElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceSpectrum, deviceWindows, deviceOutput, batch, frames, bins, fftN,
        hop, outputSamples, outputScale, center, onesided);
    rc = syncAndCopy(output.data(), deviceOutput,
                     outputElements * sizeof(float));
  }
  if (!rc) {
    std::vector<unsigned char> stored(outputElements * storageBytes(storage));
    packStorage(output, stored.data(), storage);
    unpackAxis(stored.data(), outputHost, outer, outputSamples, inner,
               storageBytes(storage));
  }
  release(deviceSpectrum, deviceWindows, deviceOutput);
  return rc;
}

extern "C" int ts_streaming_stft_hostptr_broadcast_layout_storage_amd(
    const char *digest, const void *inputHost, const void *tailHost,
    const void *windowHost, float2 *outputHost, void *nextTailHost,
    int rank, const int64_t *shape, const int64_t *strides, int axis,
    int tailSamples, int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int fftN, int hop, int frames, int storage,
    float outputScale, int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !outputHost ||
      !nextTailHost || (tailSamples && !tailHost) || !validStorage(storage) ||
      rank <= 0 || rank > 8 || axis < 0 || axis >= rank || fftN <= 0 ||
      hop <= 0 || hop > fftN || tailSamples < 0 || tailSamples >= fftN ||
      frames < 0 || (onesided != 0 && onesided != 1)) return 240;
  size_t bytes = storageBytes(storage), elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || size_t(shape[dim]) >
                               std::numeric_limits<size_t>::max() / elements)
      return 241;
    elements *= size_t(shape[dim]);
  }
  std::vector<unsigned char> contiguous(elements * bytes);
  size_t packedElements = 0;
  if (!packHostLayout(inputHost, contiguous.data(), rank, shape, strides,
                      bytes, packedElements) || packedElements != elements)
    return 241;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < rank; ++dim) {
    if (dim < axis) outer *= shape[dim];
    else if (dim > axis) inner *= shape[dim];
    if (dim != axis) batchShape.push_back(shape[dim]);
  }
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 241;
  int chunkSamples = int(shape[axis]);
  int combinedSamples = tailSamples + chunkSamples;
  int expectedFrames = combinedSamples < fftN
                           ? 0 : (combinedSamples - fftN) / hop + 1;
  if (frames != expectedFrames) return 242;
  std::vector<unsigned char> axisPacked(elements * bytes);
  packAxis(contiguous.data(), axisPacked.data(), outer, chunkSamples, inner, bytes);
  std::vector<float> chunk = unpackStorage(axisPacked.data(), elements, storage);
  std::vector<float> prior = tailSamples
      ? unpackStorage(tailHost, size_t(batch) * tailSamples, storage)
      : std::vector<float>();
  std::vector<float> combined(size_t(batch) * combinedSamples);
  for (int row = 0; row < batch; ++row) {
    for (int at = 0; at < tailSamples; ++at)
      combined[size_t(row) * combinedSamples + at] =
          prior[size_t(row) * tailSamples + at];
    for (int at = 0; at < chunkSamples; ++at)
      combined[size_t(row) * combinedSamples + tailSamples + at] =
          chunk[size_t(row) * chunkSamples + at];
  }
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, storage, windowRank, windowShape,
                         windowStrides, batchShape, fftN, windows)) return 243;
  int bins = onesided ? fftN / 2 + 1 : fftN;
  size_t outputElements = size_t(batch) * frames * bins;
  float *deviceInput = nullptr, *deviceWindows = nullptr;
  float2 *deviceOutput = nullptr;
  if (!allocate(&deviceInput, combined.size()) ||
      !allocate(&deviceWindows, windows.size()) ||
      (outputElements && !allocate(&deviceOutput, outputElements))) {
    release(deviceInput, deviceWindows, deviceOutput);
    return 244;
  }
  int rc = 0;
  if (hipMemcpy(deviceInput, combined.data(), combined.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceWindows, windows.data(), windows.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess) rc = 245;
  std::vector<float2> output(outputElements);
  if (!rc && outputElements) {
    hipLaunchKernelGGL(stftDirectPolicy,
        dim3((outputElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceInput, deviceWindows, deviceOutput, batch, combinedSamples, fftN,
        hop, frames, bins, outputScale, 0, 0);
    rc = syncAndCopy(output.data(), deviceOutput,
                     outputElements * sizeof(float2));
  }
  if (!rc && outputElements)
    unpackAxis(output.data(), outputHost, outer, int64_t(frames) * bins, inner,
               sizeof(float2));
  int nextSamples = combinedSamples - frames * hop;
  std::vector<float> next(size_t(batch) * nextSamples);
  for (int row = 0; row < batch; ++row)
    for (int at = 0; at < nextSamples; ++at)
      next[size_t(row) * nextSamples + at] =
          combined[size_t(row) * combinedSamples + frames * hop + at];
  packStorage(next, nextTailHost, storage);
  release(deviceInput, deviceWindows, deviceOutput);
  return rc;
}

extern "C" int ts_stft_backward_hostptr_broadcast_layout_storage_amd(
    const char *digest, const float2 *dyHost, const void *inputHost,
    const void *windowHost, void *dxHost, void *dwindowHost,
    int xRank, const int64_t *xShape, const int64_t *xStrides, int axis,
    int dyRank, const int64_t *dyShape, const int64_t *dyStrides,
    int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int fftN, int hop, int storage,
    float forwardScale, int center, int padMode, int onesided) {
  if (!validDigest(digest) || !dyHost || !inputHost || !windowHost ||
      !dxHost || !dwindowHost || !validStorage(storage) || xRank <= 0 ||
      xRank > 8 || dyRank != xRank + 1 || axis < 0 || axis >= xRank ||
      fftN <= 0 || hop <= 0 || (center != 0 && center != 1) ||
      (padMode != 0 && padMode != 1) || (onesided != 0 && onesided != 1))
    return 250;
  int samples = int(xShape[axis]), frames = int(dyShape[axis]);
  int bins = int(dyShape[axis + 1]);
  if (bins != (onesided ? fftN / 2 + 1 : fftN)) return 251;
  int pad = center ? fftN / 2 : 0;
  int framedSamples = std::max(samples + 2 * pad, fftN);
  if (frames != (framedSamples - fftN) / hop + 1 ||
      (center && padMode == 1 && samples <= pad)) return 251;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < xRank; ++dim) {
    if (dim < axis) outer *= xShape[dim];
    else if (dim > axis) inner *= xShape[dim];
    if (dim != axis) batchShape.push_back(xShape[dim]);
  }
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 252;
  for (int dim = 0; dim < xRank; ++dim) {
    if (dim < axis && dyShape[dim] != xShape[dim]) return 252;
    if (dim > axis && dyShape[dim + 1] != xShape[dim]) return 252;
  }
  size_t inputElements = size_t(batch) * samples;
  size_t spectralElements = size_t(batch) * frames * bins;
  size_t bytes = storageBytes(storage), packedElements = 0;
  std::vector<unsigned char> inputContiguous(inputElements * bytes);
  if (!packHostLayout(inputHost, inputContiguous.data(), xRank, xShape,
                      xStrides, bytes, packedElements) ||
      packedElements != inputElements) return 253;
  std::vector<unsigned char> inputPacked(inputElements * bytes);
  packAxis(inputContiguous.data(), inputPacked.data(), outer, samples, inner,
           bytes);
  std::vector<float> input = unpackStorage(inputPacked.data(), inputElements,
                                           storage);
  std::vector<float2> dyContiguous(spectralElements), dy(spectralElements);
  if (!packHostLayout(dyHost, dyContiguous.data(), dyRank, dyShape, dyStrides,
                      sizeof(float2), packedElements) ||
      packedElements != spectralElements) return 253;
  packAxis(dyContiguous.data(), dy.data(), outer, int64_t(frames) * bins,
           inner, sizeof(float2));
  std::vector<float> windows;
  std::vector<int> rowWindow;
  int windowRows = 0;
  if (!expandHostWindows(windowHost, storage, windowRank, windowShape,
                         windowStrides, batchShape, fftN, windows) ||
      !buildWindowRowMap(windowRank, windowShape, batchShape, rowWindow,
                         windowRows)) return 254;
  int win = int(windowShape[windowRank - 1]);
  float2 *deviceDy = nullptr;
  float *deviceInput = nullptr, *deviceWindows = nullptr, *deviceDx = nullptr;
  float *deviceDwindow = nullptr;
  int *deviceRowWindow = nullptr;
  if (!allocate(&deviceDy, spectralElements) ||
      !allocate(&deviceInput, inputElements) ||
      !allocate(&deviceWindows, windows.size()) ||
      !allocate(&deviceDx, inputElements) ||
      !allocate(&deviceDwindow, size_t(windowRows) * win) ||
      !allocate(&deviceRowWindow, rowWindow.size())) {
    release(deviceDy, deviceInput, deviceWindows, deviceDx, deviceDwindow,
            deviceRowWindow);
    return 255;
  }
  int rc = 0;
  if (hipMemcpy(deviceDy, dy.data(), spectralElements * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceInput, input.data(), inputElements * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceWindows, windows.data(), windows.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceRowWindow, rowWindow.data(),
                rowWindow.size() * sizeof(int), hipMemcpyHostToDevice) != hipSuccess)
    rc = 256;
  std::vector<float> dx(inputElements), dwindow(size_t(windowRows) * win);
  if (!rc) {
    hipLaunchKernelGGL(stftBackwardInputPolicy,
        dim3((inputElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceDy, deviceWindows, deviceDx, batch, samples, fftN, hop, frames,
        bins, forwardScale, center, padMode);
    size_t windowElements = size_t(windowRows) * win;
    hipLaunchKernelGGL(stftBackwardWindowPolicy,
        dim3((windowElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceDy, deviceInput, deviceRowWindow, deviceDwindow, batch,
        windowRows, samples, fftN, win, hop, frames, bins, forwardScale,
        center, padMode);
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(dx.data(), deviceDx, inputElements * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(dwindow.data(), deviceDwindow,
                  windowElements * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
      rc = 257;
  }
  if (!rc) {
    std::vector<unsigned char> dxStored(inputElements * bytes);
    std::vector<unsigned char> dwStored(dwindow.size() * bytes);
    packStorage(dx, dxStored.data(), storage);
    packStorage(dwindow, dwStored.data(), storage);
    unpackAxis(dxStored.data(), dxHost, outer, samples, inner, bytes);
    std::memcpy(dwindowHost, dwStored.data(), dwStored.size());
  }
  release(deviceDy, deviceInput, deviceWindows, deviceDx, deviceDwindow,
          deviceRowWindow);
  return rc;
}

extern "C" int ts_istft_backward_hostptr_broadcast_layout_storage_amd(
    const char *digest, const void *dyHost, const float2 *spectrumHost,
    const void *windowHost, float2 *dspectrumHost, void *dwindowHost,
    int dyRank, const int64_t *dyShape, const int64_t *dyStrides,
    int outputAxis, int spectrumRank, const int64_t *spectrumShape,
    const int64_t *spectrumStrides, int frameAxis, int binAxis,
    int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int fftN, int hop, int storage,
    float inverseScale, int center, int onesided) {
  if (!validDigest(digest) || !dyHost || !spectrumHost || !windowHost ||
      !dspectrumHost || !dwindowHost || !validStorage(storage) ||
      spectrumRank != dyRank + 1 || outputAxis < 0 || outputAxis >= dyRank ||
      frameAxis < 0 || binAxis != frameAxis + 1 || binAxis >= spectrumRank ||
      fftN <= 0 || hop <= 0 || (center != 0 && center != 1) ||
      (onesided != 0 && onesided != 1)) return 260;
  int frames = int(spectrumShape[frameAxis]), bins = int(spectrumShape[binAxis]);
  if (bins != (onesided ? fftN / 2 + 1 : fftN)) return 261;
  int rawSamples = (frames - 1) * hop + fftN;
  int outputSamples = int(dyShape[outputAxis]);
  if (outputSamples <= 0 || outputSamples > rawSamples - (center ? fftN : 0))
    return 261;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < dyRank; ++dim) {
    if (dim < outputAxis) outer *= dyShape[dim];
    else if (dim > outputAxis) inner *= dyShape[dim];
    if (dim != outputAxis) batchShape.push_back(dyShape[dim]);
  }
  int batch = 0;
  if (!checkedHostBatch(outer, inner, batch)) return 262;
  for (int dim = 0; dim < dyRank; ++dim) {
    if (dim < outputAxis && spectrumShape[dim] != dyShape[dim]) return 262;
    if (dim > outputAxis && spectrumShape[dim + 1] != dyShape[dim]) return 262;
  }
  size_t bytes = storageBytes(storage), packedElements = 0;
  size_t dyElements = size_t(batch) * outputSamples;
  size_t spectralElements = size_t(batch) * frames * bins;
  std::vector<unsigned char> dyContiguous(dyElements * bytes), dyPacked(dyElements * bytes);
  if (!packHostLayout(dyHost, dyContiguous.data(), dyRank, dyShape, dyStrides,
                      bytes, packedElements) || packedElements != dyElements)
    return 263;
  packAxis(dyContiguous.data(), dyPacked.data(), outer, outputSamples, inner,
           bytes);
  std::vector<float> dy = unpackStorage(dyPacked.data(), dyElements, storage);
  std::vector<float2> spectrumContiguous(spectralElements), spectrum(spectralElements);
  if (!packHostLayout(spectrumHost, spectrumContiguous.data(), spectrumRank,
                      spectrumShape, spectrumStrides, sizeof(float2),
                      packedElements) || packedElements != spectralElements)
    return 263;
  packAxis(spectrumContiguous.data(), spectrum.data(), outer,
           int64_t(frames) * bins, inner, sizeof(float2));
  std::vector<float> windows;
  std::vector<int> rowWindow;
  int windowRows = 0;
  if (!expandHostWindows(windowHost, storage, windowRank, windowShape,
                         windowStrides, batchShape, fftN, windows) ||
      !buildWindowRowMap(windowRank, windowShape, batchShape, rowWindow,
                         windowRows)) return 264;
  int win = int(windowShape[windowRank - 1]);
  float *deviceDy = nullptr, *deviceWindows = nullptr, *deviceFrames = nullptr;
  float *deviceDframes = nullptr, *deviceDwindow = nullptr;
  float2 *deviceSpectrum = nullptr, *deviceDspectrum = nullptr;
  int *deviceRowWindow = nullptr;
  size_t frameElements = size_t(batch) * frames * fftN;
  if (!allocate(&deviceDy, dyElements) ||
      !allocate(&deviceSpectrum, spectralElements) ||
      !allocate(&deviceWindows, windows.size()) ||
      !allocate(&deviceFrames, frameElements) ||
      !allocate(&deviceDframes, frameElements) ||
      !allocate(&deviceDspectrum, spectralElements) ||
      !allocate(&deviceDwindow, size_t(windowRows) * win) ||
      !allocate(&deviceRowWindow, rowWindow.size())) {
    release(deviceDy, deviceSpectrum, deviceWindows, deviceFrames,
            deviceDframes, deviceDspectrum, deviceDwindow, deviceRowWindow);
    return 265;
  }
  int rc = 0;
  if (hipMemcpy(deviceDy, dy.data(), dyElements * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceSpectrum, spectrum.data(), spectralElements * sizeof(float2),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceWindows, windows.data(), windows.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(deviceRowWindow, rowWindow.data(), rowWindow.size() * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess) rc = 266;
  std::vector<float2> dspectrum(spectralElements);
  std::vector<float> dwindow(size_t(windowRows) * win);
  if (!rc) {
    hipLaunchKernelGGL(istftFrameValuesPolicy,
        dim3((frameElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceSpectrum, deviceFrames, batch, frames, bins, fftN, inverseScale,
        onesided);
    hipLaunchKernelGGL(istftBackwardFramesPolicy,
        dim3((frameElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceDy, deviceFrames, deviceWindows, deviceDframes, batch, frames,
        fftN, hop, outputSamples, center);
    hipLaunchKernelGGL(istftBackwardSpectrumPolicy,
        dim3((spectralElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceDframes, deviceDspectrum, batch, frames, bins, fftN,
        inverseScale, onesided);
    size_t windowElements = size_t(windowRows) * win;
    hipLaunchKernelGGL(istftBackwardWindowPolicy,
        dim3((windowElements + kThreads - 1) / kThreads), dim3(kThreads), 0, 0,
        deviceDy, deviceFrames, deviceWindows, deviceRowWindow, deviceDwindow,
        batch, windowRows, frames, fftN, win, hop, outputSamples, center);
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(dspectrum.data(), deviceDspectrum,
                  spectralElements * sizeof(float2), hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(dwindow.data(), deviceDwindow,
                  windowElements * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
      rc = 267;
  }
  if (!rc) {
    unpackAxis(dspectrum.data(), dspectrumHost, outer,
               int64_t(frames) * bins, inner, sizeof(float2));
    std::vector<unsigned char> dwStored(dwindow.size() * bytes);
    packStorage(dwindow, dwStored.data(), storage);
    std::memcpy(dwindowHost, dwStored.data(), dwStored.size());
  }
  release(deviceDy, deviceSpectrum, deviceWindows, deviceFrames,
          deviceDframes, deviceDspectrum, deviceDwindow, deviceRowWindow);
  return rc;
}
