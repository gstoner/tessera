#include "tessera_nvidia_fft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;

bool validDigest(const char *digest) {
  if (digest == nullptr)
    return true;
  if (std::strlen(digest) != 64)
    return false;
  for (const char *at = digest; *at; ++at)
    if (!((*at >= '0' && *at <= '9') || (*at >= 'a' && *at <= 'f')))
      return false;
  return true;
}

template <typename T>
bool packHostLayout(const T *input, std::vector<T> &output, int rank,
                    const int64_t *shape, const int64_t *strides) {
  if (!input || !shape || !strides || rank <= 0 || rank > 8)
    return false;
  size_t elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || (shape[dim] > 1 && strides[dim] == 0) ||
        size_t(shape[dim]) > std::numeric_limits<size_t>::max() / elements)
      return false;
    elements *= size_t(shape[dim]);
  }
  output.resize(elements);
  for (size_t logical = 0; logical < elements; ++logical) {
    size_t cursor = logical;
    int64_t offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t coordinate = int64_t(cursor % size_t(shape[dim]));
      cursor /= size_t(shape[dim]);
      offset += coordinate * strides[dim];
    }
    output[logical] = input[offset];
  }
  return true;
}

bool validStorage(int storage) { return storage >= 0 && storage <= 2; }

float halfToFloat(uint16_t half) {
  const uint32_t sign = uint32_t(half & 0x8000u) << 16;
  uint32_t exponent = (half >> 10) & 0x1fu;
  uint32_t mantissa = half & 0x03ffu;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      bits = sign | uint32_t(127 - 14 - shift) << 23 | mantissa << 13;
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | mantissa << 13;
  } else {
    bits = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t floatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu)
    return uint16_t(sign | 0x7c00u | (mantissa ? 0x0200u : 0));
  int adjusted = int(exponent) - 127 + 15;
  if (adjusted >= 31)
    return uint16_t(sign | 0x7c00u);
  if (adjusted <= 0) {
    if (adjusted < -10)
      return uint16_t(sign);
    mantissa |= 0x800000u;
    const int shift = 14 - adjusted;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1)))
      ++rounded;
    return uint16_t(sign | rounded);
  }
  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1))) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      if (++adjusted >= 31)
        return uint16_t(sign | 0x7c00u);
    }
  }
  return uint16_t(sign | uint32_t(adjusted) << 10 | rounded);
}

float loadStorage(const void *input, int64_t index, int storage) {
  if (storage == 0)
    return static_cast<const float *>(input)[index];
  uint16_t bits = static_cast<const uint16_t *>(input)[index];
  if (storage == 1)
    return halfToFloat(bits);
  uint32_t wide = uint32_t(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

void storeStorage(void *output, int64_t index, int storage, float value) {
  if (storage == 0) {
    static_cast<float *>(output)[index] = value;
    return;
  }
  if (storage == 1) {
    static_cast<uint16_t *>(output)[index] = floatToHalf(value);
    return;
  }
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
  static_cast<uint16_t *>(output)[index] = uint16_t((bits + rounding) >> 16);
}

bool packStorageLayout(const void *input, std::vector<float> &output, int rank,
                       const int64_t *shape, const int64_t *strides,
                       int storage) {
  if (!input || !shape || !strides || !validStorage(storage) || rank <= 0 ||
      rank > 8)
    return false;
  size_t elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 || (shape[dim] > 1 && strides[dim] == 0) ||
        size_t(shape[dim]) > std::numeric_limits<size_t>::max() / elements)
      return false;
    elements *= size_t(shape[dim]);
  }
  output.resize(elements);
  for (size_t logical = 0; logical < elements; ++logical) {
    size_t cursor = logical;
    int64_t offset = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t coordinate = int64_t(cursor % size_t(shape[dim]));
      cursor /= size_t(shape[dim]);
      offset += coordinate * strides[dim];
    }
    output[logical] = loadStorage(input, offset, storage);
  }
  return true;
}

std::vector<int64_t> compactStrides(int rank, const int64_t *shape) {
  std::vector<int64_t> strides(size_t(rank), 1);
  for (int dim = rank - 2; dim >= 0; --dim)
    strides[size_t(dim)] = strides[size_t(dim + 1)] * shape[dim + 1];
  return strides;
}

template <typename T>
void packAxis(const T *input, T *output, int64_t outer, int64_t axisExtent,
              int64_t inner) {
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j)
      for (int64_t i = 0; i < axisExtent; ++i)
        output[(o * inner + j) * axisExtent + i] =
            input[(o * axisExtent + i) * inner + j];
}

template <typename T>
void unpackAxis(const T *input, T *output, int64_t outer, int64_t axisExtent,
                int64_t inner) {
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j)
      for (int64_t i = 0; i < axisExtent; ++i)
        output[(o * axisExtent + i) * inner + j] =
            input[(o * inner + j) * axisExtent + i];
}

bool foldedBatch(int rank, const int64_t *shape, int axis, int64_t &outer,
                 int64_t &inner, int &batch,
                 std::vector<int64_t> &batchShape) {
  outer = 1;
  inner = 1;
  batchShape.clear();
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0)
      return false;
    if (dim < axis)
      outer *= shape[dim];
    else if (dim > axis)
      inner *= shape[dim];
    if (dim != axis)
      batchShape.push_back(shape[dim]);
  }
  if (outer <= 0 || inner <= 0 || outer > INT32_MAX / inner)
    return false;
  batch = int(outer * inner);
  return true;
}

bool expandHostWindows(const float *window, int windowRank,
                       const int64_t *windowShape,
                       const int64_t *windowStrides,
                       const std::vector<int64_t> &batchShape, int nfft,
                       std::vector<float> &expanded) {
  if (!window || !windowShape || !windowStrides || windowRank < 1 ||
      windowRank > 8 || windowRank - 1 > int(batchShape.size()))
    return false;
  int64_t win = windowShape[windowRank - 1];
  if (win <= 0 || win > nfft)
    return false;
  int leading = int(batchShape.size()) - (windowRank - 1);
  size_t batch = 1;
  for (int64_t extent : batchShape) {
    if (extent <= 0 || size_t(extent) >
                           std::numeric_limits<size_t>::max() / batch)
      return false;
    batch *= size_t(extent);
  }
  for (int dim = 0; dim < windowRank - 1; ++dim)
    if (windowShape[dim] != 1 &&
        windowShape[dim] != batchShape[leading + dim])
      return false;
  expanded.assign(batch * size_t(nfft), 0.0f);
  int placement = (nfft - int(win)) / 2;
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
    for (int local = 0; local < win; ++local)
      expanded[row * size_t(nfft) + placement + local] =
          window[base + int64_t(local) * windowStrides[windowRank - 1]];
  }
  return true;
}

bool buildWindowRowMap(int windowRank, const int64_t *windowShape,
                       const std::vector<int64_t> &batchShape,
                       std::vector<int> &rowWindow, int &windowRows) {
  if (!windowShape || windowRank < 1 ||
      windowRank - 1 > int(batchShape.size()))
    return false;
  int leading = int(batchShape.size()) - (windowRank - 1);
  size_t batch = 1;
  for (int64_t extent : batchShape)
    batch *= size_t(extent);
  int64_t rows = 1;
  for (int dim = 0; dim < windowRank - 1; ++dim) {
    if (windowShape[dim] != 1 &&
        windowShape[dim] != batchShape[leading + dim])
      return false;
    if (windowShape[dim] <= 0 ||
        windowShape[dim] > std::numeric_limits<int>::max() / rows)
      return false;
    rows *= windowShape[dim];
  }
  if (rows <= 0 || rows > std::numeric_limits<int>::max())
    return false;
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
      int selected = windowShape[dim] == 1
                         ? 0
                         : int(coordinate[leading + dim]);
      logical = logical * int(windowShape[dim]) + selected;
    }
    rowWindow[row] = logical;
  }
  return true;
}

__device__ int reflectIndex(int source, int samples) {
  while (source < 0 || source >= samples)
    source = source < 0 ? -source : 2 * samples - 2 - source;
  return source;
}

__global__ void frameRealPolicy(const float *input, const float *windows,
                                float *framesOut, int batch, int samples,
                                int nfft, int hop, int frames, int center,
                                int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int source = frame * hop + local - (center ? nfft / 2 : 0);
  if ((source < 0 || source >= samples) && padMode == 1)
    source = reflectIndex(source, samples);
  float value = source >= 0 && source < samples
                    ? input[size_t(row) * samples + source]
                    : 0.0f;
  framesOut[index] = value * windows[size_t(row) * nfft + local];
}

__global__ void frameComplexPolicy(const float *input, const float *windows,
                                   cufftComplex *framesOut, int batch,
                                   int samples, int nfft, int hop, int frames,
                                   int center, int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int source = frame * hop + local - (center ? nfft / 2 : 0);
  if ((source < 0 || source >= samples) && padMode == 1)
    source = reflectIndex(source, samples);
  float value = source >= 0 && source < samples
                    ? input[size_t(row) * samples + source]
                    : 0.0f;
  framesOut[index] = make_cuFloatComplex(
      value * windows[size_t(row) * nfft + local], 0.0f);
}

__global__ void frameRealJVPPolicy(
    const float *input, const float *windows, const float *dinput,
    const float *dwindows, float *framesOut, float *dframesOut, int batch,
    int samples, int nfft, int hop, int frames, int center, int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int source = frame * hop + local - (center ? nfft / 2 : 0);
  if ((source < 0 || source >= samples) && padMode == 1)
    source = reflectIndex(source, samples);
  float value = source >= 0 && source < samples
                    ? input[size_t(row) * samples + source]
                    : 0.0f;
  float dvalue = source >= 0 && source < samples
                     ? dinput[size_t(row) * samples + source]
                     : 0.0f;
  float window = windows[size_t(row) * nfft + local];
  float dwindow = dwindows[size_t(row) * nfft + local];
  framesOut[index] = value * window;
  dframesOut[index] = dvalue * window + value * dwindow;
}

__global__ void frameComplexJVPPolicy(
    const float *input, const float *windows, const float *dinput,
    const float *dwindows, cufftComplex *framesOut,
    cufftComplex *dframesOut, int batch, int samples, int nfft, int hop,
    int frames, int center, int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int source = frame * hop + local - (center ? nfft / 2 : 0);
  if ((source < 0 || source >= samples) && padMode == 1)
    source = reflectIndex(source, samples);
  float value = source >= 0 && source < samples
                    ? input[size_t(row) * samples + source]
                    : 0.0f;
  float dvalue = source >= 0 && source < samples
                     ? dinput[size_t(row) * samples + source]
                     : 0.0f;
  float window = windows[size_t(row) * nfft + local];
  float dwindow = dwindows[size_t(row) * nfft + local];
  framesOut[index] = make_cuFloatComplex(value * window, 0.0f);
  dframesOut[index] =
      make_cuFloatComplex(dvalue * window + value * dwindow, 0.0f);
}

__global__ void scaleComplex(cufftComplex *values, size_t elements,
                             float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index].x *= scale;
    values[index].y *= scale;
  }
}

__global__ void dctDirectPolicy(const float *input, float *output, int batch,
                                int length, int dctType, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * length;
  if (index >= total)
    return;
  int row = int(index / length);
  int k = int(index % length);
  const float *values = input + size_t(row) * length;
  double result = 0.0;
  if (dctType == 1) {
    result = double(values[0]) + ((k & 1) ? -1.0 : 1.0) *
                                      double(values[length - 1]);
    for (int n = 1; n + 1 < length; ++n)
      result += 2.0 * double(values[n]) *
                cos(M_PI * double(n * k) / double(length - 1));
  } else if (dctType == 2) {
    for (int n = 0; n < length; ++n)
      result += 2.0 * double(values[n]) *
                cos(M_PI * double((2 * n + 1) * k) /
                    double(2 * length));
  } else if (dctType == 3) {
    result = double(values[0]);
    for (int n = 1; n < length; ++n)
      result += 2.0 * double(values[n]) *
                cos(M_PI * double(n * (2 * k + 1)) /
                    double(2 * length));
  } else {
    for (int n = 0; n < length; ++n)
      result += 2.0 * double(values[n]) *
                cos(M_PI * double((2 * n + 1) * (2 * k + 1)) /
                    double(4 * length));
  }
  output[index] = float(result * double(scale));
}

__global__ void stftBackwardInputPolicy(
    const cufftComplex *dy, const float *windows, float *dx, int batch,
    int samples, int nfft, int hop, int frames, int bins, float scale,
    int center, int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * samples;
  if (index >= total)
    return;
  int sampleIndex = int(index % samples);
  int row = int(index / samples);
  int pad = center ? nfft / 2 : 0;
  double result = 0.0;
  for (int frame = 0; frame < frames; ++frame)
    for (int local = 0; local < nfft; ++local) {
      int source = frame * hop + local - pad;
      if ((source < 0 || source >= samples) && padMode == 1)
        source = reflectIndex(source, samples);
      if (source != sampleIndex)
        continue;
      double gradient = 0.0;
      for (int bin = 0; bin < bins; ++bin) {
        cufftComplex upstream =
            dy[(size_t(row) * frames + frame) * bins + bin];
        double angle = 2.0 * M_PI * double(bin * local) / double(nfft);
        gradient += double(upstream.x) * cos(angle) -
                    double(upstream.y) * sin(angle);
      }
      result += gradient * double(scale) *
                double(windows[size_t(row) * nfft + local]);
    }
  dx[index] = float(result);
}

__global__ void stftBackwardWindowPolicy(
    const cufftComplex *dy, const float *input, const int *rowWindow,
    float *dwindow, int batch, int windowRows, int samples, int nfft,
    int win, int hop, int frames, int bins, float scale, int center,
    int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(windowRows) * win;
  if (index >= total)
    return;
  int localWindow = int(index % win);
  int windowRow = int(index / win);
  int local = (nfft - win) / 2 + localWindow;
  int pad = center ? nfft / 2 : 0;
  double result = 0.0;
  for (int row = 0; row < batch; ++row) {
    if (rowWindow[row] != windowRow)
      continue;
    for (int frame = 0; frame < frames; ++frame) {
      int source = frame * hop + local - pad;
      bool present = source >= 0 && source < samples;
      if (!present && padMode == 1) {
        source = reflectIndex(source, samples);
        present = true;
      }
      if (!present)
        continue;
      double gradient = 0.0;
      for (int bin = 0; bin < bins; ++bin) {
        cufftComplex upstream =
            dy[(size_t(row) * frames + frame) * bins + bin];
        double angle = 2.0 * M_PI * double(bin * local) / double(nfft);
        gradient += double(upstream.x) * cos(angle) -
                    double(upstream.y) * sin(angle);
      }
      result += gradient * double(scale) *
                double(input[size_t(row) * samples + source]);
    }
  }
  dwindow[index] = float(result);
}

__global__ void istftFrameValuesPolicy(const cufftComplex *spectrum,
                                       float *framesOut, int batch, int frames,
                                       int bins, int nfft, float inverseScale,
                                       int onesided) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  double value = 0.0;
  for (int bin = 0; bin < bins; ++bin) {
    double weight = 1.0;
    if (onesided && bin > 0 && !(nfft % 2 == 0 && bin == bins - 1))
      weight = 2.0;
    cufftComplex spectral =
        spectrum[(size_t(row) * frames + frame) * bins + bin];
    double angle = 2.0 * M_PI * double(bin * local) / double(nfft);
    value += weight * (double(spectral.x) * cos(angle) -
                       double(spectral.y) * sin(angle));
  }
  framesOut[index] = float(value * double(inverseScale));
}

__global__ void istftBackwardFramesPolicy(
    const float *dy, const float *frameValues, const float *windows,
    float *dframes, int batch, int frames, int nfft, int hop,
    int outputSamples, int center) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * nfft;
  if (index >= total)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int output = frame * hop + local - (center ? nfft / 2 : 0);
  if (output < 0 || output >= outputSamples) {
    dframes[index] = 0.0f;
    return;
  }
  int rawSample = output + (center ? nfft / 2 : 0);
  double numerator = 0.0;
  double denominator = 0.0;
  for (int other = 0; other < frames; ++other) {
    int otherLocal = rawSample - other * hop;
    if (otherLocal < 0 || otherLocal >= nfft)
      continue;
    double window = windows[size_t(row) * nfft + otherLocal];
    numerator += double(frameValues[
                     (size_t(row) * frames + other) * nfft + otherLocal]) *
                 window;
    denominator += window * window;
  }
  double safe = denominator > 1.0e-12 ? denominator : 1.0e-12;
  dframes[index] = float(double(dy[size_t(row) * outputSamples + output]) /
                         safe *
                         double(windows[size_t(row) * nfft + local]));
}

__global__ void istftBackwardSpectrumPolicy(
    const float *dframes, cufftComplex *dspectrum, int batch, int frames,
    int bins, int nfft, float inverseScale, int onesided) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * frames * bins;
  if (index >= total)
    return;
  int bin = int(index % bins);
  size_t rowFrame = index / bins;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  double real = 0.0;
  double imag = 0.0;
  for (int local = 0; local < nfft; ++local) {
    double angle = 2.0 * M_PI * double(bin * local) / double(nfft);
    double value =
        dframes[(size_t(row) * frames + frame) * nfft + local];
    real += value * cos(angle);
    imag -= value * sin(angle);
  }
  double weight = double(inverseScale);
  if (onesided && bin > 0 && !(nfft % 2 == 0 && bin == bins - 1))
    weight *= 2.0;
  dspectrum[index] = make_cuFloatComplex(float(real * weight),
                                         float(imag * weight));
}

__global__ void istftBackwardWindowPolicy(
    const float *dy, const float *frameValues, const float *windows,
    const int *rowWindow, float *dwindow, int batch, int windowRows,
    int frames, int nfft, int win, int hop, int outputSamples, int center) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(windowRows) * win;
  if (index >= total)
    return;
  int localWindow = int(index % win);
  int windowRow = int(index / win);
  int local = (nfft - win) / 2 + localWindow;
  int trim = center ? nfft / 2 : 0;
  double result = 0.0;
  for (int row = 0; row < batch; ++row) {
    if (rowWindow[row] != windowRow)
      continue;
    for (int frame = 0; frame < frames; ++frame) {
      int output = frame * hop + local - trim;
      if (output < 0 || output >= outputSamples)
        continue;
      int rawSample = output + trim;
      double numerator = 0.0;
      double denominator = 0.0;
      for (int other = 0; other < frames; ++other) {
        int otherLocal = rawSample - other * hop;
        if (otherLocal < 0 || otherLocal >= nfft)
          continue;
        double window = windows[size_t(row) * nfft + otherLocal];
        numerator += double(frameValues[
                         (size_t(row) * frames + other) * nfft + otherLocal]) *
                     window;
        denominator += window * window;
      }
      double safe = denominator > 1.0e-12 ? denominator : 1.0e-12;
      double upstream = dy[size_t(row) * outputSamples + output];
      double draw = upstream / safe;
      double dweight = denominator > 1.0e-12
                           ? -upstream * numerator / (safe * safe)
                           : 0.0;
      double window = windows[size_t(row) * nfft + local];
      double frameValue = frameValues[
          (size_t(row) * frames + frame) * nfft + local];
      result += draw * frameValue + 2.0 * dweight * window;
    }
  }
  dwindow[index] = float(result);
}

__global__ void overlapAddReal(const float *framesIn, const float *windows,
                               float *output, int batch, int frames, int nfft,
                               int hop, int outputSamples, int trim,
                               float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputSamples;
  if (index >= total)
    return;
  int row = int(index / outputSamples);
  int rawSample = int(index % outputSamples) + trim;
  double numerator = 0.0;
  double denominator = 0.0;
  for (int frame = 0; frame < frames; ++frame) {
    int local = rawSample - frame * hop;
    if (local < 0 || local >= nfft)
      continue;
    float window = windows[size_t(row) * nfft + local];
    numerator += double(framesIn[(size_t(row) * frames + frame) * nfft + local]) *
                 double(window);
    denominator += double(window) * double(window);
  }
  output[index] = float(numerator /
                        (denominator > 1.0e-12 ? denominator : 1.0e-12) *
                        double(scale));
}

__global__ void overlapAddComplex(const cufftComplex *framesIn,
                                  const float *windows, float *output,
                                  int batch, int frames, int nfft, int hop,
                                  int outputSamples, int trim, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputSamples;
  if (index >= total)
    return;
  int row = int(index / outputSamples);
  int rawSample = int(index % outputSamples) + trim;
  double numerator = 0.0;
  double denominator = 0.0;
  for (int frame = 0; frame < frames; ++frame) {
    int local = rawSample - frame * hop;
    if (local < 0 || local >= nfft)
      continue;
    float window = windows[size_t(row) * nfft + local];
    numerator += double(framesIn[(size_t(row) * frames + frame) * nfft + local].x) *
                 double(window);
    denominator += double(window) * double(window);
  }
  output[index] = float(numerator /
                        (denominator > 1.0e-12 ? denominator : 1.0e-12) *
                        double(scale));
}

template <typename Frame>
__device__ float frameReal(const Frame *frames, size_t index);

template <>
__device__ float frameReal<float>(const float *frames, size_t index) {
  return frames[index];
}

template <>
__device__ float frameReal<cufftComplex>(const cufftComplex *frames,
                                        size_t index) {
  return frames[index].x;
}

template <typename Frame>
__global__ void overlapAddJVP(
    const Frame *framesIn, const Frame *dframesIn, const float *windows,
    const float *dwindows, float *primal, float *tangent, int batch,
    int frames, int nfft, int hop, int outputSamples, int trim, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = size_t(batch) * outputSamples;
  if (index >= total)
    return;
  int row = int(index / outputSamples);
  int rawSample = int(index % outputSamples) + trim;
  double numerator = 0.0, denominator = 0.0;
  double dnumerator = 0.0, ddenominator = 0.0;
  for (int frame = 0; frame < frames; ++frame) {
    int local = rawSample - frame * hop;
    if (local < 0 || local >= nfft)
      continue;
    size_t frameIndex = (size_t(row) * frames + frame) * nfft + local;
    size_t windowIndex = size_t(row) * nfft + local;
    double value = frameReal(framesIn, frameIndex);
    double dvalue = frameReal(dframesIn, frameIndex);
    double window = windows[windowIndex];
    double dwindow = dwindows[windowIndex];
    numerator += value * window;
    denominator += window * window;
    dnumerator += dvalue * window + value * dwindow;
    ddenominator += 2.0 * window * dwindow;
  }
  double safe = denominator > 1.0e-12 ? denominator : 1.0e-12;
  primal[index] = float(numerator / safe * double(scale));
  tangent[index] = float((dnumerator / safe -
                          numerator * ddenominator / (safe * safe)) *
                         double(scale));
}

int makePlan(int batch, int nfft, cufftType type, cufftHandle &plan,
             void *&workspace) {
  plan = 0;
  workspace = nullptr;
  if (cufftCreate(&plan) != CUFFT_SUCCESS ||
      cufftSetAutoAllocation(plan, 0) != CUFFT_SUCCESS)
    return 1;
  int bins = nfft / 2 + 1;
  int inputDistance = type == CUFFT_C2R ? bins : nfft;
  int outputDistance = type == CUFFT_R2C ? bins : nfft;
  size_t workspaceBytes = 0;
  if (cufftMakePlanMany(plan, 1, &nfft, nullptr, 1, inputDistance, nullptr, 1,
                        outputDistance, type, batch, &workspaceBytes) !=
      CUFFT_SUCCESS)
    return 2;
  if (cudaMalloc(&workspace, std::max<size_t>(workspaceBytes, 1)) != cudaSuccess)
    return 3;
  return cufftSetWorkArea(plan, workspace) == CUFFT_SUCCESS ? 0 : 4;
}

void destroyPlan(cufftHandle plan, void *workspace) {
  if (workspace)
    cudaFree(workspace);
  if (plan)
    cufftDestroy(plan);
}

template <typename... Pointers> void release(Pointers... pointers) {
  ((pointers ? (void)cudaFree(pointers) : (void)0), ...);
}

bool checkedProduct(size_t a, size_t b, size_t &product) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    return false;
  product = a * b;
  return true;
}

} // namespace

extern "C" const char *tessera_nvidia_spectral_package_abi() {
  return "tessera.nvidia.spectral_policy.v1";
}

extern "C" int tessera_nvidia_spectral_arch() {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return 0;
  return properties.major * 10 + properties.minor;
}

extern "C" int tessera_nvidia_dct_policy_layout_f32(
    const char *digest, const float *inputHost, float *outputHost, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int dctType,
    float outputScale) {
  if (!validDigest(digest) || !inputHost || !outputHost || rank <= 0 ||
      rank > 8 || axis < 0 || axis >= rank || dctType < 1 || dctType > 4 ||
      (dctType == 1 && shape[axis] < 2))
    return 290;
  std::vector<float> contiguous;
  if (!packHostLayout(inputHost, contiguous, rank, shape, strides) ||
      shape[axis] > INT32_MAX)
    return 291;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(rank, shape, axis, outer, inner, batch, batchShape))
    return 291;
  int length = int(shape[axis]);
  std::vector<float> packed(contiguous.size()), output(contiguous.size());
  packAxis(contiguous.data(), packed.data(), outer, length, inner);
  float *deviceInput = nullptr, *deviceOutput = nullptr;
  cudaError_t status = cudaMalloc(&deviceInput, packed.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceOutput, output.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceInput, packed.data(), packed.size() * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    dctDirectPolicy<<<unsigned((packed.size() + kThreads - 1) / kThreads),
                      kThreads>>>(deviceInput, deviceOutput, batch, length,
                                  dctType, outputScale);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(output.data(), deviceOutput,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  release(deviceInput, deviceOutput);
  if (status != cudaSuccess)
    return 292;
  unpackAxis(output.data(), outputHost, outer, length, inner);
  return 0;
}

extern "C" int tessera_nvidia_stft_policy_broadcast_layout_f32(
    const char *digest, const float *inputHost, const float *windowHost,
    float *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int frames, float outputScale, int center, int padMode,
    int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !outputHost ||
      rank <= 0 || rank > 8 || axis < 0 || axis >= rank || nfft <= 0 ||
      hop <= 0 || frames <= 0 || (center != 0 && center != 1) ||
      (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1))
    return 300;
  std::vector<float> contiguous;
  if (!packHostLayout(inputHost, contiguous, rank, shape, strides))
    return 301;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(rank, shape, axis, outer, inner, batch, batchShape) ||
      shape[axis] > INT32_MAX)
    return 301;
  int samples = int(shape[axis]);
  int pad = center ? nfft / 2 : 0;
  if (padMode == 1 && center && samples <= pad)
    return 302;
  int64_t padded = std::max<int64_t>(int64_t(samples) + 2 * pad, nfft);
  if (padded > INT32_MAX || frames != (padded - nfft) / hop + 1)
    return 302;
  std::vector<float> packed(contiguous.size());
  packAxis(contiguous.data(), packed.data(), outer, samples, inner);
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 303;

  int bins = onesided ? nfft / 2 + 1 : nfft;
  size_t frameElements = 0, outputElements = 0;
  if (!checkedProduct(size_t(batch) * frames, size_t(nfft), frameElements) ||
      !checkedProduct(size_t(batch) * frames, size_t(bins), outputElements))
    return 304;
  float *deviceInput = nullptr, *deviceWindows = nullptr,
        *deviceRealFrames = nullptr;
  cufftComplex *deviceComplexFrames = nullptr, *deviceOutput = nullptr;
  cudaError_t status = cudaMalloc(&deviceInput, packed.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceRealFrames, frameElements * sizeof(float));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceComplexFrames,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceOutput, outputElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceInput, packed.data(), packed.size() * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    release(deviceInput, deviceWindows, deviceRealFrames, deviceComplexFrames,
            deviceOutput);
    return 305;
  }
  unsigned frameBlocks = unsigned((frameElements + kThreads - 1) / kThreads);
  if (onesided)
    frameRealPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceRealFrames, batch, samples, nfft,
        hop, frames, center, padMode);
  else
    frameComplexPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceComplexFrames, batch, samples, nfft,
        hop, frames, center, padMode);
  status = cudaGetLastError();
  cufftHandle plan = 0;
  void *workspace = nullptr;
  int planStatus = status == cudaSuccess
                       ? makePlan(batch * frames, nfft,
                                  onesided ? CUFFT_R2C : CUFFT_C2C, plan,
                                  workspace)
                       : 1;
  cufftResult fftStatus = CUFFT_INVALID_PLAN;
  if (!planStatus)
    fftStatus = onesided
                    ? cufftExecR2C(plan, deviceRealFrames, deviceOutput)
                    : cufftExecC2C(plan, deviceComplexFrames, deviceOutput,
                                   CUFFT_FORWARD);
  if (!planStatus && fftStatus == CUFFT_SUCCESS && outputScale != 1.0f)
    scaleComplex<<<unsigned((outputElements + kThreads - 1) / kThreads),
                   kThreads>>>(deviceOutput, outputElements, outputScale);
  status = cudaGetLastError();
  std::vector<cufftComplex> output(outputElements);
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaMemcpy(output.data(), deviceOutput,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  destroyPlan(plan, workspace);
  release(deviceInput, deviceWindows, deviceRealFrames, deviceComplexFrames,
          deviceOutput);
  if (planStatus || fftStatus != CUFFT_SUCCESS || status != cudaSuccess)
    return 306;
  unpackAxis(output.data(), reinterpret_cast<cufftComplex *>(outputHost), outer,
             int64_t(frames) * bins, inner);
  return 0;
}

extern "C" int tessera_nvidia_stft_jvp_broadcast_layout_f32(
    const char *digest, const float *inputHost, const float *windowHost,
    const float *dinputHost, const float *dwindowHost, float *primalHost,
    float *tangentHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int frames, float outputScale, int center, int padMode,
    int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !primalHost ||
      !tangentHost || rank <= 0 || rank > 8 || axis < 0 || axis >= rank ||
      nfft <= 0 || hop <= 0 || frames <= 0 ||
      (center != 0 && center != 1) || (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1))
    return 360;
  std::vector<float> contiguous;
  if (!packHostLayout(inputHost, contiguous, rank, shape, strides))
    return 361;
  std::vector<float> dcontiguous(contiguous.size(), 0.0f);
  if (dinputHost &&
      !packHostLayout(dinputHost, dcontiguous, rank, shape, strides))
    return 361;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(rank, shape, axis, outer, inner, batch, batchShape) ||
      shape[axis] > INT32_MAX)
    return 361;
  int samples = int(shape[axis]);
  int pad = center ? nfft / 2 : 0;
  if (padMode == 1 && center && samples <= pad)
    return 362;
  int64_t padded = std::max<int64_t>(int64_t(samples) + 2 * pad, nfft);
  if (padded > INT32_MAX || frames != (padded - nfft) / hop + 1)
    return 362;
  std::vector<float> input(contiguous.size()), dinput(dcontiguous.size());
  packAxis(contiguous.data(), input.data(), outer, samples, inner);
  packAxis(dcontiguous.data(), dinput.data(), outer, samples, inner);
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 363;
  std::vector<float> dwindows(windows.size(), 0.0f);
  if (dwindowHost &&
      !expandHostWindows(dwindowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, dwindows))
    return 363;
  int bins = onesided ? nfft / 2 + 1 : nfft;
  size_t frameElements = size_t(batch) * frames * nfft;
  size_t outputElements = size_t(batch) * frames * bins;
  float *deviceInput = nullptr, *deviceDinput = nullptr;
  float *deviceWindows = nullptr, *deviceDwindows = nullptr;
  float *deviceFrames = nullptr, *deviceDframes = nullptr;
  cufftComplex *deviceFramesComplex = nullptr, *deviceDframesComplex = nullptr;
  cufftComplex *devicePrimal = nullptr, *deviceTangent = nullptr;
  cudaError_t status = cudaMalloc(&deviceInput, input.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDinput, dinput.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDwindows, dwindows.size() * sizeof(float));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceFrames, frameElements * sizeof(float));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceDframes, frameElements * sizeof(float));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceFramesComplex,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceDframesComplex,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&devicePrimal,
                        outputElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceTangent,
                        outputElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceInput, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDinput, dinput.data(), dinput.size() * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDwindows, dwindows.data(),
                        dwindows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    release(deviceInput, deviceDinput, deviceWindows, deviceDwindows,
            deviceFrames, deviceDframes, deviceFramesComplex,
            deviceDframesComplex, devicePrimal, deviceTangent);
    return 364;
  }
  unsigned frameBlocks = unsigned((frameElements + kThreads - 1) / kThreads);
  if (onesided)
    frameRealJVPPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceDinput, deviceDwindows, deviceFrames,
        deviceDframes, batch, samples, nfft, hop, frames, center, padMode);
  else
    frameComplexJVPPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceDinput, deviceDwindows,
        deviceFramesComplex, deviceDframesComplex, batch, samples, nfft, hop,
        frames, center, padMode);
  status = cudaGetLastError();
  cufftHandle plan = 0;
  void *workspace = nullptr;
  int planStatus = status == cudaSuccess
                       ? makePlan(batch * frames, nfft,
                                  onesided ? CUFFT_R2C : CUFFT_C2C, plan,
                                  workspace)
                       : 1;
  cufftResult first = CUFFT_INVALID_PLAN, second = CUFFT_INVALID_PLAN;
  if (!planStatus) {
    if (onesided) {
      first = cufftExecR2C(plan, deviceFrames, devicePrimal);
      second = first == CUFFT_SUCCESS
                   ? cufftExecR2C(plan, deviceDframes, deviceTangent)
                   : CUFFT_INVALID_PLAN;
    } else {
      first = cufftExecC2C(plan, deviceFramesComplex, devicePrimal,
                           CUFFT_FORWARD);
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2C(plan, deviceDframesComplex, deviceTangent,
                                  CUFFT_FORWARD)
                   : CUFFT_INVALID_PLAN;
    }
  }
  if (!planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS &&
      outputScale != 1.0f) {
    unsigned blocks = unsigned((outputElements + kThreads - 1) / kThreads);
    scaleComplex<<<blocks, kThreads>>>(devicePrimal, outputElements,
                                       outputScale);
    scaleComplex<<<blocks, kThreads>>>(deviceTangent, outputElements,
                                       outputScale);
  }
  status = cudaGetLastError();
  std::vector<cufftComplex> primal(outputElements), tangent(outputElements);
  if (!planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS &&
      status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(primal.data(), devicePrimal,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(tangent.data(), deviceTangent,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  destroyPlan(plan, workspace);
  release(deviceInput, deviceDinput, deviceWindows, deviceDwindows,
          deviceFrames, deviceDframes, deviceFramesComplex,
          deviceDframesComplex, devicePrimal, deviceTangent);
  if (planStatus || first != CUFFT_SUCCESS || second != CUFFT_SUCCESS ||
      status != cudaSuccess)
    return 365;
  unpackAxis(primal.data(), reinterpret_cast<cufftComplex *>(primalHost), outer,
             int64_t(frames) * bins, inner);
  unpackAxis(tangent.data(), reinterpret_cast<cufftComplex *>(tangentHost), outer,
             int64_t(frames) * bins, inner);
  return 0;
}

extern "C" int tessera_nvidia_istft_policy_broadcast_layout_f32(
    const char *digest, const float *inputHost, const float *windowHost,
    float *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, float outputScale, int center, int outputSamples, int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !outputHost ||
      rank < 2 || rank > 8 || axis <= 0 || axis >= rank || nfft <= 0 ||
      hop <= 0 || outputSamples <= 0 || (center != 0 && center != 1) ||
      (onesided != 0 && onesided != 1))
    return 310;
  int frameAxis = axis - 1;
  int frames = int(shape[frameAxis]);
  int bins = int(shape[axis]);
  if (frames <= 0 || bins != (onesided ? nfft / 2 + 1 : nfft))
    return 311;
  std::vector<cufftComplex> contiguous;
  if (!packHostLayout(reinterpret_cast<const cufftComplex *>(inputHost),
                      contiguous, rank, shape, strides))
    return 311;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0)
      return 311;
    if (dim < frameAxis)
      outer *= shape[dim];
    else if (dim > axis)
      inner *= shape[dim];
    if (dim != frameAxis && dim != axis)
      batchShape.push_back(shape[dim]);
  }
  if (outer <= 0 || inner <= 0 || outer > INT32_MAX / inner)
    return 311;
  int batch = int(outer * inner);
  std::vector<cufftComplex> spectra(contiguous.size());
  packAxis(contiguous.data(), spectra.data(), outer,
           int64_t(frames) * bins, inner);
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 312;
  int64_t rawSamples = int64_t(frames - 1) * hop + nfft;
  int trim = center ? nfft / 2 : 0;
  int64_t available = rawSamples - 2 * trim;
  if (outputSamples > available)
    return 313;

  size_t spectrumElements = spectra.size();
  size_t frameElements = size_t(batch) * frames * nfft;
  size_t outputElements = size_t(batch) * outputSamples;
  cufftComplex *deviceSpectrum = nullptr, *deviceComplexFrames = nullptr;
  float *deviceRealFrames = nullptr, *deviceWindows = nullptr,
        *deviceOutput = nullptr;
  cudaError_t status = cudaMalloc(&deviceSpectrum,
                                  spectrumElements * sizeof(cufftComplex));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceRealFrames, frameElements * sizeof(float));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceComplexFrames,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceOutput, outputElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceSpectrum, spectra.data(),
                        spectrumElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    release(deviceSpectrum, deviceComplexFrames, deviceRealFrames,
            deviceWindows, deviceOutput);
    return 314;
  }
  cufftHandle plan = 0;
  void *workspace = nullptr;
  int planStatus = makePlan(batch * frames, nfft,
                            onesided ? CUFFT_C2R : CUFFT_C2C, plan, workspace);
  cufftResult fftStatus = CUFFT_INVALID_PLAN;
  if (!planStatus)
    fftStatus = onesided
                    ? cufftExecC2R(plan, deviceSpectrum, deviceRealFrames)
                    : cufftExecC2C(plan, deviceSpectrum, deviceComplexFrames,
                                   CUFFT_INVERSE);
  status = cudaGetLastError();
  float inverseScale = outputScale / float(nfft);
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess) {
    unsigned blocks = unsigned((outputElements + kThreads - 1) / kThreads);
    if (onesided)
      overlapAddReal<<<blocks, kThreads>>>(
          deviceRealFrames, deviceWindows, deviceOutput, batch, frames, nfft,
          hop, outputSamples, trim, inverseScale);
    else
      overlapAddComplex<<<blocks, kThreads>>>(
          deviceComplexFrames, deviceWindows, deviceOutput, batch, frames,
          nfft, hop, outputSamples, trim, inverseScale);
    status = cudaGetLastError();
  }
  std::vector<float> output(outputElements);
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaMemcpy(output.data(), deviceOutput,
                        outputElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  destroyPlan(plan, workspace);
  release(deviceSpectrum, deviceComplexFrames, deviceRealFrames, deviceWindows,
          deviceOutput);
  if (planStatus || fftStatus != CUFFT_SUCCESS || status != cudaSuccess)
    return 315;
  unpackAxis(output.data(), outputHost, outer, outputSamples, inner);
  return 0;
}

extern "C" int tessera_nvidia_dct_policy_layout_storage(
    const char *digest, const void *inputHost, void *outputHost, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int dctType,
    int storage, float outputScale) {
  if (!validStorage(storage) || !outputHost)
    return 340;
  if (storage == 0)
    return tessera_nvidia_dct_policy_layout_f32(
        digest, static_cast<const float *>(inputHost),
        static_cast<float *>(outputHost), rank, shape, strides, axis, dctType,
        outputScale);
  std::vector<float> input;
  if (!packStorageLayout(inputHost, input, rank, shape, strides, storage))
    return 341;
  std::vector<float> output(input.size());
  std::vector<int64_t> compact = compactStrides(rank, shape);
  int rc = tessera_nvidia_dct_policy_layout_f32(
      digest, input.data(), output.data(), rank, shape, compact.data(), axis,
      dctType, outputScale);
  if (!rc)
    for (size_t index = 0; index < output.size(); ++index)
      storeStorage(outputHost, int64_t(index), storage, output[index]);
  return rc;
}

extern "C" int tessera_nvidia_stft_policy_broadcast_layout_storage(
    const char *digest, const void *inputHost, const void *windowHost,
    float *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int frames, int storage, float outputScale, int center,
    int padMode, int onesided) {
  if (!validStorage(storage))
    return 342;
  if (storage == 0)
    return tessera_nvidia_stft_policy_broadcast_layout_f32(
        digest, static_cast<const float *>(inputHost),
        static_cast<const float *>(windowHost), outputHost, rank, shape,
        strides, axis, windowRank, windowShape, windowStrides, nfft, hop,
        frames, outputScale, center, padMode, onesided);
  std::vector<float> input, window;
  if (!packStorageLayout(inputHost, input, rank, shape, strides, storage) ||
      !packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 343;
  std::vector<int64_t> inputCompact = compactStrides(rank, shape);
  std::vector<int64_t> windowCompact = compactStrides(windowRank, windowShape);
  return tessera_nvidia_stft_policy_broadcast_layout_f32(
      digest, input.data(), window.data(), outputHost, rank, shape,
      inputCompact.data(), axis, windowRank, windowShape,
      windowCompact.data(), nfft, hop, frames, outputScale, center, padMode,
      onesided);
}

extern "C" int tessera_nvidia_stft_jvp_broadcast_layout_storage(
    const char *digest, const void *inputHost, const void *windowHost,
    const void *dinputHost, const void *dwindowHost, float *primalHost,
    float *tangentHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int frames, int storage, float outputScale, int center,
    int padMode, int onesided) {
  if (!validStorage(storage))
    return 346;
  if (storage == 0)
    return tessera_nvidia_stft_jvp_broadcast_layout_f32(
        digest, static_cast<const float *>(inputHost),
        static_cast<const float *>(windowHost),
        static_cast<const float *>(dinputHost),
        static_cast<const float *>(dwindowHost), primalHost, tangentHost, rank,
        shape, strides, axis, windowRank, windowShape, windowStrides, nfft,
        hop, frames, outputScale, center, padMode, onesided);
  std::vector<float> input, window;
  if (!packStorageLayout(inputHost, input, rank, shape, strides, storage) ||
      !packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 347;
  std::vector<float> dinput(input.size(), 0.0f), dwindow(window.size(), 0.0f);
  if (dinputHost &&
      !packStorageLayout(dinputHost, dinput, rank, shape, strides, storage))
    return 347;
  if (dwindowHost &&
      !packStorageLayout(dwindowHost, dwindow, windowRank, windowShape,
                         windowStrides, storage))
    return 347;
  std::vector<int64_t> inputCompact = compactStrides(rank, shape);
  std::vector<int64_t> windowCompact = compactStrides(windowRank, windowShape);
  return tessera_nvidia_stft_jvp_broadcast_layout_f32(
      digest, input.data(), window.data(), dinput.data(), dwindow.data(),
      primalHost, tangentHost, rank, shape, inputCompact.data(), axis,
      windowRank, windowShape, windowCompact.data(), nfft, hop, frames,
      outputScale, center, padMode, onesided);
}

extern "C" int tessera_nvidia_istft_policy_broadcast_layout_storage(
    const char *digest, const float *inputHost, const void *windowHost,
    void *outputHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int storage, float outputScale, int center, int outputSamples,
    int onesided) {
  if (!validStorage(storage) || !outputHost)
    return 344;
  if (storage == 0)
    return tessera_nvidia_istft_policy_broadcast_layout_f32(
        digest, inputHost, static_cast<const float *>(windowHost),
        static_cast<float *>(outputHost), rank, shape, strides, axis,
        windowRank, windowShape, windowStrides, nfft, hop, outputScale,
        center, outputSamples, onesided);
  std::vector<float> window;
  if (!packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 345;
  std::vector<int64_t> windowCompact = compactStrides(windowRank, windowShape);
  size_t outputElements = 1;
  int frameAxis = axis - 1;
  for (int dim = 0; dim < rank; ++dim)
    if (dim != frameAxis && dim != axis)
      outputElements *= size_t(shape[dim]);
  outputElements *= size_t(outputSamples);
  std::vector<float> output(outputElements);
  int rc = tessera_nvidia_istft_policy_broadcast_layout_f32(
      digest, inputHost, window.data(), output.data(), rank, shape, strides,
      axis, windowRank, windowShape, windowCompact.data(), nfft, hop,
      outputScale, center, outputSamples, onesided);
  if (!rc)
    for (size_t index = 0; index < output.size(); ++index)
      storeStorage(outputHost, int64_t(index), storage, output[index]);
  return rc;
}

extern "C" int tessera_nvidia_istft_jvp_broadcast_layout_f32(
    const char *digest, const float *inputHost, const float *windowHost,
    const float *dinputHost, const float *dwindowHost, float *primalHost,
    float *tangentHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, float outputScale, int center, int outputSamples, int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !primalHost ||
      !tangentHost || rank < 2 || rank > 8 || axis <= 0 || axis >= rank ||
      nfft <= 0 || hop <= 0 || outputSamples <= 0 ||
      (center != 0 && center != 1) || (onesided != 0 && onesided != 1))
    return 350;
  int frameAxis = axis - 1;
  int frames = int(shape[frameAxis]);
  int bins = int(shape[axis]);
  if (frames <= 0 || bins != (onesided ? nfft / 2 + 1 : nfft))
    return 351;
  std::vector<cufftComplex> contiguous;
  if (!packHostLayout(reinterpret_cast<const cufftComplex *>(inputHost),
                      contiguous, rank, shape, strides))
    return 351;
  std::vector<cufftComplex> dcontiguous(contiguous.size(),
                                       make_cuFloatComplex(0.0f, 0.0f));
  if (dinputHost &&
      !packHostLayout(reinterpret_cast<const cufftComplex *>(dinputHost),
                      dcontiguous, rank, shape, strides))
    return 351;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0)
      return 351;
    if (dim < frameAxis)
      outer *= shape[dim];
    else if (dim > axis)
      inner *= shape[dim];
    if (dim != frameAxis && dim != axis)
      batchShape.push_back(shape[dim]);
  }
  if (outer <= 0 || inner <= 0 || outer > INT32_MAX / inner)
    return 351;
  int batch = int(outer * inner);
  std::vector<cufftComplex> spectra(contiguous.size()),
      dspectra(dcontiguous.size());
  packAxis(contiguous.data(), spectra.data(), outer,
           int64_t(frames) * bins, inner);
  packAxis(dcontiguous.data(), dspectra.data(), outer,
           int64_t(frames) * bins, inner);
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 352;
  std::vector<float> dwindows(windows.size(), 0.0f);
  if (dwindowHost &&
      !expandHostWindows(dwindowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, dwindows))
    return 352;
  int64_t rawSamples = int64_t(frames - 1) * hop + nfft;
  int trim = center ? nfft / 2 : 0;
  if (outputSamples > rawSamples - 2 * trim)
    return 353;
  size_t spectrumElements = spectra.size();
  size_t frameElements = size_t(batch) * frames * nfft;
  size_t outputElements = size_t(batch) * outputSamples;
  cufftComplex *deviceSpectrum = nullptr, *deviceDspectrum = nullptr;
  cufftComplex *deviceFramesComplex = nullptr, *deviceDframesComplex = nullptr;
  float *deviceFrames = nullptr, *deviceDframes = nullptr;
  float *deviceWindows = nullptr, *deviceDwindows = nullptr;
  float *devicePrimal = nullptr, *deviceTangent = nullptr;
  cudaError_t status = cudaMalloc(&deviceSpectrum,
                                  spectrumElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDspectrum,
                        spectrumElements * sizeof(cufftComplex));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceFrames, frameElements * sizeof(float));
  if (status == cudaSuccess && onesided)
    status = cudaMalloc(&deviceDframes, frameElements * sizeof(float));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceFramesComplex,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess && !onesided)
    status = cudaMalloc(&deviceDframesComplex,
                        frameElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDwindows, dwindows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&devicePrimal, outputElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceTangent, outputElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceSpectrum, spectra.data(),
                        spectrumElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDspectrum, dspectra.data(),
                        spectrumElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDwindows, dwindows.data(),
                        dwindows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    release(deviceSpectrum, deviceDspectrum, deviceFrames,
            deviceDframes, deviceFramesComplex, deviceDframesComplex,
            deviceWindows, deviceDwindows, devicePrimal, deviceTangent);
    return 354;
  }
  cufftHandle plan = 0;
  void *workspace = nullptr;
  int planStatus = makePlan(batch * frames, nfft,
                            onesided ? CUFFT_C2R : CUFFT_C2C, plan, workspace);
  cufftResult first = CUFFT_INVALID_PLAN, second = CUFFT_INVALID_PLAN;
  if (!planStatus) {
    if (onesided) {
      first = cufftExecC2R(plan, deviceSpectrum, deviceFrames);
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2R(plan, deviceDspectrum, deviceDframes)
                   : CUFFT_INVALID_PLAN;
    } else {
      first = cufftExecC2C(plan, deviceSpectrum, deviceFramesComplex,
                           CUFFT_INVERSE);
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2C(plan, deviceDspectrum,
                                  deviceDframesComplex, CUFFT_INVERSE)
                   : CUFFT_INVALID_PLAN;
    }
  }
  status = cudaGetLastError();
  if (!planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS &&
      status == cudaSuccess) {
    unsigned blocks = unsigned((outputElements + kThreads - 1) / kThreads);
    float inverseScale = outputScale / float(nfft);
    if (onesided)
      overlapAddJVP<<<blocks, kThreads>>>(
          deviceFrames, deviceDframes, deviceWindows, deviceDwindows,
          devicePrimal, deviceTangent, batch, frames, nfft, hop,
          outputSamples, trim, inverseScale);
    else
      overlapAddJVP<<<blocks, kThreads>>>(
          deviceFramesComplex, deviceDframesComplex, deviceWindows,
          deviceDwindows, devicePrimal, deviceTangent, batch, frames, nfft,
          hop, outputSamples, trim, inverseScale);
    status = cudaGetLastError();
  }
  std::vector<float> primal(outputElements), tangent(outputElements);
  if (!planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS &&
      status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(primal.data(), devicePrimal,
                        outputElements * sizeof(float), cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(tangent.data(), deviceTangent,
                        outputElements * sizeof(float), cudaMemcpyDeviceToHost);
  destroyPlan(plan, workspace);
  release(deviceSpectrum, deviceDspectrum, deviceFrames, deviceDframes,
          deviceFramesComplex, deviceDframesComplex, deviceWindows,
          deviceDwindows, devicePrimal, deviceTangent);
  if (planStatus || first != CUFFT_SUCCESS || second != CUFFT_SUCCESS ||
      status != cudaSuccess)
    return 355;
  unpackAxis(primal.data(), primalHost, outer, outputSamples, inner);
  unpackAxis(tangent.data(), tangentHost, outer, outputSamples, inner);
  return 0;
}

extern "C" int tessera_nvidia_istft_jvp_broadcast_layout_storage(
    const char *digest, const float *inputHost, const void *windowHost,
    const float *dinputHost, const void *dwindowHost, void *primalHost,
    void *tangentHost, int rank, const int64_t *shape,
    const int64_t *strides, int axis, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int storage, float outputScale, int center, int outputSamples,
    int onesided) {
  if (!validStorage(storage) || !primalHost || !tangentHost)
    return 356;
  if (storage == 0)
    return tessera_nvidia_istft_jvp_broadcast_layout_f32(
        digest, inputHost, static_cast<const float *>(windowHost), dinputHost,
        static_cast<const float *>(dwindowHost),
        static_cast<float *>(primalHost), static_cast<float *>(tangentHost),
        rank, shape, strides, axis, windowRank, windowShape, windowStrides,
        nfft, hop, outputScale, center, outputSamples, onesided);
  std::vector<float> window, dwindow;
  if (!packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 357;
  if (dwindowHost) {
    if (!packStorageLayout(dwindowHost, dwindow, windowRank, windowShape,
                           windowStrides, storage))
      return 357;
  } else {
    dwindow.assign(window.size(), 0.0f);
  }
  std::vector<int64_t> compact = compactStrides(windowRank, windowShape);
  size_t outputElements = 1;
  int frameAxis = axis - 1;
  for (int dim = 0; dim < rank; ++dim)
    if (dim != frameAxis && dim != axis)
      outputElements *= size_t(shape[dim]);
  outputElements *= size_t(outputSamples);
  std::vector<float> primal(outputElements), tangent(outputElements);
  int rc = tessera_nvidia_istft_jvp_broadcast_layout_f32(
      digest, inputHost, window.data(), dinputHost, dwindow.data(),
      primal.data(), tangent.data(), rank, shape, strides, axis, windowRank,
      windowShape, compact.data(), nfft, hop, outputScale, center,
      outputSamples, onesided);
  if (!rc)
    for (size_t index = 0; index < outputElements; ++index) {
      storeStorage(primalHost, int64_t(index), storage, primal[index]);
      storeStorage(tangentHost, int64_t(index), storage, tangent[index]);
    }
  return rc;
}

extern "C" int tessera_nvidia_stft_backward_broadcast_layout_storage(
    const char *digest, const float *dyHost, const void *inputHost,
    const void *windowHost, void *dxHost, void *dwindowHost, int xRank,
    const int64_t *xShape, const int64_t *xStrides, int axis, int dyRank,
    const int64_t *dyShape, const int64_t *dyStrides, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, int storage, float forwardScale, int center, int padMode,
    int onesided) {
  if (!validStorage(storage) || !dxHost || !dwindowHost)
    return 358;
  if (storage == 0)
    return tessera_nvidia_stft_backward_broadcast_layout_f32(
        digest, dyHost, static_cast<const float *>(inputHost),
        static_cast<const float *>(windowHost), static_cast<float *>(dxHost),
        static_cast<float *>(dwindowHost), xRank, xShape, xStrides, axis,
        dyRank, dyShape, dyStrides, windowRank, windowShape, windowStrides,
        nfft, hop, forwardScale, center, padMode, onesided);
  std::vector<float> input, window;
  if (!packStorageLayout(inputHost, input, xRank, xShape, xStrides, storage) ||
      !packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 359;
  std::vector<int64_t> xCompact = compactStrides(xRank, xShape);
  std::vector<int64_t> windowCompact = compactStrides(windowRank, windowShape);
  std::vector<float> dx(input.size()), dwindow(window.size());
  int rc = tessera_nvidia_stft_backward_broadcast_layout_f32(
      digest, dyHost, input.data(), window.data(), dx.data(), dwindow.data(),
      xRank, xShape, xCompact.data(), axis, dyRank, dyShape, dyStrides,
      windowRank, windowShape, windowCompact.data(), nfft, hop, forwardScale,
      center, padMode, onesided);
  if (!rc) {
    for (size_t index = 0; index < dx.size(); ++index)
      storeStorage(dxHost, int64_t(index), storage, dx[index]);
    for (size_t index = 0; index < dwindow.size(); ++index)
      storeStorage(dwindowHost, int64_t(index), storage, dwindow[index]);
  }
  return rc;
}

extern "C" int tessera_nvidia_istft_backward_broadcast_layout_storage(
    const char *digest, const void *dyHost, const float *spectrumHost,
    const void *windowHost, float *dspectrumHost, void *dwindowHost,
    int dyRank, const int64_t *dyShape, const int64_t *dyStrides,
    int outputAxis, int spectrumRank, const int64_t *spectrumShape,
    const int64_t *spectrumStrides, int frameAxis, int binAxis,
    int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int nfft, int hop, int storage,
    float inverseScale, int center, int onesided) {
  if (!validStorage(storage) || !dspectrumHost || !dwindowHost)
    return 366;
  if (storage == 0)
    return tessera_nvidia_istft_backward_broadcast_layout_f32(
        digest, static_cast<const float *>(dyHost), spectrumHost,
        static_cast<const float *>(windowHost), dspectrumHost,
        static_cast<float *>(dwindowHost), dyRank, dyShape, dyStrides,
        outputAxis, spectrumRank, spectrumShape, spectrumStrides, frameAxis,
        binAxis, windowRank, windowShape, windowStrides, nfft, hop,
        inverseScale, center, onesided);
  std::vector<float> dy, window;
  if (!packStorageLayout(dyHost, dy, dyRank, dyShape, dyStrides, storage) ||
      !packStorageLayout(windowHost, window, windowRank, windowShape,
                         windowStrides, storage))
    return 367;
  std::vector<int64_t> dyCompact = compactStrides(dyRank, dyShape);
  std::vector<int64_t> windowCompact = compactStrides(windowRank, windowShape);
  size_t spectrumElements = 1;
  for (int dim = 0; dim < spectrumRank; ++dim)
    spectrumElements *= size_t(spectrumShape[dim]);
  std::vector<cufftComplex> dspectrum(spectrumElements);
  std::vector<float> dwindow(window.size());
  int rc = tessera_nvidia_istft_backward_broadcast_layout_f32(
      digest, dy.data(), spectrumHost, window.data(),
      reinterpret_cast<float *>(dspectrum.data()), dwindow.data(), dyRank,
      dyShape, dyCompact.data(), outputAxis, spectrumRank, spectrumShape,
      spectrumStrides, frameAxis, binAxis, windowRank, windowShape,
      windowCompact.data(), nfft, hop, inverseScale, center, onesided);
  if (!rc) {
    std::memcpy(dspectrumHost, dspectrum.data(),
                dspectrum.size() * sizeof(cufftComplex));
    for (size_t index = 0; index < dwindow.size(); ++index)
      storeStorage(dwindowHost, int64_t(index), storage, dwindow[index]);
  }
  return rc;
}

extern "C" int tessera_nvidia_streaming_stft_broadcast_layout_f32(
    const char *digest, const float *inputHost, const float *tailHost,
    const float *windowHost, float *outputHost, float *nextTailHost, int rank,
    const int64_t *shape, const int64_t *strides, int axis, int tailSamples,
    int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int nfft, int hop, int frames,
    float outputScale, int onesided) {
  if (!validDigest(digest) || !inputHost || !windowHost || !nextTailHost ||
      (frames > 0 && !outputHost) || (tailSamples > 0 && !tailHost) ||
      rank <= 0 || rank > 8 || axis < 0 || axis >= rank || tailSamples < 0 ||
      tailSamples >= nfft || nfft <= 0 || hop <= 0 || hop > nfft ||
      frames < 0 || (onesided != 0 && onesided != 1))
    return 316;
  std::vector<float> contiguous;
  if (!packHostLayout(inputHost, contiguous, rank, shape, strides) ||
      shape[axis] > INT32_MAX)
    return 317;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(rank, shape, axis, outer, inner, batch, batchShape))
    return 317;
  int chunkSamples = int(shape[axis]);
  int combinedSamples = tailSamples + chunkSamples;
  int expectedFrames = combinedSamples < nfft
                           ? 0
                           : (combinedSamples - nfft) / hop + 1;
  if (frames != expectedFrames)
    return 318;
  std::vector<float> chunk(contiguous.size());
  packAxis(contiguous.data(), chunk.data(), outer, chunkSamples, inner);
  std::vector<float> combined(size_t(batch) * combinedSamples);
  for (int row = 0; row < batch; ++row) {
    for (int at = 0; at < tailSamples; ++at)
      combined[size_t(row) * combinedSamples + at] =
          tailHost[size_t(row) * tailSamples + at];
    for (int at = 0; at < chunkSamples; ++at)
      combined[size_t(row) * combinedSamples + tailSamples + at] =
          chunk[size_t(row) * chunkSamples + at];
  }
  int rc = 0;
  if (frames > 0) {
    std::vector<float> logical(combined.size());
    unpackAxis(combined.data(), logical.data(), outer, combinedSamples, inner);
    std::vector<int64_t> combinedShape(shape, shape + rank);
    combinedShape[axis] = combinedSamples;
    std::vector<int64_t> combinedStrides(rank, 1);
    for (int dim = rank - 2; dim >= 0; --dim)
      combinedStrides[dim] =
          combinedStrides[dim + 1] * combinedShape[dim + 1];
    rc = tessera_nvidia_stft_policy_broadcast_layout_f32(
        digest, logical.data(), windowHost, outputHost, rank,
        combinedShape.data(), combinedStrides.data(), axis, windowRank,
        windowShape, windowStrides, nfft, hop, frames, outputScale, 0, 0,
        onesided);
  }
  if (rc)
    return rc;
  int nextSamples = combinedSamples - frames * hop;
  for (int row = 0; row < batch; ++row)
    for (int at = 0; at < nextSamples; ++at)
      nextTailHost[size_t(row) * nextSamples + at] =
          combined[size_t(row) * combinedSamples + frames * hop + at];
  return 0;
}

extern "C" int tessera_nvidia_stft_backward_broadcast_layout_f32(
    const char *digest, const float *dyHost, const float *inputHost,
    const float *windowHost, float *dxHost, float *dwindowHost, int xRank,
    const int64_t *xShape, const int64_t *xStrides, int axis, int dyRank,
    const int64_t *dyShape, const int64_t *dyStrides, int windowRank,
    const int64_t *windowShape, const int64_t *windowStrides, int nfft,
    int hop, float forwardScale, int center, int padMode, int onesided) {
  if (!validDigest(digest) || !dyHost || !inputHost || !windowHost ||
      !dxHost || !dwindowHost || xRank <= 0 || xRank > 8 ||
      dyRank != xRank + 1 || axis < 0 || axis >= xRank || nfft <= 0 ||
      hop <= 0 || (center != 0 && center != 1) ||
      (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1))
    return 320;
  int samples = int(xShape[axis]);
  int frames = int(dyShape[axis]);
  int bins = int(dyShape[axis + 1]);
  if (bins != (onesided ? nfft / 2 + 1 : nfft))
    return 321;
  int pad = center ? nfft / 2 : 0;
  int framedSamples = std::max(samples + 2 * pad, nfft);
  if (frames != (framedSamples - nfft) / hop + 1 ||
      (center && padMode == 1 && samples <= pad))
    return 321;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(xRank, xShape, axis, outer, inner, batch, batchShape))
    return 322;
  for (int dim = 0; dim < xRank; ++dim) {
    if (dim < axis && dyShape[dim] != xShape[dim])
      return 322;
    if (dim > axis && dyShape[dim + 1] != xShape[dim])
      return 322;
  }
  std::vector<float> inputContiguous;
  std::vector<cufftComplex> dyContiguous;
  if (!packHostLayout(inputHost, inputContiguous, xRank, xShape, xStrides) ||
      !packHostLayout(reinterpret_cast<const cufftComplex *>(dyHost),
                      dyContiguous, dyRank, dyShape, dyStrides))
    return 323;
  size_t inputElements = size_t(batch) * samples;
  size_t spectralElements = size_t(batch) * frames * bins;
  if (inputContiguous.size() != inputElements ||
      dyContiguous.size() != spectralElements)
    return 323;
  std::vector<float> input(inputElements);
  std::vector<cufftComplex> dy(spectralElements);
  packAxis(inputContiguous.data(), input.data(), outer, samples, inner);
  packAxis(dyContiguous.data(), dy.data(), outer,
           int64_t(frames) * bins, inner);
  std::vector<float> windows;
  std::vector<int> rowWindow;
  int windowRows = 0;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows) ||
      !buildWindowRowMap(windowRank, windowShape, batchShape, rowWindow,
                         windowRows))
    return 324;
  int win = int(windowShape[windowRank - 1]);
  size_t windowElements = size_t(windowRows) * win;
  cufftComplex *deviceDy = nullptr;
  float *deviceInput = nullptr, *deviceWindows = nullptr, *deviceDx = nullptr,
        *deviceDwindow = nullptr;
  int *deviceRowWindow = nullptr;
  cudaError_t status = cudaMalloc(&deviceDy,
                                  spectralElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceInput, inputElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDx, inputElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDwindow, windowElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceRowWindow, rowWindow.size() * sizeof(int));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDy, dy.data(),
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceInput, input.data(),
                        inputElements * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceRowWindow, rowWindow.data(),
                        rowWindow.size() * sizeof(int),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    stftBackwardInputPolicy<<<
        unsigned((inputElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceDy, deviceWindows, deviceDx, batch, samples, nfft, hop, frames,
        bins, forwardScale, center, padMode);
    stftBackwardWindowPolicy<<<
        unsigned((windowElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceDy, deviceInput, deviceRowWindow, deviceDwindow, batch,
        windowRows, samples, nfft, win, hop, frames, bins, forwardScale,
        center, padMode);
    status = cudaGetLastError();
  }
  std::vector<float> dx(inputElements), dwindow(windowElements);
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(dx.data(), deviceDx, inputElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(dwindow.data(), deviceDwindow,
                        windowElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  release(deviceDy, deviceInput, deviceWindows, deviceDx, deviceDwindow,
          deviceRowWindow);
  if (status != cudaSuccess)
    return 325;
  unpackAxis(dx.data(), dxHost, outer, samples, inner);
  std::copy(dwindow.begin(), dwindow.end(), dwindowHost);
  return 0;
}

extern "C" int tessera_nvidia_istft_backward_broadcast_layout_f32(
    const char *digest, const float *dyHost, const float *spectrumHost,
    const float *windowHost, float *dspectrumHost, float *dwindowHost,
    int dyRank, const int64_t *dyShape, const int64_t *dyStrides,
    int outputAxis, int spectrumRank, const int64_t *spectrumShape,
    const int64_t *spectrumStrides, int frameAxis, int binAxis,
    int windowRank, const int64_t *windowShape,
    const int64_t *windowStrides, int nfft, int hop, float inverseScale,
    int center, int onesided) {
  if (!validDigest(digest) || !dyHost || !spectrumHost || !windowHost ||
      !dspectrumHost || !dwindowHost || spectrumRank != dyRank + 1 ||
      outputAxis < 0 || outputAxis >= dyRank || frameAxis < 0 ||
      binAxis != frameAxis + 1 || binAxis >= spectrumRank || nfft <= 0 ||
      hop <= 0 || (center != 0 && center != 1) ||
      (onesided != 0 && onesided != 1))
    return 330;
  int frames = int(spectrumShape[frameAxis]);
  int bins = int(spectrumShape[binAxis]);
  if (frames <= 0 || bins != (onesided ? nfft / 2 + 1 : nfft))
    return 331;
  int rawSamples = (frames - 1) * hop + nfft;
  int outputSamples = int(dyShape[outputAxis]);
  if (outputSamples <= 0 ||
      outputSamples > rawSamples - (center ? nfft : 0))
    return 331;
  int64_t outer = 1, inner = 1;
  std::vector<int64_t> batchShape;
  for (int dim = 0; dim < dyRank; ++dim) {
    if (dyShape[dim] <= 0)
      return 332;
    if (dim < outputAxis)
      outer *= dyShape[dim];
    else if (dim > outputAxis)
      inner *= dyShape[dim];
    if (dim != outputAxis)
      batchShape.push_back(dyShape[dim]);
  }
  if (outer <= 0 || inner <= 0 || outer > INT32_MAX / inner)
    return 332;
  int batch = int(outer * inner);
  for (int dim = 0; dim < dyRank; ++dim) {
    if (dim < outputAxis && spectrumShape[dim] != dyShape[dim])
      return 332;
    if (dim > outputAxis && spectrumShape[dim + 1] != dyShape[dim])
      return 332;
  }
  std::vector<float> dyContiguous;
  std::vector<cufftComplex> spectrumContiguous;
  if (!packHostLayout(dyHost, dyContiguous, dyRank, dyShape, dyStrides) ||
      !packHostLayout(reinterpret_cast<const cufftComplex *>(spectrumHost),
                      spectrumContiguous, spectrumRank, spectrumShape,
                      spectrumStrides))
    return 333;
  size_t dyElements = size_t(batch) * outputSamples;
  size_t spectralElements = size_t(batch) * frames * bins;
  if (dyContiguous.size() != dyElements ||
      spectrumContiguous.size() != spectralElements)
    return 333;
  std::vector<float> dy(dyElements);
  std::vector<cufftComplex> spectrum(spectralElements);
  packAxis(dyContiguous.data(), dy.data(), outer, outputSamples, inner);
  packAxis(spectrumContiguous.data(), spectrum.data(), outer,
           int64_t(frames) * bins, inner);
  std::vector<float> windows;
  std::vector<int> rowWindow;
  int windowRows = 0;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows) ||
      !buildWindowRowMap(windowRank, windowShape, batchShape, rowWindow,
                         windowRows))
    return 334;
  int win = int(windowShape[windowRank - 1]);
  size_t frameElements = size_t(batch) * frames * nfft;
  size_t windowElements = size_t(windowRows) * win;
  float *deviceDy = nullptr, *deviceWindows = nullptr, *deviceFrames = nullptr,
        *deviceDframes = nullptr, *deviceDwindow = nullptr;
  cufftComplex *deviceSpectrum = nullptr, *deviceDspectrum = nullptr;
  int *deviceRowWindow = nullptr;
  cudaError_t status = cudaMalloc(&deviceDy, dyElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceSpectrum,
                        spectralElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceWindows, windows.size() * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceFrames, frameElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDframes, frameElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDspectrum,
                        spectralElements * sizeof(cufftComplex));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceDwindow, windowElements * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&deviceRowWindow, rowWindow.size() * sizeof(int));
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDy, dy.data(), dyElements * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceSpectrum, spectrum.data(),
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceRowWindow, rowWindow.data(),
                        rowWindow.size() * sizeof(int),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    istftFrameValuesPolicy<<<
        unsigned((frameElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceSpectrum, deviceFrames, batch, frames, bins, nfft, inverseScale,
        onesided);
    istftBackwardFramesPolicy<<<
        unsigned((frameElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceDy, deviceFrames, deviceWindows, deviceDframes, batch, frames,
        nfft, hop, outputSamples, center);
    istftBackwardSpectrumPolicy<<<
        unsigned((spectralElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceDframes, deviceDspectrum, batch, frames, bins, nfft,
        inverseScale, onesided);
    istftBackwardWindowPolicy<<<
        unsigned((windowElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceDy, deviceFrames, deviceWindows, deviceRowWindow, deviceDwindow,
        batch, windowRows, frames, nfft, win, hop, outputSamples, center);
    status = cudaGetLastError();
  }
  std::vector<cufftComplex> dspectrum(spectralElements);
  std::vector<float> dwindow(windowElements);
  if (status == cudaSuccess)
    status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(dspectrum.data(), deviceDspectrum,
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(dwindow.data(), deviceDwindow,
                        windowElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  release(deviceDy, deviceSpectrum, deviceWindows, deviceFrames,
          deviceDframes, deviceDspectrum, deviceDwindow, deviceRowWindow);
  if (status != cudaSuccess)
    return 335;
  unpackAxis(dspectrum.data(),
             reinterpret_cast<cufftComplex *>(dspectrumHost), outer,
             int64_t(frames) * bins, inner);
  std::copy(dwindow.begin(), dwindow.end(), dwindowHost);
  return 0;
}
