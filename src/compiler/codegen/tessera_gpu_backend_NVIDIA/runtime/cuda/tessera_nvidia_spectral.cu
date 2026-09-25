#include "tessera_nvidia_fft.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
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
  // Odometer walk: one strided row copy per innermost run, carrying the outer
  // coordinates incrementally. A per-element div/mod over every dimension cost
  // ~6 ms for a 250k-element strided spectrum in the unoptimized host build.
  const int last = rank - 1;
  const int64_t rowExtent = shape[last], rowStride = strides[last];
  int64_t index[8] = {0};
  int64_t offset = 0;
  for (size_t logical = 0; logical < elements; logical += size_t(rowExtent)) {
    const T *row = input + offset;
    T *destination = output.data() + logical;
    for (int64_t i = 0; i < rowExtent; ++i)
      destination[i] = row[i * rowStride];
    for (int dim = last - 1; dim >= 0; --dim) {
      offset += strides[dim];
      if (++index[dim] < shape[dim])
        break;
      offset -= strides[dim] * shape[dim];
      index[dim] = 0;
    }
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

// Whether `strides` is the compact row-major layout of `shape` (extent-1
// dimensions may carry any stride). Such an input needs no host repack.
// A shape/stride descriptor a caller may index: both pointers present, rank in
// [1, 8], every extent positive. Wrappers that size buffers from `shape` before
// the f32 entry point validates it call this first.
bool validDescriptor(int rank, const int64_t *shape, const int64_t *strides) {
  if (!shape || !strides || rank <= 0 || rank > 8)
    return false;
  for (int dim = 0; dim < rank; ++dim)
    if (shape[dim] <= 0)
      return false;
  return true;
}

// False for an invalid descriptor, so callers fall through to
// packHostLayout, which reports it as the entry point's layout error.
bool isCompactLayout(int rank, const int64_t *shape, const int64_t *strides) {
  if (!shape || !strides || rank <= 0 || rank > 8)
    return false;
  int64_t expected = 1;
  for (int dim = rank - 1; dim >= 0; --dim) {
    if (shape[dim] != 1 && strides[dim] != expected)
      return false;
    expected *= shape[dim];
  }
  return true;
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

// Axis-packed ([outer][inner][axis]) view of a strided host tensor holding
// exactly `expected` elements; nullptr when the layout is invalid or the count
// disagrees. It is the caller's own buffer when the layout is compact and the
// axis innermost, so the common case stages nothing on the host. Measured on
// the RTX 5070: the unconditional pack-then-fold cost more host time than the
// whole device side of the STFT/ISTFT JVP and backward calls.
template <typename T>
const T *stageAxis(const T *host, int rank, const int64_t *shape,
                   const int64_t *strides, int64_t outer, int64_t axisExtent,
                   int64_t inner, size_t expected, std::vector<T> &contiguous,
                   std::vector<T> &packed) {
  if (!host || !shape || !strides || rank <= 0 || rank > 8)
    return nullptr;
  size_t elements = 1;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape[dim] <= 0 ||
        size_t(shape[dim]) > std::numeric_limits<size_t>::max() / elements)
      return nullptr;
    elements *= size_t(shape[dim]);
  }
  if (elements != expected)
    return nullptr;
  const T *staged = host;
  if (!isCompactLayout(rank, shape, strides)) {
    if (!packHostLayout(host, contiguous, rank, shape, strides))
      return nullptr;
    staged = contiguous.data();
  }
  if (inner == 1)
    return staged;
  packed.resize(expected);
  packAxis(staged, packed.data(), outer, axisExtent, inner);
  return packed.data();
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

// pad_mode is a centered-framing policy (tessera.ops.stft, vjp._VJPS["stft"]):
// only a centered frame reflects at the signal edge. A non-centered frame that
// runs past the signal is zero-filled, so every reflect site below is gated on
// `center && padMode == 1`; gating on padMode alone reflected those frames.
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
  if ((source < 0 || source >= samples) && center && padMode == 1)
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
  if ((source < 0 || source >= samples) && center && padMode == 1)
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
  if ((source < 0 || source >= samples) && center && padMode == 1)
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
  if ((source < 0 || source >= samples) && center && padMode == 1)
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

// ---------------------------------------------------------------------------
// FFT-based STFT/ISTFT reverse mode. The former direct kernels (removed; the
// ROCm composite still carries the same ones) evaluated each DFT as an O(N)
// sum per output element in double precision; on the RTX 5070
// an 8x16000 (nfft 512, hop 128) STFT VJP took 1.65 s. These compute the same
// quantities with cuFFT (validated against transcriptions of the direct
// kernels to ~1e-14 in float64 for even/odd N, one-sided/full spectra and
// constant/reflect padding) and keep the direct kernels' conventions:
//   STFT backward: G[frame, local] = Re sum_bins dy_b e^{+i 2 pi b local / N},
//     every stored bin weighted once -- a C2R of dy with interior bins halved
//     (C2R counts them twice), or the real part of an unnormalized inverse C2C
//     for a full spectrum.
//   ISTFT backward: frame values = C2R(spectrum) * inverseScale (C2R's doubling
//     is the direct kernel's weight 2); dspectrum = RFFT(dframes) * weight *
//     inverseScale with that same weight.
__device__ bool interiorBin(int bin, int bins, int nfft) {
  return bin > 0 && !(nfft % 2 == 0 && bin == bins - 1);
}

// Scale each one-sided bin by (interior ? interiorWeight : 1) * scale.
__global__ void weightOnesidedBins(cufftComplex *values, size_t count,
                                   int bins, int nfft, float interiorWeight,
                                   float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  int bin = int(index % bins);
  float w = (interiorBin(bin, bins, nfft) ? interiorWeight : 1.0f) * scale;
  values[index].x *= w;
  values[index].y *= w;
}

__global__ void realPartScaled(const cufftComplex *values, float *output,
                               size_t count, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = values[index].x * scale;
}

__global__ void scaleReal(float *values, size_t count, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] *= scale;
}

__global__ void realToComplex(const float *values, cufftComplex *output,
                              size_t count) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = make_cuComplex(values[index], 0.0f);
}

// dx[row, s] = scale * sum over (frame, local) whose source maps to s of
// G[frame, local] * window[local]. A deterministic gather: a sample is reached
// directly and, under reflect padding, from at most two mirrored positions.
// One bounce suffices because reflect applies only to centered frames and a
// centered reflect requires samples > pad = nfft / 2.
__global__ void stftBackwardInputFromG(const float *g, const float *windows,
                                       float *dx, int batch, int samples,
                                       int nfft, int hop, int frames,
                                       float scale, int center, int padMode) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * samples)
    return;
  int s = int(index % samples);
  int row = int(index / samples);
  int pad = center ? nfft / 2 : 0;
  int candidates[3] = {s, -s, 2 * samples - 2 - s};
  int count = center && padMode == 1 ? 3 : 1;
  double result = 0.0;
  for (int c = 0; c < count; ++c) {
    int t = candidates[c];
    if (c > 0 && (t == s || (t >= 0 && t < samples) ||
                  reflectIndex(t, samples) != s))
      continue;
    if (c == 2 && t == candidates[1])
      continue;
    int reach = t + pad;
    int first = reach - (nfft - 1) <= 0 ? 0 : (reach - (nfft - 1) + hop - 1) / hop;
    int last = reach < 0 ? -1 : min(frames - 1, reach / hop);
    for (int frame = first; frame <= last; ++frame) {
      int local = reach - frame * hop;
      if (local < 0 || local >= nfft)
        continue;
      result += double(g[(size_t(row) * frames + frame) * nfft + local]) *
                double(windows[size_t(row) * nfft + local]);
    }
  }
  dx[index] = float(result * double(scale));
}

// Window-gradient reductions run one block per window element: each thread
// strides over the flattened (row, frame) pairs and the block folds the
// partials in a fixed tree order, so the result is deterministic run to run.
// One thread per element launched only windowRows * win threads (512 for a
// rank-one n_fft=512 window) and took 0.41 ms (STFT) / 1.59 ms (ISTFT) of
// serial fp64 on the RTX 5070.
constexpr int kReduceThreads = 256;

__device__ double blockSum(double value) {
  __shared__ double partial[kReduceThreads];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int width = kReduceThreads / 2; width > 0; width /= 2) {
    if (int(threadIdx.x) < width)
      partial[threadIdx.x] += partial[threadIdx.x + width];
    __syncthreads();
  }
  return partial[0];
}

// dwindow: the former direct kernel's reduction, with the per-frame DFT read
// from G instead of recomputed per element. Launch: windowRows * win blocks of
// kReduceThreads.
__global__ void stftBackwardWindowFromG(
    const float *g, const float *input, const int *rowWindow, float *dwindow,
    int batch, int windowRows, int samples, int nfft, int win, int hop,
    int frames, float scale, int center, int padMode) {
  size_t index = blockIdx.x;
  if (index >= size_t(windowRows) * win)
    return;
  int localWindow = int(index % win);
  int windowRow = int(index / win);
  int local = (nfft - win) / 2 + localWindow;
  int pad = center ? nfft / 2 : 0;
  double result = 0.0;
  size_t pairs = size_t(batch) * frames;
  for (size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    int row = int(pair / frames);
    int frame = int(pair % frames);
    if (rowWindow[row] != windowRow)
      continue;
    int source = frame * hop + local - pad;
    bool present = source >= 0 && source < samples;
    if (!present && center && padMode == 1) {
      source = reflectIndex(source, samples);
      present = true;
    }
    if (!present)
      continue;
    result += double(g[(size_t(row) * frames + frame) * nfft + local]) *
              double(input[size_t(row) * samples + source]);
  }
  result = blockSum(result);
  if (threadIdx.x == 0)
    dwindow[index] = float(result * double(scale));
}

// Overlap-add numerator (sum frame*window) and denominator (sum window^2) per
// raw sample, gathered once instead of per consumer element.
__global__ void istftOverlapTerms(const float *frameValues,
                                  const float *windows, double *numerator,
                                  double *denominator, int batch, int frames,
                                  int nfft, int hop, int rawSamples) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * rawSamples)
    return;
  int raw = int(index % rawSamples);
  int row = int(index / rawSamples);
  int first = raw - (nfft - 1) <= 0 ? 0 : (raw - (nfft - 1) + hop - 1) / hop;
  int last = min(frames - 1, raw / hop);
  double num = 0.0, den = 0.0;
  for (int frame = first; frame <= last; ++frame) {
    int local = raw - frame * hop;
    if (local < 0 || local >= nfft)
      continue;
    double window = windows[size_t(row) * nfft + local];
    num += double(frameValues[(size_t(row) * frames + frame) * nfft + local]) *
           window;
    den += window * window;
  }
  numerator[index] = num;
  denominator[index] = den;
}

__global__ void istftBackwardFramesFromTerms(
    const float *dy, const double *denominator, const float *windows,
    float *dframes, int batch, int frames, int nfft, int hop,
    int outputSamples, int rawSamples, int center) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * frames * nfft)
    return;
  int local = int(index % nfft);
  size_t rowFrame = index / nfft;
  int frame = int(rowFrame % frames);
  int row = int(rowFrame / frames);
  int trim = center ? nfft / 2 : 0;
  int output = frame * hop + local - trim;
  if (output < 0 || output >= outputSamples) {
    dframes[index] = 0.0f;
    return;
  }
  double den = denominator[size_t(row) * rawSamples + output + trim];
  double safe = den > 1.0e-12 ? den : 1.0e-12;
  dframes[index] = float(double(dy[size_t(row) * outputSamples + output]) /
                         safe * double(windows[size_t(row) * nfft + local]));
}

__global__ void istftBackwardWindowFromTerms(
    const float *dy, const float *frameValues, const float *windows,
    const double *numerator, const double *denominator, const int *rowWindow,
    float *dwindow, int batch, int windowRows, int frames, int nfft, int win,
    int hop, int outputSamples, int rawSamples, int center) {
  // Launch: windowRows * win blocks of kReduceThreads (see blockSum).
  size_t index = blockIdx.x;
  if (index >= size_t(windowRows) * win)
    return;
  int localWindow = int(index % win);
  int windowRow = int(index / win);
  int local = (nfft - win) / 2 + localWindow;
  int trim = center ? nfft / 2 : 0;
  double result = 0.0;
  size_t pairs = size_t(batch) * frames;
  for (size_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    int row = int(pair / frames);
    int frame = int(pair % frames);
    if (rowWindow[row] != windowRow)
      continue;
    int output = frame * hop + local - trim;
    if (output < 0 || output >= outputSamples)
      continue;
    size_t term = size_t(row) * rawSamples + output + trim;
    double num = numerator[term], den = denominator[term];
    double safe = den > 1.0e-12 ? den : 1.0e-12;
    double upstream = dy[size_t(row) * outputSamples + output];
    double draw = upstream / safe;
    double dweight = den > 1.0e-12 ? -upstream * num / (safe * safe) : 0.0;
    double window = windows[size_t(row) * nfft + local];
    double frameValue =
        frameValues[(size_t(row) * frames + frame) * nfft + local];
    result += draw * frameValue + 2.0 * dweight * window;
  }
  result = blockSum(result);
  if (threadIdx.x == 0)
    dwindow[index] = float(result);
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

// ---------------------------------------------------------------------------
// Per-device reuse of cuFFT plans and device staging buffers.
//
// Measured on the RTX 5070 (benchmarks/baselines/nvidia_spectral_20260925):
// a warm STFT spent ~2.7 ms per call in five cudaMalloc/cudaFree pairs plus a
// fresh cuFFT plan (module load + memory query) around ~15 us of kernels. Every
// entry point here is synchronous (it synchronizes before returning), so one
// library-wide lock held for the call makes buffer reuse across calls safe.
// Plans and buffers are keyed by the CUDA device current at use: a plan belongs
// to its creating context (see tessera_nvidia_fft.cu).
std::mutex &spectralMutex() {
  static std::mutex mutex;
  return mutex;
}

struct CachedPlan {
  int device;
  int type;
  int nfft;
  int batch;
  cufftHandle plan;
  void *workspace;
};

constexpr size_t kPlanCacheLimit = 16;

std::list<CachedPlan> &planCache() {
  static std::list<CachedPlan> cache;
  return cache;
}

// Caller holds spectralMutex(). 0 and a reusable plan, or makePlan's status
// (5 when the current device cannot be read).
int cachedPlan(int batch, int nfft, cufftType type, cufftHandle &plan) {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess)
    return 5;
  auto &cache = planCache();
  for (auto it = cache.begin(); it != cache.end(); ++it)
    if (it->device == device && it->type == int(type) && it->nfft == nfft &&
        it->batch == batch) {
      cache.splice(cache.begin(), cache, it);
      plan = cache.front().plan;
      return 0;
    }
  cufftHandle fresh = 0;
  void *workspace = nullptr;
  if (int status = makePlan(batch, nfft, type, fresh, workspace)) {
    destroyPlan(fresh, workspace);
    return status;
  }
  if (cache.size() >= kPlanCacheLimit) {
    destroyPlan(cache.back().plan, cache.back().workspace);
    cache.pop_back();
  }
  cache.push_front({device, int(type), nfft, batch, fresh, workspace});
  plan = fresh;
  return 0;
}

struct ScratchBuffer {
  int device;
  int slot;
  void *pointer;
  size_t bytes;
};

std::vector<ScratchBuffer> &scratchPool() {
  static std::vector<ScratchBuffer> pool;
  return pool;
}

// Caller holds spectralMutex(). A device buffer of at least `bytes` for
// (current device, slot), contents undefined; nullptr on CUDA failure. Grows
// geometrically so a slowly growing workload does not reallocate every call.
void *scratch(int slot, size_t bytes) {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess)
    return nullptr;
  bytes = std::max<size_t>(bytes, 1);
  for (auto &buffer : scratchPool()) {
    if (buffer.device != device || buffer.slot != slot)
      continue;
    if (buffer.bytes >= bytes)
      return buffer.pointer;
    size_t grown = std::max(bytes, buffer.bytes * 2);
    cudaFree(buffer.pointer);
    buffer.pointer = nullptr;
    buffer.bytes = 0;
    if (cudaMalloc(&buffer.pointer, grown) != cudaSuccess)
      return nullptr;
    buffer.bytes = grown;
    return buffer.pointer;
  }
  void *pointer = nullptr;
  if (cudaMalloc(&pointer, bytes) != cudaSuccess)
    return nullptr;
  scratchPool().push_back({device, slot, pointer, bytes});
  return pointer;
}

// FFT-based DCT-II / DCT-III (Makhoul), unnormalized like scipy.fft.dct:
//   DCT-II : v = even samples then reversed odd samples; X[k] =
//            2 Re(FFT(v)[k] e^{-i pi k / 2N}).
//   DCT-III: Z[k] = (x[k] - i x[N-k]) e^{i pi k / 2N}, x[N] = 0;
//            z = unnormalized inverse FFT(Z); y[2m] = z[m], y[2m+1] = z[N-1-m].
// Replaces an O(N^2) double-precision direct sum (11.5 ms per 64x1024 call,
// ~99% of the op, on the RTX 5070, whose fp64 rate is 1/64 of fp32).
__device__ int makhoulIndex(int m, int n) {
  int half = (n + 1) / 2;
  return m < half ? 2 * m : 2 * (n - 1 - m) + 1;
}

__global__ void dct2Reorder(const float *input, cufftComplex *values,
                            int batch, int n) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * n)
    return;
  size_t row = index / n;
  int m = int(index % n);
  values[index] = make_cuComplex(input[row * n + makhoulIndex(m, n)], 0.0f);
}

__global__ void dct2Twiddle(const cufftComplex *values, float *output,
                            int batch, int n, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * n)
    return;
  int k = int(index % n);
  float sine, cosine;
  sincospif(float(k) / float(2 * n), &sine, &cosine);
  output[index] =
      2.0f * (values[index].x * cosine + values[index].y * sine) * scale;
}

__global__ void dct3Prepare(const float *input, cufftComplex *values,
                            int batch, int n) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * n)
    return;
  size_t row = index / n;
  int k = int(index % n);
  float xk = input[row * n + k];
  float xnk = k == 0 ? 0.0f : input[row * n + (n - k)];
  float sine, cosine;
  sincospif(float(k) / float(2 * n), &sine, &cosine);
  values[index] = make_cuComplex(xk * cosine + xnk * sine,
                                 xk * sine - xnk * cosine);
}

// X[r, k] *= W[kernelRow(r), k] * scale; one kernel row broadcasts to all.
__global__ void multiplySpectra(cufftComplex *signal,
                                const cufftComplex *kernel, int rows,
                                int kernelRows, int bins, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(rows) * bins)
    return;
  size_t row = index / bins;
  size_t k = index % bins;
  cufftComplex w = kernel[(kernelRows == 1 ? 0 : row) * bins + k];
  cufftComplex x = signal[index];
  signal[index] = make_cuComplex((x.x * w.x - x.y * w.y) * scale,
                                 (x.x * w.y + x.y * w.x) * scale);
}

__global__ void dct3Scatter(const cufftComplex *values, float *output,
                            int batch, int n, float scale) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(batch) * n)
    return;
  size_t row = index / n;
  int m = int(index % n);
  output[row * n + makhoulIndex(m, n)] = values[index].x * scale;
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
  if (!validDigest(digest) || !inputHost || !outputHost || !shape ||
      !strides || rank <= 0 || rank > 8 || axis < 0 || axis >= rank ||
      dctType < 1 || dctType > 4 ||
      (dctType == 1 && shape[axis] < 2))
    return 290;
  if (shape[axis] > INT32_MAX)
    return 291;
  int64_t outer = 0, inner = 0;
  int batch = 0;
  std::vector<int64_t> batchShape;
  if (!foldedBatch(rank, shape, axis, outer, inner, batch, batchShape))
    return 291;
  int length = int(shape[axis]);
  const size_t elements = size_t(batch) * size_t(length);
  // Host staging only where the layout needs it: a compact input is read in
  // place, and with the transform axis innermost (inner == 1) the axis fold
  // and unfold are identities, so the device reads and writes the caller's
  // buffers directly (measured ~1.4 ms of host passes per 64x1024 call).
  std::vector<float> contiguous, packed, output;
  const float *staged = inputHost;
  if (!isCompactLayout(rank, shape, strides)) {
    if (!packHostLayout(inputHost, contiguous, rank, shape, strides))
      return 291;
    staged = contiguous.data();
  }
  if (inner != 1) {
    packed.resize(elements);
    packAxis(staged, packed.data(), outer, length, inner);
    staged = packed.data();
    output.resize(elements);
  }
  float *result = inner != 1 ? output.data() : outputHost;
  std::lock_guard<std::mutex> lock(spectralMutex());
  auto *deviceInput = static_cast<float *>(scratch(0, elements * sizeof(float)));
  auto *deviceOutput = static_cast<float *>(scratch(1, elements * sizeof(float)));
  if (!deviceInput || !deviceOutput)
    return 292;
  cudaError_t status = cudaMemcpy(deviceInput, staged,
                                  elements * sizeof(float),
                                  cudaMemcpyHostToDevice);
  const unsigned blocks = unsigned((elements + kThreads - 1) / kThreads);
  if (status == cudaSuccess && (dctType == 2 || dctType == 3)) {
    auto *values = static_cast<cufftComplex *>(
        scratch(2, elements * sizeof(cufftComplex)));
    cufftHandle plan = 0;
    if (!values || cachedPlan(batch, length, CUFFT_C2C, plan))
      return 292;
    if (dctType == 2)
      dct2Reorder<<<blocks, kThreads>>>(deviceInput, values, batch, length);
    else
      dct3Prepare<<<blocks, kThreads>>>(deviceInput, values, batch, length);
    status = cudaGetLastError();
    if (status == cudaSuccess &&
        cufftExecC2C(plan, values, values,
                     dctType == 2 ? CUFFT_FORWARD : CUFFT_INVERSE) !=
            CUFFT_SUCCESS)
      return 292;
    if (status == cudaSuccess) {
      if (dctType == 2)
        dct2Twiddle<<<blocks, kThreads>>>(values, deviceOutput, batch, length,
                                          outputScale);
      else
        dct3Scatter<<<blocks, kThreads>>>(values, deviceOutput, batch, length,
                                          outputScale);
      status = cudaGetLastError();
    }
  } else if (status == cudaSuccess) {
    dctDirectPolicy<<<blocks, kThreads>>>(deviceInput, deviceOutput, batch,
                                          length, dctType, outputScale);
    status = cudaGetLastError();
  }
  // The synchronous copy back orders after the kernels and reports their
  // errors, so no separate device synchronize is needed.
  if (status == cudaSuccess)
    status = cudaMemcpy(result, deviceOutput, elements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    return 292;
  if (inner != 1)
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
      !shape || !strides || rank <= 0 || rank > 8 || axis < 0 ||
      axis >= rank || nfft <= 0 || hop <= 0 || frames <= 0 || (center != 0 && center != 1) ||
      (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1))
    return 300;
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
  const size_t inputElements = size_t(batch) * size_t(samples);
  // Host staging only where the layout needs it (see the DCT entry point).
  std::vector<float> contiguous, packed;
  const float *staged = inputHost;
  if (!isCompactLayout(rank, shape, strides)) {
    if (!packHostLayout(inputHost, contiguous, rank, shape, strides))
      return 301;
    staged = contiguous.data();
  }
  if (inner != 1) {
    packed.resize(inputElements);
    packAxis(staged, packed.data(), outer, samples, inner);
    staged = packed.data();
  }
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 303;

  int bins = onesided ? nfft / 2 + 1 : nfft;
  size_t frameElements = 0, outputElements = 0;
  if (!checkedProduct(size_t(batch) * frames, size_t(nfft), frameElements) ||
      !checkedProduct(size_t(batch) * frames, size_t(bins), outputElements))
    return 304;
  std::lock_guard<std::mutex> lock(spectralMutex());
  auto *deviceInput =
      static_cast<float *>(scratch(0, inputElements * sizeof(float)));
  auto *deviceWindows =
      static_cast<float *>(scratch(1, windows.size() * sizeof(float)));
  float *deviceRealFrames = nullptr;
  cufftComplex *deviceComplexFrames = nullptr;
  if (onesided)
    deviceRealFrames =
        static_cast<float *>(scratch(2, frameElements * sizeof(float)));
  else
    deviceComplexFrames = static_cast<cufftComplex *>(
        scratch(2, frameElements * sizeof(cufftComplex)));
  auto *deviceOutput = static_cast<cufftComplex *>(
      scratch(3, outputElements * sizeof(cufftComplex)));
  if (!deviceInput || !deviceWindows || !deviceOutput ||
      !(onesided ? static_cast<void *>(deviceRealFrames)
                 : static_cast<void *>(deviceComplexFrames)))
    return 305;
  cudaError_t status = cudaMemcpy(deviceInput, staged,
                                  inputElements * sizeof(float),
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return 305;
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
  int planStatus = status == cudaSuccess
                       ? cachedPlan(batch * frames, nfft,
                                    onesided ? CUFFT_R2C : CUFFT_C2C, plan)
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
  // With the sample axis innermost the unfold is an identity: copy straight
  // into the caller's buffer. The synchronous copy orders after the kernels.
  std::vector<cufftComplex> output(inner != 1 ? outputElements : 0);
  auto *result = inner != 1 ? output.data()
                            : reinterpret_cast<cufftComplex *>(outputHost);
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaMemcpy(result, deviceOutput,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (planStatus || fftStatus != CUFFT_SUCCESS || status != cudaSuccess)
    return 306;
  if (inner != 1)
    unpackAxis(output.data(), reinterpret_cast<cufftComplex *>(outputHost),
               outer, int64_t(frames) * bins, inner);
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
      !tangentHost || !shape || !strides || rank <= 0 || rank > 8 ||
      axis < 0 || axis >= rank || nfft <= 0 || hop <= 0 || frames <= 0 ||
      (center != 0 && center != 1) || (padMode != 0 && padMode != 1) ||
      (onesided != 0 && onesided != 1))
    return 360;
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
  const size_t inputElements = size_t(batch) * size_t(samples);
  std::vector<float> contiguous, packed, dcontiguous, dpacked, zeros;
  const float *input = stageAxis(inputHost, rank, shape, strides, outer,
                                 samples, inner, inputElements, contiguous,
                                 packed);
  const float *dinput =
      dinputHost ? stageAxis(dinputHost, rank, shape, strides, outer, samples,
                             inner, inputElements, dcontiguous, dpacked)
                 : (zeros.assign(inputElements, 0.0f), zeros.data());
  if (!input || !dinput)
    return 361;
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
  size_t frameElements = 0, outputElements = 0;
  if (!checkedProduct(size_t(batch) * frames, size_t(nfft), frameElements) ||
      !checkedProduct(size_t(batch) * frames, size_t(bins), outputElements))
    return 364;
  // Pooled buffers and a cached plan, as in the forward STFT: the per-call
  // version spent ~13 ms of a ~14 ms warm call in nine cudaMalloc/cudaFree
  // pairs plus plan construction (nsys, RTX 5070, 8x16000, n_fft=512).
  std::lock_guard<std::mutex> lock(spectralMutex());
  size_t frameBytes =
      frameElements * (onesided ? sizeof(float) : sizeof(cufftComplex));
  auto *deviceInput =
      static_cast<float *>(scratch(0, inputElements * sizeof(float)));
  auto *deviceDinput =
      static_cast<float *>(scratch(1, inputElements * sizeof(float)));
  auto *deviceWindows =
      static_cast<float *>(scratch(2, windows.size() * sizeof(float)));
  auto *deviceDwindows =
      static_cast<float *>(scratch(3, dwindows.size() * sizeof(float)));
  void *framesBuffer = scratch(4, frameBytes);
  void *dframesBuffer = scratch(5, frameBytes);
  auto *devicePrimal = static_cast<cufftComplex *>(
      scratch(6, outputElements * sizeof(cufftComplex)));
  auto *deviceTangent = static_cast<cufftComplex *>(
      scratch(7, outputElements * sizeof(cufftComplex)));
  if (!deviceInput || !deviceDinput || !deviceWindows || !deviceDwindows ||
      !framesBuffer || !dframesBuffer || !devicePrimal || !deviceTangent)
    return 364;
  cudaError_t status = cudaMemcpy(deviceInput, input,
                                  inputElements * sizeof(float),
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDinput, dinput, inputElements * sizeof(float),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDwindows, dwindows.data(),
                        dwindows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return 364;
  unsigned frameBlocks = unsigned((frameElements + kThreads - 1) / kThreads);
  if (onesided)
    frameRealJVPPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceDinput, deviceDwindows,
        static_cast<float *>(framesBuffer), static_cast<float *>(dframesBuffer),
        batch, samples, nfft, hop, frames, center, padMode);
  else
    frameComplexJVPPolicy<<<frameBlocks, kThreads>>>(
        deviceInput, deviceWindows, deviceDinput, deviceDwindows,
        static_cast<cufftComplex *>(framesBuffer),
        static_cast<cufftComplex *>(dframesBuffer), batch, samples, nfft, hop,
        frames, center, padMode);
  status = cudaGetLastError();
  cufftHandle plan = 0;
  int planStatus = status == cudaSuccess
                       ? cachedPlan(batch * frames, nfft,
                                    onesided ? CUFFT_R2C : CUFFT_C2C, plan)
                       : 1;
  cufftResult first = CUFFT_INVALID_PLAN, second = CUFFT_INVALID_PLAN;
  if (!planStatus) {
    if (onesided) {
      first = cufftExecR2C(plan, static_cast<float *>(framesBuffer), devicePrimal);
      second = first == CUFFT_SUCCESS
                   ? cufftExecR2C(plan, static_cast<float *>(dframesBuffer),
                                  deviceTangent)
                   : CUFFT_INVALID_PLAN;
    } else {
      first = cufftExecC2C(plan, static_cast<cufftComplex *>(framesBuffer),
                           devicePrimal, CUFFT_FORWARD);
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2C(plan, static_cast<cufftComplex *>(dframesBuffer),
                                  deviceTangent, CUFFT_FORWARD)
                   : CUFFT_INVALID_PLAN;
    }
  }
  bool ok = !planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS;
  if (ok && outputScale != 1.0f) {
    unsigned blocks = unsigned((outputElements + kThreads - 1) / kThreads);
    scaleComplex<<<blocks, kThreads>>>(devicePrimal, outputElements,
                                       outputScale);
    scaleComplex<<<blocks, kThreads>>>(deviceTangent, outputElements,
                                       outputScale);
  }
  status = cudaGetLastError();
  // The synchronous copies order after the kernels and surface their errors.
  // With the sample axis innermost the unfold is an identity: copy straight
  // into the caller's buffers.
  std::vector<cufftComplex> primal(inner != 1 ? outputElements : 0),
      tangent(inner != 1 ? outputElements : 0);
  auto *primalOut = inner != 1 ? primal.data()
                               : reinterpret_cast<cufftComplex *>(primalHost);
  auto *tangentOut = inner != 1 ? tangent.data()
                                : reinterpret_cast<cufftComplex *>(tangentHost);
  if (ok && status == cudaSuccess)
    status = cudaMemcpy(primalOut, devicePrimal,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (ok && status == cudaSuccess)
    status = cudaMemcpy(tangentOut, deviceTangent,
                        outputElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (!ok || status != cudaSuccess)
    return 365;
  if (inner == 1)
    return 0;
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
      !shape || !strides || rank < 2 || rank > 8 || axis <= 0 ||
      axis >= rank || nfft <= 0 ||
      hop <= 0 || outputSamples <= 0 || (center != 0 && center != 1) ||
      (onesided != 0 && onesided != 1))
    return 310;
  int frameAxis = axis - 1;
  int frames = int(shape[frameAxis]);
  int bins = int(shape[axis]);
  if (frames <= 0 || bins != (onesided ? nfft / 2 + 1 : nfft))
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
  const size_t spectrumElements = size_t(batch) * size_t(frames) * size_t(bins);
  // Host staging only where the layout needs it (see the DCT entry point).
  const auto *complexInput = reinterpret_cast<const cufftComplex *>(inputHost);
  std::vector<cufftComplex> contiguous, spectra;
  const cufftComplex *staged = complexInput;
  if (!isCompactLayout(rank, shape, strides)) {
    if (!packHostLayout(complexInput, contiguous, rank, shape, strides))
      return 311;
    staged = contiguous.data();
  }
  if (inner != 1) {
    spectra.resize(spectrumElements);
    packAxis(staged, spectra.data(), outer, int64_t(frames) * bins, inner);
    staged = spectra.data();
  }
  std::vector<float> windows;
  if (!expandHostWindows(windowHost, windowRank, windowShape, windowStrides,
                         batchShape, nfft, windows))
    return 312;
  int64_t rawSamples = int64_t(frames - 1) * hop + nfft;
  int trim = center ? nfft / 2 : 0;
  int64_t available = rawSamples - 2 * trim;
  if (outputSamples > available)
    return 313;

  size_t frameElements = size_t(batch) * frames * nfft;
  size_t outputElements = size_t(batch) * outputSamples;
  std::lock_guard<std::mutex> lock(spectralMutex());
  // C2R overwrites its input; the spectrum is re-uploaded into scratch every
  // call, so the caller's data is never touched.
  auto *deviceSpectrum = static_cast<cufftComplex *>(
      scratch(0, spectrumElements * sizeof(cufftComplex)));
  float *deviceRealFrames = nullptr;
  cufftComplex *deviceComplexFrames = nullptr;
  if (onesided)
    deviceRealFrames =
        static_cast<float *>(scratch(1, frameElements * sizeof(float)));
  else
    deviceComplexFrames = static_cast<cufftComplex *>(
        scratch(1, frameElements * sizeof(cufftComplex)));
  auto *deviceWindows =
      static_cast<float *>(scratch(2, windows.size() * sizeof(float)));
  auto *deviceOutput =
      static_cast<float *>(scratch(3, outputElements * sizeof(float)));
  if (!deviceSpectrum || !deviceWindows || !deviceOutput ||
      !(onesided ? static_cast<void *>(deviceRealFrames)
                 : static_cast<void *>(deviceComplexFrames)))
    return 314;
  cudaError_t status = cudaMemcpy(deviceSpectrum, staged,
                                  spectrumElements * sizeof(cufftComplex),
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return 314;
  cufftHandle plan = 0;
  int planStatus = cachedPlan(batch * frames, nfft,
                              onesided ? CUFFT_C2R : CUFFT_C2C, plan);
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
  std::vector<float> output(inner != 1 ? outputElements : 0);
  float *result = inner != 1 ? output.data() : outputHost;
  if (!planStatus && fftStatus == CUFFT_SUCCESS && status == cudaSuccess)
    status = cudaMemcpy(result, deviceOutput, outputElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (planStatus || fftStatus != CUFFT_SUCCESS || status != cudaSuccess)
    return 315;
  if (inner != 1)
    unpackAxis(output.data(), outputHost, outer, outputSamples, inner);
  return 0;
}

// Batched real FFT convolution, full length (the scheduler's
// tessera_nvidia_spectral_conv_f32 entry). x is [rows, xLength], w is
// [kernelRows, kernelLength] with kernelRows == rows or 1, out is
// [rows, xLength + kernelLength - 1], all compact host arrays. nfft must hold
// the full convolution. `scale` is the product of the normalization factors
// (cuFFT is unnormalized both ways); the caller derives it from the norm mode.
// One upload per operand, one download: replaces three host-staged cuFFT calls
// with the pad and the spectrum multiply done in NumPy.
extern "C" int tessera_nvidia_spectral_conv_f32(
    const char *digest, const float *xHost, int rows, int xLength,
    const float *wHost, int kernelRows, int kernelLength, float *outHost,
    int nfft, float scale) {
  if (!validDigest(digest) || !xHost || !wHost || !outHost || rows <= 0 ||
      xLength <= 0 || kernelLength <= 0 ||
      (kernelRows != 1 && kernelRows != rows) || nfft <= 0)
    return 380;
  const int64_t outLength = int64_t(xLength) + kernelLength - 1;
  if (outLength > nfft || int64_t(rows) * nfft > INT32_MAX)
    return 381;
  const int bins = nfft / 2 + 1;
  const size_t realBytes = size_t(rows) * nfft * sizeof(float);
  const size_t kernelRealBytes = size_t(kernelRows) * nfft * sizeof(float);
  std::lock_guard<std::mutex> lock(spectralMutex());
  auto *deviceX = static_cast<float *>(scratch(0, realBytes));
  auto *deviceW = static_cast<float *>(scratch(1, kernelRealBytes));
  auto *spectrumX = static_cast<cufftComplex *>(
      scratch(2, size_t(rows) * bins * sizeof(cufftComplex)));
  auto *spectrumW = static_cast<cufftComplex *>(
      scratch(3, size_t(kernelRows) * bins * sizeof(cufftComplex)));
  if (!deviceX || !deviceW || !spectrumX || !spectrumW)
    return 382;
  // Zero padding: clear, then drop each row in with one strided 2-D copy.
  cudaError_t status = cudaMemset(deviceX, 0, realBytes);
  if (status == cudaSuccess)
    status = cudaMemset(deviceW, 0, kernelRealBytes);
  if (status == cudaSuccess)
    status = cudaMemcpy2D(deviceX, size_t(nfft) * sizeof(float), xHost,
                          size_t(xLength) * sizeof(float),
                          size_t(xLength) * sizeof(float), size_t(rows),
                          cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy2D(deviceW, size_t(nfft) * sizeof(float), wHost,
                          size_t(kernelLength) * sizeof(float),
                          size_t(kernelLength) * sizeof(float),
                          size_t(kernelRows), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return 383;
  cufftHandle forwardX = 0, forwardW = 0, inverse = 0;
  if (cachedPlan(rows, nfft, CUFFT_R2C, forwardX) ||
      cachedPlan(kernelRows, nfft, CUFFT_R2C, forwardW) ||
      cachedPlan(rows, nfft, CUFFT_C2R, inverse))
    return 384;
  if (cufftExecR2C(forwardX, deviceX, spectrumX) != CUFFT_SUCCESS ||
      cufftExecR2C(forwardW, deviceW, spectrumW) != CUFFT_SUCCESS)
    return 385;
  const size_t spectrumElements = size_t(rows) * bins;
  multiplySpectra<<<unsigned((spectrumElements + kThreads - 1) / kThreads),
                    kThreads>>>(spectrumX, spectrumW, rows, kernelRows, bins,
                                scale);
  if (cudaGetLastError() != cudaSuccess)
    return 385;
  // C2R consumes spectrumX and writes the padded result back over deviceX.
  if (cufftExecC2R(inverse, spectrumX, deviceX) != CUFFT_SUCCESS)
    return 385;
  // Download only the first outLength samples of each row, straight into the
  // caller's buffer; the synchronous copy orders after the transforms.
  status = cudaMemcpy2D(outHost, size_t(outLength) * sizeof(float), deviceX,
                        size_t(nfft) * sizeof(float),
                        size_t(outLength) * sizeof(float), size_t(rows),
                        cudaMemcpyDeviceToHost);
  return status == cudaSuccess ? 0 : 386;
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
  if (!validStorage(storage) || !outputHost ||
      !validDescriptor(rank, shape, strides) || axis <= 0 || axis >= rank ||
      outputSamples <= 0)
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
      !tangentHost || !shape || !strides || rank < 2 || rank > 8 ||
      axis <= 0 || axis >= rank ||
      nfft <= 0 || hop <= 0 || outputSamples <= 0 ||
      (center != 0 && center != 1) || (onesided != 0 && onesided != 1))
    return 350;
  int frameAxis = axis - 1;
  int frames = int(shape[frameAxis]);
  int bins = int(shape[axis]);
  if (frames <= 0 || bins != (onesided ? nfft / 2 + 1 : nfft))
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
  const size_t spectrumElements = size_t(batch) * size_t(frames) * size_t(bins);
  std::vector<cufftComplex> contiguous, packed, dcontiguous, dpacked, zeros;
  const cufftComplex *spectra = stageAxis(
      reinterpret_cast<const cufftComplex *>(inputHost), rank, shape, strides,
      outer, int64_t(frames) * bins, inner, spectrumElements, contiguous,
      packed);
  const cufftComplex *dspectra =
      dinputHost
          ? stageAxis(reinterpret_cast<const cufftComplex *>(dinputHost), rank,
                      shape, strides, outer, int64_t(frames) * bins, inner,
                      spectrumElements, dcontiguous, dpacked)
          : (zeros.assign(spectrumElements, make_cuFloatComplex(0.0f, 0.0f)),
             zeros.data());
  if (!spectra || !dspectra)
    return 351;
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
  size_t frameElements = 0, outputElements = 0;
  if (!checkedProduct(size_t(batch) * frames, size_t(nfft), frameElements) ||
      !checkedProduct(size_t(batch), size_t(outputSamples), outputElements))
    return 354;
  // Pooled buffers and a cached plan (see the STFT JVP above).
  std::lock_guard<std::mutex> lock(spectralMutex());
  size_t frameBytes =
      frameElements * (onesided ? sizeof(float) : sizeof(cufftComplex));
  auto *deviceSpectrum = static_cast<cufftComplex *>(
      scratch(0, spectrumElements * sizeof(cufftComplex)));
  auto *deviceDspectrum = static_cast<cufftComplex *>(
      scratch(1, spectrumElements * sizeof(cufftComplex)));
  void *framesBuffer = scratch(2, frameBytes);
  void *dframesBuffer = scratch(3, frameBytes);
  auto *deviceWindows =
      static_cast<float *>(scratch(4, windows.size() * sizeof(float)));
  auto *deviceDwindows =
      static_cast<float *>(scratch(5, dwindows.size() * sizeof(float)));
  auto *devicePrimal =
      static_cast<float *>(scratch(6, outputElements * sizeof(float)));
  auto *deviceTangent =
      static_cast<float *>(scratch(7, outputElements * sizeof(float)));
  if (!deviceSpectrum || !deviceDspectrum || !framesBuffer || !dframesBuffer ||
      !deviceWindows || !deviceDwindows || !devicePrimal || !deviceTangent)
    return 354;
  // C2R overwrites its input; the spectra are staged per call, so that is safe.
  cudaError_t status = cudaMemcpy(deviceSpectrum, spectra,
                                  spectrumElements * sizeof(cufftComplex),
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDspectrum, dspectra,
                        spectrumElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceDwindows, dwindows.data(),
                        dwindows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    return 354;
  cufftHandle plan = 0;
  int planStatus =
      cachedPlan(batch * frames, nfft, onesided ? CUFFT_C2R : CUFFT_C2C, plan);
  cufftResult first = CUFFT_INVALID_PLAN, second = CUFFT_INVALID_PLAN;
  if (!planStatus) {
    if (onesided) {
      first = cufftExecC2R(plan, deviceSpectrum, static_cast<float *>(framesBuffer));
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2R(plan, deviceDspectrum,
                                  static_cast<float *>(dframesBuffer))
                   : CUFFT_INVALID_PLAN;
    } else {
      first = cufftExecC2C(plan, deviceSpectrum,
                           static_cast<cufftComplex *>(framesBuffer),
                           CUFFT_INVERSE);
      second = first == CUFFT_SUCCESS
                   ? cufftExecC2C(plan, deviceDspectrum,
                                  static_cast<cufftComplex *>(dframesBuffer),
                                  CUFFT_INVERSE)
                   : CUFFT_INVALID_PLAN;
    }
  }
  bool ok = !planStatus && first == CUFFT_SUCCESS && second == CUFFT_SUCCESS;
  status = cudaGetLastError();
  if (ok && status == cudaSuccess) {
    unsigned blocks = unsigned((outputElements + kThreads - 1) / kThreads);
    float inverseScale = outputScale / float(nfft);
    if (onesided)
      overlapAddJVP<<<blocks, kThreads>>>(
          static_cast<float *>(framesBuffer), static_cast<float *>(dframesBuffer),
          deviceWindows, deviceDwindows, devicePrimal, deviceTangent, batch,
          frames, nfft, hop, outputSamples, trim, inverseScale);
    else
      overlapAddJVP<<<blocks, kThreads>>>(
          static_cast<cufftComplex *>(framesBuffer),
          static_cast<cufftComplex *>(dframesBuffer), deviceWindows,
          deviceDwindows, devicePrimal, deviceTangent, batch, frames, nfft,
          hop, outputSamples, trim, inverseScale);
    status = cudaGetLastError();
  }
  // Synchronous copies order after the kernels; identity unfold when the
  // sample axis is innermost.
  std::vector<float> primal(inner != 1 ? outputElements : 0),
      tangent(inner != 1 ? outputElements : 0);
  float *primalOut = inner != 1 ? primal.data() : primalHost;
  float *tangentOut = inner != 1 ? tangent.data() : tangentHost;
  if (ok && status == cudaSuccess)
    status = cudaMemcpy(primalOut, devicePrimal,
                        outputElements * sizeof(float), cudaMemcpyDeviceToHost);
  if (ok && status == cudaSuccess)
    status = cudaMemcpy(tangentOut, deviceTangent,
                        outputElements * sizeof(float), cudaMemcpyDeviceToHost);
  if (!ok || status != cudaSuccess)
    return 355;
  if (inner == 1)
    return 0;
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
  if (!validStorage(storage) || !primalHost || !tangentHost ||
      !validDescriptor(rank, shape, strides) || axis <= 0 || axis >= rank ||
      outputSamples <= 0)
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
  if (!validStorage(storage) || !dspectrumHost || !dwindowHost ||
      !validDescriptor(spectrumRank, spectrumShape, spectrumStrides))
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
      !shape || !strides || rank <= 0 || rank > 8 || axis < 0 || axis >= rank || tailSamples < 0 ||
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
      !dxHost || !dwindowHost || !xShape || !xStrides || !dyShape ||
      !dyStrides || xRank <= 0 || xRank > 8 ||
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
  size_t inputElements = size_t(batch) * samples;
  size_t spectralElements = size_t(batch) * frames * bins;
  std::vector<float> inputContiguous, inputPacked;
  std::vector<cufftComplex> dyContiguous, dyPacked;
  const float *input =
      stageAxis(inputHost, xRank, xShape, xStrides, outer, samples, inner,
                inputElements, inputContiguous, inputPacked);
  const cufftComplex *dy = stageAxis(
      reinterpret_cast<const cufftComplex *>(dyHost), dyRank, dyShape,
      dyStrides, outer, int64_t(frames) * bins, inner, spectralElements,
      dyContiguous, dyPacked);
  if (!input || !dy)
    return 323;
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
  if (windowElements == 0 || windowElements > size_t(INT32_MAX))
    return 324; // one reduction block per window element
  size_t frameElements = size_t(batch) * frames * nfft;
  std::lock_guard<std::mutex> lock(spectralMutex());
  auto *deviceDy = static_cast<cufftComplex *>(
      scratch(0, spectralElements * sizeof(cufftComplex)));
  auto *deviceInput =
      static_cast<float *>(scratch(1, inputElements * sizeof(float)));
  auto *deviceWindows =
      static_cast<float *>(scratch(2, windows.size() * sizeof(float)));
  auto *deviceDx = static_cast<float *>(scratch(3, inputElements * sizeof(float)));
  auto *deviceDwindow =
      static_cast<float *>(scratch(4, windowElements * sizeof(float)));
  auto *deviceRowWindow =
      static_cast<int *>(scratch(5, rowWindow.size() * sizeof(int)));
  auto *deviceG = static_cast<float *>(scratch(6, frameElements * sizeof(float)));
  if (!deviceDy || !deviceInput || !deviceWindows || !deviceDx ||
      !deviceDwindow || !deviceRowWindow || !deviceG)
    return 325;
  cudaError_t status = cudaMemcpy(deviceDy, dy,
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceInput, input,
                        inputElements * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceRowWindow, rowWindow.data(),
                        rowWindow.size() * sizeof(int),
                        cudaMemcpyHostToDevice);
  // G = the per-frame inverse DFT of dy (see stftBackwardInputFromG). The
  // one-sided C2R consumes the scratch copy of dy, never the caller's.
  cufftHandle plan = 0;
  if (status == cudaSuccess &&
      cachedPlan(batch * frames, nfft, onesided ? CUFFT_C2R : CUFFT_C2C, plan))
    return 325;
  if (status == cudaSuccess && onesided) {
    weightOnesidedBins<<<unsigned((spectralElements + kThreads - 1) / kThreads),
                         kThreads>>>(deviceDy, spectralElements, bins, nfft,
                                     0.5f, 1.0f);
    status = cudaGetLastError();
    if (status == cudaSuccess &&
        cufftExecC2R(plan, deviceDy, deviceG) != CUFFT_SUCCESS)
      return 325;
  } else if (status == cudaSuccess) {
    if (cufftExecC2C(plan, deviceDy, deviceDy, CUFFT_INVERSE) != CUFFT_SUCCESS)
      return 325;
    realPartScaled<<<unsigned((frameElements + kThreads - 1) / kThreads),
                     kThreads>>>(deviceDy, deviceG, frameElements, 1.0f);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    stftBackwardInputFromG<<<
        unsigned((inputElements + kThreads - 1) / kThreads), kThreads>>>(
        deviceG, deviceWindows, deviceDx, batch, samples, nfft, hop, frames,
        forwardScale, center, padMode);
    stftBackwardWindowFromG<<<unsigned(windowElements), kReduceThreads>>>(
        deviceG, deviceInput, deviceRowWindow, deviceDwindow, batch,
        windowRows, samples, nfft, win, hop, frames, forwardScale, center,
        padMode);
    status = cudaGetLastError();
  }
  // The synchronous copies below order after the kernels; with the sample
  // axis innermost the unfold is an identity, so dx lands in place.
  std::vector<float> dx(inner != 1 ? inputElements : 0);
  float *dxOut = inner != 1 ? dx.data() : dxHost;
  if (status == cudaSuccess)
    status = cudaMemcpy(dxOut, deviceDx, inputElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(dwindowHost, deviceDwindow,
                        windowElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    return 325;
  if (inner != 1)
    unpackAxis(dx.data(), dxHost, outer, samples, inner);
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
      !dspectrumHost || !dwindowHost || !dyShape || !dyStrides ||
      !spectrumShape || !spectrumStrides || dyRank <= 0 || dyRank > 7 ||
      spectrumRank != dyRank + 1 ||
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
  size_t dyElements = size_t(batch) * outputSamples;
  size_t spectralElements = size_t(batch) * frames * bins;
  std::vector<float> dyContiguous, dyPacked;
  std::vector<cufftComplex> spectrumContiguous, spectrumPacked;
  const float *dy =
      stageAxis(dyHost, dyRank, dyShape, dyStrides, outer, outputSamples,
                inner, dyElements, dyContiguous, dyPacked);
  const cufftComplex *spectrum = stageAxis(
      reinterpret_cast<const cufftComplex *>(spectrumHost), spectrumRank,
      spectrumShape, spectrumStrides, outer, int64_t(frames) * bins, inner,
      spectralElements, spectrumContiguous, spectrumPacked);
  if (!dy || !spectrum)
    return 333;
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
  if (windowElements == 0 || windowElements > size_t(INT32_MAX))
    return 334; // one reduction block per window element
  size_t termElements = size_t(batch) * rawSamples;
  std::lock_guard<std::mutex> lock(spectralMutex());
  auto *deviceDy = static_cast<float *>(scratch(0, dyElements * sizeof(float)));
  auto *deviceSpectrum = static_cast<cufftComplex *>(
      scratch(1, spectralElements * sizeof(cufftComplex)));
  auto *deviceWindows =
      static_cast<float *>(scratch(2, windows.size() * sizeof(float)));
  auto *deviceFrames =
      static_cast<float *>(scratch(3, frameElements * sizeof(float)));
  auto *deviceDframes =
      static_cast<float *>(scratch(4, frameElements * sizeof(float)));
  auto *deviceDspectrum = static_cast<cufftComplex *>(
      scratch(5, std::max(spectralElements, onesided ? size_t(0) : frameElements) *
                     sizeof(cufftComplex)));
  auto *deviceDwindow =
      static_cast<float *>(scratch(6, windowElements * sizeof(float)));
  auto *deviceRowWindow =
      static_cast<int *>(scratch(7, rowWindow.size() * sizeof(int)));
  auto *deviceNumerator =
      static_cast<double *>(scratch(8, termElements * sizeof(double)));
  auto *deviceDenominator =
      static_cast<double *>(scratch(9, termElements * sizeof(double)));
  if (!deviceDy || !deviceSpectrum || !deviceWindows || !deviceFrames ||
      !deviceDframes || !deviceDspectrum || !deviceDwindow ||
      !deviceRowWindow || !deviceNumerator || !deviceDenominator)
    return 335;
  cudaError_t status = cudaMemcpy(deviceDy, dy,
                                  dyElements * sizeof(float),
                                  cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceSpectrum, spectrum,
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceWindows, windows.data(),
                        windows.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess)
    status = cudaMemcpy(deviceRowWindow, rowWindow.data(),
                        rowWindow.size() * sizeof(int),
                        cudaMemcpyHostToDevice);
  const unsigned frameBlocks = unsigned((frameElements + kThreads - 1) / kThreads);
  cufftHandle inverse = 0, forward = 0;
  if (status == cudaSuccess &&
      (cachedPlan(batch * frames, nfft, onesided ? CUFFT_C2R : CUFFT_C2C,
                  inverse) ||
       cachedPlan(batch * frames, nfft, onesided ? CUFFT_R2C : CUFFT_C2C,
                  forward)))
    return 335;
  // Frame values: the forward ISTFT's frames (C2R consumes the scratch copy).
  if (status == cudaSuccess && onesided) {
    if (cufftExecC2R(inverse, deviceSpectrum, deviceFrames) != CUFFT_SUCCESS)
      return 335;
    scaleReal<<<frameBlocks, kThreads>>>(deviceFrames, frameElements,
                                         inverseScale);
    status = cudaGetLastError();
  } else if (status == cudaSuccess) {
    if (cufftExecC2C(inverse, deviceSpectrum, deviceSpectrum, CUFFT_INVERSE) !=
        CUFFT_SUCCESS)
      return 335;
    realPartScaled<<<frameBlocks, kThreads>>>(deviceSpectrum, deviceFrames,
                                              frameElements, inverseScale);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    istftOverlapTerms<<<unsigned((termElements + kThreads - 1) / kThreads),
                        kThreads>>>(deviceFrames, deviceWindows,
                                    deviceNumerator, deviceDenominator, batch,
                                    frames, nfft, hop, rawSamples);
    istftBackwardFramesFromTerms<<<frameBlocks, kThreads>>>(
        deviceDy, deviceDenominator, deviceWindows, deviceDframes, batch,
        frames, nfft, hop, outputSamples, rawSamples, center);
    istftBackwardWindowFromTerms<<<unsigned(windowElements),
                                   kReduceThreads>>>(
        deviceDy, deviceFrames, deviceWindows, deviceNumerator,
        deviceDenominator, deviceRowWindow, deviceDwindow, batch, windowRows,
        frames, nfft, win, hop, outputSamples, rawSamples, center);
    status = cudaGetLastError();
  }
  // dspectrum = forward DFT of dframes, weighted like the direct kernel.
  if (status == cudaSuccess && onesided) {
    if (cufftExecR2C(forward, deviceDframes, deviceDspectrum) != CUFFT_SUCCESS)
      return 335;
    weightOnesidedBins<<<unsigned((spectralElements + kThreads - 1) / kThreads),
                         kThreads>>>(deviceDspectrum, spectralElements, bins,
                                     nfft, 2.0f, inverseScale);
    status = cudaGetLastError();
  } else if (status == cudaSuccess) {
    realToComplex<<<frameBlocks, kThreads>>>(deviceDframes, deviceDspectrum,
                                             frameElements);
    status = cudaGetLastError();
    if (status == cudaSuccess &&
        cufftExecC2C(forward, deviceDspectrum, deviceDspectrum,
                     CUFFT_FORWARD) != CUFFT_SUCCESS)
      return 335;
    if (status == cudaSuccess) {
      scaleReal<<<unsigned((2 * spectralElements + kThreads - 1) / kThreads),
                  kThreads>>>(reinterpret_cast<float *>(deviceDspectrum),
                              2 * spectralElements, inverseScale);
      status = cudaGetLastError();
    }
  }
  // The synchronous copies below order after the kernels; identity unfold
  // when the bin axis is innermost.
  std::vector<cufftComplex> dspectrum(inner != 1 ? spectralElements : 0);
  auto *dspectrumOut = inner != 1
                           ? dspectrum.data()
                           : reinterpret_cast<cufftComplex *>(dspectrumHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(dspectrumOut, deviceDspectrum,
                        spectralElements * sizeof(cufftComplex),
                        cudaMemcpyDeviceToHost);
  if (status == cudaSuccess)
    status = cudaMemcpy(dwindowHost, deviceDwindow,
                        windowElements * sizeof(float),
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    return 335;
  if (inner != 1)
    unpackAxis(dspectrum.data(),
               reinterpret_cast<cufftComplex *>(dspectrumHost), outer,
               int64_t(frames) * bins, inner);
  return 0;
}
