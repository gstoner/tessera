//===- avx512_coalition_f32.cpp - shared coalition butterfly ------------===//
//
// One Yates radix-2 implementation serves subset/superset zeta and Mobius.
// fp32 values are widened to fp64 for every butterfly update, matching the
// shared numerical policy without creating four architecture emitters.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

extern "C" int tessera_x86_avx512_coalition_butterfly_f32(
    const float *input, float *output, int64_t batch, int64_t size,
    int32_t half, int32_t sign) {
  if (!input || !output || batch <= 0 || size <= 0 ||
      (size & (size - 1)) != 0 || (half != 0 && half != 1) ||
      (sign != -1 && sign != 1))
    return 1;
  std::vector<double> values;
  try {
    values.resize(static_cast<size_t>(batch * size));
  } catch (const std::bad_alloc &) {
    return 2;
  }
  for (int64_t i = 0; i < batch * size; ++i)
    values[static_cast<size_t>(i)] = static_cast<double>(input[i]);
  for (int64_t stride = 1; stride < size; stride <<= 1) {
    const int64_t group = stride << 1;
    for (int64_t system = 0; system < batch; ++system) {
      double *base = values.data() + system * size;
      for (int64_t start = 0; start < size; start += group) {
        double *destination = base + start + (half ? stride : 0);
        const double *source = base + start + (half ? 0 : stride);
        int64_t lane = 0;
#if defined(__AVX512F__)
        for (; lane + 8 <= stride; lane += 8) {
          const __m512d dst = _mm512_loadu_pd(destination + lane);
          const __m512d src = _mm512_loadu_pd(source + lane);
          const __m512d result = sign > 0 ? _mm512_add_pd(dst, src)
                                         : _mm512_sub_pd(dst, src);
          _mm512_storeu_pd(destination + lane, result);
        }
#endif
        for (; lane < stride; ++lane) {
          destination[lane] += static_cast<double>(sign) * source[lane];
        }
      }
    }
  }
  for (int64_t i = 0; i < batch * size; ++i)
    output[i] = static_cast<float>(values[static_cast<size_t>(i)]);
  return 0;
}
