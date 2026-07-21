// Portable baseline kernels for exact-host E2E proof on x86-64 machines that
// do not expose AVX-512.  Keep this translation unit free of vector intrinsics;
// its target is the compiler-owned x86_64_base image envelope.

#include <cmath>
#include <cstdint>
#include <limits>

extern "C" void tessera_x86_base_softmax_f32(const float *input,
                                              int64_t rows, int64_t columns,
                                              float *output) {
  for (int64_t row = 0; row < rows; ++row) {
    const float *source = input + row * columns;
    float *destination = output + row * columns;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t column = 0; column < columns; ++column)
      maximum = source[column] > maximum ? source[column] : maximum;
    float denominator = 0.0f;
    for (int64_t column = 0; column < columns; ++column) {
      destination[column] = std::exp(source[column] - maximum);
      denominator += destination[column];
    }
    for (int64_t column = 0; column < columns; ++column)
      destination[column] /= denominator;
  }
}

extern "C" void tessera_x86_base_reduce_f32(const float *input,
                                             int64_t rows, int64_t columns,
                                             float *output, int kind) {
  for (int64_t row = 0; row < rows; ++row) {
    const float *source = input + row * columns;
    if (kind == 1) {
      float accumulator = -std::numeric_limits<float>::infinity();
      bool hasNan = false;
      for (int64_t column = 0; column < columns; ++column) {
        hasNan |= std::isnan(source[column]);
        accumulator = source[column] > accumulator ? source[column] : accumulator;
      }
      output[row] = hasNan ? std::numeric_limits<float>::quiet_NaN() : accumulator;
      continue;
    }
    float accumulator = 0.0f;
    for (int64_t column = 0; column < columns; ++column)
      accumulator += source[column];
    output[row] = kind == 2 && columns > 0
                      ? accumulator / static_cast<float>(columns)
                      : accumulator;
  }
}
