//===- avx512_tridiagonal_f32.cpp - batched tridiagonal solve ------------===//
//
// The three diagonals are shared by `batch` independent right-hand sides.
// Thomas' recurrence remains sequential in the matrix row, but each row is
// evaluated across sixteen independent systems.  This is the CPU analogue of
// the gfx1151 PCR consumer: the Schedule contract fixes the mathematics while
// each architecture chooses its natural parallel axis.
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

extern "C" int tessera_x86_avx512_tridiagonal_solve_f32(
    const float *dl, const float *diag, const float *du, const float *rhs,
    float *output, int64_t batch, int64_t n) {
  if (!dl || !diag || !du || !rhs || !output || batch <= 0 || n <= 0)
    return 1;
  if (batch > INT32_MAX / n)
    return 2; // AVX-512 gather/scatter indices are signed i32.

  std::vector<double> cprime;
  std::vector<double> forward;
  try {
    cprime.resize(static_cast<size_t>(n));
    // Row-major by recurrence step makes the independent-system axis
    // contiguous.  The public RHS remains batch-major; packing once avoids a
    // gather in every forward/backward SIMD step.
    forward.resize(static_cast<size_t>(batch * n));
  } catch (const std::bad_alloc &) {
    return 3;
  }
  double pivot = static_cast<double>(diag[0]);
  if (pivot == 0.0f || !std::isfinite(pivot))
    return 4;
  cprime[0] = static_cast<double>(du[0]) / pivot;
  for (int64_t row = 1; row < n; ++row) {
    pivot = static_cast<double>(diag[row]) -
            static_cast<double>(dl[row]) *
                cprime[static_cast<size_t>(row - 1)];
    if (pivot == 0.0f || !std::isfinite(pivot))
      return 4;
    cprime[static_cast<size_t>(row)] =
        static_cast<double>(du[row]) / pivot;
  }

#if defined(__AVX512F__)
  for (int64_t system = 0; system < batch; ++system)
    forward[static_cast<size_t>(system)] =
        static_cast<double>(rhs[system * n]) /
        static_cast<double>(diag[0]);
  for (int64_t row = 1; row < n; ++row) {
    const double rowPivot =
        static_cast<double>(diag[row]) - static_cast<double>(dl[row]) *
                                              cprime[static_cast<size_t>(row - 1)];
    int64_t system = 0;
    for (; system + 8 <= batch; system += 8) {
      alignas(64) double packedRhs[8];
      for (int lane = 0; lane < 8; ++lane)
        packedRhs[lane] = static_cast<double>(rhs[(system + lane) * n + row]);
      const __m512d value = _mm512_div_pd(
          _mm512_sub_pd(
              _mm512_load_pd(packedRhs),
              _mm512_mul_pd(
                  _mm512_set1_pd(static_cast<double>(dl[row])),
                  _mm512_loadu_pd(&forward[static_cast<size_t>((row - 1) * batch + system)]))),
          _mm512_set1_pd(rowPivot));
      _mm512_storeu_pd(&forward[static_cast<size_t>(row * batch + system)], value);
    }
    for (; system < batch; ++system)
      forward[static_cast<size_t>(row * batch + system)] =
          (static_cast<double>(rhs[system * n + row]) -
           static_cast<double>(dl[row]) *
               forward[static_cast<size_t>((row - 1) * batch + system)]) /
          rowPivot;
  }
  for (int64_t row = n - 2; row >= 0; --row) {
    int64_t system = 0;
    for (; system + 8 <= batch; system += 8) {
      const __m512d value = _mm512_sub_pd(
          _mm512_loadu_pd(&forward[static_cast<size_t>(row * batch + system)]),
          _mm512_mul_pd(
              _mm512_set1_pd(cprime[static_cast<size_t>(row)]),
              _mm512_loadu_pd(&forward[static_cast<size_t>((row + 1) * batch + system)])));
      _mm512_storeu_pd(&forward[static_cast<size_t>(row * batch + system)], value);
    }
    for (; system < batch; ++system)
      forward[static_cast<size_t>(row * batch + system)] -=
          cprime[static_cast<size_t>(row)] *
          forward[static_cast<size_t>((row + 1) * batch + system)];
  }
#else
  for (int64_t row = 0; row < n; ++row)
    for (int64_t system = 0; system < batch; ++system) {
      const double rowPivot =
          row == 0 ? static_cast<double>(diag[0])
                   : static_cast<double>(diag[row]) -
                         static_cast<double>(dl[row]) * cprime[row - 1];
      forward[row * batch + system] =
          (static_cast<double>(rhs[system * n + row]) -
           (row ? static_cast<double>(dl[row]) *
                      forward[(row - 1) * batch + system]
                : 0.0)) /
          rowPivot;
    }
  for (int64_t row = n - 2; row >= 0; --row)
    for (int64_t system = 0; system < batch; ++system)
      forward[row * batch + system] -=
          cprime[row] * forward[(row + 1) * batch + system];
#endif
  for (int64_t system = 0; system < batch; ++system)
    for (int64_t row = 0; row < n; ++row)
      output[system * n + row] =
          static_cast<float>(forward[row * batch + system]);
  return 0;
}
