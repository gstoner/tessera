// Architecture-owned AD-SOLVER-IFT-1 pilot.
//
// Residual: R(theta, x) = x*x - theta.
// Matrix-free transposed solve: (dR/dx)^T lambda = cotangent.
// Residual adjoint: dtheta = -lambda * dR/dtheta = lambda.

#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

extern "C" void tessera_x86_avx512_solver_ift_sqrt_f32(
    const float *theta, const float *solution, const float *cotangent,
    float *residual, float *linear_solution, float *parameter_cotangent,
    int64_t n) {
  int64_t i = 0;
#if defined(__AVX512F__)
  const __m512 two = _mm512_set1_ps(2.0f);
  for (; i + 16 <= n; i += 16) {
    const __m512 t = _mm512_loadu_ps(theta + i);
    const __m512 x = _mm512_loadu_ps(solution + i);
    const __m512 c = _mm512_loadu_ps(cotangent + i);
    const __m512 r = _mm512_sub_ps(_mm512_mul_ps(x, x), t);
    const __m512 lambda = _mm512_div_ps(c, _mm512_mul_ps(two, x));
    _mm512_storeu_ps(residual + i, r);
    _mm512_storeu_ps(linear_solution + i, lambda);
    _mm512_storeu_ps(parameter_cotangent + i, lambda);
  }
#endif
  for (; i < n; ++i) {
    const float r = solution[i] * solution[i] - theta[i];
    const float lambda = cotangent[i] / (2.0f * solution[i]);
    residual[i] = r;
    linear_solution[i] = lambda;
    parameter_cotangent[i] = lambda;
  }
}
