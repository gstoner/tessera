//===- apple_gpu_runtime_stub.cpp - Non-Darwin Apple GPU runtime stub ---===//
//
// Phase 8.3 — portable fallback. On non-Darwin builds Metal is not available,
// so this TU provides reference implementations of the C-ABI symbols that the
// AppleGPUToMPS lowering pass emits. Same ABI as apple_gpu_runtime.mm, just no
// MPS dispatch — pure CPU compute. Keeps the static library buildable on
// Linux CI and lets the Python ctypes layer be platform-agnostic.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstring>

#if !defined(__APPLE__)

namespace {

inline void reference_gemm_f32(const float* A, const float* B, float* C,
                               int32_t M, int32_t N, int32_t K) {
  std::memset(C, 0, sizeof(float) * static_cast<std::size_t>(M) *
                        static_cast<std::size_t>(N));
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(m) * K + k];
      for (int32_t n = 0; n < N; ++n) {
        C[static_cast<std::size_t>(m) * N + n] +=
            a * B[static_cast<std::size_t>(k) * N + n];
      }
    }
  }
}

inline void reference_rope_f32(const float* X, const float* Theta, float* Out,
                               int32_t M, int32_t K) {
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t pair = 0; pair < K / 2; ++pair) {
      std::size_t even = static_cast<std::size_t>(row) * K + pair * 2;
      std::size_t odd = even + 1;
      float xe = X[even];
      float xo = X[odd];
      float c = std::cos(Theta[even]);
      float s = std::sin(Theta[even]);
      Out[even] = xe * c - xo * s;
      Out[odd]  = xe * s + xo * c;
    }
  }
}

} // namespace

extern "C" int32_t tessera_apple_gpu_runtime_has_metal(void) { return 0; }

extern "C" void tessera_apple_gpu_mps_matmul_f32(const float* A,
                                                 const float* B, float* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  reference_gemm_f32(A, B, C, M, N, K);
}

extern "C" void tessera_apple_gpu_rope_f32(const float* X, const float* Theta,
                                           float* Out, int32_t M, int32_t K) {
  reference_rope_f32(X, Theta, Out, M, K);
}

extern "C" int32_t tessera_apple_gpu_runtime_msl_cache_size(void) {
  // No Metal -> no MSL cache. Tests gate this on platform.
  return -1;
}

#endif // !__APPLE__
