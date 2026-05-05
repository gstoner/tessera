//===- apple_cpu_runtime.cpp - Accelerate-backed Apple CPU runtime ------===//
//
// Phase 8.2 — Apple CPU native execution.
//
// C-ABI symbols emitted as call targets by MatmulToAppleCPUPass. Each symbol
// is a thin shim around an Accelerate routine on Darwin and a portable
// fallback on every other platform (so the static library can be built on
// Linux CI without breaking).
//
// ABI contract (matches MatmulToAppleCPUPass.cpp):
//
//   void tessera_apple_cpu_gemm_f32(
//       const float* A,    // i64 raw pointer (row-major M*K)
//       const float* B,    // i64 raw pointer (row-major K*N)
//       float*       C,    // i64 raw pointer (row-major M*N, written)
//       int32_t M, int32_t N, int32_t K)
//
// The pass passes pointers as i64; this TU receives them as `void*` to keep
// the symbol table compatible with both i64 and uintptr-style ABIs.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__APPLE__)
   // Opt into the modern Accelerate CBLAS/LAPACK headers (macOS 13.3+).
   // Without this, `cblas_sgemm` is marked deprecated.
#  ifndef ACCELERATE_NEW_LAPACK
#    define ACCELERATE_NEW_LAPACK
#  endif
#  include <Accelerate/Accelerate.h>
#  define TESSERA_APPLE_CPU_HAVE_ACCELERATE 1
#else
#  define TESSERA_APPLE_CPU_HAVE_ACCELERATE 0
#endif

namespace {

// Portable row-major fallback (used on non-Apple builds and as a sanity
// reference). Not optimized — Phase 8.2 expects Accelerate to be live.
[[maybe_unused]] inline void reference_gemm_f32(const float* A, const float* B,
                                                float* C, int32_t M, int32_t N,
                                                int32_t K) {
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

} // namespace

extern "C" void tessera_apple_cpu_gemm_f32(const float* A, const float* B,
                                           float* C, int32_t M, int32_t N,
                                           int32_t K) {
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  // C := 1.0 * A * B + 0.0 * C, row-major, no transpose.
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              /*M=*/M, /*N=*/N, /*K=*/K,
              /*alpha=*/1.0f,
              A, /*lda=*/K,
              B, /*ldb=*/N,
              /*beta=*/0.0f,
              C, /*ldc=*/N);
#else
  reference_gemm_f32(A, B, C, M, N, K);
#endif
}

// Capability probe: returns 1 when Accelerate is the active backend, 0 when
// the reference fallback is in use. Useful for the Python `execute=True`
// machinery to decide whether to skip-or-warn on non-Darwin hosts.
extern "C" int32_t tessera_apple_cpu_runtime_has_accelerate(void) {
  return TESSERA_APPLE_CPU_HAVE_ACCELERATE;
}
