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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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

// Portable row-major lower Cholesky (Cholesky–Banachiewicz). Used on non-Apple
// builds and as a sanity reference. `L` enters holding the symmetric SPD input
// A and exits with the lower-triangular factor (A = L Lᵀ) in its lower triangle;
// the strict upper triangle is left as-is (the caller zeroes it). Returns 0 on
// success, 1 if the matrix is not positive definite.
[[maybe_unused]] inline int32_t reference_cholesky_f32(float* L, int32_t N) {
  for (int32_t j = 0; j < N; ++j) {
    float diag = L[static_cast<std::size_t>(j) * N + j];
    for (int32_t k = 0; k < j; ++k) {
      float ljk = L[static_cast<std::size_t>(j) * N + k];
      diag -= ljk * ljk;
    }
    if (!(diag > 0.0f)) return 1; // not positive definite (or NaN)
    float ljj = std::sqrt(diag);
    L[static_cast<std::size_t>(j) * N + j] = ljj;
    for (int32_t i = j + 1; i < N; ++i) {
      float s = L[static_cast<std::size_t>(i) * N + j];
      for (int32_t k = 0; k < j; ++k)
        s -= L[static_cast<std::size_t>(i) * N + k] *
             L[static_cast<std::size_t>(j) * N + k];
      L[static_cast<std::size_t>(i) * N + j] = s / ljj;
    }
  }
  return 0;
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

// Batched f32 GEMM: A is (batch, M, K), B is (batch, K, N), C is (batch, M, N),
// row-major, contiguous within each batch. Strides count elements (not bytes)
// from one batch slice to the next; for tightly-packed inputs that is M*K /
// K*N / M*N respectively.
//
// Phase 8.2 Item #3: loops over batches calling cblas_sgemm. The kernel itself
// is multi-threaded inside Accelerate, so for typical attention-style workloads
// with large per-batch GEMMs this is competitive with cblas_sgemm_batch_strided.
// We can swap to the batch-level API later if profiling shows tail-latency
// pressure on small batches.
extern "C" void tessera_apple_cpu_gemm_f32_batched(
    const float* A, const float* B, float* C,
    int32_t batch, int32_t M, int32_t N, int32_t K,
    int32_t strideA, int32_t strideB, int32_t strideC) {
  for (int32_t b = 0; b < batch; ++b) {
    const float* Ab = A + static_cast<std::size_t>(b) * strideA;
    const float* Bb = B + static_cast<std::size_t>(b) * strideB;
    float*       Cb = C + static_cast<std::size_t>(b) * strideC;
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                /*M=*/M, /*N=*/N, /*K=*/K,
                /*alpha=*/1.0f,
                Ab, /*lda=*/K,
                Bb, /*ldb=*/N,
                /*beta=*/0.0f,
                Cb, /*ldc=*/N);
#else
    reference_gemm_f32(Ab, Bb, Cb, M, N, K);
#endif
  }
}

// L-series linalg pilot (2026-06-02) — Apple CPU Cholesky factorization.
//
// ABI (matches TileToApple.cpp's tessera_apple.cpu.vector_op symbol
// "tessera_apple_cpu_cholesky_f32"):
//
//   int32_t tessera_apple_cpu_cholesky_f32(
//       const float* A,   // i64 raw pointer (row-major N*N, symmetric SPD)
//       float*       L,    // i64 raw pointer (row-major N*N, lower factor out)
//       int32_t N)         // matrix order
//
// Returns 0 on success; >0 if the leading minor of that order is not positive
// definite (LAPACK `info`), mirroring numpy.linalg.LinAlgError. Produces the
// lower-triangular L with A = L Lᵀ (numpy.linalg.cholesky convention); the
// strict upper triangle of L is zeroed.
//
// LAPACK is column-major. A row-major symmetric buffer reinterpreted as
// column-major is the same matrix (A = Aᵀ), and LAPACK UPLO='U' on that
// column-major view writes the factor into the column-major upper triangle —
// which is exactly the row-major *lower* triangle. Reading that result back
// row-major yields the lower L we want, with no explicit transpose.
extern "C" int32_t tessera_apple_cpu_cholesky_f32(const float* A, float* L,
                                                  int32_t N) {
  if (N <= 0) return 0;
  const std::size_t nn =
      static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
  if (L != A) std::memcpy(L, A, sizeof(float) * nn);

#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  char uplo = 'U'; // col-major 'U' ≡ row-major lower factor
  __LAPACK_int n = static_cast<__LAPACK_int>(N);
  __LAPACK_int lda = n;
  __LAPACK_int info = 0;
  spotrf_(&uplo, &n, L, &lda, &info);
  if (info != 0) return static_cast<int32_t>(info);
#else
  int32_t info = reference_cholesky_f32(L, N);
  if (info != 0) return info;
#endif

  // Zero the strict upper triangle (row-major) so L matches numpy's lower L.
  for (int32_t i = 0; i < N; ++i)
    for (int32_t j = i + 1; j < N; ++j)
      L[static_cast<std::size_t>(i) * N + j] = 0.0f;
  return 0;
}

// Capability probe: returns 1 when Accelerate is the active backend, 0 when
// the reference fallback is in use. Useful for the Python `execute=True`
// machinery to decide whether to skip-or-warn on non-Darwin hosts.
extern "C" int32_t tessera_apple_cpu_runtime_has_accelerate(void) {
  return TESSERA_APPLE_CPU_HAVE_ACCELERATE;
}

//===---------------------------------------------------------------------===//
// fp16 matmul (Phase 8.2 Item #4)
//===---------------------------------------------------------------------===//
//
// Inputs/output use the IEEE-754 half encoding (numpy float16 layout). At the
// ABI boundary they appear as uint16_t* — no implicit casts.
//
// On Apple, prefer BNNSMatMul with f16 descriptors so fp16 stays in fp16
// throughout the computation. If BNNS rejects the call (e.g. older OS without
// the f16 path) we fall through to a cblas_sgemm dispatch with fp32 conversion
// at the boundaries. Either way the C ABI is the same.
//
// On non-Apple builds this collapses to the fp32-conversion path against the
// portable reference kernel.

#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
#  include <Accelerate/Accelerate.h>
#endif

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace {

// Convert a single IEEE-754 half-precision value to float, manually.
// Avoids depending on _Float16 being available (older toolchains, MSVC).
inline float half_to_float(uint16_t h) {
  // Bit-twiddle conversion. Branches on subnormal/inf/nan are inlined and
  // predictable; called at most M*K + K*N times per fp16 GEMM dispatch.
  uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
  uint32_t exp  = (uint32_t(h) & 0x7C00u) >> 10;
  uint32_t frac = uint32_t(h) & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (frac == 0) {
      f = sign;
    } else {
      // subnormal half -> normal float
      while ((frac & 0x0400u) == 0) { frac <<= 1; exp -= 1; }
      exp = exp + 1;
      frac &= ~0x0400u;
      f = sign | ((exp + 112) << 23) | (frac << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (frac << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (frac << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t float_to_half(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000u;
  int32_t  exp  = int32_t((f >> 23) & 0xFFu) - 127 + 15;
  uint32_t frac = f & 0x007FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    frac = (frac | 0x00800000u) >> (1 - exp);
    if (frac & 0x00001000u) frac += 0x00002000u;
    return uint16_t(sign | (frac >> 13));
  }
  if (exp >= 0x1F) {
    if (((f >> 23) & 0xFFu) == 0xFFu) {
      // Inf or NaN
      return uint16_t(sign | 0x7C00u | (frac ? (frac >> 13) | 0x200u : 0));
    }
    return uint16_t(sign | 0x7C00u);
  }
  if (frac & 0x00001000u) {
    frac += 0x00002000u;
    if (frac & 0x00800000u) {
      frac = 0;
      exp += 1;
      if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);
    }
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | (frac >> 13));
}

// bf16 ↔ float conversion (Phase 8.2 Item: bf16 GEMM).
// bf16 is IEEE binary32 with the bottom 16 mantissa bits truncated, so the
// conversion is a simple shift. Round-to-nearest-even on the truncated bits.
inline float bfloat16_to_float(uint16_t b) {
  uint32_t f = static_cast<uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t float_to_bfloat16(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  // Handle NaN: keep at least one mantissa bit set, otherwise NaN -> Inf.
  if ((f & 0x7FC00000u) == 0x7F800000u && (f & 0x007FFFFFu) != 0) {
    return static_cast<uint16_t>((f >> 16) | 0x40u);
  }
  // Round-to-nearest-even on bottom 16 bits.
  uint32_t lsb = (f >> 16) & 1u;
  uint32_t rounded = f + 0x7FFFu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
}

// fp32 reference path used when BNNS isn't available or rejects the input.
// Converts both inputs to fp32, runs cblas_sgemm (or the portable reference
// kernel), then converts back. Preserves correctness at the cost of one
// f16->f32->f16 round-trip per matrix.
inline void cblas_fp16_via_fp32(const uint16_t* A, const uint16_t* B,
                                uint16_t* C, int32_t M, int32_t N, int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float(B[i]);
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, 1.0f, Af.data(), K, Bf.data(), N, 0.0f, Cf.data(), N);
#else
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
#endif
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_half(Cf[i]);
}

#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
// Try BNNSMatMul with f16 descriptors. Returns true on success. The BNNS
// row-major BLAS-equivalent layout is BNNSDataLayout2DFirstMajor where size[]
// is (rows, cols) in BLAS order.
inline bool bnns_gemm_f16(const uint16_t* A, const uint16_t* B, uint16_t* C,
                          int32_t M, int32_t N, int32_t K) {
  // BNNSDataLayout2DFirstMajor: value (i, j) at j*stride[0] + i*stride[1].
  // For C row-major storage where value(i,j) is at i*ld + j, we need
  //   stride[0] = 1 (j step)
  //   stride[1] = leading dim (i step)
  // size[0] = first dim (i / row count), size[1] = second dim (j / col count).
  BNNSNDArrayDescriptor a{}, b{}, c{};
  a.layout = BNNSDataLayout2DFirstMajor;
  a.data_type = BNNSDataTypeFloat16;
  a.size[0] = static_cast<size_t>(M);
  a.size[1] = static_cast<size_t>(K);
  a.stride[0] = 1;
  a.stride[1] = static_cast<size_t>(K);
  a.data = const_cast<uint16_t*>(A);

  b.layout = BNNSDataLayout2DFirstMajor;
  b.data_type = BNNSDataTypeFloat16;
  b.size[0] = static_cast<size_t>(K);
  b.size[1] = static_cast<size_t>(N);
  b.stride[0] = 1;
  b.stride[1] = static_cast<size_t>(N);
  b.data = const_cast<uint16_t*>(B);

  c.layout = BNNSDataLayout2DFirstMajor;
  c.data_type = BNNSDataTypeFloat16;
  c.size[0] = static_cast<size_t>(M);
  c.size[1] = static_cast<size_t>(N);
  c.stride[0] = 1;
  c.stride[1] = static_cast<size_t>(N);
  c.data = C;

  ssize_t ws_size = BNNSMatMulWorkspaceSize(/*transA=*/false, /*transB=*/false,
                                            /*alpha=*/1.0f, &a, &b, &c, nullptr);
  if (ws_size < 0) return false;
  std::vector<uint8_t> workspace(ws_size > 0 ? static_cast<std::size_t>(ws_size) : 0);
  int rc = BNNSMatMul(/*transA=*/false, /*transB=*/false,
                      /*alpha=*/1.0f, &a, &b, &c,
                      ws_size > 0 ? workspace.data() : nullptr, nullptr);
  return rc == 0;
}
#endif

} // namespace

extern "C" void tessera_apple_cpu_gemm_f16(const uint16_t* A, const uint16_t* B,
                                           uint16_t* C, int32_t M, int32_t N,
                                           int32_t K) {
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  if (bnns_gemm_f16(A, B, C, M, N, K)) return;
  // BNNS path failed (older OS, unexpected layout, etc.) — fall through.
#endif
  cblas_fp16_via_fp32(A, B, C, M, N, K);
}

//===---------------------------------------------------------------------===//
// bf16 matmul (Phase 8.2 follow-up)
//===---------------------------------------------------------------------===//
// Inputs/output use the IEEE-style brain-float (bf16) encoding. At the ABI
// boundary they appear as uint16_t* — same convention as fp16. ml_dtypes'
// numpy bfloat16 dtype is byte-compatible with this layout.
//
// Apple path: BNNSMatMul with BNNSDataTypeBFloat16 (macOS 12+). Same descriptor
// shape as the fp16 path, just a different data_type field. Internally BNNS
// does fp32 accumulation, matching the typical mixed-precision contract.
//
// Fallback path: convert to fp32 with a bit-shift, run cblas_sgemm (or the
// portable reference kernel), convert back with round-to-nearest-even.

namespace {

inline void cblas_bf16_via_fp32(const uint16_t* A, const uint16_t* B,
                                uint16_t* C, int32_t M, int32_t N, int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float(B[i]);
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, 1.0f, Af.data(), K, Bf.data(), N, 0.0f, Cf.data(), N);
#else
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
#endif
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16(Cf[i]);
}

#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
inline bool bnns_gemm_bf16(const uint16_t* A, const uint16_t* B, uint16_t* C,
                           int32_t M, int32_t N, int32_t K) {
  // Same descriptor convention as bnns_gemm_f16: row-major storage maps
  // to BNNSDataLayout2DFirstMajor with stride[0]=1, stride[1]=leading-dim.
  BNNSNDArrayDescriptor a{}, b{}, c{};
  a.layout = BNNSDataLayout2DFirstMajor;
  a.data_type = BNNSDataTypeBFloat16;
  a.size[0] = static_cast<size_t>(M);
  a.size[1] = static_cast<size_t>(K);
  a.stride[0] = 1;
  a.stride[1] = static_cast<size_t>(K);
  a.data = const_cast<uint16_t*>(A);

  b.layout = BNNSDataLayout2DFirstMajor;
  b.data_type = BNNSDataTypeBFloat16;
  b.size[0] = static_cast<size_t>(K);
  b.size[1] = static_cast<size_t>(N);
  b.stride[0] = 1;
  b.stride[1] = static_cast<size_t>(N);
  b.data = const_cast<uint16_t*>(B);

  c.layout = BNNSDataLayout2DFirstMajor;
  c.data_type = BNNSDataTypeBFloat16;
  c.size[0] = static_cast<size_t>(M);
  c.size[1] = static_cast<size_t>(N);
  c.stride[0] = 1;
  c.stride[1] = static_cast<size_t>(N);
  c.data = C;

  ssize_t ws_size = BNNSMatMulWorkspaceSize(/*transA=*/false, /*transB=*/false,
                                            /*alpha=*/1.0f, &a, &b, &c, nullptr);
  if (ws_size < 0) return false;
  std::vector<uint8_t> workspace(ws_size > 0 ? static_cast<std::size_t>(ws_size) : 0);
  int rc = BNNSMatMul(/*transA=*/false, /*transB=*/false,
                      /*alpha=*/1.0f, &a, &b, &c,
                      ws_size > 0 ? workspace.data() : nullptr, nullptr);
  return rc == 0;
}
#endif

} // namespace

extern "C" void tessera_apple_cpu_gemm_bf16(const uint16_t* A, const uint16_t* B,
                                            uint16_t* C, int32_t M, int32_t N,
                                            int32_t K) {
#if TESSERA_APPLE_CPU_HAVE_ACCELERATE
  if (bnns_gemm_bf16(A, B, C, M, N, K)) return;
#endif
  cblas_bf16_via_fp32(A, B, C, M, N, K);
}

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
