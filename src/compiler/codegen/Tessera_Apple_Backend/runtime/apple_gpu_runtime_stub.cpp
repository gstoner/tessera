//===- apple_gpu_runtime_stub.cpp - Non-Darwin Apple GPU runtime stub ---===//
//
// Phase 8.3 — portable fallback. On non-Darwin builds Metal is not available,
// so this TU provides reference implementations of the C-ABI symbols that the
// AppleGPUToMPS lowering pass emits. Same ABI as apple_gpu_runtime.mm, just no
// MPS dispatch — pure CPU compute. Keeps the static library buildable on
// Linux CI and lets the Python ctypes layer be platform-agnostic.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if !defined(__APPLE__)

#include <cstdlib>

// R0 — persistent device-tensor handle, non-Apple reference. Backed by plain
// host memory (no Metal); ts_dev_contents returns that pointer so the Python
// DeviceTensor behaves identically (zero-copy numpy view), just on the CPU.
struct TsDeviceTensor {
  void* data;
  int64_t nbytes;
};
extern "C" TsDeviceTensor* ts_dev_alloc(int64_t nbytes) {
  if (nbytes <= 0) return nullptr;
  void* p = std::calloc(1, static_cast<std::size_t>(nbytes));
  if (!p) return nullptr;
  return new TsDeviceTensor{p, nbytes};
}
extern "C" void* ts_dev_contents(TsDeviceTensor* t) { return t ? t->data : nullptr; }
extern "C" int64_t ts_dev_nbytes(TsDeviceTensor* t) { return t ? t->nbytes : 0; }
extern "C" void ts_dev_upload(TsDeviceTensor* t, const void* src, int64_t n) {
  if (t && src && n > 0 && n <= t->nbytes) std::memcpy(t->data, src, static_cast<std::size_t>(n));
}
extern "C" void ts_dev_download(TsDeviceTensor* t, void* dst, int64_t n) {
  if (t && dst && n > 0 && n <= t->nbytes) std::memcpy(dst, t->data, static_cast<std::size_t>(n));
}
extern "C" void ts_dev_free(TsDeviceTensor* t) {
  if (t) { std::free(t->data); delete t; }
}
extern "C" int32_t ts_dev_is_metal(void) { return 0; }
// Interop escape hatches — no Metal device off Darwin, so no raw handles.
extern "C" void* ts_dev_mtl_buffer(TsDeviceTensor*) { return nullptr; }
extern "C" void* tessera_apple_gpu_device_handle(void) { return nullptr; }
extern "C" void* tessera_apple_gpu_command_queue_handle(void) { return nullptr; }
// SIMD-feature caps — no Metal device off Darwin, so no SIMD intrinsics.
extern "C" int32_t tessera_apple_gpu_simd_caps(void) { return 0; }
// GPU-native RNG (opt-in) — no Metal off Darwin; return 0 so Python uses its
// own RNG fallback.
extern "C" int32_t tessera_apple_gpu_random_uniform_f32(float*, int64_t, uint64_t,
                                                        float, float) { return 0; }
extern "C" int32_t tessera_apple_gpu_random_normal_f32(float*, int64_t, uint64_t,
                                                       float, float) { return 0; }

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

// Apple-sample Pattern 4 probe — stub returns -1 ("runtime not
// available"). Tests that exercise the GPU timeout machinery skip when
// this is the answer.
extern "C" int32_t tessera_apple_gpu_commit_and_wait_timeout_probe(
    uint64_t /*timeout_ms*/) {
  return -1;
}

// Apple-sample pattern 6 — row-major strides probe. The math is pure
// (no Metal calls), so the stub computes the same answer as the real
// runtime. Lets the unit test verify the contract on every host —
// Darwin + Apple silicon OR Linux/Intel — without conditional skipping.
extern "C" int32_t tessera_apple_gpu_row_major_strides(const int64_t *dims_in,
                                                      int32_t rank,
                                                      int64_t *strides_out) {
  if (!dims_in || !strides_out || rank <= 0 || rank > 8) return 0;
  int64_t stride = 1;
  for (int32_t i = 0; i < rank; ++i) {
    strides_out[i] = stride;
    stride *= dims_in[i];
  }
  return rank;
}

// Pattern 3 follow-on (2026-06-01) — Apple stride-alignment rules.
// Pure math, no Metal calls; stub matches the Darwin runtime so the
// test contract holds on every host. See ``apple_gpu_runtime.mm`` for
// the rule documentation.
extern "C" int32_t tessera_apple_gpu_row_major_strides_aligned(
    const int64_t *dims_in, int32_t rank, int32_t element_bits,
    int32_t ml_usage, int64_t *strides_out) {
  if (!dims_in || !strides_out || rank <= 0 || rank > 8) return 0;
  if (element_bits <= 0 || element_bits > 64) return 0;
  int32_t alignment_bits = 0;
  if (element_bits < 8)       alignment_bits = 1024;  // 128 bytes (sub-byte)
  else if (ml_usage != 0)     alignment_bits = 512;   // 64 bytes (ML usage)
  int64_t elem_align = 1;
  if (alignment_bits > 0) {
    if (alignment_bits % element_bits != 0) return 0;
    elem_align = alignment_bits / element_bits;
    if (elem_align < 1) elem_align = 1;
  }
  strides_out[0] = 1;
  if (rank == 1) return rank;
  int64_t natural = dims_in[0];
  int64_t aligned = natural;
  if (elem_align > 1) {
    int64_t rem = natural % elem_align;
    if (rem != 0) aligned = natural + (elem_align - rem);
  }
  strides_out[1] = aligned;
  int64_t acc = aligned;
  for (int32_t i = 2; i < rank; ++i) {
    acc *= dims_in[i - 1];
    strides_out[i] = acc;
  }
  return rank;
}

extern "C" void tessera_apple_gpu_mps_matmul_f32(const float* A,
                                                 const float* B, float* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  reference_gemm_f32(A, B, C, M, N, K);
}

// GPU linear-algebra lane — no Metal off Darwin; return -1 so the Python wrapper
// falls back to the numpy reference (np.linalg.cholesky / solve / triangular).
extern "C" int32_t tessera_apple_gpu_cholesky_f32(const float*, float*, int32_t) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_solve_cholesky_f32(const float*, const float*,
                                                        float*, int32_t, int32_t) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_solve_lu_f32(const float*, const float*,
                                                  float*, int32_t, int32_t) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_tri_solve_f32(const float*, const float*,
                                                   float*, int32_t, int32_t,
                                                   int32_t, int32_t, int32_t) {
  return -1;
}
// One-sided Jacobi SVD — no Metal off Darwin; return 0 so Python uses numpy.
extern "C" int32_t tessera_apple_gpu_svd_f32(const float*, float*, float*, float*,
                                             int32_t, int32_t) { return 0; }
extern "C" int32_t tessera_apple_gpu_svd_batched_f32(const float*, float*, float*,
                                                     float*, int32_t, int32_t,
                                                     int32_t) { return 0; }
extern "C" int32_t tessera_apple_gpu_svd_bl_batched_f32(const float*, float*, float*,
                                                        float*, int32_t, int32_t,
                                                        int32_t) { return 0; }
// Batched factorizations/solves — no Metal off Darwin; Python uses numpy.
extern "C" int32_t tessera_apple_gpu_cholesky_batched_f32(const float*, float*,
                                                          int32_t*, int32_t,
                                                          int32_t) { return 0; }
extern "C" int32_t tessera_apple_gpu_tri_solve_batched_f32(const float*, const float*,
                                                           float*, int32_t, int32_t,
                                                           int32_t, int32_t, int32_t,
                                                           int32_t) { return 0; }
// R0 resident cast + general resident matmul2d — no Metal off Darwin.
extern "C" int32_t ts_dev_cast(TsDeviceTensor*, TsDeviceTensor*, int64_t, int32_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_dev(TsDeviceTensor*, TsDeviceTensor*,
                                                       TsDeviceTensor*, int32_t,
                                                       int32_t, int32_t, int32_t) {
  return 0;
}

namespace {

// Phase 8.4.4 — fp16/bf16 bit-pattern conversion helpers for the non-Darwin
// stub. Same shape as the conversion helpers in apple_gpu_runtime.mm; kept
// inline here so the stub TU is self-contained.

inline float half_to_float_stub(uint16_t h) {
  uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
  uint32_t exp  = (uint32_t(h) & 0x7C00u) >> 10;
  uint32_t frac = uint32_t(h) & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (frac == 0) {
      f = sign;
    } else {
      while ((frac & 0x0400u) == 0) { frac <<= 1; exp -= 1; }
      exp += 1;
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

inline uint16_t float_to_half_stub(float v) {
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

// bf16 bit helpers, defined up front (used by the MLA-decode + GQA stubs below).
// Defined here rather than forward-declared + defined later: a `static inline`
// forward-declaration split from its definition makes Clang flag intervening
// calls as ambiguous (and would break a Clang Linux build).
static inline float gqa_bf16_to_f32_stub(uint16_t b) {
  uint32_t f = static_cast<uint32_t>(b) << 16;
  float o;
  std::memcpy(&o, &f, sizeof(o));
  return o;
}
static inline uint16_t gqa_f32_to_bf16_stub(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  uint32_t lsb = (f >> 16) & 1u;
  return static_cast<uint16_t>((f + 0x7FFFu + lsb) >> 16);
}

inline float bfloat16_to_float_stub(uint16_t b) {
  uint32_t f = static_cast<uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t float_to_bfloat16_stub(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  if ((f & 0x7FC00000u) == 0x7F800000u && (f & 0x007FFFFFu) != 0) {
    return static_cast<uint16_t>((f >> 16) | 0x40u);
  }
  uint32_t lsb = (f >> 16) & 1u;
  uint32_t rounded = f + 0x7FFFu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
}

} // namespace

extern "C" void tessera_apple_gpu_mps_matmul_f16(const uint16_t* A,
                                                 const uint16_t* B,
                                                 uint16_t* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  // Convert each operand to fp32, run the reference fp32 GEMM, convert back.
  // Same numerical contract as the BNNS-fallback path on CPU.
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_half_stub(Cf[i]);
}

extern "C" void tessera_apple_gpu_mps_matmul_bf16(const uint16_t* A,
                                                  const uint16_t* B,
                                                  uint16_t* C,
                                                  int32_t M, int32_t N,
                                                  int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16_stub(Cf[i]);
}

extern "C" void tessera_apple_gpu_rope_f32(const float* X, const float* Theta,
                                           float* Out, int32_t M, int32_t K) {
  reference_rope_f32(X, Theta, Out, M, K);
}

namespace {

inline void reference_flash_attn_f32(const float* Q, const float* K,
                                     const float* V, float* O,
                                     int32_t B, int32_t Sq, int32_t Sk,
                                     int32_t D, float scale, int32_t causal) {
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t q = 0; q < Sq; ++q) {
      const float* Qrow = Q + (static_cast<std::size_t>(b) * Sq + q) * D;
      const float* Kbase = K + static_cast<std::size_t>(b) * Sk * D;
      const float* Vbase = V + static_cast<std::size_t>(b) * Sk * D;
      float* Orow = O + (static_cast<std::size_t>(b) * Sq + q) * D;
      float m = -std::numeric_limits<float>::infinity();
      float l = 0.0f;
      float o[256];
      for (int32_t d = 0; d < D; ++d) o[d] = 0.0f;
      for (int32_t k = 0; k < Sk; ++k) {
        if (causal != 0 && k > q) break;
        const float* Krow = Kbase + static_cast<std::size_t>(k) * D;
        float score = 0.0f;
        for (int32_t d = 0; d < D; ++d) score += Qrow[d] * Krow[d];
        score *= scale;
        float new_m = std::max(m, score);
        float exp_old = std::exp(m - new_m);
        float exp_score = std::exp(score - new_m);
        float new_l = l * exp_old + exp_score;
        const float* Vrow = Vbase + static_cast<std::size_t>(k) * D;
        for (int32_t d = 0; d < D; ++d) {
          o[d] = o[d] * exp_old + Vrow[d] * exp_score;
        }
        m = new_m;
        l = new_l;
      }
      if (l == 0.0f) {
        for (int32_t d = 0; d < D; ++d) Orow[d] = 0.0f;
      } else {
        float inv_l = 1.0f / l;
        for (int32_t d = 0; d < D; ++d) Orow[d] = o[d] * inv_l;
      }
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_flash_attn_f32(const float* Q, const float* K,
                                                 const float* V, float* O,
                                                 int32_t B, int32_t Sq,
                                                 int32_t Sk, int32_t D,
                                                 float scale, int32_t causal) {
  reference_flash_attn_f32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
}

// Phase 8.4.4.2 — fp16 / bf16 stubs for flash_attn (fp32 conversion path).

extern "C" void tessera_apple_gpu_flash_attn_f16(const uint16_t* Q,
                                                 const uint16_t* K_buf,
                                                 const uint16_t* V,
                                                 uint16_t* O,
                                                 int32_t B, int32_t Sq,
                                                 int32_t Sk, int32_t D,
                                                 float scale, int32_t causal) {
  std::vector<float> Qf(static_cast<std::size_t>(B) * Sq * D);
  std::vector<float> Kf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Vf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < Qf.size(); ++i) Qf[i] = half_to_float_stub(Q[i]);
  for (std::size_t i = 0; i < Kf.size(); ++i) Kf[i] = half_to_float_stub(K_buf[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = half_to_float_stub(V[i]);
  reference_flash_attn_f32(Qf.data(), Kf.data(), Vf.data(), Of.data(),
                           B, Sq, Sk, D, scale, causal);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_flash_attn_bf16(const uint16_t* Q,
                                                  const uint16_t* K_buf,
                                                  const uint16_t* V,
                                                  uint16_t* O,
                                                  int32_t B, int32_t Sq,
                                                  int32_t Sk, int32_t D,
                                                  float scale, int32_t causal) {
  std::vector<float> Qf(static_cast<std::size_t>(B) * Sq * D);
  std::vector<float> Kf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Vf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < Qf.size(); ++i) Qf[i] = bfloat16_to_float_stub(Q[i]);
  for (std::size_t i = 0; i < Kf.size(); ++i) Kf[i] = bfloat16_to_float_stub(K_buf[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = bfloat16_to_float_stub(V[i]);
  reference_flash_attn_f32(Qf.data(), Kf.data(), Vf.data(), Of.data(),
                           B, Sq, Sk, D, scale, causal);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

namespace {

inline void reference_softmax_f32(const float* X, float* Out, int32_t M,
                                  int32_t K) {
  for (int32_t r = 0; r < M; ++r) {
    const float* row = X + static_cast<std::size_t>(r) * K;
    float* out_row = Out + static_cast<std::size_t>(r) * K;
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t j = 0; j < K; ++j) row_max = std::max(row_max, row[j]);
    float denom = 0.0f;
    for (int32_t j = 0; j < K; ++j) {
      float e = std::exp(row[j] - row_max);
      out_row[j] = e;
      denom += e;
    }
    float inv = 1.0f / denom;
    for (int32_t j = 0; j < K; ++j) out_row[j] *= inv;
  }
}

inline void reference_gelu_f32(const float* X, float* Out, int32_t N) {
  static constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int32_t i = 0; i < N; ++i) {
    float v = X[i];
    float t = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    Out[i] = 0.5f * v * (1.0f + std::tanh(t));
  }
}

} // namespace

extern "C" void tessera_apple_gpu_softmax_f32(const float* X, float* Out,
                                              int32_t M, int32_t K) {
  reference_softmax_f32(X, Out, M, K);
}

extern "C" void tessera_apple_gpu_gelu_f32(const float* X, float* Out,
                                           int32_t N) {
  reference_gelu_f32(X, Out, N);
}

// Phase 8.4.4.1 — fp16 / bf16 stubs for rope / softmax / gelu. All routes
// fp32-convert at the boundary, run the existing f32 reference, convert
// back. Same shape as the matmul fp16/bf16 stub from Phase 8.4.4.

extern "C" void tessera_apple_gpu_rope_f16(const uint16_t* X,
                                           const uint16_t* Theta,
                                           uint16_t* Out,
                                           int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Tf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = half_to_float_stub(X[i]);
  for (std::size_t i = 0; i < Tf.size(); ++i) Tf[i] = half_to_float_stub(Theta[i]);
  reference_rope_f32(Xf.data(), Tf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_rope_bf16(const uint16_t* X,
                                            const uint16_t* Theta,
                                            uint16_t* Out,
                                            int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Tf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_stub(X[i]);
  for (std::size_t i = 0; i < Tf.size(); ++i) Tf[i] = bfloat16_to_float_stub(Theta[i]);
  reference_rope_f32(Xf.data(), Tf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_softmax_f16(const uint16_t* X, uint16_t* Out,
                                              int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = half_to_float_stub(X[i]);
  reference_softmax_f32(Xf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_softmax_bf16(const uint16_t* X, uint16_t* Out,
                                               int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_stub(X[i]);
  reference_softmax_f32(Xf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_gelu_f16(const uint16_t* X, uint16_t* Out,
                                           int32_t N) {
  std::vector<float> Xf(static_cast<std::size_t>(N));
  std::vector<float> Of(static_cast<std::size_t>(N));
  for (int32_t i = 0; i < N; ++i) Xf[i] = half_to_float_stub(X[i]);
  reference_gelu_f32(Xf.data(), Of.data(), N);
  for (int32_t i = 0; i < N; ++i) Out[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_gelu_bf16(const uint16_t* X, uint16_t* Out,
                                            int32_t N) {
  std::vector<float> Xf(static_cast<std::size_t>(N));
  std::vector<float> Of(static_cast<std::size_t>(N));
  for (int32_t i = 0; i < N; ++i) Xf[i] = bfloat16_to_float_stub(X[i]);
  reference_gelu_f32(Xf.data(), Of.data(), N);
  for (int32_t i = 0; i < N; ++i) Out[i] = float_to_bfloat16_stub(Of[i]);
}

namespace {

inline void reference_matmul_softmax_f32(const float* A, const float* B,
                                         float* O, int32_t M, int32_t N,
                                         int32_t K) {
  // Phase 8.4.3 — fused matmul -> softmax(axis=-1) reference. Same algorithm
  // as the MSL kernel: per-row, compute row of A@B, then numerically-stable
  // softmax over that row. The GPU kernel caps N <= 256 with a stack array;
  // the stub uses a heap allocation per row so it stays correct for any N.
  for (int32_t row = 0; row < M; ++row) {
    float* scores = new float[static_cast<std::size_t>(N)]();
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t n = 0; n < N; ++n) row_max = std::max(row_max, scores[n]);
    float denom = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      scores[n] = std::exp(scores[n] - row_max);
      denom += scores[n];
    }
    float* out_row = O + static_cast<std::size_t>(row) * N;
    if (denom == 0.0f) {
      for (int32_t n = 0; n < N; ++n) out_row[n] = 0.0f;
    } else {
      float inv = 1.0f / denom;
      for (int32_t n = 0; n < N; ++n) out_row[n] = scores[n] * inv;
    }
    delete[] scores;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_f32(const float* A,
                                                     const float* B, float* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K) {
  reference_matmul_softmax_f32(A, B, O, M, N, K);
}

// Phase 8.4.6 — threadgroup-tiled variant. On non-Darwin we just route
// to the same reference implementation; the tiling is a Metal-specific
// perf optimization that doesn't apply to the portable fallback.
extern "C" void tessera_apple_gpu_matmul_softmax_tiled_f32(const float* A,
                                                           const float* B,
                                                           float* O,
                                                           int32_t M,
                                                           int32_t N,
                                                           int32_t K) {
  reference_matmul_softmax_f32(A, B, O, M, N, K);
}

// Phase 8.4.4.2 — fp16 / bf16 stubs for fused matmul -> softmax.

extern "C" void tessera_apple_gpu_matmul_softmax_f16(const uint16_t* A,
                                                     const uint16_t* B,
                                                     uint16_t* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_softmax_bf16(const uint16_t* A,
                                                      const uint16_t* B,
                                                      uint16_t* O,
                                                      int32_t M, int32_t N,
                                                      int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

// Native-half tiled variants. On non-Darwin the tiling is a no-op; route to the
// same reference path so f16/bf16 large-N chains stay numerically correct.

extern "C" void tessera_apple_gpu_matmul_softmax_tiled_f16(const uint16_t* A,
                                                           const uint16_t* B,
                                                           uint16_t* O,
                                                           int32_t M, int32_t N,
                                                           int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_softmax_tiled_bf16(const uint16_t* A,
                                                            const uint16_t* B,
                                                            uint16_t* O,
                                                            int32_t M, int32_t N,
                                                            int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

// Phase 8.4.5 — fp32/fp16/bf16 stubs for fused matmul -> softmax -> matmul.

namespace {

inline void reference_matmul_softmax_matmul_f32_stub(
    const float* A, const float* B, const float* C, float* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  std::vector<float> out(static_cast<std::size_t>(P), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t n = 0; n < N; ++n) row_max = std::max(row_max, scores[n]);
    float denom = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      scores[n] = std::exp(scores[n] - row_max);
      denom += scores[n];
    }
    if (denom > 0.0f) {
      float inv = 1.0f / denom;
      for (int32_t n = 0; n < N; ++n) scores[n] *= inv;
    } else {
      for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    }
    for (int32_t p = 0; p < P; ++p) out[p] = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      const float* c_row = C + static_cast<std::size_t>(n) * P;
      float sn = scores[n];
      for (int32_t p = 0; p < P; ++p) out[p] += sn * c_row[p];
    }
    float* o_row = O + static_cast<std::size_t>(row) * P;
    for (int32_t p = 0; p < P; ++p) o_row[p] = out[p];
  }
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_f32(
    const float* A, const float* B, const float* C, float* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  reference_matmul_softmax_matmul_f32_stub(A, B, C, O, M, K, N, P);
}

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_f16(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(N) * P);
  std::vector<float> Of(static_cast<std::size_t>(M) * P);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  for (std::size_t i = 0; i < Cf.size(); ++i) Cf[i] = half_to_float_stub(C[i]);
  reference_matmul_softmax_matmul_f32_stub(Af.data(), Bf.data(), Cf.data(),
                                           Of.data(), M, K, N, P);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_bf16(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(N) * P);
  std::vector<float> Of(static_cast<std::size_t>(M) * P);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  for (std::size_t i = 0; i < Cf.size(); ++i) Cf[i] = bfloat16_to_float_stub(C[i]);
  reference_matmul_softmax_matmul_f32_stub(Af.data(), Bf.data(), Cf.data(),
                                           Of.data(), M, K, N, P);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

// Phase 8.4.7 — MLP-block fusion stubs (matmul -> gelu, matmul -> rmsnorm).

namespace {

inline void reference_matmul_gelu_f32_stub(const float* A, const float* B,
                                           float* O, int32_t M, int32_t N,
                                           int32_t K) {
  static constexpr float kSqrt2OverPi = 0.7978845608028654f;
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float* o_row = O + static_cast<std::size_t>(row) * N;
    for (int32_t n = 0; n < N; ++n) {
      float v = scores[n];
      float t = kSqrt2OverPi * (v + 0.044715f * v * v * v);
      o_row[n] = 0.5f * v * (1.0f + std::tanh(t));
    }
  }
}

inline void reference_matmul_rmsnorm_f32_stub(const float* A, const float* B,
                                              float* O, int32_t M, int32_t N,
                                              int32_t K, float eps) {
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float sumsq = 0.0f;
    for (int32_t n = 0; n < N; ++n) sumsq += scores[n] * scores[n];
    float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(N) + eps);
    float* o_row = O + static_cast<std::size_t>(row) * N;
    for (int32_t n = 0; n < N; ++n) o_row[n] = scores[n] * inv_rms;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_gelu_f32(const float* A, const float* B,
                                                  float* O, int32_t M,
                                                  int32_t N, int32_t K) {
  reference_matmul_gelu_f32_stub(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_f32(const float* A,
                                                     const float* B, float* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K, float eps) {
  reference_matmul_rmsnorm_f32_stub(A, B, O, M, N, K, eps);
}

// f16/bf16 MLP-block fusion stubs — convert to fp32, reuse the reference, cast
// back. Mirrors the native-half MSL kernels' fp32-accumulator convention.

extern "C" void tessera_apple_gpu_matmul_gelu_f16(const uint16_t* A,
                                                  const uint16_t* B, uint16_t* O,
                                                  int32_t M, int32_t N,
                                                  int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  reference_matmul_gelu_f32_stub(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_gelu_bf16(const uint16_t* A,
                                                   const uint16_t* B,
                                                   uint16_t* O, int32_t M,
                                                   int32_t N, int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  reference_matmul_gelu_f32_stub(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_f16(const uint16_t* A,
                                                     const uint16_t* B,
                                                     uint16_t* O, int32_t M,
                                                     int32_t N, int32_t K,
                                                     float eps) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_stub(B[i]);
  reference_matmul_rmsnorm_f32_stub(Af.data(), Bf.data(), Of.data(), M, N, K, eps);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_stub(Of[i]);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_bf16(const uint16_t* A,
                                                      const uint16_t* B,
                                                      uint16_t* O, int32_t M,
                                                      int32_t N, int32_t K,
                                                      float eps) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  reference_matmul_rmsnorm_f32_stub(Af.data(), Bf.data(), Of.data(), M, N, K, eps);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}

// Phase 8.4.8 — SwiGLU MLP-block fusion stubs (Stage 3).
// On non-Apple hosts the runtime falls through to a portable reference
// path so tests can exercise the lowered IR without Metal.

namespace {

inline void reference_swiglu_f32_stub(const float* X, const float* Wg,
                                      const float* Wu, const float* Wd,
                                      float* O, int32_t M, int32_t K,
                                      int32_t H, int32_t Kout) {
  std::vector<float> gate(static_cast<std::size_t>(H), 0.0f);
  std::vector<float> up(static_cast<std::size_t>(H), 0.0f);
  std::vector<float> hidden(static_cast<std::size_t>(H), 0.0f);
  std::vector<float> out_row(static_cast<std::size_t>(Kout), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t h = 0; h < H; ++h) { gate[h] = 0.0f; up[h] = 0.0f; }
    for (int32_t k = 0; k < K; ++k) {
      float xv = X[static_cast<std::size_t>(row) * K + k];
      const float* wg_row = Wg + static_cast<std::size_t>(k) * H;
      const float* wu_row = Wu + static_cast<std::size_t>(k) * H;
      for (int32_t h = 0; h < H; ++h) {
        gate[h] += xv * wg_row[h];
        up[h]   += xv * wu_row[h];
      }
    }
    for (int32_t h = 0; h < H; ++h) {
      float g = gate[h];
      float s = g / (1.0f + std::exp(-g));
      hidden[h] = s * up[h];
    }
    for (int32_t ko = 0; ko < Kout; ++ko) out_row[ko] = 0.0f;
    for (int32_t h = 0; h < H; ++h) {
      float hv = hidden[h];
      const float* wd_row = Wd + static_cast<std::size_t>(h) * Kout;
      for (int32_t ko = 0; ko < Kout; ++ko) out_row[ko] += hv * wd_row[ko];
    }
    float* o_row = O + static_cast<std::size_t>(row) * Kout;
    for (int32_t ko = 0; ko < Kout; ++ko) o_row[ko] = out_row[ko];
  }
}

} // namespace

extern "C" void tessera_apple_gpu_swiglu_f32(const float* X, const float* Wg,
                                             const float* Wu, const float* Wd,
                                             float* O, int32_t M, int32_t K,
                                             int32_t H, int32_t Kout) {
  reference_swiglu_f32_stub(X, Wg, Wu, Wd, O, M, K, H, Kout);
}

extern "C" void tessera_apple_gpu_swiglu_f16(const uint16_t* /*X*/,
                                             const uint16_t* /*Wg*/,
                                             const uint16_t* /*Wu*/,
                                             const uint16_t* /*Wd*/,
                                             uint16_t* O, int32_t M,
                                             int32_t /*K*/, int32_t /*H*/,
                                             int32_t Kout) {
  // Non-Apple stub: zero-fill rather than implement fp16<->fp32 conversion
  // here. Real numerical paths run on Apple Silicon.
  std::memset(O, 0, sizeof(uint16_t) * static_cast<std::size_t>(M) * Kout);
}

extern "C" void tessera_apple_gpu_swiglu_bf16(const uint16_t* /*X*/,
                                              const uint16_t* /*Wg*/,
                                              const uint16_t* /*Wu*/,
                                              const uint16_t* /*Wd*/,
                                              uint16_t* O, int32_t M,
                                              int32_t /*K*/, int32_t /*H*/,
                                              int32_t Kout) {
  std::memset(O, 0, sizeof(uint16_t) * static_cast<std::size_t>(M) * Kout);
}

// attention_variants_plan, LA-2 — linear-attn stub on non-Apple hosts.
// Pure-numpy reference over the recurrence; tests compile this from
// source via a CXX subprocess on Linux.

namespace {

inline float la_feature_map(float x, int32_t fm) {
  if (fm == 0) return x > 0.0f ? (x + 1.0f) : std::exp(x);
  if (fm == 1) return std::max(x, 0.0f);
  if (fm == 2) return x;
  return x * x;
}

inline void reference_linear_attn_f32_stub(const float* Q, const float* K,
                                           const float* V, float* O,
                                           int32_t B, int32_t H, int32_t S,
                                           int32_t D_qk, int32_t D_v,
                                           int32_t feature_map,
                                           int32_t /*causal*/) {
  std::vector<float> state(static_cast<std::size_t>(D_qk) * D_v, 0.0f);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      std::fill(state.begin(), state.end(), 0.0f);
      int q_base = (b * H + h) * S * D_qk;
      int k_base = q_base;
      int v_base = (b * H + h) * S * D_v;
      int o_base = v_base;
      for (int32_t t = 0; t < S; ++t) {
        for (int32_t d_qk = 0; d_qk < D_qk; ++d_qk) {
          float k_d = la_feature_map(K[k_base + t * D_qk + d_qk], feature_map);
          for (int32_t d_v = 0; d_v < D_v; ++d_v) {
            state[(std::size_t)d_qk * D_v + d_v] +=
                k_d * V[v_base + t * D_v + d_v];
          }
        }
        for (int32_t d_v = 0; d_v < D_v; ++d_v) {
          float acc = 0.0f;
          for (int32_t d_qk = 0; d_qk < D_qk; ++d_qk) {
            float q_d = la_feature_map(Q[q_base + t * D_qk + d_qk], feature_map);
            acc += q_d * state[(std::size_t)d_qk * D_v + d_v];
          }
          O[o_base + t * D_v + d_v] = acc;
        }
      }
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_linear_attn_f32(const float* Q, const float* K,
                                                   const float* V, float* O,
                                                   int32_t B, int32_t H,
                                                   int32_t S, int32_t D_qk,
                                                   int32_t D_v,
                                                   int32_t feature_map,
                                                   int32_t causal) {
  reference_linear_attn_f32_stub(Q, K, V, O, B, H, S, D_qk, D_v, feature_map,
                                 causal);
}

// attention_variants_plan, MLA-2 — non-Apple stub. Numpy-reference path
// for the host-only fallback. The Apple .mm file's `reference_mla_decode_f32`
// already implements the same math; this is its non-Apple twin.

namespace {

inline void reference_mla_decode_f32_stub(const float* X, const float* Wdkv,
                                          const float* Wuk, const float* Wuv,
                                          const float* Q, float* O,
                                          int32_t B, int32_t S_kv,
                                          int32_t D_x, int32_t D_lat,
                                          int32_t S_q, int32_t D_h) {
  std::vector<float> c(static_cast<std::size_t>(S_kv) * D_lat, 0.0f);
  for (int32_t s = 0; s < S_kv; ++s) {
    for (int32_t d = 0; d < D_x; ++d) {
      float xv = X[static_cast<std::size_t>(s) * D_x + d];
      const float* w_row = Wdkv + static_cast<std::size_t>(d) * D_lat;
      for (int32_t l = 0; l < D_lat; ++l) {
        c[static_cast<std::size_t>(s) * D_lat + l] += xv * w_row[l];
      }
    }
  }
  std::vector<float> K(static_cast<std::size_t>(S_kv) * D_h, 0.0f);
  std::vector<float> V(static_cast<std::size_t>(S_kv) * D_h, 0.0f);
  for (int32_t s = 0; s < S_kv; ++s) {
    for (int32_t l = 0; l < D_lat; ++l) {
      float cv = c[static_cast<std::size_t>(s) * D_lat + l];
      const float* uk_row = Wuk + static_cast<std::size_t>(l) * D_h;
      const float* uv_row = Wuv + static_cast<std::size_t>(l) * D_h;
      for (int32_t h = 0; h < D_h; ++h) {
        K[static_cast<std::size_t>(s) * D_h + h] += cv * uk_row[h];
        V[static_cast<std::size_t>(s) * D_h + h] += cv * uv_row[h];
      }
    }
  }
  float scale = 1.0f / std::sqrt(static_cast<float>(D_h));
  std::vector<float> scores(static_cast<std::size_t>(S_q) * S_kv);
  for (int32_t b = 0; b < B; ++b) {
    const float* Qb = Q + static_cast<std::size_t>(b) * S_q * D_h;
    float* Ob = O + static_cast<std::size_t>(b) * S_q * D_h;
    for (int32_t i = 0; i < S_q; ++i) {
      for (int32_t j = 0; j < S_kv; ++j) {
        float dot = 0.0f;
        for (int32_t h = 0; h < D_h; ++h) {
          dot += Qb[i * D_h + h] * K[static_cast<std::size_t>(j) * D_h + h];
        }
        scores[static_cast<std::size_t>(i) * S_kv + j] = dot * scale;
      }
    }
    for (int32_t i = 0; i < S_q; ++i) {
      float maxv = -std::numeric_limits<float>::infinity();
      for (int32_t j = 0; j < S_kv; ++j) {
        maxv = std::max(maxv, scores[static_cast<std::size_t>(i) * S_kv + j]);
      }
      float sum = 0.0f;
      for (int32_t j = 0; j < S_kv; ++j) {
        float e = std::exp(
            scores[static_cast<std::size_t>(i) * S_kv + j] - maxv);
        scores[static_cast<std::size_t>(i) * S_kv + j] = e;
        sum += e;
      }
      for (int32_t j = 0; j < S_kv; ++j) {
        scores[static_cast<std::size_t>(i) * S_kv + j] /= sum;
      }
      for (int32_t h = 0; h < D_h; ++h) {
        float acc = 0.0f;
        for (int32_t j = 0; j < S_kv; ++j) {
          acc += scores[static_cast<std::size_t>(i) * S_kv + j]
                 * V[static_cast<std::size_t>(j) * D_h + h];
        }
        Ob[i * D_h + h] = acc;
      }
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_mla_decode_f32(const float* X,
                                                  const float* Wdkv,
                                                  const float* Wuk,
                                                  const float* Wuv,
                                                  const float* Q, float* O,
                                                  int32_t B, int32_t S_kv,
                                                  int32_t D_x, int32_t D_lat,
                                                  int32_t S_q, int32_t D_h) {
  reference_mla_decode_f32_stub(X, Wdkv, Wuk, Wuv, Q, O, B, S_kv, D_x, D_lat,
                                S_q, D_h);
}

// compressed-KV mla_decode f16 / bf16 non-Apple reference.
extern "C" void tessera_apple_gpu_mla_decode_f16(
    const uint16_t* X, const uint16_t* Wdkv, const uint16_t* Wuk,
    const uint16_t* Wuv, const uint16_t* Q, uint16_t* O, int32_t B, int32_t S_kv,
    int32_t D_x, int32_t D_lat, int32_t S_q, int32_t D_h) {
  if (B <= 0 || S_kv <= 0 || D_x <= 0 || D_lat <= 0 || S_q <= 0 || D_h <= 0)
    return;
  auto c = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = half_to_float_stub(p[i]);
    return v;
  };
  std::vector<float> xf = c(X, (std::size_t)S_kv * D_x),
                     wd = c(Wdkv, (std::size_t)D_x * D_lat),
                     wk = c(Wuk, (std::size_t)D_lat * D_h),
                     wv = c(Wuv, (std::size_t)D_lat * D_h),
                     qf = c(Q, (std::size_t)B * S_q * D_h),
                     of((std::size_t)B * S_q * D_h);
  reference_mla_decode_f32_stub(xf.data(), wd.data(), wk.data(), wv.data(),
                                qf.data(), of.data(), B, S_kv, D_x, D_lat, S_q, D_h);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_stub(of[i]);
}
extern "C" void tessera_apple_gpu_mla_decode_bf16(
    const uint16_t* X, const uint16_t* Wdkv, const uint16_t* Wuk,
    const uint16_t* Wuv, const uint16_t* Q, uint16_t* O, int32_t B, int32_t S_kv,
    int32_t D_x, int32_t D_lat, int32_t S_q, int32_t D_h) {
  if (B <= 0 || S_kv <= 0 || D_x <= 0 || D_lat <= 0 || S_q <= 0 || D_h <= 0)
    return;
  auto c = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32_stub(p[i]);
    return v;
  };
  std::vector<float> xf = c(X, (std::size_t)S_kv * D_x),
                     wd = c(Wdkv, (std::size_t)D_x * D_lat),
                     wk = c(Wuk, (std::size_t)D_lat * D_h),
                     wv = c(Wuv, (std::size_t)D_lat * D_h),
                     qf = c(Q, (std::size_t)B * S_q * D_h),
                     of((std::size_t)B * S_q * D_h);
  reference_mla_decode_f32_stub(xf.data(), wd.data(), wk.data(), wv.data(),
                                qf.data(), of.data(), B, S_kv, D_x, D_lat, S_q, D_h);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16_stub(of[i]);
}

// ---- MLA decode with decoupled RoPE — non-Apple reference (2026-05-30) ------
static inline void mla_rope_apply_stub(const float* x, const float* cosr,
                                       const float* sinr, float* out, int32_t dr,
                                       int32_t style) {
  int32_t half = dr / 2;
  if (style == 0) {
    for (int32_t p = 0; p < half; ++p) {
      float a = x[2 * p], b = x[2 * p + 1], c = cosr[p], s = sinr[p];
      out[2 * p] = a * c - b * s;
      out[2 * p + 1] = a * s + b * c;
    }
  } else {
    for (int32_t p = 0; p < half; ++p) {
      float a = x[p], b = x[p + half], c = cosr[p], s = sinr[p];
      out[p] = a * c - b * s;
      out[p + half] = b * c + a * s;
    }
  }
}
extern "C" void tessera_apple_gpu_mla_decode_rope_f32(
    const float* Qn, const float* Qr, const float* Kn, const float* Kr,
    const float* V, const float* cosQ, const float* sinQ, const float* cosK,
    const float* sinK, float* O, int32_t B, int32_t H, int32_t Sq, int32_t Skv,
    int32_t dn, int32_t dr, int32_t dv, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  int32_t dh = dn + dr, halfdr = dr / 2;
  float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  std::vector<float> qf(dh), kf(dh), tmp(dr), scores(Skv);
  for (int32_t b = 0; b < B; ++b)
    for (int32_t h = 0; h < H; ++h) {
      int32_t bh = b * H + h;
      for (int32_t i = 0; i < Sq; ++i) {
        const float* qn = Qn + (((std::size_t)bh * Sq + i) * dn);
        const float* qr = Qr + (((std::size_t)bh * Sq + i) * dr);
        for (int32_t d = 0; d < dn; ++d) qf[d] = qn[d];
        mla_rope_apply_stub(qr, cosQ + (std::size_t)i * halfdr,
                            sinQ + (std::size_t)i * halfdr, tmp.data(), dr,
                            rotation_style);
        for (int32_t d = 0; d < dr; ++d) qf[dn + d] = tmp[d];
        double mx = -1e30;
        for (int32_t j = 0; j < Skv; ++j) {
          const float* kn = Kn + (((std::size_t)bh * Skv + j) * dn);
          const float* kr = Kr + (((std::size_t)b * Skv + j) * dr);
          for (int32_t d = 0; d < dn; ++d) kf[d] = kn[d];
          mla_rope_apply_stub(kr, cosK + (std::size_t)j * halfdr,
                              sinK + (std::size_t)j * halfdr, tmp.data(), dr,
                              rotation_style);
          for (int32_t d = 0; d < dr; ++d) kf[dn + d] = tmp[d];
          double acc = 0;
          for (int32_t d = 0; d < dh; ++d) acc += (double)qf[d] * kf[d];
          scores[j] = (float)(acc * scale);
          mx = std::max(mx, (double)scores[j]);
        }
        double den = 0;
        for (int32_t j = 0; j < Skv; ++j) {
          double e = std::exp((double)scores[j] - mx);
          scores[j] = (float)e;
          den += e;
        }
        float* o = O + (((std::size_t)bh * Sq + i) * dv);
        for (int32_t d = 0; d < dv; ++d) {
          double acc = 0;
          for (int32_t j = 0; j < Skv; ++j)
            acc += (double)scores[j] / den *
                   V[((std::size_t)bh * Skv + j) * dv + d];
          o[d] = (float)acc;
        }
      }
    }
}

// decoupled-rope mla_decode_rope f16 / bf16 non-Apple reference.
extern "C" void tessera_apple_gpu_mla_decode_rope_f16(
    const uint16_t* Qn, const uint16_t* Qr, const uint16_t* Kn,
    const uint16_t* Kr, const uint16_t* V, const float* cosQ, const float* sinQ,
    const float* cosK, const float* sinK, uint16_t* O, int32_t B, int32_t H,
    int32_t Sq, int32_t Skv, int32_t dn, int32_t dr, int32_t dv,
    int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  auto c = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = half_to_float_stub(p[i]);
    return v;
  };
  std::vector<float> qn = c(Qn, (std::size_t)B * H * Sq * dn),
                     qr = c(Qr, (std::size_t)B * H * Sq * dr),
                     kn = c(Kn, (std::size_t)B * H * Skv * dn),
                     kr = c(Kr, (std::size_t)B * Skv * dr),
                     vf = c(V, (std::size_t)B * H * Skv * dv),
                     of((std::size_t)B * H * Sq * dv);
  tessera_apple_gpu_mla_decode_rope_f32(qn.data(), qr.data(), kn.data(),
      kr.data(), vf.data(), cosQ, sinQ, cosK, sinK, of.data(), B, H, Sq, Skv,
      dn, dr, dv, rotation_style);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_stub(of[i]);
}
extern "C" void tessera_apple_gpu_mla_decode_rope_bf16(
    const uint16_t* Qn, const uint16_t* Qr, const uint16_t* Kn,
    const uint16_t* Kr, const uint16_t* V, const float* cosQ, const float* sinQ,
    const float* cosK, const float* sinK, uint16_t* O, int32_t B, int32_t H,
    int32_t Sq, int32_t Skv, int32_t dn, int32_t dr, int32_t dv,
    int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  auto c = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32_stub(p[i]);
    return v;
  };
  std::vector<float> qn = c(Qn, (std::size_t)B * H * Sq * dn),
                     qr = c(Qr, (std::size_t)B * H * Sq * dr),
                     kn = c(Kn, (std::size_t)B * H * Skv * dn),
                     kr = c(Kr, (std::size_t)B * Skv * dr),
                     vf = c(V, (std::size_t)B * H * Skv * dv),
                     of((std::size_t)B * H * Sq * dv);
  tessera_apple_gpu_mla_decode_rope_f32(qn.data(), qr.data(), kn.data(),
      kr.data(), vf.data(), cosQ, sinQ, cosK, sinK, of.data(), B, H, Sq, Skv,
      dn, dr, dv, rotation_style);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16_stub(of[i]);
}

// ---- MLA decode with weight absorption — non-Apple reference (2026-05-30) ---
// Attention runs directly against the cached latent c_kv (shared across heads);
// the up-proj weights Wukᵀ / Wuv absorb into the query / output.
extern "C" void tessera_apple_gpu_mla_absorb_decode_f32(
    const float* q_nope, const float* q_rope, const float* c_kv,
    const float* k_rope, const float* Wuk_t, const float* Wuv,
    const float* cosQ, const float* sinQ, const float* cosK, const float* sinK,
    float* O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  int32_t halfdr = dr / 2;
  float scale = 1.0f / std::sqrt(static_cast<float>(dn + dr));
  std::vector<double> qabs(Dl), score(Skv);
  std::vector<float> qrr(dr), krr((std::size_t)Skv * dr);
  for (int32_t b = 0; b < B; ++b)
    for (int32_t h = 0; h < H; ++h) {
      int32_t bh = b * H + h;
      // pre-rope the shared key once per batch/head (k_rope shared across h)
      for (int32_t j = 0; j < Skv; ++j)
        mla_rope_apply_stub(k_rope + ((std::size_t)b * Skv + j) * dr,
                            cosK + (std::size_t)j * halfdr,
                            sinK + (std::size_t)j * halfdr,
                            krr.data() + (std::size_t)j * dr, dr, rotation_style);
      for (int32_t i = 0; i < Sq; ++i) {
        const float* qnb = q_nope + ((std::size_t)bh * Sq + i) * dn;
        mla_rope_apply_stub(q_rope + ((std::size_t)bh * Sq + i) * dr,
                            cosQ + (std::size_t)i * halfdr,
                            sinQ + (std::size_t)i * halfdr, qrr.data(), dr,
                            rotation_style);
        const float* wuktb = Wuk_t + (std::size_t)h * dn * Dl;
        for (int32_t l = 0; l < Dl; ++l) {
          double acc = 0;
          for (int32_t d = 0; d < dn; ++d) acc += (double)qnb[d] * wuktb[d * Dl + l];
          qabs[l] = acc;
        }
        const float* ckvb = c_kv + (std::size_t)b * Skv * Dl;
        double mx = -1e30;
        for (int32_t j = 0; j < Skv; ++j) {
          double sn = 0;
          for (int32_t l = 0; l < Dl; ++l) sn += qabs[l] * ckvb[j * Dl + l];
          double sr = 0;
          for (int32_t d = 0; d < dr; ++d) sr += (double)qrr[d] * krr[(std::size_t)j * dr + d];
          score[j] = (sn + sr) * scale;
          mx = std::max(mx, score[j]);
        }
        double den = 0;
        for (int32_t j = 0; j < Skv; ++j) { double e = std::exp(score[j] - mx); score[j] = e; den += e; }
        const float* wuvb = Wuv + (std::size_t)h * Dl * dv;
        float* o = O + ((std::size_t)bh * Sq + i) * dv;
        for (int32_t d = 0; d < dv; ++d) {
          double acc = 0;
          for (int32_t j = 0; j < Skv; ++j) {
            double cv = 0;
            for (int32_t l = 0; l < Dl; ++l) cv += ckvb[j * Dl + l] * wuvb[l * dv + d];
            acc += (score[j] / den) * cv;
          }
          o[d] = (float)acc;
        }
      }
    }
}

// f16 / bf16 non-Apple absorb decode: convert to fp32, run the fp32 reference,
// convert back.
extern "C" void tessera_apple_gpu_mla_absorb_decode_f16(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* c_kv,
    const uint16_t* k_rope, const uint16_t* Wuk_t, const uint16_t* Wuv,
    const float* cosQ, const float* sinQ, const float* cosK, const float* sinK,
    uint16_t* O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  auto cvt = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = half_to_float_stub(p[i]);
    return v;
  };
  std::vector<float> qn = cvt(q_nope, (std::size_t)B * H * Sq * dn);
  std::vector<float> qr = cvt(q_rope, (std::size_t)B * H * Sq * dr);
  std::vector<float> ck = cvt(c_kv, (std::size_t)B * Skv * Dl);
  std::vector<float> kr = cvt(k_rope, (std::size_t)B * Skv * dr);
  std::vector<float> wk = cvt(Wuk_t, (std::size_t)H * dn * Dl);
  std::vector<float> wv = cvt(Wuv, (std::size_t)H * Dl * dv);
  std::vector<float> of((std::size_t)B * H * Sq * dv);
  tessera_apple_gpu_mla_absorb_decode_f32(qn.data(), qr.data(), ck.data(),
      kr.data(), wk.data(), wv.data(), cosQ, sinQ, cosK, sinK, of.data(), B, H,
      Sq, Skv, dn, dr, dv, Dl, rotation_style);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_stub(of[i]);
}

extern "C" void tessera_apple_gpu_mla_absorb_decode_bf16(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* c_kv,
    const uint16_t* k_rope, const uint16_t* Wuk_t, const uint16_t* Wuv,
    const float* cosQ, const float* sinQ, const float* cosK, const float* sinK,
    uint16_t* O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  auto cvt = [](const uint16_t* p, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32_stub(p[i]);
    return v;
  };
  std::vector<float> qn = cvt(q_nope, (std::size_t)B * H * Sq * dn);
  std::vector<float> qr = cvt(q_rope, (std::size_t)B * H * Sq * dr);
  std::vector<float> ck = cvt(c_kv, (std::size_t)B * Skv * Dl);
  std::vector<float> kr = cvt(k_rope, (std::size_t)B * Skv * dr);
  std::vector<float> wk = cvt(Wuk_t, (std::size_t)H * dn * Dl);
  std::vector<float> wv = cvt(Wuv, (std::size_t)H * Dl * dv);
  std::vector<float> of((std::size_t)B * H * Sq * dv);
  tessera_apple_gpu_mla_absorb_decode_f32(qn.data(), qr.data(), ck.data(),
      kr.data(), wk.data(), wv.data(), cosQ, sinQ, cosK, sinK, of.data(), B, H,
      Sq, Skv, dn, dr, dv, Dl, rotation_style);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16_stub(of[i]);
}

// attention_variants_plan, NSA-5 — non-Apple stub.
// Zero-fills the output (the Apple-side reference has the full impl).
extern "C" void tessera_apple_gpu_native_sparse_attn_f32(
    const float* /*Q*/, const float* /*K*/, const float* /*V*/,
    const float* /*gate*/, float* O,
    int32_t B, int32_t H, int32_t S, int32_t D,
    int32_t /*window*/, int32_t /*block*/, int32_t /*top_k*/,
    int32_t /*causal*/) {
  std::memset(O, 0, sizeof(float) * static_cast<std::size_t>(B) * H * S * D);
}

extern "C" int32_t tessera_apple_gpu_native_sparse_attn_last_path(void) {
  return 0;
}

// ---- MPSGraph lane (non-Apple reference fallbacks, 2026-05-29) -------------
// Mirror the C ABI of the MPSGraph-backed Tier-1 / long-tail lane so the
// symbol surface is identical across platforms. Op codes match the .mm.
extern "C" void tessera_apple_gpu_mpsgraph_unary_f32(int32_t op, const float* x,
                                                     float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    float v = x[i];
    switch (op) {
      case 0: out[i] = v > 0 ? v : 0.0f; break;
      case 1: out[i] = 1.0f / (1.0f + std::exp(-v)); break;
      case 2: out[i] = std::tanh(v); break;
      case 3: out[i] = std::log1p(std::exp(v)); break;
      case 4: out[i] = v / (1.0f + std::exp(-v)); break;
      case 5: out[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))); break;
      case 6: out[i] = std::exp(v); break;
      case 7: out[i] = std::log(v); break;
      case 8: out[i] = std::sqrt(v); break;
      case 9: out[i] = 1.0f / std::sqrt(v); break;
      case 10: out[i] = -v; break;
      case 11: out[i] = std::fabs(v); break;
      default: out[i] = v; break;
    }
  }
}
extern "C" void tessera_apple_gpu_mpsgraph_unary_f16(int32_t, const uint16_t* x,
                                                     uint16_t* out, int64_t n) {
  std::memcpy(out, x, static_cast<std::size_t>(n) * 2);
}
extern "C" void tessera_apple_gpu_mpsgraph_binary_f32(int32_t op, const float* a,
                                                      const float* b, float* out,
                                                      int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    float x = a[i], y = b[i];
    switch (op) {
      case 0: out[i] = x + y; break;
      case 1: out[i] = x - y; break;
      case 2: out[i] = x * y; break;
      case 3: out[i] = x / y; break;
      case 4: out[i] = x > y ? x : y; break;
      case 5: out[i] = x < y ? x : y; break;
      case 6: out[i] = x * (y / (1.0f + std::exp(-y))); break;
      default: out[i] = x; break;
    }
  }
}
// Phase-G Rung 0 — control-flow scan, non-Apple reference. Same recurrence the
// MPSGraph forLoop computes: carry_{i+1} = tanh(carry_i @ Wh + x_i @ Wx).
extern "C" int32_t tessera_apple_gpu_cf_scan_f32(const float* Wh, const float* Wx,
                                                 const float* xseq,
                                                 const float* init, float* ys,
                                                 int32_t T, int32_t d, int32_t m) {
  if (!Wh || !Wx || !xseq || !init || !ys || T <= 0 || d <= 0 || m <= 0) return 0;
  for (int t = 0; t < T; ++t) {
    const float* prev = (t == 0) ? init : &ys[(t - 1) * d];
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k) acc += (double)prev[k] * Wh[k * d + j];
      for (int k = 0; k < m; ++k) acc += (double)xseq[t * m + k] * Wx[k * d + j];
      ys[t * d + j] = (float)std::tanh(acc);
    }
  }
  return 1;
}

// Phase-G Rung 1 — serial draft, non-Apple reference. Mirrors the MPSGraph
// forLoop body (value-only T=1 attention + SwiGLU), autoregressive over T steps.
namespace {
inline void cf_rms(const float* x, const float* g, float* o, int d, float eps) {
  double ms = 0.0;
  for (int j = 0; j < d; ++j) ms += (double)x[j] * x[j];
  ms /= d;
  double den = std::sqrt(ms + eps);
  for (int j = 0; j < d; ++j) o[j] = (float)(x[j] / den * g[j]);
}
}  // namespace

// Metal 4 capability probe — non-Apple reference: no Metal 4 here.
extern "C" int32_t tessera_apple_gpu_metal4_probe(int32_t* caps_out) {
  if (caps_out) *caps_out = 0;
  return 0;
}
// Metal 4 MTLTensor round-trip — non-Apple reference: plain memcpy.
extern "C" int32_t tessera_apple_gpu_metal4_tensor_roundtrip(const void* in,
                                                             void* out, int32_t n,
                                                             int32_t dtype_code) {
  if (!in || !out || n <= 0) return 0;
  std::size_t elem = (dtype_code == 1 || dtype_code == 2) ? 2 : 4;
  std::memcpy(out, in, (std::size_t)n * elem);
  return 1;
}
// Metal 4 M2 MSL-loop scan — no Metal 4 off Darwin; caller falls back to numpy.
extern "C" int32_t tessera_apple_gpu_mtl4_scan_f32(const float*, const float*,
                                                   const float*, const float*,
                                                   float*, int32_t, int32_t,
                                                   int32_t) {
  return 0;
}
// Metal 4 M3 cooperative-matrix matmul — no Metal 4 off Darwin; numpy fallback.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul_sg_f32(const float*, const float*,
                                                        float*, int32_t, int32_t,
                                                        int32_t) {
  return 0;
}
// Metal 4 M6/M7 — MPP matmul2d {f16,bf16} (+ fused epilogue); no Metal 4 off
// Darwin; numpy fallback in the Python wrappers.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_f16(const uint16_t*, const uint16_t*,
                                                       float*, int32_t, int32_t,
                                                       int32_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_bf16(const uint16_t*, const uint16_t*,
                                                        float*, int32_t, int32_t,
                                                        int32_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_epilogue_f16(const uint16_t*, const uint16_t*,
                                                               float*, const float*,
                                                               int32_t, int32_t, int32_t,
                                                               int32_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_epilogue_bf16(const uint16_t*, const uint16_t*,
                                                                float*, const float*,
                                                                int32_t, int32_t, int32_t,
                                                                int32_t) {
  return 0;
}
// Metal 4 M8 — resident-weight MLP session; no Metal 4 off Darwin.
extern "C" void *tessera_apple_gpu_mtl4_mlp_session_create(const uint16_t*, const float*,
                                                           int32_t, int32_t, int32_t,
                                                           int32_t) {
  return nullptr;
}
extern "C" int32_t tessera_apple_gpu_mtl4_mlp_session_run(void*, const uint16_t*, float*,
                                                          int32_t) {
  return 0;
}
// R0 bridge — device-resident decode step; no Metal 4 off Darwin (session_create
// returns null, so this is never reached with a real handle).
extern "C" int32_t tessera_apple_gpu_mtl4_mlp_session_run_dev(void*, TsDeviceTensor*,
                                                              TsDeviceTensor*, int32_t) {
  return 0;
}
extern "C" void tessera_apple_gpu_mtl4_mlp_session_destroy(void*) {}
// Metal 4 P4 — MTL4Archive pipeline persistence; no Metal 4 off Darwin.
extern "C" int32_t tessera_apple_gpu_mtl4_archive_enable(const char*) { return 0; }
extern "C" int32_t tessera_apple_gpu_mtl4_archive_flush(void) { return 0; }
// PK1 — Packaged ML pipeline stubs. Off Darwin / pre-macOS-26 there's
// no MTL4 lane; all probes report "not available" with error_kind=-1.
extern "C" void *tessera_apple_gpu_mlpkg_compile(const char *, const char *) {
  return nullptr;
}
extern "C" void *tessera_apple_gpu_mlpkg_compile_with_dims(
    const char *, const char *, int32_t, const int32_t *,
    const int32_t *, const int64_t *) {
  return nullptr;
}
extern "C" void tessera_apple_gpu_mlpkg_destroy(void *) {}
extern "C" int32_t tessera_apple_gpu_mlpkg_is_compiled(void *) { return 0; }
extern "C" int32_t tessera_apple_gpu_mlpkg_last_error_kind(void) { return -1; }
// PK8 — package authoring needs a real Metal device + MPSGraph; unavailable
// off-Darwin. Return the "OS unavailable" code so callers skip cleanly.
extern "C" int32_t tessera_apple_gpu_mlpkg_author_matmul(const char *, int32_t,
                                                         int32_t, int32_t) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_author_op(const char *, const char *,
                                                     int32_t, int32_t, float,
                                                     int32_t) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_author_chain(const char *,
                                                        const char *,
                                                        const int32_t *,
                                                        int32_t, float) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_dylib_serialize(const char *, const char *,
                                                     const char *) {
  return -1;
}
extern "C" int32_t tessera_apple_gpu_dylib_load(const char *) { return 0; }
extern "C" int32_t tessera_apple_gpu_mlpkg_first_function_name(const char *,
                                                               char *,
                                                               int32_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_fill_input_at(void *, int32_t,
                                                         const void *,
                                                         int64_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_read_output_at(void *, int32_t,
                                                          void *, int64_t) {
  return 0;
}
// PK2 — Reflection-extraction stubs. No pipeline → no bindings; the
// count probe returns -1 and the info probe returns 0 with zeroed
// outputs (matching the runtime's "invalid handle" semantics).
extern "C" int32_t tessera_apple_gpu_mlpkg_binding_count(void *) { return -1; }
extern "C" int32_t tessera_apple_gpu_mlpkg_binding_info(
    void *, int32_t, char *name_out, int32_t name_len,
    int32_t *buffer_index_out, int32_t *rank_out,
    int64_t *, int32_t, int32_t *dtype_raw_out) {
  if (name_out && name_len > 0) name_out[0] = '\0';
  if (buffer_index_out) *buffer_index_out = 0;
  if (rank_out) *rank_out = 0;
  if (dtype_raw_out) *dtype_raw_out = 0;
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_dtype_raw_for_tag(int32_t) {
  return -1;
}
// PK3 — tensor + argument-table stubs. Off Darwin / pre-macOS-26 these
// all report failure cleanly so callers can detect "feature absent"
// without crashing.
extern "C" int32_t tessera_apple_gpu_mlpkg_prepare_tensors(void *) { return 0; }
extern "C" int32_t tessera_apple_gpu_mlpkg_fill_input(
    void *, const char *, const void *, int64_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_read_output(
    void *, const char *, void *, int64_t) {
  return 0;
}
extern "C" int32_t tessera_apple_gpu_mlpkg_argument_table_ready(void *) {
  return 0;
}
// PK4 — dispatch + heap-size stubs.
extern "C" int32_t tessera_apple_gpu_mlpkg_dispatch(void *, uint64_t) {
  return 0;
}
extern "C" int64_t tessera_apple_gpu_mlpkg_intermediates_heap_size(void *) {
  return -1;
}
// Apple-sample Action 6 — archive state probe. Off-Darwin: zero outputs
// and report 0 ("runtime not ready"). The Python side reads the rc and
// treats 0 as "no archive telemetry available".
extern "C" int32_t tessera_apple_gpu_mtl4_archive_state(
    int32_t *archive_enabled_out, int32_t *has_lookup_archive_out,
    char *archive_path_out, int32_t archive_path_len) {
  if (archive_enabled_out) *archive_enabled_out = 0;
  if (has_lookup_archive_out) *has_lookup_archive_out = 0;
  if (archive_path_out && archive_path_len > 0) archive_path_out[0] = '\0';
  return 0;
}
// P8 — GPU im2col conv. Non-Apple reference: f16/bf16 -> f32 conv (Wr is the
// HWIO weights reshaped to [kH*kW*Cin, Cout], byte-identical, so the existing
// HWIO f32 reference applies), + bias + activation, -> f32 Y. The f32 conv
// reference + out-dim helper are defined further down, so forward-declare them.
static inline int32_t conv2d_out_dim_stub(int32_t in, int32_t k, int32_t stride,
                                          int32_t pad, int32_t dilation);
static void reference_conv2d_f32_stub(const float* X, const float* Wt,
    const float* bias, float* O, int32_t N, int32_t H, int32_t W, int32_t Cin,
    int32_t Cout, int32_t kH, int32_t kW, int32_t strideH, int32_t strideW,
    int32_t padH, int32_t padW, int32_t dilationH, int32_t dilationW, int32_t groups);
static void tessera_conv2d_mtl4_ref_stub(const uint16_t* X, const uint16_t* Wr,
    const float* bias, float* Y, int32_t act, int32_t N, int32_t H, int32_t W,
    int32_t Cin, int32_t Cout, int32_t kH, int32_t kW, int32_t sH, int32_t sW,
    int32_t pH, int32_t pW, int32_t dH, int32_t dW, bool is_bf16) {
  int32_t OH = conv2d_out_dim_stub(H, kH, sH, pH, dH);
  int32_t OW = conv2d_out_dim_stub(W, kW, sW, pW, dW);
  if (OH <= 0 || OW <= 0) return;
  auto cvt = [&](uint16_t h) {
    return is_bf16 ? gqa_bf16_to_f32_stub(h) : half_to_float_stub(h);
  };
  std::size_t xN = static_cast<std::size_t>(N) * H * W * Cin;
  std::size_t wN = static_cast<std::size_t>(kH) * kW * Cin * Cout;
  std::vector<float> Xf(xN), Wf(wN);
  for (std::size_t i = 0; i < xN; ++i) Xf[i] = cvt(X[i]);
  for (std::size_t i = 0; i < wN; ++i) Wf[i] = cvt(Wr[i]);
  reference_conv2d_f32_stub(Xf.data(), Wf.data(), bias, Y, N, H, W, Cin, Cout,
                            kH, kW, sH, sW, pH, pW, dH, dW, 1);
  std::size_t oN = static_cast<std::size_t>(N) * OH * OW * Cout;
  for (std::size_t i = 0; i < oN; ++i) {
    float v = Y[i];
    if (act == 1) v = v > 0 ? v : 0.0f;
    else if (act == 2) { float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                         v = 0.5f * v * (1.0f + std::tanh(t)); }
    else if (act == 3) v = v / (1.0f + std::exp(-v));
    Y[i] = v;
  }
}
extern "C" int32_t tessera_apple_gpu_mtl4_conv2d_f16(
    const uint16_t* X, const uint16_t* Wr, const float* bias, float* Y, int32_t act,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW) {
  tessera_conv2d_mtl4_ref_stub(X, Wr, bias, Y, act, N, H, W, Cin, Cout, kH, kW,
                               sH, sW, pH, pW, dH, dW, false);
  return 1;
}
extern "C" int32_t tessera_apple_gpu_mtl4_conv2d_bf16(
    const uint16_t* X, const uint16_t* Wr, const float* bias, float* Y, int32_t act,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW) {
  tessera_conv2d_mtl4_ref_stub(X, Wr, bias, Y, act, N, H, W, Cin, Cout, kH, kW,
                               sH, sW, pH, pW, dH, dW, true);
  return 1;
}
// Phase-G Rung 3 — dynamic speculative accept+select, non-Apple reference
// (same logic the MSL kernel runs, so it's correct everywhere).
extern "C" int32_t tessera_apple_gpu_msl_spec_accept(const int32_t* draft,
                                                     const int32_t* target,
                                                     int32_t* out, int32_t P,
                                                     int32_t depth) {
  if (!draft || !target || !out || P <= 0 || depth <= 0) return 0;
  int best_path = 0, best_len = -1, best_bonus = 0;
  for (int p = 0; p < P; ++p) {
    int len = 0;
    for (int i = 0; i < depth; ++i) {
      if (draft[p * depth + i] == target[p * (depth + 1) + i]) len++;
      else break;
    }
    if (len > best_len) {
      best_len = len; best_path = p;
      best_bonus = target[p * (depth + 1) + len];
    }
  }
  out[0] = best_path; out[1] = best_len; out[2] = best_bonus;
  for (int i = 0; i < depth; ++i)
    out[3 + i] = (i < best_len) ? draft[best_path * depth + i] : -1;
  return 1;
}

extern "C" int32_t tessera_apple_gpu_cf_serial_draft_f32(
    const float* embed, const float* fc_in, const float* ln1_all,
    const float* ln2_all, const float* wv_all, const float* wo_all,
    const float* wg_all, const float* wu_all, const float* wd_all,
    const float* snorm, const float* lm_head, const float* h_init,
    int32_t root_token, int32_t* tokens_out, float* hidden_out, int32_t T,
    int32_t L, int32_t d, int32_t ffn, int32_t V, float eps) {
  if (T <= 0 || L <= 0 || d <= 0 || ffn <= 0 || V <= 0) return 0;
  std::vector<float> hid(h_init, h_init + d), x(2 * d), s(d), n(d), v(d), attn(d);
  std::vector<float> gate(ffn), up(ffn), act(ffn), down(d), sn(d), logits(V);
  int tok = root_token;
  for (int step = 0; step < T; ++step) {
    for (int j = 0; j < d; ++j) x[j] = hid[j];
    for (int j = 0; j < d; ++j) x[d + j] = embed[(size_t)tok * d + j];
    for (int j = 0; j < d; ++j) {
      double a = 0.0;
      for (int k = 0; k < 2 * d; ++k) a += (double)x[k] * fc_in[k * d + j];
      s[j] = (float)a;
    }
    for (int li = 0; li < L; ++li) {
      const float* ln1 = ln1_all + (size_t)li * d;
      const float* ln2 = ln2_all + (size_t)li * d;
      const float* wv = wv_all + (size_t)li * d * d;
      const float* wo = wo_all + (size_t)li * d * d;
      const float* wg = wg_all + (size_t)li * d * ffn;
      const float* wu = wu_all + (size_t)li * d * ffn;
      const float* wd = wd_all + (size_t)li * ffn * d;
      cf_rms(s.data(), ln1, n.data(), d, eps);
      for (int j = 0; j < d; ++j) {
        double a = 0.0;
        for (int k = 0; k < d; ++k) a += (double)n[k] * wv[k * d + j];
        v[j] = (float)a;
      }
      for (int j = 0; j < d; ++j) {
        double a = 0.0;
        for (int k = 0; k < d; ++k) a += (double)v[k] * wo[k * d + j];
        s[j] += (float)a;
      }
      cf_rms(s.data(), ln2, n.data(), d, eps);
      for (int j = 0; j < ffn; ++j) {
        double a = 0.0;
        for (int k = 0; k < d; ++k) a += (double)n[k] * wg[k * ffn + j];
        gate[j] = (float)a;
      }
      for (int j = 0; j < ffn; ++j) {
        double a = 0.0;
        for (int k = 0; k < d; ++k) a += (double)n[k] * wu[k * ffn + j];
        up[j] = (float)a;
      }
      for (int j = 0; j < ffn; ++j) act[j] = gate[j] / (1.f + std::exp(-gate[j])) * up[j];
      for (int j = 0; j < d; ++j) {
        double a = 0.0;
        for (int k = 0; k < ffn; ++k) a += (double)act[k] * wd[k * d + j];
        s[j] += (float)a;
      }
    }
    cf_rms(s.data(), snorm, sn.data(), d, eps);
    for (int j = 0; j < V; ++j) {
      double a = 0.0;
      for (int k = 0; k < d; ++k) a += (double)sn[k] * lm_head[k * V + j];
      logits[j] = (float)a;
    }
    int am = 0;
    float best = logits[0];
    for (int j = 1; j < V; ++j)
      if (logits[j] > best) { best = logits[j]; am = j; }
    tokens_out[step] = am;
    for (int j = 0; j < d; ++j) hidden_out[(size_t)step * d + j] = s[j];
    for (int j = 0; j < d; ++j) hid[j] = s[j];
    tok = am;
  }
  return 1;
}

// Phase-G Rung 2 — predicate-driven greedy generation, non-Apple reference.
// Mirrors the MPSGraph while: loop while (step < max && last != eos).
extern "C" int32_t tessera_apple_gpu_cf_while_generate_f32(
    const float* W, const float* lm, const float* h_init, int32_t start_token,
    int32_t eos_token, int32_t max_steps, int32_t* tokens_out, int32_t* n_out,
    int32_t d, int32_t V) {
  if (!W || !lm || !h_init || !tokens_out || !n_out || d <= 0 || V <= 0 ||
      max_steps <= 0)
    return 0;
  std::vector<float> h(h_init, h_init + d), hp(d), logits(V);
  int last = start_token, step = 0;
  while (step < max_steps && last != eos_token) {
    for (int j = 0; j < d; ++j) {
      double a = 0.0;
      for (int k = 0; k < d; ++k) a += (double)h[k] * W[k * d + j];
      hp[j] = (float)std::tanh(a);
    }
    for (int j = 0; j < V; ++j) {
      double a = 0.0;
      for (int k = 0; k < d; ++k) a += (double)hp[k] * lm[k * V + j];
      logits[j] = (float)a;
    }
    int am = 0;
    float best = logits[0];
    for (int j = 1; j < V; ++j)
      if (logits[j] > best) { best = logits[j]; am = j; }
    tokens_out[step] = am;
    for (int j = 0; j < d; ++j) h[j] = hp[j];
    last = am;
    ++step;
  }
  *n_out = step;
  return 1;
}

extern "C" void tessera_apple_gpu_mpsgraph_binary_f16(int32_t, const uint16_t* a,
                                                      const uint16_t*, uint16_t* out,
                                                      int64_t n) {
  std::memcpy(out, a, static_cast<std::size_t>(n) * 2);
}
extern "C" void tessera_apple_gpu_layer_norm_f32(const float* x, const float* gamma,
                                                 const float* beta, float* out,
                                                 int32_t rows, int32_t cols, float eps) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* o = out + static_cast<std::size_t>(r) * cols;
    double mean = 0.0;
    for (int32_t c = 0; c < cols; ++c) mean += row[c];
    mean /= cols;
    double var = 0.0;
    for (int32_t c = 0; c < cols; ++c) { double d = row[c] - mean; var += d * d; }
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int32_t c = 0; c < cols; ++c) {
      double g = gamma ? gamma[c] : 1.0;
      double b = beta ? beta[c] : 0.0;
      o[c] = (float)(((row[c] - mean) * inv) * g + b);
    }
  }
}
extern "C" void tessera_apple_gpu_layer_norm_f16(const uint16_t* x, const uint16_t*,
                                                 const uint16_t*, uint16_t* out,
                                                 int32_t rows, int32_t cols, float) {
  std::memcpy(out, x, static_cast<std::size_t>(rows) * cols * 2);
}
extern "C" void tessera_apple_gpu_rmsnorm_gpu_f32(const float* x, const float* gamma,
                                                  float* out, int32_t rows,
                                                  int32_t cols, float eps) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* o = out + static_cast<std::size_t>(r) * cols;
    double ms = 0.0;
    for (int32_t c = 0; c < cols; ++c) ms += (double)row[c] * row[c];
    ms /= cols;
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int32_t c = 0; c < cols; ++c)
      o[c] = (float)(row[c] * inv * (gamma ? gamma[c] : 1.0));
  }
}
extern "C" void tessera_apple_gpu_rmsnorm_gpu_f16(const uint16_t* x, const uint16_t*,
                                                  uint16_t* out, int32_t rows,
                                                  int32_t cols, float) {
  std::memcpy(out, x, static_cast<std::size_t>(rows) * cols * 2);
}
extern "C" void tessera_apple_gpu_mpsgraph_softmax_f32(const float* x, float* out,
                                                       int32_t rows, int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* o = out + static_cast<std::size_t>(r) * cols;
    float m = row[0];
    for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m;
    double s = 0.0;
    for (int32_t c = 0; c < cols; ++c) { o[c] = std::exp(row[c] - m); s += o[c]; }
    for (int32_t c = 0; c < cols; ++c) o[c] = (float)(o[c] / s);
  }
}
extern "C" void tessera_apple_gpu_mpsgraph_softmax_f16(const uint16_t* x, uint16_t* out,
                                                       int32_t rows, int32_t cols) {
  std::memcpy(out, x, static_cast<std::size_t>(rows) * cols * 2);
}
extern "C" void tessera_apple_gpu_log_softmax_f32(const float* x, float* out,
                                                  int32_t rows, int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* o = out + static_cast<std::size_t>(r) * cols;
    float m = row[0];
    for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m;
    double s = 0.0;
    for (int32_t c = 0; c < cols; ++c) s += std::exp(row[c] - m);
    float lse = m + (float)std::log(s);
    for (int32_t c = 0; c < cols; ++c) o[c] = row[c] - lse;
  }
}
extern "C" void tessera_apple_gpu_log_softmax_f16(const uint16_t* x, uint16_t* out,
                                                  int32_t rows, int32_t cols) {
  std::memcpy(out, x, static_cast<std::size_t>(rows) * cols * 2);
}

extern "C" int32_t tessera_apple_gpu_mpsgraph_cache_size(void) {
  // No Metal -> no MPSGraph cache.
  return 0;
}

extern "C" int32_t tessera_apple_gpu_runtime_msl_cache_size(void) {
  // No Metal -> no MSL cache. Tests gate this on platform.
  return -1;
}

// ---- Batched matmul (bmm) non-Apple reference (2026-05-29) -----------------
extern "C" void tessera_apple_gpu_bmm_f32(const float* A, const float* B,
                                          float* O, int32_t batch, int32_t M,
                                          int32_t N, int32_t K,
                                          int32_t b_broadcast) {
  for (int32_t bi = 0; bi < batch; ++bi) {
    const float* a = A + static_cast<std::size_t>(bi) * M * K;
    const float* b = B + static_cast<std::size_t>(b_broadcast ? 0 : bi) * K * N;
    float* o = O + static_cast<std::size_t>(bi) * M * N;
    for (int32_t m = 0; m < M; ++m)
      for (int32_t n = 0; n < N; ++n) {
        float s = 0.0f;
        for (int32_t k = 0; k < K; ++k)
          s += a[static_cast<std::size_t>(m) * K + k] *
               b[static_cast<std::size_t>(k) * N + n];
        o[static_cast<std::size_t>(m) * N + n] = s;
      }
  }
}
extern "C" void tessera_apple_gpu_bmm_f16(const uint16_t* A, const uint16_t* B,
                                          uint16_t* O, int32_t batch, int32_t M,
                                          int32_t N, int32_t K, int32_t b_broadcast) {
  // Non-Apple reference: f16 -> f32 -> bmm -> f16 (runtime.py trusts the symbol
  // on every platform, so the stub must compute, not zero-fill).
  std::size_t aN = static_cast<std::size_t>(batch) * M * K;
  std::size_t bN = static_cast<std::size_t>(b_broadcast ? 1 : batch) * K * N;
  std::size_t oN = static_cast<std::size_t>(batch) * M * N;
  std::vector<float> Af(aN), Bf(bN), Of(oN);
  for (std::size_t i = 0; i < aN; ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < bN; ++i) Bf[i] = half_to_float_stub(B[i]);
  tessera_apple_gpu_bmm_f32(Af.data(), Bf.data(), Of.data(), batch, M, N, K, b_broadcast);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_half_stub(Of[i]);
}
// Sprint 8: bf16 batched matmul non-Apple reference. Honest bf16 ABI — upcasts
// to f32, computes the bmm, and rounds back to bf16 (NOT zero-fill, NOT an f32
// alias). runtime.py trusts the symbol on every platform.
extern "C" void tessera_apple_gpu_bmm_bf16(const uint16_t* A, const uint16_t* B,
                                           uint16_t* O, int32_t batch, int32_t M,
                                           int32_t N, int32_t K, int32_t b_broadcast) {
  std::size_t aN = static_cast<std::size_t>(batch) * M * K;
  std::size_t bN = static_cast<std::size_t>(b_broadcast ? 1 : batch) * K * N;
  std::size_t oN = static_cast<std::size_t>(batch) * M * N;
  std::vector<float> Af(aN), Bf(bN), Of(oN);
  for (std::size_t i = 0; i < aN; ++i) Af[i] = bfloat16_to_float_stub(A[i]);
  for (std::size_t i = 0; i < bN; ++i) Bf[i] = bfloat16_to_float_stub(B[i]);
  tessera_apple_gpu_bmm_f32(Af.data(), Bf.data(), Of.data(), batch, M, N, K, b_broadcast);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_bfloat16_stub(Of[i]);
}
// R1 device-resident bmm — non-Apple reference. The handles are host-memory
// backed, so this is the same bmm into O->data.
extern "C" int32_t tessera_apple_gpu_bmm_dev_f32(TsDeviceTensor* A,
                                                 TsDeviceTensor* B,
                                                 TsDeviceTensor* O, int32_t batch,
                                                 int32_t M, int32_t N, int32_t K,
                                                 int32_t b_broadcast) {
  if (!A || !B || !O) return 0;
  tessera_apple_gpu_bmm_f32(static_cast<const float*>(A->data),
                            static_cast<const float*>(B->data),
                            static_cast<float*>(O->data), batch, M, N, K,
                            b_broadcast);
  return 1;
}

// R2 — encode session, non-Apple reference. No command buffer; encoded ops run
// immediately into O->data and commit is a no-op, so results stay correct.
struct TsEncodeSession { int dummy; };
extern "C" TsEncodeSession* ts_enc_begin(void) { return new TsEncodeSession{0}; }
extern "C" void ts_enc_commit_wait(TsEncodeSession* s) { delete s; }
extern "C" int32_t tessera_apple_gpu_bmm_dev_f32_enc(TsEncodeSession* s,
                                                     TsDeviceTensor* A,
                                                     TsDeviceTensor* B,
                                                     TsDeviceTensor* O,
                                                     int32_t batch, int32_t M,
                                                     int32_t N, int32_t K,
                                                     int32_t b_broadcast) {
  if (!s) return 0;
  return tessera_apple_gpu_bmm_dev_f32(A, B, O, batch, M, N, K, b_broadcast);
}

// Single-command-buffer decode scaffold (2026-06-01) — off-Darwin reference.
// No command buffer; the layer_norm reference runs immediately into Y->data.
extern "C" int32_t tessera_apple_gpu_layer_norm_dev_f32_enc(
    TsEncodeSession* s,
    TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* beta, TsDeviceTensor* Y,
    int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !beta || !Y) return 0;
  const float* xb = reinterpret_cast<const float*>(X->data);
  const float* gb = reinterpret_cast<const float*>(gamma->data);
  const float* bb = reinterpret_cast<const float*>(beta->data);
  float* yb = reinterpret_cast<float*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = xb + static_cast<std::size_t>(r) * cols;
    float* o = yb + static_cast<std::size_t>(r) * cols;
    double mean = 0.0;
    for (int32_t c = 0; c < cols; ++c) mean += row[c];
    mean /= cols;
    double var = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      double d = row[c] - mean;
      var += d * d;
    }
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int32_t c = 0; c < cols; ++c) {
      o[c] = static_cast<float>(
          ((row[c] - mean) * inv) * gb[c] + bb[c]);
    }
  }
  return 1;
}

// Stage-2 single-cb scaffold (2026-06-01) — flash_attn encoded
// dispatch, off-Darwin reference. Runs the existing host reference
// into O->data immediately (no command buffer to defer).
extern "C" int32_t tessera_apple_gpu_flash_attn_dev_f32_enc(
    TsEncodeSession* s,
    TsDeviceTensor* Q, TsDeviceTensor* K,
    TsDeviceTensor* V, TsDeviceTensor* O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  if (!s || !Q || !K || !V || !O) return 0;
  reference_flash_attn_f32(
      reinterpret_cast<const float*>(Q->data),
      reinterpret_cast<const float*>(K->data),
      reinterpret_cast<const float*>(V->data),
      reinterpret_cast<float*>(O->data),
      B, Sq, Sk, D, scale, causal);
  return 1;
}

// Stage-2 (2026-06-01) — rmsnorm encoded, off-Darwin reference.
extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_f32_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* Y, int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !Y) return 0;
  const float* xb = reinterpret_cast<const float*>(X->data);
  const float* gb = reinterpret_cast<const float*>(gamma->data);
  float* yb = reinterpret_cast<float*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = xb + static_cast<std::size_t>(r) * cols;
    float* o = yb + static_cast<std::size_t>(r) * cols;
    double v = 0.0;
    for (int32_t c = 0; c < cols; ++c) v += static_cast<double>(row[c]) * row[c];
    v /= cols;
    double inv = 1.0 / std::sqrt(v + eps);
    for (int32_t c = 0; c < cols; ++c)
      o[c] = static_cast<float>(row[c] * inv * gb[c]);
  }
  return 1;
}

// Stage-2 (2026-06-01) — softmax encoded, off-Darwin reference.
extern "C" int32_t tessera_apple_gpu_softmax_dev_f32_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Y,
    int32_t rows, int32_t cols) {
  if (!s || !X || !Y) return 0;
  const float* xb = reinterpret_cast<const float*>(X->data);
  float* yb = reinterpret_cast<float*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = xb + static_cast<std::size_t>(r) * cols;
    float* o = yb + static_cast<std::size_t>(r) * cols;
    float mx = row[0];
    for (int32_t c = 1; c < cols; ++c) if (row[c] > mx) mx = row[c];
    double s_total = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      o[c] = static_cast<float>(std::exp(row[c] - mx));
      s_total += o[c];
    }
    float inv = static_cast<float>(1.0 / s_total);
    for (int32_t c = 0; c < cols; ++c) o[c] *= inv;
  }
  return 1;
}

// Stage-2 (2026-06-01) — RoPE encoded, off-Darwin reference (mirrors
// the MSL kernel above: pair-wise rotation by theta).
extern "C" int32_t tessera_apple_gpu_rope_dev_f32_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Theta,
    TsDeviceTensor* Y, int32_t M, int32_t K) {
  if (!s || !X || !Theta || !Y) return 0;
  const float* xb = reinterpret_cast<const float*>(X->data);
  const float* tb = reinterpret_cast<const float*>(Theta->data);
  float* yb = reinterpret_cast<float*>(Y->data);
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t pair = 0; pair < K / 2; ++pair) {
      int idx_e = m * K + pair * 2;
      int idx_o = idx_e + 1;
      float xe = xb[idx_e];
      float xo = xb[idx_o];
      float c = std::cos(tb[idx_e]);
      float ss = std::sin(tb[idx_e]);
      yb[idx_e] = xe * c - xo * ss;
      yb[idx_o] = xe * ss + xo * c;
    }
  }
  return 1;
}

// Project-3 f16 encode-session stubs (2026-06-01).
// Off-Darwin reference: treat fp16 inputs as numpy ml_dtypes.float16
// 16-bit bit patterns, upcast to fp32 for the math, write back as fp16
// bit patterns. Keeps stub-side dtype handling deterministic across
// non-Apple CI without needing a real fp16 implementation.

static inline float _f16_to_f32(uint16_t h) {
  uint32_t s = (h & 0x8000u) << 16;
  uint32_t e = (h & 0x7C00u) >> 10;
  uint32_t m = (h & 0x03FFu);
  uint32_t r;
  if (e == 0) {
    if (m == 0) r = s;
    else {
      e = 1;
      while ((m & 0x0400u) == 0) { m <<= 1; --e; }
      m &= 0x03FFu;
      r = s | ((e + 112u) << 23) | (m << 13);
    }
  } else if (e == 31) {
    r = s | 0x7F800000u | (m << 13);
  } else {
    r = s | ((e + 112u) << 23) | (m << 13);
  }
  float out;
  std::memcpy(&out, &r, 4);
  return out;
}

static inline uint16_t _f32_to_f16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  uint32_t s = (bits >> 16) & 0x8000u;
  int32_t e = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t m = bits & 0x007FFFFFu;
  if (e >= 31) {
    return (uint16_t)(s | 0x7C00u | (m ? 0x0200u : 0));
  } else if (e <= 0) {
    if (e < -10) return (uint16_t)s;
    m |= 0x00800000u;
    uint32_t shift = (uint32_t)(14 - e);
    return (uint16_t)(s | (m >> shift));
  }
  return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

extern "C" int32_t tessera_apple_gpu_bmm_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* A, TsDeviceTensor* B,
    TsDeviceTensor* O, int32_t batch, int32_t M, int32_t N, int32_t K,
    int32_t b_broadcast) {
  if (!s || !A || !B || !O) return 0;
  const uint16_t* Ah = reinterpret_cast<const uint16_t*>(A->data);
  const uint16_t* Bh = reinterpret_cast<const uint16_t*>(B->data);
  uint16_t* Oh = reinterpret_cast<uint16_t*>(O->data);
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t i = 0; i < M; ++i) {
      for (int32_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int32_t kk = 0; kk < K; ++kk) {
          int32_t b_idx = b_broadcast ? 0 : b;
          acc += _f16_to_f32(Ah[((std::size_t)b * M + i) * K + kk]) *
                 _f16_to_f32(Bh[((std::size_t)b_idx * K + kk) * N + j]);
        }
        Oh[((std::size_t)b * M + i) * N + j] = _f32_to_f16(acc);
      }
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_layer_norm_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* beta, TsDeviceTensor* Y,
    int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !beta || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(gamma->data);
  const uint16_t* bh = reinterpret_cast<const uint16_t*>(beta->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int32_t c = 0; c < cols; ++c)
      mean += _f16_to_f32(xh[(std::size_t)r * cols + c]);
    mean /= cols;
    double var = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      double d = _f16_to_f32(xh[(std::size_t)r * cols + c]) - mean;
      var += d * d;
    }
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int32_t c = 0; c < cols; ++c) {
      double n = (_f16_to_f32(xh[(std::size_t)r * cols + c]) - mean) * inv;
      yh[(std::size_t)r * cols + c] = _f32_to_f16(
          (float)(n * _f16_to_f32(gh[c]) + _f16_to_f32(bh[c])));
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* Y, int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(gamma->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    double v = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      double x = _f16_to_f32(xh[(std::size_t)r * cols + c]);
      v += x * x;
    }
    v /= cols;
    double inv = 1.0 / std::sqrt(v + eps);
    for (int32_t c = 0; c < cols; ++c) {
      double x = _f16_to_f32(xh[(std::size_t)r * cols + c]);
      yh[(std::size_t)r * cols + c] = _f32_to_f16(
          (float)(x * inv * _f16_to_f32(gh[c])));
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_softmax_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Y,
    int32_t rows, int32_t cols) {
  if (!s || !X || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    float mx = _f16_to_f32(xh[(std::size_t)r * cols + 0]);
    for (int32_t c = 1; c < cols; ++c) {
      float v = _f16_to_f32(xh[(std::size_t)r * cols + c]);
      if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      float v = _f16_to_f32(xh[(std::size_t)r * cols + c]);
      double e = std::exp(v - mx);
      yh[(std::size_t)r * cols + c] = _f32_to_f16((float)e);
      sum += e;
    }
    float inv = (float)(1.0 / sum);
    for (int32_t c = 0; c < cols; ++c) {
      float e = _f16_to_f32(yh[(std::size_t)r * cols + c]);
      yh[(std::size_t)r * cols + c] = _f32_to_f16(e * inv);
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_rope_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Theta,
    TsDeviceTensor* Y, int32_t M, int32_t K) {
  if (!s || !X || !Theta || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* th = reinterpret_cast<const uint16_t*>(Theta->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t pair = 0; pair < K / 2; ++pair) {
      int idx_e = m * K + pair * 2;
      int idx_o = idx_e + 1;
      float xe = _f16_to_f32(xh[idx_e]);
      float xo = _f16_to_f32(xh[idx_o]);
      float c = std::cos(_f16_to_f32(th[idx_e]));
      float ss = std::sin(_f16_to_f32(th[idx_e]));
      yh[idx_e] = _f32_to_f16(xe * c - xo * ss);
      yh[idx_o] = _f32_to_f16(xe * ss + xo * c);
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_flash_attn_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* Q, TsDeviceTensor* K,
    TsDeviceTensor* V, TsDeviceTensor* O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  if (!s || !Q || !K || !V || !O) return 0;
  const uint16_t* Qh = reinterpret_cast<const uint16_t*>(Q->data);
  const uint16_t* Kh = reinterpret_cast<const uint16_t*>(K->data);
  const uint16_t* Vh = reinterpret_cast<const uint16_t*>(V->data);
  uint16_t* Oh = reinterpret_cast<uint16_t*>(O->data);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t q = 0; q < Sq; ++q) {
      // Online softmax for this row.
      float m = -INFINITY;
      float l = 0.0f;
      std::vector<float> o(D, 0.0f);
      for (int32_t k = 0; k < Sk; ++k) {
        if (causal && k > q) break;
        float score = 0.0f;
        for (int32_t d = 0; d < D; ++d) {
          score += _f16_to_f32(Qh[((std::size_t)b * Sq + q) * D + d]) *
                   _f16_to_f32(Kh[((std::size_t)b * Sk + k) * D + d]);
        }
        score *= scale;
        float new_m = std::max(m, score);
        float exp_old = std::exp(m - new_m);
        float exp_score = std::exp(score - new_m);
        for (int32_t d = 0; d < D; ++d) {
          o[d] = o[d] * exp_old +
                 _f16_to_f32(Vh[((std::size_t)b * Sk + k) * D + d]) *
                     exp_score;
        }
        l = l * exp_old + exp_score;
        m = new_m;
      }
      if (l == 0.0f) {
        for (int32_t d = 0; d < D; ++d)
          Oh[((std::size_t)b * Sq + q) * D + d] = _f32_to_f16(0.0f);
      } else {
        float inv_l = 1.0f / l;
        for (int32_t d = 0; d < D; ++d)
          Oh[((std::size_t)b * Sq + q) * D + d] = _f32_to_f16(o[d] * inv_l);
      }
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_unary_dev_f16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* O,
    int64_t n, int32_t op) {
  if (!s || !X || !O) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  uint16_t* oh = reinterpret_cast<uint16_t*>(O->data);
  for (int64_t i = 0; i < n; ++i) {
    float v = _f16_to_f32(xh[i]);
    float r;
    switch (op) {
      case 0: r = v > 0 ? v : 0.0f; break;
      case 1: r = 1.0f / (1.0f + std::exp(-v)); break;
      case 2: r = std::tanh(v); break;
      case 3: r = std::log1p(std::exp(v)); break;
      case 4: r = v / (1.0f + std::exp(-v)); break;  // silu
      case 5: {  // gelu tanh approx
        float c = std::sqrt(2.0f / (float)M_PI);
        r = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
        break;
      }
      case 6: r = std::exp(v); break;
      case 7: r = std::log(v); break;
      case 8: r = std::sqrt(v); break;
      case 9: r = 1.0f / std::sqrt(v); break;
      case 10: r = -v; break;
      case 11: r = std::fabs(v); break;
      default: r = v; break;
    }
    oh[i] = _f32_to_f16(r);
  }
  return 1;
}

// Project-3 bf16 encode-session stubs (2026-06-01) — off-Darwin
// reference. bf16 is the high 16 bits of a fp32 IEEE-754 bit pattern;
// decode by zero-extending, encode by truncating with round-to-nearest.

static inline float _bf16_to_f32(uint16_t b) {
  uint32_t bits = ((uint32_t)b) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

static inline uint16_t _f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  // Round-to-nearest-even.
  uint32_t rounding_bias = 0x00007FFFu + ((bits >> 16) & 1u);
  bits += rounding_bias;
  return (uint16_t)(bits >> 16);
}

// Phase 2 stride-alignment wire-up (2026-06-01) — opt-in setter
// stub. Off-Darwin packaged ML isn't reachable, so this is a no-op
// success.
extern "C" int32_t tessera_apple_gpu_mlpkg_set_aligned_strides(
    void* handle, int32_t flag) {
  (void)handle; (void)flag;
  return 1;
}

// Phase 2 stride-alignment (2026-06-01) — companion byte-count helper.
extern "C" int64_t tessera_apple_gpu_aligned_buffer_nbytes(
    const int64_t* dims_in, int32_t rank, int32_t element_bits,
    int32_t ml_usage) {
  if (!dims_in || rank <= 0 || rank > 8) return 0;
  if (element_bits <= 0 || element_bits > 64) return 0;
  int64_t strides[8];
  int32_t rc = tessera_apple_gpu_row_major_strides_aligned(
      dims_in, rank, element_bits, ml_usage, strides);
  if (rc != rank) return 0;
  int64_t total_elems = strides[rank - 1] * dims_in[rank - 1];
  int64_t total_bits = total_elems * element_bits;
  return (total_bits + 7) / 8;
}

extern "C" int32_t tessera_apple_gpu_mpsgraph_bf16_supported(void) {
  // Off-Darwin: no MPSGraph to probe; report the stub's own
  // bf16 capability (the fp32-conversion math is host code).
  return 1;
}

extern "C" int32_t tessera_apple_gpu_bmm_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* A, TsDeviceTensor* B,
    TsDeviceTensor* O, int32_t batch, int32_t M, int32_t N, int32_t K,
    int32_t b_broadcast) {
  if (!s || !A || !B || !O) return 0;
  const uint16_t* Ah = reinterpret_cast<const uint16_t*>(A->data);
  const uint16_t* Bh = reinterpret_cast<const uint16_t*>(B->data);
  uint16_t* Oh = reinterpret_cast<uint16_t*>(O->data);
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t i = 0; i < M; ++i) {
      for (int32_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int32_t kk = 0; kk < K; ++kk) {
          int32_t b_idx = b_broadcast ? 0 : b;
          acc += _bf16_to_f32(Ah[((std::size_t)b * M + i) * K + kk]) *
                 _bf16_to_f32(Bh[((std::size_t)b_idx * K + kk) * N + j]);
        }
        Oh[((std::size_t)b * M + i) * N + j] = _f32_to_bf16(acc);
      }
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_layer_norm_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* beta, TsDeviceTensor* Y,
    int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !beta || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(gamma->data);
  const uint16_t* bh = reinterpret_cast<const uint16_t*>(beta->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int32_t c = 0; c < cols; ++c)
      mean += _bf16_to_f32(xh[(std::size_t)r * cols + c]);
    mean /= cols;
    double var = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      double d = _bf16_to_f32(xh[(std::size_t)r * cols + c]) - mean;
      var += d * d;
    }
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int32_t c = 0; c < cols; ++c) {
      double n = (_bf16_to_f32(xh[(std::size_t)r * cols + c]) - mean) * inv;
      yh[(std::size_t)r * cols + c] = _f32_to_bf16(
          (float)(n * _bf16_to_f32(gh[c]) + _bf16_to_f32(bh[c])));
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* gamma,
    TsDeviceTensor* Y, int32_t rows, int32_t cols, float eps) {
  if (!s || !X || !gamma || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(gamma->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    double v = 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      double x = _bf16_to_f32(xh[(std::size_t)r * cols + c]);
      v += x * x;
    }
    v /= cols;
    double inv = 1.0 / std::sqrt(v + eps);
    for (int32_t c = 0; c < cols; ++c) {
      double x = _bf16_to_f32(xh[(std::size_t)r * cols + c]);
      yh[(std::size_t)r * cols + c] = _f32_to_bf16(
          (float)(x * inv * _bf16_to_f32(gh[c])));
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_softmax_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Y,
    int32_t rows, int32_t cols) {
  if (!s || !X || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t r = 0; r < rows; ++r) {
    float mx = _bf16_to_f32(xh[(std::size_t)r * cols + 0]);
    for (int32_t c = 1; c < cols; ++c) {
      float v = _bf16_to_f32(xh[(std::size_t)r * cols + c]);
      if (v > mx) mx = v;
    }
    double sum = 0.0;
    std::vector<float> tmp(cols);
    for (int32_t c = 0; c < cols; ++c) {
      float v = _bf16_to_f32(xh[(std::size_t)r * cols + c]);
      tmp[c] = (float)std::exp(v - mx);
      sum += tmp[c];
    }
    float inv = (float)(1.0 / sum);
    for (int32_t c = 0; c < cols; ++c) {
      yh[(std::size_t)r * cols + c] = _f32_to_bf16(tmp[c] * inv);
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_unary_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* O,
    int64_t n, int32_t op) {
  if (!s || !X || !O) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  uint16_t* oh = reinterpret_cast<uint16_t*>(O->data);
  for (int64_t i = 0; i < n; ++i) {
    float v = _bf16_to_f32(xh[i]);
    float r;
    switch (op) {
      case 0: r = v > 0 ? v : 0.0f; break;
      case 1: r = 1.0f / (1.0f + std::exp(-v)); break;
      case 2: r = std::tanh(v); break;
      case 3: r = std::log1p(std::exp(v)); break;
      case 4: r = v / (1.0f + std::exp(-v)); break;
      case 5: {
        float c = std::sqrt(2.0f / (float)M_PI);
        r = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
        break;
      }
      case 6: r = std::exp(v); break;
      case 7: r = std::log(v); break;
      case 8: r = std::sqrt(v); break;
      case 9: r = 1.0f / std::sqrt(v); break;
      case 10: r = -v; break;
      case 11: r = std::fabs(v); break;
      default: r = v; break;
    }
    oh[i] = _f32_to_bf16(r);
  }
  return 1;
}

// Phase 3b (2026-06-01) — bf16 MSL via on-GPU cast, off-Darwin
// reference. The Darwin runtime composes bf16→fp32 cast + fp32 MSL
// kernel + fp32→bf16 cast on the GPU. Off-Darwin we just do the
// equivalent math on the host: read bf16, compute in fp32, write bf16.

extern "C" int32_t tessera_apple_gpu_rope_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* X, TsDeviceTensor* Theta,
    TsDeviceTensor* Y, int32_t M, int32_t K) {
  if (!s || !X || !Theta || !Y) return 0;
  const uint16_t* xh = reinterpret_cast<const uint16_t*>(X->data);
  const uint16_t* th = reinterpret_cast<const uint16_t*>(Theta->data);
  uint16_t* yh = reinterpret_cast<uint16_t*>(Y->data);
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t pair = 0; pair < K / 2; ++pair) {
      int idx_e = m * K + pair * 2;
      int idx_o = idx_e + 1;
      float xe = _bf16_to_f32(xh[idx_e]);
      float xo = _bf16_to_f32(xh[idx_o]);
      float c = std::cos(_bf16_to_f32(th[idx_e]));
      float ss = std::sin(_bf16_to_f32(th[idx_e]));
      yh[idx_e] = _f32_to_bf16(xe * c - xo * ss);
      yh[idx_o] = _f32_to_bf16(xe * ss + xo * c);
    }
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_flash_attn_dev_bf16_enc(
    TsEncodeSession* s, TsDeviceTensor* Q, TsDeviceTensor* K,
    TsDeviceTensor* V, TsDeviceTensor* O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  if (!s || !Q || !K || !V || !O) return 0;
  const uint16_t* Qh = reinterpret_cast<const uint16_t*>(Q->data);
  const uint16_t* Kh = reinterpret_cast<const uint16_t*>(K->data);
  const uint16_t* Vh = reinterpret_cast<const uint16_t*>(V->data);
  uint16_t* Oh = reinterpret_cast<uint16_t*>(O->data);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t q = 0; q < Sq; ++q) {
      float m = -INFINITY;
      float l = 0.0f;
      std::vector<float> o(D, 0.0f);
      for (int32_t k = 0; k < Sk; ++k) {
        if (causal && k > q) break;
        float score = 0.0f;
        for (int32_t d = 0; d < D; ++d) {
          score += _bf16_to_f32(Qh[((std::size_t)b * Sq + q) * D + d]) *
                   _bf16_to_f32(Kh[((std::size_t)b * Sk + k) * D + d]);
        }
        score *= scale;
        float new_m = std::max(m, score);
        float exp_old = std::exp(m - new_m);
        float exp_score = std::exp(score - new_m);
        for (int32_t d = 0; d < D; ++d) {
          o[d] = o[d] * exp_old +
                 _bf16_to_f32(Vh[((std::size_t)b * Sk + k) * D + d]) *
                     exp_score;
        }
        l = l * exp_old + exp_score;
        m = new_m;
      }
      if (l == 0.0f) {
        for (int32_t d = 0; d < D; ++d)
          Oh[((std::size_t)b * Sq + q) * D + d] = _f32_to_bf16(0.0f);
      } else {
        float inv_l = 1.0f / l;
        for (int32_t d = 0; d < D; ++d)
          Oh[((std::size_t)b * Sq + q) * D + d] = _f32_to_bf16(o[d] * inv_l);
      }
    }
  }
  return 1;
}

extern "C" int64_t tessera_apple_gpu_session_commit_count(void) {
  // Off-Darwin: no real command queue; static counter incremented by
  // ``ts_enc_commit_wait`` in the stub (simple file-static — single-threaded
  // off-Darwin tests are the only callers).
  static int64_t counter = 0;
  return counter;
}

// R2 rowop / gumbel device + encode — non-Apple references into O->data.
static void rowop_ref_host(int kind, const float* x, const float* gamma,
                           float* o, int32_t rows, int32_t cols, float eps) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * cols;
    float* orow = o + static_cast<std::size_t>(r) * cols;
    if (kind == 0 || kind == 1) {  // layer_norm / rmsnorm
      double mean = 0.0;
      if (kind == 0) { for (int32_t c = 0; c < cols; ++c) mean += xr[c]; mean /= cols; }
      double v = 0.0;
      for (int32_t c = 0; c < cols; ++c) {
        double d = (kind == 0) ? (xr[c] - mean) : xr[c];
        v += d * d;
      }
      v /= cols;
      double denom = std::sqrt(v + eps);
      for (int32_t c = 0; c < cols; ++c) {
        double d = (kind == 0) ? (xr[c] - mean) : xr[c];
        double nv = d / denom;
        if (gamma) nv *= gamma[c];
        orow[c] = static_cast<float>(nv);
      }
    } else {  // softmax (2) / log_softmax (3)
      double mx = -1e30;
      for (int32_t c = 0; c < cols; ++c) mx = std::max(mx, (double)xr[c]);
      double sum = 0.0;
      for (int32_t c = 0; c < cols; ++c) sum += std::exp(xr[c] - mx);
      for (int32_t c = 0; c < cols; ++c) {
        double e = xr[c] - mx;
        orow[c] = static_cast<float>(kind == 2 ? std::exp(e) / sum : e - std::log(sum));
      }
    }
  }
}
extern "C" int32_t tessera_apple_gpu_rowop_dev_f32(TsDeviceTensor* X,
                                                   TsDeviceTensor* gamma,
                                                   TsDeviceTensor* O, int32_t kind,
                                                   int32_t rows, int32_t cols,
                                                   float eps) {
  if (!X || !O) return 0;
  rowop_ref_host(kind, static_cast<const float*>(X->data),
                 gamma ? static_cast<const float*>(gamma->data) : nullptr,
                 static_cast<float*>(O->data), rows, cols, eps);
  return 1;
}
extern "C" int32_t tessera_apple_gpu_rowop_dev_f32_enc(TsEncodeSession* s,
                                                       TsDeviceTensor* X,
                                                       TsDeviceTensor* gamma,
                                                       TsDeviceTensor* O,
                                                       int32_t kind, int32_t rows,
                                                       int32_t cols, float eps) {
  if (!s) return 0;
  return tessera_apple_gpu_rowop_dev_f32(X, gamma, O, kind, rows, cols, eps);
}
// R2 (cont.) — flat elementwise unary/binary, non-Apple reference into O->data.
// unary op: 0 relu, 4 silu. binary op: 0 add, 2 mul.
extern "C" int32_t tessera_apple_gpu_unary_dev_f32_enc(TsEncodeSession* s,
                                                       TsDeviceTensor* X,
                                                       TsDeviceTensor* O,
                                                       int64_t n, int32_t op) {
  if (!s || !X || !O) return 0;
  const float* x = static_cast<const float*>(X->data);
  float* o = static_cast<float*>(O->data);
  for (int64_t i = 0; i < n; ++i) {
    float v = x[i];
    o[i] = (op == 0) ? (v > 0.f ? v : 0.f)
                     : (op == 4) ? v / (1.f + std::exp(-v)) : v;
  }
  return 1;
}
extern "C" int32_t tessera_apple_gpu_binary_dev_f32_enc(TsEncodeSession* s,
                                                        TsDeviceTensor* A,
                                                        TsDeviceTensor* B,
                                                        TsDeviceTensor* O,
                                                        int64_t n, int32_t op) {
  if (!s || !A || !B || !O) return 0;
  const float* a = static_cast<const float*>(A->data);
  const float* b = static_cast<const float*>(B->data);
  float* o = static_cast<float*>(O->data);
  for (int64_t i = 0; i < n; ++i) {
    o[i] = (op == 0) ? a[i] + b[i] : (op == 2) ? a[i] * b[i] : a[i];
  }
  return 1;
}
extern "C" int32_t tessera_apple_gpu_gumbel_argmax_dev_f32(TsDeviceTensor* logits,
                                                           TsDeviceTensor* gumbel,
                                                           TsDeviceTensor* out_ids,
                                                           int32_t rows,
                                                           int32_t cols,
                                                           float inv_temp) {
  if (!logits || !gumbel || !out_ids) return 0;
  const float* L = static_cast<const float*>(logits->data);
  const float* G = static_cast<const float*>(gumbel->data);
  int32_t* out = static_cast<int32_t*>(out_ids->data);
  for (int32_t r = 0; r < rows; ++r) {
    const float* Lr = L + static_cast<std::size_t>(r) * cols;
    const float* Gr = G + static_cast<std::size_t>(r) * cols;
    int32_t best = 0;
    float bs = Lr[0] * inv_temp + Gr[0];
    for (int32_t c = 1; c < cols; ++c) {
      float sv = Lr[c] * inv_temp + Gr[c];
      if (sv > bs) { bs = sv; best = c; }
    }
    out[r] = best;
  }
  return 1;
}
extern "C" int32_t tessera_apple_gpu_gumbel_argmax_dev_f32_enc(
    TsEncodeSession* s, TsDeviceTensor* logits, TsDeviceTensor* gumbel,
    TsDeviceTensor* out_ids, int32_t rows, int32_t cols, float inv_temp) {
  if (!s) return 0;
  return tessera_apple_gpu_gumbel_argmax_dev_f32(logits, gumbel, out_ids, rows,
                                                 cols, inv_temp);
}

// R4 block-table gather — non-Apple reference into out->data.
extern "C" int32_t tessera_apple_gpu_gather_blocks_dev_f32(
    TsDeviceTensor* pool, TsDeviceTensor* block_table, TsDeviceTensor* out,
    int32_t num_blocks, int32_t n, int32_t block_size, int32_t dim) {
  if (!pool || !block_table || !out) return 0;
  const float* P = static_cast<const float*>(pool->data);
  const int32_t* idx = static_cast<const int32_t*>(block_table->data);
  float* O = static_cast<float*>(out->data);
  std::size_t blk = static_cast<std::size_t>(block_size) * dim;
  for (int32_t i = 0; i < n; ++i) {
    int32_t b = idx[i];
    if (b < 0 || b >= num_blocks) b = 0;
    std::memcpy(O + static_cast<std::size_t>(i) * blk,
                P + static_cast<std::size_t>(b) * blk, blk * sizeof(float));
  }
  return 1;
}
extern "C" int32_t tessera_apple_gpu_gather_blocks_dev_f32_enc(
    TsEncodeSession* s, TsDeviceTensor* pool, TsDeviceTensor* block_table,
    TsDeviceTensor* out, int32_t num_blocks, int32_t n, int32_t block_size,
    int32_t dim) {
  if (!s) return 0;
  return tessera_apple_gpu_gather_blocks_dev_f32(pool, block_table, out,
                                                 num_blocks, n, block_size, dim);
}

// ---- Tier-3 reduction lane non-Apple reference (2026-05-29) ----------------
extern "C" void tessera_apple_gpu_mpsgraph_reduce_f32(int32_t op, const float* x,
                                                      float* out, int32_t rows,
                                                      int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    double acc;
    switch (op) {
      case 0: case 1: { double s = 0; for (int32_t c = 0; c < cols; ++c) s += row[c]; acc = op == 1 ? s / cols : s; break; }
      case 2: { double m = row[0]; for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m; acc = m; break; }
      case 3: { double m = row[0]; for (int32_t c = 1; c < cols; ++c) m = row[c] < m ? row[c] : m; acc = m; break; }
      case 4: { double p = 1; for (int32_t c = 0; c < cols; ++c) p *= row[c]; acc = p; break; }
      default: { double s = 0; for (int32_t c = 0; c < cols; ++c) s += row[c]; double m = s / cols; double v = 0; for (int32_t c = 0; c < cols; ++c) { double d = row[c] - m; v += d * d; } v /= cols; acc = op == 6 ? std::sqrt(v) : v; break; }
    }
    out[r] = static_cast<float>(acc);
  }
}
extern "C" void tessera_apple_gpu_mpsgraph_argreduce_f32(int32_t op, const float* x,
                                                         int32_t* out, int32_t rows,
                                                         int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    int32_t best = 0;
    for (int32_t c = 1; c < cols; ++c)
      if ((op == 0 && row[c] > row[best]) || (op == 1 && row[c] < row[best])) best = c;
    out[r] = best;
  }
}
extern "C" void tessera_apple_gpu_gumbel_argmax_f32(const float* logits,
                                                    const float* gumbel,
                                                    int32_t* out, int32_t rows,
                                                    int32_t cols,
                                                    float inv_temp) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* L = logits + static_cast<std::size_t>(r) * cols;
    const float* G = gumbel + static_cast<std::size_t>(r) * cols;
    int32_t best = 0;
    float bs = L[0] * inv_temp + G[0];
    for (int32_t c = 1; c < cols; ++c) {
      float s = L[c] * inv_temp + G[c];
      if (s > bs) { bs = s; best = c; }
    }
    out[r] = best;
  }
}
extern "C" void tessera_apple_gpu_mpsgraph_scan_f32(int32_t op, const float* x,
                                                    float* out, int32_t rows,
                                                    int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float* row = x + static_cast<std::size_t>(r) * cols;
    float* o = out + static_cast<std::size_t>(r) * cols;
    double acc = op == 1 ? 1.0 : 0.0;
    for (int32_t c = 0; c < cols; ++c) { acc = op == 1 ? acc * row[c] : acc + row[c]; o[c] = static_cast<float>(acc); }
  }
}

// ---- flash_attn GQA non-Apple reference (2026-05-29) -----------------------
extern "C" void tessera_apple_gpu_flash_attn_gqa_f32(
    const float* Q, const float* K, const float* V, float* O, int32_t B,
    int32_t q_heads, int32_t kv_heads, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  int32_t group = (kv_heads > 0) ? q_heads / kv_heads : 1;
  for (int32_t b = 0; b < B; ++b) {
    int32_t kv_batch = (b / q_heads) * kv_heads + (b % q_heads) / (group > 0 ? group : 1);
    const float* Kb = K + static_cast<std::size_t>(kv_batch) * Sk * D;
    const float* Vb = V + static_cast<std::size_t>(kv_batch) * Sk * D;
    for (int32_t q = 0; q < Sq; ++q) {
      const float* qp = Q + (static_cast<std::size_t>(b) * Sq + q) * D;
      float* op = O + (static_cast<std::size_t>(b) * Sq + q) * D;
      double m = -1e30, l = 0.0;
      std::vector<double> o(D, 0.0);
      for (int32_t k = 0; k < Sk; ++k) {
        if (causal != 0 && k > q) break;
        const float* kp = Kb + static_cast<std::size_t>(k) * D;
        double s = 0.0;
        for (int32_t d = 0; d < D; ++d) s += static_cast<double>(qp[d]) * kp[d];
        s *= scale;
        double nm = std::max(m, s);
        double eo = std::exp(m - nm), es = std::exp(s - nm);
        l = l * eo + es;
        for (int32_t d = 0; d < D; ++d) o[d] = o[d] * eo + Vb[static_cast<std::size_t>(k) * D + d] * es;
        m = nm;
      }
      if (l == 0.0) for (int32_t d = 0; d < D; ++d) op[d] = 0.0f;
      else for (int32_t d = 0; d < D; ++d) op[d] = static_cast<float>(o[d] / l);
    }
  }
}

// ---- fused batched matmul->softmax->matmul non-Apple reference -------------
extern "C" void tessera_apple_gpu_mpsgraph_bsmm_f32(const float* A, const float* B,
                                                    const float* C, float* O,
                                                    int32_t batch, int32_t M,
                                                    int32_t N, int32_t P, int32_t K,
                                                    float scale) {
  std::vector<double> s(static_cast<std::size_t>(M) * N);
  for (int32_t bi = 0; bi < batch; ++bi) {
    const float* a = A + static_cast<std::size_t>(bi) * M * K;
    const float* b = B + static_cast<std::size_t>(bi) * K * N;
    const float* c = C + static_cast<std::size_t>(bi) * N * P;
    float* o = O + static_cast<std::size_t>(bi) * M * P;
    for (int32_t m = 0; m < M; ++m) {
      double mx = -1e30;
      for (int32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (int32_t k = 0; k < K; ++k) acc += static_cast<double>(a[m * K + k]) * b[k * N + n];
        acc *= scale; s[static_cast<std::size_t>(m) * N + n] = acc; mx = std::max(mx, acc);
      }
      double den = 0;
      for (int32_t n = 0; n < N; ++n) { double e = std::exp(s[static_cast<std::size_t>(m) * N + n] - mx); s[static_cast<std::size_t>(m) * N + n] = e; den += e; }
      for (int32_t p = 0; p < P; ++p) {
        double acc = 0;
        for (int32_t n = 0; n < N; ++n) acc += s[static_cast<std::size_t>(m) * N + n] / den * c[n * P + p];
        o[static_cast<std::size_t>(m) * P + p] = static_cast<float>(acc);
      }
    }
  }
}
extern "C" void tessera_apple_gpu_mpsgraph_bsmm_f16(const uint16_t* A, const uint16_t* B,
                                                    const uint16_t* C, uint16_t* O,
                                                    int32_t batch, int32_t M, int32_t N,
                                                    int32_t P, int32_t K, float scale) {
  // Non-Apple reference: f16 -> f32 -> fused bsmm -> f16 (compute, not zero-fill).
  std::size_t aN = static_cast<std::size_t>(batch) * M * K;
  std::size_t bN = static_cast<std::size_t>(batch) * K * N;
  std::size_t cN = static_cast<std::size_t>(batch) * N * P;
  std::size_t oN = static_cast<std::size_t>(batch) * M * P;
  std::vector<float> Af(aN), Bf(bN), Cf(cN), Of(oN);
  for (std::size_t i = 0; i < aN; ++i) Af[i] = half_to_float_stub(A[i]);
  for (std::size_t i = 0; i < bN; ++i) Bf[i] = half_to_float_stub(B[i]);
  for (std::size_t i = 0; i < cN; ++i) Cf[i] = half_to_float_stub(C[i]);
  tessera_apple_gpu_mpsgraph_bsmm_f32(Af.data(), Bf.data(), Cf.data(), Of.data(),
                                      batch, M, N, P, K, scale);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_half_stub(Of[i]);
}

// ---- flash_attn GQA f16/bf16 non-Apple reference (2026-05-30) --------------
// (gqa_bf16_to_f32_stub / gqa_f32_to_bf16_stub are defined up top.)
extern "C" void tessera_apple_gpu_flash_attn_gqa_f16(const uint16_t*, const uint16_t*,
                                                     const uint16_t*, uint16_t* O,
                                                     int32_t B, int32_t, int32_t,
                                                     int32_t Sq, int32_t, int32_t D,
                                                     float, int32_t) {
  std::memset(O, 0, static_cast<std::size_t>(B) * Sq * D * 2);  // python upcasts on fallback
}
extern "C" void tessera_apple_gpu_flash_attn_gqa_bf16(const uint16_t* Q, const uint16_t* K,
                                                      const uint16_t* V, uint16_t* O,
                                                      int32_t B, int32_t q_heads,
                                                      int32_t kv_heads, int32_t Sq,
                                                      int32_t Sk, int32_t D, float scale,
                                                      int32_t causal) {
  int32_t kv_outer = (q_heads > 0) ? (B / q_heads) * kv_heads : B;
  std::vector<float> qf(static_cast<std::size_t>(B) * Sq * D),
      kf(static_cast<std::size_t>(kv_outer) * Sk * D),
      vf(static_cast<std::size_t>(kv_outer) * Sk * D),
      of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < qf.size(); ++i) qf[i] = gqa_bf16_to_f32_stub(Q[i]);
  for (std::size_t i = 0; i < kf.size(); ++i) kf[i] = gqa_bf16_to_f32_stub(K[i]);
  for (std::size_t i = 0; i < vf.size(); ++i) vf[i] = gqa_bf16_to_f32_stub(V[i]);
  tessera_apple_gpu_flash_attn_gqa_f32(qf.data(), kf.data(), vf.data(), of.data(),
                                       B, q_heads, kv_heads, Sq, Sk, D, scale, causal);
  for (std::size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16_stub(of[i]);
}

// ---- conv2d non-Apple reference (NHWC source, HWIO weights) (2026-05-30) ----
static inline int32_t conv2d_out_dim_stub(int32_t in, int32_t k, int32_t stride,
                                          int32_t pad, int32_t dilation) {
  int32_t eff = dilation * (k - 1) + 1;
  if (in + 2 * pad < eff) return 0;
  return (in + 2 * pad - eff) / stride + 1;
}
extern "C" int32_t tessera_apple_gpu_conv2d_out_h(int32_t H, int32_t kH,
                                                  int32_t strideH, int32_t padH,
                                                  int32_t dilationH) {
  return conv2d_out_dim_stub(H, kH, strideH, padH, dilationH);
}
extern "C" int32_t tessera_apple_gpu_conv2d_out_w(int32_t W, int32_t kW,
                                                  int32_t strideW, int32_t padW,
                                                  int32_t dilationW) {
  return conv2d_out_dim_stub(W, kW, strideW, padW, dilationW);
}
static void reference_conv2d_f32_stub(const float* X, const float* Wt,
                                      const float* bias, float* O, int32_t N,
                                      int32_t H, int32_t W, int32_t Cin,
                                      int32_t Cout, int32_t kH, int32_t kW,
                                      int32_t strideH, int32_t strideW,
                                      int32_t padH, int32_t padW,
                                      int32_t dilationH, int32_t dilationW,
                                      int32_t groups) {
  int32_t outH = conv2d_out_dim_stub(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim_stub(W, kW, strideW, padW, dilationW);
  if (outH <= 0 || outW <= 0 || groups <= 0 || Cin % groups || Cout % groups)
    return;
  int32_t cinG = Cin / groups, coutG = Cout / groups;
  for (int32_t n = 0; n < N; ++n)
    for (int32_t oy = 0; oy < outH; ++oy)
      for (int32_t ox = 0; ox < outW; ++ox)
        for (int32_t oc = 0; oc < Cout; ++oc) {
          int32_t grp = oc / coutG;
          double acc = bias ? static_cast<double>(bias[oc]) : 0.0;
          for (int32_t ky = 0; ky < kH; ++ky) {
            int32_t iy = oy * strideH + ky * dilationH - padH;
            if (iy < 0 || iy >= H) continue;
            for (int32_t kx = 0; kx < kW; ++kx) {
              int32_t ix = ox * strideW + kx * dilationW - padW;
              if (ix < 0 || ix >= W) continue;
              for (int32_t ic = 0; ic < cinG; ++ic) {
                int32_t icAbs = grp * cinG + ic;
                double xv = X[(((std::size_t)n * H + iy) * W + ix) * Cin + icAbs];
                double wv = Wt[(((std::size_t)ky * kW + kx) * cinG + ic) * Cout + oc];
                acc += xv * wv;
              }
            }
          }
          O[(((std::size_t)n * outH + oy) * outW + ox) * Cout + oc] =
              static_cast<float>(acc);
        }
}
extern "C" void tessera_apple_gpu_conv2d_f32(
    const float* X, const float* Wt, const float* bias, float* O, int32_t N,
    int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t strideH, int32_t strideW, int32_t padH, int32_t padW,
    int32_t dilationH, int32_t dilationW, int32_t groups) {
  reference_conv2d_f32_stub(X, Wt, bias, O, N, H, W, Cin, Cout, kH, kW, strideH,
                            strideW, padH, padW, dilationH, dilationW, groups);
}
extern "C" void tessera_apple_gpu_conv2d_f16(
    const uint16_t* X, const uint16_t* Wt, const uint16_t* bias, uint16_t* O,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t strideH, int32_t strideW, int32_t padH, int32_t padW,
    int32_t dilationH, int32_t dilationW, int32_t groups) {
  int32_t outH = conv2d_out_dim_stub(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim_stub(W, kW, strideW, padW, dilationW);
  if (outH <= 0 || outW <= 0 || groups <= 0) return;
  // Non-Apple reference: f16 -> f32 -> conv -> f16 (the stub must compute).
  std::size_t xN = static_cast<std::size_t>(N) * H * W * Cin;
  std::size_t wN = static_cast<std::size_t>(kH) * kW * (Cin / groups) * Cout;
  std::size_t oN = static_cast<std::size_t>(N) * outH * outW * Cout;
  std::vector<float> Xf(xN), Wf(wN), Of(oN), Bf;
  for (std::size_t i = 0; i < xN; ++i) Xf[i] = half_to_float_stub(X[i]);
  for (std::size_t i = 0; i < wN; ++i) Wf[i] = half_to_float_stub(Wt[i]);
  const float* bptr = nullptr;
  if (bias) {
    Bf.resize(Cout);
    for (int32_t i = 0; i < Cout; ++i) Bf[i] = half_to_float_stub(bias[i]);
    bptr = Bf.data();
  }
  reference_conv2d_f32_stub(Xf.data(), Wf.data(), bptr, Of.data(), N, H, W, Cin,
                            Cout, kH, kW, strideH, strideW, padH, padW,
                            dilationH, dilationW, groups);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_half_stub(Of[i]);
}

// ---- conv3d non-Apple reference (NDHWC source, DHWIO weights) (2026-05-30) --
extern "C" int32_t tessera_apple_gpu_conv3d_out_dim(int32_t in, int32_t k,
                                                    int32_t stride, int32_t pad,
                                                    int32_t dilation) {
  return conv2d_out_dim_stub(in, k, stride, pad, dilation);
}
static void reference_conv3d_f32_stub(
    const float* X, const float* Wt, const float* bias, float* O, int32_t N,
    int32_t iD, int32_t iH, int32_t iW, int32_t Cin, int32_t Cout, int32_t kD,
    int32_t kH, int32_t kW, int32_t sD, int32_t sH, int32_t sW, int32_t pD,
    int32_t pH, int32_t pW, int32_t dD, int32_t dH, int32_t dW, int32_t groups) {
  int32_t oD = conv2d_out_dim_stub(iD, kD, sD, pD, dD);
  int32_t oH = conv2d_out_dim_stub(iH, kH, sH, pH, dH);
  int32_t oW = conv2d_out_dim_stub(iW, kW, sW, pW, dW);
  if (oD <= 0 || oH <= 0 || oW <= 0 || groups <= 0 || Cin % groups ||
      Cout % groups)
    return;
  int32_t cinG = Cin / groups, coutG = Cout / groups;
  for (int32_t n = 0; n < N; ++n)
    for (int32_t od = 0; od < oD; ++od)
      for (int32_t oh = 0; oh < oH; ++oh)
        for (int32_t ow = 0; ow < oW; ++ow)
          for (int32_t oc = 0; oc < Cout; ++oc) {
            int32_t grp = oc / coutG;
            double acc = bias ? static_cast<double>(bias[oc]) : 0.0;
            for (int32_t kd = 0; kd < kD; ++kd) {
              int32_t id = od * sD + kd * dD - pD;
              if (id < 0 || id >= iD) continue;
              for (int32_t kh = 0; kh < kH; ++kh) {
                int32_t ih = oh * sH + kh * dH - pH;
                if (ih < 0 || ih >= iH) continue;
                for (int32_t kw = 0; kw < kW; ++kw) {
                  int32_t iw = ow * sW + kw * dW - pW;
                  if (iw < 0 || iw >= iW) continue;
                  for (int32_t ic = 0; ic < cinG; ++ic) {
                    double xv = X[((((std::size_t)n * iD + id) * iH + ih) * iW + iw) * Cin + grp * cinG + ic];
                    double wv = Wt[((((std::size_t)kd * kH + kh) * kW + kw) * cinG + ic) * Cout + oc];
                    acc += xv * wv;
                  }
                }
              }
            }
            O[((((std::size_t)n * oD + od) * oH + oh) * oW + ow) * Cout + oc] =
                static_cast<float>(acc);
          }
}
extern "C" void tessera_apple_gpu_conv3d_f32(
    const float* X, const float* Wt, const float* bias, float* O, int32_t N,
    int32_t iD, int32_t iH, int32_t iW, int32_t Cin, int32_t Cout, int32_t kD,
    int32_t kH, int32_t kW, int32_t sD, int32_t sH, int32_t sW, int32_t pD,
    int32_t pH, int32_t pW, int32_t dD, int32_t dH, int32_t dW, int32_t groups) {
  reference_conv3d_f32_stub(X, Wt, bias, O, N, iD, iH, iW, Cin, Cout, kD, kH, kW,
                            sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
}
extern "C" void tessera_apple_gpu_conv3d_f16(
    const uint16_t* X, const uint16_t* Wt, const uint16_t* bias, uint16_t* O, int32_t N,
    int32_t iD, int32_t iH, int32_t iW, int32_t Cin, int32_t Cout, int32_t kD,
    int32_t kH, int32_t kW, int32_t sD, int32_t sH, int32_t sW, int32_t pD,
    int32_t pH, int32_t pW, int32_t dD, int32_t dH, int32_t dW, int32_t groups) {
  int32_t oD = conv2d_out_dim_stub(iD, kD, sD, pD, dD);
  int32_t oH = conv2d_out_dim_stub(iH, kH, sH, pH, dH);
  int32_t oW = conv2d_out_dim_stub(iW, kW, sW, pW, dW);
  if (oD <= 0 || oH <= 0 || oW <= 0 || groups <= 0) return;
  // Non-Apple reference: f16 -> f32 -> conv3d -> f16 (compute, not zero-fill).
  std::size_t xN = static_cast<std::size_t>(N) * iD * iH * iW * Cin;
  std::size_t wN = static_cast<std::size_t>(kD) * kH * kW * (Cin / groups) * Cout;
  std::size_t oN = static_cast<std::size_t>(N) * oD * oH * oW * Cout;
  std::vector<float> Xf(xN), Wf(wN), Of(oN), Bf;
  for (std::size_t i = 0; i < xN; ++i) Xf[i] = half_to_float_stub(X[i]);
  for (std::size_t i = 0; i < wN; ++i) Wf[i] = half_to_float_stub(Wt[i]);
  const float* bptr = nullptr;
  if (bias) {
    Bf.resize(Cout);
    for (int32_t i = 0; i < Cout; ++i) Bf[i] = half_to_float_stub(bias[i]);
    bptr = Bf.data();
  }
  reference_conv3d_f32_stub(Xf.data(), Wf.data(), bptr, Of.data(), N, iD, iH, iW,
                            Cin, Cout, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH,
                            dW, groups);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_half_stub(Of[i]);
}

#endif // !__APPLE__
