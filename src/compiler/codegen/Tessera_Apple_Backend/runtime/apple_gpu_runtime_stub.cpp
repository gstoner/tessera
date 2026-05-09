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

extern "C" int32_t tessera_apple_gpu_runtime_msl_cache_size(void) {
  // No Metal -> no MSL cache. Tests gate this on platform.
  return -1;
}

#endif // !__APPLE__
