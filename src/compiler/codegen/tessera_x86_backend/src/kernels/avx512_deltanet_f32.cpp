// AVX-512 DeltaNet / gated-delta linear-attention causal scan (f32).
//
// The x86 analog of the ROCm rocm_deltanet_compiled lane; matches the numpy
// reference tessera._delta_attention_impl exactly (the executor forces f64 there
// as the oracle). Per (b, h) a Dqk×Dv state S is scanned causally over S steps:
//
//   erase:      v̂_e = Σ_d k_d · S_{d,e};  target_e = v_e − α·v̂_e   (α = decay_t or 1)
//   decay:      S_{d,e} *= α_t
//   delta:      Δ_{d,e} = k_d · target_e
//   modified:   Δ /= (1 + ‖Δ‖_F);  ‖k⊗target‖_F = ‖k‖·‖target‖  (exact, cheap)
//   update:     S_{d,e} += weight · Δ_{d,e}          (weight = beta_t or 1)
//   output:     O_{t,e} = Σ_d Q_{t,d} · S_{d,e}
//   gate:       O_{t,e} *= sigmoid(gate_{t,e})
//
// erase=false is gated linear attention (the shipped default); erase=true is the
// genuine delta rule. `modified` bounds the update (Kimi-style). The inner loops
// are over the contiguous Dv (e) dimension, which the library's AVX-512 build
// flags (-O3 -mavx512f) auto-vectorize; the scan is sequential over t, so the
// loop nest is (b,h) → t → d → e. Single-threaded, race-free (the GPU lane uses
// atomics). Optional operands (gate/beta/decay) are broadcast to full arrays by
// the caller; a null pointer + has_* = 0 means absent.

#include <cmath>
#include <cstdint>
#include <vector>

namespace {
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

extern "C" void tessera_x86_deltanet_f32(
    const float* Q, const float* K, const float* V,
    const float* gate, const float* beta, const float* decay,
    float* O,
    int64_t B, int64_t Hh, int64_t Sseq, int64_t Dqk, int64_t Dv,
    int32_t erase, int32_t modified,
    int32_t has_gate, int32_t has_beta, int32_t has_decay) {
  const int64_t bh = B * Hh;
  std::vector<float> S(static_cast<size_t>(Dqk) * Dv);
  std::vector<float> target(Dv), vhat(Dv);

  for (int64_t g = 0; g < bh; ++g) {
    const int64_t base_qk = g * Sseq * Dqk;
    const int64_t base_v = g * Sseq * Dv;
    const int64_t base_s = g * Sseq;                 // beta/decay: [B,H,S]
    std::fill(S.begin(), S.end(), 0.0f);

    for (int64_t t = 0; t < Sseq; ++t) {
      const float* q = Q + base_qk + t * Dqk;
      const float* k = K + base_qk + t * Dqk;
      const float* v = V + base_v + t * Dv;
      float* o = O + base_v + t * Dv;

      for (int64_t e = 0; e < Dv; ++e) target[e] = v[e];

      const float a_t = has_decay ? decay[base_s + t] : 1.0f;
      if (erase) {
        for (int64_t e = 0; e < Dv; ++e) vhat[e] = 0.0f;
        for (int64_t d = 0; d < Dqk; ++d) {
          const float kd = k[d];
          const float* srow = &S[static_cast<size_t>(d) * Dv];
          for (int64_t e = 0; e < Dv; ++e) vhat[e] += kd * srow[e];
        }
        for (int64_t e = 0; e < Dv; ++e) target[e] = v[e] - a_t * vhat[e];
      }
      if (has_decay) {
        for (size_t i = 0; i < S.size(); ++i) S[i] *= a_t;
      }

      float dscale = has_beta ? beta[base_s + t] : 1.0f;
      if (modified) {
        float kn = 0.0f, tn = 0.0f;
        for (int64_t d = 0; d < Dqk; ++d) kn += k[d] * k[d];
        for (int64_t e = 0; e < Dv; ++e) tn += target[e] * target[e];
        const float nrm = std::sqrt(kn) * std::sqrt(tn);   // ‖k⊗target‖_F
        dscale /= (1.0f + nrm);
      }
      for (int64_t d = 0; d < Dqk; ++d) {
        const float kd = k[d] * dscale;
        float* srow = &S[static_cast<size_t>(d) * Dv];
        for (int64_t e = 0; e < Dv; ++e) srow[e] += kd * target[e];
      }

      for (int64_t e = 0; e < Dv; ++e) o[e] = 0.0f;
      for (int64_t d = 0; d < Dqk; ++d) {
        const float qd = q[d];
        const float* srow = &S[static_cast<size_t>(d) * Dv];
        for (int64_t e = 0; e < Dv; ++e) o[e] += qd * srow[e];
      }
      if (has_gate) {
        const float* gt = gate + base_v + t * Dv;
        for (int64_t e = 0; e < Dv; ++e) o[e] *= sigmoidf(gt[e]);
      }
    }
  }
}
