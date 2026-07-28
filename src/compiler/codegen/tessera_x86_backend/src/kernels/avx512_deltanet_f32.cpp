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
#include <algorithm>
#include <thread>
#include <vector>

namespace {
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

template <typename Fn>
static void parallel_for(int64_t count, int32_t workers, Fn fn) {
  workers = std::max<int32_t>(1, std::min<int64_t>(workers, count));
  if (workers == 1) {
    for (int64_t i = 0; i < count; ++i) fn(i);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (int32_t worker = 0; worker < workers; ++worker)
    pool.emplace_back([=, &fn]() {
      for (int64_t i = worker; i < count; i += workers) fn(i);
    });
  for (auto& thread : pool) thread.join();
}

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

// Resident-workspace reverse program. For erase-free recurrences each chunk
// independently constructs the affine state transform
//   state_out = chunk_scale * state_in + chunk_update
// and a deterministic prefix composes chunk boundaries. Token checkpoints are
// then materialized in parallel from those boundaries. Erase-dependent chunks
// use the exact sequential checkpoint because target depends on incoming state.
extern "C" void tessera_x86_deltanet_bwd_f32(
    const float* Q, const float* K, const float* V,
    const float* gate, const float* beta, const float* decay, const float* DY,
    float* DQ, float* DK, float* DV, float* DG, float* DB, float* DA,
    float* HIST, float* DS, float* DN, float* DU,
    float* chunk_scale, float* chunk_update,
    int64_t B, int64_t Hh, int64_t Sseq, int64_t Dqk, int64_t Dv,
    int32_t erase, int32_t modified,
    int32_t has_gate, int32_t has_beta, int32_t has_decay,
    int64_t chunk_size, int32_t workers) {
  const int64_t bh = B * Hh;
  const int64_t state_size = Dqk * Dv;
  const int64_t chunks = (Sseq + chunk_size - 1) / chunk_size;
  const int64_t hist_stride = (Sseq + 1) * state_size;

  auto target_at = [&](int64_t g, int64_t t, const float* state,
                       int64_t e) {
    const int64_t qk = (g * Sseq + t) * Dqk;
    const int64_t vv = (g * Sseq + t) * Dv;
    float target = V[vv + e];
    if (erase) {
      float vhat = 0.0f;
      for (int64_t d = 0; d < Dqk; ++d)
        vhat += K[qk + d] * state[d * Dv + e];
      const float a = has_decay ? decay[g * Sseq + t] : 1.0f;
      target -= a * vhat;
    }
    return target;
  };
  auto inverse_denom = [&](int64_t g, int64_t t, const float* state) {
    if (!modified) return 1.0f;
    const int64_t qk = (g * Sseq + t) * Dqk;
    float norm2 = 0.0f;
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e) {
        const float update = K[qk + d] * target_at(g, t, state, e);
        norm2 += update * update;
      }
    return 1.0f / (1.0f + std::sqrt(norm2));
  };
  auto advance = [&](int64_t g, int64_t t, const float* old, float* next) {
    const int64_t qk = (g * Sseq + t) * Dqk;
    const float a = has_decay ? decay[g * Sseq + t] : 1.0f;
    const float weight = has_beta ? beta[g * Sseq + t] : 1.0f;
    const float inv = inverse_denom(g, t, old);
    for (int64_t d = 0; d < Dqk; ++d)
      for (int64_t e = 0; e < Dv; ++e)
        next[d * Dv + e] =
            a * old[d * Dv + e] +
            weight * inv * K[qk + d] * target_at(g, t, old, e);
  };

  const bool compose_chunks =
      !erase && chunks > 1 && workers > 1 && chunk_scale && chunk_update;
  if (compose_chunks) {
    parallel_for(bh * chunks, workers, [&](int64_t task) {
      const int64_t g = task / chunks;
      const int64_t chunk = task % chunks;
      const int64_t begin = chunk * chunk_size;
      const int64_t end = std::min<int64_t>(Sseq, begin + chunk_size);
      float* update = chunk_update + task * state_size;
      std::fill(update, update + state_size, 0.0f);
      float scale = 1.0f;
      for (int64_t t = begin; t < end; ++t) {
        const int64_t qk = (g * Sseq + t) * Dqk;
        const int64_t vv = (g * Sseq + t) * Dv;
        const float a = has_decay ? decay[g * Sseq + t] : 1.0f;
        const float weight = has_beta ? beta[g * Sseq + t] : 1.0f;
        float inv = 1.0f;
        if (modified) {
          float norm2 = 0.0f;
          for (int64_t d = 0; d < Dqk; ++d)
            for (int64_t e = 0; e < Dv; ++e) {
              const float value = K[qk + d] * V[vv + e];
              norm2 += value * value;
            }
          inv = 1.0f / (1.0f + std::sqrt(norm2));
        }
        for (int64_t i = 0; i < state_size; ++i) update[i] *= a;
        for (int64_t d = 0; d < Dqk; ++d)
          for (int64_t e = 0; e < Dv; ++e)
            update[d * Dv + e] +=
                weight * inv * K[qk + d] * V[vv + e];
        scale *= a;
      }
      chunk_scale[task] = scale;
    });
    for (int64_t g = 0; g < bh; ++g) {
      float* state = HIST + g * hist_stride;
      std::fill(state, state + state_size, 0.0f);
      for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        const int64_t task = g * chunks + chunk;
        float* next = HIST + g * hist_stride +
                      std::min<int64_t>(Sseq, (chunk + 1) * chunk_size) *
                          state_size;
        const float* update = chunk_update + task * state_size;
        for (int64_t i = 0; i < state_size; ++i)
          next[i] = chunk_scale[task] * state[i] + update[i];
        state = next;
      }
    }
    parallel_for(bh * chunks, workers, [&](int64_t task) {
      const int64_t g = task / chunks;
      const int64_t chunk = task % chunks;
      const int64_t begin = chunk * chunk_size;
      const int64_t end = std::min<int64_t>(Sseq, begin + chunk_size);
      std::vector<float> current(state_size), next_state(state_size);
      std::copy(HIST + g * hist_stride + begin * state_size,
                HIST + g * hist_stride + (begin + 1) * state_size,
                current.begin());
      for (int64_t t = begin; t < end; ++t) {
        advance(g, t, current.data(), next_state.data());
        if (t + 1 < end)
          std::copy(next_state.begin(), next_state.end(),
                    HIST + g * hist_stride + (t + 1) * state_size);
        current.swap(next_state);
      }
    });
  } else {
    parallel_for(bh, workers, [&](int64_t g) {
      float* state = HIST + g * hist_stride;
      std::fill(state, state + state_size, 0.0f);
      for (int64_t t = 0; t < Sseq; ++t)
        advance(g, t, state + t * state_size,
                state + (t + 1) * state_size);
    });
  }

  parallel_for(bh, workers, [&](int64_t g) {
    float* ds = DS + g * state_size;
    float* dn = DN + g * state_size;
    float* du = DU + g * state_size;
    std::fill(ds, ds + state_size, 0.0f);
    const int64_t qk_base = g * Sseq * Dqk;
    const int64_t v_base = g * Sseq * Dv;
    for (int64_t chunk_rev = 0; chunk_rev < chunks; ++chunk_rev) {
      const int64_t chunk = chunks - 1 - chunk_rev;
      const int64_t begin = chunk * chunk_size;
      const int64_t end = std::min<int64_t>(Sseq, begin + chunk_size);
      for (int64_t t = end; t-- > begin;) {
        const int64_t qk = qk_base + t * Dqk;
        const int64_t vv = v_base + t * Dv;
        const int64_t scalar = g * Sseq + t;
        const float* old = HIST + g * hist_stride + t * state_size;
        const float* next = old + state_size;
        const float a = has_decay ? decay[scalar] : 1.0f;
        const float weight = has_beta ? beta[scalar] : 1.0f;
        for (int64_t e = 0; e < Dv; ++e) {
          float raw = 0.0f;
          for (int64_t d = 0; d < Dqk; ++d)
            raw += Q[qk + d] * next[d * Dv + e];
          float dy = DY[vv + e];
          if (has_gate) {
            const float sig = sigmoidf(gate[vv + e]);
            DG[vv + e] = dy * raw * sig * (1.0f - sig);
            dy *= sig;
          }
          for (int64_t d = 0; d < Dqk; ++d)
            dn[d * Dv + e] = ds[d * Dv + e] + Q[qk + d] * dy;
        }
        const float inv = inverse_denom(g, t, old);
        float projection = 0.0f;
        if (modified)
          for (int64_t d = 0; d < Dqk; ++d)
            for (int64_t e = 0; e < Dv; ++e)
              projection +=
                  weight * dn[d * Dv + e] * K[qk + d] *
                  target_at(g, t, old, e);
        const float norm = modified ? (1.0f / inv - 1.0f) : 0.0f;
        const float safe_norm = std::max(norm, 1.0f);
        for (int64_t d = 0; d < Dqk; ++d)
          for (int64_t e = 0; e < Dv; ++e) {
            float value = weight * dn[d * Dv + e];
            if (modified) {
              const float update =
                  K[qk + d] * target_at(g, t, old, e);
              const float correction =
                  norm > 0.0f ? update * projection * inv * inv / safe_norm
                              : 0.0f;
              value = value * inv - correction;
            }
            du[d * Dv + e] = value;
          }
        for (int64_t e = 0; e < Dv; ++e) {
          float dtarget = 0.0f;
          for (int64_t d = 0; d < Dqk; ++d)
            dtarget += du[d * Dv + e] * K[qk + d];
          DV[vv + e] = dtarget;
        }
        float db = 0.0f, da = 0.0f;
        for (int64_t d = 0; d < Dqk; ++d) {
          float dq = 0.0f, dk = 0.0f;
          for (int64_t e = 0; e < Dv; ++e) {
            float dy = DY[vv + e];
            if (has_gate) dy *= sigmoidf(gate[vv + e]);
            dq += next[d * Dv + e] * dy;
            const float target = target_at(g, t, old, e);
            dk += du[d * Dv + e] * target;
            float previous = a * dn[d * Dv + e];
            if (erase) {
              float vhat = 0.0f;
              for (int64_t rd = 0; rd < Dqk; ++rd)
                vhat += K[qk + rd] * old[rd * Dv + e];
              const float dvhat = -a * DV[vv + e];
              dk += old[d * Dv + e] * dvhat;
              previous += K[qk + d] * dvhat;
              if (d == 0) da -= DV[vv + e] * vhat;
            }
            ds[d * Dv + e] = previous;
            db += dn[d * Dv + e] * K[qk + d] * target * inv;
            da += dn[d * Dv + e] * old[d * Dv + e];
          }
          DQ[qk + d] = dq;
          DK[qk + d] = dk;
        }
        DB[scalar] = db;
        DA[scalar] = da;
      }
    }
  });
}
