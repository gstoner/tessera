// AVX-512 fused optimizer-step kernels (f32) — the optimized CPU lane for the
// S-series `functional_optimizer_step` family (P3 of S_SERIES_GAP_CLOSURE_PLAN).
// A single flat per-parameter update over one parameter tensor; the pytree
// orchestration + the bias-correction scalars (1-β^t) are computed on the host
// (the optimizer carries `step`), so the kernel is pure elementwise.
//
//   kind 0 sgd      : p -= lr·g
//   kind 1 momentum : v = μ·v + g ; p -= lr·v
//   kind 2 adam     : m = β1·m+(1-β1)g ; v = β2·v+(1-β2)g² ;
//                     p -= lr·(m/b1c)/(√(v/b2c)+eps)
//   kind 3 adamw    : p *= (1-lr·wd) ; then the adam update (decoupled decay)
//   kind 4 lion     : u = β1·m+(1-β1)g ; m = β2·m+(1-β2)g ;
//                     p *= (1-lr·wd) ; p -= lr·sign(u)
//
// `m`/`v` are in/out state buffers (written to m_out/v_out, may alias in). β1/β2
// = momentum/decay; for momentum, μ is passed as beta1 and the v slot is used.
// √ via the exact hardware `_mm512_sqrt_ps`. A scalar reference is alongside.

#include <immintrin.h>
#include <cstddef>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSgd = 0, kMomentum = 1, kAdam = 2, kAdamW = 3, kLion = 4,
              kNesterov = 5;

inline float signf(float x) { return (x > 0.0f) - (x < 0.0f); }
}  // namespace

extern "C" void tessera_x86_reference_optimizer_f32(
    const float* p, const float* g, const float* m, const float* v, int64_t n,
    int kind, float lr, float beta1, float beta2, float eps, float wd,
    float b1c, float b2c, float* p_out, float* m_out, float* v_out) {
    for (int64_t i = 0; i < n; ++i) {
        float pi = p[i], gi = g[i];
        if (kind == kSgd) { p_out[i] = pi - lr * gi; continue; }
        if (kind == kMomentum) {
            float vv = beta1 * v[i] + gi;
            v_out[i] = vv;
            p_out[i] = pi - lr * vv;
            continue;
        }
        if (kind == kNesterov) {
            float vv = beta1 * v[i] + gi;            // new velocity
            v_out[i] = vv;
            p_out[i] = pi - lr * (gi + beta1 * vv);  // look-ahead update
            continue;
        }
        if (kind == kLion) {
            float u = beta1 * m[i] + (1.0f - beta1) * gi;
            m_out[i] = beta2 * m[i] + (1.0f - beta2) * gi;
            if (wd != 0.0f) pi *= (1.0f - lr * wd);
            p_out[i] = pi - lr * signf(u);
            continue;
        }
        // adam / adamw
        float mm = beta1 * m[i] + (1.0f - beta1) * gi;
        float vv = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m_out[i] = mm;
        v_out[i] = vv;
        if (kind == kAdamW && wd != 0.0f) pi *= (1.0f - lr * wd);
        float upd = (mm / b1c) / (std::sqrt(vv / b2c) + eps);
        p_out[i] = pi - lr * upd;
    }
}

extern "C" void tessera_x86_optimizer_f32(
    const float* p, const float* g, const float* m, const float* v, int64_t n,
    int kind, float lr, float beta1, float beta2, float eps, float wd,
    float b1c, float b2c, float* p_out, float* m_out, float* v_out) {
    const __m512 vlr = _mm512_set1_ps(lr);
    const __m512 vb1 = _mm512_set1_ps(beta1);
    const __m512 vb2 = _mm512_set1_ps(beta2);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vom1 = _mm512_set1_ps(1.0f - beta1);
    const __m512 vom2 = _mm512_set1_ps(1.0f - beta2);
    const __m512 veps = _mm512_set1_ps(eps);
    const __m512 vb1c = _mm512_set1_ps(b1c);
    const __m512 vb2c = _mm512_set1_ps(b2c);
    const __m512 vdecay = _mm512_set1_ps(1.0f - lr * wd);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 pi = _mm512_loadu_ps(p + i);
        __m512 gi = _mm512_loadu_ps(g + i);
        if (kind == kSgd) {
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, gi, pi));
            continue;
        }
        if (kind == kMomentum) {
            __m512 vv = _mm512_fmadd_ps(vb1, _mm512_loadu_ps(v + i), gi);
            _mm512_storeu_ps(v_out + i, vv);
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, vv, pi));
            continue;
        }
        if (kind == kNesterov) {
            __m512 vv = _mm512_fmadd_ps(vb1, _mm512_loadu_ps(v + i), gi);
            _mm512_storeu_ps(v_out + i, vv);
            __m512 upd = _mm512_fmadd_ps(vb1, vv, gi);   // g + beta1*vv
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, upd, pi));
            continue;
        }
        if (kind == kLion) {
            __m512 mi = _mm512_loadu_ps(m + i);
            __m512 u = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(vom1, gi));
            _mm512_storeu_ps(m_out + i,
                             _mm512_fmadd_ps(vb2, mi, _mm512_mul_ps(vom2, gi)));
            // sign(u) = (u>0) - (u<0)
            __mmask16 pos = _mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_GT_OQ);
            __mmask16 neg = _mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_LT_OQ);
            __m512 sgn = _mm512_mask_blend_ps(
                pos, _mm512_mask_blend_ps(neg, _mm512_setzero_ps(),
                                          _mm512_set1_ps(-1.0f)),
                vone);
            if (wd != 0.0f) pi = _mm512_mul_ps(pi, vdecay);
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, sgn, pi));
            continue;
        }
        // adam / adamw
        __m512 mi = _mm512_loadu_ps(m + i);
        __m512 vi = _mm512_loadu_ps(v + i);
        __m512 mm = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(vom1, gi));
        __m512 vv = _mm512_fmadd_ps(vb2, vi,
                                    _mm512_mul_ps(vom2, _mm512_mul_ps(gi, gi)));
        _mm512_storeu_ps(m_out + i, mm);
        _mm512_storeu_ps(v_out + i, vv);
        if (kind == kAdamW && wd != 0.0f) pi = _mm512_mul_ps(pi, vdecay);
        __m512 denom = _mm512_add_ps(
            _mm512_sqrt_ps(_mm512_div_ps(vv, vb2c)), veps);
        __m512 upd = _mm512_div_ps(_mm512_div_ps(mm, vb1c), denom);
        _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, upd, pi));
    }
    if (i < n)
        tessera_x86_reference_optimizer_f32(
            p + i, g + i, m + i, v + i, n - i, kind, lr, beta1, beta2, eps, wd,
            b1c, b2c, p_out + i, m_out + i, v_out + i);
}

extern "C" void tessera_x86_avx512_sgd_bwd_f32(
    const float* dy, int64_t n, float lr, float* dparam, float* dgrad) {
    const __m512 neg_lr = _mm512_set1_ps(-lr);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 incoming = _mm512_loadu_ps(dy + i);
        _mm512_storeu_ps(dparam + i, incoming);
        _mm512_storeu_ps(dgrad + i, _mm512_mul_ps(neg_lr, incoming));
    }
    for (; i < n; ++i) {
        dparam[i] = dy[i];
        dgrad[i] = -lr * dy[i];
    }
}

extern "C" void tessera_x86_avx512_momentum_bwd_f32(
    const float* dparam_out, const float* dvelocity_out, int64_t n,
    float lr, float momentum, int nesterov, float* dparam, float* dgrad,
    float* dvelocity) {
    const __m512 neg_lr = _mm512_set1_ps(-lr);
    const __m512 mu = _mm512_set1_ps(momentum);
    const __m512 grad_factor =
        _mm512_set1_ps(nesterov ? 1.0f + momentum : 1.0f);
    const __m512 velocity_factor =
        _mm512_set1_ps(nesterov ? momentum : 1.0f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 dp = _mm512_loadu_ps(dparam_out + i);
        const __m512 dv = _mm512_loadu_ps(dvelocity_out + i);
        const __m512 from_param = _mm512_mul_ps(neg_lr, dp);
        _mm512_storeu_ps(dparam + i, dp);
        _mm512_storeu_ps(
            dgrad + i, _mm512_fmadd_ps(grad_factor, from_param, dv));
        const __m512 velocity_base =
            _mm512_fmadd_ps(velocity_factor, from_param, dv);
        _mm512_storeu_ps(dvelocity + i, _mm512_mul_ps(mu, velocity_base));
    }
    for (; i < n; ++i) {
        const float from_param = -lr * dparam_out[i];
        dparam[i] = dparam_out[i];
        dgrad[i] = (nesterov ? 1.0f + momentum : 1.0f) * from_param
                   + dvelocity_out[i];
        dvelocity[i] = momentum *
            ((nesterov ? momentum : 1.0f) * from_param + dvelocity_out[i]);
    }
}

extern "C" void tessera_x86_avx512_lion_bwd_f32(
    const float* dparam_out, const float* dmoment_out, int64_t n, float lr,
    float beta2, float weight_decay, float* dparam, float* dgrad,
    float* dmoment) {
    const __m512 parameter_factor =
        _mm512_set1_ps(1.0f - lr * weight_decay);
    const __m512 gradient_factor = _mm512_set1_ps(1.0f - beta2);
    const __m512 moment_factor = _mm512_set1_ps(beta2);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 dp = _mm512_loadu_ps(dparam_out + i);
        const __m512 dm = _mm512_loadu_ps(dmoment_out + i);
        _mm512_storeu_ps(dparam + i, _mm512_mul_ps(parameter_factor, dp));
        _mm512_storeu_ps(dgrad + i, _mm512_mul_ps(gradient_factor, dm));
        _mm512_storeu_ps(dmoment + i, _mm512_mul_ps(moment_factor, dm));
    }
    for (; i < n; ++i) {
        dparam[i] = (1.0f - lr * weight_decay) * dparam_out[i];
        dgrad[i] = (1.0f - beta2) * dmoment_out[i];
        dmoment[i] = beta2 * dmoment_out[i];
    }
}

extern "C" void tessera_x86_avx512_adafactor_factored_f32(
    const float* parameter, const float* gradient, const float* old_row,
    const float* old_col, int64_t rows, int64_t cols, float lr, float beta2,
    float eps, float* output, float* new_row, float* new_col) {
    const float one_minus_beta2 = 1.0f - beta2;
    for (int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            const float g = gradient[r * cols + c];
            sum += g * g;
        }
        new_row[r] = beta2 * old_row[r]
                   + one_minus_beta2 * (sum / static_cast<float>(cols));
    }
    for (int64_t c = 0; c < cols; ++c) {
        float sum = 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            const float g = gradient[r * cols + c];
            sum += g * g;
        }
        new_col[c] = beta2 * old_col[c]
                   + one_minus_beta2 * (sum / static_cast<float>(rows));
    }
    float row_mean = 0.0f;
    for (int64_t r = 0; r < rows; ++r) row_mean += new_row[r];
    row_mean /= static_cast<float>(rows);
    const float safe_mean = std::fmax(row_mean, eps);
    for (int64_t r = 0; r < rows; ++r) {
        const float safe_row = std::fmax(new_row[r], eps);
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t i = r * cols + c;
            const float scale =
                safe_row * std::fmax(new_col[c], eps) / safe_mean;
            const float denom = std::sqrt(scale) + eps;
            output[i] = parameter[i] - lr * gradient[i] / denom;
        }
    }
}

extern "C" void tessera_x86_avx512_adafactor_full_f32(
    const float* parameter, const float* gradient, const float* old_moment,
    int64_t n, float lr, float beta2, float eps, float* output,
    float* new_moment) {
    const float one_minus_beta2 = 1.0f - beta2;
    for (int64_t i = 0; i < n; ++i) {
        const float g = gradient[i];
        const float moment =
            beta2 * old_moment[i] + one_minus_beta2 * g * g;
        new_moment[i] = moment;
        output[i] =
            parameter[i] - lr * g / (std::sqrt(std::fmax(moment, eps)) + eps);
    }
}

extern "C" void tessera_x86_avx512_adafactor_factored_bwd_f32(
    const float* gradient, const float* old_row, const float* old_col,
    const float* cotangent, int64_t rows, int64_t cols, float lr, float beta2,
    float eps, float* dparam, float* dgrad, float* d_old_row,
    float* d_old_col) {
    float* row = new float[static_cast<size_t>(rows)];
    float* col = new float[static_cast<size_t>(cols)];
    float* drow = new float[static_cast<size_t>(rows)]();
    float* dcol = new float[static_cast<size_t>(cols)]();
    const float one_minus_beta2 = 1.0f - beta2;
    for (int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            const float g = gradient[r * cols + c];
            sum += g * g;
        }
        row[r] = beta2 * old_row[r]
               + one_minus_beta2 * sum / static_cast<float>(cols);
    }
    for (int64_t c = 0; c < cols; ++c) {
        float sum = 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            const float g = gradient[r * cols + c];
            sum += g * g;
        }
        col[c] = beta2 * old_col[c]
               + one_minus_beta2 * sum / static_cast<float>(rows);
    }
    float mean = 0.0f;
    for (int64_t r = 0; r < rows; ++r) mean += row[r];
    mean /= static_cast<float>(rows);
    const float safe_mean = std::fmax(mean, eps);
    float dmean = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        const float safe_row = std::fmax(row[r], eps);
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t i = r * cols + c;
            const float safe_col = std::fmax(col[c], eps);
            const float scale = safe_row * safe_col / safe_mean;
            const float root = std::sqrt(scale);
            const float denom = root + eps;
            const float ds = lr * cotangent[i] * gradient[i]
                           / (2.0f * std::fmax(root, 1.0e-30f)
                              * denom * denom);
            drow[r] += ds * safe_col / safe_mean;
            dcol[c] += ds * safe_row / safe_mean;
            dmean -= ds * safe_row * safe_col / (safe_mean * safe_mean);
        }
    }
    for (int64_t r = 0; r < rows; ++r) {
        if (row[r] > eps && mean > eps)
            drow[r] += dmean / static_cast<float>(rows);
        else if (row[r] <= eps)
            drow[r] = 0.0f;
        d_old_row[r] = beta2 * drow[r];
    }
    for (int64_t c = 0; c < cols; ++c) {
        if (col[c] <= eps) dcol[c] = 0.0f;
        d_old_col[c] = beta2 * dcol[c];
    }
    for (int64_t r = 0; r < rows; ++r) {
        const float safe_row = std::fmax(row[r], eps);
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t i = r * cols + c;
            const float scale =
                safe_row * std::fmax(col[c], eps) / safe_mean;
            const float denom = std::sqrt(scale) + eps;
            const float state =
                drow[r] / static_cast<float>(cols)
                + dcol[c] / static_cast<float>(rows);
            dparam[i] = cotangent[i];
            dgrad[i] = -lr * cotangent[i] / denom
                     + 2.0f * one_minus_beta2 * gradient[i] * state;
        }
    }
    delete[] row;
    delete[] col;
    delete[] drow;
    delete[] dcol;
}

extern "C" void tessera_x86_avx512_adafactor_full_bwd_f32(
    const float* gradient, const float* old_moment, const float* cotangent,
    int64_t n, float lr, float beta2, float eps, float* dparam, float* dgrad,
    float* d_old_moment) {
    const float one_minus_beta2 = 1.0f - beta2;
    for (int64_t i = 0; i < n; ++i) {
        const float g = gradient[i];
        const float moment =
            beta2 * old_moment[i] + one_minus_beta2 * g * g;
        const float root = std::sqrt(std::fmax(moment, eps));
        const float denom = root + eps;
        const float dm =
            moment > eps
                ? lr * cotangent[i] * g
                      / (2.0f * std::fmax(root, 1.0e-30f)
                         * denom * denom)
                : 0.0f;
        dparam[i] = cotangent[i];
        dgrad[i] = -lr * cotangent[i] / denom
                 + 2.0f * one_minus_beta2 * g * dm;
        d_old_moment[i] = beta2 * dm;
    }
}
