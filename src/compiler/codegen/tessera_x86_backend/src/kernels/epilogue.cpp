#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" void tessera_x86_epilogue_bias_fp32(float* C, const float* bias, int M, int N) {
    if (!bias) return;
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            C[i*N + j] += bias[j];
        }
    }
}

// tanh-based GELU approx
static inline float gelu(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x3 = x*x*x;
    float t  = kAlpha * (x + kBeta * x3);
    float g  = 0.5f * x * (1.0f + std::tanh(t));
    return g;
}

extern "C" void tessera_x86_epilogue_bias_gelu_fp32(float* C, const float* bias, int M, int N) {
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            float v = C[i*N + j] + (bias ? bias[j] : 0.0f);
            C[i*N + j] = gelu(v);
        }
    }
}
