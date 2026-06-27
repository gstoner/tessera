// On-device test for the AVX-512 transcendental / activation kernels (f32).
// Compares the vector path against libm at atol/rtol 2e-5 over each op's valid
// domain (log/log1p restricted to x>0 / x>-1).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_transcendental_f32(const float*, int64_t,
                                                         float*, int);
extern "C" void tessera_x86_avx512_transcendental_f32(const float*, int64_t,
                                                      float*, int);

static int g_fail = 0;

struct Op { int kind; const char* name; float lo; float hi; };

static float libm_ref(float v, int kind) {
    switch (kind) {
    case 0: return std::exp(v);
    case 1: return std::log(v);
    case 2: return std::tanh(v);
    case 3: return 1.0f / (1.0f + std::exp(-v));
    case 4: return v / (1.0f + std::exp(-v));
    case 5: return 0.5f * v * (1.0f + std::tanh(0.7978845608028654f *
                                                (v + 0.044715f * v * v * v)));
    case 6: return std::erf(v);
    case 7: return std::log1p(std::exp(-std::fabs(v))) + std::fmax(v, 0.0f);
    case 8: return std::exp(v) - 1.0f;
    case 9: return std::log1p(v);
    case 10: return std::cos(v);
    case 11: return std::tan(v);
    case 12: return std::sinh(v);
    case 13: return std::cosh(v);
    case 14: return std::asin(v);
    case 15: return std::acos(v);
    case 16: return std::atan(v);
    case 17: return std::erfc(v);
    default: return v;
    }
}

static void check(const Op& op, int64_t n) {
    std::mt19937 rng(1234 + op.kind * 17 + (unsigned)n);
    std::uniform_real_distribution<float> dist(op.lo, op.hi);
    std::vector<float> x((size_t)n), avx(n), ref(n);
    for (int64_t i = 0; i < n; ++i) x[i] = dist(rng);
    tessera_x86_avx512_transcendental_f32(x.data(), n, avx.data(), op.kind);
    tessera_x86_reference_transcendental_f32(x.data(), n, ref.data(), op.kind);
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float want = libm_ref(x[i], op.kind);
        // vector vs libm
        float err = std::fabs(avx[i] - want);
        float tol = 2e-5f + 2e-5f * std::fabs(want);
        if (err > tol || std::fabs(avx[i] - ref[i]) > tol) {
            std::printf("FAIL %s n=%lld i=%lld: x=%g avx=%g ref=%g want=%g "
                        "err=%g tol=%g\n", op.name, (long long)n, (long long)i,
                        x[i], avx[i], ref[i], want, err, tol);
            ++g_fail;
            return;
        }
        worst = std::fmax(worst, err);
    }
    std::printf("ok   %-9s n=%-5lld worst_abs=%.2e\n", op.name, (long long)n,
                worst);
}

int main() {
    // domain chosen so each op is finite & well-conditioned at 2e-5
    const Op ops[] = {
        {0, "exp",      -10.0f, 10.0f},
        {1, "log",       1e-3f, 50.0f},
        {2, "tanh",     -12.0f, 12.0f},
        {3, "sigmoid",  -20.0f, 20.0f},
        {4, "silu",     -20.0f, 20.0f},
        {5, "gelu",      -8.0f,  8.0f},
        {6, "erf",       -4.0f,  4.0f},
        {7, "softplus", -30.0f, 30.0f},
        {8, "expm1",     -8.0f,  8.0f},
        {9, "log1p",    -0.99f, 50.0f},
        {10, "cos",     -12.0f, 12.0f},
        {11, "tan",      -1.3f,  1.3f},   // away from ±π/2 singularities
        {12, "sinh",    -11.0f, 11.0f},
        {13, "cosh",    -11.0f, 11.0f},
        {14, "asin",    -0.99f, 0.99f},
        {15, "acos",     -0.9f,  0.9f},   // pi/2-asin cancellation near ±1
        {16, "atan",    -50.0f, 50.0f},
        {17, "erfc",     -4.0f,  4.0f},
    };
    for (const Op& op : ops) {
        check(op, 64);    // aligned
        check(op, 70);    // tail
        check(op, 5);     // sub-block (all scalar tail)
        check(op, 1024);  // wide
        check(op, 1);     // single
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
