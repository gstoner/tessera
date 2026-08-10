// On-device test for the AVX-512 radix-2 C2C FFT vs a scalar DFT reference.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_fft_c2c_f32(float*, int64_t, int64_t, int);
extern "C" int tessera_x86_fft_mixed_c2c_f32(float*, int64_t, int64_t, int);
extern "C" void tessera_x86_avx512_spectral_filter_bwd_c64(
    const float*, const float*, const float*, float*, float*, int64_t);
extern "C" void tessera_x86_avx512_spectral_conv_bwd_f32(
    const float*, const float*, const float*, float*, float*, int64_t,
    int64_t, int64_t, int64_t, float);

static int g_fail = 0;

// naive O(n^2) DFT reference (interleaved complex), sign<0 forward.
static void dft_ref(const std::vector<float>& in, std::vector<float>& out,
                    int64_t n, double sign) {
    out.assign(2 * n, 0.0f);
    for (int64_t k = 0; k < n; ++k) {
        double re = 0, im = 0;
        for (int64_t j = 0; j < n; ++j) {
            double ang = sign * 2.0 * M_PI * (double)(k * j) / (double)n;
            double c = std::cos(ang), s = std::sin(ang);
            re += in[2 * j] * c - in[2 * j + 1] * s;
            im += in[2 * j] * s + in[2 * j + 1] * c;
        }
        out[2 * k] = (float)re; out[2 * k + 1] = (float)im;
    }
}

static void check(int64_t n, int inverse, int64_t batch) {
    std::mt19937 rng((unsigned)(n * 7 + inverse + batch));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> data(batch * 2 * n), orig;
    for (auto& v : data) v = d(rng);
    orig = data;
    tessera_x86_fft_c2c_f32(data.data(), batch, n, inverse);
    double sign = inverse ? 1.0 : -1.0;
    float worst = 0.0f;
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<float> row(orig.begin() + b * 2 * n,
                               orig.begin() + (b + 1) * 2 * n), ref;
        dft_ref(row, ref, n, sign);
        for (int64_t i = 0; i < 2 * n; ++i) {
            float got = data[b * 2 * n + i];
            float tol = 2e-3f + 2e-3f * std::fabs(ref[i]);
            if (std::fabs(got - ref[i]) > tol) {
                std::printf("FAIL n=%lld inv=%d b=%lld i=%lld got=%g want=%g\n",
                            (long long)n, inverse, (long long)b, (long long)i,
                            got, ref[i]); ++g_fail; return;
            }
            worst = std::fmax(worst, std::fabs(got - ref[i]));
        }
    }
    std::printf("ok   n=%-5lld inv=%d batch=%lld worst=%.2e\n", (long long)n,
                inverse, (long long)batch, worst);
}

static void check_mixed(int64_t n, int inverse, int64_t batch) {
    std::mt19937 rng((unsigned)(n * 11 + inverse + batch));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> data(batch * 2 * n), orig;
    for (auto& v : data) v = d(rng);
    orig = data;
    if (tessera_x86_fft_mixed_c2c_f32(data.data(), batch, n, inverse) != 0) {
        std::printf("FAIL mixed n=%lld declined\n", (long long)n);
        ++g_fail;
        return;
    }
    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<float> row(orig.begin() + b * 2 * n,
                               orig.begin() + (b + 1) * 2 * n), ref;
        dft_ref(row, ref, n, sign);
        for (int64_t i = 0; i < 2 * n; ++i) {
            const float got = data[b * 2 * n + i];
            const float tol = 4e-3f + 4e-3f * std::fabs(ref[i]);
            if (std::fabs(got - ref[i]) > tol) {
                std::printf("FAIL mixed n=%lld inv=%d b=%lld i=%lld got=%g want=%g\n",
                            (long long)n, inverse, (long long)b, (long long)i,
                            got, ref[i]);
                ++g_fail;
                return;
            }
        }
    }
    std::printf("ok   mixed n=%-5lld inv=%d batch=%lld\n",
                (long long)n, inverse, (long long)batch);
}

static void check_compound_backward() {
    const float dy[] = {2.0f, -1.0f, 0.5f, 3.0f};
    const float x[] = {1.0f, 4.0f, -2.0f, 0.25f};
    const float filter[] = {3.0f, 2.0f, -1.0f, 5.0f};
    float dx[4] = {}, df[4] = {};
    tessera_x86_avx512_spectral_filter_bwd_c64(
        dy, x, filter, dx, df, 2);
    const float wantDx[] = {4.0f, -7.0f, 14.5f, -5.5f};
    const float wantDf[] = {-2.0f, -9.0f, -0.25f, -6.125f};
    for (int i = 0; i < 4; ++i)
        if (std::fabs(dx[i] - wantDx[i]) > 1e-6f ||
            std::fabs(df[i] - wantDf[i]) > 1e-6f)
            ++g_fail;

    const float realDy[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float realX[] = {2.0f, -1.0f, 3.0f};
    const float realKernel[] = {4.0f, 5.0f};
    float realDx[3] = {}, realDk[2] = {};
    tessera_x86_avx512_spectral_conv_bwd_f32(
        realDy, realX, realKernel, realDx, realDk, 1, 4, 3, 2, 0.5f);
    const float wantRealDx[] = {7.0f, 11.5f, 16.0f};
    const float wantRealDk[] = {4.5f, 6.5f};
    for (int i = 0; i < 3; ++i)
        if (std::fabs(realDx[i] - wantRealDx[i]) > 1e-6f) ++g_fail;
    for (int i = 0; i < 2; ++i)
        if (std::fabs(realDk[i] - wantRealDk[i]) > 1e-6f) ++g_fail;
}

int main() {
    for (int64_t n : {2, 4, 8, 16, 64, 256, 1024}) {
        check(n, 0, 3);
        check(n, 1, 3);
    }
    check(4096, 0, 1);
    // Exercise every compile-time codelet admitted by FFTPlan, including its
    // intentionally supported composite odd divisors 9 and 15.  N=68 also
    // proves a multi-stage radix-4/radix-17 transform rather than isolated
    // one-stage DFTs only.
    for (int64_t n : {2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 68}) {
        check_mixed(n, 0, 2);
        check_mixed(n, 1, 2);
    }
    check_compound_backward();
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
