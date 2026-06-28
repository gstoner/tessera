// On-device test for the AVX-512 radix-2 C2C FFT vs a scalar DFT reference.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_fft_c2c_f32(float*, int64_t, int64_t, int);

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

int main() {
    for (int64_t n : {2, 4, 8, 16, 64, 256, 1024}) {
        check(n, 0, 3);
        check(n, 1, 3);
    }
    check(4096, 0, 1);
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
