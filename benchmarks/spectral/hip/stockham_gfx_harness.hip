//===- stockham_gfx_harness.hip ------------------------------*- HIP -*-===//
//
// GPU correctness + roofline harness for the Spectral solver's mixed-radix
// Stockham FFT on AMD ROCm (proven on gfx1151 / Strix Halo, RDNA 3.5).
//
// It links the *shipping* target hook — lib/TargetHooks/AMD/StockhamRadix4.hip
// (ts_fft_stockham_amd) — so it certifies exactly the kernels the compiler
// lowers to, not a private copy.  For each size N it:
//
//   * runs the fp32 GPU FFT,
//   * validates against an fp64 CPU reference (naive O(N^2) DFT for the small
//     correctness sizes; the same-math fp64 Stockham for large perf sizes),
//   * validates a forward+inverse round-trip back to the input,
//   * times the GPU FFT and reports achieved GFLOP/s (5 N log2 N model, the
//     same estimate benchmarks/spectral/spectral_math.py uses) and effective
//     GB/s (device traffic ~= 2*log-stages * N * 8 bytes).
//
// Output is machine-parseable key=value (scraped by spectral_correctness.py /
// gen_roofline.py) and exits non-zero on any correctness regression.
//
// Build:
//   hipcc --offload-arch=gfx1151 -O3 -std=c++17 \
//     benchmarks/spectral/hip/stockham_gfx_harness.hip \
//     src/solvers/spectral/lib/TargetHooks/AMD/StockhamRadix4.hip \
//     -o /tmp/stockham_gfx_harness
// Run (WSL2 needs the DXG shim):
//   HSA_ENABLE_DXG_DETECTION=1 /tmp/stockham_gfx_harness
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using cf64 = std::complex<double>;
static const double kPi = 3.14159265358979323846;

// Declared in the shipping target hook (StockhamRadix4.hip).
extern "C" void ts_fft_stockham_amd(const float2 *d_in, float2 *d_out,
                                    float2 *d_scratch, int N, int sign,
                                    hipStream_t stream);

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                         \
    if (e != hipSuccess) {                                                      \
      printf("hip_error=%s at %s:%d\n", hipGetErrorString(e), __FILE__,         \
             __LINE__);                                                         \
      std::exit(3);                                                             \
    }                                                                           \
  } while (0)

static std::vector<cf64> make_input(int N, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<cf64> x(N);
  for (auto &v : x) v = cf64(d(rng), d(rng));
  return x;
}

// fp64 naive O(N^2) DFT — golden reference for small correctness sizes.
static std::vector<cf64> naive_dft(const std::vector<cf64> &x) {
  int N = (int)x.size();
  std::vector<cf64> y(N);
  for (int k = 0; k < N; ++k) {
    cf64 acc(0, 0);
    for (int n = 0; n < N; ++n) {
      double ph = -2.0 * kPi * k * n / N;
      acc += x[n] * cf64(std::cos(ph), std::sin(ph));
    }
    y[k] = acc;
  }
  return y;
}

// fp64 CPU mixed-radix Stockham (same butterfly as the kernels) — golden for
// large sizes where O(N^2) is too slow.  Verified against naive_dft offline.
static void cpu_stage(const cf64 *a, cf64 *b, int N, int L, int radix,
                      int sign) {
  int m = N / (radix * L);
  for (int j = 0; j < L; ++j)
    for (int k = 0; k < m; ++k) {
      int i = k * L + j, o = k * (radix * L) + j;
      if (radix == 4) {
        cf64 c0 = a[i + 0 * L * m], c1 = a[i + 1 * L * m];
        cf64 c2 = a[i + 2 * L * m], c3 = a[i + 3 * L * m];
        double ang = sign * 2.0 * kPi * j / (4.0 * L);
        cf64 w1(std::cos(ang), std::sin(ang)), w2 = w1 * w1, w3 = w2 * w1;
        c1 *= w1; c2 *= w2; c3 *= w3;
        cf64 t0 = c0 + c2, t1 = c0 - c2, t2 = c1 + c3, dd = c1 - c3;
        cf64 t3 = (sign < 0) ? cf64(dd.imag(), -dd.real())
                             : cf64(-dd.imag(), dd.real());
        b[o + 0 * L] = t0 + t2; b[o + 1 * L] = t1 + t3;
        b[o + 2 * L] = t0 - t2; b[o + 3 * L] = t1 - t3;
      } else {
        cf64 c0 = a[i + 0 * L * m], c1 = a[i + 1 * L * m];
        double ang = sign * 2.0 * kPi * j / (2.0 * L);
        c1 *= cf64(std::cos(ang), std::sin(ang));
        b[o + 0 * L] = c0 + c1; b[o + 1 * L] = c0 - c1;
      }
    }
}
static std::vector<cf64> cpu_stockham(std::vector<cf64> x, int sign) {
  int N = (int)x.size();
  std::vector<cf64> y(N);
  cf64 *a = x.data(), *b = y.data();
  int L = 1, n = N;
  auto step = [&](int r) { cpu_stage(a, b, N, L, r, sign); L *= r; std::swap(a, b); };
  while (n % 4 == 0) { step(4); n /= 4; }
  while (n % 2 == 0) { step(2); n /= 2; }
  std::vector<cf64> out(a, a + N);
  if (sign > 0) for (auto &v : out) v /= (double)N;
  return out;
}

struct Result {
  int N;
  double fwd_max_err, roundtrip_err, time_ms, gflops, gbs;
  bool pass;
};

static Result run_size(int N, uint32_t seed, int repeats, int warmup,
                       double tol) {
  auto x = make_input(N, seed);
  std::vector<cf64> ref = (N <= 2048) ? naive_dft(x) : cpu_stockham(x, -1);

  std::vector<float2> h_in(N);
  for (int i = 0; i < N; ++i)
    h_in[i] = make_float2((float)x[i].real(), (float)x[i].imag());

  float2 *d_in, *d_out, *d_scr;
  HIP_CHECK(hipMalloc(&d_in, N * sizeof(float2)));
  HIP_CHECK(hipMalloc(&d_out, N * sizeof(float2)));
  HIP_CHECK(hipMalloc(&d_scr, N * sizeof(float2)));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), N * sizeof(float2),
                      hipMemcpyHostToDevice));

  // Forward transform + correctness.
  ts_fft_stockham_amd(d_in, d_out, d_scr, N, -1, 0);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float2> h_out(N);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, N * sizeof(float2),
                      hipMemcpyDeviceToHost));
  double fwd_max = 0;
  for (int i = 0; i < N; ++i) {
    double dr = h_out[i].x - ref[i].real(), di = h_out[i].y - ref[i].imag();
    fwd_max = std::max(fwd_max, std::sqrt(dr * dr + di * di));
  }

  // Round-trip: inverse of the forward result back to the input.
  ts_fft_stockham_amd(d_out, d_in, d_scr, N, +1, 0);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float2> h_rt(N);
  HIP_CHECK(hipMemcpy(h_rt.data(), d_in, N * sizeof(float2),
                      hipMemcpyDeviceToHost));
  double rt_max = 0;
  for (int i = 0; i < N; ++i) {
    double dr = h_rt[i].x - x[i].real(), di = h_rt[i].y - x[i].imag();
    rt_max = std::max(rt_max, std::sqrt(dr * dr + di * di));
  }

  // Timing (forward only).
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), N * sizeof(float2),
                      hipMemcpyHostToDevice));
  for (int i = 0; i < warmup; ++i)
    ts_fft_stockham_amd(d_in, d_out, d_scr, N, -1, 0);
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  HIP_CHECK(hipEventRecord(t0, 0));
  for (int i = 0; i < repeats; ++i)
    ts_fft_stockham_amd(d_in, d_out, d_scr, N, -1, 0);
  HIP_CHECK(hipEventRecord(t1, 0));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms_total = 0;
  HIP_CHECK(hipEventElapsedTime(&ms_total, t0, t1));
  double ms = ms_total / repeats;

  double flops = 5.0 * N * std::log2((double)N);
  int nstages = 0;
  for (int n = N; n % 4 == 0; n /= 4) ++nstages;
  for (int n = N; n % 4 != 0 && n % 2 == 0; n /= 2) ++nstages;
  double bytes = 2.0 * nstages * (double)N * sizeof(float2); // r+w per stage
  double gflops = flops / (ms * 1e6);
  double gbs = bytes / (ms * 1e6);

  hipFree(d_in); hipFree(d_out); hipFree(d_scr);
  hipEventDestroy(t0); hipEventDestroy(t1);

  bool pass = (fwd_max <= tol) && (rt_max <= tol);
  return {N, fwd_max, rt_max, ms, gflops, gbs, pass};
}

int main(int argc, char **argv) {
  int repeats = 100, warmup = 20;
  double tol = 1e-2; // fp32 GPU vs fp64 ref; loosens with N (see spectral tol)
  (void)argc; (void)argv;

  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    printf("verdict=fail reason=no_device\n");
    return 3;
  }
  printf("tool=ts-spectral-gfx-harness\n");
  printf("backend=amd\n");
  printf("device=%s\n", prop.name);
  printf("arch=%s\n", prop.gcnArchName);
  printf("fast_kernel=ts_fft_stockham_amd\n");
  printf("dtype=complex64\n");
  printf("abs_tol=%g\n", tol);

  int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536,
                 262144, 1048576};
  bool all = true;
  for (int N : sizes) {
    Result r = run_size(N, 0xDEADBEEFu, repeats, warmup, tol);
    printf("size=%d fwd_max_err=%.3e roundtrip_err=%.3e time_ms=%.4f "
           "gflops=%.2f gbs=%.2f pass=%d\n",
           r.N, r.fwd_max_err, r.roundtrip_err, r.time_ms, r.gflops, r.gbs,
           r.pass ? 1 : 0);
    all &= r.pass;
  }
  printf("verdict=%s\n", all ? "pass" : "fail");
  return all ? 0 : 1;
}
