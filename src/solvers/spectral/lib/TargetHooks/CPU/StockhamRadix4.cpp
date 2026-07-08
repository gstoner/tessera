//===- StockhamRadix4.cpp (CPU target hook) ------------------*- C++ -*-===//
//
// Complete mixed-radix Stockham *autosort* FFT for the CPU backend.
//
// This is the reference fast-path the Spectral solver's correctness
// sentinel certifies against a naive O(N^2) DFT (see
// benchmarks/correctness_microbench.cpp).  Unlike the previous v0
// placeholder (a single twiddle-less radix-4 butterfly), this file
// computes a *full* N-point transform:
//
//   * radix-4 stages drain all factors of 4, then a single radix-2
//     stage handles the residual factor of 2 (so every power-of-two N
//     is covered: 64=4^3, 128=4^3*2, 256=4^4, 512=4^4*2, ...).
//   * autosort (Stockham) layout => no bit-reversal permutation; each
//     stage ping-pongs between two buffers with unit-stride access,
//     which is exactly the structure the GPU hooks (AMD/NVIDIA) mirror.
//
// Sign convention matches numpy.fft: forward uses W_N = exp(-2pi i/N);
// inverse uses the conjugate and scales by 1/N.
//
// Symbols (C ABI) — the names LowerSpectralToTargetIRPass emits for the
// "cpu" backend:
//   ts_stockham_r4_cpu(in, out, N, L, sign)   one radix-4 stage
//   ts_stockham_r2_cpu(in, out, N, L, sign)   one radix-2 stage
//   ts_fft_stockham_cpu(in, out, N, sign)     full transform (driver)
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstring>
#include <vector>

namespace {
struct cf {
  float x, y;
};
inline cf mul(cf a, cf b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
inline cf add(cf a, cf b) { return {a.x + b.x, a.y + b.y}; }
inline cf sub(cf a, cf b) { return {a.x - b.x, a.y - b.y}; }
// rotate by -i (forward, sign<0) or +i (inverse, sign>0)
inline cf rot(cf a, int sign) {
  return sign < 0 ? cf{a.y, -a.x} : cf{-a.y, a.x};
}
constexpr double kPi = 3.14159265358979323846;
} // namespace

// One radix-4 Stockham stage.  L = size of sub-DFTs already completed.
extern "C" void ts_stockham_r4_cpu(const void *in_, void *out_, int N, int L,
                                   int sign) {
  const cf *a = static_cast<const cf *>(in_);
  cf *b = static_cast<cf *>(out_);
  int m = N / (4 * L);
  for (int j = 0; j < L; ++j) {
    double ang = sign * 2.0 * kPi * j / (4.0 * L);
    cf w1{(float)std::cos(ang), (float)std::sin(ang)};
    cf w2 = mul(w1, w1);
    cf w3 = mul(w2, w1);
    for (int k = 0; k < m; ++k) {
      int i = k * L + j;
      cf c0 = a[i + 0 * L * m];
      cf c1 = mul(a[i + 1 * L * m], w1);
      cf c2 = mul(a[i + 2 * L * m], w2);
      cf c3 = mul(a[i + 3 * L * m], w3);
      cf t0 = add(c0, c2), t1 = sub(c0, c2);
      cf t2 = add(c1, c3), t3 = rot(sub(c1, c3), sign);
      int o = k * (4 * L) + j;
      b[o + 0 * L] = add(t0, t2);
      b[o + 1 * L] = add(t1, t3);
      b[o + 2 * L] = sub(t0, t2);
      b[o + 3 * L] = sub(t1, t3);
    }
  }
}

// One radix-2 Stockham stage (residual factor of two).
extern "C" void ts_stockham_r2_cpu(const void *in_, void *out_, int N, int L,
                                   int sign) {
  const cf *a = static_cast<const cf *>(in_);
  cf *b = static_cast<cf *>(out_);
  int m = N / (2 * L);
  for (int j = 0; j < L; ++j) {
    double ang = sign * 2.0 * kPi * j / (2.0 * L);
    cf w1{(float)std::cos(ang), (float)std::sin(ang)};
    for (int k = 0; k < m; ++k) {
      int i = k * L + j;
      cf c0 = a[i + 0 * L * m];
      cf c1 = mul(a[i + 1 * L * m], w1);
      int o = k * (2 * L) + j;
      b[o + 0 * L] = add(c0, c1);
      b[o + 1 * L] = sub(c0, c1);
    }
  }
}

// Full transform host driver: loops radix-4 stages then a radix-2 tail,
// ping-ponging between `out` and a scratch buffer.  sign=-1 forward,
// sign=+1 inverse (scaled by 1/N).  Result is written to `out`.
extern "C" void ts_fft_stockham_cpu(const void *in_, void *out_, int N,
                                    int sign) {
  std::vector<cf> scratch(N);
  const cf *src = static_cast<const cf *>(in_);
  cf *dst = static_cast<cf *>(out_);
  cf *a = scratch.data();
  cf *b = dst;
  std::memcpy(a, src, N * sizeof(cf));

  int L = 1, n = N;
  auto step = [&](int radix) {
    if (radix == 4)
      ts_stockham_r4_cpu(a, b, N, L, sign);
    else
      ts_stockham_r2_cpu(a, b, N, L, sign);
    L *= radix;
    std::swap(a, b);
  };
  while (n % 4 == 0) { step(4); n /= 4; }
  while (n % 2 == 0) { step(2); n /= 2; }

  if (a != dst)
    std::memcpy(dst, a, N * sizeof(cf));
  if (sign > 0)
    for (int i = 0; i < N; ++i) { dst[i].x /= N; dst[i].y /= N; }
}
