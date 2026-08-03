//===- StockhamRadix4.cpp (CPU target hook) ------------------*- C++ -*-===//
//
// Mixed-radix Stockham *autosort* FFT + Bluestein, for the CPU backend.
//
// Planning (which radices, and whether N factors at all) lives in
// TargetHooks/Common/FFTPlan.h and is shared with the AMD and NVIDIA hooks.
// This file is the CPU *execution*: butterflies, twiddle tables, buffers.
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
//   * a generic radix-r stage covers the odd small primes (3..13) using a
//     precomputed r-point DFT matrix -- r^2 entries, built once per stage
//     rather than recomputing sin/cos inside every butterfly.
//   * Bluestein (chirp-z) covers large primes and anything else that does
//     not factor, by turning the transform into a power-of-two circular
//     convolution the radix-4/2 path already computes.
//
// Previously this file drained factors of 4 and 2 and then STOPPED, so any
// other N returned a partially-transformed buffer and reported success --
// measured against numpy, relative error ~1.0 at N = 3, 12, 24, 48, 100, 255,
// 257 while every power of two agreed to ~1e-7.
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

#include "../Common/FFTPlan.h"

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

// One generic radix-r Stockham stage, for the odd small primes.
//
// The r-point DFT matrix is built ONCE per stage (r^2 entries) instead of
// calling sin/cos inside every butterfly, which is what a plan is for: the
// previous shape recomputed the same r^2 twiddles N/r times per stage.
extern "C" void ts_stockham_rn_cpu(const void *in_, void *out_, int N, int L,
                                   int r, int sign) {
  const cf *a = static_cast<const cf *>(in_);
  cf *b = static_cast<cf *>(out_);
  const int m = N / (r * L);

  std::vector<cf> dft(static_cast<size_t>(r) * r);
  for (int p = 0; p < r; ++p)
    for (int q = 0; q < r; ++q) {
      // (p*q) % r keeps the angle small so the cosine holds its precision at
      // the larger radices.
      const double ang = sign * 2.0 * kPi * ((p * q) % r) / (double)r;
      dft[(size_t)p * r + q] = cf{(float)std::cos(ang), (float)std::sin(ang)};
    }

  std::vector<cf> c(r);
  for (int j = 0; j < L; ++j) {
    // Stage twiddles W_{rL}^{j*q}: r values, reused across all m butterflies.
    std::vector<cf> stage(r);
    for (int q = 0; q < r; ++q) {
      const double ang = sign * 2.0 * kPi * (double)j * (double)q /
                         ((double)r * (double)L);
      stage[q] = cf{(float)std::cos(ang), (float)std::sin(ang)};
    }
    for (int k = 0; k < m; ++k) {
      const int i = k * L + j;
      for (int q = 0; q < r; ++q)
        c[q] = q == 0 ? a[i] : mul(a[i + (size_t)q * L * m], stage[q]);
      const int o = k * (r * L) + j;
      for (int p = 0; p < r; ++p) {
        cf acc{0.0f, 0.0f};
        for (int q = 0; q < r; ++q)
          acc = add(acc, mul(c[q], dft[(size_t)p * r + q]));
        b[o + (size_t)p * L] = acc;
      }
    }
  }
}

// Exposed for tests: the plan for N, so a gate can assert WHICH path a size
// takes rather than inferring it from timing.  Returns the stage count, or -1
// when N needs Bluestein.
extern "C" int ts_fft_radices_cpu(int N, int *radices, int cap) {
  const tessera::spectral::FFTPlan plan = tessera::spectral::fft_plan(N);
  if (plan.kind == tessera::spectral::kFFTBluestein)
    return -1;
  if (plan.stage_count > cap)
    return -1;
  for (int i = 0; i < plan.stage_count; ++i)
    radices[i] = plan.stages[i];
  return plan.stage_count;
}

extern "C" void ts_fft_stockham_cpu(const void *in_, void *out_, int N,
                                    int sign);

namespace {
// Bluestein (chirp-z).  With nk = (n^2 + k^2 - (k-n)^2)/2 the transform
// becomes a convolution, evaluated circularly at M = next_pow2(2N-1) -- long
// enough that the wrap-around cannot alias into the first N outputs.
void bluestein(const cf *x, cf *out, int N, int sign, int M) {
  std::vector<cf> a(M, cf{0.0f, 0.0f}), b(M, cf{0.0f, 0.0f}), chirp(N);
  for (int j = 0; j < N; ++j) {
    const double ang = tessera::spectral::fft_chirp_angle(j, N, sign);
    chirp[j] = cf{(float)std::cos(ang), (float)std::sin(ang)};
  }
  for (int j = 0; j < N; ++j) {
    a[j] = mul(x[j], cf{chirp[j].x, -chirp[j].y});
    b[j] = chirp[j];
    if (j)
      b[M - j] = chirp[j]; // symmetric tail for the circular wrap
  }

  std::vector<cf> fa(M), fb(M), fc(M), conv(M);
  ts_fft_stockham_cpu(a.data(), fa.data(), M, -1);
  ts_fft_stockham_cpu(b.data(), fb.data(), M, -1);
  for (int j = 0; j < M; ++j)
    fc[j] = mul(fa[j], fb[j]);
  // Inverse via the forward path: ifft(y) = conj(fft(conj(y)))/M.  Calling the
  // driver's own inverse here would apply the 1/M scale a second time.
  for (int j = 0; j < M; ++j)
    fc[j].y = -fc[j].y;
  ts_fft_stockham_cpu(fc.data(), conv.data(), M, -1);
  const float inv = 1.0f / (float)M;
  for (int j = 0; j < M; ++j)
    conv[j] = cf{conv[j].x * inv, -conv[j].y * inv};

  for (int k = 0; k < N; ++k)
    out[k] = mul(conv[k], cf{chirp[k].x, -chirp[k].y});
}
} // namespace

// Full transform host driver.  Plans first (shared FFTPlan.h), then either
// runs the stage list or routes to Bluestein.  sign=-1 forward, sign=+1
// inverse (scaled by 1/N).  Result is written to `out`.
extern "C" void ts_fft_stockham_cpu(const void *in_, void *out_, int N,
                                    int sign) {
  const cf *src = static_cast<const cf *>(in_);
  cf *dst = static_cast<cf *>(out_);
  if (N <= 0)
    return;
  if (N == 1) {
    dst[0] = src[0];
    return;
  }

  const tessera::spectral::FFTPlan plan = tessera::spectral::fft_plan(N);
  if (plan.kind == tessera::spectral::kFFTBluestein) {
    std::vector<cf> tmp(N);
    std::memcpy(tmp.data(), src, N * sizeof(cf));
    bluestein(tmp.data(), dst, N, sign, plan.bluestein_m);
    if (sign > 0)
      for (int i = 0; i < N; ++i) { dst[i].x /= N; dst[i].y /= N; }
    return;
  }

  std::vector<cf> scratch(N);
  cf *a = scratch.data();
  cf *b = dst;
  std::memcpy(a, src, N * sizeof(cf));

  int L = 1;
  for (int s = 0; s < plan.stage_count; ++s) {
    const int radix = plan.stages[s];
    if (radix == 4)
      ts_stockham_r4_cpu(a, b, N, L, sign);
    else if (radix == 2)
      ts_stockham_r2_cpu(a, b, N, L, sign);
    else
      ts_stockham_rn_cpu(a, b, N, L, radix, sign);
    L *= radix;
    std::swap(a, b);
  }

  if (a != dst)
    std::memcpy(dst, a, N * sizeof(cf));
  if (sign > 0)
    for (int i = 0; i < N; ++i) { dst[i].x /= N; dst[i].y /= N; }
}
