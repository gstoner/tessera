//===- FFTPlan.h - target-independent FFT planning ------------*- C++ -*-===//
//
// The part of an FFT that is NOT target-specific: how to factor N into
// stages, whether it factors at all, and the chirp table Bluestein needs when
// it does not.
//
// WHY THIS EXISTS.  The CPU, AMD and NVIDIA hooks each carried their own copy
// of the driver loop:
//
//     while (n % 4 == 0) { stage(4); n /= 4; }
//     while (n % 2 == 0) { stage(2); n /= 2; }
//
// Three copies of one decision, compiled by three different compilers, and all
// three shared the same defect: any N with a factor outside {2,4} fell out of
// the loop with the transform half-finished, and the driver returned success.
// Measured against numpy, powers of two agreed to ~1e-7 while 3, 12, 24, 48,
// 100, 255 and 257 came back with relative error ~1.0.
//
// Fixing that in three places would have left three places to drift.  Planning
// is arithmetic on N -- it has nothing to do with threads, streams or device
// memory -- so it belongs in one header the three backends include (Decision
// #31: one implementation per boundary).  What stays per-target is what is
// genuinely per-target: how a butterfly is executed and where the buffers
// live.
//
// Header-only and host-side only.  No device qualifiers, no target headers, so
// nvcc, hipcc and the host compiler can all include it.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_SPECTRAL_TARGETHOOKS_FFTPLAN_H
#define TESSERA_SPECTRAL_TARGETHOOKS_FFTPLAN_H

#include <cmath>

namespace tessera {
namespace spectral {

/// Largest prime factor the generic radix stage handles directly.
///
/// Above this, an r-point DFT inside each butterfly costs O(r^2) and Bluestein
/// -- O(M log M) on a power-of-two M >= 2N-1 -- wins.  13 keeps the common
/// audio/scientific sizes (multiples of 3, 5, 7, 11, 13, 17) on the direct path
/// while sending genuine large primes to the algorithm meant for them.
constexpr int kMaxRadix = 17;

/// Upper bound on stages.  N < 2^31 has at most 31 factors (all 2s), so 32 is
/// sufficient for any int N and the plan can live on the stack.
constexpr int kMaxStages = 32;

enum FFTPlanKind {
  /// N factors entirely within the radix set; run `stage_count` stages.
  kFFTMixedRadix = 0,
  /// N has a factor above kMaxRadix; run Bluestein over `bluestein_m`.
  kFFTBluestein = 1,
};

struct FFTPlan {
  int n = 0;
  int kind = kFFTMixedRadix;
  int stage_count = 0;
  int stages[kMaxStages] = {};
  /// Power-of-two convolution length for Bluestein; 0 for the mixed-radix path.
  int bluestein_m = 0;
};

/// Smallest power of two >= v.
inline int fft_next_pow2(int v) {
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

inline bool fft_is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

/// Plan a length-`n` transform.
///
/// Radices are drained largest-first -- 4, then 2, then the odd primes -- so
/// the cheap hand-written butterflies do as much of the work as possible and
/// the generic O(r^2) stage handles only what is left.
///
/// A plan is always returned.  When `n` does not factor within the radix set
/// the plan says `kFFTBluestein` rather than reporting a shorter stage list,
/// which is the specific failure this replaces: a truncated stage list reads
/// like a valid plan.
inline FFTPlan fft_plan(int n) {
  FFTPlan plan;
  plan.n = n;
  if (n <= 1) {
    plan.kind = kFFTMixedRadix;
    plan.stage_count = 0;
    return plan;
  }

  int rest = n;
  int count = 0;
  auto push = [&](int radix) -> bool {
    if (count >= kMaxStages)
      return false;
    plan.stages[count++] = radix;
    return true;
  };

  bool ok = true;
  while (ok && rest % 4 == 0) {
    ok = push(4);
    rest /= 4;
  }
  while (ok && rest % 2 == 0) {
    ok = push(2);
    rest /= 2;
  }
  for (int radix = 3; ok && radix <= kMaxRadix; radix += 2)
    while (ok && rest % radix == 0) {
      ok = push(radix);
      rest /= radix;
    }

  if (ok && rest == 1) {
    plan.kind = kFFTMixedRadix;
    plan.stage_count = count;
    plan.bluestein_m = 0;
  } else {
    plan.kind = kFFTBluestein;
    plan.stage_count = 0;
    plan.bluestein_m = fft_next_pow2(2 * n - 1);
  }
  return plan;
}

/// The Bluestein chirp phase for index `j` of a length-`n` transform.
///
/// `chirp[j] = exp(-sign * pi * i * j^2 / n)`, returned as an angle so each
/// backend can use its own sincos.
///
/// Uses `(j*j) mod 2n` rather than `j*j`.  For n in the tens of thousands
/// `j^2` leaves the exactly-representable range of a double and the error
/// appears as a slow phase drift across the spectrum -- a wrong answer that
/// still looks like a spectrum, which is the hardest kind to notice.  The
/// modulus is exact in `long long` and loses nothing, since the phase is
/// periodic in 2n.
inline double fft_chirp_angle(int j, int n, int sign) {
  const long long jj = (static_cast<long long>(j) * static_cast<long long>(j)) %
                       (2LL * static_cast<long long>(n));
  return -static_cast<double>(sign) * 3.14159265358979323846 *
         static_cast<double>(jj) / static_cast<double>(n);
}

} // namespace spectral
} // namespace tessera

#endif // TESSERA_SPECTRAL_TARGETHOOKS_FFTPLAN_H
