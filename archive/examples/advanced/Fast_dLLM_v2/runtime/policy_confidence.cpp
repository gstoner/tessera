#include "policy_confidence.h"
#include <algorithm>
#include <cmath>

using namespace tessera::runtime;

// Very small illustrative placeholder; real impl would run on device or with reductions.
PrefixDecision validate_and_merge(const float* stats[], int K, const ConfidenceParams& p) {
  PrefixDecision d{};
  d.lcp_len = 0;
  // Find per-position agreement by comparing argmax across branches; stop at first disagreement above tau.
  // (Placeholder: assume stats[k][i] encodes confidence margin; require >= tau across branches)
  for (int i = 0; i < 4096; ++i) {
    bool ok = true;
    for (int k = 0; k < K; ++k) ok &= (stats[k][i] >= p.tau);
    if (ok) d.lcp_len = i + 1; else break;
  }
  for (int k = 0; k < K; ++k) d.keep[k] = true;
  return d;
}
