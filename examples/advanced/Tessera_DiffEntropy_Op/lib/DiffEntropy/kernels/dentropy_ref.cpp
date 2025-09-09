//===- dentropy_ref.cpp --------------------------------------------------===//
// Simple, numerically stable CPU reference for testing.
// NOTE: This is not vectorized; it is intended for correctness.
#include <vector>
#include <cmath>
#include <cassert>

namespace tessera { namespace diffentropy {

// Soft indicator for halfspace: sigma((w·x + b) / alpha)
static inline double soft_ind_half(double dot, double alpha) {
  const double z = dot / alpha;
  return 1.0 / (1.0 + std::exp(-z));
}

// Entropy: -sum p log(p + eps)
static inline double entropy(const std::vector<double>& p) {
  double H = 0.0;
  const double eps = 1e-12;
  for (double v : p) if (v > 0.0) H -= v * std::log(v + eps);
  return H;
}

// Range-family aware soft partition (very simplified):
// For 'balls', params = anchors KxD; p_c ∝ exp(-||x - a_c||^2 / alpha)
// For 'halfspaces', params = (w_c[0..D-1], b_c); cells via product of soft_ind_half
double range_entropy_soft_ref(const std::vector<double>& X, // N*D
                              const std::vector<double>& P, // K*D or K*(D+1)
                              int N, int D, int K,
                              double alpha,
                              const char* family) {
  std::vector<double> masses(K, 0.0);
  // Compute soft assignments and accumulate masses
  for (int i = 0; i < N; ++i) {
    std::vector<double> s(K);
    double sum = 0.0;
    for (int c = 0; c < K; ++c) {
      double score = 0.0;
      if (std::string(family)=="balls") {
        double dist2 = 0.0;
        for (int d = 0; d < D; ++d) {
          double dx = X[i*D+d] - P[c*D+d];
          dist2 += dx*dx;
        }
        score = -dist2 / alpha;
      } else { // halfspaces (use logistic of signed distance once as a soft gate)
        double dot = 0.0;
        for (int d = 0; d < D; ++d) dot += P[c*(D+1)+d] * X[i*D+d];
        dot += P[c*(D+1)+D]; // bias
        score = std::log(soft_ind_half(dot, alpha) + 1e-12);
      }
      s[c] = std::exp(score);
      sum += s[c];
    }
    for (int c = 0; c < K; ++c) masses[c] += s[c] / (sum + 1e-12);
  }
  // Normalize masses
  double total = 0.0; for (auto v : masses) total += v;
  if (total <= 0.0) return 0.0;
  for (auto &v : masses) v /= total;
  return entropy(masses);
}

}} // namespace
