#pragma once
#include <vector>
#include <random>
#include <cassert>
#include <cstdio>

namespace tessera { namespace ebt {
// Simple row-major matrix W (DxD) and vector y (D), energy E=0.5*y^T W y (symmetric W)
struct QuadEnergy {
  int D;
  std::vector<float> W;
  explicit QuadEnergy(int d): D(d), W(d*d, 0.f) {}
  float step(std::vector<float>& y, float eta) const {
    std::vector<float> Wy(D,0.f);
    for (int i=0;i<D;i++) for (int j=0;j<D;j++) Wy[i] += W[i*D+j]*y[j];
    // grad = Wy (assuming W symmetric)
    for (int i=0;i<D;i++) y[i] -= eta * Wy[i];
    // energy
    float E=0.f;
    for (int i=0;i<D;i++) for (int j=0;j<D;j++) E += 0.5f * y[i]*W[i*D+j]*y[j];
    return E;
  }
};

inline void check_monotone_descent(int D, int T, float eta) {
  QuadEnergy q(D);
  // fill W as diag(1..D) to keep positive-definite
  for (int i=0;i<D;i++) q.W[i*D+i] = 1.0f + 0.1f*i;
  std::vector<float> y(D, 0.0f);
  for (int i=0;i<D;i++) y[i] = 0.5f; // init
  float prevE = 1e30f;
  for (int t=0;t<T;t++) {
    float E = q.step(y, eta);
    std::printf("// t=%d E=%f\n", t, E);
    assert(E <= prevE + 1e-5f); // non-increasing
    prevE = E;
  }
}
}} // ns
