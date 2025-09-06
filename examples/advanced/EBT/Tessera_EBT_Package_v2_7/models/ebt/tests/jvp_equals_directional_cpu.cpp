#include <cmath>
#include <cstdio>
#include <vector>
#include <cassert>
#include <random>

// Energy: E(y) = 0.5 * y^T W y  (W SPD diagonal for simplicity)
// grad_y = W y
// JVP(y; v) = grad_y(y) · v = (W y) · v
static float energy(const std::vector<float>& y, const std::vector<float>& Wdiag) {
  float E=0.f;
  for (size_t i=0;i<y.size();++i) E += 0.5f * Wdiag[i] * y[i]*y[i];
  return E;
}
static void grad(const std::vector<float>& y, const std::vector<float>& Wdiag, std::vector<float>& g) {
  size_t D=y.size(); g.assign(D,0.f);
  for (size_t i=0;i<D;++i) g[i] = Wdiag[i]*y[i];
}
static float jvp_scalar(const std::vector<float>& y, const std::vector<float>& v, const std::vector<float>& Wdiag) {
  // directional derivative grad·v
  std::vector<float> g; grad(y,Wdiag,g);
  float s=0.f; for (size_t i=0;i<y.size();++i) s += g[i]*v[i];
  return s;
}

int main(){
  const int D=32; const float eps=1e-4f; const float tol=5e-3f;
  std::vector<float> y(D,0.0f), v(D,0.0f), Wdiag(D,0.0f);
  for (int i=0;i<D;++i){ y[i]=0.3f+0.01f*i; v[i]=((i%2)?1.f:-1.f)*0.5f; Wdiag[i]=1.0f+0.1f*i; }
  float E0 = energy(y,Wdiag);
  // E(y + eps v)
  std::vector<float> y2=y; for (int i=0;i<D;++i) y2[i]+=eps*v[i];
  float E1 = energy(y2,Wdiag);
  float fd = (E1 - E0)/eps; // finite-diff directional derivative
  float j = jvp_scalar(y,v,Wdiag);
  std::printf("// fd=%.6f jvp=%.6f\n", fd, j);
  float rel = std::fabs(fd-j)/std::max(1.0f, std::fabs(fd));
  assert(rel < tol);
  std::puts("PASS JVP equals directional derivative (within tol).");
  return 0;
}
