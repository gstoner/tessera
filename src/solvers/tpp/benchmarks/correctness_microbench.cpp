//===- correctness_microbench.cpp - TPP space-time correctness sentinel --===//
//
// Numerical correctness sentinel for the TPP space-time solver, analogous to
// the Spectral solver's FFT microbench.  The TPP passes are IR transforms and
// do not execute, so this binary implements the *semantics* those ops denote
// — a central-difference gradient (`tpp.grad`), a periodic boundary wrap
// (`tpp.bc.enforce<periodic>` + the local `tpp.halo.exchange`), and explicit
// time stepping (`tpp.time.step`) — and checks them against exact references:
//
//   1. grad-convergence : central-difference d/dx of sin(kx) vs the analytic
//                         k*cos(kx).  Verifies the stencil is 2nd-order
//                         accurate (error drops ~4x per grid doubling).
//   2. periodic-bc      : the periodic ghost wrap makes the boundary-cell
//                         gradient as accurate as the interior (no edge blow-up
//                         from a one-sided/again-zero stencil).
//   3. swe-conservation : the linearised shallow-water system conserves total
//                         mass and energy over a long run (RK4 in time).
//   4. swe-traveling-wave: a Riemann-invariant right-moving wave advects at the
//                         gravity-wave speed c=sqrt(gH); numeric h(x,T) matches
//                         the analytic A*cos(k(x-cT)).
//
// Output is machine-parseable key=value; exit is non-zero on any regression.
//
// Build: cmake --build <build> --target tpp-correctness
// Run:   ./tpp-correctness
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// Central-difference d/dx on a periodic 1D grid (radius-1 stencil == tpp.grad
// with an inferred halo of 1 + a periodic tpp.halo.exchange).
std::vector<double> ddx_periodic(const std::vector<double> &f, double dx) {
  int n = (int)f.size();
  std::vector<double> g(n);
  for (int i = 0; i < n; ++i) {
    int ip = (i + 1) % n;       // periodic wrap (ghost cell)
    int im = (i - 1 + n) % n;
    g[i] = (f[ip] - f[im]) / (2.0 * dx);
  }
  return g;
}

double max_abs_err(const std::vector<double> &a, const std::vector<double> &b) {
  double e = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    e = std::max(e, std::fabs(a[i] - b[i]));
  return e;
}

// ---- Test 1 + 2: gradient accuracy + periodic-boundary correctness --------
struct GradReport {
  double err;         // interior max error
  double boundary_err;// error at the two wrap cells (i=0, i=n-1)
};

GradReport grad_test(int n) {
  double L = 2.0 * kPi, dx = L / n;
  std::vector<double> f(n), exact(n);
  for (int i = 0; i < n; ++i) {
    double x = i * dx;
    f[i] = std::sin(x);
    exact[i] = std::cos(x); // d/dx sin(x)
  }
  auto g = ddx_periodic(f, dx);
  double interior = 0.0, boundary = 0.0;
  for (int i = 0; i < n; ++i) {
    double e = std::fabs(g[i] - exact[i]);
    if (i == 0 || i == n - 1)
      boundary = std::max(boundary, e);
    else
      interior = std::max(interior, e);
  }
  return {interior, boundary};
}

// ---- Test 3 + 4: linearised shallow-water (wave) system -------------------
// h_t = -H u_x ,  u_t = -g h_x   (periodic).  Wave speed c = sqrt(gH).
struct SWEParams {
  int n = 512;
  double L = 2.0 * kPi;
  double g = 1.0, H = 1.0;
  int k = 1;          // wavenumber
  double A = 0.1;     // amplitude
};

struct SWEState {
  std::vector<double> h, u;
};

void swe_rhs(const SWEState &s, double g, double H, double dx, SWEState &d) {
  auto ux = ddx_periodic(s.u, dx);
  auto hx = ddx_periodic(s.h, dx);
  int n = (int)s.h.size();
  d.h.resize(n);
  d.u.resize(n);
  for (int i = 0; i < n; ++i) {
    d.h[i] = -H * ux[i];
    d.u[i] = -g * hx[i];
  }
}

void swe_rk4_step(SWEState &s, double g, double H, double dx, double dt) {
  int n = (int)s.h.size();
  SWEState k1, k2, k3, k4, tmp;
  auto axpy = [&](const SWEState &base, const SWEState &k, double a,
                  SWEState &out) {
    out.h.resize(n);
    out.u.resize(n);
    for (int i = 0; i < n; ++i) {
      out.h[i] = base.h[i] + a * k.h[i];
      out.u[i] = base.u[i] + a * k.u[i];
    }
  };
  swe_rhs(s, g, H, dx, k1);
  axpy(s, k1, dt / 2, tmp); swe_rhs(tmp, g, H, dx, k2);
  axpy(s, k2, dt / 2, tmp); swe_rhs(tmp, g, H, dx, k3);
  axpy(s, k3, dt, tmp);     swe_rhs(tmp, g, H, dx, k4);
  for (int i = 0; i < n; ++i) {
    s.h[i] += dt / 6 * (k1.h[i] + 2 * k2.h[i] + 2 * k3.h[i] + k4.h[i]);
    s.u[i] += dt / 6 * (k1.u[i] + 2 * k2.u[i] + 2 * k3.u[i] + k4.u[i]);
  }
}

struct SWEReport {
  double mass_drift;   // |sum h(T) - sum h(0)|
  double energy_drift; // relative energy drift
  double wave_err;     // max |h_numeric(T) - A cos(k(x - cT))|
};

SWEReport swe_test(const SWEParams &p, double periods) {
  int n = p.n;
  double dx = p.L / n;
  double c = std::sqrt(p.g * p.H);
  double kx = 2.0 * kPi * p.k / p.L;
  // Right-moving Riemann-invariant init: u = sqrt(g/H) h => R- = 0.
  SWEState s;
  s.h.resize(n);
  s.u.resize(n);
  double ratio = std::sqrt(p.g / p.H);
  for (int i = 0; i < n; ++i) {
    double x = i * dx;
    s.h[i] = p.A * std::cos(kx * x);
    s.u[i] = ratio * s.h[i];
  }
  // Conserved diagnostics at t=0.
  auto mass = [&](const SWEState &st) {
    double m = 0;
    for (double v : st.h) m += v;
    return m;
  };
  auto energy = [&](const SWEState &st) {
    double e = 0;
    for (int i = 0; i < n; ++i)
      e += 0.5 * (p.g * st.h[i] * st.h[i] + p.H * st.u[i] * st.u[i]);
    return e;
  };
  double m0 = mass(s), e0 = energy(s);

  double T = periods * (p.L / c); // integrate `periods` wave crossings
  double dt = 0.25 * dx / c;      // CFL-safe
  int steps = (int)std::ceil(T / dt);
  dt = T / steps;                 // land exactly on T
  for (int t = 0; t < steps; ++t)
    swe_rk4_step(s, p.g, p.H, dx, dt);

  // Analytic right-moving wave at time T.
  std::vector<double> exact(n);
  for (int i = 0; i < n; ++i) {
    double x = i * dx;
    exact[i] = p.A * std::cos(kx * (x - c * T));
  }
  SWEReport r;
  r.mass_drift = std::fabs(mass(s) - m0);
  r.energy_drift = std::fabs(energy(s) - e0) / e0;
  r.wave_err = max_abs_err(s.h, exact);
  return r;
}

} // namespace

int main() {
  printf("tool=tpp-correctness\n");
  printf("system=linearized_shallow_water\n");
  printf("stencil=central_difference_r1_periodic\n");
  printf("integrator=rk4\n");

  bool all_pass = true;

  // --- Test 1 + 2: gradient convergence + periodic boundary ---
  printf("test=grad_convergence\n");
  double prev_err = -1.0;
  bool grad_ok = true, bc_ok = true;
  for (int n : {32, 64, 128, 256}) {
    GradReport g = grad_test(n);
    double order =
        (prev_err > 0) ? std::log2(prev_err / g.err) : 0.0;
    // Boundary must be as accurate as interior (periodic wrap correct).
    double bc_ratio = g.boundary_err / (g.err + 1e-30);
    bool bc_here = bc_ratio < 2.0;
    bc_ok &= bc_here;
    printf("size=%d grad_max_err=%.3e boundary_err=%.3e order=%.2f "
           "bc_ratio=%.2f\n",
           n, g.err, g.boundary_err, order, bc_ratio);
    if (prev_err > 0 && order < 1.8) // expect ~2nd order
      grad_ok = false;
    prev_err = g.err;
  }
  printf("grad_convergence_pass=%d\n", grad_ok ? 1 : 0);
  printf("periodic_bc_pass=%d\n", bc_ok ? 1 : 0);
  all_pass &= grad_ok && bc_ok;

  // --- Test 3 + 4: shallow-water conservation + traveling wave ---
  SWEParams p;
  SWEReport s = swe_test(p, /*periods=*/2.0);
  bool mass_ok = s.mass_drift < 1e-9;
  bool energy_ok = s.energy_drift < 1e-4;
  bool wave_ok = s.wave_err < 1e-3;
  printf("test=shallow_water\n");
  printf("n=%d periods=2 mass_drift=%.3e energy_drift=%.3e wave_err=%.3e\n",
         p.n, s.mass_drift, s.energy_drift, s.wave_err);
  printf("mass_conservation_pass=%d\n", mass_ok ? 1 : 0);
  printf("energy_conservation_pass=%d\n", energy_ok ? 1 : 0);
  printf("traveling_wave_pass=%d\n", wave_ok ? 1 : 0);
  all_pass &= mass_ok && energy_ok && wave_ok;

  printf("verdict=%s\n", all_pass ? "pass" : "fail");
  return all_pass ? 0 : 1;
}
