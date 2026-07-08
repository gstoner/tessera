//===- Stencil.cpp (CPU target hook) --------------------------*- C++ -*-===//
//
// Reference CPU implementations of the TPP stencil Target-IR primitives — the
// concrete kernels behind the symbols `-lower-tpp-to-target-ir` emits
// (`ts_stencil_grad_cpu`, ...).  These are what the D1 arbiter's CPU stencil
// candidate runs and F4-gates against a numpy central-difference reference, so
// the Target-IR symbol the pass names has a real, verified implementation.
//
// Convention: unit grid spacing, periodic boundary (the wrap the local
// `tpp.halo.exchange` denotes).  Central difference of accuracy `order`:
//   order 2:  d/dx f[i] = (f[i+1] - f[i-1]) / 2
//   order 4:  d/dx f[i] = (-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / 12
// which matches, elementwise, the roll-based numpy reference in
// python/tessera/compiler/emit/tpp_candidates.py.
//
//===----------------------------------------------------------------------===//

extern "C" void ts_stencil_grad_cpu(const float *in, float *out, int nx,
                                    int ny, int axis, int order) {
  auto idx = [ny](int i, int j) { return i * ny + j; };
  int n = (axis == 0) ? nx : ny; // extent along the differentiated axis
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      // Coordinate along the differentiated axis + a helper to fetch a
      // neighbour `d` cells away (periodic wrap).
      int c = (axis == 0) ? i : j;
      auto at = [&](int off) -> float {
        int k = ((c + off) % n + n) % n;
        return (axis == 0) ? in[idx(k, j)] : in[idx(i, k)];
      };
      float g;
      if (order >= 4)
        g = (-at(2) + 8.0f * at(1) - 8.0f * at(-1) + at(-2)) / 12.0f;
      else
        g = (at(1) - at(-1)) * 0.5f;
      out[idx(i, j)] = g;
    }
  }
}
