//===- Stencil.cpp (CPU target hook) --------------------------*- C++ -*-===//
//
// Reference CPU implementations of the TPP stencil Target-IR primitives — the
// concrete kernels behind the symbols `-lower-tpp-to-target-ir` emits
// (`ts_stencil_grad_cpu`, ...).  These are what the D1 arbiter's CPU stencil
// candidate runs and F4-gates against a numpy central-difference reference, so
// the Target-IR symbol the pass names has a real, verified implementation.
//
// Convention: explicit positive grid spacing, periodic boundary (the wrap the
// local `tpp.halo.exchange` denotes).  Central difference of accuracy `order`:
//   order 2:  d/dx f[i] = (f[i+1] - f[i-1]) / (2 h)
//   order 4:  d/dx f[i] = (-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / (12 h)
// which matches, elementwise, the roll-based numpy reference in
// python/tessera/compiler/emit/tpp_candidates.py.
//
//===----------------------------------------------------------------------===//

// Status codes for the C ABI. Mirrored by
// python/tessera/compiler/emit/tpp_candidates.py, which raises on non-zero.
#define TS_STENCIL_BAD_ARGS 1
#define TS_STENCIL_UNSUPPORTED_ORDER 2

// Returns 0 on success, TS_STENCIL_BAD_ARGS for an unusable argument, and
// TS_STENCIL_UNSUPPORTED_ORDER for an accuracy order this kernel does not
// implement. `order` selects the operator's truncation error, so silently
// substituting a different tap set returns a number the caller will read as
// the order it asked for (Decision #21a).
extern "C" int ts_stencil_grad_cpu(const float *in, float *out, int nx,
                                   int ny, int axis, int order,
                                   float spacing) {
  if (!in || !out || nx <= 0 || ny <= 0 || axis < 0 || axis > 1 ||
      spacing <= 0.0f)
    return TS_STENCIL_BAD_ARGS;
  if (order != 2 && order != 4 && order != 6 && order != 8)
    return TS_STENCIL_UNSUPPORTED_ORDER;
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
      // Central first-derivative tap sets. Coefficients at +1/+2/+3/+4 are
      // 1/2 | 2/3, -1/12 | 3/4, -3/20, 1/60 | 4/5, -1/5, 4/105, -1/280.
      float g;
      if (order == 8)
        g = (672.0f * (at(1) - at(-1)) - 168.0f * (at(2) - at(-2)) +
             32.0f * (at(3) - at(-3)) - 3.0f * (at(4) - at(-4))) /
            (840.0f * spacing);
      else if (order == 6)
        g = (45.0f * (at(1) - at(-1)) - 9.0f * (at(2) - at(-2)) +
             (at(3) - at(-3))) /
            (60.0f * spacing);
      else if (order == 4)
        g = (-at(2) + 8.0f * at(1) - 8.0f * at(-1) + at(-2)) /
            (12.0f * spacing);
      else
        g = (at(1) - at(-1)) * (0.5f / spacing);
      out[idx(i, j)] = g;
    }
  }
  return 0;
}
