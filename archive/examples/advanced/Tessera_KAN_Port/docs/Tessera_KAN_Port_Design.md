<!-- MERGE-START: Tessera_KAN_Port_Design.md (part 1 of 1) -->
# Tessera KAN Port — Design & Mapping (Starter)

**Scope.** This starter ports the core ideas from Efficient-KAN (B‑spline basis + linear mixing) into Tessera via:
1) a **Python layer** (`tessera_kan.KANLinear`) that builds Tessera graphs and falls back to NumPy for CPU tests,
2) a **Target‑IR dialect stub** (`kan_ops.td`) with two ops: `kan.bspline_eval` and `kan.linear_mix`,
3) a **lowering pass** (`LowerKANToTessera.cpp`) that rewrites to standard Tessera/Linalg ops (eltwise + matmul),
4) a **kernel stub** for a tiled bspline evaluation,
5) **FileCheck tests** and a tiny **MNIST‑like** example.

## Background recap (KAN)
KAN layers replace a static activation with a learnable univariate function per edge, parameterized as a linear
combination of **B‑spline** basis functions over a fixed grid. Efficient‑KAN computes the common basis responses
once, then combines them linearly into the output weight matrix, turning the core compute into GEMMs.

## Tessera mapping
- **Op 1: `kan.bspline_eval(x)`** — Given `x: tensor<[B, I]>`, degree `k`, `grid_min/max`, `grid_size` → returns
  `phi: tensor<[B, I, M]>` where `M = grid_size + k - 1` (uniform knots). This uses a fast iterative Cox‑de Boor.
- **Op 2: `kan.linear_mix(phi, W_base, W_spline)`** — Produces output `y: tensor<[B, O]>` by contracting the shared
  basis responses with per‑edge mixing weights. In practice we pre‑mix `(W_eff = W_base + (W_spline @ S))` so the
  end becomes a batched GEMM: `y = (phi_flat @ W_phi) @ W_out` or a single fused contraction depending on shapes.

### Lowering strategy
1. `kan.bspline_eval` → tiled elementwise kernel or vectorized loop nest; emits bufferized ops on GPU with shared
   memory tiling; on CPU emits vector dialect.
2. `kan.linear_mix` → reshape/contract to **linalg.matmul** or **tessera.matmul** + optional epilogue.
3. Optional L1/L2 penalties attach as attributes/side‑effects for the training loop (not compiled into kernels).

### Autodiff
- Forward: as above.
- Backward: gradients for B‑splines are local and can be expressed via the same recurrence with degree‑1. We either
  (a) rely on Tessera autodiff if available, or (b) register **adjoint patterns** for both ops (see pass TODOs).

## Shapes & parameters
- Inputs: `x ∈ ℝ^{B×I}`
- Outputs: `y ∈ ℝ^{B×O}`
- Grid: uniform, degree `k∈{1,2,3}`, knots count `M = grid_size + k - 1`.
- Trainables: `W_base ∈ ℝ^{I×O}`, `W_spline ∈ ℝ^{I×M×O}`, optional `spline_scale ∈ ℝ^{I×M}`.

## Files in this starter
- `python/tessera_kan/layers.py` — KANLinear with Tessera build + NumPy fallback.
- `mlir/kan_ops.td` — Minimal Target‑IR dialect.
- `mlir/LowerKANToTessera.cpp` — Pass skeleton + pattern placeholders.
- `src/runtime/kernels/kan_bspline_ref.cpp` — CPU reference (vectorized loops).
- `examples/mnist_kan_tessera.py` — Tiny toy example with random data (drop‑in for real MNIST later).
- `tests/kan_lowering.mlir` — IR → IR FileCheck sample.

## Build (sketch)
```bash
# CMake (add to your tessera superbuild)
cmake -S . -B build -DTESSERA_BUILD_KAN=ON
cmake --build build -j
```

## Upstream credit
Efficient‑KAN (MIT) inspired the basis precompute + linear mixing design.
<!-- MERGE-END: Tessera_KAN_Port_Design.md -->