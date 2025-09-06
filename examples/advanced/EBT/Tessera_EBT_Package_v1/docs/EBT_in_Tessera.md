<!-- MERGE_START -->
# Energy‑Based Transformer in the Tessera Programming Model

**Document intent.** Provide a practical mapping of Energy‑Based Transformers (EBTs) into Tessera’s IR stack and runtime. The goal is to make EBTs a first‑class pattern with clean passes, tests, and a portable execution loop.

## 1. Concept recap (informative)
EBTs learn an **energy function** `E(x, y; θ)` that scores candidate outputs `y` given input `x`. **Prediction** is performed by *minimizing* energy over `y`. EBTs support *System‑2 thinking* by running extra inner‑loop optimization (or sampling) steps and **self‑verification** by generating multiple candidates and taking the minimum‑energy one.

> See official site and paper for background and empirical results.

### Core algorithmic skeleton
1. Encode `x` (tokens, patches) with a Transformer‑like stack to produce context features `h`.
2. Define learnable energy `E(h, y)` (attention‑conditioned decoder scoring the full output sequence or token).
3. **Inner loop**: initialize `y₀` (e.g., from a base model or noise), then iterate:  
   `y_{t+1} = y_t - η ∂E/∂y_t + ξ_t`  (gradient step + optional noise; or use MCMC/Langevin/coordinate updates)
4. **Self‑verify (optional)**: draw `K` candidate trajectories, evaluate energies, return `argmin_k E(h, y_K^k)`.

### Losses (train time)
- Contrastive / NCE on positives vs. negatives from the inner loop.
- Score matching or denoising‑style objectives for continuous `y`.
- Cross‑entropy auxiliaries on teacher proposals (optional).

## 2. Mapping to Tessera IR

Tessera levels:
- **Graph IR**: high‑level ops; training/inference control‑flow.
- **Schedule IR**: loop structure over candidates/steps; parallel policies.
- **Tile IR**: kernels: attention blocks, energy heads, gradient/JVPs, logits/energies.
- **Target IR**: backend lowering (NVIDIA/ROCm/CPU/TPU/etc.).

### 2.1 Graph IR: new ops & patterns
We model EBT as a *pattern*, using a small dialect overlay `tessera.ebt`:

- `ebt.energy(%h, %y) -> tensor<...>` : returns scalar or per‑token energy.
- `ebt.inner_step(%y, %grad, %eta) -> %y'` : pluggable update (SGD/Langevin/coordinate).
- `ebt.self_verify(%candE: tensor<Kxf32>, %candY: tensor<Kx...>) -> %y_best`.
- `ebt.decode_init(%x) -> %y0` : initializer from a base policy or noise.
- `ebt.grad_y(%h, %y) -> %grad` : gradient of energy w.r.t. `y` (autodiff or custom).

We keep the rest as existing Tessera ops (attention, MLP, layer‑norm).

### 2.2 Schedule IR
The control structure below is canonical for inference; `K` candidates, `T` steps:
```mlir
// Graph→Schedule lower: explicit loops (pseudo‑MLIR)
%y0s = ebt.decode_init %x : (tensor<...>) -> tensor<Kx...>
scf.for %k = 0 to K {
  %y = tensor.extract %y0s[%k]
  scf.for %t = 0 to T {
    %g = ebt.grad_y %h, %y
    %y = ebt.inner_step %y, %g, %eta
  }
  %Ek = ebt.energy %h, %y     // final energy
  // collect (%Ek, %y) per k
}
%y_best = ebt.self_verify %E_all, %Y_all
```

Parallelization policies:
- Map the **K** candidates across devices/streams.
- Map **T** steps with software pipelining (double‑buffer states).
- Fuse `grad_y` with attention/MLP tiles to reuse activations.

### 2.3 Tile IR
Key kernels:
- **Energy head**: attention‑conditioned scoring (e.g., bilinear head or small MLP).
- **Grad/JVP**: reuse backward paths of attention/MLP tiles; expose fast‑path JVP for inner‑loop steps.
- **Sampler**: vectorized Langevin/coordinate updates (bf16/fp8 storage; fp32 accum).

We provide reference Tile IR snippets in `models/ebt/ir/ebt_ir_samples.mlir` (see below).

## 3. Runtime design

A tiny runner abstracts the inner loop:
```c++
// ebt_runner.h
struct EBTStepConfig { int T; int K; float eta; float noise; bool self_verify; };
void ebt_infer(const Tensor& x, const EBTStepConfig& cfg, Tensor* y_out);
```
Backends may implement specialized kernels; on GPU we batch candidates across streams and record ranges to Perfetto/NVTX.

## 4. Pass pipelines (mlir-opt style)

- `-tessera-ebt-canonicalize`:
  - Inline `ebt.decode_init` when it’s a simple noise/base‑model call.
  - Normalize `ebt.self_verify` to min‑reduce over `K` energies.
  - Hoist loop‑invariants (context `h`) and rematerialize light ops.

- `-tessera-ebt-lower`:
  - Graph→Schedule: materialize `scf.for` loops for `K` and `T`.
  - Schedule→Tile: tile attention/MLP; generate `grad_y` kernels.
  - Tile→Target: pick backend (NV/PTX, ROCm/MFMA, CPU/AVX2) via existing Tessera lowers.

## 5. Tests (FileCheck)

- Canonicalization: shape‑safe hoists, normalized verify.
- Lowering: loop emission; candidate/step mapping attrs.
- Backend smoke: Tile→Target patterns appear with expected ops.

## 6. Open choices

- Exact energy head form (bilinear vs. MLP); both sketched.
- Update rule: pure gradient vs. Langevin (noise); pluggable policy attr.
- Token‑wise vs. sequence‑wise energy; we provide both.

---

## Appendix A — IR snippets (abbrev.)

```mlir
// Encode → h
%h = tessera.encode %x : (tensor<BxLxD>) -> tensor<BxLxD>

// Init K candidates
%y0 = ebt.decode_init %x : (tensor<BxLxD>) -> tensor<BxKxLxD>

// Inner loop
scf.for %k = 0 to %K {
  %y = tensor.extract %y0[%k]
  scf.for %t = 0 to %T {
    %g = ebt.grad_y %h, %y : (tensor<BxLxD>, tensor<BxLxD>) -> tensor<BxLxD>
    %y = ebt.inner_step %y, %g, %eta
  }
  %Ek = ebt.energy %h, %y : (tensor<BxLxD>, tensor<BxLxD>) -> tensor<Bxf32>
  // collect
}
%y_best = ebt.self_verify %E_all, %Y_all
```

<!-- MERGE_END -->
