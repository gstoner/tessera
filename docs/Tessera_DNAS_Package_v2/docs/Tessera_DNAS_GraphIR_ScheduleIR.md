# Tessera DNAS — GraphIR & ScheduleIR Expansion (v2)

This document extends the DNAS design with **IR-level details** for GraphIR and ScheduleIR, including
dialect sketches, lowering paths, and a small prototype that jointly optimizes **architecture** and **schedule** knobs.

---

## 1. GraphIR Dialect Sketch

### 1.1 Dialect Overview
- **Namespace:** `tessera.graph`
- **Key ops:** `arch.parameter`, `arch.gumbel_softmax`, `arch.softmax`, `arch.hard_concrete`,
  `arch.weighted_sum`, `arch.switch`, and common compute ops `op.matmul`, `op.attention`, `op.ffn`, ...

### 1.2 Canonical MixedOp Encoding
```mlir
tessera.graph.func @block(%x: tensor<?xDxf32>) -> tensor<?xDxf32> {
  %tau     = tessera.graph.constant 4.0 : f32
  %alpha   = tessera.graph.arch.parameter {num_candidates = 4} : tensor<4xf32>
  %gate    = tessera.graph.arch.gumbel_softmax(%alpha, %tau) : tensor<4xf32>

  %y0 = tessera.graph.op.flash_attention(%x)   : tensor<?xDxf32>
  %y1 = tessera.graph.op.performer_attention(%x) : tensor<?xDxf32>
  %y2 = tessera.graph.op.mqa(%x)               : tensor<?xDxf32>
  %y3 = tessera.graph.op.gmlp(%x)              : tensor<?xDxf32>

  %y  = tessera.graph.arch.weighted_sum %gate, [%y0, %y1, %y2, %y3]
         : tensor<4xf32>, tensor<?xDxf32> -> tensor<?xDxf32>
  return %y : tensor<?xDxf32>
}
```

**Notes**
- `arch.parameter` participates in reverse-mode AD; its gradients are reduced across the data-parallel mesh with deterministic rules (ordered/Kahan).
- `arch.switch` (not shown) creates a single-branch execution with STE; use when you want hard routing during training.

### 1.3 AD Semantics
- All `arch.*` ops are **differentiable** (except hard variants, which use STE).
- Gradients w.r.t. `alpha` flow through `gumbel_softmax`/`softmax` into `weighted_sum`.

---

## 2. ScheduleIR Dialect Sketch (Joint NAS + Scheduling)

### 2.1 Dialect Overview
- **Namespace:** `tessera.schedule`
- **Concepts:** choices over **tile sizes**, **pipeline stages**, **layouts**, **fusion plans**.
- **Relaxed choices:** same `arch.*` family for differentiable gates, reused here.

### 2.2 Attaching Schedule Choices
```mlir
// Define schedule knobs (can be relaxed/differentiable)
%tm = tessera.schedule.choice @tile_m {values = [64, 128]} : i64
%tn = tessera.schedule.choice @tile_n {values = [128, 256]} : i64
%tk = tessera.schedule.choice @tile_k {values = [32, 64]}   : i64
%st = tessera.schedule.choice @stages {values = [2, 3, 4]}  : i64
%ly = tessera.schedule.choice @layout {values = ["row", "col", "tiled64"]} : !tessera.layout

// Apply to a MatMul site in the GraphIR
tessera.schedule.apply @matmul0 tile(%tm, %tn, %tk) stages(%st) layout(%ly)
```

**Lowering behavior**
1. GraphIR → ScheduleIR: Annotate MatMul/Attention sites with **schedule handles**.
2. ScheduleIR → TileIR: Emit tiled loops, cp.async stages, Tensor Core fragments (WMMA/WGMMA) according to the chosen values.
3. TileIR → TargetIR: Lower to NVVM/PTX (NVIDIA), ROCm-LLVM (AMD), CPU backends.

### 2.3 Differentiable Scheduling
- Each `choice` can be **relaxed** by mapping discrete values to a probability vector `π_s` and then generating an **expected cost**.
- The generated code for evaluation uses a **single variant** at runtime; differentiation uses the cost model (see §3).

---

## 3. Cost Model & Autotuner Coupling

### 3.1 Two-tier cost model
- **Analytical** (fast): FLOPs/bytes/bandwidth + rule-based latency/energy approximation.
- **Surrogate** (smooth): `f_φ(features) → {lat, energy, mem}` trained from **on-device measurements**.

### 3.2 Autotuner Loop
1. Build feature vector from GraphIR + ScheduleIR (shape, dtype, tiles, stages, arch α).
2. Predict `{lat, energy, mem}` with surrogate for gradient flow.
3. Periodically **measure** real kernels (per shape) and **update φ**.
4. Persist **best schedule per (shape, arch choice, sm_arch)** in a cache.

**Cache key example**
```
key = {
  "op": "matmul",
  "shape": [M,K,N],
  "dtype": "bf16",
  "sm": "sm_90",
  "arch": hash(alpha_probs_discretized)
}
```

### 3.3 Determinism
- When measuring, use fixed seeds, fixed stream order, and identical reduction order.
- Store measurement metadata (clock rate, temperature, driver version) alongside results.

---

## 4. Discretization & Specialization

- Freeze both **architecture** and **schedule** candidates:
  - `choices_op = arch.argmax(model)`
  - `choices_sched = schedule.argmax(plan)`
- `tessera.lower(frozen_model, frozen_schedule)` → optimized TileIR → TargetIR
- Re-run the autotuner **only** within the narrowed discrete space to squeeze last-mile perf.

---

## 5. Example MLIR Sketch (Graph→Schedule)

```mlir
// GraphIR
%y = call @block(%x) : (tensor<?xDxf32>) -> tensor<?xDxf32>

// Block expansion (see §1.2)...

// ScheduleIR (attached to a MatMul inside block)
%tm = tessera.schedule.choice @tile_m {values=[64,128]} : i64
%tn = tessera.schedule.choice @tile_n {values=[128,256]} : i64
%tk = tessera.schedule.choice @tile_k {values=[32,64]}   : i64
%st = tessera.schedule.choice @stages {values=[2,3,4]}   : i64
tessera.schedule.apply @matmul0 tile(%tm,%tn,%tk) stages(%st)
```

See `examples/dnas_graphir_sketch.mlir`.

---

## 6. Minimal Prototype (Python)

We provide a **PyTorch-only** prototype `examples/dnas_schedule_autotune.py` that mirrors Tessera’s API:
- One **MixedOp** (Linear vs GatedMLP).
- A **ScheduleSpace** with tile choices and stages.
- A **surrogate cost** (FLOPs/bytes + schedule penalty) with a JSON **cache** keyed by shape.
- Alternating updates for **weights**, **arch logits**, and **schedule probs**.

This prepares the integration path to the real Tessera compiler passes and runtime.
