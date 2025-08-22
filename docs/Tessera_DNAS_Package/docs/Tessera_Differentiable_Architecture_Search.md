# Tessera Differentiable Architecture Search (DNAS)

This document describes how Tessera can support Differentiable Neural Architecture Search (DNAS).

1) Search space as first-class Graph IR

Mixed operators (continuous relaxations)

Treat every “choice” as a MixedOp that contains K candidate ops and a learnable architecture logit vector α ∈ ℝ^K.

```python
from tessera import graph as tg, ops as T, arch

class MixedOp(tg.Module):
    def __init__(self, candidates, relax="gumbel", temperature=5.0):
        super().__init__()
        self.candidates = tg.ModuleList(candidates)     # e.g., conv3x3, conv5x5, identity
        self.alpha = arch.Parameter(len(candidates))    # architecture logits
        self.relax = arch.GumbelSoftmax(self.alpha, temperature=temperature) if relax=="gumbel" \
                     else arch.Softmax(self.alpha)

    def forward(self, x):
        gate = self.relax()      # shape [K], differentiable probs
        outs = [op(x) for op in self.candidates]
        return T.weighted_sum(outs, gate)   # Σ_i gate[i] * op_i(x)
```
Tessera Graph IR needs:

	-	arch.Parameter (lives alongside model weights but separate optimizer & schedulers).
	-	arch.GumbelSoftmax, arch.HardConcrete, arch.STEOneHot (straight-through) building blocks.
	-	arch.weighted_sum / arch.switch operators that are autodiff-aware.

What’s searchable?

	-	Op choice: attention variant (Flash, Performer/Kerna, SDPA), MLP vs. gated MLP.
	-	Dimensions: hidden size, FFN expansion ratio, #heads, groups, kernel sizes.
	-	Topology: skip vs. residual mix, depth expansion (layer replication with gates).
	-	Schedule knobs (optional): tile sizes/stages, tensor layouts—via Schedule IR (see §4).

⸻

2) Bilevel optimization (weight vs. architecture)

Classic DARTS-style bilevel: inner loop updates weights W on train split, outer loop updates architecture logits α on val split.
```python
opt_w  = tg.Adam(model.weight_parameters(), lr=3e-4)
opt_a  = tg.Adam(model.arch_parameters(),   lr=1e-2, weight_decay=0.0)

for step, (train_batch, val_batch) in tg.zipcycle(train_loader, val_loader):
    # Inner: update W on training loss
    with tg.grad_enabled(True):
        loss_train = T.cross_entropy(model(train_batch.x), train_batch.y)
        opt_w.zero_grad(); loss_train.backward(wrt="weights"); opt_w.step()

    # Outer: update α on validation objective (task + hardware costs)
    with tg.no_grad_for("weights"):   # freeze W
        lat, energy, mem = arch.hw_cost(model, device="auto")   # §3
        loss_val = T.cross_entropy(model(val_batch.x), val_batch.y) \
                 + 1e-3 * lat + 1e-4 * energy + 1e-4 * mem
        opt_a.zero_grad(); loss_val.backward(wrt="arch"); opt_a.step()
```

Options:

	-	Unrolled K-step inner optimization (memory-heavy but exact).
	-	Implicit gradient (Hessian-vector via CG/Neumann) for lower memory.
	-	Temperature annealing on Gumbel-Softmax to move from soft → near-discrete.

Stability helpers:

	-	Clip ∥∇α∥, constrain skip-connect logits, L0/HG (Hard-Concrete) sparsity on edges to avoid collapse.
	-	Keep arch params in fp32 regardless of AMP; weights can be mixed precision.

⸻

3) Hardware-aware differentiable objective

You need a smooth proxy for latency/energy/memory that can participate in gradients:

Two-tier cost model

	1.	Analytical/IR-aware estimator (fast, differentiable):
From Schedule IR decisions + Graph IR features (tensor sizes, flops/bytes) → predict {lat, energy, mem} via parametric formulas.
	2.	Learned surrogate (smooth):

Small MLP f_φ(features(α, shape, arch)) → {lat, energy, mem} trained online from on-device measurements.
```python
lat_pred, energy_pred, mem_pred = arch.cost_model.predict(model)
loss_val = task_loss + λ1*lat_pred + λ2*energy_pred + λ3*mem_pred
```

Closing the loop: measure → fit → use
```python
meas = arch.measure(model, reps=5, metric=("latency","energy"))
arch.cost_model.update(model.features(), meas)   # fit φ by SGD; keeps differentiability for α
```
For constraints (e.g., latency ≤ 3.5 ms), use soft constraints (hinge/barrier) or Augmented Lagrangian with dual variables that are also optimized.

⸻

4) Joint search over architecture and schedule

Many wins are at the schedule/layout level. Let Tessera expose Schedule IR knobs as differentiable gates or slow outer variables:
```python
sched = arch.ScheduleSpace({
  "tile_m": [64, 128], "tile_n": [128, 256], "tile_k": [32, 64],
  "stages": [2, 3, 4],  "layout": ["row", "col", "tiled(64)"]
})

with arch.relax(sched):    # internally uses softmax/gumbel over discrete options
    y = model(x)           # codegen consults current relaxed choices
    cost = arch.cost_model.predict_from_schedule(sched.current())
    loss = task(y, y_) + λ*cost
```
Practical recipe:

	-	Alternate: a few steps on W/α (ops), then a few on schedule α_s (tiling/layout).
	-	Or nest: architecture is outer loop, schedule tuned by Tessera’s autotuner with a differentiable surrogate and occasional real measurements (non-diff).

⸻

5) From soft choices → discrete specialization

When validation stabilizes:

	1.	Select per-edge argmax (or sample with temperature→0).
	2.	Prune unused ops/branches.
	3.	Specialize & lower: freeze to Graph IR, re-run Schedule IR autotuning, generate Tile IR/Target IR.
	4.	Retrain (short finetune) with the discrete network.

```python
choices = arch.argmax(model)                       # {edge: index}
frozen  = tg.specialize(model, choices)            # deletes unused ops/params
plan    = tg.compile(frozen).autotune().lower()    # emits CUDA/ROCm/CPU
```

6) Distributed & deterministic

	•	Gradients on α: allreduce across DP mesh; use Kahan or ordered reductions for determinism.
	•	Population NAS (optional): split submeshes each exploring different α; periodically average logits (elastic DNAS).
	•	Checkpointing: store {W, α, cost_model_φ}; resume exactly (cost model seeds, RNG states).

⸻

7) Example: Attention-block search (Flash vs. Performer vs. Gated-MLP)
```python
def make_search_attn(embed, n_heads):
    return MixedOp([
        T.flash_attention(embed, n_heads, causal=True),
        T.performer_attention(embed, n_heads, kernel="relu"),
        T.multi_query_attention(embed, n_heads//2),
        T.gmlp_block(embed, expansion=4)             # a non-attention alternative
    ], relax="gumbel", temperature=4.0)

class SearchTransformerBlock(tg.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = T.layer_norm(d)
        self.attn  = make_search_attn(d, heads)
        self.norm2 = T.layer_norm(d)
        self.mlp   = MixedOp([
            T.ffn(d, 4*d, act="gelu"),
            T.ffn(d, 2*d, act="silu"),
            T.gated_ffn(d, 4*d)
        ])
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```
Hardware term: FlashAttention is often faster but more memory-bound at long seq; Performer scales linearly; gMLP has different arithmetic intensity. The differentiable cost will nudge α toward the variant best for your sequence length, batch, and GPU.

⸻

8) Practical knobs & guardrails

	-	Regularize α: entropy bonus early (explore), L0 penalty late (sparsify).
	-	Temperature schedule: τ: 5 → 0.3 over ~10–30% of training.
	-	Early stopping on α if probabilities saturate.
	-	Precision: keep α & cost model in fp32; AMP for weights only.
	-	Logging: track per-edge probs, expected FLOPs/bytes, predicted vs. measured latency drift.

⸻

9) What to implement in Tessera (checklist)

Graph IR / TSOL

	-	arch.Parameter, GumbelSoftmax, HardConcrete, STEOneHot
	-	arch.MixedOp, arch.relax(sched), weighted_sum, switch
	-	bilevel_step() helper (unrolled & implicit variants)
	-	Feature extractor for {flops, bytes, params, tiles, seq_len, sm_arch, bw, clock}

Schedule IR

	-	Knob objects with relaxed gates; export features for cost model
	-	Ties to autotuner (measurements → dataset for surrogate)

Runtime

	-	Deterministic allreduce for α
	-	Efficient small-tensor updates (α are tiny)

Tooling

	-	Dashboards for α histograms & HW cost curves
	-	Checkpointing of {W, α, φ}

⸻

10) Minimal training loop (all together)
```python
model = SearchTransformerBlock(d=2048, heads=16)
opt_w = tg.Adam(model.weight_parameters(), lr=3e-4)
opt_a = tg.Adam(model.arch_parameters(),   lr=5e-3)

for step in range(steps):
    # === train weights ===
    x, y = next(train_loader)
    with tg.autocast(): yhat = model(x)
    loss_w = T.cross_entropy(yhat, y)
    opt_w.zero_grad(); loss_w.backward(wrt="weights"); opt_w.step()

    # === update arch on val + HW cost ===
    if step % 2 == 0:
        xv, yv = next(val_loader)
        with tg.no_grad_for("weights"):
            yhatv = model(xv)
            task = T.cross_entropy(yhatv, yv)
            lat, energy, mem = arch.hw_cost(model)   # surrogate + occasional measure
            loss_a = task + 1e-3*lat + 1e-4*energy + 1e-4*mem
            opt_a.zero_grad(); loss_a.backward(wrt="arch"); opt_a.step()

    arch.anneal_temperature(model, schedule="cosine", t=step/steps)

# Freeze and specialize
choices = arch.argmax(model)
frozen  = tg.specialize(model, choices)
compiled = tg.compile(frozen).autotune().lower()
```
Variants you may want

	•	ProxylessNAS / Path Binarization (binary gates + STE) for even lower overhead.
	•	Once-For-All supernet export: train once, derive many subnets for different devices by adjusting constraints/λ.
	•	Population NAS (multiple α replicas per submesh) + periodic logit averaging (elastic).

## Key Ideas

- MixedOp with differentiable gates (e.g., Gumbel-Softmax).
- Bilevel optimization: weights trained on training set, architecture logits on validation set + hardware cost.
- Surrogate cost model: latency, memory, energy estimated differentiably.
- Annealing schedule for gate sharpness.
- Discretization step to finalize architecture.

## Example Flow

1. Define search space as a set of candidate ops.
2. Relax discrete choice into soft gates (softmax).
3. Train weights + gates with alternating updates.
4. Evaluate with hardware-aware loss.
5. Collapse gates into hard choices for final network.
