# Tessera Lifelong Learning Guide

Lifelong learning a first-class citizen in Tessera, tied into Graph IR, Schedule IR, the runtime, and the inference server.

Goals
	•	No catastrophic forgetting while the model keeps learning.
	•	Fast online adaptation with bounded compute/memory.
	•	Native meta-learning (MAML/implicit grads/hypernets).
	•	Few-shot readiness (PEFT, prompt/adapter routing, retrieval).
	•	Deterministic, resumable, multi-GPU friendly.

⸻

Core abstractions (new in Tessera)
	•	Anchors & Deltas
	•	arch.Anchor(name, params, stats) snapshot of weights/statistics (and Fisher info).
	•	arch.Delta(layer_mask|LoRA|IA3|Prefix) parameter-efficient overlays.
	•	Experience Store
	•	data.ReplayBuffer(capacity, policy="reservoir|prioritized")
	•	data.Stream(shards, window, drift_detector=...)
	•	Continual Regularizers
	•	loss.ewc(anchor, λ), loss.si(anchor, c), loss.lwf(prev_logits, T, λ), loss.l2sp(anchor, λ)
	•	opt.ogd(project="orthogonal") (orthogonal gradient projection)
	•	Meta-Learning Primitives
	•	meta.maml_step(model, batch, inner_opt, K, create_graph=True)
	•	meta.implicit_bilevel(inner_solve="CG|Neumann", hvp="pearlmutter")
	•	meta.hypernet(encoder → fast-weights), meta.fastweights(scope)
	•	Few-Shot / PEFT
	•	peft.lora(modules, r, α), peft.prefix(num_tokens), peft.prompt(vecs)
	•	retrieval.KVCache(index, adapter=...) for RAG/FiD-style conditioning

⸻

Catastrophic Forgetting Prevention

1) Regularization (EWC, SI, LwF, L2SP)
```python
from tessera import graph as tg, ops as T, loss, arch

# After Task t:
anchor = arch.anchor_from(model, name=f"task{t}", fisher="diagonal")  # stores θ_t and F_t

@tg.training_step
def step(batch):
    yhat = model(batch.x)
    task_loss = T.cross_entropy(yhat, batch.y)

    # EWC penalty: Σ_i λ/2 * F_i * (θ - θ*_i)^2  (diagonal Fisher)
    reg = loss.ewc(model, anchor, lam=1e-2)

    # (optional) LwF: distill onto older model’s logits at temperature T
    with tg.no_grad():
        y_prev = anchor.model(batch.x)
    kd = loss.kl_divergence(T.log_softmax(yhat/2.0), T.softmax(y_prev/2.0)) * (2.0**2)

    return task_loss + reg + 0.3*kd
```
2) Rehearsal & Generative Replay
```python 
buf = data.ReplayBuffer(capacity=200_000, policy="reservoir")
# write streaming
buf.add(batch.x, batch.y)

# mixed batch: new + replay
x_new,y_new = batch.x, batch.y
x_rep,y_rep = buf.sample_like(x_new, ratio=0.25)
x_mix = T.concat([x_new, x_rep]); y_mix = T.concat([y_new, y_rep])
```
3) Parameter Isolation (PEFT) + OGD
	•	Keep a frozen base; learn deltas per domain/task via LoRA/IA³/prefix.
	•	Apply Orthogonal Gradient Descent to discourage interference:
```python
opt = tg.Adam(model.parameters(), lr=2e-4, plugin=opt.ogd())
```
Online Adaptation
	•	Streaming optimizer & stats: per-stream EMA, streaming BN/LayerNorm with decay β_stream.
	•	Recency-weighted losses: loss = (1-ρ)*L_replay + ρ*L_new with ρ scheduled by drift.
	•	Drift detection: page-Hinkley/ADWIN hooks on validation loss → change adaptation rate.
	•	Hot recompile small deltas: enqueue only changed ops to autotuner; persist shape+arch cache.
```python
sched = tg.OnlineSchedule(allow_hot_swap=True, max_latency_ms=10)
with sched.stream("updates"):    # low-priority stream
    autotuner.try_update(model, shapes=observed, budget_ms=2)
```
Meta-Learning Primitives

1) First-order & higher-order MAML
```python
from tessera import meta

def bilevel_step(task_batch, val_batch):
    # inner loop: adapt K steps
    adapted, meta_state = meta.maml_step(model, task_batch, inner_opt=tg.SGD(1e-2), K=5,
                                         create_graph=True)  # higher-order
    # outer loss on held-out
    yhat = adapted(val_batch.x)
    outer = T.cross_entropy(yhat, val_batch.y)
    return outer
```
- Efficient HVP via Pearlmutter; implicit gradients to avoid storing full unrolled graphs:
```python
outer = meta.implicit_bilevel(model, inner=..., val_batch=...)
```
2) Hypernetworks & Fast Weights
```python
task_embed = meta.task_encoder(context)     # few examples → embedding
fast = meta.hypernet(task_embed)            # produces low-rank deltas
with meta.fastweights(model, fast):         # scoped overlay
    yhat = model(x_support)
```
Few-Shot Learning Built-In
	•	Adapters/Prompts/LoRA toggled per request; delta params kept small & cacheable.
	•	Retrieval conditioning: retrieval.index.encode(corpus), retrieval.query(x) returns adapters/prompts to compose.
	•	Inference server supports ephemeral session deltas: attach, time-limit, GC.
```python
# 3-shot adaptation at inference
delta = peft.lora(model, r=8)
peft.fit(delta, shots)           # tiny inner loop on support examples
yhat = peft.apply(model, delta)(x_query)
```
IR & Runtime hooks

	-	Graph IR: arch.parameter, loss.ewc/si/lwf, peft.delta_apply, meta.maml_step are first-class ops with AD rules.
	-	Schedule IR: a low-priority “background” pass queue for incremental autotuning; shape/arch persistent cache by SM arch.
	-	Tile/Target IR: no special changes—lifelong patterns are mostly Graph/Schedule level; kernels remain the same.
	-	Checkpointing: versioned {weights, anchors(Fisher), deltas, replay metadata, RNG}; supports elastic re-sharding.

⸻

Distributed + Inference Server

	-	Anchors sharded with DP/TP; Fisher diagonals reduced deterministically.
	-	Replay buffer can be sharded with local sampling + periodic merge.
	-	Inference server:
	-	Hot-swap deltas per tenant/namespace,
	-	A/B gating and rollback,
	-	Budget caps for online steps (lat/energy constraints),
	-	Deterministic replay for post-mortems.

⸻

Minimal worked snippets

A) EWC + Replay (streaming)
```python
buf = data.ReplayBuffer(200_000, "reservoir")
anchor = arch.anchor_from(model, "pre_task")

@tg.training_step
def continual(batch):
    buf.add(batch.x, batch.y)
    x_r, y_r = buf.sample_like(batch.x, ratio=0.25)
    x = T.concat([batch.x, x_r]); y = T.concat([batch.y, y_r])

    logits = model(x)
    task = T.cross_entropy(logits, y)
    reg  = loss.ewc(model, anchor, lam=5e-3)
    return task + reg
```
B) Few-shot with LoRA overlay
```python
delta = peft.lora(model.layers.attn, r=8, alpha=16)
peft.fit(delta, support_batch, steps=20, lr=5e-3)   # quick inner loop
pred = peft.apply(model, delta)(query.x)
```
C) Meta-learning (implicit bilevel)
```python
outer = meta.implicit_bilevel(
    model,
    inner=lambda m: tg.SGD(m, lr=1e-2, steps=5),
    train_batch=task_support,
    val_batch=task_query,
    hvp="pearlmutter", tol=1e-3
)
outer.backward(); meta_opt.step()
```
Measurement & QA

	-	Report Avg Acc, Backward Transfer (BWT), Forward Transfer (FWT), Forgetting per task stream.
	-	CI: tiny stream dataset; ensure replay/anchors/deltas are versioned and deterministic (seed + ordered reductions).
	-	Perf budgets: online step ≤ X ms, ≤ Y joules (use Tessera profiler counters).

⸻

What I can add next

	-	A Lifelong Learning Guide markdown with these APIs and 3 runnable examples:

            1.	EWC+Replay on a stream benchmark,
            2.	Few-shot LoRA at inference server,
            3.	MAML with implicit gradients.

	-	Hooks into the unified skeleton: unified/examples/continual/ + CTest sanity runs.

**Scope.** Practical patterns and APIs for continual, online, and meta-learning in Tessera. This guide mirrors CUDA’s pragmatic style: conceptual overview → API surface → worked examples → best practices.

> Status: Draft. APIs shown as *Tessera-style* pseudocode with runnable PyTorch prototypes in `examples/`.

---
(abridged — see examples for details)
