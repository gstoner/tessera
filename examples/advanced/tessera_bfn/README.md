# Tessera Support Plan & Starter Scaffold for Bayesian Flow Networks (BFN)

> Scaffold generated for integrating **Bayesian Flow Networks** into the Tessera Programming Model.
>
> This repository contains:
> - A **design & API plan** to map BFN concepts to Tessera IR and runtime,
> - A minimal **Python API** (stubs) for training/sampling,
> - A **dialect sketch** and **lowering notes** for compiler integration,
> - A couple of **tests** to lock in the public surface as it evolves.

---

## 1) Executive summary

Bayesian Flow Networks (BFN) operate on **distribution parameters** rather than directly on noisy data. At each step:
1. We feed the parameters of a **factorized input distribution** into a neural net to obtain an **interdependent output distribution**.
2. We **convolve** the output distribution with the noise schedule to form a **receiver**.
3. A **sender** distribution is obtained by adding noise to the (possibly discrete) data per schedule.
4. The per‑step **loss** is the KL(receiver || sender).
5. We draw a sender sample and perform a **closed‑form Bayesian update** to the input distribution, and repeat for *n* steps.

This yields a generative process akin to the reverse process in diffusion models, but **requires no forward diffusion**, supports **continuous or discrete‑time losses**, and handles **discrete data** natively by operating on the probability simplex.

---

## 2) Minimal Tessera API (Python stubs)

```
tessera.models.bfn
├─ BFNConfig
├─ accuracy: schedules (beta(t)) + utilities
├─ ops: sender, receiver (convolution), kl_div, bayes_update, step
├─ loss: discrete_time_loss, continuous_time_loss
└─ sampling: generate(..., n_steps)
```

### Example (API sketch)

```python
from tessera.models.bfn import BFNConfig, step, discrete_time_loss, sampling

cfg = BFNConfig(
    data_kind="discrete",                # "discrete" | "discretized" | "continuous"
    family="categorical",               # "categorical" | "gaussian" | "discretized_bins"
    n_steps=100,
    accuracy_schedule="cosine",         # "linear" | "cosine" | "poly" | custom lambda
    net="unet_small",                   # any Tessera module; opaque to BFN
)

# tensors:
# params_in: factorized parameters (e.g., probabilities for categorical)
# x:         raw data (ints for discrete, float for continuous, etc.)
# t:         discrete step or continuous time; broadcastable

loss = discrete_time_loss(params_in, x, t, cfg)   # returns scalar loss
params_next = step(params_in, x, t, cfg)          # single BFN update step
samples = sampling.generate(params_in, n_steps=50, cfg=cfg)
```

**Notes**
- For *discrete* data, `params_in` are **simplex** tensors (per‑site probability vectors).
- For *continuous* data, `params_in` usually store mean/variance per variable.
- The **network** is pluggable: BFN only assumes it maps factorized input params (+ optional `t`) → output params.

---

## 3) IR sketch (dialect + ops)

We introduce a few ops in a `tessera.bfn` dialect (names tentative):

```
%b0   = bfn.paramize %x                  : Converts raw data → factorized params, if needed
%recv = bfn.receiver %out, %beta         : Convolution of output with noise (receiver)
%send = bfn.sender  %x,   %beta          : Noisy sender distribution from data
%kl   = bfn.kl %recv, %send              : KL(receiver || sender)
%post = bfn.bayes_update %prior, %samp   : Closed‑form update of factorized prior
%next = bfn.flow_step %prior, %x, %t, ^net_region(...) : Composes the above, calls net
```

**Types**
- `!tsr.simplex<N>`: statically‑sized probability simplex vector (sum=1, >=0)
- `!tsr.gauss{mean: f32, var: f32}`: packed mean/variance representation
- `!tsr.acc` (accuracy): scalar controlling sender/receiver noise, with **additivity** property

**Patterns / Lowerings**
- Fuse `bfn.receiver` + `bfn.kl` to a single tiled kernel when shapes align.
- Specialize categorical paths to **log‑space softmax + logsumexp‑stable** kernels.
- Lower `bfn.bayes_update` to family‑specific elementwise kernels (categorical, Gaussian, discretized bins).

---

## 4) Compiler pipeline integration

**Graph IR → Schedule IR**
- Outline `bfn.flow_step` regions, hoist accuracy schedule, and annotate fusion groups (net matmuls + bfn ops).
- Introduce a `bfn-cleanup` pipeline to fold constants, normalize simplex types, and inline tiny updates.

**Schedule IR → Tile IR**
- Tile categorical dimensions (alphabet size) for cache‑friendly softmax/logsumexp.
- Co‑tile network matmuls with adjacent BFN ops; prefer **MMA tiles** to keep data on‑chip.
- Insert software pipelining (double‑buffering) across steps for few‑step generation.

**Tile IR → Target IR**
- **NVIDIA**: WGMMA/WMMA tiles for network matmuls; tensor‑core friendly softmax/softplus; use TMA for bulk copies.
- **ROCm**: MFMA coverage + fused epilogues; map LDS barriers around receiver/kl fusions.
- **CPU**: AVX2/AVX‑512/AMX paths for softmax/logsumexp/kl; threaded over batch/time.

---

## 5) Numerical notes

- Keep **simplex** in log‑space where feasible; normalize with `logsumexp`.
- Accuracy schedule `beta(t)` must be **additive** across steps; expose as attribute & precompute table.
- Continuous‑time loss: sample `t ~ U(0,1)`; evaluate instantaneous KL‑rate; accumulate by MC average.
- Mixed precision: store params in FP16/BF16 but compute KL in FP32; guard small probabilities with eps.

---

## 6) Tests & datasets

- MNIST (discrete), CIFAR‑10 (discretized), text8 (discrete) parity tests for loss and sample generation.
- Golden values: small toy problems (2–4 categorical states; 1D Gaussian) with analytic KL and updates.
- Determinism: Philox seed plumbing, CPU reference path, tolerance bands per dtype/backend.

---

## 7) Roadmap

- [ ] Implement categorical and Gaussian families end‑to‑end (sender/receiver/kl/update).
- [ ] Add continuous‑time loss path & benchmarks (few‑step vs many‑step quality).
- [ ] Kernel fusions for `receiver+kl` and log‑space softmax/normalization.
- [ ] End‑to‑end examples for MNIST/CIFAR10/text8 using Tessera runners.
- [ ] Perf micro‑benches (n_steps sweep; alphabet‑size sweep; dtype sweep).

---

## 8) File layout

```
tessera_bfn_starter/
├── README.md                         # this file
├── tessera/models/bfn/__init__.py
├── tessera/models/bfn/config.py
├── tessera/models/bfn/accuracy.py
├── tessera/models/bfn/ops.py         # sender, receiver, kl, bayes_update, step (stubs)
├── tessera/models/bfn/loss.py        # discrete & continuous time loss (sketch)
├── tessera/models/bfn/sampling.py    # few-step generator
├── mlir/bfn/bfn_dialect.td           # dialect sketch
├── mlir/bfn/passes.td
├── mlir/bfn/lowering_notes.md
└── tests/test_bfn_shapes.py
```

This is a focused scaffold intended to drop under `tessera/` in your repo and evolve incrementally.
