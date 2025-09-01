# Tessera Jet‑Nemotron (Unofficial Port – Draft)
This repo provides a **programmer‑oriented port scaffold** of NVIDIA's **Jet‑Nemotron** concepts into the **Tessera Programming Model**, including a working **JetBlock** module (dynamic linear attention + dynamic conv mixing), streaming state, and a **PostNAS**-style search scaffold.

> ⚠️ **Status**: Draft implementation for experimentation. The official Jet‑Nemotron code has not yet shipped at the time of writing. This port follows the public paper and README descriptions and stays conservative where details are unspecified.

## References
- Paper: Jet‑Nemotron: Efficient Language Model with Post Neural Architecture Search (2025‑08‑21). arXiv:2508.15884.  
- Repo: NVlabs/Jet‑Nemotron (README highlights PostNAS and JetBlock; code pending).

## What’s here
- `jetblock.py`: JetBlock module combining **dynamic linear attention** and **dynamic depthwise conv** with a learnable gate.
- `dynamic_linear_attention.py`: Linear attention core (feature maps, streaming summaries, Tessera tile kernels).
- `postnas_pipeline.py`: A lightweight **PostNAS** scaffold: freeze MLPs, search attention type/placement and JetBlock hyperparams with Tessera autotune & hardware counters.
- `hf_convert.py`: Helpers to load a HF transformer, freeze MLPs, and replace attention blocks with JetBlock.
- `nvl72_mesh_example.py`: Example of mapping a Jet‑Nemotron stack to an **NVL72 superpod** mesh in Tessera.
- `tests/test_sanity.py`: Quick numeric sanity checks for JetBlock vs. softmax attention on short sequences.

## Quick start (pseudo‑code)
```python
from tessera import jit, autodiff
from tessera_jetnemotron.jetblock import JetBlockConfig, JetBlock

cfg = JetBlockConfig(d_model=2048, n_heads=16, head_dim=128,
                     feature_map="elu1", conv_ks=7, gate="token",
                     attn_dropout=0.0, dtype="fp8_e4m3", accum="fp32")

blk = JetBlock(cfg)

@jit @autodiff
def step(x, state=None):
    y, state = blk(x, state=state, causal=True, streaming=True)
    return y, state
```


## Next steps you can take
1. **Numerical checks**: A/B on short sequences vs softmax attention to sanity‑check outputs; extend `tests/` accordingly.
2. **Precision policies**: Try **fp6/fp4 (Blackwell)** in the type annotations with **fp32 accum**; calibrate with a small KL minimization step.
3. **Scheduling**: Narrow autotune spaces for your target (e.g., **B200 vs H100**), then export tuned artifacts with our **Schedule IR cache**.
4. **NVL72**: Use the mesh layout to **shard long contexts across nodes** and enable **overlap of collectives with compute** (per our Chapter 8 model).

## If you want, I can extend this scaffold with
- A **full Transformer block** that swaps between **full‑attn** and **JetBlock** by layer (to run a full PostNAS sweep).
- A **conversion script** that ports **Llama‑family weights** (Q,K,V, MLP) into the Tessera module layout.
- A **Tessera kernel** that implements a more advanced φ (e.g., **random features**) and block‑wise streaming with **KV‑rolling windows**.


## New in this update
- **Full Transformer block** (`transformer_block.py`) that can **swap per-layer** between **full-attn** and **JetBlock** (for PostNAS sweeps).
- **Llama-family conversion script** (`llama_convert.py`) to map HF weights into Tessera modules.
- **Advanced φ** with **random features** + **block-wise streaming** and **KV-rolling windows** (`advanced_feature_maps.py`).
- `JetBlock` now accepts `feature_map="rf"` to activate random-features linear attention.

### Example: build a mixed-attention Transformer
```python
from tessera_jetnemotron.transformer_block import Transformer, TransformerConfig

tcfg = TransformerConfig(
    d_model=2048, n_heads=16, head_dim=128, mlp_hidden=8192,
    n_layers=24, attn_types=["full" if i % 6 == 0 else "jet" for i in range(24)],
    dropout_p=0.0, dtype="fp8_e4m3", accum="fp32"
)
model = Transformer(tcfg)
```

### Example: enable random-features φ in JetBlock
```python
from tessera_jetnemotron.jetblock import JetBlockConfig, JetBlock

jcfg = JetBlockConfig(d_model=2048, n_heads=16, head_dim=128,
                      feature_map="rf", conv_ks=7, gate="token",
                      attn_dropout=0.0, dtype="fp8_e4m3", accum="fp32")
blk = JetBlock(jcfg)
```

### Example: sliding window streaming
`advanced_feature_maps.linear_attention_rf` maintains a **rolling window** of per-block summaries and subtracts stale contributions when the window advances, enabling **block-wise streaming** over long contexts.


## End-to-end example
Run the minimal example:
```
python examples/e2e_infer.py
```

## PostNAS driver
Run a tiny search and log results:
```
python tools/postnas_driver.py
```
Outputs go to `logs/postnas_results.json` and `.csv`.

## Deterministic RF projections across shards
To enable **on-device** per-head projection generation with distributed-deterministic seeding:
```python
from tessera_jetnemotron.jetblock import JetBlockConfig, JetBlock

jcfg = JetBlockConfig(d_model=2048, n_heads=16, head_dim=128, feature_map="rf")
blk = JetBlock(jcfg).set_rf_on_device(True, base_seed=1234)
```
The seed is **broadcast across the mesh**, so all shards see identical RF projections.


## Tiny training loop
A minimal next-token training script with **loss, backward, and gradient clipping**:
```
python examples/train_tiny.py
```
It uses a synthetic dataset and an AdamW-like optimizer to demonstrate the full loop.


## Training (with grad accumulation, AMP scaler, validation perplexity)
Run the tiny training loop with gradient accumulation, mixed-precision loss scaling, and a validation pass:
```
python examples/train_tiny.py
```
Logs include tokens/sec, grad-norm, and current loss-scale. A periodic **validation loop** computes average loss and **perplexity**.


## Autocast, Checkpointing, and LR Schedulers
- **Autocast**: training loop wraps forward pass in a mixed-precision autocast shim (`utils/autocast.py`).
- **Checkpointing**: `utils/checkpoint.py` saves/loads JSON payloads with `ir_hash` and a stubbed **schedule cache export**.
- **LR Schedulers**: `CosineLR` and `ConstWithWarmup` included; pick in `train_tiny.py`.

Check it out:
```
python examples/train_tiny.py
```
Checkpoints go to `logs/ckpt_stepXXXX.json`; schedule cache metadata to `logs/sched_cache/`.


## CLI + Resume
The tiny trainer now supports **CLI flags** and **resume-from-checkpoint**:
```
python examples/train_tiny.py --vocab 4096 --seq-len 256 --d-model 512 --heads 8 --layers 12 \
  --lr 1e-3 --min-lr 5e-5 --warmup 100 --sched cosine \
  --train-batches 200 --val-batches 20 --accum-steps 8 --grad-clip 1.0 \
  --save-every 50 --sched-cache-dir logs/sched_cache
# Resume:
python examples/train_tiny.py --resume logs/ckpt_step0050.json
```
