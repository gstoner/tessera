# Nemotron‑Nano‑12B‑v2‑Base → Tessera Port (Starter Kit)

> This is a **repo‑ready starter** for porting **NVIDIA‑Nemotron‑Nano‑12B‑v2‑Base** to the **Tessera Programming Model**.
> It includes: config mirroring, a hybrid (Mamba2/MLP/Attention) graph builder, a HF→Tessera checkpoint
> converter skeleton, a minimal inference runner, and MLIR Graph‑IR snippets to drive Tessera’s pipelines.

**What you get**

- `configs/nemotron_nano_12b_v2_base.yaml` — mirrors HF `config.json` (heads/layers/dims/pattern).
- `tessera/model/nemotron_h/` — hybrid block definitions (Mamba2, GQA Attention, MLP[ReLU^2]) and a stack
  assembler driven by `hybrid_override_pattern`.
- `scripts/convert_hf_checkpoint.py` — skeleton to snapshot/download HF weights and export a Tessera‑friendly
  shard layout (NPZ per layer + manifest).
- `tessera/graph_examples/nemotron_2L_graph.mlir` — tiny 2‑layer demo graph showing the three block kinds.
- `tests/smoke_random.py` — shape/prog‑flow smoke test with random weights and 2‑layer toy config.
- `LICENSE-THIRD-PARTY-README.md` — license pointers and usage notes (weights not included).

> ⚠️ This is *not* a full performance port yet. The Mamba2 mixer is provided as a reference kernel stub with clear
> hook points to implement real Tile‑IR / Target‑IR kernels (GPU/CPU/ROCm/NVIDIA backends). Attention uses a
> generic GQA path; swap in your Flash‑Attention Tile kernels when ready.

## Quick start (convert + run tiny smoke)

```bash
# 1) Create a venv and install deps (Transformers for conversion only)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip torch transformers huggingface_hub safetensors numpy

# 2) Convert HF weights (manifest + shards) — defaults to BASE ckpt
python scripts/convert_hf_checkpoint.py --repo nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base --out ./artifacts

# 3) Run a 2‑layer toy graph for smoke testing (random weights)
python tests/smoke_random.py
```

## How this maps to Nemotron‑H (v2‑Base)

- **Hybrid pattern** drives which block type to instantiate per layer:
  `M` = Mamba‑2 mixer, `*` = Attention (GQA), `-` = MLP(ReLU²).
- **128K context** supported by streaming state in Mamba2 and standard KV‑cache for the few attention layers.
- **Key dims** (from HF config): hidden=5120, heads=40, head_dim=128, kv_heads=8, layers=62, ff=20480.

See `MODEL_SPECS.md` for details and citations.

## Next steps (what to implement in Tessera)

- Replace the reference Mamba2 mixer with a real **Tile‑IR** pipeline:
  1) gate & projections, 2) depthwise causal‑conv, 3) selective state update (A,B,C,Δ), 4) chunked scan
  (size=128), 5) output projection. Provide **Target‑IR** specializations for NVIDIA (WGMMA + TMA), ROCm (MFMA),
  and x86 (AMX/VNNI where applicable).
- Swap the generic attention module with your **FlashAttention Tile‑IR** (GQA, optional paged‑KV).
- Add **sequence‑parallel + tensor‑parallel** sharding export (fits your Shardy‑native path).
- Extend `scripts/convert_hf_checkpoint.py` to strict parity (layer‑wise tests vs HF logits on a 1‑2 layer toy).

_This kit is intentionally small so you can drop it into `tessera/models/` and iterate._
