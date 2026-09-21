# Nemotron-Nano-12B-v2-Base -> Tessera Port (Starter Kit)

This active directory contains the maintained, dependency-light NumPy reference
and current-compiler smoke for the NVIDIA Nemotron Nano hybrid pattern.

**What you get**

- `configs/nemotron_nano_12b_v2_base.yaml` — mirrors HF `config.json` (heads/layers/dims/pattern).
- `nemotron_nano/` — current-compiler smoke package with a NumPy reference path and Graph IR builder.
- `tessera/graph_examples/nemotron_2L_graph.mlir` — tiny current-dialect graph using registered `tessera.*` ops.
- `tests/smoke_random.py` — no-PyTorch shape/prog-flow and Apple Target IR smoke test.
- `LICENSE-THIRD-PARTY-README.md` — license pointers and usage notes (weights not included).

> ⚠️ This is *not* a full performance port yet. The Mamba2 mixer is provided as a reference kernel stub with clear
> hook points to implement real Tile‑IR / Target‑IR kernels (GPU/CPU/ROCm/NVIDIA backends). Attention uses a
> generic GQA path; swap in your Flash‑Attention Tile kernels when ready.

## Quick Start

```bash
# From the repository root, use the Tessera venv.
PYTHONPATH=python python3 \
  examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py

# Validate the checked-in Graph IR with the current tessera-opt.
PATH="$PWD/build/tools/tessera-opt:/opt/homebrew/opt/llvm@23/bin:$PATH" \
  tessera-opt examples/advanced/Nemotron_Nano_12B_v2/tessera/graph_examples/nemotron_2L_graph.mlir >/tmp/nemotron_graph.mlir
```

Expected smoke output:

```text
OK nemotron tiny: (2, 16, 257) apple_cpu cpu_accelerate
```

The former PyTorch model skeleton and incomplete Hugging Face converter are
preserved under `archive/examples/advanced/Nemotron_Nano_12B_v2/`; they are not
part of the active examples contract.

## How this maps to Nemotron‑H (v2‑Base)

- **Hybrid pattern** drives which block type to instantiate per layer:
  `M` = Mamba‑2 mixer, `*` = Attention (GQA), `-` = MLP(ReLU²).
- **128K context** supported by streaming state in Mamba2 and standard KV‑cache for the few attention layers.
- **Key dims** (from HF config): hidden=5120, heads=40, head_dim=128, kv_heads=8, layers=62, ff=20480.

See `MODEL_SPECS.md` for details and citations.

## Next steps (what to implement in Tessera)

- Replace the current compiler smoke Mamba2 approximation with a real **Tile-IR** pipeline:
  1) gate & projections, 2) depthwise causal‑conv, 3) selective state update (A,B,C,Δ), 4) chunked scan
  (size=128), 5) output projection. Provide **Target‑IR** specializations for NVIDIA (WGMMA + TMA), ROCm (MFMA),
  and x86 (AMX/VNNI where applicable).
- Swap the generic attention module with your **FlashAttention Tile‑IR** (GQA, optional paged‑KV).
- Add **sequence‑parallel + tensor‑parallel** sharding export (fits your Shardy‑native path).
- Extend `scripts/convert_hf_checkpoint.py` to strict parity (layer‑wise tests vs HF logits on a 1‑2 layer toy).

_This kit is intentionally small so you can drop it into `tessera/models/` and iterate._
