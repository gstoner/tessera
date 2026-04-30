# Multi-Layer GRPO (MGRPO) — Minimal Sample

This repo shows a minimal, two-layer GRPO training loop for reasoning/self-correction, inspired by:
Ding et al., *Multi-Layer GRPO: Enhancing Reasoning and Self-Correction in Large Language Models* (2025).

## What it does
- **Layer 1 (GRPO):** sample K responses per prompt, compute outcome reward (rule-based), update policy via GRPO.
- **Layer 2 (GRPO):** concat original query + Layer‑1 response + guiding phrase; sample M corrections; reward successful **corrections/confirmations** and update the **same policy**.
- **Shared policy** across both layers; **reference policy** is frozen for KL penalty.
- **Group baseline**: advantages = reward − mean(reward_group).
- Minimal **math rewarder** looks for `<answer>...</answer>` and checks numeric equivalence.

## Quick start
```bash
python -m pip install -U transformers accelerate datasets torch numpy pyyaml
python scripts/train_mgrpo.py --config configs/mgrpo_math.yaml
```

- Edit `configs/mgrpo_math.yaml` to choose a small model (e.g., `sshleifer/tiny-gpt2` for CPU smoke tests).
- Replace the rule-based rewarder for your own domain; plug in datasets via JSONL (`data/sample_math.jsonl`).

License: MIT
