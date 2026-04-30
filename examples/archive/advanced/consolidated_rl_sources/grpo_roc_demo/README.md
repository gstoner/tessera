# GRPO-RoC Demo (Minimal Working Example)

A small, runnable reference that implements **Group Relative Policy Optimization with Resample-on-Correct (GRPO‑RoC)** for an agent that solves integer math questions with optional "tool calls" into a restricted Python sandbox.

## What this includes
- A minimal multi‑turn rollout loop with a function‑call style tool interface (`tool_call{...}` / `/tool_call`).
- Binary answer‑only rewards (correct/incorrect) with integer answer checking.
- GRPO objective with group‑relative advantages and Clip‑Higher style clipping (`eps_low=0.2`, `eps_high=0.28`).
- RoC sampler: oversample rollouts, keep half negatives uniformly, keep half positives by inverse penalty (tool‑error ratio + format violations).
- A tiny LSTM toy policy (no external downloads) and an optional HF Transformers policy stub you can wire up.
- Synthetic integer‑math dataset to make it runnable on CPU.

### Quick start
```bash
cd grpo_roc_demo
python -m venv .venv && source .venv/bin/activate
pip install -U torch pyyaml
python run_demo.py --steps 5 --device cpu
```
