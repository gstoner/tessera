# RLVR Reasoning Suite

This example consolidates the earlier GRPO, multi-layer GRPO, and FlowRL drops into
one Tessera-facing reasoning-training suite.

The goal is a small but serious reference for reinforcement learning with verifiable
rewards:

- batched rollouts for math and code-style tasks
- answer and program verifiers
- GRPO-style group-relative advantages
- optional resample-on-correct filtering
- reward and verifier telemetry
- hook points for Tessera kernels, profiler feedback, and distributed rollout workers

The archived source material lives under:

```text
examples/archive/advanced/consolidated_rl_sources/
```

## Quick Start

```bash
python3 examples/advanced/rlvr_reasoning_suite/run_demo.py --steps 4 --group-size 6
```

The default run uses a deterministic toy policy so the example has no external model
download. Swap `ToyPolicy` with a Hugging Face or Tessera policy once the runtime is
connected.

## Layout

```text
rlvr_suite/
  dataset.py      tiny generated math/code tasks
  verifier.py     exact answer and unit-test verifiers
  rollout.py      grouped rollout collection
  trainer.py      GRPO/RoC update accounting
  telemetry.py    JSONL reward logging
```

## Tessera Integration Points

- Lower verifier batches to a sandboxed test kernel when code tasks dominate.
- Use Tessera schedule/autotune to batch rollouts by prompt length and verifier cost.
- Export group advantage and reward histograms to the profiler so failed reasoning
  modes can drive prompt/template or kernel changes.
- Replace the toy policy with a Tessera model using the same `sample(prompt, n)` API.
