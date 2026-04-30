# Advanced Examples

This directory is now focused on modern AI research examples that map naturally to
Tessera's compiler, runtime, and hardware-aware scheduling model.

## Active Research Examples

- `Fast_dLLM_v2/` - diffusion LLM inference, confidence-aware parallel decoding,
  and approximate KV-cache planning.
- `rlvr_reasoning_suite/` - consolidated GRPO/RLVR reasoning suite with rollout
  batching, verifiers, reward telemetry, and trainer accounting.
- `Jet_nemotron/` - hybrid efficient LM scaffold with PostNAS, linear attention,
  streaming state, and hardware-aware search hooks.
- `Nemotron_Nano_12B_v2/` - hybrid Mamba2/GQA/MLP model-port starter with 128K
  context-oriented state handling.
- `mla/` - Multi-Latent Attention / FlashMLA notes and implementation sketches.
- `power_retention/` - active PowerAttention/retention port; older versioned
  drops are archived.
- `Tessera_Empirical_Software_Agent/` - tree-search agent scaffold now including
  a concrete kernel-autotuning task.
- `long_context_attention/` - retrieval-head vs streaming-head specialization.
- `kv_cache_serving/` - TurboQuant/DuoAttention/Mooncake-style cache compression
  and disaggregated long-context serving planner.
- `speculative_decoding/` - Yggdrasil/Medusa/EAGLE-style tree decoding scheduler.

## Archive

Older, duplicate, or less-current examples were preserved under:

```text
examples/archive/advanced/
```

The RL source drops consolidated into `rlvr_reasoning_suite/` are under:

```text
examples/archive/advanced/consolidated_rl_sources/
```
