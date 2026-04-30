# Tessera Examples

This directory contains examples and tutorials for using Tessera.

## Getting Started

- [`basic_tensor_ops.py`](getting_started/basic_tensor_ops.py) - Basic tensor operations
- [`flash_attention_demo.py`](getting_started/flash_attention_demo.py) - Flash Attention usage
- [`first_model.py`](getting_started/first_model.py) - Your first Tessera model

## Advanced Examples

- [`advanced/Fast_dLLM_v2/`](advanced/Fast_dLLM_v2/) - diffusion LLM inference and parallel decoding
- [`advanced/Diffusion_LLM/`](advanced/Diffusion_LLM/) - diffusion language model package
- [`advanced/rlvr_reasoning_suite/`](advanced/rlvr_reasoning_suite/) - consolidated GRPO/RLVR reasoning suite
- [`advanced/Jet_nemotron/`](advanced/Jet_nemotron/) - hybrid efficient language model and PostNAS scaffold
- [`advanced/Nemotron_Nano_12B_v2/`](advanced/Nemotron_Nano_12B_v2/) - hybrid Mamba2/GQA/MLP model port
- [`advanced/mla/`](advanced/mla/) - Multi-Latent Attention / FlashMLA examples
- [`advanced/power_retention/`](advanced/power_retention/) - retention and PowerAttention kernels
- [`advanced/long_context_attention/`](advanced/long_context_attention/) - retrieval-head vs streaming-head planning
- [`advanced/kv_cache_serving/`](advanced/kv_cache_serving/) - compressed KV-cache and disaggregated serving planner
- [`advanced/speculative_decoding/`](advanced/speculative_decoding/) - tree/speculative decoding scheduler
- [`advanced/Tessera_Empirical_Software_Agent/`](advanced/Tessera_Empirical_Software_Agent/) - agentic kernel autotuning loop

Archived advanced examples are preserved under [`archive/advanced/`](archive/advanced/).

## Optimization

- [`autotuning_demo.py`](optimization/autotuning_demo.py) - Automatic parameter tuning
- [`memory_optimization.py`](optimization/memory_optimization.py) - Memory efficiency
- [`kernel_fusion.py`](optimization/kernel_fusion.py) - Operation fusion

## Running Examples

```bash
# Basic examples
cd examples/getting_started
python3 flash_attention_demo.py

# Advanced examples
python3 examples/advanced/rlvr_reasoning_suite/run_demo.py --steps 4
python3 examples/advanced/kv_cache_serving/demo.py --requests 8
```
