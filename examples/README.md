# Tessera Examples

This directory contains examples and tutorials for using Tessera.

## Getting Started

- [`basic_tensor_ops.py`](getting_started/basic_tensor_ops.py) - Basic tensor operations
- [`tessera_flash_attention_demo/`](getting_started/tessera_flash_attention_demo/) - Flash Attention usage

## Compiler Examples

- [`compiler/dnas/`](compiler/dnas/) - differentiable NAS Graph IR and schedule-search examples
- [`compiler/ir_pipeline_tutorial/`](compiler/ir_pipeline_tutorial/) - Python to Graph/Schedule/Tile/Target IR walkthrough

## Conformance Examples

- [`conformance/apple_path_ga_ebm_demos.py`](conformance/apple_path_ga_ebm_demos.py) - visible GA + EBM demos for the Apple-path story: rotation-invariant point-cloud features and EBT-tiny inner-loop refinement

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
- [`advanced/gumiho/`](advanced/gumiho/) - Gumiho (ICML'25) hybrid speculative decoding on the Apple GPU/CPU backend
- [`advanced/Tessera_Empirical_Software_Agent/`](advanced/Tessera_Empirical_Software_Agent/) - agentic kernel autotuning loop

Archived advanced examples are preserved under [`archive/examples/advanced/`](../archive/examples/advanced/).

## Optimization

- [`src/01_loop_tiling_blocking.cpp`](optimization/src/01_loop_tiling_blocking.cpp) - Loop tiling and blocking
- [`src/02_vectorization_intrinsics.cpp`](optimization/src/02_vectorization_intrinsics.cpp) - Vectorization intrinsics
- [`src/05_async_copy_tma_wgmma.cu`](optimization/src/05_async_copy_tma_wgmma.cu) - Async copy, TMA, and WGMMA sketch
- [`src/08_mlir_gemm_tiled.mlir`](optimization/src/08_mlir_gemm_tiled.mlir) - MLIR tiled GEMM example

## Running Examples

```bash
# Basic examples
PYTHONPATH=python python3 examples/getting_started/basic_tensor_ops.py

# Advanced examples
python3 examples/advanced/rlvr_reasoning_suite/run_demo.py --steps 4
python3 examples/advanced/kv_cache_serving/demo.py --requests 8

# Compiler examples
PYTHONPATH=python python3 examples/compiler/dnas/dnas_schedule_autotune.py
PYTHONPATH=python python3 examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py

# Conformance demos
.venv/bin/python examples/conformance/apple_path_ga_ebm_demos.py
```
