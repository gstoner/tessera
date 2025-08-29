# Tessera Examples

This directory contains examples and tutorials for using Tessera.

## Getting Started

- [`basic_tensor_ops.py`](getting_started/basic_tensor_ops.py) - Basic tensor operations
- [`flash_attention_demo.py`](getting_started/flash_attention_demo.py) - Flash Attention usage
- [`first_model.py`](getting_started/first_model.py) - Your first Tessera model

## Advanced Examples

- [`transformer/`](advanced/transformer/) - Complete transformer implementation
- [`mla/`](advanced/mla/) - Multi-Latent Attention examples  
- [`hrm/`](advanced/hrm/) - Hierarchical Reasoning Models

## Optimization

- [`autotuning_demo.py`](optimization/autotuning_demo.py) - Automatic parameter tuning
- [`memory_optimization.py`](optimization/memory_optimization.py) - Memory efficiency
- [`kernel_fusion.py`](optimization/kernel_fusion.py) - Operation fusion

## Running Examples

```bash
# Basic examples
cd examples/getting_started
python flash_attention_demo.py

# Advanced examples  
cd examples/advanced/transformer
python model.py
```
