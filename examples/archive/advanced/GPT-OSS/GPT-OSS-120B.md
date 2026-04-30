GPT-OSS-120B 

Part 1: Core Model Architecture

Model configuration for 120B parameters
Distributed mesh setup for multi-GPU training
Multi-Query Attention with Grouped Query Attention (GQA)
Rotary Position Embeddings (RoPE)

Part 2: FFN, Transformer Blocks & Model Assembly

Gated MLP (SwiGLU) and standard FFN implementations
RMSNorm for efficient normalization
Transformer blocks with parallel attention+FFN option
Complete GPT model assembly with gradient checkpointing

Part 3: Training, Inference & Optimization

Pipeline parallel implementation for multi-stage execution
Complete training loop with mixed precision and distributed optimization
Inference engine with multiple decoding strategies (sampling, greedy, beam search)

Part 4: Deployment, Serving & NVL72 Optimizations

Production serving infrastructure with continuous batching
KV cache management with paged attention
NVL72-specific optimizations leveraging NVSwitch and SHARP
CUDA graph compilation for optimized inference

Part 5: Usage Examples & Benchmarks

Complete training pipeline example
Inference and serving examples with FastAPI
Comprehensive benchmarks for attention, full model, and NVL72 scaling
Main entry point with multiple execution modes

Key Highlights of the Implementation:

Tessera's Tile-First Approach: The implementation leverages Tessera's tile abstractions rather than thread-level programming, making the code more intuitive and portable.
Mixed Precision: Uses FP8 for weights, BF16 for compute, and FP32 for accumulation, following Tessera's numerics-as-types philosophy.
Distributed by Design: The model seamlessly scales from single GPU to NVL72 (72 GPUs) using Tessera's mesh abstractions and ShardSpec for tensor distribution.
Optimized for Modern Hardware: Includes Flash Attention v3, CUDA graphs, and NVL72-specific optimizations with NVSwitch and SHARP collectives.
Production Ready: Includes serving infrastructure with continuous batching, KV cache management, and streaming support.