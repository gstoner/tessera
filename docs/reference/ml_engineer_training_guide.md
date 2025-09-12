# Tessera ML Training Guide - Distributed Training at Scale

This guide covers training ML models with Tessera, from single-GPU training to massive distributed setups with 72+ GPUs, including data parallelism, tensor parallelism, pipeline parallelism, and mixed strategies.

## Table of Contents

1. [Single GPU Training](#single-gpu-training)
2. [Data Parallel Training](#data-parallel-training)
3. [Tensor Parallel Training](#tensor-parallel-training)
4. [Pipeline Parallel Training](#pipeline-parallel-training)
5. [Hybrid Parallelism Strategies](#hybrid-parallelism-strategies)
6. [Memory Optimization](#memory-optimization)
7. [Training Loops and Optimization](#training-loops-and-optimization)
8. [Monitoring and Debugging](#monitoring-and-debugging)

## Single GPU Training

### Basic Training Setup

```python
import tessera as ts
import numpy as np
from typing import Dict, Any

@ts.function
def compute_loss(
    model_fn: callable,
    inputs: ts.Tensor["B", "S", ts.int32],
    targets: ts.Tensor["B", "S", ts.int32],
    weights: Dict[str, ts.Tensor]
) -> ts.Tensor[ts.f32]:
    """Compute cross-entropy loss for language modeling."""
    
    # Forward pass
    logits = model_fn(inputs, weights)  # [B, S, vocab_size]
    
    # Reshape for loss computation
    logits_flat = logits.reshape(-1, logits.shape[-1])  # [B*S, vocab_size]
    targets_flat = targets.reshape(-1)  # [B*S]
    
    # Cross-entropy loss
    loss = ts.nn.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
    
    return loss

@ts.function
def training_step(
    model_fn: callable,
    batch: Dict[str, ts.Tensor],
    weights: Dict[str, ts.Tensor],
    optimizer_state: Dict[str, ts.Tensor],
    config: Dict[str, Any]
) -> tuple[ts.Tensor, Dict[str, ts.Tensor], Dict[str, ts.Tensor]]:
    """Single training step with gradient computation."""
    
    # Forward pass and loss computation
    loss = compute_loss(model_fn, batch["input_ids"], batch["labels"], weights)
    
    # Backward pass
    gradients = ts.grad(loss, weights)
    
    # Gradient clipping
    if config.get("gradient_clip_norm", 0) > 0:
        gradients = clip_gradients(gradients, config["gradient_clip_norm"])
    
    # Optimizer step
    updated_weights, updated_optimizer_state = optimizer_step(
        weights, gradients, optimizer_state, config
    )
    
    return loss, updated_weights, updated_optimizer_state

def create_training_config():
    """Create training configuration."""
    return {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "warmup_steps": 1000,
        "total_steps": 100000,
        "batch_size": 32,
        "sequence_length": 2048,
        "mixed_precision": True,
        "gradient_checkpointing": True
    }

# Example single GPU training loop
def train_single_gpu():
    """Complete single GPU training example."""
    
    # Initialize model and data
    config = create_training_config()
    model_weights = initialize_model_weights(config)
    optimizer_state = initialize_optimizer_state(model_weights, config)
    dataloader = create_dataloader(config)
    
    # JIT compile for performance
    compiled_step = ts.jit(training_step)
    
    # Training loop
    for step, batch in enumerate(dataloader):
        if step >= config["total_steps"]:
            break
        
        # Training step
        loss, model_weights, optimizer_state = compiled_step(
            model_fn=transformer_model,  # Your model function
            batch=batch,
            weights=model_weights,
            optimizer_state=optimizer_state,
            config=config
        )
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Checkpointing
        if step % 5000 == 0:
            save_checkpoint(model_weights, optimizer_state, step)
    
    return model_weights
```

### Mixed Precision Training

```python
@ts.function
def mixed_precision_step(
    model_fn: callable,
    batch: Dict[str, ts.Tensor],
    weights: Dict[str, ts.Tensor],
    optimizer_state: Dict[str, ts.Tensor]
) -> tuple[ts.Tensor, Dict[str, ts.Tensor]]:
    """Training step with automatic mixed precision."""
    
    # Enable automatic mixed precision
    with ts.autocast(dtype=ts.bf16, enabled=True):
        # Forward pass in reduced precision
        logits = model_fn(batch["input_ids"], weights)
        loss = ts.nn.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"].reshape(-1)
        )
    
    # Scale loss to prevent gradient underflow
    scaled_loss = loss * optimizer_state["loss_scale"]
    
    # Backward pass (gradients computed in FP32)
    gradients = ts.grad(scaled_loss, weights)
    
    # Unscale gradients
    gradients = {k: v / optimizer_state["loss_scale"] for k, v in gradients.items()}
    
    # Check for overflow
    grad_norm = compute_gradient_norm(gradients)
    
    if ts.isfinite(grad_norm):
        # Update weights
        updated_weights = apply_optimizer_update(weights, gradients, optimizer_state)
        # Gradually increase loss scale
        optimizer_state["loss_scale"] = ts.minimum(
            optimizer_state["loss_scale"] * 1.0001,
            ts.tensor(65536.0)  # Max loss scale
        )
    else:
        # Skip update and reduce loss scale
        updated_weights = weights
        optimizer_state["loss_scale"] *= 0.5
        print(f"Gradient overflow detected, reducing loss scale to {optimizer_state['loss_scale']}")
    
    return loss, updated_weights
```

## Data Parallel Training

### Multi-GPU Data Parallelism

```python
@ts.function
@ts.distribute(strategy="data_parallel")
def data_parallel_step(
    model_fn: callable,
    batch: Dict[str, ts.Tensor],
    weights: Dict[str, ts.Tensor],
    config: Dict[str, Any]
) -> tuple[ts.Tensor, Dict[str, ts.Tensor]]:
    """Data parallel training step with automatic gradient synchronization."""
    
    # Each GPU processes a portion of the batch
    # batch is automatically sharded across GPUs
    
    # Forward pass
    logits = model_fn(batch["input_ids"], weights)
    loss = ts.nn.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1)
    )
    
    # Backward pass with automatic gradient allreduce
    gradients = ts.grad(loss, weights)
    
    # Gradients are automatically averaged across all GPUs
    # Update weights with averaged gradients
    updated_weights = adam_optimizer(weights, gradients, config)
    
    return loss, updated_weights

def setup_data_parallel_training():
    """Setup multi-GPU data parallel training."""
    
    # Automatically detect available GPUs
    mesh = ts.mesh.auto()
    print(f"Detected {mesh.size} GPUs: {mesh.device_ids}")
    
    # Configure for data parallelism
    config = create_training_config()
    config["global_batch_size"] = config["batch_size"] * mesh.size
    config["batch_size_per_gpu"] = config["batch_size"]
    
    # Initialize model on all GPUs
    model_weights = initialize_model_weights(config)
    optimizer_state = initialize_optimizer_state(model_weights, config)
    
    # Create distributed dataloader
    dataloader = create_distributed_dataloader(config, mesh)
    
    return mesh, model_weights, optimizer_state, dataloader

# Example 8-GPU data parallel training
def train_data_parallel():
    """8-GPU data parallel training example."""
    
    mesh, weights, optimizer_state, dataloader = setup_data_parallel_training()
    
    # Compile distributed training step
    compiled_step = ts.jit(data_parallel_step)
    
    for step, batch in enumerate(dataloader):
        # Distributed training step
        loss, weights = compiled_step(
            model_fn=transformer_model,
            batch=batch,
            weights=weights,
            config=config
        )
        
        # Loss is automatically averaged across GPUs
        if step % 100 == 0:
            print(f"Step {step}: Global Loss = {loss.item():.4f}")
```

### Advanced Data Parallel Features

```python
@ts.function
def gradient_accumulation_step(
    model_fn: callable,
    micro_batches: list[Dict[str, ts.Tensor]],
    weights: Dict[str, ts.Tensor],
    accumulation_steps: int
) -> tuple[ts.Tensor, Dict[str, ts.Tensor]]:
    """Gradient accumulation for larger effective batch sizes."""
    
    accumulated_gradients = None
    total_loss = 0.0
    
    for i, micro_batch in enumerate(micro_batches):
        # Forward pass
        logits = model_fn(micro_batch["input_ids"], weights)
        loss = ts.nn.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            micro_batch["labels"].reshape(-1)
        )
        
        # Scale loss by accumulation steps
        scaled_loss = loss / accumulation_steps
        total_loss += loss
        
        # Backward pass
        micro_gradients = ts.grad(scaled_loss, weights)
        
        # Accumulate gradients
        if accumulated_gradients is None:
            accumulated_gradients = micro_gradients
        else:
            accumulated_gradients = {
                k: accumulated_gradients[k] + micro_gradients[k]
                for k in accumulated_gradients.keys()
            }
    
    return total_loss / len(micro_batches), accumulated_gradients

@ts.function
@ts.distribute(strategy="zero_redundancy_optimizer")
def zero_optimizer_step(
    weights: Dict[str, ts.Tensor],
    gradients: Dict[str, ts.Tensor],
    optimizer_state: Dict[str, ts.Tensor]
) -> Dict[str, ts.Tensor]:
    """ZeRO optimizer for memory-efficient data parallelism."""
    
    # ZeRO-2: Partition optimizer states across GPUs
    # Each GPU only stores optimizer state for subset of parameters
    
    updated_weights = {}
    
    for param_name, gradient in gradients.items():
        # Determine which GPU owns this parameter's optimizer state
        owner_rank = hash(param_name) % ts.mesh.current().size
        
        if ts.mesh.current().rank == owner_rank:
            # This GPU owns the optimizer state for this parameter
            param_optimizer_state = optimizer_state[param_name]
            
            # Apply optimizer update
            updated_param = adam_update_single_param(
                weights[param_name], gradient, param_optimizer_state
            )
        else:
            # Other GPUs receive the updated parameter
            updated_param = ts.distributed.recv_from(owner_rank, param_name)
        
        updated_weights[param_name] = updated_param
    
    return updated_weights
```

## Tensor Parallel Training

### Tensor Parallelism Implementation

```python
@ts.function
@ts.distribute(strategy="tensor_parallel", mesh_axes=["tp"])
def tensor_parallel_linear(
    x: ts.Tensor["B", "S", "D_in", ts.bf16],
    weight: ts.Tensor["D_in", "D_out", ts.bf16],
    bias: ts.Tensor["D_out", ts.bf16] = None
) -> ts.Tensor["B", "S", "D_out", ts.bf16]:
    """Tensor parallel linear layer."""
    
    # weight is automatically sharded along output dimension
    # Each GPU computes partial output
    partial_output = ts.matmul(x, weight)
    
    if bias is not None:
        partial_output = partial_output + bias
    
    # All-gather to get complete output across all GPUs
    complete_output = ts.distributed.all_gather(partial_output, axis="tp")
    
    return complete_output

@ts.function
@ts.distribute(strategy="tensor_parallel", mesh_axes=["tp"])
def tensor_parallel_attention(
    x: ts.Tensor["B", "S", "D", ts.bf16],
    w_qkv: ts.Tensor["D", "3*D_head*H", ts.bf16],  # Sharded along head dimension
    w_out: ts.Tensor["D_head*H", "D", ts.bf16],    # Sharded along input dimension
    num_heads: int
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Tensor parallel multi-head attention."""
    
    B, S, D = x.shape
    local_heads = num_heads // ts.mesh.current().size  # Heads per GPU
    head_dim = D // num_heads
    
    # QKV projection (column-parallel)
    # Each GPU computes subset of attention heads
    local_qkv = ts.matmul(x, w_qkv)  # [B, S, 3*D_head*local_heads]
    
    # Split into Q, K, V
    q, k, v = ts.split(local_qkv, sections=3, dim=-1)
    
    # Reshape for attention
    q = q.reshape(B, S, local_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, S, local_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, S, local_heads, head_dim).transpose(1, 2)
    
    # Flash attention on local heads
    local_attn_out = ts.nn.flash_attention(q, k, v, causal=True)
    local_attn_out = local_attn_out.transpose(1, 2).reshape(B, S, -1)
    
    # Output