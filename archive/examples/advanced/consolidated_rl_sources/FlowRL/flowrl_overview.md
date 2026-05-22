# FlowRL-Tessera Implementation - Document 1: Overview and Architecture

This document introduces the Tessera implementation of FlowRL: Matching Reward Distributions for LLM Reasoning, providing a comprehensive framework for training language models with flow-based reward distribution matching.

## Overview

FlowRL addresses a fundamental challenge in reinforcement learning from human feedback (RLHF): traditional methods like PPO optimize for expected rewards, potentially missing the rich distributional structure of human preferences. Our Tessera implementation provides a scalable, high-performance framework for flow-based reward distribution matching.

### Key Innovations

1. **Flow-Based Distribution Matching**: Match entire reward distributions rather than just expected values
2. **Tessera-Optimized Training**: Leverage Tessera's tile-first programming model for efficient transformer training
3. **Multi-GPU Scaling**: Distributed training across NVL72 and other large-scale systems
4. **Numerical Stability**: Safe mixed-precision training with FP4/FP6/FP8 support
5. **Production Ready**: End-to-end training pipeline with monitoring and checkpointing

## FlowRL Algorithm Fundamentals

### Core Concept

FlowRL models the reward function as a flow that transports probability mass from a reference distribution to the target reward distribution. This enables:

- **Richer Reward Modeling**: Capture full distribution of human preferences
- **Better Exploration**: Maintain diversity through distributional matching
- **Stable Training**: Avoid reward hacking through distribution constraints

### Mathematical Framework

The FlowRL objective minimizes the Wasserstein distance between reward distributions:

```
L_flow = W_2(μ_θ, μ_target)
```

Where:
- `μ_θ` is the learned reward distribution
- `μ_target` is the target distribution from human preferences
- `W_2` is the 2-Wasserstein distance

## Tessera Architecture Design

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FlowRL Training Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │
│  │   Language  │   Reward    │   Flow      │   Distribution      │  │
│  │   Model     │   Model     │   Network   │   Matcher           │  │
│  │   (LLM)     │   (RM)      │   (FN)      │   (DM)              │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│              Tessera Distributed Training Runtime                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐  │
│  │  Flash      │   Tensor    │  Gradient   │   Memory            │  │
│  │  Attention  │  Parallel   │  Sync       │   Management        │  │
│  │  Kernels    │  GEMM       │  (NCCL)     │   (HBM Pooling)     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                     NVIDIA H100/A100 Hardware                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tessera Programming Model Integration

```python
import tessera as ts
from tessera import dist, autodiff, jit

# Define mesh for distributed training
mesh = dist.mesh(
    devices=[f"cuda:{i}" for i in range(72)],  # NVL72 configuration
    axes=("dp", "tp", "pp"),
    shape=(8, 9, 1)  # 8-way DP, 9-way TP, 1-way PP
)

@jit @autodiff
def flowrl_step(
    # Language model parameters
    lm_params: dict,
    # Reward model parameters  
    rm_params: dict,
    # Flow network parameters
    flow_params: dict,
    # Training batch
    batch: dict,
    # Hyperparameters
    config: dict
) -> dict:
    """Single FlowRL training step with Tessera optimization."""
    
    # Forward pass through language model
    lm_outputs = language_model_forward(lm_params, batch["prompts"])
    
    # Compute rewards using reward model
    rewards = reward_model_forward(rm_params, lm_outputs, batch["responses"])
    
    # Flow-based distribution matching
    flow_loss = flow_distribution_matching(
        flow_params, rewards, batch["target_distribution"]
    )
    
    # Combined loss
    total_loss = config.alpha * flow_loss + config.beta * lm_loss(lm_outputs, batch)
    
    return {
        "loss": total_loss,
        "flow_loss": flow_loss,
        "lm_outputs": lm_outputs,
        "rewards": rewards
    }
```

## Core Components

### 1. Language Model (LLM)

```python
@ts.function
def transformer_layer(
    x: ts.Tensor["B", "S", "D", ts.bf16 @ts.accum(ts.f32)],
    attention_weights: ts.Tensor["D", "D", ts.bf16],
    mlp_weights: ts.Tensor["D", "4*D", ts.bf16],
    config: dict
) -> ts.Tensor["B", "S", "D", ts.bf16]:
    """Tessera-optimized transformer layer."""
    
    # Self-attention with Flash Attention
    h = rmsnorm_safe(x)
    attn_out = flash_attention_layer(h, attention_weights, config)
    x = x + attn_out
    
    # MLP with tensor parallelism
    h = rmsnorm_safe(x)
    mlp_out = mlp_layer(h, mlp_weights, config)
    x = x + mlp_out
    
    return x

@ts.distribute(mesh=mesh, strategy="tensor_parallel")
def language_model_forward(
    params: dict,
    input_ids: ts.Tensor["B", "S", ts.int32]
) -> ts.Tensor["B", "S", "V", ts.f32]:
    """Forward pass through the language model."""
    
    # Embedding lookup with TP
    x = embedding_lookup(params["embeddings"], input_ids)
    
    # Transformer layers
    for layer_params in params["layers"]:
        x = transformer_layer(x, layer_params["attention"], 
                            layer_params["mlp"], config)
    
    # Output projection
    logits = output_projection(x, params["output_weights"])
    
    return logits
```

### 2. Reward Model (RM)

```python
@ts.function
def reward_model_forward(
    params: dict,
    responses: ts.Tensor["B", "S", "D", ts.bf16],
    prompts: ts.Tensor["B", "P", "D", ts.bf16]
) -> ts.Tensor["B", ts.f32]:
    """Compute rewards for response-prompt pairs."""
    
    # Concatenate prompt and response
    sequence = ts.concatenate([prompts, responses], axis=1)
    
    # Encode through reward transformer
    hidden = reward_transformer_forward(params["encoder"], sequence)
    
    # Global pooling (mean over sequence)
    pooled = ts.mean(hidden, axis=1)
    
    # Reward head
    rewards = ts.gemm(pooled, params["reward_head"])
    
    return ts.squeeze(rewards, axis=-1)

@ts.kernel
def reward_loss_kernel(
    predicted_rewards: ts.Tile["B", ts.f32],
    preference_labels: ts.Tile["B", ts.int32],
    output_loss: ts.Tile["1", ts.f32]
):
    """Efficient reward model loss computation."""
    
    # Bradley-Terry model loss
    batch_size = ts.tile.size(predicted_rewards, 0)
    loss_sum = ts.tile.zeros(ts.f32)
    
    for i in ts.tile.range(0, batch_size, 2):
        reward_chosen = predicted_rewards[i]
        reward_rejected = predicted_rewards[i + 1]
        
        # Preference probability
        logit_diff = reward_chosen - reward_rejected
        prob_chosen = ts.sigmoid(logit_diff)
        
        # Binary cross-entropy loss
        label = preference_labels[i]  # 1 if chosen, 0 if rejected
        loss_sum += -label * ts.log(prob_chosen) - (1 - label) * ts.log(1 - prob_chosen)
    
    output_loss[0] = loss_sum / (batch_size / 2)
```

### 3. Flow Network (FN)

```python
@ts.function
def flow_network_forward(
    params: dict,
    rewards: ts.Tensor["B", ts.f32],
    context: ts.Tensor["B", "D", ts.bf16]
) -> ts.Tensor["B", ts.f32]:
    """Flow network for reward distribution transformation."""
    
    # Context encoding
    context_encoded = flow_encoder(params["encoder"], context)
    
    # Combine rewards and context
    combined = ts.concatenate([
        ts.expand_dims(rewards, axis=-1),
        context_encoded
    ], axis=-1)
    
    # Flow transformation layers
    flow_output = combined
    for layer_params in params["flow_layers"]:
        flow_output = flow_layer(layer_params, flow_output)
    
    # Output flow field
    flow_field = ts.squeeze(
        ts.gemm(flow_output, params["output_projection"]), 
        axis=-1
    )
    
    return flow_field

@ts.function
def flow_layer(
    params: dict,
    x: ts.Tensor["B", "D", ts.f32]
) -> ts.Tensor["B", "D", ts.f32]:
    """Single flow transformation layer."""
    
    # Normalizing flow with coupling layer
    x_a, x_b = ts.split(x, 2, axis=-1)
    
    # Transform x_b conditioned on x_a
    shift_scale = coupling_network(params["coupling"], x_a)
    shift, scale = ts.split(shift_scale, 2, axis=-1)
    
    x_b_transformed = x_b * ts.exp(scale) + shift
    
    # Recombine
    return ts.concatenate([x_a, x_b_transformed], axis=-1)
```

### 4. Distribution Matcher (DM)

```python
@ts.function
def wasserstein_distance(
    samples_1: ts.Tensor["N", ts.f32],
    samples_2: ts.Tensor["M", ts.f32]
) -> ts.Tensor["1", ts.f32]:
    """Compute 2-Wasserstein distance between sample sets."""
    
    # Sort both sample sets
    sorted_1 = ts.sort(samples_1)
    sorted_2 = ts.sort(samples_2)
    
    # Interpolate to common grid
    n_grid = max(len(samples_1), len(samples_2))
    grid_1 = ts.interpolate(sorted_1, n_grid)
    grid_2 = ts.interpolate(sorted_2, n_grid)
    
    # L2 distance between CDFs
    diff_squared = ts.square(grid_1 - grid_2)
    wasserstein_dist = ts.sqrt(ts.mean(diff_squared))
    
    return wasserstein_dist

@ts.function
def flow_distribution_matching_loss(
    flow_params: dict,
    current_rewards: ts.Tensor["B", ts.f32],
    target_samples: ts.Tensor["T", ts.f32],
    context: ts.Tensor["B", "D", ts.bf16]
) -> ts.Tensor["1", ts.f32]:
    """FlowRL distribution matching loss."""
    
    # Apply flow transformation to current rewards
    transformed_rewards = flow_network_forward(
        flow_params, current_rewards, context
    )
    
    # Compute Wasserstein distance to target distribution
    flow_loss = wasserstein_distance(transformed_rewards, target_samples)
    
    # Add flow regularization
    flow_reg = flow_regularization(flow_params, current_rewards, context)
    
    return flow_loss + 0.1 * flow_reg

@ts.kernel
def optimal_transport_kernel(
    source_samples: ts.Tile["N", ts.f32],
    target_samples: ts.Tile["M", ts.f32],
    transport_plan: ts.Tile["N", "M", ts.f32],
    cost_matrix: ts.Tile["N", "M", ts.f32]
):
    """Efficient optimal transport computation using Sinkhorn algorithm."""
    
    # Initialize transport plan
    n_source = ts.tile.size(source_samples, 0)
    n_target = ts.tile.size(target_samples, 0)
    
    # Compute cost matrix (L2 distances)
    for i in ts.tile.range(n_source):
        for j in ts.tile.range(n_target):
            cost_matrix[i, j] = ts.square(source_samples[i] - target_samples[j])
    
    # Sinkhorn iterations
    epsilon = 0.01  # Regularization parameter
    max_iters = 100
    
    # Initialize dual variables
    u = ts.tile.ones(ts.f32, n_source)
    v = ts.tile.ones(ts.f32, n_target)
    
    for iter in ts.tile.range(max_iters):
        # Update u
        for i in ts.tile.range(n_source):
            sum_v = ts.tile.zeros(ts.f32)
            for j in ts.tile.range(n_target):
                sum_v += v[j] * ts.exp(-cost_matrix[i, j] / epsilon)
            u[i] = 1.0 / sum_v
        
        # Update v
        for j in ts.tile.range(n_target):
            sum_u = ts.tile.zeros(ts.f32)
            for i in ts.tile.range(n_source):
                sum_u += u[i] * ts.exp(-cost_matrix[i, j] / epsilon)
            v[j] = 1.0 / sum_u
    
    # Compute final transport plan
    for i in ts.tile.range(n_source):
        for j in ts.tile.range(n_target):
            transport_plan[i, j] = u[i] * ts.exp(-cost_matrix[i, j] / epsilon) * v[j]
```

## Training Pipeline Architecture

### Distributed Training Setup

```python
class FlowRLTrainer:
    """Main FlowRL training coordinator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.mesh = self._setup_mesh()
        self.models = self._initialize_models()
        self.optimizers = self._setup_optimizers()
        
    def _setup_mesh(self):
        """Configure distributed mesh for training."""
        total_gpus = self.config.total_gpus
        
        # Automatic mesh configuration based on model size
        if self.config.model_size == "7B":
            dp, tp, pp = 4, 2, 1
        elif self.config.model_size == "70B":
            dp, tp, pp = 2, 8, 4
        elif self.config.model_size == "405B":
            dp, tp, pp = 1, 18, 4
        
        return dist.mesh(
            devices=[f"cuda:{i}" for i in range(total_gpus)],
            axes=("dp", "tp", "pp"),
            shape=(dp, tp, pp)
        )
    
    @ts.jit(capture_graph=True)
    def training_step(self, batch: dict) -> dict:
        """Single training step with graph capture for performance."""
        
        # Forward pass
        outputs = flowrl_step(
            self.models["language_model"],
            self.models["reward_model"],
            self.models["flow_network"],
            batch,
            self.config
        )
        
        # Backward pass
        grads = ts.grad(lambda: outputs["loss"])()
        
        # Optimizer step
        self.optimizers["language_model"].step(grads["language_model"])
        self.optimizers["reward_model"].step(grads["reward_model"])
        self.optimizers["flow_network"].step(grads["flow_network"])
        
        return outputs
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        for batch in dataloader:
            # Prepare batch for distributed training
            batch = self._prepare_batch(batch)
            
            # Training step
            outputs = self.training_step(batch)
            
            # Logging and monitoring
            self._log_metrics(outputs)
            
            # Checkpointing
            if self._should_checkpoint():
                self._save_checkpoint()

    def _prepare_batch(self, batch: dict) -> dict:
        """Prepare batch for distributed training."""
        return {
            "prompts": ts.from_numpy(batch["prompts"]).to_mesh(self.mesh),
            "responses": ts.from_numpy(batch["responses"]).to_mesh(self.mesh),
            "target_distribution": ts.from_numpy(batch["target_distribution"]).to_mesh(self.mesh),
            "preference_labels": ts.from_numpy(batch["preference_labels"]).to_mesh(self.mesh)
        }
```

## Performance Optimizations

### Memory-Efficient Training

```python
@ts.checkpoint
def memory_efficient_forward(
    params: dict,
    inputs: ts.Tensor,
    config: dict
) -> ts.Tensor:
    """Memory-efficient forward pass with activation checkpointing."""
    
    # Checkpoint every N layers to save memory
    checkpoint_every = config.get("checkpoint_every", 4)
    
    x = inputs
    for i, layer_params in enumerate(params["layers"]):
        if i % checkpoint_every == 0:
            # Checkpoint this layer
            x = ts.checkpoint(transformer_layer)(x, layer_params, config)
        else:
            x = transformer_layer(x, layer_params, config)
    
    return x

@ts.function
def gradient_accumulation_step(
    accumulated_grads: dict,
    current_grads: dict,
    accumulation_steps: int
) -> dict:
    """Accumulate gradients across microbatches."""
    
    for param_name in accumulated_grads:
        accumulated_grads[param_name] = (
            accumulated_grads[param_name] + 
            current_grads[param_name] / accumulation_steps
        )
    
    return accumulated_grads
```

### Numerical Stability

```python
@ts.function
def safe_log_softmax(
    logits: ts.Tensor["B", "V", ts.f32]
) -> ts.Tensor["B", "V", ts.f32]:
    """Numerically stable log softmax."""
    
    # Subtract max for numerical stability
    max_logits = ts.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    
    # Compute log softmax
    exp_shifted = ts.exp(shifted_logits)
    sum_exp = ts.sum(exp_shifted, axis=-1, keepdims=True)
    log_sum_exp = ts.log(sum_exp)
    
    return shifted_logits - log_sum_exp

@ts.function
def stable_reward_computation(
    logits: ts.Tensor["B", "S", "V", ts.f32],
    response_tokens: ts.Tensor["B", "S", ts.int32]
) -> ts.Tensor["B", ts.f32]:
    """Stable reward computation with mixed precision."""
    
    # Convert to BF16 for computation, accumulate in FP32
    logits_bf16 = ts.cast(logits, ts.bf16)
    log_probs = safe_log_softmax(ts.cast(logits_bf16, ts.f32))
    
    # Gather log probabilities for response tokens
    response_log_probs = ts.gather(log_probs, response_tokens, axis=-1)
    
    # Sum log probabilities (in FP32 for stability)
    sequence_log_prob = ts.sum(response_log_probs, axis=-1)
    
    return sequence_log_prob
```

## Configuration and Hyperparameters

```python
DEFAULT_CONFIG = {
    # Model architecture
    "model_size": "7B",
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": 128,
    
    # Training configuration
    "batch_size": 64,
    "sequence_length": 2048,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    
    # FlowRL specific
    "flow_layers": 4,
    "flow_hidden_size": 512,
    "distribution_matching_weight": 1.0,
    "flow_regularization_weight": 0.1,
    "reward_model_weight": 0.5,
    
    # Distributed training
    "total_gpus": 8,
    "tensor_parallel_size": 2,
    "data_parallel_size": 4,
    "pipeline_parallel_size": 1,
    
    # Optimization
    "mixed_precision": "bf16",
    "gradient_accumulation_steps": 4,
    "checkpoint_every": 4,
    "use_flash_attention": True,
    
    # Numerical stability
    "epsilon": 1e-8,
    "reward_clip_value": 10.0,
    "flow_clip_value": 5.0
}
```

## Next Steps

This document provides the foundational architecture for FlowRL in Tessera. The following documents will cover:

- **Document 2**: Detailed kernel implementations and optimizations
- **Document 3**: Training pipeline and distributed execution
- **Document 4**: Evaluation metrics and experimental results
- **Document 5**: Production deployment and scaling

The complete implementation leverages Tessera's strengths in:
- High-performance kernel generation for attention and matrix operations
- Distributed training across large GPU clusters
- Mixed precision training with numerical stability
- Memory-efficient execution with activation checkpointing
- Production-ready deployment with monitoring and checkpointing

## Benefits Over Traditional RLHF

1. **Richer Reward Modeling**: Captures full distribution of human preferences
2. **Better Sample Efficiency**: More efficient exploration through distributional matching
3. **Reduced Reward Hacking**: Distribution constraints prevent exploitation
4. **Scalable Implementation**: Tessera's optimizations enable large-scale training
5. **Numerical Stability**: Safe mixed-precision training prevents gradient explosions