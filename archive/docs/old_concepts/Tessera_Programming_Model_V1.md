# Tessera Programming Model: Detailed Documentation

## Chapter 1: Core Philosophy & Mental Model

### 1.1 The Tessera Way of Thinking

Tessera fundamentally changes how we think about deep learning computation. Instead of viewing models as collections of layers that process data, Tessera treats computation as algebraic transformations with guaranteed properties.

```python
# Traditional thinking: Imperative layer stacking
def traditional_model(x):
    x = linear_layer_1(x)  # What device? What precision? What if NaN?
    x = relu(x)            # Is this fused? Should it be?
    x = linear_layer_2(x)  # How does this parallelize?
    return x


# Tessera thinking: Declarative transformation with contracts
@tessera.function
def tessera_model(x: Tensor["B", "D_in"]) -> Tensor["B", "D_out"]:
    """A pure function with guaranteed properties:
    - Shape correctness verified at compile time
    - Numerical stability enforced
    - Automatic optimization applied
    - Parallelization inferred
    """
    return linear(relu(linear(x, D_hidden)), D_out)
```

###    1.2 The Principle of Least Surprise

Every Tessera operation behaves predictably:

```python
# Numerical behavior is explicit
x: Tensor[B, D, bf16 @accum(f32)]  # BF16 storage, FP32 accumulation

# Data location is explicit
with device("gpu:0"), prefetch(weights, "L2_cache"):
    y = matmul(x, weights)

# Parallelism is explicit
with mesh.axis("data"):
    loss = compute_loss(predictions, labels)
1.3 Progressive Disclosure of Complexity
Tessera follows a three-tier complexity model:
python# Tier 1: Beginner - Everything just works
model = tessera.models.GPT2()
output = model(input)

# Tier 2: Intermediate - Control what matters
model = tessera.models.GPT2(
    precision="mixed",
    attention="flash2",
    compile_mode="optimize"
)

# Tier 3: Expert - Control everything
@tessera.custom
def expert_model(x):
    with tessera.schedule(tile_size=128, warps=8):
        with tessera.numerics(accumulate=tf32, round="stochastic"):
            return custom_kernel(x)
```

### 1.4 Composition Over Configuration

Instead of configuration files, Tessera uses compositional programming:

``` python
# Not this:
config = {
    "model": {
        "type": "transformer",
        "layers": 12,
        "optimizer": {
            "type": "adam",
            "lr": 1e-4
        }
    }
}

# But this:
model = compose(
    transformer(layers=12),
    optimize_with(adam(lr=1e-4)),
    distribute_across(mesh),
    checkpoint_every(1000)
)
```

## Chapter 2: Type System Deep Dive

### 2.1 Shape-Aware Types

Tessera's type system tracks tensor shapes through every operation, catching errors at compile time that would traditionally fail at runtime.
``` python
# Shape variables (like generics in other languages)
def attention[B, S, D, H](
    query: Tensor[B, H, S, D],
    key: Tensor[B, H, S, D],
    value: Tensor[B, H, S, D]
) -> Tensor[B, H, S, D]:
    """
    B = batch size (can be dynamic)
    S = sequence length (can be dynamic)
    D = dimension per head (must be static)
    H = number of heads (must be static)
    """
    # Compiler knows the output shape WITHOUT running the code
    scores = query @ key.transpose(-2, -1)  # [B, H, S, S]
    weights = softmax(scores / sqrt(D))      # [B, H, S, S]
    return weights @ value                    # [B, H, S, D]
```
 Broadcasting Rules
```python
# Tessera enforces NumPy broadcasting semantics at compile time
def broadcast_example(
    x: Tensor["B", "S", "D"],
    y: Tensor["D"],           # Will broadcast to [B, S, D]
    z: Tensor["1", "S", "1"]  # Will broadcast to [B, S, D]
) -> Tensor["B", "S", "D"]:
    return x + y + z  # All broadcasts verified at compile time

# Compile-time error example:
def invalid_broadcast(
    x: Tensor["B", "S", "D"],
    y: Tensor["B", "K", "D"]  # K != S
) -> Tensor["B", "S", "D"]:
    return x + y  # ERROR: Cannot broadcast S with K
    # Compile error: Shape mismatch in dimension 1: S vs K
```
Dynamic vs Static Dimensions

```python
# Static dimensions (known at compile time)
def static_model(x: Tensor["?", 768]) -> Tensor["?", 10]:
    """? means dynamic batch size, 768 and 10 are static"""
    # Compiler can optimize knowing hidden size is always 768
    return linear(x, weight_768x10)

# Dynamic dimensions with constraints
def dynamic_model[D: int](
    x: Tensor["?", D],
    constraint: D % 8 == 0  # D must be divisible by 8
) -> Tensor["?", D]:
    """D is dynamic but constrained"""
    # Allows tile optimizations knowing alignment
    return optimized_operation(x)

# Runtime shape validation
def runtime_checked(x: Tensor) -> Tensor:
    # Shape is fully dynamic, checked at runtime
    assert x.shape[-1] == 768, "Last dim must be 768"
    return process(x)
```
### 2.2 Numerical Types and Policies

Tessera separates storage format from computation format:
``` python
@dataclass
class NumericalPolicy:
    """Complete specification of numerical behavior"""
    storage: DType          # How tensor is stored in memory
    compute: DType          # How operations are performed
    accumulate: DType       # How reductions accumulate
    rounding: RoundingMode  # How to round after ops
    saturate: bool          # Clamp to dtype range?
    denorm_mode: str        # "flush" or "preserve"
    
# Applied to tensors
x: Tensor[B, D, 
    Policy(
        storage=fp8_e4m3,
        compute=bf16,
        accumulate=f32,
        rounding="stochastic",
        saturate=True
    )
]

# Hierarchical policies
@tessera.numerics.policy("training")
class TrainingPolicy(NumericalPolicy):
    """Standard mixed precision training"""
    storage = bf16
    compute = bf16
    accumulate = f32
    rounding = "nearest"
    
@tessera.numerics.policy("inference") 
class InferencePolicy(NumericalPolicy):
    """Optimized inference"""
    storage = fp8_e4m3
    compute = fp8_e4m3
    accumulate = bf16
    rounding = "stochastic"
    saturate = True

# Use policies
with tessera.numerics(TrainingPolicy):
    loss = train_step(batch)
    
with tessera.numerics(InferencePolicy):
    output = model.generate(prompt)
2.3 Effect Types
Effects make side effects explicit and controllable:
python# Pure functions (no effects)
@tessera.pure
def add(x: Tensor, y: Tensor) -> Tensor:
    return x + y  # Can be called anywhere, anytime

# Functions with effects
@tessera.effects(["random", "io"])
def dropout(x: Tensor, p: float = 0.5) -> Tensor:
    """Has 'random' effect - results vary per call"""
    mask = tessera.random.bernoulli(p, x.shape)
    return x * mask

@tessera.effects(["memory"])
def prefetch_weights(weights: Tensor) -> None:
    """Has 'memory' effect - affects cache state"""
    tessera.memory.prefetch_to_l2(weights)

# Effect composition
@tessera.effects(["all"])
def training_step(batch):
    """Combines all effects"""
    with tessera.effects.suspend("io"):  # Temporarily disable IO
        # No logging allowed here
        loss = compute_loss(batch)
    
    tessera.log(loss)  # IO effect allowed again
    return loss

# Effect inference
def composed_function(x):
    y = add(x, x)        # Pure, no effects
    z = dropout(y)       # Infers "random" effect
    return z             # Function has "random" effect
```

## Chapter 3: Functions and Composition

### 3.1 Function Definition and Decoration

```python
# Basic function definition
@tessera.function
def layer_norm(x: Tensor["...", "D"]) -> Tensor["...", "D"]:
    """... means any number of leading dimensions"""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-5)

# Function with multiple decorators (order matters!)
@tessera.compile          # 4. Finally compile to kernels
@tessera.checkpoint       # 3. Then add checkpointing  
@tessera.distribute       # 2. Then distribute
@tessera.function         # 1. First make it a Tessera function
def transformer_layer(x):
    return attention(x) + mlp(x)

# Conditional compilation
@tessera.function
@tessera.specialize_for("A100", "H100")  # Custom kernels per GPU
def optimized_attention(x: Tensor) -> Tensor:
    if tessera.current_device() == "A100":
        return flash_attention_v1(x)
    else:  # H100
        return flash_attention_v2(x)
```

### 3.2 Function Composition Patterns

```python
# Function composition as a first-class concept
from tessera.functional import compose, pipe, curry

# Mathematical composition: (f ∘ g)(x) = f(g(x))
normalize_and_scale = compose(
    scale(factor=2.0),
    normalize(method="layer")
)

# Pipeline composition (reads left-to-right)
process = pipe(
    load_data,
    preprocess,
    augment,
    normalize,
    batch
)

# Partial application
attention_128 = curry(attention, dim=128, heads=8)
# Now can call: attention_128(query, key, value)

# Monadic composition for effects
@tessera.maybe
def safe_divide(x: Tensor, y: Tensor) -> Maybe[Tensor]:
    if (y == 0).any():
        return Nothing()
    return Just(x / y)

result = (
    safe_divide(x, y)
    .map(lambda z: z + 1)
    .unwrap_or(default_value)
)
```

### 3.3 Higher-Order Functions

```python 
# Map over tensor dimensions
@tessera.vmap(in_dims=0, out_dims=0)
def per_batch_operation(x: Tensor["D"]) -> Tensor["D"]:
    """Applied to each batch element independently"""
    return complex_operation(x)

# Scan for sequential processing
@tessera.scan
def rnn_cell(carry: Tensor["H"], x: Tensor["D"]) -> Tuple[Tensor["H"], Tensor["D"]]:
    new_carry = activation(carry @ W_h + x @ W_x)
    output = new_carry @ W_o
    return new_carry, output

# Fold for reductions
@tessera.fold
def tree_reduce(x: Tensor, y: Tensor) -> Tensor:
    return x + y  # Automatically uses tree reduction for stability

# Custom control flow
@tessera.while_loop
def iterative_refinement(
    x: Tensor,
    condition: Callable[[Tensor], bool],
    max_iters: int = 100
) -> Tensor:
    @tessera.jit
    def body(x, i):
        return improve(x), i + 1
    
    return tessera.while_loop(
        cond=lambda x, i: condition(x) and i < max_iters,
        body=body,
        init=(x, 0)
    )[0]
```
### 3.4 Function Transformations


```python
# Automatic differentiation
@tessera.function
def loss_fn(params, x, y):
    predictions = model(params, x)
    return mse(predictions, y)

# Get gradients
grad_fn = tessera.grad(loss_fn, argnums=0)  # Gradient w.r.t. params
gradients = grad_fn(params, x, y)

# Higher-order derivatives
hessian_fn = tessera.grad(tessera.grad(loss_fn))

# Value and gradient together
loss_value, gradients = tessera.value_and_grad(loss_fn)(params, x, y)

# Custom VJP (vector-Jacobian product)
@tessera.custom_vjp
def stable_log1p(x):
    return log(1 + x)

def stable_log1p_fwd(x):
    y = log(1 + x)
    return y, x  # Return primal and residuals

def stable_log1p_bwd(x, grad_y):
    return grad_y / (1 + x)  # More stable than automatic

stable_log1p.defvjp(stable_log1p_fwd, stable_log1p_bwd)

# JIT compilation with shape specialization
@tessera.jit(static_argnums=(1,))  # Hidden size is static
def flexible_model(x: Tensor["?", "?"], hidden_size: int):
    return process(x, hidden_size)

# Checkpointing for memory efficiency
@tessera.checkpoint(policy="selective")
def memory_efficient_model(x):
    # Only checkpoint activations at layer boundaries
    for layer in layers:
        x = tessera.checkpoint_here(layer(x))
    return x
```

## Chapter 4: State and Memory Management

### 4.1 Stateful Computations

```python
# Explicit state management
class RNNCell(tessera.Stateful):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.state = tessera.zeros(hidden_size)
    
    @tessera.stateful
    def forward(self, x: Tensor) -> Tensor:
        # State updates are tracked
        self.state = tanh(self.state @ self.W_h + x @ self.W_x)
        return self.state @ self.W_o
    
    def reset(self):
        self.state = tessera.zeros(self.hidden_size)

# Functional state management
@tessera.with_state
def functional_rnn(
    x: Tensor["S", "D"],
    initial_state: Tensor["H"]
) -> Tuple[Tensor["S", "D"], Tensor["H"]]:
    def step(state, x_t):
        new_state = tanh(state @ W_h + x_t @ W_x)
        output = new_state @ W_o
        return new_state, output
    
    final_state, outputs = tessera.scan(step, initial_state, x)
    return outputs, final_state
```
### 4.2 Memory Hierarchies

```python
# Explicit memory placement
@tessera.memory
class MemoryHierarchy:
    """Explicit control over memory placement"""
    
    # On-chip SRAM (fastest, smallest)
    registers: Tensor["128", f32] @ memory("register")
    shared: Tensor["1024", bf16] @ memory("shared")
    
    # Off-chip DRAM (slower, larger)
    global_mem: Tensor["1M", fp8] @ memory("global")
    
    # Host memory (slowest, largest)
    host_mem: Tensor["1G", int8] @ memory("host")
    
    # Persistent storage
    disk: Tensor @ memory("disk")

# Memory movement operations
@tessera.function
def efficient_compute(x: Tensor @ memory("global")):
    # Explicit prefetch to faster memory
    with tessera.memory.prefetch(x, to="shared"):
        # x is now in shared memory for this block
        result = expensive_operation(x)
    
    # Async copy while computing
    future = tessera.memory.async_copy(result, to="global")
    other_work()
    future.wait()
    
    return result

# Memory pooling for efficiency
pool = tessera.memory.Pool(
    size="8GB",
    device="gpu:0",
    growth_factor=1.5
)

with tessera.memory.use_pool(pool):
    # All allocations come from pool
    tensors = [tessera.zeros(1000, 1000) for _ in range(100)]
    # Automatic defragmentation when needed
```
### 4.3 Cache Management

```python
# KV Cache for attention
class KVCache(tessera.Cache):
    def __init__(self, max_batch: int, max_seq: int, dim: int):
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.dim = dim
        
        # Ring buffer for efficient memory use
        self.keys = tessera.RingBuffer(max_batch, max_seq, dim)
        self.values = tessera.RingBuffer(max_batch, max_seq, dim)
        
        # Page table for non-contiguous storage
        self.page_table = tessera.PageTable(
            page_size=256,
            num_pages=max_seq // 256
        )
    
    @tessera.cache.eviction_policy("lru")
    def get(self, position: int) -> Tuple[Tensor, Tensor]:
        page = self.page_table.get_page(position)
        return self.keys[page], self.values[page]
    
    @tessera.cache.write_through
    def update(self, position: int, k: Tensor, v: Tensor):
        page = self.page_table.allocate_page(position)
        self.keys[page] = k
        self.values[page] = v

# Automatic caching of expensive operations
@tessera.memoize(maxsize=1000, typed=True)
def expensive_computation(x: Tensor) -> Tensor:
    # Result cached based on tensor hash
    return very_expensive_operation(x)

# Persistent cache across runs
@tessera.persistent_cache("~/.tessera/cache.db")
def cached_model_forward(x: Tensor, model_version: str) -> Tensor:
    # Cached based on input hash and version
    return model(x)
```

## Chapter 5: Parallelism Primitives

## 5.1 Data Parallelism

```python
# Simple data parallelism
@tessera.data_parallel(devices=["gpu:0", "gpu:1", "gpu:2", "gpu:3"])
def train_step(batch: Tensor["B", "..."]) -> Tensor:
    """Automatically splits batch across devices"""
    # Each device gets B/4 samples
    loss = model(batch)
    # Gradients automatically averaged across devices
    return loss

# Fine-grained control
@tessera.function
def custom_data_parallel(
    batch: Tensor["B", "S", "D"],
    devices: List[Device]
) -> Tensor:
    # Manual sharding
    shards = tessera.shard(batch, axis=0, devices=devices)
    
    # Parallel execution
    outputs = tessera.parallel_map(
        lambda shard, device: model.on(device)(shard),
        shards, devices
    )
    
    # Custom reduction
    return tessera.reduce(outputs, op="mean", axis=0)

# Gradient accumulation for large batches
@tessera.gradient_accumulation(steps=4)
def large_batch_training(batch):
    # Automatically splits batch into 4 micro-batches
    # Accumulates gradients before optimizer step
    return model(batch)
5.2 Model Parallelism
python# Tensor model parallelism
class ParallelLinear(tessera.Module):
    def __init__(self, in_features: int, out_features: int, devices: List):
        # Automatically shard weights across devices
        self.weight = tessera.sharded_parameter(
            shape=(in_features, out_features),
            devices=devices,
            axis=1  # Column-wise sharding
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # Computation automatically distributed
        return tessera.distributed_matmul(x, self.weight)

# Pipeline parallelism
@tessera.pipeline(stages=4, devices=["gpu:0", "gpu:1", "gpu:2", "gpu:3"])
class PipelineModel(tessera.Module):
    def __init__(self):
        # Each stage on different device
        self.stage1 = EmbeddingLayer().on("gpu:0")
        self.stage2 = TransformerBlock().on("gpu:1")
        self.stage3 = TransformerBlock().on("gpu:2")
        self.stage4 = OutputLayer().on("gpu:3")
    
    @tessera.pipeline.schedule("1f1b")  # One forward, one backward
    def forward(self, x):
        # Automatic micro-batching and scheduling
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

# Expert parallelism (MoE)
class MixtureOfExperts(tessera.Module):
    def __init__(self, num_experts: int, devices: List):
        self.experts = [
            Expert().on(devices[i % len(devices)])
            for i in range(num_experts)
        ]
        self.router = Router()
    
    @tessera.expert_parallel
    def forward(self, x: Tensor) -> Tensor:
        # Route tokens to experts
        routing_weights, expert_indices = self.router(x)
        
        # All-to-all communication
        x_dispatched = tessera.all_to_all(
            x, expert_indices, 
            src_axis="batch", dst_axis="expert"
        )
        
        # Parallel expert computation
        expert_outputs = tessera.parallel_apply(
            self.experts, x_dispatched
        )
        
        # Combine results
        return tessera.all_to_all(
            expert_outputs, routing_weights,
            src_axis="expert", dst_axis="batch"
        )
```
### 5.3 Mesh Parallelism

```python
# Define logical mesh
mesh = tessera.Mesh(
    devices=np.array(range(16)).reshape(2, 4, 2),
    axis_names=["data", "model", "pipeline"]
)

@tessera.on_mesh(mesh)
class MeshParallelModel(tessera.Module):
    def __init__(self):
        # Specify how parameters are sharded
        self.embedding = tessera.MeshParameter(
            shape=(vocab_size, dim),
            mesh_axes={"row": "model", "col": None}
        )
        
        self.attention = tessera.MeshParameter(
            shape=(dim, dim),
            mesh_axes={"row": None, "col": "model"}
        )
    
    def forward(self, x: tessera.MeshTensor) -> tessera.MeshTensor:
        # Automatic SPMD execution
        with mesh.axis("data"):
            # Data parallel region
            x = self.embedding(x)
        
        with mesh.axis("model"):
            # Model parallel region
            x = self.attention(x)
        
        with mesh.axis("pipeline"):
            # Pipeline parallel region
            x = self.output(x)
        
        return x

# Collective operations on mesh
@tessera.on_mesh(mesh)
def mesh_reduce(x: MeshTensor) -> MeshTensor:
    # Reduce across specific axes
    x = tessera.mesh_reduce(x, axis="data", op="mean")
    x = tessera.mesh_reduce(x, axis="model", op="sum")
    return x
```
## Chapter 6: Numerical Computing

### 6.1 Precision Management

```python
# Mixed precision context managers
with tessera.precision("mixed"):
    # Automatically uses FP16 compute, FP32 accumulate
    loss = model(batch)

with tessera.precision(compute=bf16, accumulate=f32, store=fp8):
    # Fine-grained control
    output = inference(input)

# Dynamic precision based on gradient magnitude
class AdaptivePrecision(tessera.PrecisionPolicy):
    def select_precision(self, tensor: Tensor, grad_scale: float) -> DType:
        if grad_scale > 1e3:
            return f32  # High gradients need full precision
        elif grad_scale > 1:
            return bf16  # Normal range
        else:
            return fp8  # Small gradients can use lower precision

@tessera.adaptive_precision(AdaptivePrecision())
def adaptive_training(batch):
    return model(batch)

# Quantization-aware training
@tessera.quantization_aware(
    weights=int8,
    activations=int8,
    accumulate=int32
)
def quantized_model(x):
    # Fake quantization during training
    # Real quantization during inference
    return model(x)
```
### 6.2 Numerical Stability

```python
# Numerically stable primitives
from tessera.stable import *

# Stable softmax (avoids overflow)
def stable_softmax(x: Tensor) -> Tensor:
    x_max = x.max(dim=-1, keepdim=True)
    exp_x = exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# Stable log-sum-exp
def stable_logsumexp(x: Tensor) -> Tensor:
    x_max = x.max()
    return x_max + log(exp(x - x_max).sum())

# Stable normalization
@tessera.numerically_stable
def stable_layer_norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    # Uses Welford's algorithm for numerical stability
    mean = tessera.stable.mean(x)
    var = tessera.stable.variance(x, mean)
    return (x - mean) / sqrt(var + eps)

# Kahan summation for high precision
@tessera.kahan_sum
def precise_sum(x: Tensor) -> Tensor:
    """Uses compensation for floating point errors"""
    return sum(x)  # Automatically uses Kahan algorithm

# Stable attention computation
@tessera.stable.attention
def stable_attention(q, k, v, temperature=1.0):
    """
    Numerically stable attention that handles:
    - Large magnitude queries/keys
    - Very long sequences
    - Low precision computation
    """
    # Automatic scaling and stabilization
    return tessera.stable.scaled_dot_product_attention(
        q, k, v, 
        temperature=temperature,
        use_log_space=True  # For extreme cases
    )
```
### 6.3 Deterministic Computation

```python
# Enforce determinism
@tessera.deterministic(seed=42)
def reproducible_training(dataset):
    """
    Guarantees bit-identical results across runs:
    - Fixed reduction order
    - Deterministic GPU algorithms
    - Consistent floating point rounding
    """
    model = create_model()
    
    for batch in dataset:
        with tessera.deterministic.scope():
            # All operations are deterministic
            loss = model(batch)
            
            # Even parallel reductions are ordered
            total_loss = tessera.deterministic.all_reduce(loss)
            
            # Consistent random number generation
            dropout = tessera.deterministic.dropout(x, p=0.1)
    
    return model

# Determinism verification
def verify_determinism(fn, input, num_runs=10):
    """Verify function produces identical outputs"""
    results = []
    for i in range(num_runs):
        with tessera.deterministic(seed=0):
            results.append(fn(input))
    
    # Check all results are identical
    for i in range(1, num_runs):
        assert tessera.allclose(results[0], results[i], rtol=0, atol=0)
    
    print("✓ Function is deterministic")

# Deterministic debugging
@tessera.debug.deterministic
def debug_nondeterminism(x):
    """
    Automatically detects source of nondeterminism:
    - Flags operations that vary across runs
    - Identifies race conditions
    - Points to specific kernels
    """
    return model(x)
```
##Chapter 7: Automatic Differentiation

### 7.1 Basic Autodiff

```python
# Forward mode differentiation
def forward_diff(f, x, v):
    """Compute Jacobian-vector product: ∂f/∂x · v"""
    return tessera.jvp(f, (x,), (v,))[1]

# Reverse mode differentiation (backprop)
def reverse_diff(f, x):
    """Compute gradient ∇f(x)"""
    return tessera.grad(f)(x)

# Higher-order derivatives
def hessian(f, x):
    """Compute Hessian matrix"""
    return tessera.jacfwd(tessera.grad(f))(x)

# Directional derivatives
def directional_derivative(f, x, direction):
    """Derivative in specific direction"""
    _, jvp = tessera.jvp(f, (x,), (direction,))
    return jvp

# Per-example gradients
@tessera.vmap(in_dims=0, out_dims=0)
def per_example_grad(params, x, y):
    """Gradient for each example in batch"""
    def loss(params, x, y):
        pred = model(params, x)
        return mse(pred, y)
    
    return tessera.grad(loss)(params, x, y)
```
### 7.2 Custom Gradients

```python
# Custom VJP (reverse mode)
@tessera.custom_vjp
def custom_relu(x):
    return tessera.maximum(x, 0)

def custom_relu_fwd(x):
    y = tessera.maximum(x, 0)
    return y, (x > 0)  # Save mask for backward

def custom_relu_bwd(mask, grad_y):
    return grad_y * mask

custom_relu.defvjp(custom_relu_fwd, custom_relu_bwd)

# Custom JVP (forward mode)
@tessera.custom_jvp
def stable_div(x, y):
    return x / y

def stable_div_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    z = x / y
    dz = (dx * y - x * dy) / (y * y)
    return z, dz

stable_div.defjvp(stable_div_jvp)

# Stop gradients
def stop_gradient_example(x):
    y = complex_operation(x)
    # Gradient won't flow through y_no_grad
    y_no_grad = tessera.stop_gradient(y)
    return another_operation(y_no_grad)

# Gradient checkpointing
@tessera.checkpoint
def checkpointed_layer(x):
    """Trade compute for memory"""
    # Forward: compute and discard intermediates
    #RetryGContinueEditpython   # Forward: compute and discard intermediates
   # Backward: recompute intermediates as needed
   x = expensive_layer_1(x)
   x = expensive_layer_2(x)
   x = expensive_layer_3(x)
   return x

# Selective checkpointing
@tessera.checkpoint.policy("selective")
def smart_checkpoint(x):
   """Only checkpoint memory-intensive operations"""
   x = small_op(x)  # Not checkpointed
   x = tessera.checkpoint_here(large_op(x))  # Checkpointed
   x = small_op_2(x)  # Not checkpointed
   return x
```

### 7.3 Advanced Autodiff Patterns

```python
# Implicit differentiation
@tessera.implicit_diff
def solve_fixed_point(f, x0, max_iters=100):
    """Differentiate through fixed point iteration"""
    x = x0
    for _ in range(max_iters):
        x_new = f(x)
        if tessera.allclose(x, x_new):
            break
        x = x_new
    return x

# Differentiate through optimization
@tessera.diff_through_opt
def argmin_layer(x, objective):
    """Differentiate through optimization problem"""
    # Solve: y* = argmin_y objective(x, y)
    y_star = tessera.optimize(
        lambda y: objective(x, y),
        init=tessera.zeros_like(x)
    )
    # Gradient flows through optimal solution
    return y_star

# Straight-through estimator
@tessera.straight_through
def quantize(x):
    """Gradient passes through unchanged"""
    # Forward: quantize
    # Backward: identity
    return tessera.round(x)

# Gradient penalty
def gradient_penalty(discriminator, real, fake, lambda_gp=10):
    """WGAN-GP gradient penalty"""
    batch_size = real.shape[0]
    alpha = tessera.random.uniform((batch_size, 1, 1, 1))
    interpolated = alpha * real + (1 - alpha) * fake
    
    # Compute gradient w.r.t. input
    grad = tessera.grad(discriminator, argnums=0)(interpolated)
    grad_norm = tessera.norm(grad, dim=[1, 2, 3])
    
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return penalty
```
## Chapter 8: Kernel Programming Model

###8.1 Tile Abstraction

```python
# Tiles are the fundamental unit of computation
@tessera.kernel
def tile_matmul(
    A: Tile["M", "K", bf16],
    B: Tile["K", "N", bf16],
    C: Tile["M", "N", f32]
):
    """
    Tiles abstract away thread/warp complexity
    M, N, K are tile dimensions (e.g., 64x64)
    """
    # Get tile configuration
    ctx = tile.context()
    
    # Tiles handle:
    # - Thread block assignment
    # - Shared memory allocation
    # - Warp synchronization
    # - Memory coalescing
    
    # Load tiles (automatic vectorization)
    a_tile = tile.load(A, shape=(ctx.M, ctx.K))
    b_tile = tile.load(B, shape=(ctx.K, ctx.N))
    
    # Compute (maps to tensor cores)
    c_tile = tile.mma(a_tile, b_tile)
    
    # Store (automatic coalescing)
    tile.store(C, c_tile)

# Hierarchical tiles
@tessera.kernel
def hierarchical_kernel(X: Tile3D["D1", "D2", "D3"]):
    """Multi-level tiling for cache optimization"""
    # L2 cache tile
    for l2_tile in tile.split(X, level="L2"):
        # L1 cache tile
        for l1_tile in tile.split(l2_tile, level="L1"):
            # Register tile
            for reg_tile in tile.split(l1_tile, level="register"):
                process(reg_tile)
```
### 8.2 Memory Spaces and Movement

```python
@tessera.kernel
def memory_hierarchy_kernel(
    input: Tile @ memory.global,
    output: Tile @ memory.global
):
    """Explicit memory hierarchy management"""
    
    # Shared memory (48KB on A100)
    shared_buffer = tile.alloc_shared(shape=(64, 64), dtype=bf16)
    
    # Registers (256KB per SM)
    register_buffer = tile.alloc_register(shape=(8, 8), dtype=f32)
    
    # Async copy global -> shared (hides latency)
    tile.async_copy(
        src=input,
        dst=shared_buffer,
        size=(64, 64),
        stage=2  # Double buffering
    )
    
    # Wait for async copy
    tile.wait_async(stage=0)
    
    # Load shared -> register (very fast)
    register_buffer = tile.load_shared(shared_buffer)
    
    # Compute in registers
    result = compute(register_buffer)
    
    # Store back through hierarchy
    tile.store_shared(shared_buffer, result)
    tile.async_copy(shared_buffer, output)

# Memory access patterns
@tessera.kernel
def coalesced_access(data: Tile):
    """Optimized memory access patterns"""
    # Coalesced access (all threads access contiguous memory)
    coalesced = tile.load_coalesced(data, threads=32)
    
    # Strided access (less efficient)
    strided = tile.load_strided(data, stride=2)
    
    # Random access (very slow)
    random = tile.load_gather(data, indices)
    
    # Broadcast (single load, shared by all)
    broadcasted = tile.load_broadcast(data[0])

# Bank conflict avoidance
@tessera.kernel
def avoid_bank_conflicts(data: Tile):
    """Shared memory bank conflict prevention"""
    # Automatic padding to avoid conflicts
    shared = tile.alloc_shared(
        shape=(32, 33),  # 33 instead of 32 for padding
        dtype=f32,
        swizzle="xor"  # XOR-based swizzling
    )
```
### 8.3 Warp and Block Primitives

```python
@tessera.kernel
def warp_primitives(x: Tile):
    """Warp-level operations (32 threads)"""
    
    # Warp-level reductions (very fast)
    warp_sum = tile.warp.reduce_sum(x)
    warp_max = tile.warp.reduce_max(x)
    
    # Warp shuffle (register-to-register)
    shuffled = tile.warp.shuffle(x, lane_id=5)
    butterfly = tile.warp.butterfly(x, mask=0xf)
    
    # Warp vote (consensus operations)
    all_positive = tile.warp.all(x > 0)
    any_zero = tile.warp.any(x == 0)
    
    # Warp match (find threads with same value)
    mask = tile.warp.match(x)

@tessera.kernel
def block_primitives(x: Tile):
    """Block-level operations (multiple warps)"""
    
    # Block-level synchronization
    tile.block.sync()
    
    # Block-level reduction
    block_sum = tile.block.reduce_sum(x)
    
    # Cooperative groups
    group = tile.cooperative_group(size=64)
    group_result = group.reduce(x, op="sum")
    
    # Block-wide broadcast
    value = tile.block.broadcast(x, src_thread=0)
```
### 8.4 Tensor Core Programming

```python
@tessera.kernel
def tensor_core_kernel(
    A: Tile["M", "K", bf16],
    B: Tile["K", "N", bf16],
    C: Tile["M", "N", f32]
):
    """Direct tensor core programming"""
    
    # Fragment types for tensor cores
    a_frag = tile.fragment.a(shape=(16, 16), dtype=bf16)
    b_frag = tile.fragment.b(shape=(16, 16), dtype=bf16)
    c_frag = tile.fragment.c(shape=(16, 16), dtype=f32)
    
    # Load matrix fragments
    tile.load_matrix(a_frag, A, stride=16)
    tile.load_matrix(b_frag, B, stride=16)
    
    # Tensor core MMA (matrix multiply-accumulate)
    tile.mma(c_frag, a_frag, b_frag, c_frag)
    
    # Store result
    tile.store_matrix(C, c_frag, stride=16)

# Mixed precision tensor cores
@tessera.kernel
def mixed_precision_mma(
    A: Tile[m, k, fp8_e4m3],
    B: Tile[k, n, fp8_e5m2],
    C: Tile[m, n, f32]
):
    """FP8 computation with FP32 accumulation"""
    # Automatic mixed precision handling
    result = tile.mma_mixed(
        A, B,
        accumulate=f32,
        saturate=True,
        rounding="rte"  # Round to nearest even
    )
    tile.store(C, result)
```
## Chapter 9: Scheduling and Autotuning

## 9.1 Schedule Specification

``` python
# Schedule as first-class object
schedule = tessera.Schedule(
    # Tiling parameters
    tile_m=128,
    tile_n=128,
    tile_k=32,
    
    # Thread block configuration
    block_m=256,
    block_n=128,
    warps=8,
    
    # Memory hierarchy
    stages=3,  # Pipeline stages
    vector_width=8,  # Vectorization
    
    # Optimization flags
    unroll_factor=4,
    prefetch_distance=2,
    swizzle_pattern="xor"
)

# Apply schedule to kernel
@tessera.kernel.with_schedule(schedule)
def scheduled_kernel(A, B, C):
    return tile.matmul(A, B, C)

# Schedule composition
base_schedule = tessera.Schedule(tile_m=64, tile_n=64)
specialized = base_schedule.derive(
    tile_k=16,  # Override specific parameters
    warps=4
)

# Schedule algebra
schedule_a = tessera.Schedule(tile_m=128)
schedule_b = tessera.Schedule(tile_n=128)
combined = schedule_a | schedule_b  # Merge schedules
```
### 9.2 Autotuning Framework
```python
# Basic autotuning
@tessera.autotune(
    configs=[
        {"tile_m": 64, "tile_n": 64, "warps": 4},
        {"tile_m": 128, "tile_n": 128, "warps": 8},
        {"tile_m": 256, "tile_n": 64, "warps": 4},
    ],
    metric="latency"
)
def autotuned_kernel(A, B, C):
    return matmul(A, B, C)

# Advanced autotuning with search space
@tessera.autotune.search_space(
    tile_m=[64, 128, 256],
    tile_n=[64, 128, 256],
    tile_k=[16, 32, 64],
    warps=[2, 4, 8],
    stages=[2, 3, 4],
    # Constraints
    constraints=[
        lambda c: c.tile_m * c.tile_n <= 65536,  # Shared memory limit
        lambda c: c.warps * 32 <= c.tile_m,      # Enough work per warp
    ]
)
def complex_kernel(A, B, C):
    pass

# ML-guided autotuning
class MLAutotuner(tessera.Autotuner):
    """Uses machine learning to predict good configurations"""
    
    def __init__(self):
        self.model = tessera.ml.PerformancePredictor()
        self.history = []
    
    def suggest_configs(self, kernel, input_shapes):
        # Use ML model to predict performance
        features = self.extract_features(kernel, input_shapes)
        predictions = self.model.predict(features)
        
        # Return top-k predicted configurations
        return self.model.top_k_configs(predictions, k=10)
    
    def update(self, config, performance):
        # Online learning from results
        self.history.append((config, performance))
        if len(self.history) % 100 == 0:
            self.model.retrain(self.history)

# Evolutionary autotuning
@tessera.autotune.evolutionary(
    population_size=50,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.7
)
def evolutionary_tuned(A, B, C):
    """Uses genetic algorithm for optimization"""
    pass
```
### 9.3 Performance Modeling

```python
# Analytical performance model
class RooflineModel(tessera.PerformanceModel):
    """Roofline model for performance prediction"""
    
    def __init__(self, device):
        self.peak_flops = device.peak_flops()  # e.g., 19.5 TFLOPS
        self.peak_bandwidth = device.peak_bandwidth()  # e.g., 1.6 TB/s
    
    def predict(self, kernel):
        flops = kernel.count_flops()
        bytes = kernel.count_memory_bytes()
        
        # Arithmetic intensity
        ai = flops / bytes
        
        # Roofline prediction
        if ai < self.peak_flops / self.peak_bandwidth:
            # Memory bound
            return bytes / self.peak_bandwidth
        else:
            # Compute bound
            return flops / self.peak_flops

# Empirical performance model
class EmpiricalModel(tessera.PerformanceModel):
    """Learn from measurements"""
    
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def measure(self, kernel, config):
        # Run kernel multiple times
        times = []
        for _ in range(100):
            start = tessera.time()
            kernel.run(config)
            times.append(tessera.time() - start)
        
        # Use median to avoid outliers
        return statistics.median(times)
    
    def predict(self, kernel, config):
        # Interpolate from nearby measurements
        similar = self.find_similar_configs(config)
        if similar:
            return self.interpolate(similar, config)
        else:
            # Fall back to measurement
            return self.measure(kernel, config)
```
## Chapter 10: Distribution Strategies

## 10.1 Mesh Topology

```python
# Define physical topology
devices = tessera.get_devices()  # [gpu:0, gpu:1, ..., gpu:15]

# 2D mesh (4x4)
mesh_2d = tessera.Mesh(
    devices=np.array(devices).reshape(4, 4),
    axis_names=["data", "model"]
)

# 3D mesh (2x4x2)
mesh_3d = tessera.Mesh(
    devices=np.array(devices).reshape(2, 4, 2),
    axis_names=["pipeline", "data", "model"]
)

# Hierarchical mesh (multi-level)
hierarchical_mesh = tessera.HierarchicalMesh({
    "node": 4,      # 4 nodes
    "gpu": 4,       # 4 GPUs per node
    "replica": 1    # No replication
})

# Logical view of physical devices
@tessera.mesh_view(mesh_2d)
def distributed_computation(x: MeshTensor):
    """Work with logical mesh coordinates"""
    # Get current position in mesh
    coord = tessera.mesh_coordinate()  # e.g., (1, 2)
    
    # Neighbor communication
    north = tessera.mesh_shift(x, axis=0, shift=-1)
    south = tessera.mesh_shift(x, axis=0, shift=1)
    east = tessera.mesh_shift(x, axis=1, shift=1)
    west = tessera.mesh_shift(x, axis=1, shift=-1)
    
    # Compute using neighbors
    return (x + north + south + east + west) / 5
```
### 10.2 SPMD Programming

```python
# Single Program Multiple Data
@tessera.spmd
def spmd_program(
    global_input: Tensor["1024", "1024"],
    mesh: Mesh
) -> Tensor["1024", "1024"]:
    """
    Same program runs on all devices
    Each device works on its shard
    """
    # Automatic sharding
    local_input = tessera.shard(global_input, mesh)
    
    # Local computation (same on all devices)
    local_output = local_computation(local_input)
    
    # Collective communication
    global_output = tessera.all_gather(local_output, mesh)
    
    return global_output

# Manual SPMD control
@tessera.manual_spmd
def manual_spmd(x: Tensor, mesh: Mesh):
    # Get device rank
    rank = tessera.get_rank()
    size = tessera.get_world_size()
    
    # Compute local portion
    chunk_size = x.shape[0] // size
    start = rank * chunk_size
    end = start + chunk_size
    
    local_x = x[start:end]
    local_y = process(local_x)
    
    # Manual all-gather
    gathered = tessera.empty((size, chunk_size, *local_y.shape[1:]))
    tessera.all_gather_into(gathered, local_y)
    
    return gathered.reshape(x.shape)
```
### 10.3 Collective Operations

```python
# Basic collectives
@tessera.distributed
def collective_examples(x: Tensor, mesh: Mesh):
    # Reduction operations
    sum_all = tessera.all_reduce(x, op="sum", mesh=mesh)
    max_all = tessera.all_reduce(x, op="max", mesh=mesh)
    
    # Gather operations
    gathered = tessera.all_gather(x, mesh=mesh, axis=0)
    
    # Scatter operations
    scattered = tessera.scatter(x, mesh=mesh, axis=0)
    
    # Broadcast
    broadcasted = tessera.broadcast(x, src=0, mesh=mesh)
    
    # All-to-all (for MoE)
    permuted = tessera.all_to_all(x, mesh=mesh)
    
    # Reduce-scatter (for ZeRO)
    reduced_scattered = tessera.reduce_scatter(x, mesh=mesh)

# Hierarchical collectives
@tessera.hierarchical_collective
def hierarchical_reduce(x: Tensor, mesh: HierarchicalMesh):
    """Optimize for network topology"""
    # First reduce within node (fast NVLink)
    x = tessera.reduce(x, mesh.axis("gpu"), op="sum")
    
    # Then reduce across nodes (slower InfiniBand)
    x = tessera.reduce(x, mesh.axis("node"), op="sum")
    
    return x

# Overlapped collectives
@tessera.overlap_communication
def overlapped_compute(x: Tensor, mesh: Mesh):
    """Hide communication latency"""
    # Start async all-reduce
    handle = tessera.all_reduce_async(x, mesh=mesh)
    
    # Do computation while communicating
    y = expensive_local_computation(x)
    
    # Wait for communication
    x_reduced = handle.wait()
    
    return combine(y, x_reduced)
```
## Chapter 11: Compilation Pipeline

### 11.1 Graph-Level Optimizations

```python
# Graph optimization passes
@tessera.compiler.graph_pass
class OperatorFusion(GraphPass):
    """Fuse compatible operations"""
    
    def run(self, graph: GraphIR) -> GraphIR:
        # Pattern matching
        patterns = [
            # Fuse matmul + bias + activation
            (["matmul", "add", "relu"], fused_linear_relu),
            # Fuse normalization + activation
            (["layernorm", "gelu"], fused_norm_gelu),
        ]
        
        for pattern, replacement in patterns:
            graph = self.replace_pattern(graph, pattern, replacement)
        
        return graph

@tessera.compiler.graph_pass
class ConstantFolding(GraphPass):
    """Evaluate constants at compile time"""
    
    def run(self, graph: GraphIR) -> GraphIR:
        for node in graph.nodes:
            if self.all_inputs_constant(node):
                # Compute at compile time
                result = self.evaluate(node)
                graph.replace_with_constant(node, result)
        
        return graph

# Dead code elimination
@tessera.compiler.graph_pass
class DeadCodeElimination(GraphPass):
    """Remove unused operations"""
    
    def run(self, graph: GraphIR) -> GraphIR:
        used = self.find_used_nodes(graph)
        return graph.keep_only(used)
```
### 11.2 Schedule Generation
```python
# Schedule generation strategies
class ScheduleGenerator:
    """Generate optimized schedules"""
    
    def generate(self, kernel: Kernel, target: Target) -> Schedule:
        # Analyze kernel characteristics
        compute_intensity = kernel.arithmetic_intensity()
        memory_pattern = kernel.memory_access_pattern()
        
        # Choose strategy based on analysis
        if compute_intensity > 10:
            # Compute bound: maximize occupancy
            return self.compute_bound_schedule(kernel, target)
        else:
            # Memory bound: maximize bandwidth
            return self.memory_bound_schedule(kernel, target)
    
    def compute_bound_schedule(self, kernel, target):
        return Schedule(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warps=8,
            stages=2,  # Less staging for compute bound
            vector_width=4
        )
    
    def memory_bound_schedule(self, kernel, target):
        return Schedule(
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warps=4,
            stages=4,  # More staging for memory bound
            vector_width=8  # Wider vectors for bandwidth
        )

# Schedule optimization
@tessera.compiler.schedule_pass
class ScheduleOptimizer:
    """Optimize existing schedules"""
    
    def optimize(self, schedule: Schedule, kernel: Kernel) -> Schedule:
        # Loop transformations
        schedule = self.tile_loops(schedule)
        schedule = self.unroll_loops(schedule)
        schedule = self.vectorize(schedule)
        
        # Memory optimizations
        schedule = self.optimize_data_layout(schedule)
        schedule = self.insert_prefetching(schedule)
        
        # Parallelization
        schedule = self.parallelize(schedule)
        
        return schedule
```
### 11.3 Code Generation

```python
# Target-specific code generation
class PTXCodeGen(tessera.CodeGenerator):
    """Generate PTX for NVIDIA GPUs"""
    
    def generate(self, tile_ir: TileIR) -> str:
        ptx = []
        
        # Generate kernel signature
        ptx.append(self.gen_signature(tile_ir))
        
        # Allocate registers
        regs = self.allocate_registers(tile_ir)
        ptx.append(self.gen_register_decls(regs))
        
        # Generate body
        for op in tile_ir.operations:
            if op.type == "load":
                ptx.append(self.gen_load(op, regs))
            elif op.type == "compute":
                ptx.append(self.gen_compute(op, regs))
            elif op.type == "store":
                ptx.append(self.gen_store(op, regs))
        
        return "\n".join(ptx)
    
    def gen_compute(self, op, regs):
        if op.is_tensor_core:
            # Use mma instruction
            return f"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                   f"{{{regs[op.output]}}}, " \
                   f"{{{regs[op.input_a]}}}, " \
                   f"{{{regs[op.input_b]}}}, " \
                   f"{{{regs[op.input_c]}}};"
        else:
            # Regular instruction
            return self.gen_regular_compute(op, regs)

# Multi-target compilation
@tessera.compiler
class UniversalCompiler:
    """Compile to multiple targets"""
    
    def compile(self, graph: GraphIR, targets: List[Target]) -> Dict[Target, Binary]:
        binaries = {}
        
        for target in targets:
            # Target-specific optimization
            optimized = self.optimize_for_target(graph, target)
            
            # Generate schedule
            schedule = self.generate_schedule(optimized, target)
            
            # Lower to target IR
            if target.type == "cuda":
                target_ir = self.lower_to_ptx(optimized, schedule)
            elif target.type == "rocm":
                target_ir = self.lower_to_gcn(optimized, schedule)
            elif target.type == "cpu":
                target_ir = self.lower_to_llvm(optimized, schedule)
            
            # Generate binary
            binaries[target] = self.codegen(target_ir, target)
        
        return binaries
```
## Chapter 12: Error Handling and Debugging

### 12.1 Compile-Time Error Messages

```python
# Shape error messages
"""
Error: Shape mismatch in function 'transformer_block' at line 42

    Expected: Tensor["B", "S", "D"] where D = 768
    Received: Tensor["B", "S", 512]
    
    41 | def transformer_block(x: Tensor["B", "S", "D"]) -> Tensor["B", "S", "D"]:
    42 |     attention_out = attention(x)  # ← Error here
         |                    ~~~~~~~~~~~~
    43 |     return x + attention_out
    
The attention function expects dimension D=768 but received D=512.

Possible fixes:
1. Ensure input tensor has correct dimensions
2. Add a projection layer: x = linear(x, out_dim=768)
3. Update the model configuration to use D=512

Stack trace:
  File "model.py", line 42, in transformer_block
  File "attention.py", line 15, in attention
    Expected D=768 from config
"""

# Type error messages
"""
Error: Numerical type mismatch in 'mixed_precision_model'

    Cannot implicitly convert from bf16 to fp8_e4m3
    
    28 | x = compute_in_bf16(x)
    29 | y = compute_in_fp8(x)  # ← Error here
         |     ~~~~~~~~~~~~~~~
    
    Tensor 'x' has type bf16 but function expects fp8_e4m3
    
Suggestion: Add explicit cast:
    y = compute_in_fp8(x.to(fp8_e4m3))
    
Or use automatic mixed precision:
    with tessera.precision("mixed"):
        y = compute_in_fp8(x)
"""
```
### 12.2 Runtime Debugging

```python
# Interactive debugger
@tessera.debug.interactive
def debug_model(x):
    """Drop into debugger on error"""
    try:
        return model(x)
    except tessera.NumericalError as e:
        # Automatic breakpoint with context
        tessera.debug.break_here(locals())
        """
        Tessera Debugger
        ----------------
        Numerical instability detected at layer_4.attention
        
        > print(x.stats())
        Tensor stats:
          Shape: (32, 512, 768)
          Mean: 0.0001
          Std: 127.3  ← Very high!
          Min: -6834.2
          Max: 8923.1
          NaN count: 0
          Inf count: 47  ← Problem!
        
        > tessera.debug.trace_back()
        The instability originated at:
          layer_2.mlp.activation -> exp() overflow
          Input to exp: 89.3 (exceeds fp16 max)
        
        > tessera.debug.fix_suggestion()
        Add gradient clipping or use stable_gelu instead of gelu
        """

# Tensor inspection
@tessera.debug.watch("x", "gradients", "activations")
def monitored_training(batch):
    """Monitor tensor values during execution"""
    # Automatically logs statistics for watched tensors
    loss = model(batch)
    
    # Alert on anomalies
    """
    Warning: Gradient explosion detected
      Layer: transformer.layer_3.attention
      Gradient norm: 1834.2 (threshold: 10.0)
      Suggestion: Enable gradient clipping
    """
    
    return loss

# Memory debugging
@tessera.debug.memory
def find_memory_leak():
    """Track all allocations"""
    for i in range(100):
        output = model(input)
        # Automatic memory tracking
    
    tessera.debug.memory_report()
    """
    Memory Report:
      Total allocated: 8.3 GB
      Total freed: 7.9 GB
      Leaked: 400 MB
      
    Leak sources:
      1. model.py:45 - 200 MB (50 allocations not freed)
         Tensor["1024", "1024"] allocated in loop
         Suggestion: Use del or context manager
      
      2. attention.py:23 - 200 MB (KV cache growing)
         Cache not cleared between batches
         Suggestion: Call cache.clear() after each batch
    """
```
### 12.3 Performance Debugging

```python
# Performance profiler
@tessera.profile(level="detailed")
def profile_model(x):
    """Detailed performance analysis"""
    output = model(x)
    
    tessera.profile.report()
    """
    Performance Profile:
    
    Total time: 145.3ms
    
    Breakdown by operation:
      attention: 89.2ms (61.4%)
        - qkv_projection: 12.1ms
        - softmax: 45.3ms  ← Bottleneck!
        - output_projection: 31.8ms
      
      mlp: 34.1ms (23.5%)
        - linear_1: 15.2ms
        - activation: 3.7ms
        - linear_2: 15.2ms
      
      layernorm: 22.0ms (15.1%)
    
    Bottleneck Analysis:
      Softmax is taking 45.3ms due to:
        - Not using flash attention
        - Low arithmetic intensity (1.2 ops/byte)
        - Poor memory access pattern
    
    Optimization suggestions:
      1. Enable flash attention: ~3x speedup
      2. Fuse with other operations: ~1.5x speedup
      3. Use fp8 precision: ~2x speedup
    
    Estimated optimized time: 24.2ms (6x faster)
    """

# Kernel debugging
@tessera.debug.kernel
def debug_custom_kernel():
    """Debug at kernel level"""RetryGContinueEditpython   """Debug at kernel level"""
   with tessera.debug.kernel_trace():
       output = custom_kernel(input)
   
   tessera.debug.kernel_report()
   """
   Kernel Execution Trace:
   
   custom_kernel<<<(256,1,1), (128,1,1)>>>
     
   Resource usage:
     Registers: 96/255 (37.6%)
     Shared memory: 48KB/48KB (100%)  ← Maxed out!
     Occupancy: 25% (limited by shared memory)
   
   Instruction mix:
     Memory: 67.3%  ← Memory bound!
     Compute: 23.1%
     Control: 9.6%
   
   Memory access patterns:
     L1 cache hit rate: 45.2%
     L2 cache hit rate: 78.3%
     Bank conflicts: 1,247 (high!)
     Coalescing efficiency: 67%
   
   Warp efficiency:
     Active warps: 8/32
     Stall reasons:
       - Memory dependency: 45%
       - Execution dependency: 23%
       - Synchronization: 12%
   
   Suggestions:
     1. Reduce shared memory usage to increase occupancy
     2. Use padding to avoid bank conflicts
     3. Improve memory coalescing with different data layout
   """
```
## Chapter 13: Optimization Techniques

### 13.1 Operation Fusion

```python
# Automatic fusion
@tessera.fusion.auto
def unfused_model(x):
    """Automatically fuses compatible operations"""
    x = layernorm(x)      # These three operations
    x = linear(x, 4*dim)  # will be fused into
    x = gelu(x)           # single kernel
    x = linear(x, dim)    
    return x

# Manual fusion specification
@tessera.fusion.manual([
    ["layernorm", "matmul", "add", "gelu"],  # Fuse these
    ["attention", "add"],                     # And these
])
def manually_fused(x):
    return transformer_block(x)

# Custom fusion patterns
@tessera.fusion.pattern
class CustomFusion:
    """Define custom fusion rules"""
    
    @staticmethod
    def should_fuse(op1, op2):
        # Fusion heuristics
        if op1.memory_bound and op2.compute_bound:
            return True  # Good balance
        if op1.output_size == op2.input_size:
            return True  # No intermediate storage
        return False
    
    @staticmethod
    def fuse(ops):
        """Generate fused kernel"""
        return tessera.kernel.generate(
            inputs=ops[0].inputs,
            outputs=ops[-1].outputs,
            body=tessera.kernel.sequence(ops)
        )

# Vertical fusion (producer-consumer)
@tessera.fusion.vertical
def vertical_fusion(x):
    """Fuse producer-consumer chains"""
    # Producer
    y = expensive_compute(x)
    # Consumer (fused with producer)
    z = simple_transform(y)
    return z

# Horizontal fusion (independent ops)
@tessera.fusion.horizontal
def horizontal_fusion(x, y):
    """Fuse independent operations"""
    # These run in parallel in single kernel
    a = operation_1(x)
    b = operation_2(y)
    return a, b
```
### 13.2 Memory Optimization

```python
# Activation checkpointing
@tessera.checkpoint(policy="selective")
def memory_efficient_model(x):
    """Trade compute for memory"""
    # Only checkpoint at strategic points
    x = layer_1(x)
    x = tessera.checkpoint_here(layer_2(x))  # Checkpoint
    x = layer_3(x)
    x = tessera.checkpoint_here(layer_4(x))  # Checkpoint
    return x

# Memory reuse
@tessera.memory.reuse
def reuse_buffers(x):
    """Reuse memory buffers when possible"""
    # Buffer for x can be reused for y
    y = inplace_operation(x)  # Modifies x in-place
    
    # Explicit buffer management
    with tessera.memory.buffer(size="1GB") as buf:
        temp1 = buf.allocate(x.shape)
        compute_into(x, temp1)
        
        # Reuse same memory for temp2
        temp2 = buf.reuse(temp1)
        compute_into(temp1, temp2)
    
    return temp2

# Memory pipelining
@tessera.memory.pipeline(stages=3)
def pipelined_compute(data_stream):
    """Pipeline memory transfers with compute"""
    for i, batch in enumerate(data_stream):
        # Stage 1: Load next batch
        if i < len(data_stream) - 1:
            next_batch = tessera.async_load(data_stream[i + 1])
        
        # Stage 2: Compute current batch
        result = compute(batch)
        
        # Stage 3: Store previous result
        if i > 0:
            tessera.async_store(prev_result)
        
        prev_result = result
    
    return results

# ZeRO optimization
@tessera.zero(stage=3)
def zero_optimizer(model, gradients):
    """
    ZeRO Stage 3: Partition everything
    - Parameters partitioned across devices
    - Gradients partitioned
    - Optimizer states partitioned
    """
    # Automatic partitioning and gathering
    params = tessera.zero.gather_params(model)
    grads = tessera.zero.reduce_scatter_grads(gradients)
    
    # Update only local shard
    local_params = tessera.zero.update_local(params, grads)
    
    # Broadcast updated params
    return tessera.zero.all_gather(local_params)
```
### 13.3 Numerical Optimization
```python
# Mixed precision training
@tessera.mixed_precision(
    compute=bf16,
    accumulate=f32,
    grad_scale=1024
)
def mixed_precision_training(batch):
    """Automatic mixed precision"""
    # Forward pass in bf16
    with tessera.autocast():
        output = model(batch)
        loss = loss_fn(output, target)
    
    # Backward pass with scaling
    scaled_loss = loss * tessera.grad_scale()
    grads = tessera.grad(scaled_loss)
    
    # Unscale gradients
    grads = tessera.unscale_grads(grads)
    
    # Update in fp32
    optimizer.step(grads)
    
    return loss

# Quantization aware training
@tessera.quantization.aware(
    weights=int8,
    activations=int8,
    calibrate=True
)
def quantization_aware_training(x):
    """Train with simulated quantization"""
    # Forward pass with fake quantization
    x_quant = tessera.fake_quantize(x, bits=8)
    output = model(x_quant)
    
    # Backward pass through straight-through estimator
    return output

# Stochastic rounding
@tessera.rounding.stochastic
def stochastic_rounding_training(x):
    """Use stochastic rounding for better convergence"""
    # Automatically applies stochastic rounding
    # to all operations
    return model(x)

# Kahan summation for high precision
@tessera.summation.kahan
def high_precision_reduction(x: Tensor) -> Tensor:
    """Compensated summation for accuracy"""
    # Automatically uses Kahan algorithm
    return x.sum()
```
## Chapter 14: Production Deployment

### 14.1 Model Export

```python
# Export for different targets
class ModelExporter:
    """Export trained models for deployment"""
    
    def export_server(self, model, path):
        """Optimize for server deployment"""
        optimized = tessera.optimize(
            model,
            target="server",
            optimizations=[
                "graph_optimization",
                "kernel_fusion",
                "memory_planning",
                "fp8_quantization"
            ]
        )
        
        # Pre-compile for common batch sizes
        for batch_size in [1, 8, 32, 128]:
            optimized.compile(batch_size=batch_size)
        
        # Save with metadata
        tessera.save(
            optimized,
            path,
            metadata={
                "version": "1.0",
                "requirements": {"memory": "8GB", "compute": "7.0"},
                "performance": {"throughput": "1M tokens/sec"}
            }
        )
    
    def export_edge(self, model, path):
        """Optimize for edge deployment"""
        optimized = tessera.optimize(
            model,
            target="edge",
            constraints={
                "memory": "100MB",
                "power": "5W",
                "latency": "10ms"
            },
            optimizations=[
                "pruning",
                "quantization_int8",
                "layer_fusion",
                "constant_folding"
            ]
        )
        
        tessera.save(optimized, path, format="tflite")
    
    def export_web(self, model, path):
        """Export for browser deployment"""
        optimized = tessera.optimize(
            model,
            target="webgpu",
            optimizations=[
                "graph_simplification",
                "webgpu_kernels",
                "memory_minimization"
            ]
        )
        
        # Generate WASM + WebGPU
        tessera.export_web(
            optimized,
            path,
            split_size="4MB",  # Split for CDN
            compression="brotli"
        )
```
### 14.2 Serving Infrastructure

```python
# Production server
class ProductionServer:
    """High-performance model serving"""
    
    def __init__(self, model_path, config):
        self.model = tessera.load(model_path)
        self.config = config
        
        # Request batching
        self.batcher = tessera.serving.DynamicBatcher(
            max_batch_size=config.max_batch,
            timeout_ms=config.batch_timeout,
            padding_strategy="left"
        )
        
        # Memory management
        self.cache_manager = tessera.serving.CacheManager(
            cache_size=config.cache_size,
            eviction_policy="lru"
        )
        
        # Load balancing
        self.load_balancer = tessera.serving.LoadBalancer(
            replicas=config.num_replicas,
            strategy="least_connections"
        )
    
    async def serve(self, request):
        """Handle incoming requests"""
        # Automatic batching
        batch = await self.batcher.add(request)
        
        # Cache lookup
        cached = self.cache_manager.get(batch.hash())
        if cached:
            return cached
        
        # Load balance across replicas
        replica = self.load_balancer.select()
        
        # Execute with monitoring
        with tessera.monitor.request(request.id):
            result = await replica.infer(batch)
        
        # Cache result
        self.cache_manager.put(batch.hash(), result)
        
        return result

# Continuous batching for LLMs
class ContinuousBatchingServer:
    """Optimized for text generation"""
    
    def __init__(self, model):
        self.model = model
        self.active_sequences = {}
        
    async def generate_stream(self, prompt, max_length=2048):
        """Streaming generation with continuous batching"""
        seq_id = tessera.uuid()
        self.active_sequences[seq_id] = {
            "tokens": tokenize(prompt),
            "cache": KVCache(),
            "position": 0
        }
        
        while self.active_sequences[seq_id]["position"] < max_length:
            # Batch all active sequences
            batch = self.prepare_batch()
            
            # Single forward pass for all sequences
            next_tokens = await self.model.generate_next(batch)
            
            # Update sequences
            for sid, token in zip(batch.seq_ids, next_tokens):
                self.active_sequences[sid]["tokens"].append(token)
                self.active_sequences[sid]["position"] += 1
                
                # Yield token for this sequence
                if sid == seq_id:
                    yield token
                
                # Check stopping condition
                if token == EOS or self.should_stop(sid):
                    del self.active_sequences[sid]
```
### 14.3 Monitoring and Observability

```python
# Production monitoring
class ModelMonitor:
    """Monitor model in production"""
    
    def __init__(self):
        self.metrics = tessera.monitoring.MetricsCollector()
        self.alerting = tessera.monitoring.AlertManager()
        self.logging = tessera.monitoring.Logger()
    
    def monitor_inference(self, model):
        """Track inference metrics"""
        
        @self.metrics.track
        def monitored_inference(input):
            # Automatic metric collection
            output = model(input)
            
            # Custom metrics
            self.metrics.record("input_length", len(input))
            self.metrics.record("output_confidence", output.max())
            
            # Data drift detection
            if self.detect_drift(input):
                self.alerting.trigger("data_drift", severity="warning")
            
            # Performance degradation
            if self.metrics.latency_p99 > SLA_THRESHOLD:
                self.alerting.trigger("sla_violation", severity="critical")
            
            return output
        
        return monitored_inference
    
    def detect_drift(self, input):
        """Detect distribution shift"""
        current_stats = self.compute_statistics(input)
        baseline_stats = self.baseline_statistics
        
        # KL divergence for drift detection
        kl_div = tessera.stats.kl_divergence(
            current_stats, 
            baseline_stats
        )
        
        return kl_div > self.drift_threshold

# A/B testing
class ABTestFramework:
    """A/B test model variants"""
    
    def __init__(self, variants):
        self.variants = variants
        self.metrics = defaultdict(list)
        
    async def route_request(self, request):
        """Route to appropriate variant"""
        # Determine variant (hash-based or random)
        variant = self.select_variant(request.user_id)
        
        # Execute with tracking
        start_time = time.time()
        result = await self.variants[variant].infer(request)
        latency = time.time() - start_time
        
        # Record metrics
        self.metrics[variant].append({
            "latency": latency,
            "user_id": request.user_id,
            "timestamp": time.time()
        })
        
        # Statistical significance testing
        if len(self.metrics[variant]) % 1000 == 0:
            self.analyze_results()
        
        return result
    
    def analyze_results(self):
        """Statistical analysis of A/B test"""
        from scipy import stats
        
        # Compare variants
        for variant_a, variant_b in combinations(self.variants.keys(), 2):
            metrics_a = self.metrics[variant_a]
            metrics_b = self.metrics[variant_b]
            
            # T-test for latency
            t_stat, p_value = stats.ttest_ind(
                [m["latency"] for m in metrics_a],
                [m["latency"] for m in metrics_b]
            )
            
            if p_value < 0.05:
                print(f"Significant difference between {variant_a} and {variant_b}")
                print(f"Mean latency A: {np.mean(metrics_a):.2f}ms")
                print(f"Mean latency B: {np.mean(metrics_b):.2f}ms")
```
## Chapter 15: Advanced Patterns

### 15.1 Meta-Programming
```python
# Code generation
@tessera.meta.generate
def generate_optimized_kernel(config):
    """Generate kernel code from configuration"""
    
    code = tessera.meta.Template("""
    @tessera.kernel
    def generated_kernel_{name}(
        input: Tile[{tile_m}, {tile_k}],
        output: Tile[{tile_m}, {tile_n}]
    ):
        # Generated code
        {body}
    """)
    
    # Generate optimal code based on config
    if config.use_tensor_cores:
        body = generate_tensor_core_code(config)
    else:
        body = generate_simd_code(config)
    
    return code.render(
        name=config.name,
        tile_m=config.tile_m,
        tile_k=config.tile_k,
        tile_n=config.tile_n,
        body=body
    )

# Domain-specific language embedding
@tessera.dsl
class AttentionDSL:
    """DSL for attention variants"""
    
    def flash_attention(self, q, k, v):
        """Flash attention pattern"""
        return self.tiled_attention(
            q, k, v,
            tile_size=64,
            use_online_softmax=True
        )
    
    def sliding_window(self, q, k, v, window=256):
        """Sliding window attention"""
        return self.windowed_attention(
            q, k, v,
            window_size=window,
            overlap=32
        )
    
    def compile(self):
        """Compile DSL to Tessera kernels"""
        return tessera.compile_dsl(self)

# Polymorphic functions
@tessera.polymorphic
def polymorphic_add(x: T, y: T) -> T:
    """Works for any tensor type T"""
    return x + y

# Specialized for specific types
@polymorphic_add.specialize(Tensor[Float32])
def add_f32(x, y):
    return tessera.f32.add(x, y)

@polymorphic_add.specialize(Tensor[Int8])
def add_i8(x, y):
    return tessera.i8.saturating_add(x, y)
```
### 15.2 Advanced Parallelism Patterns

```python
# Work stealing
class WorkStealingScheduler:
    """Dynamic load balancing"""
    
    def __init__(self, num_workers):
        self.workers = [Worker() for _ in range(num_workers)]
        self.queues = [deque() for _ in range(num_workers)]
    
    def schedule(self, tasks):
        # Initial distribution
        for i, task in enumerate(tasks):
            self.queues[i % len(self.workers)].append(task)
        
        # Work stealing loop
        while any(self.queues):
            for worker_id, worker in enumerate(self.workers):
                if not self.queues[worker_id]:
                    # Steal from busiest queue
                    victim = max(range(len(self.queues)), 
                               key=lambda i: len(self.queues[i]))
                    if self.queues[victim]:
                        task = self.queues[victim].pop()
                        self.queues[worker_id].append(task)
                
                if self.queues[worker_id]:
                    task = self.queues[worker_id].popleft()
                    worker.execute(task)

# Nested parallelism
@tessera.parallel.nested
def nested_parallel(data):
    """Parallelism at multiple levels"""
    
    # Outer level: across nodes
    @tessera.parallel(level="node")
    def process_shard(shard):
        
        # Inner level: within node
        @tessera.parallel(level="gpu")
        def process_chunk(chunk):
            
            # Innermost: within GPU
            @tessera.parallel(level="warp")
            def process_element(element):
                return compute(element)
            
            return process_element(chunk)
        
        return process_chunk(shard)
    
    return process_shard(data)

# Speculative execution
@tessera.speculative
def speculative_decode(model, prompt, draft_model):
    """Speculative decoding for faster inference"""
    
    # Generate draft tokens quickly
    draft_tokens = draft_model.generate(
        prompt, 
        num_tokens=4,
        temperature=0
    )
    
    # Verify in parallel with main model
    logits = model(prompt + draft_tokens)
    
    # Accept/reject draft tokens
    accepted = []
    for i, token in enumerate(draft_tokens):
        if tessera.sample(logits[i]) == token:
            accepted.append(token)
        else:
            break
    
    return accepted
```
### 15.3 Research Extensions

```python
# Neural architecture search
@tessera.nas
class NeuralArchitectureSearch:
    """Automated architecture discovery"""
    
    def __init__(self, search_space):
        self.search_space = search_space
        self.controller = tessera.nas.Controller()
        
    def search(self, dataset, epochs=100):
        for epoch in range(epochs):
            # Sample architecture
            arch = self.controller.sample()
            
            # Train and evaluate
            model = self.build_model(arch)
            accuracy = self.train_and_eval(model, dataset)
            
            # Update controller
            self.controller.update(arch, accuracy)
        
        return self.controller.best_architecture()
    
    def build_model(self, arch):
        """Construct model from architecture"""
        layers = []
        for op in arch.operations:
            if op == "conv3x3":
                layers.append(Conv2d(3, 3))
            elif op == "conv5x5":
                layers.append(Conv2d(5, 5))
            elif op == "maxpool":
                layers.append(MaxPool2d())
        
        return tessera.Sequential(layers)

# Differential privacy
@tessera.privacy.differential(epsilon=1.0, delta=1e-5)
def private_training(model, dataset):
    """Training with differential privacy"""
    
    # Gradient clipping for privacy
    max_grad_norm = tessera.privacy.compute_clip_norm(
        epsilon=1.0,
        delta=1e-5,
        dataset_size=len(dataset)
    )
    
    for batch in dataset:
        # Per-example gradients
        grads = tessera.per_example_grad(model, batch)
        
        # Clip gradients
        clipped = tessera.clip_grad_norm(grads, max_grad_norm)
        
        # Add noise for privacy
        noisy_grads = tessera.privacy.add_gaussian_noise(
            clipped,
            sensitivity=max_grad_norm,
            epsilon=1.0
        )
        
        # Update model
        optimizer.step(noisy_grads)

# Federated learning
@tessera.federated
class FederatedLearning:
    """Distributed learning with privacy"""
    
    def train_round(self, clients, global_model):
        client_updates = []
        
        # Local training on each client
        for client in clients:
            local_model = copy(global_model)
            
            # Train on local data
            for batch in client.local_data:
                loss = local_model(batch)
                local_model.backward(loss)
            
            # Compute update
            update = local_model.weights - global_model.weights
            client_updates.append(update)
        
        # Secure aggregation
        aggregated = tessera.federated.secure_aggregate(
            client_updates,
            method="homomorphic"
        )
        
        # Update global model
        global_model.weights += aggregated
        
        return global_model
```

## 
Conclusion
This comprehensive documentation of the Tessera programming model represents a fundamental rethinking of how we build deep learning systems. By treating numerical precision, parallelism, memory management, and correctness as first-class concerns rather than afterthoughts, Tessera enables developers to write models that are:

Correct by construction through compile-time shape checking and type safety
Efficient by default through automatic optimization and autotuning
Portable across hardware through multi-level IR compilation
Production-ready with built-in monitoring, debugging, and deployment tools

The progressive complexity model ensures that beginners can be productive immediately while experts have full control over every aspect of execution. The two-layer architecture separates the concerns of model developers from performance engineers while allowing seamless collaboration.
Most importantly, Tessera is designed not just for today's models but for the next decade of machine learning innovation, with support for probabilistic programming, neural architecture search, federated learning, and whatever comes next.
The future of deep learning deserves a programming model built for it. That's Tessera.

For the latest updates, examples, and community resources, visit tessera.aiRetryClaude can make mistakes. Please double-check responses.