# Tessera Migration Guide Part 1 - Assessment, Planning, and Basic Migration

This comprehensive guide helps developers migrate existing CUDA and PyTorch applications to Tessera, providing step-by-step instructions, code transformation examples, and best practices for a smooth transition.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Pre-Migration Assessment](#pre-migration-assessment)
3. [CUDA to Tessera Migration](#cuda-to-tessera-migration)
4. [PyTorch to Tessera Migration](#pytorch-to-tessera-migration)
5. [Common Migration Patterns](#common-migration-patterns)
6. [Performance Optimization During Migration](#performance-optimization-during-migration)

---

## Migration Overview

### Why Migrate to Tessera?

**Productivity Benefits:**
- **Unified Programming Model**: Single language for kernels, models, and distributed execution
- **Automatic Optimization**: Built-in autotuning and performance optimization
- **Simplified Debugging**: Integrated profiling and debugging tools
- **Reduced Maintenance**: No separate CUDA kernel development and maintenance

**Performance Benefits:**
- **Architecture Adaptation**: Automatic optimization for different GPU generations
- **Advanced Features**: Native support for latest GPU features (WGMMA, TMA, etc.)
- **Memory Efficiency**: Automatic memory layout optimization
- **Distributed Scaling**: Seamless scaling from single GPU to NVL72

**Future-Proofing Benefits:**
- **Multi-Architecture Support**: Single codebase for NVIDIA, AMD, Intel GPUs
- **Automatic Adaptation**: New hardware features automatically utilized
- **Numerical Stability**: Built-in numerical safety and precision management

### Migration Strategy Options

1. **Complete Rewrite**: Full migration to Tessera (recommended for new projects)
2. **Gradual Migration**: Incremental replacement of components
3. **Hybrid Approach**: Tessera for new features, maintain existing CUDA/PyTorch code
4. **Proof of Concept**: Migrate critical kernels first to validate benefits

### Expected Timeline and Effort

| Project Type | Timeline | Effort Level | Expected Gains |
|--------------|----------|--------------|----------------|
| **Research Prototype** | 2-4 weeks | Low | 2-3x dev speed |
| **Single GPU Application** | 1-3 months | Medium | 1.5-2x performance |
| **Multi-GPU Training** | 3-6 months | High | 1.4-2.5x performance |
| **Production System** | 6-12 months | Very High | 1.3-2x performance + maintenance |

---

## Pre-Migration Assessment

### Automated Codebase Analysis

```python
import tessera as ts

# Automated codebase analysis
analyzer = ts.migration.CodebaseAnalyzer()

# Scan existing CUDA/PyTorch code
analysis_report = analyzer.analyze_project("/path/to/your/project")

print("Migration Analysis Report:")
print(f"Total CUDA kernels found: {analysis_report.cuda_kernels}")
print(f"PyTorch modules found: {analysis_report.pytorch_modules}")
print(f"Estimated migration effort: {analysis_report.effort_estimate}")
print(f"Migration complexity: {analysis_report.complexity_level}")
print(f"Potential performance gains: {analysis_report.performance_potential}")

# Detailed recommendations
for recommendation in analysis_report.recommendations:
    print(f"- {recommendation.component}: {recommendation.suggestion}")
```

### Migration Readiness Assessment

```python
class MigrationReadinessAssessment:
    """Comprehensive assessment of migration readiness."""
    
    def __init__(self, project_path):
        self.project_path = project_path
        self.assessment_results = {}
        
    def run_full_assessment(self):
        """Run complete migration readiness assessment."""
        
        # Technical readiness
        self.assessment_results['technical'] = self.assess_technical_readiness()
        
        # Code complexity analysis
        self.assessment_results['complexity'] = self.analyze_code_complexity()
        
        # Performance bottleneck identification
        self.assessment_results['bottlenecks'] = self.identify_bottlenecks()
        
        # Dependencies analysis
        self.assessment_results['dependencies'] = self.analyze_dependencies()
        
        # Generate migration plan
        self.assessment_results['plan'] = self.generate_migration_plan()
        
        return self.assessment_results
    
    def assess_technical_readiness(self):
        """Assess technical prerequisites for migration."""
        
        technical_assessment = {
            'gpu_compatibility': self.check_gpu_compatibility(),
            'software_environment': self.check_software_environment(),
            'testing_infrastructure': self.check_testing_infrastructure(),
            'performance_benchmarks': self.check_performance_benchmarks()
        }
        
        return technical_assessment
    
    def check_gpu_compatibility(self):
        """Check GPU compatibility with Tessera."""
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            gpu_info = []
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                
                # Get compute capability
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
                
                gpu_info.append({
                    'name': name,
                    'compute_capability': compute_capability,
                    'tessera_compatible': float(compute_capability) >= 7.0
                })
            
            return {
                'gpus_found': len(gpu_info),
                'gpu_details': gpu_info,
                'all_compatible': all(gpu['tessera_compatible'] for gpu in gpu_info)
            }
            
        except Exception as e:
            return {
                'error': f"Could not assess GPU compatibility: {str(e)}",
                'recommendation': "Please ensure NVIDIA drivers and pynvml are installed"
            }
    
    def analyze_code_complexity(self):
        """Analyze complexity of existing codebase."""
        
        complexity_metrics = {
            'total_lines_of_code': 0,
            'cuda_kernel_count': 0,
            'pytorch_module_count': 0,
            'custom_autograd_functions': 0,
            'distributed_training_usage': False,
            'mixed_precision_usage': False
        }
        
        # Scan project files
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(('.py', '.cu', '.cpp', '.h')):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path, complexity_metrics)
        
        # Calculate complexity score
        complexity_score = self.calculate_complexity_score(complexity_metrics)
        
        return {
            'metrics': complexity_metrics,
            'complexity_score': complexity_score,
            'estimated_effort': self.estimate_migration_effort(complexity_score)
        }
    
    def calculate_complexity_score(self, metrics):
        """Calculate migration complexity score (0-100)."""
        
        score = 0
        
        # Base complexity from code size
        if metrics['total_lines_of_code'] > 10000:
            score += 30
        elif metrics['total_lines_of_code'] > 5000:
            score += 20
        else:
            score += 10
        
        # CUDA kernel complexity
        score += min(metrics['cuda_kernel_count'] * 5, 25)
        
        # PyTorch module complexity
        score += min(metrics['pytorch_module_count'] * 2, 20)
        
        # Advanced features
        if metrics['custom_autograd_functions'] > 0:
            score += 15
        
        if metrics['distributed_training_usage']:
            score += 10
        
        if metrics['mixed_precision_usage']:
            score += 5
        
        return min(score, 100)
```

### Migration Readiness Checklist

**Technical Readiness:**
- [ ] CUDA Compute Capability 7.0+ (Volta architecture or newer)
- [ ] Python 3.8+ environment
- [ ] Current PyTorch/CUDA versions documented
- [ ] Comprehensive test suite available
- [ ] Performance benchmarks established

**Team Readiness:**
- [ ] Development team trained on Tessera fundamentals
- [ ] Migration timeline and milestones defined
- [ ] Rollback plans prepared
- [ ] Code review processes adapted for Tessera

**Infrastructure Readiness:**
- [ ] CI/CD pipelines support Tessera compilation
- [ ] Performance monitoring infrastructure available
- [ ] Tessera development environment set up
- [ ] Documentation standards updated

---

## CUDA to Tessera Migration

### Basic CUDA Kernel Migration

#### Before: CUDA Vector Addition

```cuda
// CUDA kernel for vector addition
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code
void launch_vector_add(float* a, float* b, float* c, int n) {
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    vector_add<<<gridSize, blockSize>>>(a, b, c, n);
    cudaDeviceSynchronize();
}
```

#### After: Tessera Vector Addition

```python
import tessera as ts

@ts.kernel
def vector_add(a: ts.Tensor["N", ts.f32], 
               b: ts.Tensor["N", ts.f32], 
               c: ts.Tensor["N", ts.f32]):
    """Vector addition in Tessera."""
    idx = ts.program_id(0) * ts.block_size(0) + ts.thread_id(0)
    
    if idx < a.shape[0]:
        c[idx] = a[idx] + b[idx]

# Automatic launch configuration - Tessera handles block/grid sizing
def launch_vector_add(a, b, c):
    vector_add(a, b, c)
```

### Advanced CUDA Kernel Migration

#### Before: CUDA Matrix Multiplication with Shared Memory

```cuda
#define TILE_SIZE 32

__global__ void matmul_shared(float* A, float* B, float* C, 
                             int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

#### After: Tessera Matrix Multiplication with Autotuning

```python
@ts.kernel.autotune(
    space={
        "TILE_M": [64, 128, 256],
        "TILE_N": [64, 128, 256], 
        "TILE_K": [32, 64, 128],
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4]
    },
    key=["M", "N", "K"]
)
def matmul_optimized(A: ts.Tensor["M", "K", ts.f32],
                    B: ts.Tensor["K", "N", ts.f32],
                    C: ts.Tensor["M", "N", ts.f32]):
    """Optimized matrix multiplication with autotuning."""
    
    # Get autotuned parameters
    TILE_M = ts.autotune.get_param("TILE_M")
    TILE_N = ts.autotune.get_param("TILE_N") 
    TILE_K = ts.autotune.get_param("TILE_K")
    
    # Allocate shared memory with optimal layout
    As = ts.shared.alloc[ts.f32](TILE_M, TILE_K, swizzle="xor")
    Bs = ts.shared.alloc[ts.f32](TILE_K, TILE_N, swizzle="xor")
    
    # Get thread coordinates
    row = ts.program_id(0) * TILE_M + ts.thread_id(0)
    col = ts.program_id(1) * TILE_N + ts.thread_id(1)
    
    acc = 0.0
    
    # Tiled computation with async copy optimization
    for k_tile in ts.range(0, K, TILE_K):
        # Async copy to shared memory (automatically optimized)
        ts.copy_async(A[ts.program_id(0)*TILE_M:(ts.program_id(0)+1)*TILE_M, 
                       k_tile:k_tile+TILE_K], As)
        ts.copy_async(B[k_tile:k_tile+TILE_K, 
                       ts.program_id(1)*TILE_N:(ts.program_id(1)+1)*TILE_N], Bs)
        
        # Wait for copies and synchronize
        ts.wait_group(0)
        ts.barrier()
        
        # Compute tile multiplication (automatically vectorized)
        for k in ts.range(TILE_K):
            acc += As[ts.thread_id(0), k] * Bs[k, ts.thread_id(1)]
        
        ts.barrier()
    
    # Store result
    if row < M and col < N:
        C[row, col] = acc
```

### CUDA Memory Management Migration

#### Before: Manual CUDA Memory Management

```cuda
// CUDA memory management
float *d_a, *d_b, *d_c;
size_t size = N * sizeof(float);

// Allocate device memory
cudaMalloc(&d_a, size);
cudaMalloc(&d_b, size);
cudaMalloc(&d_c, size);

// Copy data to device
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

// Launch kernel
vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

// Copy result back
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

// Free memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

#### After: Tessera Automatic Memory Management

```python
# Tessera automatic memory management
def compute_vector_add(a_host, b_host):
    # Automatic device memory allocation and transfers
    a_device = ts.from_numpy(a_host)  # Automatic host->device transfer
    b_device = ts.from_numpy(b_host)
    c_device = ts.zeros_like(a_device)
    
    # Launch kernel (automatic optimization)
    vector_add(a_device, b_device, c_device)
    
    # Automatic device->host transfer when needed
    c_host = c_device.numpy()  # Only transferred when accessed
    
    return c_host  # Memory automatically freed when out of scope

# Manual control when needed
@ts.memory.manual_management
def compute_with_manual_control(a_host, b_host):
    with ts.device(0):  # Explicit device selection
        # Explicit memory allocation
        a_device = ts.empty(a_host.shape, dtype=ts.f32, device="cuda:0")
        b_device = ts.empty(b_host.shape, dtype=ts.f32, device="cuda:0")
        c_device = ts.empty(a_host.shape, dtype=ts.f32, device="cuda:0")
        
        # Explicit transfers
        a_device.copy_from(a_host)
        b_device.copy_from(b_host)
        
        # Kernel launch
        vector_add(a_device, b_device, c_device)
        
        # Explicit transfer back
        c_host = ts.empty(a_host.shape, dtype=ts.f32, device="cpu")
        c_host.copy_from(c_device)
        
        return c_host
```

### Multi-GPU CUDA Migration

#### Before: CUDA Multi-GPU with NCCL

```cuda
// CUDA multi-GPU setup
#include <nccl.h>

void setup_multi_gpu(int num_gpus) {
    ncclComm_t* comms = new ncclComm_t[num_gpus];
    int* devs = new int[num_gpus];
    
    // Initialize devices
    for (int i = 0; i < num_gpus; i++) {
        devs[i] = i;
    }
    
    // Initialize NCCL
    ncclCommInitAll(comms, num_gpus, devs);
    
    // Launch kernels on each GPU
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        my_kernel<<<grid, block>>>(data[i]);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    
    // Allreduce operation
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        ncclAllReduce(data[i], result[i], count, ncclFloat, ncclSum, comms[i], 0);
    }
}
```

#### After: Tessera Distributed Computing

```python
# Tessera multi-GPU (automatic setup)
@ts.distributed.data_parallel(devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
def distributed_computation(data):
    """Automatic data parallel execution."""
    # Data automatically sharded across GPUs
    result = my_kernel(data)
    
    # Automatic gradient allreduce during backward pass
    return result

# Manual multi-GPU control when needed
def manual_multi_gpu_setup():
    # Create mesh for 4 GPUs
    mesh = ts.Mesh(
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        axis_names=["data_parallel"],
        shape=[4]
    )
    
    @ts.distribute.with_mesh(mesh)
    def distributed_kernel(data: ts.DistributedTensor["N"]):
        # Kernel automatically runs on all GPUs
        local_result = my_kernel(data.local_shard)
        
        # Explicit collective operations
        global_result = ts.allreduce(local_result, axis="data_parallel")
        
        return global_result
    
    return distributed_kernel
```

---

## PyTorch to Tessera Migration

### Basic PyTorch Module Migration

#### Before: PyTorch Linear Layer

```python
import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return torch.mm(x, self.weight.t()) + self.bias

# Usage
layer = CustomLinear(512, 256)
output = layer(input_tensor)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

#### After: Tessera Module

```python
import tessera as ts

class CustomLinear(ts.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Tessera automatically handles parameter initialization
        self.weight = ts.Parameter(ts.randn(out_features, in_features))
        self.bias = ts.Parameter(ts.randn(out_features))
    
    @ts.compile  # Automatic optimization
    def forward(self, x):
        # Tessera automatically optimizes for target hardware
        return ts.linear(x, self.weight, self.bias)

# Usage with automatic mixed precision and optimization
layer = CustomLinear(512, 256)

# Automatic gradient computation and optimization
@ts.compile
@ts.autodiff
def training_step(input_tensor, target):
    output = layer(input_tensor)
    loss = ts.cross_entropy(output, target)
    return loss

# Automatic backward pass and optimizer step
loss_fn = training_step
optimizer = ts.optimizers.Adam(layer.parameters())

loss = loss_fn(input_tensor, target)
gradients = ts.grad(loss_fn)(input_tensor, target)
optimizer.step(gradients)
```

### Complex PyTorch Model Migration

#### Before: PyTorch Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.w_o(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

#### After: Tessera Transformer (Optimized)

```python
import tessera as ts

class OptimizedMultiHeadAttention(ts.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Fused QKV projection for better efficiency
        self.qkv_proj = ts.Linear(d_model, 3 * d_model)
        self.output_proj = ts.Linear(d_model, d_model)
        
    @ts.compile
    @ts.optimize.memory_efficient  # Automatic memory optimization
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Fused QKV computation (more efficient than separate projections)
        qkv = self.qkv_proj(x)
        q, k, v = ts.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Flash Attention (memory-efficient, numerically stable)
        attn_output = ts.flash_attention(
            q, k, v,
            causal=True,  # For autoregressive models
            scale=1.0 / (self.d_k ** 0.5)
        )
        
        # Reshape and project output
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        return self.output_proj(attn_output)

class OptimizedTransformerBlock(ts.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = OptimizedMultiHeadAttention(d_model, num_heads)
        
        # Pre-norm architecture (more stable training)
        self.norm1 = ts.RMSNorm(d_model)  # RMSNorm is more efficient
        self.norm2 = ts.RMSNorm(d_model)
        
        # SwiGLU activation (better than ReLU for transformers)
        self.ffn = ts.SwiGLU(d_model, d_ff)
        
    @ts.compile
    @ts.autodiff  # Automatic gradient computation
    def forward(self, x, mask=None):
        # Pre-norm architecture
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, mask)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with pre-norm
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output  # Residual connection
        
        return x

# Distributed transformer for large models
@ts.distributed.tensor_parallel(num_partitions=8)
class DistributedTransformer(ts.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        # Embedding layer (automatically sharded)
        self.embedding = ts.Embedding(vocab_size, d_model)
        
        # Transformer blocks (automatically sharded)
        self.blocks = ts.ModuleList([
            OptimizedTransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output projection (automatically sharded)
        self.output_proj = ts.Linear(d_model, vocab_size)
        
    @ts.compile
    @ts.checkpoint.automatic  # Automatic activation checkpointing
    def forward(self, input_ids):
        # Embedding lookup
        x = self.embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
```

### Training Loop Migration

#### Before: PyTorch Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Training setup
model = TransformerModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
```

#### After: Tessera Training Loop

```python
import tessera as ts

# Training setup with automatic optimization
model = DistributedTransformer(
    vocab_size=50000,
    d_model=768