# Tessera NVL72 Programming Guide

## Table of Contents
1. [Overview](#overview)
2. [NVL72 Architecture](#nvl72-architecture)
3. [Mesh Configuration](#mesh-configuration)
4. [Communication Optimization](#communication-optimization)
5. [Memory Management](#memory-management)
6. [Workload Distribution](#workload-distribution)
7. [Performance Optimization](#performance-optimization)
8. [Debugging and Profiling](#debugging-and-profiling)
9. [Best Practices](#best-practices)

## Overview

NVL72 represents NVIDIA's 72-GPU supernode built from GB200 superchips, connected via NVSwitch. This guide covers optimal programming strategies for this massive parallel system using Tessera.

### Key Specifications

- **72 GPUs**: 9 NVSwitch domains × 8 GPUs
- **NVSwitch Fabric**: 1.8 TB/s bisection bandwidth
- **Memory**: 144GB HBM3e per GPU (10.3TB total)
- **Compute**: 20 PetaFLOPS FP8 (1.44 ExaFLOPS total)
- **SHARP Support**: In-network reductions
- **Topology**: All-to-all connectivity within domain

## NVL72 Architecture

### Hardware Topology

```python
import tessera as ts
from tessera.nvl72 import NVL72Config

# NVL72 topology configuration
config = NVL72Config(
    num_gpus=72,
    nvswitch_domains=9,
    gpus_per_domain=8,
    nvlink_bandwidth="900 GB/s",
    nvswitch_bandwidth="1.8 TB/s",
    sharp_enabled=True
)

# Visualize topology
def visualize_nvl72_topology():
    """
    NVL72 Topology:
    
    Domain 0 (8 GPUs) ←→ NVSwitch ←→ Domain 1 (8 GPUs)
           ↓                ↓                ↓
    Domain 2 (8 GPUs) ←→ NVSwitch ←→ Domain 3 (8 GPUs)
           ↓                ↓                ↓
                        ... ...
           ↓                ↓                ↓
    Domain 8 (8 GPUs) ←→ NVSwitch ←→ ...
    
    Total: 72 GPUs with all-to-all connectivity
    """
    return config.get_topology_graph()
```

### Communication Hierarchy

```python
class NVL72Hierarchy:
    """Communication hierarchy for NVL72"""
    
    # Latency and bandwidth at different levels
    INTRA_GPU = {
        "latency_us": 0,
        "bandwidth_gbps": "memory_bandwidth"
    }
    
    INTRA_DOMAIN = {  # Within 8-GPU NVSwitch domain
        "latency_us": 2,
        "bandwidth_gbps": 900
    }
    
    INTER_DOMAIN = {  # Across NVSwitch domains
        "latency_us": 5,
        "bandwidth_gbps": 450  # Effective per-GPU
    }
    
    GLOBAL = {  # All 72 GPUs
        "latency_us": 10,
        "bandwidth_gbps": 225  # With contention
    }
```

## Mesh Configuration

### Optimal Mesh Layouts

```python
# Create optimal mesh for NVL72
def create_nvl72_mesh(parallelism_config):
    """Create mesh optimized for NVL72 topology"""
    
    # Standard configurations for different model sizes
    configs = {
        "small": (72, 1, 1),     # 72-way data parallel
        "medium": (9, 8, 1),     # 9 DP × 8 TP
        "large": (4, 9, 2),      # 4 DP × 9 TP × 2 PP
        "xlarge": (2, 9, 4),     # 2 DP × 9 TP × 4 PP
        "xxlarge": (1, 9, 8),    # 9 TP × 8 PP (model parallel only)
    }
    
    dp, tp, pp = configs.get(parallelism_config, configs["large"])
    
    # Create mesh with topology awareness
    mesh = ts.dist.mesh(
        devices=[f"cuda:{i}" for i in range(72)],
        axes=("dp", "tp", "pp"),
        shape=(dp, tp, pp),
        topology="nvl72"  # Enables NVL72-specific optimizations
    )
    
    # Set communication backend
    mesh.set_backend("nccl", {
        "NCCL_ALGO": "RING",  # Or "TREE" for reductions
        "NCCL_PROTO": "LL128",  # Low-latency 128B protocol
        "NCCL_NET_GDR_LEVEL": "5",  # GPU Direct RDMA
        "NCCL_NVLS_ENABLE": "1",  # NVLink SHARP
    })
    
    return mesh

# Example: 175B parameter model configuration
mesh_175b = create_nvl72_mesh("xlarge")  # 2 DP × 9 TP × 4 PP

# Example: 1T parameter model configuration  
mesh_1t = create_nvl72_mesh("xxlarge")  # 9 TP × 8 PP
```

### Hierarchical Mesh Design

```python
class HierarchicalMesh:
    """Hierarchical mesh for NVL72"""
    
    def __init__(self):
        # Create 3-level hierarchy
        self.levels = {
            "domain": self.create_domain_mesh(),
            "rack": self.create_rack_mesh(),
            "global": self.create_global_mesh()
        }
    
    def create_domain_mesh(self):
        """8-GPU mesh within NVSwitch domain"""
        return ts.dist.mesh(
            devices=[f"cuda:{i}" for i in range(8)],
            axes=("tp",),  # Tensor parallel within domain
            shape=(8,)
        )
    
    def create_rack_mesh(self):
        """Cross-domain mesh (9 domains)"""
        meshes = []
        for domain in range(9):
            start = domain * 8
            devices = [f"cuda:{i}" for i in range(start, start + 8)]
            meshes.append(ts.dist.mesh(devices, ("dp",), (8,)))
        return meshes
    
    def create_global_mesh(self):
        """Full 72-GPU mesh"""
        return ts.dist.mesh(
            devices=[f"cuda:{i}" for i in range(72)],
            axes=("global",),
            shape=(72,)
        )
    
    def get_optimal_collective(self, op_type, data_size):
        """Select optimal collective strategy"""
        if op_type == "allreduce":
            if data_size < 1024 * 1024:  # < 1MB
                return "ring"  # Low latency
            else:
                return "sharp"  # In-network reduction
        elif op_type == "allgather":
            return "hierarchical"  # Domain-then-global
        return "auto"
```

## Communication Optimization

### SHARP In-Network Reductions

```python
from tessera.nvl72 import SHARPConfig

# Configure SHARP for in-network reductions
sharp_config = SHARPConfig(
    enable=True,
    min_message_size="256KB",
    max_message_size="128MB",
    reduction_ops=["sum", "max", "min"],
    tree_topology="optimal"
)

@ts.jit
def sharp_optimized_allreduce(tensor, mesh):
    """AllReduce optimized for NVL72 with SHARP"""
    
    # Use SHARP for large reductions
    if tensor.numel() * tensor.element_size() > sharp_config.min_message_size:
        return ts.sharp_allreduce(tensor, mesh=mesh)
    else:
        # Fall back to ring for small messages
        return ts.ring_allreduce(tensor, mesh=mesh)

# Gradient synchronization with SHARP
@ts.jit  
def sync_gradients_nvl72(model, mesh):
    """Optimized gradient sync for NVL72"""
    
    # Group small gradients for efficiency
    small_grads = []
    large_grads = []
    
    for param in model.parameters():
        size = param.grad.numel() * param.grad.element_size()
        if size < 1024 * 1024:  # < 1MB
            small_grads.append(param.grad)
        else:
            large_grads.append(param.grad)
    
    # Fuse small gradients
    if small_grads:
        fused = ts.cat(small_grads)
        fused = sharp_optimized_allreduce(fused, mesh)
        # Unfuse
        offset = 0
        for i, grad in enumerate(small_grads):
            size = grad.numel()
            small_grads[i].copy_(fused[offset:offset+size].view_as(grad))
            offset += size
    
    # SHARP for large gradients
    for grad in large_grads:
        grad.copy_(sharp_optimized_allreduce(grad, mesh))
```

### Hierarchical Collectives

```python
class HierarchicalCollectives:
    """Hierarchical collective operations for NVL72"""
    
    @staticmethod
    def hierarchical_allreduce(tensor, mesh):
        """Two-level allreduce: intra-domain then inter-domain"""
        
        # Level 1: Reduce within each 8-GPU domain
        domain_id = mesh.get_coordinate("domain")
        intra_domain_tensor = ts.allreduce(
            tensor, 
            group=mesh.get_domain_group(domain_id),
            op="sum"
        )
        
        # Level 2: Reduce across domain leaders
        if mesh.is_domain_leader():
            inter_domain_tensor = ts.allreduce(
                intra_domain_tensor,
                group=mesh.get_leader_group(),
                op="sum"
            )
        else:
            inter_domain_tensor = intra_domain_tensor
        
        # Level 3: Broadcast within domains
        final_tensor = ts.broadcast(
            inter_domain_tensor,
            src=mesh.get_domain_leader(),
            group=mesh.get_domain_group(domain_id)
        )
        
        return final_tensor
    
    @staticmethod
    def ring_allgather(tensor_list, mesh):
        """Ring-based allgather optimized for NVL72"""
        
        world_size = 72
        rank = mesh.get_rank()
        
        # Use ring algorithm for bandwidth optimization
        for i in range(world_size - 1):
            send_rank = (rank + i) % world_size
            recv_rank = (rank - i + world_size) % world_size
            
            ts.send(tensor_list[send_rank], dest=(rank + 1) % world_size)
            tensor_list[recv_rank] = ts.recv(src=(rank - 1 + world_size) % world_size)
        
        return tensor_list
```

### Communication Scheduling

```python
class CommunicationScheduler:
    """Schedule communications to avoid congestion"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.schedule = self.compute_optimal_schedule()
    
    def compute_optimal_schedule(self):
        """Compute congestion-free communication schedule"""
        
        schedule = []
        num_domains = 9
        
        # Phase 1: Intra-domain communications
        for domain in range(num_domains):
            schedule.append({
                "phase": 1,
                "domain": domain,
                "type": "intra_domain",
                "participants": list(range(domain * 8, (domain + 1) * 8))
            })
        
        # Phase 2: Inter-domain communications (avoid congestion)
        for step in range(num_domains - 1):
            pairs = []
            for domain in range(num_domains):
                partner = (domain + step + 1) % num_domains
                if domain < partner:
                    pairs.append((domain, partner))
            
            schedule.append({
                "phase": 2,
                "step": step,
                "type": "inter_domain",
                "pairs": pairs
            })
        
        return schedule
    
    def execute_communication(self, tensor, operation):
        """Execute communication following schedule"""
        
        results = []
        for phase in self.schedule:
            if phase["phase"] == 1 and self.mesh.is_in_domain(phase["domain"]):
                # Intra-domain operation
                result = operation(tensor, group=phase["participants"])
                results.append(result)
            
            elif phase["phase"] == 2:
                # Inter-domain operation
                for src, dst in phase["pairs"]:
                    if self.mesh.is_in_domain(src) or self.mesh.is_in_domain(dst):
                        result = operation(tensor, src=src, dst=dst)
                        results.append(result)
        
        return results
```

## Memory Management

### Distributed Memory Pool

```python
class NVL72MemoryPool:
    """Memory pool optimized for NVL72"""
    
    def __init__(self, total_memory_gb=10368):  # 144GB × 72
        self.total_memory = total_memory_gb * 1024 * 1024 * 1024
        self.pools = self.create_hierarchical_pools()
    
    def create_hierarchical_pools(self):
        """Create memory pools at different hierarchy levels"""
        
        pools = {
            "local": [],      # Per-GPU pools
            "domain": [],     # Per-domain shared pools  
            "global": None    # Global shared pool
        }
        
        # Local pools (144GB per GPU)
        for gpu in range(72):
            pools["local"].append(ts.cuda.MemoryPool(
                size="140GB",  # Leave 4GB for system
                device=f"cuda:{gpu}",
                growth_factor=1.5
            ))
        
        # Domain pools (shared within 8 GPUs)
        for domain in range(9):
            pools["domain"].append(ts.cuda.MemoryPool(
                size="1TB",  # Distributed across 8 GPUs
                devices=[f"cuda:{i}" for i in range(domain*8, (domain+1)*8)],
                allocation_strategy="balanced"
            ))
        
        # Global pool for large allocations
        pools["global"] = ts.cuda.MemoryPool(
            size="10TB",
            devices=[f"cuda:{i}" for i in range(72)],
            allocation_strategy="first_fit"
        )
        
        return pools
    
    def allocate(self, size, level="local", device=None):
        """Allocate memory from appropriate pool"""
        
        if level == "local":
            return self.pools["local"][device].allocate(size)
        elif level == "domain":
            domain = device // 8
            return self.pools["domain"][domain].allocate(size)
        else:
            return self.pools["global"].allocate(size)
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        stats = {
            "total_allocated": 0,
            "total_free": 0,
            "fragmentation": 0
        }
        
        for pool in self.pools["local"]:
            pool_stats = pool.get_stats()
            stats["total_allocated"] += pool_stats["allocated"]
            stats["total_free"] += pool_stats["free"]
        
        return stats
```

### Model Sharding Strategy

```python
class NVL72ModelSharding:
    """Optimal model sharding for NVL72"""
    
    @staticmethod
    def compute_sharding_plan(model_size_gb, mesh):
        """Compute optimal sharding plan"""
        
        plan = {
            "embedding": None,
            "attention": None,
            "feedforward": None,
            "output": None
        }
        
        # For models > 1TB, use full model parallelism
        if model_size_gb > 1024:
            # Shard across all 72 GPUs
            plan["embedding"] = ts.ShardSpec(
                partition=("vocab", "hidden"),
                mesh_axes=("tp", "pp")
            )
            plan["attention"] = ts.ShardSpec(
                partition=("heads", "hidden"),
                mesh_axes=("tp", "pp")
            )
            plan["feedforward"] = ts.ShardSpec(
                partition=("hidden", "intermediate"),
                mesh_axes=("tp", "pp")
            )
        
        # For models 100GB-1TB, use hybrid parallelism
        elif model_size_gb > 100:
            # TP within domains, PP across domains
            plan["embedding"] = ts.ShardSpec(
                partition=("hidden",),
                mesh_axes=("tp",)
            )
            plan["attention"] = ts.ShardSpec(
                partition=("heads",),
                mesh_axes=("tp",)
            )
            plan["feedforward"] = ts.ShardSpec(
                partition=("intermediate",),
                mesh_axes=("tp",)
            )
        
        # For smaller models, use data parallelism
        else:
            # Replicate model, shard data
            plan["embedding"] = ts.ShardSpec(replicate=True)
            plan["attention"] = ts.ShardSpec(replicate=True)
            plan["feedforward"] = ts.ShardSpec(replicate=True)
        
        return plan
```

## Workload Distribution

### Load Balancing

```python
class NVL72LoadBalancer:
    """Load balancing for NVL72"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.gpu_loads = [0.0] * 72
        self.communication_matrix = self.build_comm_matrix()
    
    def build_comm_matrix(self):
        """Build communication cost matrix"""
        matrix = {}
        
        for i in range(72):
            for j in range(72):
                if i == j:
                    cost = 0
                elif i // 8 == j // 8:  # Same domain
                    cost = 1
                else:  # Different domains
                    cost = 5
                
                matrix[(i, j)] = cost
        
        return matrix
    
    def assign_work(self, tasks):
        """Assign tasks to minimize communication"""
        
        assignments = {}
        
        for task in tasks:
            # Find GPU with lowest load and communication cost
            best_gpu = None
            best_cost = float('inf')
            
            for gpu in range(72):
                load_cost = self.gpu_loads[gpu]
                comm_cost = self.compute_comm_cost(task, gpu)
                total_cost = load_cost + comm_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_gpu = gpu
            
            assignments[task.id] = best_gpu
            self.gpu_loads[best_gpu] += task.compute_cost
        
        return assignments
    
    def compute_comm_cost(self, task, gpu):
        """Compute communication cost for task placement"""
        
        cost = 0
        for dep in task.dependencies:
            dep_gpu = dep.assigned_gpu
            if dep_gpu is not None:
                cost += self.communication_matrix[(dep_gpu, gpu)]
        
        return cost
```

### Pipeline Parallelism

```python
class NVL72Pipeline:
    """Pipeline parallel execution for NVL72"""
    
    def __init__(self, num_stages=8, num_micro_batches=72):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.stage_assignment = self.compute_stage_assignment()
    
    def compute_stage_assignment(self):
        """Assign pipeline stages to GPUs"""
        
        # Each stage gets 9 GPUs (one domain)
        assignment = {}
        for stage in range(self.num_stages):
            assignment[stage] = list(range(stage * 9, (stage + 1) * 9))
        
        return assignment
    
    @ts.jit
    def pipeline_forward(self, micro_batches, model_stages):
        """1F1B pipeline schedule"""
        
        num_stages = len(model_stages)
        in_flight = []
        outputs = []
        
        # Warmup phase
        for i in range(num_stages):
            if i < len(micro_batches):
                mb = micro_batches[i]
                for stage_id in range(i + 1):
                    mb = model_stages[stage_id](mb)
                    if stage_id < i:
                        # Send to next stage
                        ts.send(mb, dest=self.stage_assignment[stage_id + 1][0])
                in_flight.append(mb)
        
        # Steady state (1F1B)
        for i in range(num_stages, self.num_micro_batches):
            # Forward
            mb = micro_batches[i]
            for stage_id in range(num_stages):
                mb = model_stages[stage_id](mb)
                if stage_id < num_stages - 1:
                    ts.send(mb, dest=self.stage_assignment[stage_id + 1][0])
            
            # Backward (process oldest in-flight)
            oldest = in_flight.pop(0)
            grad = compute_gradient(oldest)
            for stage_id in range(num_stages - 1, -1, -1):
                grad = model_stages[stage_id].backward(grad)
                if stage_id > 0:
                    ts.send(grad, dest=self.stage_assignment[stage_id - 1][0])
            
            in_flight.append(mb)
            outputs.append(mb)
        
        # Cooldown phase
        while in_flight:
            oldest = in_flight.pop(0)
            grad = compute_gradient(oldest)
            for stage_id in range(num_stages - 1, -1, -1):
                grad = model_stages[stage_id].backward(grad)
                if stage_id > 0:
                    ts.send(grad, dest=self.stage_assignment[stage_id - 1][0])
            outputs.append(oldest)
        
        return outputs
```

## Performance Optimization

### Kernel Optimization for NVL72

```python
@ts.kernel.nvl72_optimized
def nvl72_attention(Q, K, V):
    """Attention kernel optimized for NVL72"""
    
    # Use larger tile sizes for H100 GPUs
    BLOCK_M = 256  # Larger than typical 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    # Enable Hopper-specific features
    ts.tile.enable_tma()  # Tensor Memory Accelerator
    ts.tile.enable_wgmma()  # Warp Group MMA
    ts.tile.enable_cluster_mode(shape=(2, 2, 1))
    
    # Allocate distributed shared memory
    smem_q = ts.tile.alloc_distributed_shared(
        (BLOCK_M, head_dim),
        dtype=ts.bf16,
        swizzle="hopper_optimal"
    )
    
    # ... rest of attention implementation
```

### Overlapping Computation and Communication

```python
class OverlappedExecution:
    """Overlap computation with communication on NVL72"""
    
    @ts.jit
    def overlapped_allreduce(self, tensor, compute_fn, mesh):
        """Overlap allreduce with computation"""
        
        # Split tensor for overlapping
        chunks = ts.chunk(tensor, chunks=4)
        
        # Start first allreduce
        handle0 = ts.allreduce_async(chunks[0], mesh=mesh)
        
        results = []
        for i in range(len(chunks)):
            # Start next allreduce
            if i + 1 < len(chunks):
                handle_next = ts.allreduce_async(chunks[i + 1], mesh=mesh)
            
            # Wait for current chunk
            reduced = ts.wait(handle0 if i == 0 else handle_next)
            
            # Compute while next chunk is being reduced
            with ts.cuda.stream(ts.cuda.Stream()):
                computed = compute_fn(reduced)
                results.append(computed)
            
            if i + 1 < len(chunks):
                handle0 = handle_next
        
        return ts.cat(results)
```

### Performance Profiling

```python
class NVL72Profiler:
    """Profiling tools for NVL72"""
    
    def profile_communication(self, mesh):
        """Profile communication patterns"""
        
        results = {
            "intra_domain_bandwidth": [],
            "inter_domain_bandwidth": [],
            "collective_performance": {}
        }
        
        # Test intra-domain bandwidth
        for domain in range(9):
            start_gpu = domain * 8
            tensor = ts.randn(1024 * 1024 * 256, device=f"cuda:{start_gpu}")  # 1GB
            
            start = ts.cuda.Event()
            end = ts.cuda.Event()
            
            start.record()
            ts.broadcast(tensor, src=start_gpu, 
                        group=list(range(start_gpu, start_gpu + 8)))
            end.record()
            
            ts.cuda.synchronize()
            time_ms = start.elapsed_time(end)
            bandwidth = (1024 * 8) / time_ms  # GB/s
            results["intra_domain_bandwidth"].append(bandwidth)
        
        # Test collective operations
        for size in [1, 10, 100, 1000]:  # MB
            tensor = ts.randn(size * 1024 * 256, device="cuda:0")
            
            # AllReduce
            start.record()
            ts.allreduce(tensor, mesh=mesh)
            end.record()
            ts.cuda.synchronize()
            
            time_ms = start.elapsed_time(end)
            results["collective_performance"][f"allreduce_{size}MB"] = {
                "time_ms": time_ms,
                "bandwidth_gbps": (size * 72 * 2) / time_ms  # Approximate
            }
        
        return results
```

## Debugging and Profiling

### Distributed Debugging

```python
class NVL72Debugger:
    """Debugging tools for NVL72"""
    
    def __init__(self):
        self.breakpoints = {}
        self.watch_tensors = {}
    
    def distributed_breakpoint(self, rank, condition=None):
        """Set breakpoint for specific rank"""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                current_rank = ts.distributed.get_rank()
                
                if current_rank == rank:
                    if condition is None or condition():
                        print(f"Breakpoint hit at rank {rank}")
                        import pdb; pdb.set_trace()
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def check_tensor_consistency(self, tensor, mesh, tolerance=1e-5):
        """Check if tensor is consistent across all GPUs"""
        
        all_tensors = ts.all_gather(tensor, mesh=mesh)
        
        reference = all_tensors[0]
        for i, t in enumerate(all_tensors[1:], 1):
            diff = ts.abs(t - reference).max()
            if diff > tolerance:
                print(f"Inconsistency detected: GPU 0 vs GPU {i}, max diff: {diff}")
                return False
        
        return True
    
    def profile_gpu_utilization(self, duration_seconds=10):
        """Profile GPU utilization across all 72 GPUs"""
        
        import nvidia_smi
        nvidia_smi.nvmlInit()
        
        utilization = {}
        for gpu in range(72):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            
            utilization[gpu] = {
                "gpu": util.gpu,
                "memory": util.memory,
                "temperature": nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
            }
        
        return utilization
```

### Performance Analysis

```python
class NVL72PerformanceAnalyzer:
    """Analyze performance bottlenecks on NVL72"""
    
    def analyze_model(self, model, batch_size, seq_len):
        """Comprehensive model analysis"""
        
        analysis = {
            "compute_time": {},
            "communication_time": {},
            "memory_usage": {},
            "bottlenecks": []
        }
        
        # Profile forward pass
        with ts.profile() as prof:
            input_data = ts.randn(batch_size, seq_len, model.hidden_size)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
        
        # Analyze profile
        for event in prof.events():
            if "allreduce" in event.name:
                analysis["communication_time"][event.name] = event.cuda_time_ms
            elif "gemm" in event.name or "mma" in event.name:
                analysis["compute_time"][event.name] = event.cuda_time_ms
        
        # Identify bottlenecks
        total_compute = sum(analysis["compute_time"].values())
        total_comm = sum(analysis["communication_time"].values())
        
        if total_comm > total_compute:
            analysis["bottlenecks"].append("communication_bound")
            analysis["recommendations"] = [
                "Increase compute intensity",
                "Use gradient accumulation",
                "Enable SHARP reductions"
            ]
        else:
            analysis["bottlenecks"].append("compute_bound")
            analysis["recommendations"] = [
                "Use mixed precision",
                "Enable tensor cores",
                "Increase batch size"
            ]
        
        return analysis
```

## Best Practices

### Configuration Guidelines

```python
# Optimal configurations for different scenarios

# Training large language models (>100B parameters)
LLM_CONFIG = {
    "mesh": (4, 9, 2),  # 4 DP × 9 TP × 2 PP
    "micro_batch_size": 1,
    "gradient_accumulation": 8,
    "communication_backend": "nccl",
    "enable_sharp": True,
    "memory_optimization": "gradient_checkpointing",
    "precision": "bf16",
    "optimizer": "fused_adamw"
}

# Training vision transformers
VIT_CONFIG = {
    "mesh": (72, 1, 1),  # 72-way data parallel
    "batch_size": 4096,
    "communication_backend": "nccl",
    "enable_sharp": False,  # Not needed for DP
    "precision