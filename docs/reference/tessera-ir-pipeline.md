# Tessera IR Stack and Compilation Pipeline

## Table of Contents
1. [Overview](#overview)
2. [Graph IR](#graph-ir)
3. [Schedule IR](#schedule-ir)
4. [Tile IR](#tile-ir)
5. [Target IR](#target-ir)
6. [Compilation Flow](#compilation-flow)
7. [Optimization Passes](#optimization-passes)
8. [IR Inspection and Debugging](#ir-inspection-and-debugging)
9. [Custom Passes](#custom-passes)

## Overview

Tessera uses a multi-level intermediate representation (IR) stack to transform high-level Python code into optimized machine code. Each IR level serves a specific purpose in the compilation pipeline.

### IR Stack Architecture

```
Python/Tessera DSL
        ↓
   Graph IR (High-level operations, autodiff, effects)
        ↓
  Schedule IR (Loop tiling, memory placement, parallelization)
        ↓
    Tile IR (Hardware-aware operations, intrinsics, barriers)
        ↓
   Target IR (Architecture-specific optimization)
        ↓
  Machine Code (PTX, CUDA Tile IR, x86, etc.)
```

### Design Principles

1. **Progressive Lowering**: Each level makes specific decisions
2. **Preserving Semantics**: Mathematical correctness maintained
3. **Optimization Opportunities**: Each level enables different optimizations
4. **Retargetability**: Lower levels abstract hardware differences

## Graph IR

Graph IR represents the highest level of abstraction, capturing the mathematical computation graph with effects and dependencies.

### Graph IR Structure

```python
# Example: Graph IR representation
@ts.trace
def example_function(x, W, b):
    """Example function to show Graph IR"""
    y = ts.matmul(x, W)
    y = y + b
    y = ts.relu(y)
    return y

# Inspect Graph IR
graph_ir = ts.inspect_ir(example_function, level="graph")
print(graph_ir)
```

Output:
```mlir
func @example_function(%x: tensor<?x?xf32>, %W: tensor<?x?xf32>, %b: tensor<?xf32>) -> tensor<?x?xf32> {
  %0 = "tsg.matmul"(%x, %W) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tsg.broadcast"(%b) {dimensions = [0]} : (tensor<?xf32>) -> tensor<?x?xf32>
  %2 = "tsg.add"(%0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tsg.relu"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
```

### Graph IR Operations

```python
class GraphIRBuilder:
    """Builder for Graph IR operations"""
    
    def build_operation(self, op_type, inputs, attributes=None):
        """Build a Graph IR operation"""
        return {
            "op": op_type,
            "inputs": inputs,
            "attributes": attributes or {},
            "effects": self.infer_effects(op_type, inputs)
        }
    
    def infer_effects(self, op_type, inputs):
        """Infer effects for operations"""
        effects = []
        
        if op_type in ["allreduce", "allgather", "broadcast"]:
            effects.append("collective")
        
        if op_type in ["randn", "randint", "dropout"]:
            effects.append("random")
        
        if op_type in ["assign", "scatter", "accumulate"]:
            effects.append("stateful")
        
        return effects
```

### Autodiff in Graph IR

```python
# Autodiff annotations in Graph IR
def build_autodiff_graph(forward_graph):
    """Build backward graph from forward graph"""
    
    backward_ops = []
    
    for op in reversed(forward_graph.ops):
        if op.type == "matmul":
            # Generate gradient operations for matmul
            grad_x = GraphOp("matmul", [op.grad_output, op.inputs[1].T])
            grad_W = GraphOp("matmul", [op.inputs[0].T, op.grad_output])
            backward_ops.extend([grad_x, grad_W])
            
        elif op.type == "relu":
            # Generate gradient for ReLU
            mask = GraphOp("greater", [op.inputs[0], 0])
            grad = GraphOp("where", [mask, op.grad_output, 0])
            backward_ops.append(grad)
            
        elif op.type == "add":
            # Generate gradient for addition
            grad_a = op.grad_output
            grad_b = GraphOp("reduce_sum", [op.grad_output], {"axis": 0})
            backward_ops.extend([grad_a, grad_b])
    
    return GraphIR(backward_ops)
```

### Effect System

```python
@dataclass
class Effect:
    """Effect annotation for Graph IR operations"""
    type: str  # "read", "write", "collective", "random", "io"
    scope: str  # "local", "device", "global"
    ordering_constraint: bool = False

class EffectAnalysis:
    """Analyze and track effects in Graph IR"""
    
    def analyze_effects(self, graph):
        effects_map = {}
        
        for op in graph.ops:
            effects = []
            
            # Memory effects
            if op.type in ["load", "gather"]:
                effects.append(Effect("read", "device"))
            elif op.type in ["store", "scatter"]:
                effects.append(Effect("write", "device"))
            
            # Collective effects
            elif op.type in ["allreduce", "allgather"]:
                effects.append(Effect("collective", "global", ordering_constraint=True))
            
            # Random effects
            elif op.type in ["dropout", "randn"]:
                effects.append(Effect("random", "local"))
            
            effects_map[op.id] = effects
        
        return effects_map
    
    def can_reorder(self, op1, op2, effects_map):
        """Check if two operations can be reordered"""
        effects1 = effects_map[op1.id]
        effects2 = effects_map[op2.id]
        
        # Check for conflicting effects
        for e1 in effects1:
            for e2 in effects2:
                if self.effects_conflict(e1, e2):
                    return False
        return True
    
    def effects_conflict(self, e1, e2):
        """Check if two effects conflict"""
        # Write-write conflict
        if e1.type == "write" and e2.type == "write":
            return True
        
        # Read-write conflict
        if (e1.type == "read" and e2.type == "write") or \
           (e1.type == "write" and e2.type == "read"):
            return True
        
        # Ordering constraints
        if e1.ordering_constraint or e2.ordering_constraint:
            return True
        
        return False
```

## Schedule IR

Schedule IR represents scheduling decisions: how operations are tiled, parallelized, and mapped to memory.

### Schedule IR Representation

```python
# Schedule IR example
@ts.schedule
def scheduled_gemm(A, B, C):
    """GEMM with explicit scheduling"""
    M, N, K = get_problem_size(A, B)
    
    # Scheduling decisions
    schedule = ts.Schedule()
    
    # Tiling
    schedule.tile(M, tile_size=128, name="M_outer", inner="M_inner")
    schedule.tile(N, tile_size=128, name="N_outer", inner="N_inner")
    schedule.tile(K, tile_size=32, name="K_outer", inner="K_inner")
    
    # Memory placement
    schedule.cache_read(A, "shared", [M_inner, K_inner])
    schedule.cache_read(B, "shared", [K_inner, N_inner])
    schedule.cache_write(C, "local", [M_inner, N_inner])
    
    # Parallelization
    schedule.parallel(M_outer, "blockIdx.y")
    schedule.parallel(N_outer, "blockIdx.x")
    schedule.parallel(M_inner, "threadIdx.y")
    schedule.parallel(N_inner, "threadIdx.x")
    
    # Unrolling
    schedule.unroll(K_inner)
    
    # Vectorization
    schedule.vectorize(N_inner, vector_width=4)
    
    return schedule
```

### Schedule IR Operations

```mlir
// Schedule IR representation
func @scheduled_gemm {
  // Loop nest structure
  tss.for %m_outer = 0 to %M step 128 {
    tss.for %n_outer = 0 to %N step 128 {
      // Shared memory allocation
      %smem_a = tss.alloc_shared [128, 32] : memref<128x32xf32, 3>
      %smem_b = tss.alloc_shared [32, 128] : memref<32x128xf32, 3>
      
      tss.for %k_outer = 0 to %K step 32 {
        // Load tiles to shared memory
        tss.copy_async %A[%m_outer, %k_outer], %smem_a
        tss.copy_async %B[%k_outer, %n_outer], %smem_b
        tss.barrier
        
        // Compute tile
        tss.for %m_inner = 0 to 128 {
          tss.for %n_inner = 0 to 128 {
            %acc = tss.alloc_register : f32
            
            tss.for %k_inner = 0 to 32 unroll {
              %a = tss.load %smem_a[%m_inner, %k_inner]
              %b = tss.load %smem_b[%k_inner, %n_inner]
              %prod = tss.mul %a, %b
              %acc = tss.add %acc, %prod
            }
            
            tss.store %acc, %C[%m_outer + %m_inner, %n_outer + %n_inner]
          }
        }
      }
    }
  }
}
```

### Schedule Primitives

```python
class SchedulePrimitives:
    """Core scheduling primitives"""
    
    def tile(self, loop, tile_size):
        """Tile a loop into outer and inner loops"""
        return {
            "type": "tile",
            "loop": loop,
            "tile_size": tile_size,
            "outer": f"{loop}_outer",
            "inner": f"{loop}_inner"
        }
    
    def fuse(self, loop1, loop2):
        """Fuse two loops into one"""
        return {
            "type": "fuse",
            "loops": [loop1, loop2],
            "fused": f"{loop1}_{loop2}_fused"
        }
    
    def reorder(self, loops):
        """Reorder loop nest"""
        return {
            "type": "reorder",
            "order": loops
        }
    
    def cache(self, tensor, memory_level, indices):
        """Cache tensor in specified memory"""
        return {
            "type": "cache",
            "tensor": tensor,
            "memory": memory_level,
            "indices": indices
        }
    
    def parallel(self, loop, mapping):
        """Map loop to parallel execution"""
        return {
            "type": "parallel",
            "loop": loop,
            "mapping": mapping  # blockIdx, threadIdx, etc.
        }
    
    def pipeline(self, stages, depth):
        """Pipeline execution stages"""
        return {
            "type": "pipeline",
            "stages": stages,
            "depth": depth
        }
```

### Autotuning in Schedule IR

```python
class Autotuner:
    """Autotuning for schedule selection"""
    
    def __init__(self, search_space):
        self.search_space = search_space
        self.best_config = None
        self.best_time = float('inf')
    
    def tune(self, kernel, inputs, iterations=100):
        """Find best schedule configuration"""
        
        for config in self.generate_configs():
            # Apply schedule
            scheduled_kernel = self.apply_schedule(kernel, config)
            
            # Compile and run
            compiled = ts.compile(scheduled_kernel)
            
            # Measure performance
            times = []
            for _ in range(iterations):
                start = ts.cuda.Event()
                end = ts.cuda.Event()
                
                start.record()
                compiled(*inputs)
                end.record()
                
                ts.cuda.synchronize()
                times.append(start.elapsed_time(end))
            
            avg_time = sum(times) / len(times)
            
            if avg_time < self.best_time:
                self.best_time = avg_time
                self.best_config = config
        
        return self.best_config
    
    def generate_configs(self):
        """Generate configuration search space"""
        configs = []
        
        for tile_m in self.search_space["tile_m"]:
            for tile_n in self.search_space["tile_n"]:
                for tile_k in self.search_space["tile_k"]:
                    for num_warps in self.search_space["num_warps"]:
                        configs.append({
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "tile_k": tile_k,
                            "num_warps": num_warps
                        })
        
        return configs
```

## Tile IR

Tile IR represents hardware-aware operations with explicit memory hierarchy and synchronization.

### Tile IR Structure

```python
# Tile IR representation
@ts.tile_ir
def tile_gemm(A, B, C):
    """GEMM in Tile IR"""
    
    # Thread and block indices
    tid = ts.tile.thread_id()
    bid = ts.tile.block_id()
    
    # Shared memory allocation
    smem_a = ts.tile.alloc_shared((128, 32), dtype=ts.f32)
    smem_b = ts.tile.alloc_shared((32, 128), dtype=ts.f32)
    
    # Register allocation
    acc = ts.tile.alloc_register((8, 8), dtype=ts.f32)
    
    # Main computation loop
    for k in range(0, K, 32):
        # Async copy to shared memory
        ts.tile.cp_async(smem_a, A[bid.y*128:(bid.y+1)*128, k:k+32])
        ts.tile.cp_async(smem_b, B[k:k+32, bid.x*128:(bid.x+1)*128])
        
        # Wait for async copies
        ts.tile.cp_wait_group(0)
        ts.tile.barrier()
        
        # Compute using tensor cores
        ts.tile.mma(smem_a, smem_b, acc)
    
    # Store result
    ts.tile.store(C[bid.y*128:(bid.y+1)*128, bid.x*128:(bid.x+1)*128], acc)
```

### Tile IR Operations

```mlir
// Tile IR MLIR representation
func @tile_gemm(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  // Get tile coordinates
  %tid_x = tsi.thread_id_x : index
  %tid_y = tsi.thread_id_y : index
  %bid_x = tsi.block_id_x : index
  %bid_y = tsi.block_id_y : index
  
  // Allocate shared memory
  %smem_a = tsi.alloc_shared [128, 32] : memref<128x32xf32, 3>
  %smem_b = tsi.alloc_shared [32, 128] : memref<32x128xf32, 3>
  
  // Allocate registers
  %acc = tsi.alloc_register [8, 8] : memref<8x8xf32, 5>
  
  // Initialize accumulator
  tsi.fill %acc, 0.0 : memref<8x8xf32, 5>
  
  // K-loop
  tsi.for %k = 0 to %K step 32 {
    // Async memory operations
    tsi.cp_async %A[%bid_y*128, %k], %smem_a : memref<128x32xf32, 3>
    tsi.cp_async %B[%k, %bid_x*128], %smem_b : memref<32x128xf32, 3>
    
    // Synchronization
    tsi.cp_wait_group 0
    tsi.barrier
    
    // Tensor core operation
    tsi.mma %smem_a, %smem_b, %acc : 
      memref<128x32xf32, 3>, memref<32x128xf32, 3>, memref<8x8xf32, 5>
  }
  
  // Store result
  tsi.store %acc, %C[%bid_y*128, %bid_x*128] : memref<8x8xf32, 5>
}
```

### Memory Operations in Tile IR

```python
class TileMemoryOps:
    """Memory operations in Tile IR"""
    
    @staticmethod
    def load_matrix(ptr, shape, layout="row_major"):
        """Load matrix from global memory"""
        return {
            "op": "tile.load",
            "ptr": ptr,
            "shape": shape,
            "layout": layout,
            "space": "global"
        }
    
    @staticmethod
    def store_matrix(ptr, data, layout="row_major"):
        """Store matrix to global memory"""
        return {
            "op": "tile.store",
            "ptr": ptr,
            "data": data,
            "layout": layout,
            "space": "global"
        }
    
    @staticmethod
    def async_copy(src, dst, size):
        """Asynchronous copy operation"""
        return {
            "op": "tile.cp_async",
            "src": src,
            "dst": dst,
            "size": size,
            "completion_mechanism": "mbarrier"
        }
    
    @staticmethod
    def prefetch(ptr, size, cache_level="L2"):
        """Prefetch data to cache"""
        return {
            "op": "tile.prefetch",
            "ptr": ptr,
            "size": size,
            "cache": cache_level
        }
```

### Synchronization in Tile IR

```python
class TileSyncOps:
    """Synchronization operations in Tile IR"""
    
    @staticmethod
    def barrier(scope="block"):
        """Synchronization barrier"""
        return {
            "op": "tile.barrier",
            "scope": scope  # "warp", "block", "cluster", "grid"
        }
    
    @staticmethod
    def fence(scope="gpu", semantics="release"):
        """Memory fence"""
        return {
            "op": "tile.fence",
            "scope": scope,
            "semantics": semantics  # "acquire", "release", "acq_rel"
        }
    
    @staticmethod
    def wait_group(group_id):
        """Wait for async operation group"""
        return {
            "op": "tile.wait_group",
            "group": group_id
        }
    
    @staticmethod
    def arrive_barrier(barrier_id, count=1):
        """Arrive at named barrier"""
        return {
            "op": "tile.arrive",
            "barrier": barrier_id,
            "count": count
        }
```

## Target IR

Target IR is the lowest level, with architecture-specific optimizations and code generation.

### Target IR for NVIDIA GPUs

```python
# PTX generation from Target IR
class PTXGenerator:
    """Generate PTX assembly from Target IR"""
    
    def generate(self, target_ir):
        """Generate PTX code"""
        ptx = []
        
        # PTX header
        ptx.append(".version 8.0")
        ptx.append(".target sm_80")
        ptx.append(".address_size 64")
        
        # Generate kernel
        for func in target_ir.functions:
            ptx.extend(self.generate_function(func))
        
        return "\n".join(ptx)
    
    def generate_function(self, func):
        """Generate PTX for a function"""
        code = []
        
        # Function declaration
        code.append(f".visible .entry {func.name}(")
        for param in func.params:
            code.append(f"  .param .{param.type} {param.name},")
        code[-1] = code[-1].rstrip(",")  # Remove trailing comma
        code.append(") {")
        
        # Register declarations
        for reg in func.registers:
            code.append(f"  .reg .{reg.type} {reg.name}<{reg.count}>;")
        
        # Shared memory
        for smem in func.shared_memory:
            code.append(f"  .shared .align {smem.align} .b8 {smem.name}[{smem.size}];")
        
        # Generate instructions
        for instr in func.instructions:
            code.append(f"  {self.generate_instruction(instr)}")
        
        code.append("}")
        return code
    
    def generate_instruction(self, instr):
        """Generate PTX instruction"""
        if instr.op == "ld.global":
            return f"ld.global.{instr.type} {instr.dst}, [{instr.src}];"
        elif instr.op == "st.global":
            return f"st.global.{instr.type} [{instr.dst}], {instr.src};"
        elif instr.op == "mad":
            return f"mad.{instr.type} {instr.dst}, {instr.src1}, {instr.src2}, {instr.src3};"
        elif instr.op == "wmma.mma":
            return self.generate_wmma(instr)
        # ... more instructions
```

### CUDA Tile IR Generation

```python
# CUDA Tile IR for Hopper GPUs
class CUDATileIRGenerator:
    """Generate CUDA Tile IR for Hopper architecture"""
    
    def generate(self, target_ir):
        """Generate CUDA Tile IR"""
        code = []
        
        # Enable Hopper features
        code.append("// Enable Hopper features")
        code.append("#pragma once")
        code.append("#include <cuda/barrier>")
        code.append("#include <cuda/pipeline>")
        
        for func in target_ir.functions:
            code.extend(self.generate_function(func))
        
        return "\n".join(code)
    
    def generate_wgmma(self, op):
        """Generate WGMMA operations for Hopper"""
        return f"""
        // WGMMA operation
        {{
            using namespace cute;
            auto tiled_mma = make_tiled_mma(
                GMMA::ss_op_selector<{op.m}, {op.n}, {op.k}, {op.dtype}>()
            );
            
            auto [aMatrix, bMatrix, cMatrix] = tiled_mma.get_slice({op.warp_id});
            tiled_mma({op.a_frag}, {op.b_frag}, {op.c_frag});
        }}
        """
    
    def generate_tma(self, op):
        """Generate TMA operations"""
        return f"""
        // TMA bulk copy
        {{
            __shared__ cuda::barrier<cuda::thread_scope_block> bar;
            if (threadIdx.x == 0) {{
                init(&bar, blockDim.x);
            }}
            __syncthreads();
            
            cuda::memcpy_async(
                {op.dst}, {op.src},
                cuda::aligned_size_t<16>({op.size}),
                bar
            );
            
            bar.arrive_and_wait();
        }}
        """
```

## Compilation Flow

### Complete Compilation Pipeline

```python
class TesseraCompiler:
    """Main compilation pipeline"""
    
    def compile(self, source_code, target="cuda", optimization_level=3):
        """Compile Tessera code to target"""
        
        # Parse Python/Tessera code
        ast = self.parse(source_code)
        
        # Generate Graph IR
        graph_ir = self.generate_graph_ir(ast)
        
        # Optimization passes on Graph IR
        graph_ir = self.optimize_graph_ir(graph_ir, level=optimization_level)
        
        # Lower to Schedule IR
        schedule_ir = self.lower_to_schedule_ir(graph_ir)
        
        # Apply scheduling optimizations
        schedule_ir = self.optimize_schedule_ir(schedule_ir)
        
        # Lower to Tile IR
        tile_ir = self.lower_to_tile_ir(schedule_ir)
        
        # Tile-level optimizations
        tile_ir = self.optimize_tile_ir(tile_ir)
        
        # Lower to Target IR
        target_ir = self.lower_to_target_ir(tile_ir, target)
        
        # Target-specific optimizations
        target_ir = self.optimize_target_ir(target_ir, target)
        
        # Code generation
        if target == "cuda":
            return self.generate_cuda(target_ir)
        elif target == "cpu":
            return self.generate_cpu(target_ir)
        else:
            raise ValueError(f"Unknown target: {target}")
    
    def optimize_graph_ir(self, ir, level):
        """Apply Graph IR optimizations"""
        passes = []
        
        if level >= 1:
            passes.extend([
                DeadCodeElimination(),
                CommonSubexpressionElimination(),
                ConstantFolding(),
            ])
        
        if level >= 2:
            passes.extend([
                OperatorFusion(),
                MemoryPlanning(),
                CollectiveOptimization(),
            ])
        
        if level >= 3:
            passes.extend([
                AutodiffOptimization(),
                NumericalStabilization(),
                EffectReordering(),
            ])
        
        for pass_obj in passes:
            ir = pass_obj.run(ir)
        
        return ir
```

## Optimization Passes

### Graph IR Optimizations

```python
class OperatorFusion:
    """Fuse compatible operations in Graph IR"""
    
    def run(self, graph_ir):
        """Run fusion pass"""
        fused_graph = graph_ir.copy()
        
        # Find fusable patterns
        patterns = self.find_fusable_patterns(fused_graph)
        
        for pattern in patterns:
            if self.can_fuse(pattern):
                fused_op = self.create_fused_op(pattern)
                fused_graph.replace_pattern(pattern, fused_op)
        
        return fused_graph
    
    def find_fusable_patterns(self, graph):
        """Find operations that can be fused"""
        patterns = []
        
        # Elementwise fusion
        for op in graph.ops:
            if op.type in ["add", "mul", "relu", "tanh"]:
                chain = self.find_elementwise_chain(op, graph)
                if len(chain) > 1:
                    patterns.append(chain)
        
        # GEMM + bias + activation
        for op in graph.ops:
            if op.type == "matmul":
                if self.has_bias_add(op) and self.has_activation(op):
                    patterns.append(self.get_gemm_pattern(op))
        
        return patterns
    
    def can_fuse(self, pattern):
        """Check if pattern can be fused"""
        # Check memory requirements
        total_memory = sum(op.memory_requirement for op in pattern)
        if total_memory > MAX_SHARED_MEMORY:
            return False
        
        # Check for conflicting effects
        effects = [op.effects for op in pattern]
        if any("collective" in e for e in effects):
            return False
        
        return True
```

### Schedule IR Optimizations

```python
class LoopTiling:
    """Optimize loop tiling in Schedule IR"""
    
    def run(self, schedule_ir):
        """Run tiling optimization"""
        optimized = schedule_ir.copy()
        
        for loop_nest in optimized.loop_nests:
            # Analyze loop properties
            analysis = self.analyze_loop(loop_nest)
            
            # Determine optimal tile sizes
            tile_sizes = self.compute_tile_sizes(analysis)
            
            # Apply tiling
            tiled_loop = self.apply_tiling(loop_nest, tile_sizes)
            optimized.replace_loop(loop_nest, tiled_loop)
        
        return optimized
    
    def compute_tile_sizes(self, analysis):
        """Compute optimal tile sizes"""
        # Consider cache sizes
        l1_size = 128 * 1024  # 128 KB
        l2_size = 40 * 1024 * 1024  # 40 MB
        
        # Consider register pressure
        max_registers = 255
        
        # Compute tile sizes that fit in cache
        tile_m = min(analysis.m_size, int(math.sqrt(l1_size / 4)))
        tile_n = min(analysis.n_size, int(math.sqrt(l1_size / 4)))
        tile_k = min(analysis.k_size, 32)  # Small K for accumulation
        
        return {
            "M": tile_m,
            "N": tile_n,
            "K": tile_k
        }
```

### Tile IR Optimizations

```python
class BankConflictElimination:
    """Eliminate shared memory bank conflicts"""
    
    def run(self, tile_ir):
        """Run bank conflict elimination"""
        optimized = tile_ir.copy()
        
        for smem_alloc in optimized.shared_memory_allocations:
            if self.has_bank_conflicts(smem_alloc):
                # Apply padding or swizzling
                if smem_alloc.size < 16384:
                    self.apply_padding(smem_alloc)
                else:
                    self.apply_swizzling(smem_alloc)
        
        return optimized
    
    def apply_swizzling(self, smem_alloc):
        """Apply XOR swizzling pattern"""
        smem_alloc.layout = "xor_swizzle"
        smem_alloc.swizzle_mask = 0x7  # XOR with lower 3 bits
```

## IR Inspection and Debugging

### IR Inspection Tools

```python
class IRInspector:
    """Tools for inspecting IR at different levels"""
    
    def inspect(self, function, level="all", format="text"):
        """Inspect IR at specified level"""
        
        if level == "all":
            return {
                "graph": self.inspect_graph_ir(function),
                "schedule": self.inspect_schedule_ir(function),
                "tile": self.inspect_tile_ir(function),
                "target": self.inspect_target_ir(function)
            }
        elif level == "graph":
            return self.inspect_graph_ir(function)
        elif level == "schedule":
            return self.inspect_schedule_ir(function)
        elif level == "tile":
            return self.inspect_tile_ir(function)
        elif level == "target":
            return self.inspect_target_ir(function)
        else:
            raise ValueError(f"Unknown IR level: {level}")
    
    def inspect_graph_ir(self, function):
        """Inspect Graph IR"""
        graph_ir = ts.trace_to_graph_ir(function)
        return self.format_graph_ir(graph_ir)
    
    def format_graph_ir(self, graph_ir):
        """Format Graph IR for display"""
        output = []
        output.append("=== Graph IR ===")
        
        for op in graph_ir.operations:
            output.append(f"{op.id}: {op.type}")
            output.append(f"  Inputs: {op.inputs}")
            output.append(f"  Outputs: {op.outputs}")
            output.append(f"  Attributes: {op.attributes}")
            output.append(f"  Effects: {op.effects}")
        
        return "\n".join(output)

# Example usage
@ts.trace
def example_kernel(x, W):
    y = ts.matmul(x, W)
    y = ts.relu(y)
    return y

inspector = IRInspector()

# Inspect all IR levels
all_ir = inspector.inspect(example_kernel, level="all")
print(all_ir["graph"])

# Inspect specific level
tile_ir = inspector.inspect(example_kernel, level="tile")
print(tile_ir)
```

### IR Debugging

```python
class IRDebugger:
    """Debugging tools for IR"""
    
    def __init__(self):
        self.breakpoints = []
        self.watches = []
        self.step_mode = False
    
    def set_breakpoint(self, ir_level, op_type):
        """Set breakpoint at specific operation"""
        self.breakpoints.append({
            "level": ir_level,
            "op_type": op_type
        })
    
    def watch(self, variable):
        """Watch variable through compilation"""
        self.watches.append(variable)
    
    def step_through(self, ir, level):
        """Step through IR execution"""
        for op in ir.operations:
            if self.should_break(op, level):
                self.show_state(op, ir)
                input("Press Enter to continue...")
            
            if self.step_mode:
                self.show_operation(op)
                input("Next step...")
    
    def validate_ir(self, ir, level):
        """Validate IR correctness"""
        validators = {
            "graph": self.validate_graph_ir,
            "schedule": self.validate_schedule_ir,
            "tile": self.validate_tile_ir,
            "target": self.validate_target_ir
        }
        
        validator = validators.get(level)
        if validator:
            errors = validator(ir)
            if errors:
                self.report_errors(errors)
                return False
        return True
    
    def validate_tile_ir(self, tile_ir):
        """Validate Tile IR correctness"""
        errors = []
        
        # Check shared memory limits
        total_smem = 0
        for alloc in tile_ir.shared_allocations:
            total_smem += alloc.size
        
        if total_smem > 228 * 1024:  # Hopper limit
            errors.append(f"Shared memory exceeds limit: {total_smem} bytes")
        
        # Check register limits
        for func in tile_ir.functions:
            if func.register_count > 255:
                errors.append(f"Register count exceeds limit: {func.register_count}")
        
        # Check synchronization correctness
        for barrier in tile_ir.barriers:
            if not self.is_barrier_correct(barrier):
                errors.append(f"Incorrect barrier placement: {barrier}")
        
        return errors
```

### Performance Analysis in IR

```python
class IRPerformanceAnalyzer:
    """Analyze performance characteristics from IR"""
    
    def analyze(self, ir, target_arch):
        """Analyze IR performance"""
        metrics = {}
        
        # Compute intensity
        metrics["flops"] = self.count_flops(ir)
        metrics["memory_ops"] = self.count_memory_ops(ir)
        metrics["compute_intensity"] = metrics["flops"] / metrics["memory_ops"]
        
        # Memory analysis
        metrics["shared_memory_usage"] = self.analyze_shared_memory(ir)
        metrics["register_pressure"] = self.analyze_register_pressure(ir)
        
        # Parallelism analysis
        metrics["parallelism"] = self.analyze_parallelism(ir)
        metrics["occupancy"] = self.estimate_occupancy(ir, target_arch)
        
        # Bottleneck analysis
        metrics["bottleneck"] = self.identify_bottleneck(metrics)
        
        return metrics
    
    def count_flops(self, ir):
        """Count floating point operations"""
        flops = 0
        
        for op in ir.operations:
            if op.type == "matmul":
                # 2*M*N*K flops for matmul
                m, n, k = op.get_dimensions()
                flops += 2 * m * n * k
            elif op.type in ["add", "mul", "sub", "div"]:
                flops += op.num_elements
            elif op.type == "mma":
                # Tensor core operations
                flops += op.mma_flops
        
        return flops
    
    def identify_bottleneck(self, metrics):
        """Identify performance bottleneck"""
        if metrics["compute_intensity"] < 10:
            return "memory_bound"
        elif metrics["occupancy"] < 0.5:
            return "latency_bound"
        else:
            return "compute_bound"
```

## Custom Passes

### Creating Custom Optimization Passes

```python
class CustomPass:
    """Base class for custom optimization passes"""
    
    def __init__(self, name):
        self.name = name
        self.statistics = {}
    
    def run(self, ir):
        """Run the optimization pass"""
        raise NotImplementedError
    
    def report_statistics(self):
        """Report pass statistics"""
        return self.statistics

class MyCustomFusionPass(CustomPass):
    """Example custom fusion pass"""
    
    def __init__(self):
        super().__init__("MyCustomFusion")
    
    def run(self, graph_ir):
        """Run custom fusion logic"""
        fused_count = 0
        
        # Look for specific pattern
        for i in range(len(graph_ir.ops) - 2):
            op1 = graph_ir.ops[i]
            op2 = graph_ir.ops[i + 1]
            op3 = graph_ir.ops[i + 2]
            
            # Pattern: LayerNorm -> GEMM -> GELU
            if (op1.type == "layernorm" and 
                op2.type == "matmul" and 
                op3.type == "gelu"):
                
                # Create fused operation
                fused_op = self.create_fused_op(op1, op2, op3)
                
                # Replace in graph
                graph_ir.replace_sequence([op1, op2, op3], fused_op)
                fused_count += 1
        
        self.statistics["fusions"] = fused_count
        return graph_ir
    
    def create_fused_op(self, ln_op, gemm_op, gelu_op):
        """Create fused LayerNorm-GEMM-GELU operation"""
        return GraphOp(
            type="fused_ln_gemm_gelu",
            inputs=[ln_op.inputs[0], gemm_op.inputs[1]],
            outputs=gelu_op.outputs,
            attributes={
                "epsilon": ln_op.attributes.get("epsilon", 1e-5),
                "use_bias": gemm_op.attributes.get("use_bias", False)
            }
        )

# Register custom pass
ts.register_pass(MyCustomFusionPass(), level="graph", priority=10)
```

### Custom Lowering Rules

```python
class CustomLoweringRules:
    """Define custom lowering rules between IR levels"""
    
    @staticmethod
    def lower_custom_op(op, target_level):
        """Lower custom operation to target level"""
        
        if target_level == "schedule":
            return CustomLoweringRules.lower_to_schedule(op)
        elif target_level == "tile":
            return CustomLoweringRules.lower_to_tile(op)
        elif target_level == "target":
            return CustomLoweringRules.lower_to_target(op)
    
    @staticmethod
    def lower_to_tile(op):
        """Lower custom op to Tile IR"""
        
        if op.type == "fused_ln_gemm_gelu":
            # Generate Tile IR for fused operation
            tile_ops = []
            
            # LayerNorm part
            tile_ops.append(TileOp("compute_mean", op.inputs[0]))
            tile_ops.append(TileOp("compute_variance", op.inputs[0]))
            tile_ops.append(TileOp("normalize", op.inputs[0]))
            
            # GEMM part
            tile_ops.append(TileOp("tile_gemm", [op.inputs[0], op.inputs[1]]))
            
            # GELU part
            tile_ops.append(TileOp("gelu_activation", op.outputs[0]))
            
            return tile_ops
        
        return None

# Register custom lowering rule
ts.register_lowering_rule("fused_ln_gemm_gelu", CustomLoweringRules.lower_custom_op)
```

### Pass Manager Configuration

```python
class PassManager:
    """Manage optimization passes"""
    
    def __init__(self):
        self.passes = {
            "graph": [],
            "schedule": [],
            "tile": [],
            "target": []
        }
        self.config = {}
    
    def add_pass(self, pass_obj, level, priority=50):
        """Add optimization pass"""
        self.passes[level].append({
            "pass": pass_obj,
            "priority": priority
        })
        # Sort by priority
        self.passes[level].sort(key=lambda x: x["priority"])
    
    def configure(self, **kwargs):
        """Configure pass manager"""
        self.config.update(kwargs)
    
    def run_passes(self, ir, level):
        """Run all passes for a level"""
        result = ir
        
        for pass_info in self.passes[level]:
            pass_obj = pass_info["pass"]
            
            # Check if pass should run
            if self.should_run_pass(pass_obj, level):
                result = pass_obj.run(result)
                
                # Validate after each pass
                if self.config.get("validate", False):
                    validator = IRDebugger()
                    if not validator.validate_ir(result, level):
                        raise RuntimeError(f"Pass {pass_obj.name} produced invalid IR")
        
        return result
    
    def should_run_pass(self, pass_obj, level):
        """Check if pass should run"""
        # Check optimization level
        opt_level = self.config.get("optimization_level", 2)
        if hasattr(pass_obj, "min_opt_level"):
            if opt_level < pass_obj.min_opt_level:
                return False
        
        # Check target architecture
        target = self.config.get("target", "cuda")
        if hasattr(pass_obj, "supported_targets"):
            if target not in pass_obj.supported_targets:
                return False
        
        return True

# Example configuration
pm = PassManager()
pm.configure(
    optimization_level=3,
    target="cuda",
    validate=True,
    profile=True
)

# Add passes
pm.add_pass(DeadCodeElimination(), "graph", priority=10)
pm.add_pass(OperatorFusion(), "graph", priority=20)
pm.add_pass(MyCustomFusionPass(), "graph", priority=25)
pm.add_pass(LoopTiling(), "schedule", priority=10)
pm.add_pass(BankConflictElimination(), "tile", priority=10)
```

## Examples

### Complete Compilation Example

```python
# Example: Compile a transformer block
@ts.compile(target="cuda", optimization_level=3)
def transformer_block(x, W_qkv, W_o):
    """Transformer block to demonstrate compilation"""
    
    # Multi-head attention
    qkv = ts.matmul(x, W_qkv)
    q, k, v = ts.split(qkv, 3, dim=-1)
    
    # Reshape for heads
    q = ts.reshape(q, (batch, heads, seq, dim))
    k = ts.reshape(k, (batch, heads, seq, dim))
    v = ts.reshape(v, (batch, heads, seq, dim))
    
    # Attention
    scores = ts.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    attn = ts.softmax(scores, dim=-1)
    out = ts.matmul(attn, v)
    
    # Output projection
    out = ts.reshape(out, (batch, seq, hidden))
    out = ts.matmul(out, W_o)
    
    return out

# Inspect compilation stages
inspector = IRInspector()

# View Graph IR
graph_ir = inspector.inspect(transformer_block, level="graph")
print("Graph IR:")
print(graph_ir)

# View Schedule IR  
schedule_ir = inspector.inspect(transformer_block, level="schedule")
print("\nSchedule IR:")
print(schedule_ir)

# View Tile IR
tile_ir = inspector.inspect(transformer_block, level="tile")
print("\nTile IR:")
print(tile_ir)

# View Target IR (PTX)
target_ir = inspector.inspect(transformer_block, level="target")
print("\nTarget IR (PTX):")
print(target_ir)

# Analyze performance
analyzer = IRPerformanceAnalyzer()
metrics = analyzer.analyze(tile_ir, "sm_80")
print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

### Custom Compilation Pipeline

```python
class MyCompiler(TesseraCompiler):
    """Custom compiler with additional passes"""
    
    def __init__(self):
        super().__init__()
        
        # Add custom passes
        self.add_graph_pass(MyCustomFusionPass())
        self.add_tile_pass(MyBankConflictOptimizer())
    
    def optimize_for_my_hardware(self, ir):
        """Hardware-specific optimizations"""
        # Custom optimization logic
        return ir

# Use custom compiler
compiler = MyCompiler()
compiled = compiler.compile(
    source_code=my_kernel,
    target="my_accelerator",
    optimization_level=3
)
```

## Best Practices

1. **IR Level Selection**
   - Use Graph IR for algorithmic optimizations
   - Use Schedule IR for parallelization and tiling
   - Use Tile IR for memory and synchronization
   - Use Target IR for architecture-specific code

2. **Pass Ordering**
   - Run cleanup passes first (DCE, CSE)
   - Apply fusion before scheduling
   - Apply memory optimizations after tiling
   - Run target-specific passes last

3. **Debugging**
   - Validate IR after each major transformation
   - Use step-through debugging for complex passes
   - Profile pass execution time
   - Keep statistics on transformations

4. **Performance**
   - Focus on memory bandwidth optimization
   - Maximize parallelism at all levels
   - Use architecture-specific features
   - Profile and iterate

## Conclusion

The Tessera IR stack provides a powerful framework for compiling high-level code to efficient machine code. Through progressive lowering and targeted optimizations at each level, it achieves both portability and performance across diverse hardware architectures.