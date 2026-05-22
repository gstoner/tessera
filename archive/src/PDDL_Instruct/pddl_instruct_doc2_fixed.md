# PDDL-Instruct for Tessera Programming Model - Document 2: Domain Modeling

This document explores how PDDL-Instruct can be applied to model GPU kernel domains in Tessera, providing structured approaches to define computational problems, constraints, and optimization objectives for automated kernel generation.

## Overview of Domain Modeling in Tessera

Domain modeling in the context of Tessera kernel development involves:

1. **Problem Space Definition**: Characterizing the computational problem structure
2. **Resource Constraints**: GPU memory hierarchy, compute units, and architectural limits  
3. **Performance Objectives**: Throughput, latency, energy efficiency, and numerical accuracy
4. **Implementation Strategies**: Tiling patterns, memory layouts, and parallelization schemes

PDDL-Instruct provides a formal framework to express these domains systematically.

## Core Domain Components

### 2.1 State Space Definition

In Tessera kernel domains, the state space encompasses:

```pddl
(define (domain tessera-kernel-optimization)
  (:requirements :strips :typing :fluents :derived-predicates)
  
  (:types
    tensor - object
    memory-tier - object
    compute-unit - object
    layout-pattern - object
    tile-size - object
    numerical-precision - object
    parallelization-strategy - object
  )
  
  (:constants
    global-memory shared-memory register-memory - memory-tier
    cuda-core tensor-core - compute-unit
    row-major col-major blocked swizzled - layout-pattern
    fp32 fp16 bf16 fp8 int8 - numerical-precision
    data-parallel tensor-parallel pipeline-parallel - parallelization-strategy
  )
  
  (:predicates
    (tensor-in-memory ?t - tensor ?m - memory-tier)
    (tensor-has-layout ?t - tensor ?l - layout-pattern)
    (tensor-has-precision ?t - tensor ?p - numerical-precision)
    (compute-unit-available ?c - compute-unit)
    (memory-tier-has-capacity ?m - memory-tier ?size - number)
    (tile-fits-in-memory ?ts - tile-size ?m - memory-tier)
    (operation-uses-precision ?op - object ?p - numerical-precision)
    (layout-enables-coalescing ?l - layout-pattern ?m - memory-tier)
    (parallelization-fits-problem ?ps - parallelization-strategy ?prob-size - number)
  )
  
  (:functions
    (memory-bandwidth ?m - memory-tier) - number
    (compute-throughput ?c - compute-unit ?p - numerical-precision) - number
    (memory-usage ?t - tensor) - number
    (access-latency ?m - memory-tier) - number
    (parallelization-efficiency ?ps - parallelization-strategy) - number
    (tile-size-value ?ts - tile-size) - number
    (operation-intensity ?op - object) - number
  )
)
```

### 2.2 Memory Hierarchy Modeling

Tessera's memory hierarchy requires explicit modeling of different tiers and their characteristics:

```pddl
(:predicates
  ; Memory tier relationships and constraints
  (memory-tier-faster-than ?m1 ?m2 - memory-tier)
  (memory-tier-closer-to-compute ?m1 ?m2 - memory-tier)
  (memory-supports-broadcast ?m - memory-tier)
  (memory-supports-reduction ?m - memory-tier)
  
  ; Data movement and coherence
  (can-transfer-direct ?m1 ?m2 - memory-tier)
  (transfer-preserves-layout ?m1 ?m2 - memory-tier ?l - layout-pattern)
  (coherency-maintained ?t - tensor ?m1 ?m2 - memory-tier)
  
  ; Bank conflicts and access patterns
  (layout-avoids-bank-conflicts ?l - layout-pattern ?m - memory-tier)
  (access-pattern-coalesced ?t - tensor ?l - layout-pattern)
  (swizzle-pattern-optimal ?l - layout-pattern ?tile - tile-size)
)

(:functions
  ; Quantitative memory characteristics
  (memory-capacity ?m - memory-tier) - number
  (memory-bandwidth-peak ?m - memory-tier) - number
  (bank-conflict-penalty ?m - memory-tier ?l - layout-pattern) - number
  (transfer-cost ?m1 ?m2 - memory-tier ?size - number) - number
  (cache-hit-rate ?m - memory-tier ?pattern - layout-pattern) - number
)
```

### 2.3 Compute Unit Characterization

Different compute units have varying capabilities and optimal usage patterns:

```pddl
(:predicates
  ; Compute unit capabilities
  (compute-unit-supports-precision ?c - compute-unit ?p - numerical-precision)
  (compute-unit-supports-operation ?c - compute-unit ?op - object)
  (compute-unit-supports-vector-width ?c - compute-unit ?width - number)
  
  ; Throughput and utilization
  (compute-unit-saturated ?c - compute-unit)
  (pipeline-depth-sufficient ?c - compute-unit ?depth - number)
  (warp-utilization-optimal ?c - compute-unit ?occupancy - number)
  
  ; Tensor core specific predicates
  (tensor-core-shape-supported ?shape-m ?shape-n ?shape-k - number)
  (tensor-core-precision-combination-valid ?in-prec ?acc-prec - numerical-precision)
)

(:functions
  ; Performance characteristics
  (peak-throughput ?c - compute-unit ?p - numerical-precision) - number
  (utilization-efficiency ?c - compute-unit ?workload - number) - number
  (pipeline-latency ?c - compute-unit ?op - object) - number
  (energy-per-operation ?c - compute-unit ?op - object) - number
)
```

## Problem-Specific Domain Extensions

### 2.4 Matrix Multiplication Domain

For matrix multiplication kernels, we extend the base domain:

```pddl
(define (domain tessera-gemm)
  (:extends tessera-kernel-optimization)
  
  (:types
    matrix - tensor
    gemm-algorithm - object
  )
  
  (:constants
    naive-gemm blocked-gemm wmma-gemm wgmma-gemm - gemm-algorithm
  )
  
  (:predicates
    ; Matrix properties
    (matrix-dimensions ?m - matrix ?rows ?cols - number)
    (matrix-transpose-required ?m - matrix)
    (matrix-sparse ?m - matrix ?sparsity - number)
    
    ; GEMM algorithm applicability
    (algorithm-supports-shapes ?alg - gemm-algorithm ?m ?n ?k - number)
    (algorithm-requires-precision ?alg - gemm-algorithm ?p - numerical-precision)
    (algorithm-optimal-for-size ?alg - gemm-algorithm ?prob-size - number)
    
    ; Tiling and blocking
    (tile-shape-compatible ?tm ?tn ?tk - number ?alg - gemm-algorithm)
    (blocking-reduces-memory-traffic ?tm ?tn ?tk - number)
    (tile-fits-shared-memory ?tm ?tn ?tk - number)
  )
  
  (:functions
    ; Performance modeling
    (gemm-theoretical-flops ?m ?n ?k - number) - number
    (algorithm-efficiency ?alg - gemm-algorithm ?m ?n ?k - number) - number
    (memory-traffic-volume ?alg - gemm-algorithm ?m ?n ?k - number) - number
    (arithmetic-intensity ?alg - gemm-algorithm ?m ?n ?k - number) - number
  )
)
```

### 2.5 Attention Mechanism Domain

Flash Attention requires specialized domain modeling:

```pddl
(define (domain tessera-attention)
  (:extends tessera-kernel-optimization)
  
  (:types
    attention-tensor - tensor
    attention-algorithm - object
    softmax-strategy - object
  )
  
  (:constants
    query key value output - attention-tensor
    flash-attention standard-attention sparse-attention - attention-algorithm
    online-softmax safe-softmax fused-softmax - softmax-strategy
  )
  
  (:predicates
    ; Attention-specific properties
    (sequence-length ?seq-len - number)
    (attention-heads ?num-heads - number)
    (head-dimension ?head-dim - number)
    (causal-mask-required)
    
    ; Algorithm constraints
    (algorithm-supports-sequence-length ?alg - attention-algorithm ?seq-len - number)
    (algorithm-memory-efficient ?alg - attention-algorithm)
    (softmax-numerically-stable ?strat - softmax-strategy)
    
    ; Tiling for attention
    (attention-tile-valid ?q-tile ?k-tile - number)
    (tile-enables-recomputation ?q-tile ?k-tile - number)
    (block-sparse-pattern-applicable ?sparsity-pattern - object)
  )
  
  (:functions
    ; Attention-specific metrics
    (attention-memory-complexity ?alg - attention-algorithm ?seq-len - number) - number
    (softmax-numerical-error ?strat - softmax-strategy ?seq-len - number) - number
    (attention-arithmetic-intensity ?seq-len ?head-dim - number) - number
  )
)
```

## Constraint Modeling

### 2.6 Resource Constraints

GPU kernels must operate within strict resource limitations:

```pddl
; Memory constraints
(define (constraint memory-capacity-limit)
  (forall (?m - memory-tier)
    (<= (sum (?t - tensor) 
           (if (tensor-in-memory ?t ?m) (memory-usage ?t) 0))
        (memory-capacity ?m))))

; Register pressure constraints  
(define (constraint register-pressure-limit)
  (forall (?thread - object)
    (<= (register-usage-per-thread ?thread) 255)))

; Occupancy constraints
(define (constraint minimum-occupancy)
  (>= (thread-block-occupancy) 0.5))

; Shared memory banking constraints
(define (constraint bank-conflict-avoidance)
  (forall (?t - tensor ?l - layout-pattern)
    (implies (and (tensor-in-memory ?t shared-memory)
                  (tensor-has-layout ?t ?l))
             (layout-avoids-bank-conflicts ?l shared-memory))))
```

### 2.7 Performance Constraints

Performance objectives translate to PDDL constraints:

```pddl
; Throughput requirements
(define (constraint minimum-throughput)
  (>= (achieved-throughput) (required-throughput)))

; Memory bandwidth utilization
(define (constraint memory-bandwidth-efficiency)
  (forall (?m - memory-tier)
    (>= (/ (utilized-bandwidth ?m) (memory-bandwidth-peak ?m)) 0.8)))

; Numerical accuracy requirements
(define (constraint numerical-precision-maintained)
  (forall (?op - object ?p - numerical-precision)
    (implies (operation-uses-precision ?op ?p)
             (<= (numerical-error ?op ?p) (acceptable-error-threshold)))))

; Energy efficiency constraints
(define (constraint energy-budget)
  (<= (total-energy-consumption) (energy-budget-limit)))
```

## Optimization Objectives

### 2.8 Multi-Objective Optimization

Tessera kernel optimization involves multiple competing objectives:

```pddl
; Performance optimization objectives
(:goal-specification
  ; Primary objective: maximize throughput
  (maximize (achieved-throughput))
  
  ; Secondary objectives with weights
  (maximize (* 0.8 (memory-bandwidth-efficiency)))
  (minimize (* 0.3 (energy-per-operation)))
  (minimize (* 0.2 (numerical-error-magnitude)))
  (maximize (* 0.6 (thread-block-occupancy)))
  
  ; Pareto front considerations
  (find-pareto-optimal-solutions 
    (throughput memory-efficiency energy-efficiency numerical-accuracy))
)
```

### 2.9 Problem-Specific Objectives

Different kernel types have specialized optimization goals:

```pddl
; GEMM-specific optimization
(define (objective gemm-optimization)
  (:parameters ?m ?n ?k - number)
  (:goal
    (and 
      ; Maximize arithmetic intensity
      (maximize (/ (gemm-theoretical-flops ?m ?n ?k) 
                  (memory-traffic-volume ?m ?n ?k)))
      
      ; Minimize memory stalls  
      (minimize (memory-stall-cycles))
      
      ; Maximize tensor core utilization
      (maximize (tensor-core-utilization-percentage))
      
      ; Ensure numerical accuracy
      (< (accumulated-numerical-error) 1e-5)
    )
  )
)

; Attention-specific optimization
(define (objective attention-optimization)
  (:parameters ?seq-len ?head-dim - number)
  (:goal
    (and
      ; Memory efficiency is critical for long sequences
      (minimize (attention-memory-complexity ?seq-len))
      
      ; Numerical stability for softmax
      (maximize (softmax-numerical-stability))
      
      ; Minimize recomputation overhead
      (minimize (recomputation-factor))
      
      ; Causal mask efficiency
      (maximize (causal-mask-efficiency))
    )
  )
)
```

## Domain Adaptation Patterns

### 2.10 Architecture-Specific Adaptations

Different GPU architectures require domain specializations:

```pddl
; Hopper-specific domain extensions
(define (domain tessera-kernel-hopper)
  (:extends tessera-kernel-optimization)
  
  (:constants
    wgmma-m64n256k32 wgmma-m128n256k32 - compute-unit
    tma-transfer async-copy - transfer-mechanism
    thread-block-cluster - parallelization-strategy
  )
  
  (:predicates
    (wgmma-shape-supported ?m ?n ?k - number)
    (tma-transfer-beneficial ?size - number)
    (cluster-mode-advantageous ?cluster-size - number)
    (distributed-shared-memory-available)
  )
  
  (:functions
    (wgmma-throughput ?m ?n ?k - number ?prec - numerical-precision) - number
    (tma-bandwidth-effective) - number
    (cluster-synchronization-overhead ?cluster-size - number) - number
  )
)

; Ampere-specific adaptations
(define (domain tessera-kernel-ampere)  
  (:extends tessera-kernel-optimization)
  
  (:constants
    wmma-m16n16k16 wmma-m32n8k16 - compute-unit
    cp-async-transfer - transfer-mechanism
  )
  
  (:predicates
    (wmma-accumulator-reuse-possible)
    (async-copy-pipeline-depth-optimal ?depth - number)
    (sparsity-acceleration-available)
  )
)
```

### 2.11 Precision-Aware Domain Modeling

Mixed precision introduces additional complexity:

```pddl
(:predicates
  ; Precision compatibility and conversion
  (precision-conversion-lossless ?p1 ?p2 - numerical-precision)
  (precision-supports-accumulation ?storage ?accumulate - numerical-precision)
  (mixed-precision-strategy-valid ?strategy - object)
  
  ; Numerical stability with mixed precision  
  (operation-numerically-stable-with-precision ?op - object ?p - numerical-precision)
  (gradient-scaling-required ?p - numerical-precision)
  (overflow-underflow-risk ?op - object ?p - numerical-precision)
)

(:functions
  ; Quantitative precision effects
  (precision-speedup-factor ?p1 ?p2 - numerical-precision) - number
  (precision-accuracy-loss ?p1 ?p2 - numerical-precision) - number
  (mixed-precision-complexity-overhead ?strategy - object) - number
)
```

## Domain Validation and Testing

### 2.12 Consistency Checking

Domain models must be validated for consistency:

```pddl
; Consistency axioms
(define (axiom memory-hierarchy-ordering)
  (and (memory-tier-faster-than register-memory shared-memory)
       (memory-tier-faster-than shared-memory global-memory)
       (memory-tier-closer-to-compute register-memory shared-memory)
       (memory-tier-closer-to-compute shared-memory global-memory)))

(define (axiom precision-ordering)
  (and (precision-conversion-lossless fp32 fp16)
       (precision-conversion-lossless fp16 bf16)
       (not (precision-conversion-lossless fp16 fp32))))

; Resource limit validation
(define (validation resource-limits-feasible)
  (forall (?config - configuration)
    (implies (configuration-valid ?config)
             (and (<= (total-shared-memory-usage ?config) 
                     (max-shared-memory-per-block))
                  (<= (registers-per-thread ?config) 255)
                  (>= (threads-per-block ?config) 32)
                  (<= (threads-per-block ?config) 1024)))))
```

### 2.13 Performance Model Validation

Ensuring performance models reflect reality:

```pddl
; Performance model consistency
(define (validation performance-model-bounds)
  (forall (?config - configuration)
    (and 
      ; Roofline model compliance
      (<= (achieved-throughput ?config)
          (min (peak-compute-throughput)
               (* (memory-bandwidth) (arithmetic-intensity ?config))))
      
      ; Occupancy limits
      (<= (thread-block-occupancy ?config) 1.0)
      (>= (thread-block-occupancy ?config) 0.0)
      
      ; Energy conservation
      (>= (energy-per-operation ?config) (theoretical-minimum-energy))
    )
  )
)
```

## Integration with Planning Systems

### 2.14 Domain-to-Problem Translation

Converting high-level kernel requirements to PDDL problems:

```pddl
(define (problem flash-attention-kernel-generation)
  (:domain tessera-attention)
  
  (:objects
    Q K V O - attention-tensor
    sm90 - architecture
    bf16 fp32 - numerical-precision
    flash-alg - attention-algorithm
    online-softmax - softmax-strategy
  )
  
  (:init
    ; Problem parameters
    (sequence-length 2048)
    (attention-heads 32)
    (head-dimension 128)
    (causal-mask-required)
    
    ; Resource constraints
    (memory-capacity shared-memory 228000)  ; 228KB shared memory
    (memory-bandwidth-peak global-memory 3000)  ; 3TB/s HBM
    
    ; Available algorithms and strategies
    (algorithm-supports-sequence-length flash-alg 2048)
    (algorithm-memory-efficient flash-alg)
    (softmax-numerically-stable online-softmax)
  )
  
  (:goal
    (and 
      ; Generate working kernel
      (kernel-implements-attention flash-alg)
      (tensor-has-precision Q bf16)
      (tensor-has-precision O bf16)
      (operation-uses-precision attention-computation fp32)
      
      ; Performance requirements
      (>= (achieved-throughput) 800)  ; 800 TFLOPS minimum
      (< (memory-usage-total) 200000)  ; Under 200KB shared memory
      (<= (numerical-error-magnitude) 1e-6)
    )
  )
)
```

This domain modeling framework provides the foundation for PDDL-Instruct to systematically reason about GPU kernel optimization problems, enabling automated generation of high-performance Tessera kernels through formal planning approaches.

## Summary

The domain modeling component of PDDL-Instruct for Tessera establishes:

1. **Comprehensive State Spaces**: Covering memory hierarchies, compute units, and precision policies
2. **Resource Constraint Modeling**: Formal representation of GPU limitations and capabilities  
3. **Multi-Objective Optimization**: Balancing performance, efficiency, and numerical accuracy
4. **Architecture Adaptation**: Specialized domains for different GPU generations
5. **Validation Frameworks**: Consistency checking and performance model validation
6. **Problem Integration**: Translation from requirements to formal planning problems

This structured approach enables systematic exploration of the kernel optimization space, providing the foundation for the chain-of-thought reasoning explored in Document 3.