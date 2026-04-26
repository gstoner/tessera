# PDDL-Instruct for Tessera Programming Model
## Document 1: Architecture and Core Framework

This document introduces **Tessera-PDDL**, an adaptation of the PDDL-Instruct framework specifically designed for GPU kernel development within the Tessera programming model. Our approach leverages logical chain-of-thought reasoning to teach LLMs to plan and optimize GPU kernel implementations, from high-level mathematical specifications down to hardware-specific optimizations.

---

## 1. Executive Summary

The Tessera-PDDL framework extends the original PDDL-Instruct approach to address the unique challenges of GPU kernel development:

- **Domain-Specific Planning**: Adaptation from general symbolic planning to GPU computation planning
- **Multi-Level Reasoning**: Chain-of-thought reasoning across Tessera's IR stack (Graph → Schedule → Tile → Target)
- **Hardware-Aware Optimization**: Integration of hardware capabilities and constraints into planning
- **Performance-Driven Goals**: Planning objectives focused on throughput, occupancy, and efficiency

### Key Innovations

1. **GPU Domain Definition Language (GDDL)**: A PDDL-inspired formal language for GPU computation
2. **Optimization Chain-of-Thought**: Structured reasoning about performance transformations
3. **Hardware State Modeling**: Formal representation of GPU architecture states
4. **Verification-Driven Planning**: Self-correction through performance validation

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tessera-PDDL Framework                           │
├─────────────────────────────────────────────────────────────────────┤
│  High-Level Specification (Python/Tessera DSL)                    │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                GDDL Domain Modeling                          │   │
│  │  • Computation Primitives    • Memory Hierarchies           │   │
│  │  • Action Schemas            • Hardware Constraints         │   │
│  │  • State Representations     • Performance Metrics          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Chain-of-Thought Planner                       │   │
│  │  • Logical Reasoning Engine  • Optimization Strategies      │   │
│  │  • State Transition Logic    • Performance Prediction       │   │
│  │  • Action Applicability      • Error Correction             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            Multi-Level Code Generation                       │   │
│  │  Graph IR → Schedule IR → Tile IR → Target IR              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │           Verification and Refinement                       │   │
│  │  • Performance Testing       • Numerical Validation         │   │
│  │  • Correctness Verification  • Feedback Integration         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Principles

1. **Symbolic Reasoning**: Formal representation of GPU computation problems
2. **Logical Inference**: Step-by-step reasoning about optimization transformations
3. **State-Space Navigation**: Systematic exploration of optimization possibilities
4. **Performance-Guided Search**: Goal-directed planning toward optimal implementations

---

## 3. GPU Domain Definition Language (GDDL)

### 3.1 Domain Structure

GDDL adapts PDDL concepts to GPU kernel development:

```gddl
(define (domain gpu-kernel-optimization)
  (:requirements 
    :strips :typing :conditional-effects :fluents :preferences
    :timed-initial-literals :numeric-fluents :action-costs
  )
  
  (:types
    computation-node - object
    memory-region - object  
    hardware-resource - object
    optimization-pass - object
    performance-metric - object
    ir-representation - object
  )
  
  (:predicates
    ;; Computation Structure
    (matrix-multiply ?comp - computation-node)
    (reduction-operation ?comp - computation-node)
    (elementwise-operation ?comp - computation-node)
    (memory-bound ?comp - computation-node)
    (compute-bound ?comp - computation-node)
    
    ;; Memory Hierarchy
    (in-register ?data - memory-region)
    (in-shared-memory ?data - memory-region)
    (in-global-memory ?data - memory-region)
    (cached ?data - memory-region)
    (coalesced-access ?data - memory-region)
    
    ;; Hardware Resources
    (tensor-core-available ?hw - hardware-resource)
    (async-copy-available ?hw - hardware-resource)
    (warp-shuffle-available ?hw - hardware-resource)
    (shared-memory-capacity ?hw - hardware-resource ?size - number)
    (register-file-capacity ?hw - hardware-resource ?regs - number)
    
    ;; Optimization State
    (applied ?pass - optimization-pass)
    (occupancy-level ?hw - hardware-resource ?level - number)
    (bandwidth-utilization ?hw - hardware-resource ?util - number)
    (compute-utilization ?hw - hardware-resource ?util - number)
  )
  
  (:functions
    ;; Performance Metrics
    (execution-time ?comp - computation-node) - number
    (memory-bandwidth ?comp - computation-node) - number
    (flops-achieved ?comp - computation-node) - number
    (occupancy ?comp - computation-node) - number
    (register-pressure ?comp - computation-node) - number
    
    ;; Resource Usage
    (shared-memory-used ?comp - computation-node) - number
    (registers-per-thread ?comp - computation-node) - number
    (warps-per-sm ?comp - computation-node) - number
  )
  
  ;; Actions defined in next section...
)
```

### 3.2 Action Schemas

GDDL actions represent optimization transformations:

```gddl
;; Tensor Core Optimization
(:action apply-tensor-core-optimization
  :parameters (?comp - computation-node ?hw - hardware-resource)
  :precondition (and
    (matrix-multiply ?comp)
    (tensor-core-available ?hw)
    (>= (matrix-size ?comp) 16)  ; Minimum tile size
    (not (applied tensor-core-opt))
  )
  :effect (and
    (applied tensor-core-opt)
    (increase (flops-achieved ?comp) (* (flops-achieved ?comp) 8))
    (decrease (execution-time ?comp) (/ (execution-time ?comp) 4))
    (compute-bound ?comp)
  )
)

;; Memory Coalescing
(:action apply-memory-coalescing
  :parameters (?data - memory-region ?comp - computation-node)
  :precondition (and
    (in-global-memory ?data)
    (not (coalesced-access ?data))
    (memory-bound ?comp)
  )
  :effect (and
    (coalesced-access ?data)
    (increase (memory-bandwidth ?comp) (* (memory-bandwidth ?comp) 3))
    (decrease (execution-time ?comp) (* (execution-time ?comp) 0.4))
  )
)

;; Async Copy Pipeline
(:action apply-async-copy-pipeline
  :parameters (?comp - computation-node ?hw - hardware-resource)
  :precondition (and
    (async-copy-available ?hw)
    (memory-bound ?comp)
    (>= (data-size ?comp) 1024)  ; Minimum size for async benefit
  )
  :effect (and
    (applied async-copy-pipeline)
    (decrease (execution-time ?comp) (* (execution-time ?comp) 0.7))
    (increase (occupancy ?comp) (* (occupancy ?comp) 1.2))
  )
)

;; Shared Memory Tiling
(:action apply-shared-memory-tiling
  :parameters (?comp - computation-node ?data - memory-region ?hw - hardware-resource)
  :precondition (and
    (matrix-multiply ?comp)
    (in-global-memory ?data)
    (<= (tile-size ?comp) (shared-memory-capacity ?hw))
  )
  :effect (and
    (in-shared-memory ?data)
    (not (in-global-memory ?data))
    (decrease (execution-time ?comp) (* (execution-time ?comp) 0.6))
    (increase (shared-memory-used ?comp) (tile-size ?comp))
  )
)

;; Loop Unrolling
(:action apply-loop-unrolling
  :parameters (?comp - computation-node ?factor - number)
  :precondition (and
    (has-loop ?comp)
    (<= (* (registers-per-thread ?comp) ?factor) 200)  ; Register limit
  )
  :effect (and
    (applied loop-unrolling)
    (increase (registers-per-thread ?comp) (* (registers-per-thread ?comp) ?factor))
    (decrease (execution-time ?comp) (* (execution-time ?comp) (/ 1 ?factor)))
    (increase (compute-utilization ?comp) (* (compute-utilization ?comp) 1.3))
  )
)
```

---

## 4. Chain-of-Thought Reasoning Framework

### 4.1 Logical Reasoning Process

The framework uses structured chain-of-thought reasoning to navigate optimization decisions:

```python
class GPUOptimizationChainOfThought:
    def __init__(self, domain_knowledge, hardware_spec):
        self.domain = domain_knowledge
        self.hardware = hardware_spec
        self.reasoning_trace = []
    
    def reason_about_optimization(self, kernel_spec, performance_goal):
        """
        Main reasoning loop following PDDL-Instruct methodology
        """
        # Step 1: Problem Analysis
        problem_analysis = self.analyze_kernel_characteristics(kernel_spec)
        self.log_reasoning(f"Analysis: {problem_analysis}")
        
        # Step 2: State Space Construction
        initial_state = self.construct_initial_state(kernel_spec)
        goal_state = self.construct_goal_state(performance_goal)
        self.log_reasoning(f"Initial State: {initial_state}")
        self.log_reasoning(f"Goal State: {goal_state}")
        
        # Step 3: Action Applicability Reasoning
        applicable_actions = self.determine_applicable_actions(initial_state)
        self.log_reasoning(f"Applicable Actions: {applicable_actions}")
        
        # Step 4: Effect Prediction
        for action in applicable_actions:
            predicted_effects = self.predict_action_effects(action, initial_state)
            self.log_reasoning(f"Action {action.name} → Effects: {predicted_effects}")
        
        # Step 5: Plan Construction
        optimization_plan = self.construct_optimization_plan(
            initial_state, goal_state, applicable_actions)
        
        return optimization_plan
    
    def analyze_kernel_characteristics(self, kernel_spec):
        """
        Analyze kernel to determine fundamental characteristics
        """
        analysis = {
            "computation_pattern": self.identify_computation_pattern(kernel_spec),
            "memory_access_pattern": self.analyze_memory_access(kernel_spec),
            "bottleneck_type": self.identify_bottleneck(kernel_spec),
            "hardware_features": self.identify_useful_hardware_features(kernel_spec)
        }
        
        # Chain-of-thought reasoning for each characteristic
        reasoning = self.reason_about_characteristics(analysis)
        return reasoning
    
    def reason_about_characteristics(self, analysis):
        """
        Structured reasoning about kernel characteristics
        """
        reasoning = []
        
        # Reasoning about computation pattern
        if analysis["computation_pattern"] == "matrix_multiply":
            reasoning.append({
                "thought": "Kernel performs matrix multiplication",
                "implication": "Can benefit from tensor cores",
                "condition": "Matrix dimensions >= 16x16",
                "action_suggested": "apply-tensor-core-optimization"
            })
        
        # Reasoning about memory access
        if analysis["memory_access_pattern"] == "strided":
            reasoning.append({
                "thought": "Memory access is strided (non-coalesced)",
                "implication": "Memory bandwidth is underutilized",
                "condition": "Can be restructured for coalescing",
                "action_suggested": "apply-memory-coalescing"
            })
        
        # Reasoning about bottleneck
        if analysis["bottleneck_type"] == "memory_bound":
            reasoning.append({
                "thought": "Kernel is memory-bound",
                "implication": "Computation resources are underutilized",
                "condition": "Can overlap memory and compute",
                "action_suggested": "apply-async-copy-pipeline"
            })
        
        return reasoning
```

### 4.2 State Transition Logic

```python
class StateTransitionReasoner:
    def __init__(self):
        self.transition_rules = self.load_transition_rules()
    
    def apply_logical_reasoning(self, current_state, action):
        """
        Apply logical reasoning to determine state transitions
        """
        # Check preconditions
        precondition_check = self.verify_preconditions(current_state, action)
        if not precondition_check.valid:
            return ReasoningResult(
                valid=False, 
                reason=f"Precondition failed: {precondition_check.failed_conditions}"
            )
        
        # Predict effects
        effects = self.predict_effects(current_state, action)
        new_state = self.apply_effects(current_state, effects)
        
        # Validate invariants
        invariant_check = self.verify_invariants(new_state)
        if not invariant_check.valid:
            return ReasoningResult(
                valid=False,
                reason=f"Invariant violation: {invariant_check.violations}"
            )
        
        return ReasoningResult(
            valid=True,
            new_state=new_state,
            reasoning_trace=self.get_reasoning_trace()
        )
    
    def verify_preconditions(self, state, action):
        """
        Verify action preconditions with logical reasoning
        """
        reasoning_trace = []
        
        for precondition in action.preconditions:
            if precondition.type == "resource_availability":
                available = self.check_resource_availability(state, precondition)
                reasoning_trace.append({
                    "condition": precondition,
                    "result": available,
                    "reasoning": f"Hardware resource {precondition.resource} available: {available}"
                })
            
            elif precondition.type == "numeric_constraint":
                satisfied = self.evaluate_numeric_constraint(state, precondition)
                reasoning_trace.append({
                    "condition": precondition,
                    "result": satisfied,
                    "reasoning": f"Numeric constraint {precondition.constraint} satisfied: {satisfied}"
                })
        
        all_satisfied = all(step["result"] for step in reasoning_trace)
        return PreconditionResult(valid=all_satisfied, trace=reasoning_trace)
```

---

## 5. Integration with Tessera IR Stack

### 5.1 Multi-Level Planning

The framework integrates with Tessera's multi-level IR system:

```python
class TesseraLevelPlanner:
    """
    Plans optimizations across Tessera's IR levels
    """
    
    def __init__(self):
        self.level_planners = {
            "graph_ir": GraphIRPlanner(),
            "schedule_ir": ScheduleIRPlanner(),
            "tile_ir": TileIRPlanner(),
            "target_ir": TargetIRPlanner()
        }
    
    def plan_multi_level_optimization(self, initial_spec, performance_target):
        """
        Plan optimizations across all IR levels
        """
        optimization_plan = MultiLevelPlan()
        
        # Graph IR Level Planning
        graph_optimizations = self.plan_graph_level(initial_spec)
        optimization_plan.add_level("graph_ir", graph_optimizations)
        
        # Schedule IR Level Planning
        schedule_optimizations = self.plan_schedule_level(
            graph_optimizations.output_spec)
        optimization_plan.add_level("schedule_ir", schedule_optimizations)
        
        # Tile IR Level Planning
        tile_optimizations = self.plan_tile_level(
            schedule_optimizations.output_spec)
        optimization_plan.add_level("tile_ir", tile_optimizations)
        
        # Target IR Level Planning
        target_optimizations = self.plan_target_level(
            tile_optimizations.output_spec, performance_target)
        optimization_plan.add_level("target_ir", target_optimizations)
        
        return optimization_plan
    
    def plan_graph_level(self, spec):
        """
        Plan Graph IR optimizations using GDDL reasoning
        """
        # Graph-level optimization actions
        actions = [
            "fuse-elementwise-operations",
            "optimize-memory-layout", 
            "insert-padding-for-vectorization",
            "schedule-collective-operations"
        ]
        
        # Apply chain-of-thought reasoning
        planner = self.level_planners["graph_ir"]
        return planner.plan_optimizations(spec, actions)
    
    def plan_schedule_level(self, spec):
        """
        Plan Schedule IR optimizations
        """
        # Schedule-level optimization actions
        actions = [
            "apply-loop-tiling",
            "optimize-memory-scheduling",
            "insert-prefetch-operations",
            "apply-loop-fusion"
        ]
        
        planner = self.level_planners["schedule_ir"]
        return planner.plan_optimizations(spec, actions)
    
    def plan_tile_level(self, spec):
        """
        Plan Tile IR optimizations
        """
        # Tile-level optimization actions
        actions = [
            "apply-tensor-core-mapping",
            "optimize-shared-memory-layout",
            "apply-async-copy-scheduling",
            "optimize-barrier-placement"
        ]
        
        planner = self.level_planners["tile_ir"]
        return planner.plan_optimizations(spec, actions)
    
    def plan_target_level(self, spec, performance_target):
        """
        Plan Target IR optimizations
        """
        # Target-level optimization actions
        actions = [
            "select-optimal-instruction-sequence",
            "optimize-register-allocation",
            "apply-architecture-specific-optimizations",
            "generate-optimal-memory-access-patterns"
        ]
        
        planner = self.level_planners["target_ir"]
        return planner.plan_optimizations(spec, actions, performance_target)
```

### 5.2 Hardware-Aware Planning

```python
class HardwareAwarePlanner:
    """
    Incorporates hardware capabilities into planning process
    """
    
    def __init__(self, hardware_spec):
        self.hardware = hardware_spec
        self.capability_model = self.build_capability_model()
    
    def build_capability_model(self):
        """
        Build formal model of hardware capabilities
        """
        capabilities = {}
        
        # Tensor Core Capabilities
        if self.hardware.has_tensor_cores:
            capabilities["tensor_cores"] = {
                "supported_types": ["bf16", "fp16", "int8", "fp8"],
                "min_dimensions": (16, 16, 16),
                "max_throughput": "312 TFLOPS",  # H100 example
                "applicable_operations": ["matrix_multiply", "convolution"]
            }
        
        # Memory Hierarchy
        capabilities["memory"] = {
            "shared_memory_size": self.hardware.shared_memory_per_sm,
            "max_bandwidth": self.hardware.memory_bandwidth,
            "cache_hierarchy": self.hardware.cache_levels,
            "async_copy_support": self.hardware.supports_async_copy
        }
        
        # Compute Capabilities
        capabilities["compute"] = {
            "cuda_cores": self.hardware.cuda_cores_per_sm,
            "special_functions": self.hardware.special_function_units,
            "warp_size": 32,
            "max_occupancy": self.hardware.max_warps_per_sm
        }
        
        return capabilities
    
    def evaluate_action_applicability(self, action, current_state):
        """
        Evaluate if action is applicable given hardware constraints
        """
        if action.name == "apply-tensor-core-optimization":
            return self.evaluate_tensor_core_applicability(action, current_state)
        elif action.name == "apply-async-copy-pipeline":
            return self.evaluate_async_copy_applicability(action, current_state)
        elif action.name == "apply-shared-memory-tiling":
            return self.evaluate_shared_memory_applicability(action, current_state)
        
        return ApplicabilityResult(applicable=True)
    
    def evaluate_tensor_core_applicability(self, action, state):
        """
        Check if tensor core optimization is applicable
        """
        reasoning = []
        
        # Check hardware support
        if not self.capability_model["tensor_cores"]:
            reasoning.append("Hardware does not support tensor cores")
            return ApplicabilityResult(applicable=False, reasoning=reasoning)
        
        # Check data types
        kernel_dtype = state.get_primary_dtype()
        supported_types = self.capability_model["tensor_cores"]["supported_types"]
        if kernel_dtype not in supported_types:
            reasoning.append(f"Data type {kernel_dtype} not supported by tensor cores")
            return ApplicabilityResult(applicable=False, reasoning=reasoning)
        
        # Check matrix dimensions
        matrix_dims = state.get_matrix_dimensions()
        min_dims = self.capability_model["tensor_cores"]["min_dimensions"]
        if any(d < min_d for d, min_d in zip(matrix_dims, min_dims)):
            reasoning.append(f"Matrix dimensions {matrix_dims} below minimum {min_dims}")
            return ApplicabilityResult(applicable=False, reasoning=reasoning)
        
        reasoning.append("All tensor core requirements satisfied")
        return ApplicabilityResult(applicable=True, reasoning=reasoning)
```

---

## 6. Self-Correction and Refinement

### 6.1 Verification-Driven Planning

```python
class VerificationDrivenPlanner:
    """
    Implements self-correction through verification feedback
    """
    
    def __init__(self):
        self.verification_suite = VerificationSuite()
        self.refinement_strategies = RefinementStrategies()
    
    def plan_with_verification(self, initial_plan):
        """
        Iteratively refine plan based on verification results
        """
        current_plan = initial_plan
        iteration = 0
        max_iterations = 5
        
        while iteration < max_iterations:
            # Generate code from current plan
            generated_code = self.generate_code_from_plan(current_plan)
            
            # Verify generated code
            verification_result = self.verification_suite.verify(generated_code)
            
            if verification_result.all_passed:
                return PlanResult(plan=current_plan, verified=True)
            
            # Self-correct based on verification failures
            correction_reasoning = self.reason_about_failures(verification_result)
            corrected_plan = self.apply_corrections(current_plan, correction_reasoning)
            
            current_plan = corrected_plan
            iteration += 1
        
        return PlanResult(plan=current_plan, verified=False, max_iterations_reached=True)
    
    def reason_about_failures(self, verification_result):
        """
        Apply logical reasoning to understand verification failures
        """
        reasoning = []
        
        for failure in verification_result.failures:
            if failure.type == "performance_regression":
                reasoning.append({
                    "issue": "Performance worse than baseline",
                    "cause": failure.details,
                    "correction": "Reduce optimization aggressiveness",
                    "action": "rollback-last-optimization"
                })
            
            elif failure.type == "correctness_error":
                reasoning.append({
                    "issue": "Numerical accuracy degraded",
                    "cause": failure.details,
                    "correction": "Use higher precision accumulation",
                    "action": "increase-precision-policy"
                })
            
            elif failure.type == "resource_overflow":
                reasoning.append({
                    "issue": "Exceeded hardware resource limits",
                    "cause": failure.details,
                    "correction": "Reduce resource usage",
                    "action": "reduce-register-pressure"
                })
        
        return reasoning
    
    def apply_corrections(self, plan, correction_reasoning):
        """
        Apply corrections to the optimization plan
        """
        corrected_plan = plan.copy()
        
        for reasoning in correction_reasoning:
            action = reasoning["action"]
            
            if action == "rollback-last-optimization":
                corrected_plan = self.rollback_last_step(corrected_plan)
            elif action == "increase-precision-policy":
                corrected_plan = self.increase_precision(corrected_plan)
            elif action == "reduce-register-pressure":
                corrected_plan = self.reduce_register_usage(corrected_plan)
        
        return corrected_plan
```

---

## 7. Performance Evaluation Framework

### 7.1 Goal-Directed Planning

```python
class PerformanceGoalPlanner:
    """
    Plans optimizations toward specific performance goals
    """
    
    def __init__(self):
        self.performance_model = PerformanceModel()
        self.goal_decomposer = GoalDecomposer()
    
    def plan_for_performance_goal(self, kernel_spec, performance_goal):
        """
        Plan optimizations to achieve specific performance targets
        """
        # Decompose high-level goal into specific metrics
        metric_goals = self.goal_decomposer.decompose(performance_goal)
        
        # Example: "Achieve 80% of peak FLOPS" becomes:
        # - Maximize tensor core utilization
        # - Minimize memory bandwidth waste  
        # - Maintain high occupancy (>75%)
        # - Minimize instruction overhead
        
        optimization_plan = OptimizationPlan()
        
        for metric_goal in metric_goals:
            # Plan optimizations for each metric
            metric_plan = self.plan_for_metric(kernel_spec, metric_goal)
            optimization_plan.merge(metric_plan)
        
        # Resolve conflicts between different metric optimizations
        resolved_plan = self.resolve_conflicts(optimization_plan)
        
        return resolved_plan
    
    def plan_for_metric(self, kernel_spec, metric_goal):
        """
        Plan optimizations for a specific performance metric
        """
        if metric_goal.metric == "compute_utilization":
            return self.plan_compute_optimization(kernel_spec, metric_goal.target)
        elif metric_goal.metric == "memory_bandwidth":
            return self.plan_memory_optimization(kernel_spec, metric_goal.target)
        elif metric_goal.metric == "occupancy":
            return self.plan_occupancy_optimization(kernel_spec, metric_goal.target)
    
    def plan_compute_optimization(self, spec, target_utilization):
        """
        Plan optimizations to maximize compute utilization
        """
        actions = []
        
        # Chain-of-thought reasoning for compute optimization
        if self.has_matrix_operations(spec):
            actions.append({
                "action": "apply-tensor-core-optimization",
                "reasoning": "Matrix operations can benefit from tensor cores",
                "expected_impact": "8x compute throughput increase"
            })
        
        if self.has_elementwise_operations(spec):
            actions.append({
                "action": "apply-vectorization",
                "reasoning": "Elementwise ops can be vectorized",
                "expected_impact": "4x throughput increase"
            })
        
        if self.has_underutilized_warps(spec):
            actions.append({
                "action": "increase-occupancy",
                "reasoning": "More warps hide latency better",
                "expected_impact": "Better compute utilization"
            })
        
        return ComputeOptimizationPlan(actions)
```

---

## 8. Integration Points and APIs

### 8.1 Tessera Integration

```python
class TesseraPDDLIntegration:
    """
    Main integration point with Tessera compiler infrastructure
    """
    
    def __init__(self, tessera_context):
        self.tessera = tessera_context
        self.pddl_planner = TesseraPDDLPlanner()
        self.code_generator = TesseraCodeGenerator()
    
    @tessera.planning_pass
    def apply_pddl_optimization(self, module, optimization_level="O3"):
        """
        Apply PDDL-based optimization planning to Tessera module
        """
        # Extract kernel specifications
        kernel_specs = self.extract_kernel_specs(module)
        
        # Plan