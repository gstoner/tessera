# Tessera Numerical Behavior Guide
## Chapter 2: Deterministic Execution

---

### 2.1 Sources of Non-Determinism

In parallel GPU execution, results can vary across runs due to:

- **Floating-Point Non-Associativity**  
  `(a + b) + c ≠ a + (b + c)`  
  Different reduction orders yield different results.

- **Atomic Operations**  
  Atomics resolve in arbitrary thread order → nondeterministic accumulation.

- **Parallel Reductions**  
  Warp/block reductions vary in execution order depending on scheduling.

- **Hardware Differences**  
  Rounding behavior differs slightly between NVIDIA PTX and AMD ROCm backends.

---

### 2.2 Tessera Deterministic Modes

Tessera provides global control via **numerical policies**:

```python
from tessera import numerics

numerics.policy("deterministic")
```
This enforces:
	•	Fixed reduction trees (balanced binary across warps).
	•	Ordered atomics replaced with segmented reductions.
	•	Consistent rounding modes across backends.
	•	Reproducibility across runs on identical topologies.

⸻

2.3 Deterministic Reductions

Tessera uses tree-structured reductions to guarantee order:
```
      a+b
     /   \
   a       b
  / \     / \
 x   y   z   w
 ```
 This ensures reproducibility independent of warp scheduling.

2.4 Cross-Device Determinism

On multi-GPU systems:

	•	All-Reduce is performed with fixed reduction ordering (ring or tree).
	•	NVLink/NVSwitch collectives preserve order across ranks.
	•	Tessera runtime guarantees bitwise identical results across data-parallel shards.

⸻

2.5 Debug vs. Release Modes

	•	Debug Mode (Strict)
	•	Full determinism.
	•	Stable results for validation & debugging.
	•	Lower performance.
	•	Release Mode (Fast)
	•	Atomics and reductions optimized for throughput.
	•	Non-deterministic but faster.

Switching modes:

```python
numerics.policy("fast")          # Default: highest speed
numerics.policy("deterministic") # Strict reproducibility
numerics.policy("strict")        # Bitwise stable, slower than deterministic
```
2.6 Example: Deterministic Training Loop

```python
from tessera import numerics, graph, op

numerics.policy("deterministic")

@graph.training_step(module="StableModel")
def step(batch):
    out = model(batch["input"])
    loss = op.cross_entropy(out, batch["labels"])
    grads = graph.backward(loss)   # Uses deterministic reduction trees
    return grads, {"loss": loss}
```
2.7 Performance Considerations

	•	Deterministic execution can cost 10–30% throughput loss due to ordered operations.
	•	Recommended only for:
	•	Research reproducibility.
	•	Debugging unstable training.
	•	Cross-hardware benchmarking.

⸻

2.8 Summary

	•	Non-determinism arises from reductions, atomics, and rounding differences.
	•	Tessera exposes policies to enforce determinism.
	•	Users can balance speed vs. reproducibility.
	•	Determinism is critical for science, debugging, and reproducibility.