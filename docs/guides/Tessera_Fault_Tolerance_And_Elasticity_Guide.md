---
status: Informative
classification: Guide
authority: Fault tolerance, elastic membership, and runtime checkpointing workflows
last_updated: 2026-04-28
---

# Tessera Fault Tolerance and Elasticity Guide

Tessera treats resilience as a compiler/runtime contract, not only an
orchestrator concern. The compiler must expose quiescent points, checkpoint
metadata, mesh/shard plans, deterministic replay metadata, and schedule hashes.
The runtime uses those artifacts to survive failures and re-form distributed
jobs.

This guide is practical and status-aware: the Python foundation exists today for
policies, manifests, failure injection, and elastic metadata. Full distributed
communicator reformation and on-device restart are production-runtime work.

---

## 1. Goals and Scope

Tessera resilience aims to:

- Survive GPU/process/node failures without losing all training progress.
- Support elastic world size when workers join or leave.
- Provide atomic, sharded checkpoints with manifests.
- Preserve deterministic replay metadata when deterministic mode is enabled.
- Interoperate with Kubernetes, SLURM, Ray, and rendezvous services.

Out of scope:

- High availability of object stores, metadata databases, or external
  schedulers.
- Business continuity and cross-region disaster recovery.

---

## 2. Failure Model

Tessera classifies:

- **GPU/process crash:** OOM, illegal memory access, driver reset.
- **Node loss:** power loss, network loss, kernel panic.
- **Network fault:** latency spike, partition, dropped collective.
- **Preemption:** spot/low-priority eviction with short notice.
- **Partial degradation:** thermal throttling, link down, reduced bandwidth.

Assumptions:

- Durable storage exists for checkpoints.
- The cluster manager can restart or reschedule workers.
- Deterministic mode can persist RNG streams, reduction plans, graph hashes, and
  schedule hashes for bitwise replay when strict reproducibility is required.

---

## 3. Failure Handling

Use `tessera.fault` to attach policy metadata:

```python
from tessera import fault

@fault.on_failure(policy="drain_then_resume", max_retries=3)
def train_step(batch):
    ...
```

Supported policies:

- `drain_then_resume`: cancel in-flight step, reach a quiescent point, resume
  from the last committed checkpoint.
- `fail_fast`: abort and let the scheduler restart the job.
- `pause_for_manual`: stop scheduling new work and wait for an operator.

Preemption hooks:

```python
@fault.on_preempt(grace_s=30, action="checkpoint_then_exit")
def main():
    ...
```

Failure injection for tests/staging:

```python
with fault.inject(drop_device=3):
    train_step(batch)

with fault.inject(network_partition=[(0, 64)]):
    op.all_gather(x)
```

---

## 4. Elastic Membership

Elastic execution is coordinated through rendezvous metadata:

```python
from tessera import elastic

elastic.configure(
    backend="k8s",
    group="exp-gpt",
    min_ranks=64,
    max_ranks=256,
    rebalance_on_join=True,
    rebalance_on_exit=True,
)
```

Context form:

```python
with elastic.elastic(rendezvous="k8s://tessera-rdzv", min_ranks=16, max_ranks=256):
    run_training()
```

The `tessera.dist` convenience namespace also exposes:

```python
with tessera.dist.elastic(rendezvous="k8s://tessera-rdzv", min_ranks=16, max_ranks=256):
    ...
```

Elastic resizing should happen at epoch boundaries when strict deterministic
replay is required. Step-boundary resizing is faster but may change reduction
order and therefore floating-point results.

---

## 5. Mesh Reconfiguration and Resharding

When membership changes, Tessera re-plans mesh axes:

- DP: all-reduce/reduce-scatter groups are recomputed.
- TP: weight shards may be all-gathered and repartitioned.
- PP: stage placement and microbatch schedule may be rebalanced.
- MoE: experts should move by consistent hashing to minimize data movement.

Python planning surface:

```python
plan = elastic.reshard(
    policy="consistent_hash",
    migrate_async=True,
    old_mesh={"dp": 8, "tp": 8},
    new_mesh={"dp": 16, "tp": 8},
)
print(plan.moved_fraction())
```

`consistent_hash` should be the default for optimizer shards and MoE experts
because it minimizes movement during resize.

---

## 6. Runtime Checkpointing

There are two checkpoint concepts:

- `tessera.compiler.checkpoint`: activation checkpointing/rematerialization
  markers for memory reduction.
- `tessera.checkpoint`: runtime checkpoint/restart manifests for resilience.

Save an atomic sharded checkpoint:

```python
from tessera import checkpoint

manifest = checkpoint.save(
    tag="step_120000",
    tensors=model.parameters(),
    optimizer=optimizer.state_dict(),
    mesh={"dp": 16, "tp": 8, "pp": 4},
    atomic=True,
    numerics="deterministic",
    rng={"global": 1234},
    reduce_tree_id="ring-16x8x4-v3",
)
```

Load and remap:

```python
state = checkpoint.load("step_120000", remap_to={"dp": 32, "tp": 8, "pp": 2})
model.load_parameters(state.tensors)
optimizer.load_state_dict(state.optimizer)
```

Async policy:

```python
checkpoint.enable_async(max_bandwidth_gbps=4.0, flush_interval_s=5.0)
```

Production checkpointing should use two-phase commit:

1. Write shard payloads.
2. Write manifest.
3. Publish an atomic `COMMITTED` marker.
4. On restart, load only the latest committed manifest.

---

## 7. What Gets Checkpointed

A resilient checkpoint manifest should include:

- Model parameters and shard metadata.
- Optimizer state, including ZeRO partitioning.
- RNG state per rank.
- Deterministic reduction plan ID.
- Graph hash and schedule hash where available.
- Autotune cache summary.
- Data loader offset, epoch, and seed when training.
- Optional KV cache state for inference serving.

Example manifest fragment:

```yaml
version: 1
tag: step_120000
backend: ptx
mesh: {dp: 16, tp: 8, pp: 4}
numerics: deterministic
rng: {global: 1234}
reduce_tree_id: ring-16x8x4-v3
parameters:
  - name: transformer.w_qkv
    shape: [8192, 24576]
    dtype: bf16
    sharding: {axis: ["tp"], part: 8}
    path: transformer_w_qkv.tp0.safetensors
autotune_cache:
  arch: sm90
  entries: 1824
committed: true
```

---

## 8. Orchestrator Integration

Tessera should interoperate with:

- Kubernetes: StatefulSets/Jobs, readiness gates, PodDisruptionBudgets.
- SLURM: `srun`, `scontrol requeue`, elastic partitions.
- Ray: placement groups and actor restart.
- etcd/Redis/K8s rendezvous adapters.

Topology changes can invalidate kernels and tuning entries:

```python
policy = elastic.on_topology_change(retune_autotuner=True, invalidate_kernels=True)
```

When hardware changes across recovery, checkpoints remain hardware-agnostic, but
compiled kernels and autotune artifacts should be invalidated or retuned.

---

## 9. Observability

Track:

- Failure events: rank lost, retry count, partition healed.
- Recovery time and MTTR.
- Checkpoint duration, bandwidth, and age/RPO.
- Elastic resize events and reshard bytes moved.
- Autotune retune duration after topology changes.
- Determinism divergence flags and checksums.

Example:

```python
monitor.emit({
    "event": "elastic_resize",
    "world_size": tessera.dist.world_size(),
    "reshard_mib": stats.reshard_bytes / 2**20,
})
```

---

## 10. Testing and Validation

Required staging tests:

- Drop rank during all-reduce, then verify regroup or rollback.
- Kill node mid-pipeline, then verify drain and resume.
- Inject network partition, then verify timeout classification.
- Simulate preemption and ensure checkpoint completes inside grace period.
- Resize at epoch boundary and verify deterministic replay.
- Resize at step boundary and verify nondeterminism is explicitly allowed.

Use `tessera.fault.inject(...)` for local policy testing and
`tessera.testing.ChaosEvent` for QA contracts.

---

## 11. Compiler Integration Contract

| Layer | Responsibility |
|-------|----------------|
| Graph IR | Identify checkpointable state, RNG streams, cache objects, and quiescent barriers |
| Schedule IR | Place checkpoint barriers away from heavy collective windows |
| Tile/Target IR | Surface device faults with enough metadata to classify recovery |
| Runtime | Execute two-phase checkpoints, heartbeat ranks, detect timeouts |
| Distributed planner | Recompute mesh groups and reshard state |
| Autotuner | Invalidate or retune artifacts after topology/architecture change |

Strict determinism requires checkpoint metadata for RNG streams, reduction tree,
collective ordering, graph hash, schedule hash, and world-size assumptions.

---

## 12. Quick Reference

```python
fault.on_failure(policy="drain_then_resume")
fault.on_preempt(grace_s=30, action="checkpoint_then_exit")

with fault.inject(drop_device=3):
    train_step(batch)

elastic.configure(backend="k8s", group="exp", min_ranks=64, max_ranks=256)
with elastic.elastic(rendezvous="k8s://tessera-rdzv", min_ranks=16, max_ranks=256):
    run_training()

elastic.reshard(policy="consistent_hash", migrate_async=True)
elastic.on_topology_change(retune_autotuner=True, invalidate_kernels=True)

checkpoint.save(tag="step_1000", atomic=True)
state = checkpoint.load("step_1000", remap_to=elastic.current_mesh())
checkpoint.enable_async(max_bandwidth_gbps=4.0, flush_interval_s=5)
```

---

## 13. Best Practices

- Use atomic sharded checkpoints with committed manifests.
- Resize at epoch boundaries for strict determinism.
- Prefer consistent hashing for sharded params and MoE experts.
- Enable async/delta checkpoints to reduce I/O stalls.
- Integrate rendezvous with the scheduler, not ad hoc rank files.
- Monitor MTTR, checkpoint age/RPO, and reshard bytes moved.
- Test failure injection before production rollout.
