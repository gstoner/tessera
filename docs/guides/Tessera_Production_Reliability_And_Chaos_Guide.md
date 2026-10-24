---
status: Informative
classification: Informative
authority: Production reliability, stress, chaos, node-scale, and rack-scale QA guide
last_updated: 2026-10-24
---

# Tessera Production Reliability And Chaos Guide

Development QA proves that a Tessera program is correct under controlled
conditions. Production reliability proves that it stays correct, observable,
reproducible, and recoverable under real cluster pressure.

This guide extends `docs/guides/Tessera_QA_Reliability_Guide.md` with
production, stress, chaos, node-scale, and rack-scale validation practices.
Current API names and implementation status remain governed by `docs/README.md`
and the specs under `docs/spec/`.

## 1. Production Reliability Goals

Production Tessera workloads should provide evidence for:

- **Monitoring**: execution status, kernel latency, memory use, collective
  latency, and throughput are tracked continuously.
- **Regression detection**: accuracy and performance drift are caught before
  rollout.
- **Replay debugging**: seeds, graph IR, schedule artifact hashes, runtime
  config, and selected kernels are captured for deterministic reproduction.
- **Observability**: traces and metrics can be exported to standard systems.
- **Recovery**: checkpoints, restart regions, and fallback paths are validated.

## 2. Monitoring And Health Checks

Track health at the same granularity the compiler/runtime reasons about:

- model step: loss, accuracy, tokens/sec, samples/sec
- runtime: device memory, stream backlog, launch latency, errors
- compiler: graph hash, schedule artifact hash, kernel variant
- distributed: all-reduce/all-gather/all-to-all latency, bytes, retries
- hardware: ECC errors, throttling, temperature, lost devices

Example metric payload:

```python
metrics = {
    "loss": 2.31,
    "gpu_memory_bytes": 41_000_000_000,
    "kernel_latency_ms": 0.42,
    "all_reduce_ms": 3.8,
    "tokens_per_sec": 148_000,
    "schedule_hash": "0123456789abcdef",
}
```

Health checks should be low overhead and always safe to run in production.
Expensive profiling belongs in sampled traces or canary jobs.

## 3. Automated Regression Detection

Regression checks compare current metrics to versioned baselines. The baseline
must include target arch, shape, dtype/numeric policy, backend, and schedule
hash.

```python
from tessera.testing import RegressionBaseline

baseline = RegressionBaseline(
    name="matmul_4096_bf16_sm90",
    latency_ms=2.0,
    accuracy=0.999,
    max_latency_regression=0.20,
    max_accuracy_drop=0.001,
)
baseline.validate({"latency_ms": 2.25, "accuracy": 0.9985})
```

Recommended gates:

- block rollout on correctness failures
- block rollout on deterministic replay mismatch
- warn on performance drift above 5%
- block rollout on performance drift above 20%, unless explicitly approved

## 4. Replay Debugging

Replay capture should record the minimal information needed to reproduce a run:

- global seed and per-rank RNG stream mapping
- graph IR/module hash
- schedule artifact hash and autotune cache key
- target arch and backend
- input batch identity or sanitized input sample
- environment variables that affect collectives/runtime
- checkpoint path and step number

```python
from tessera.testing import ReplayManifest

manifest = ReplayManifest(
    run_id="session1",
    seed=42,
    graph_hash="graph-a",
    schedule_hash="sched-b",
    target="sm90",
    backend="cuda",
)
manifest.validate()
```

`ReplayManifest` is a hand-filled record that validates required fields. To
capture one from a compiled program, `tessera.debug.replay_manifest(fn)` (alias
`replay_capture`) accepts an `@tessera.jit` wrapper, runtime artifact, or compile
bundle and returns a JSON-serializable manifest with IR hashes, the compiler
lowering trace, and the debug environment snapshot; tensor payloads are excluded.

Production incidents should attach replay manifests to logs or ticket artifacts.

## 5. Observability And Profiling

Tessera production deployments should integrate with:

- Chrome Trace Event / Perfetto for timeline traces
- Prometheus and Grafana for service metrics
- TensorBoard or training dashboards for model metrics
- profiler exports for kernel occupancy, bandwidth, FLOPs, and collectives

Trace spans should include graph hash, schedule hash, device/rank, mesh axis,
collective kind, and kernel variant when available.

Status: Chrome Trace Event / Perfetto-compatible JSON export exists today
(`tessera-prof --trace`, `tools/profiler/`). Tessera ships no Prometheus,
Grafana, or TensorBoard exporter; those integrations are the deployment's
responsibility and are planned, not provided.

## 6. Fault Tolerance In Production

Validate the recovery path before depending on it:

- checkpoint/restart works from a recent step
- replay regions reproduce the same outputs
- fallback scheduling is explicit and visible in logs
- failed rank/device context is preserved
- partial collective failure does not deadlock the job

Fallback should be a policy decision, not accidental behavior. Some workloads
should fail fast instead of silently moving to a slower backend. In the current
compiler, a `@tessera.jit` program that did not run its native path records a
`tessera.compiler.fallback.FallbackReason` on `last_fallback_reason` (for example
`FRONTEND_UNLOWERED_CALL` when the front end dropped a call and the original
Python function ran); `@tessera.jit(..., native_required=True)` turns that
fallback into a raised error.

## 7. Stress Testing

Stress tests validate long-running stability and resource behavior.

Scenarios:

- max-occupancy kernels over long runtimes
- full-memory allocation plus compute
- mixed-precision training under high load
- I/O-heavy distributed workloads
- concurrent jobs on shared nodes
- repeated autotune/cache lookup cycles

Track:

- throughput stability
- memory growth and fragmentation
- error rate
- thermal/throttling signals
- schedule reuse versus retuning

Stress tests should run in staging before production rollout and on scheduled
GPU runners for long-lived kernels.

## 8. Chaos Testing

Chaos testing injects controlled failures to validate recovery behavior.

Example event types:

- kill or hide a device
- inject collective timeout
- add artificial network latency
- simulate ECC or memory error
- drop or delay a pipeline stage
- restart a data-parallel worker mid-epoch

```python
from tessera.testing import ChaosEvent

event = ChaosEvent(kind="kill_device", target="gpu:5", expected_recovery="checkpoint_restart")
event.validate()
```

`ChaosEvent` is a test-plan descriptor: `validate()` checks the event against
its accepted kinds (`kill_device`, `network_latency`, `collective_timeout`,
`ecc_error`, `drop_rank`, `restart_worker`) and requires a target and expected
recovery. Tessera does not ship a fault injector; the harness that actually
kills devices, delays links, or drops a pipeline stage is planned and must
come from the deployment today.

Every chaos test should define:

- injected event
- expected recovery path
- maximum recovery time
- checkpoint rollback expectations
- metrics that prove the job did not silently corrupt outputs

## 9. Node-Scale QA

Node-scale validation covers one multi-GPU server.

Required checks:

- collectives are correct and deterministic across all local devices
- MIG/SR-IOV partitions are detected as isolated devices
- no cross-partition memory leakage
- autotune cache keys include arch, shape, dtype, backend, and device class
- concurrent workloads do not corrupt results or leak memory

Run node-scale collectives with representative tensor sizes, edge sizes, and
contention from other jobs.

## 10. Rack-Scale And NVL72 QA

Rack-scale validation adds topology and failure-domain concerns. No rack-scale
or NVL72 system is in the current validation fleet, so this section is a
checklist for future validation, not a description of tested behavior.

Required checks:

- mesh axes match the physical topology
- tensor, pipeline, and data parallel layouts reconstruct correctly
- deterministic reductions remain stable across all ranks
- NVLink/NVSwitch bandwidth meets expected bands
- collectives survive retry/failure policies without deadlock
- checkpoint/restart works after rank or node loss

NVL72-specific tests should record:

- mesh shape and rank ordering
- NVSwitch domain placement
- collective algorithm and protocol
- achieved bandwidth and latency
- SHARP/NVLS enablement where available
- schedule artifact hash for tuned kernels

## 11. Rollout Checklist

Before production rollout:

- QA guide checks pass for the operator/model.
- Production replay manifest is captured.
- Performance baselines are versioned.
- Monitoring dashboards show model, runtime, and collective metrics.
- Stress tests have run for the expected duration.
- Chaos tests validate the recovery policy.
- Node-scale tests pass on the target server class.
- Rack-scale tests pass for workloads that use rack-scale meshes.
- Fallback behavior is documented and visible.
