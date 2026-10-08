---
classification: Backend architecture reference
authority: NVIDIA / CUDA reader entry point
last_updated: 2026-07-13
---

# NVIDIA Backend

This page is the reader-facing entry point for Tessera's CUDA target, from
generic compiler-emitted CUDA through the target-specific tensor-core lanes.

## Current evidence

- [Runtime execution matrix](../../audit/generated/runtime_execution_matrix.md)
  is current execution and placement truth.
- [NVIDIA target map](../../audit/generated/nvidia_sm90_target_map.md) is the
  generated target view.
- [Kernel inventory](kernel-inventory.md) explains the SM90+ planned contract;
  [sm_120 guide](sm120-kernel-guide.md) records the separately proven consumer
  Blackwell lane.

## Architecture and decisions

[NVIDIA audit](../../audit/backend/nvidia/NVIDIA_AUDIT.md) owns current target
decisions and execution deltas. [Blackwell execution plan](../../audit/backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md)
is the active implementation plan; archival material stays under its audit
folder.

## Explicit asynchronous saved-LSE reverse execution

An O/LSE program returned by compile_native_attention_vjp can capture resident Q/K/V
and supported bias with asynchronous=True. The default remains synchronous.

~~~python
with program.capture(q, k, v, asynchronous=True) as frame:
    output, lse = frame.primal
    gradients = frame.backward((output_cotangent, lse_cotangent))
    frame.wait_on(consumer.stream)
    # Enqueue consumer reads on this stream before leaving the frame.
    consumer.synchronize()
~~~

Async views advertise the private producer stream. Use wait_on to register
external consumers; close waits for them before releasing buffers. Producer
and consumer streams must remain alive until close; enqueue external reads
before close. synchronize completes private work and releases source references.
Inputs without producer streams are refused in asynchronous mode.
This is the static f32 saved-LSE reverse envelope, not general dynamic,
composed or higher-order attention AD. Evidence:
[Owning packet](../../../benchmarks/baselines/nvidia_attention_async_owner_20261008/README.md).
