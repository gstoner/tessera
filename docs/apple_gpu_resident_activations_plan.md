# Apple GPU — GPU-resident activations / persistent device-handle model (scoping)

> Status: **scoping only — not yet implemented.** This doc sizes the
> runtime-architecture change needed to stop activations round-tripping to the
> host between ops and across decode steps. It follows the Gumbel-sampler
> finding that the sampler (and, by extension, the whole per-token decode loop)
> is **sync-bound, not compute-bound**, in the current per-op host-dispatch
> runtime.

## 1. Current runtime model (verified in-tree)

The Apple GPU runtime is **per-op, host-orchestrated**:

- `runtime.py::_execute_apple_gpu_mps_metadata` walks an op list holding a
  `values` dict of **numpy arrays**. Every op does
  `_as_numpy(values[name])` → calls a C ABI symbol → the symbol
  `newBufferWithBytes`-uploads inputs, runs, and `readBytes`-downloads the
  result back into a numpy array stored in `values`.
- The decode *loop* is Python orchestration **across many `@jit` calls**, with
  the KV cache (`MLAPagedDecoder` / `MLABlockPagedCache`) holding **numpy**
  state between steps.

Two facts make the cost profile Apple-specific:

1. **Unified memory.** Buffers are `MTLResourceStorageModeShared` — CPU and GPU
   share physical RAM. A host↔device "transfer" is *not* a PCIe copy; the only
   real copies are the redundant `newBufferWithBytes` (host→buffer) and
   `readBytes` (buffer→host) memcpys, both avoidable by wrapping the numpy
   memory as a shared `MTLBuffer` in place.
2. **Synchronous dispatch.** 13 of 15 GPU runs use
   `runWithMTLCommandQueue:` (blocks the CPU until the GPU finishes) vs. 2
   `encodeToCommandBuffer:`. So **every op pays a full CPU↔GPU round-trip
   sync**. In a per-token decode chain (projections → attention → MLP → logits
   → sample) that is ~6–12 syncs/token, each flushing the GPU pipeline.

**Conclusion:** the bottleneck at small batch is **(a) per-op synchronization**
and **(b) redundant memcpys of values that are already in unified memory** — not
FLOPs and not (on Apple) PCIe bandwidth. The Gumbel sampler PR measured exactly
this: an argmax that is trivially cheap was *slower* on GPU because of the
per-call upload + sync.

## 2. The opportunity

Keep a value produced by op *N* **resident as a device buffer** so op *N+1*
consumes it without a readback + re-upload, and **batch the op chain into one
command buffer** so the CPU syncs once per decode step instead of once per op.

On unified memory this is unusually favorable: "resident" mostly means *don't
copy and don't sync*, not *move data across a bus*.

## 3. Proposed model — `DeviceTensor` handle + buffer registry

A small, opaque device-tensor handle that the dispatch loop threads instead of
numpy arrays.

### C ABI (runtime shim)

```
typedef struct TsDeviceTensor TsDeviceTensor;   // wraps id<MTLBuffer> + shape + dtype + valid flags

TsDeviceTensor* ts_dev_alloc(int64_t nbytes);                 // shared MTLBuffer
TsDeviceTensor* ts_dev_wrap(const void* host_ptr, int64_t n); // zero-copy: shared buffer over host memory
void            ts_dev_upload(TsDeviceTensor*, const void* host_ptr, int64_t n);
void            ts_dev_download(TsDeviceTensor*, void* host_ptr, int64_t n);  // materialize to host
void            ts_dev_free(TsDeviceTensor*);
```

Plus **handle-taking variants of the hot kernels** (or a uniform
"buffers-by-id" dispatch) so a kernel reads/writes `TsDeviceTensor` directly
instead of `const float*` + internal `newBufferWithBytes`.

### Python

- A `DeviceTensor` class wrapping the handle (shape/dtype + `valid_on ∈
  {host, device, both}`).
- The dispatch loop's `values` dict holds `DeviceTensor`s. `_as_numpy(handle)`
  **lazily downloads** only when the host genuinely needs the data — the
  program's final output, or a host-only op.
- Producers return `DeviceTensor`; consumers accept it. Intermediates never
  touch host.

### Command-buffer batching

The highest-value, hardest piece: replace per-op `runWithMTLCommandQueue:` with
`encodeToCommandBuffer:` so an entire op chain encodes into **one** command
buffer, committed once with a single completion wait. MPSGraph supports
`encodeToCommandBuffer:feeds:targetTensors:executionDescriptor:`. This is what
removes the per-op syncs.

## 4. Two layers of the win

1. **Within a `@jit` op-program** — chain ops device→device via the `values`
   dict of handles; one command buffer per program. (Already-fused kernels like
   `matmul_softmax_matmul`, `swiglu`, `mla_absorb_decode` get *some* of this
   internally; this generalizes it to arbitrary chains.)
2. **Across decode steps** — logits stay on device and feed the Gumbel sampler
   directly (the sampler becomes a real win); the KV cache (already
   device-shaped: `c_kv` / `k_rope`) lives in device buffers; only the sampled
   token id (one int) is read back per step.

## 5. Phasing

| Phase | Scope | Effort | Risk |
|---|---|---|---|
| **R0 ✅** | **DONE** — `DeviceTensor` handle + `ts_dev_alloc/contents/nbytes/upload/download/free/is_metal`; shared-buffer storage with zero-copy `.numpy()` view + lazy host materialization; non-Apple host-memory parity. `tests/unit/test_apple_gpu_device_tensor.py` (13). | Med | Low |
| **R1 🟡** | **Started** — first handle-taking entry point: `tessera_apple_gpu_bmm_dev_f32(TsDeviceTensor A,B,O,…)` consumes the inputs' shared buffers in place and writes the MPSGraph result straight into the output buffer (`resultsDictionary`) — **no host upload, no readback** — so a chain of `_apple_gpu_bmm_device` calls keeps intermediates on-GPU. Shares the bmm graph cache; host-ptr path kept. `tests/unit/test_apple_gpu_resident_bmm.py` (6, incl. a 4-deep chain). **Remaining:** extend handle entry points to absorb_decode / rowops / gumbel and thread handles through the metadata dispatch loop with lazy materialization. | Med–Large | Med |
| **R2** | **Command-buffer batching** — one `encodeToCommandBuffer:` per op-chain, single commit + wait. The core perf lever. | Large | **High** |
| **R3** | Persistent decode-loop state — logits resident → Gumbel sampler consumes a device handle → only the token id reads back. | Med | Med |
| **R4** | Device-resident KV cache — `c_kv`/`k_rope` and the paged blocks live in device buffers; gather/append happen on-device. | Large | Med–High |

## 6. Risks / hard parts

- **MPSGraph run model.** Moving from synchronous `runWithMTLCommandQueue:` to
  `encodeToCommandBuffer:` + manual commit is a rework of *every* dispatcher and
  must interoperate with the graph cache. This is the crux of R2.
- **Buffer lifetime.** Today the RAII pool (`TS_METAL_BUF_ACQUIRE`) auto-releases
  per call. Persistent handles need explicit ownership + a different pool
  discipline (ref-counting or arena-per-decode-step), or the pool will recycle a
  buffer that a live handle still points at.
- **Core dispatch contract.** The numpy-centric `values` dict + `_as_numpy`
  everywhere is load-bearing; introducing handles touches the most-used path and
  every dispatcher. Must stay transparent (handle aliasing, in-place ops,
  equality).
- **Debuggability / determinism.** On-device intermediates are harder to inspect;
  need a `.numpy()` escape hatch and a debug mode that forces materialization.
- **bf16 representation.** bf16 is stored as fp32 on host today; a device-resident
  bf16 needs one consistent representation across the handle boundary.
- **Portability.** The numpy fallback path (non-Apple / no-Metal) must keep
  working — handles degrade to numpy-backed.

## 7. Expected payoff (and where it *doesn't* help)

- **Biggest win at batch = 1** single-stream decode, where per-op sync overhead
  is the largest fraction of per-token latency. Removing ~6–12 syncs/token and
  the logits readback should cut per-token latency materially.
- **Shrinking win as batch grows** — at large batch the matmuls already dominate
  wall-clock, so the relative sync overhead is smaller (though the absolute
  savings remain).
- **Makes the Gumbel sampler a genuine win** (it's upload-bound only because the
  logits aren't resident).
- Does **not** change FLOPs or peak throughput of any single kernel; it removes
  *orchestration* overhead.

## 8. Narrower alternative — "fused decode step" (R2-lite)

If the full handle model is too much surface area, a targeted version captures
most of the batch-1 win: encode the **fixed per-token op chain** (the decoder
layer + sampler) into **one command buffer** behind a single
`tessera_apple_gpu_decode_step` entry point, with the activations living in
runtime-owned scratch buffers for the duration of the step. Less general (no
arbitrary `@jit` chains, no cross-step residency beyond the cache), but far less
core-dispatch disruption and it directly attacks the dominant cost.

## 9. Decision points

1. **Is decode *latency* the actual goal?** The current model is correct and
   already runs the MLA stack; this is a latency optimization for small-batch
   serving, not a correctness or capability gap.
2. **Full handle model (R0–R4) vs. fused-decode-step (R2-lite)?** The handle
   model is general and reusable across all ops; R2-lite is cheaper and targets
   the same batch-1 cost with much less risk.
3. **Dual ABI or migrate?** Keep host-ptr kernels alongside handle kernels (safe,
   more code) vs. migrate the hot ops (cleaner, riskier).

## 10. Recommendation

Start with **R0** (handle + registry + zero-copy wrap) — it is low-risk, useful
on its own (kills the redundant memcpys immediately), and is the prerequisite
for everything else. Prove it end-to-end on the absorbed MLA decode + Gumbel
sampler (logits resident → sampler consumes a handle), then decide between the
full R1→R2 build-out and the R2-lite fused-decode-step based on the measured R0
gain. Defer R2 (command-buffer batching) until R0/R1 numbers justify the
high-risk rework.
