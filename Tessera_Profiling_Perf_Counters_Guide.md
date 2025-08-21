# Tessera Profiling & Performance Counters Guide
**Status:** v1.0 (Informative & Practical)**  
**Audience:** Kernel authors, performance engineers, SRE

This guide explains Tessera’s **profiling stack**, the **low‑level performance counters** it exposes (SM occupancy, warp divergence, memory stalls, tensor‑core utilization, copy engine metrics, collective timings), and how to use the **Tessera Profiler** (both CLI and Python) — including a **PyTorch‑profiler‑style** API — to collect timelines and analyze bottlenecks. It also documents backend mappings to Nsight/CUPTI, rocprof, and Level Zero.

---

## 1. Overview

Tessera provides a layered profiling model:
- **On‑device counters** → SM/warp/memory/TC/copy engines/collectives.
- **Runtime events** → streams, async copies, graph launches, collectives.
- **Trace export** → Chrome Trace Event (`.json`), OpenTelemetry (`OTLP`), CSV.
- **Backend adapters** → CUPTI (NVIDIA), rocprof/rocTX (AMD), Level Zero Metrics (Intel), NVML/ROCm‑SMI power/thermals.

The profiler can run in:
- **Timeline mode** (fine‑grained events + counters per kernel/copy/collective)
- **Sampling mode** (low overhead periodic sampling)
- **Summary mode** (per‑op and per‑graph aggregates; roofline helpers)

---

## 2. Counter Taxonomy (What You Can Measure)

### 2.1 Core GPU/SM Counters
- **SM Occupancy**: active warps / max resident warps (time‑weighted).
- **Warp Launch / Active / Eligible** counts.
- **Warp Divergence**: fraction of divergent branches; reconvergence stalls.
- **Issue/Execution Slots**: instruction issue efficiency (% peak).

### 2.2 Memory System
- **DRAM BW (read/write)**: achieved GB/s; utilization vs peak.
- **L2 Hit/Miss**, L2 BW; **L1/SMEM** transactions; bank conflicts.
- **Memory Stalls**: percentage of cycles stalled on memory dependency.
- **Async Copy Depth**: in‑flight `cp.async`/prefetch queues.

### 2.3 Tensor Core / Vector Units
- **MMA Utilization** (tensor core % busy).
- **FMA/ALU Utilization** (non‑TC math % busy).
- **Mixed‑Precision Accumulation** (fp16/bf16→fp32).

### 2.4 Copy & DMA Engines
- **Memcpy Throughput (H2D/D2H/D2D)**; overlap with compute.
- **Pinned vs pageable hit rate** (H2D/D2H).
- **PCIe/NVLink link util** (if available).

### 2.5 Collectives
- **All‑reduce / all‑gather / reduce‑scatter** duration & bytes.
- **Topology path** (intra‑node vs inter‑node), **alg** (ring, tree).
- **Overlap** with compute (time overlapped vs serialized).

### 2.6 Power & Thermals
- **Board power (W)**, **rail voltages**, **clock domains**.
- **Temperature (°C)**, **throttle reasons** (power/thermal/reliability).

---

## 3. APIs & Usage

Tessera exposes a Python API and a CLI. Most users start with the **Python context manager** to collect detailed traces with low code changes.

### 3.1 Python: Quick Start (PyTorch‑Profiler‑Style)

```python
from tessera import profile, op

with profile.session(
    activities=["gpu", "memcpy", "collective"],
    record_shapes=True,
    with_stack=True,
    sample_interval_ms=5,
    exporters=[profile.chrome("trace.json"), profile.summary("summary.csv")]
) as sess:
    y = op.matmul(A, B)
    y = op.flash_attention(Q, K, V)
    sess.mark("post_attention")   # user annotation
```

- `activities`: which domains to record.
- `record_shapes`: capture tensor shapes for each op.
- `with_stack`: capture Python stack (costly; use selectively).
- `sample_interval_ms`: enable sampling mode (optional).
- `exporters`: choose trace formats.

### 3.2 Streams & Events Annotations

```python
from tessera import stream, event, memcpy

s0 = stream.create(name="compute")
s1 = stream.create(name="h2d")

ev = event.create()
memcpy.async(dst=X_dev, src=X_host, stream=s1)
ev.record(s1)

ev.wait(s0)                   # compute waits for H2D
y = op.matmul(A, B, stream=s0)
```

### 3.3 Region Scopes & NVTX‑like Markers

```python
from tessera import profile

with profile.region("forward/attention"):
    y = op.flash_attention(Q, K, V)
```

### 3.4 CLI

```bash
# Attach to a running process by PID and sample counters
tessera-prof --pid 12345 --sample-ms 10 --gpu --mem --tc --collectives   --export trace.json --summary perf.csv
```

---

## 4. Interpreting Key Counters

### 4.1 SM Occupancy
- **Goal**: high but not maximal; focus on **latency hiding** rather than peak resident warps.
- Low occupancy + high memory stalls → increase tiling or reduce registers per thread.

### 4.2 Warp Divergence
- High divergence → refactor control flow or use predication / warp‑level primitives.
- For reductions: use warp‑shuffle (`shfl`) over shared‑mem when appropriate.

### 4.3 Memory Stalls
- High memory stalls + low DRAM BW → **uncoalesced** accesses or cache thrash.
- Remedies: data layout (tiled/blocked), vectorized loads, `cp.async` prefetch, double buffering.

### 4.4 Tensor Core Utilization
- Low TC util in matmul/attention → tile mismatch or precision mismatch.
- Verify MMA shapes (e.g., m16n8k8, m16n16k16) and alignment; ensure accumulator FP32.

### 4.5 Collectives Overlap
- If collectives dominate, enable **reduce‑scatter/gather** patterns and overlap.
- Use **hierarchical collectives** to limit inter‑node time; check overlap ratio.

### 4.6 Power/Thermals
- Frequent **power throttle** → enable DVFS shaping; apply stagger/jitter (see Energy Guide).
- Track **crest factor** indicators during bursts.

---

## 5. Exposed Counter Set (Portable Facade)

Tessera normalizes counters across backends with a stable schema. Fields include:
- `kernel_name`, `stream_id`, `start_us`, `end_us`
- `sm_occupancy`, `warp_divergence_pct`, `inst_issue_eff_pct`
- `dram_read_gbps`, `dram_write_gbps`, `l2_hit_rate`, `smem_bank_conflicts`
- `tc_util_pct`, `alu_util_pct`, `cp_async_inflight`
- `memcpy_dir`, `memcpy_gbps`
- `collective_type`, `collective_bytes`, `collective_overlap_pct`
- `power_w`, `temp_c`, `throttle_reason`

```python
for rec in sess.records():
    if rec.kind == "kernel":
        print(rec.kernel_name, rec.sm_occupancy, rec.tc_util_pct)
```

---

## 6. Backend Mappings

| Domain        | NVIDIA (PTX)                      | AMD (ROCm)                       | Intel (Level Zero)           |
|---------------|-----------------------------------|----------------------------------|------------------------------|
| GPU counters  | **CUPTI** Metrics/Events          | **rocprof**, SMI, rocTX          | L0 Metrics, ITT              |
| Tracing       | CUDA Graphs, CUPTI Activity API   | rocTracer                         | Level Zero Tracing           |
| Power/Thermal | NVML                              | ROCm‑SMI                          | L0 + sysfs                   |
| Export Tools  | Nsight Systems/Compute compatible | rocprof CSV                       | VTune/ITT export             |

Tessera’s profiler adapters translate raw metrics → normalized schema. Where a metric is unavailable, it is **null** with a `capability` tag.

---

## 7. Tessera Profiler for TensorBoard

Tessera provides a **TensorBoard plugin** (similar to PyTorch Profiler) that visualizes:
- Operator timeline (stacked by stream)
- Per‑op tables with top‑k by **self time**
- Roofline chart (operational intensity vs achieved FLOPs/BW)
- Stall breakdown (memory/dep/structural)
- Collective heatmap (bytes vs duration vs overlap)

```python
with profile.session(tensorboard_dir="runs/exp1") as sess:
    train_epoch()
```

Then:
```bash
tensorboard --logdir runs/exp1
```

---

## 8. Recipes (Common Analyses)

### 8.1 Roofline
```python
from tessera import analyze
analyze.roofline(trace="trace.json", arch="sm90")
```

### 8.2 Memory Stall Breakdown
```python
analyze.stalls(trace="trace.json", group_by="kernel")
```

### 8.3 Divergence Scan
```python
analyze.divergence(trace="trace.json", threshold=0.2)
```

### 8.4 Collective Overlap
```python
analyze.collectives(trace="trace.json", min_bytes=64<<20)
```

---

## 9. Overhead & Best Practices
- Prefer **sampling mode** for long runs; use timeline mode for short windows.
- Disable `with_stack=True` unless debugging CPU scheduling overhead.
- Use **shape bucketing** and **CUDA Graph capture** to reduce kernel launch noise.
- Always collect **power/thermal** when validating energy/thermal changes.

---

## 10. Programmatic Annotations & Counter Hooks

### 10.1 User Counters
```python
profile.counter(name="loss", value=float(loss), unit="scalar")
profile.counter(name="tokens_per_s", value=tps, unit="token/s")
```

### 10.2 Step Markers
```python
sess.mark("start/step_1200")
...
sess.mark("end/step_1200")
```

---

## 11. Example End‑to‑End

```python
from tessera import profile, op

with profile.session(
    activities=["gpu","memcpy","collective","power"],
    exporters=[profile.chrome("step.json"), profile.summary("step.csv")],
    tensorboard_dir="runs/exp2",
) as sess:
    # H2D
    memcpy.async(dst=X_dev, src=X_host, stream=s_load)
    # Compute
    Y = op.gemm(A, B, stream=s_comp)
    Z = op.flash_attention(Q, K, V, stream=s_comp)
    # Collective
    G = op.all_reduce(grad, op="sum", stream=s_comm)

print("Summary at runs/exp2 and step.json")
```

---

## 12. Limitations & Notes
- Some counters are **architecture‑specific**; not all are portable.
- On certain drivers, **privileged counters** require elevated permissions.
- Sampling period < 1 ms may add overhead; default 5–10 ms.
- Collectives attribution across ranks uses **clock sync**; ensure NTP/PTP.

---

## 13. Appendix: Chrome Trace Schema (Excerpt)

```json
{
  "traceEvents": [
    { "name": "gemm", "ph": "X", "ts": 12345, "dur": 321, "cat": "kernel",
      "args": { "sm_occupancy": 0.71, "tc_util_pct": 0.83 } },
    { "name": "all_reduce", "ph": "X", "ts": 12400, "dur": 210, "cat": "collective",
      "args": { "bytes": 134217728, "overlap_pct": 0.46 } }
  ]
}
```

---

**Summary:** The Tessera Profiler provides a **CUPTI‑like** capability with portable counters, a **PyTorch‑profiler‑style** API for ease of use, and exports compatible with industry tools. Use it to pinpoint occupancy, divergence, memory stalls, and collective overlap — and to verify energy/thermal improvements.
