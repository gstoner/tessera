# Tessera v1.4 — Runtime Wiring & Tracing

**New:**
- Execution runtime that wires **Policy LUT** + **TokenLimiter** so `qos.acquire/release` conceptually gate chunk submission.
- Optional NCCL/RCCL adapters (guarded by CMake options) for nonblocking submit + callback completion.
- Real `pack_cast`: FP32→BF16 (RNE) and FP32→FP8(E4M3, clamped) with optional simple RLE compression.
- Perfetto exporter enriched with **PIDs/TIDs per device/stream** and **per-chunk byte counters**.

**Demos:**
- `tessera-exec-demo`: uses TokenLimiter-gated submissions and writes `tessera_exec_trace.json`.
- `tessera-trace-demo`: minimal Perfetto JSON with roofline bands.
