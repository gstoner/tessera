---
title: Distributed MegaMoE — expert-parallel Mixture-of-Experts
classification: Architecture / Distributed Workload
last_updated: 2026-06-09
status: implemented (single-device hardware-runtime on Apple GPU; multi-rank via in-process mock collectives)
---

# Distributed MegaMoE

Tessera ships a full Mixture-of-Experts (MoE) stack, from a single-device expert
feed-forward block up to a distributed, expert-parallel forward with token
all-to-all dispatch/combine, FP8×FP4 mixed precision, and **real wall-clock
comm/compute overlap**. The heavy expert compute runs on the Apple GPU through a
fused MoE-SwiGLU kernel; the data-dependent routing / permute / combine are
host-side index math (the same host-fallback policy `argmax` uses).

This document is the canonical reference for that stack. Source lives in
`python/tessera/nn/functional.py` (single-device layer),
`python/tessera/distributed/moe.py` (router + distributed forward),
`python/tessera/compiler/grouped_layout.py` (grouped-layout + quant contract),
and the fused kernel in
`src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`.

## Build-up (rung by rung)

The stack was built bottom-up; each rung is independently tested and the upper
rungs reuse the lower ones verbatim.

| Rung | Surface | What it adds |
|------|---------|--------------|
| Grouped GEMM | `ops.grouped_gemm(x, weights, group_sizes, *, kind, alignment, quant)` | Ragged per-expert matmul (the MoE compute core); `contiguous`/`dense` kinds, `masked`/`k_grouped` rejected with a Decision-#21 diagnostic; the grouped-layout + scale-layout contract. |
| Fused block | `ops.moe_swiglu_block(x, W_gate, W_up, W_down, group_sizes, …)` | The expert FFN `grouped_gemm → silu_mul → grouped_gemm` as a first-class Graph-IR op (`tessera.moe_swiglu_block`) with a verifier. One fused MSL kernel on Apple GPU. |
| Single-device layer | `nn.functional.moe_layer(x, W_router, W_gate, W_up, W_down, *, top_k, …)` | Router → top-k → gather tokens into expert order → `moe_swiglu_block` → weighted scatter-combine. The full local MoE forward, no comm. |
| Distributed forward | `distributed.moe.megamoe_forward` / `megamoe_layer` | Expert-parallel: experts sharded across ranks, tokens routed via a 2× all-to-all (GShard / Switch). |
| Mixed precision | `quant="fp8xfp4"` (and `fp8_e4m3` / `nvfp4`) | FP8 activations × FP4 weights — the Blackwell / DeepGEMM MoE scheme — flows through every rung. |
| Comm/compute overlap | `megamoe_forward_pipelined` / `megamoe_layer_pipelined` | Real wall-clock overlap: chunk c's GPU expert FFN runs async while chunk c+1's dispatch all-to-all issues. |

## The fused MoE-SwiGLU kernel

`tessera_apple_gpu_moe_swiglu_f32` collapses the three grouped-GEMM + `silu_mul`
dispatches of the expert FFN into a **single MSL dispatch** — the grouped analog
of the dense `swiglu_f32` kernel. One thread per token `t` with expert
`e = Eids[t]`:

```
gate   = x[t] @ Wg[e]            # (K → H)
up     = x[t] @ Wu[e]            # (K → H)
hidden = silu(gate) * up         # (H)
O[t]   = hidden @ Wd[e]          # (H → Kout)
```

`Wg`/`Wu` are `(E,K,H)`, `Wd` is `(E,H,Kout)`. Per-row stack buffers cap
`H, Kout ≤ 256`; past that the C symbol early-returns to its CPU reference and
the dispatcher falls back to the composed lanes. The runtime takes the fused
fast path for f32 / no-quant / `H,Kout ≤ 256`, and the composed grouped-GEMM +
`silu_mul` lanes otherwise (quant keeps exact per-GEMM scale semantics the
single fused kernel cannot express). The kernel has a non-Darwin reference
parity in `apple_gpu_runtime_stub.cpp` and is gated by the buffer-pool RAII
invariant.

## Expert-parallel forward (GShard 2× all-to-all)

Experts are sharded across `world_size` ranks: rank `r` owns the contiguous
expert block `[r·Ep, (r+1)·Ep)` (`Ep = num_experts / world_size`) and holds
**only those experts' weights** — `local_W_*` are shaped `(Ep, …)`, not the full
expert set (the memory win of expert parallelism). The router (`W_router`) is
replicated.

```
1. route this rank's tokens through the global router (top-k)
2. scatter tokens into a capacity-padded dispatch buffer keyed by the
   destination expert's owner rank             →  (R, Ep, C, K)
3. all-to-all DISPATCH      — every rank receives the tokens bound for its
   local experts, gathered from all source ranks
4. local expert FFN         — Ep ragged groups through the fused GPU
   moe_swiglu_block in one dispatch
5. all-to-all COMBINE       — send each result back to the originating rank
6. weighted scatter-combine — each token's top-k expert outputs
```

**Capacity-based dispatch** keeps every exchange buffer a fixed size so the
all-to-all is uniform (the only kind the mock thread group expresses): each
expert reserves `capacity_factor × (global token-slots / num_experts)` slots —
overflow is dropped (reported as `MegaMoEResult.n_dropped`), underflow
zero-padded. Per [Decision #6](../../../CLAUDE.md), multi-rank tests run in-process
via `MockRankGroup` (threads), so this is the production-shaped forward **and**
its own test harness.

**Correctness anchor:** with capacity large enough to drop nothing, the gathered
distributed output equals the single-device `nn.functional.moe_layer` *exactly*
(verified across `world_size` 1/2/4 × `top_k` 1/2).

## FP8×FP4 mixed precision

`grouped_layout.quant_scheme_for(quant)` resolves a `quant` spelling to an
`(act_dtype, weight_dtype)` pair. A plain dtype (`fp8_e4m3` / `nvfp4`) applies to
both operands; a mixed `<act>x<weight>` scheme splits them:

| Scheme | Activations | Weights |
|--------|-------------|---------|
| `fp8xfp4` / `fp8_e4m3xnvfp4` | `fp8_e4m3` | `nvfp4` (1×16 block scale) |
| `fp8_e5m2xnvfp4` | `fp8_e5m2` | `nvfp4` |

This is the Blackwell / DeepGEMM MoE pattern: activations carry the dynamic
range in FP8 while weights compress to FP4. The scale-layout contract
(`scale_layout_for`: FP8 → 1×128 / `ue8m0`, NVFP4 → 1×16 / `e4m3`) covers both.
Distributed FP8×FP4 error sits between pure-FP8 (~0.05) and pure-FP4 (~0.30), as
expected. The quant path is dequant-on-host (quantize → dequantize → f32 GEMM).

## Real comm/compute overlap

The enabling fact on Apple: the Metal command buffer runs asynchronously on the
GPU, and the `ctypes.CDLL` call into the runtime **releases the GIL** for its
duration. So running a chunk's fused `moe_swiglu_block` on a worker thread
(`_AsyncCompute`) lets the calling thread issue the next chunk's dispatch
all-to-all **concurrently** — the GPU command buffer *is* the async stream, and
the worker-thread + GIL-release is how the CPU comm proceeds alongside it.

`megamoe_forward_pipelined` decomposes each micro-batch into dispatch / compute /
combine phases and pipelines them. The overlap depth is set by `pipeline_stages`:

- **1-stage** — chunk c+1's **dispatch** all-to-all issues while chunk c's GPU
  compute is in flight; chunk c's **combine** runs after its compute completes
  (exposed).
- **2-stage** (default) — chunk c's combine is **deferred one iteration**, so
  while chunk c's compute is in flight the CPU issues BOTH chunk c+1's dispatch
  AND chunk c-1's combine. Both comms hide under compute, roughly doubling the
  overlap window (the classic software-pipelined MoE schedule). The per-rank
  all-to-all order stays fixed (`d0, d1, d2, c0, d3, c1, …, c_{n-2}, c_{n-1}`),
  so the cross-rank barrier protocol remains in lockstep.

`PipelineStats` reports per-chunk compute thread-ids, `all_offloaded` (proof the
GPU work ran off the comm thread), `pipeline_stages`, and `overlapped_combines`
(= `nc-1` for 2-stage, `0` for 1-stage — the combine comms that ran concurrently
with a compute).

`comm_latency_s` models per-all-to-all interconnect transfer latency — the
single-machine mock collective has none; real multi-device comm does. It is the
cost the pipeline hides.

**Measured** (M-series, `8192×256×8×256`, world_size=2, num_chunks=4):

| modeled comm / a2a | sequential-chunked | 1-stage | 2-stage | 2-stage vs 1-stage |
|---|---|---|---|---|
| 0 ms | 228.2 ms | 219.2 ms | 198.8 ms | 1.10× |
| 6 ms | 285.2 ms | 254.3 ms | 210.9 ms | 1.21× |
| 12 ms | 343.6 ms | 290.3 ms | 236.8 ms | **1.23×** |

In a compute-dominant regime the 2-stage's **exposed comm** (time-with-latency
minus time-without) is ≈4× smaller than the 1-stage's (e.g. 10.6 ms vs 41.9 ms
at 6 ms/a2a) — the combine comm is now hidden too; only the pipeline fill/drain
(the first dispatch + last combine) stays exposed. The pure GPU-vs-CPU-comm
overlap measures **1.85×** on a heavy CPU comm proxy.

**Honest scope:** the wall-clock win *requires* real comm latency. On a single
machine the mock all-to-all is a microsecond memcpy, so chunking alone is a wash
(1.07×); the async **engine** is real and the overlap is genuine the moment comm
has cost (real multi-device interconnect, or the modeled latency here). The
benchmark `benchmarks/apple_gpu/benchmark_megamoe_overlap.py` sweeps the
comm:compute ratio and reports the speedup curve.

## Tests & benchmarks

| Surface | Location |
|---------|----------|
| Grouped GEMM + scale-layout contract | `tests/unit/test_grouped_gemm_contract.py` |
| Fused MoE-SwiGLU block (incl. fused-kernel parity, large-H fallback) | `tests/unit/test_moe_swiglu_block.py` |
| Single-device `moe_layer` | `tests/unit/test_moe_layer.py` |
| Distributed forward + FP8×FP4 + overlap + async pipeline | `tests/unit/test_megamoe_distributed.py` |
| Overlap benchmark | `benchmarks/apple_gpu/benchmark_megamoe_overlap.py` |

## Open frontier

- **Real multi-device comm** — the in-process mock collective has no wire
  latency; a native NCCL/RCCL (or Apple multi-GPU) lane is where the modeled
  `comm_latency_s` becomes the actual interconnect cost. Gated on the Phase G/H
  hardware lanes.
- **Deeper pipelines** — the 2-stage schedule exposes only the pipeline
  fill/drain (first dispatch + last combine). A prologue/epilogue-software-
  pipelined variant could shave even those at the cost of more in-flight buffers,
  but the win is marginal once comm is already hidden under compute.
