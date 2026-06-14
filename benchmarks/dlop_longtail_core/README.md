# dlop_longtail_core

A DLOP-Bench-style benchmark for **long-tail operator fusion**, modeled on
[DeepLink DLOP-Bench](https://github.com/DeepLink-org/DLOP-Bench). DLOP's lesson:
the expensive operators are *long-tail composites* — an attention block, a
SwiGLU FFN, a CV bbox transform — that decompose into many basic ops, and
running them unfused pays a kernel-launch per basic op. DLOP grades each in
Stage-1 (eager) vs Stage-2 (graph/JIT-fused).

That decomposition-overhead story **is Tessera's fusion thesis**
(matmul→softmax→matmul, `moe_swiglu_block`). This core turns the claim into
measured rows, reusing the dispatch-count + metamorphic-equivalence telemetry
from `long_memory_core`'s resident-vs-recompute row.

## What each row reports

| Composite | Family | Primitives (unfused) | Fused lane | Reduction |
|-----------|--------|----------------------|------------|-----------|
| `attention_block` | attention | matmul·scale·softmax·matmul (4) | `flash_attn` | 4× |
| `swiglu_ffn` | moe | matmul·silu·matmul·mul·matmul (5) | `moe_swiglu_block` | 5× |
| `rmsnorm_linear` | normalization | square·mean·rsqrt·mul·matmul (5) | `matmul_rmsnorm` | 5× |
| `bbox2delta` | cv_longtail | 13 elementwise ops | **none** | 1× (gap) |

Each row carries: `eager_dispatches`, `fused_dispatches`, `decomposition_factor`
(primitives-per-composite), `dispatch_reduction_x`, the real `fused_apple_gpu_lane`,
and `metamorphic_equivalent` (fused ≡ eager-decomposed — the correctness gate on
the reduction claim).

## Honest gaps

`bbox2delta` (the CV long-tail op, in DLOP's spirit) has **no dedicated fused
kernel** — it reports 1× (no reduction) and is named in `host_composed_gaps`,
not a fabricated fusion win. That's the actionable surface: a long-tail op a
fused kernel would accelerate but which Tessera composes on the host today.

## Run

```bash
python benchmarks/dlop_longtail_core/benchmark_dlop_longtail.py
```

Guarded by `tests/unit/test_dlop_longtail_core.py`.
