# JIT reverse attention native integration

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1; sync NVIDIA-JIT-ATTENTION-VJP-2026-10-02.

@jit(autodiff='reverse', wrt=...) now exposes compile_native_attention_vjp.
The trace enters native paired AD, exports saved O/LSE checkpoint products,
and follows verified Graph -> Schedule -> Tile -> NVIDIA Target/NVVM/PTX.
capture owns private Q/K/V/O/LSE allocations; backward returns requested
Q/K/V gradients in wrt order. Native code computes the complete Q/K/V product
even when the request selects fewer results; no work-elimination claim.

Integration fixes: generated checkpoint pairing compares both O and LSE
against the correct backward slots; selected/reordered AD returns retain
paired lineage while exporting the canonical physical gradient ABI. The
canonical flash_attn shape rule preserves query axes and takes the trailing
width from V. Public eager, tracing and CPU evaluation preserve storage dtype.
Eager grouped-query mapping now repeats each KV head over its query-head group.

## Evidence

Super-Bear RTX 5070 SM120. Eighteen full/causal, regular/ragged, grouped-query
rows, including batch two and D != Dv, pass independent float64 output/LSE
and gradient oracles. Private saved state survives resident caller mutation
and repeated backward. Post-timing output/LSE and all native gradients pass.
404 focused frontend/AD/shape/operator/dtype/diagnostic/pass/device tests pass.
Adjacent attention/audit gates passed 69 tests with 3 environment-gated skips.
All 32 generated views passed their drift check; the compiler-plan navigation
gate passed. Graphify refresh was attempted but the CLI is unavailable in WSL.

| Shape B/Hq/Hkv/Sq/Sk/D/Dv | Causal | wrt | Forward window ms | Backward window ms |
|---|---|---|---:|---:|
| 1/2/1/3/5/4/3 | False | q | 0.009049 | 0.009849 |
| 1/2/1/3/5/4/3 | False | k,q | 0.009358 | 0.010473 |
| 1/2/1/3/5/4/3 | False | v,q,k | 0.017510 | 0.016895 |
| 1/2/1/3/5/4/3 | True | q | 0.008906 | 0.010547 |
| 1/2/1/3/5/4/3 | True | k,q | 0.015141 | 0.016413 |
| 1/2/1/3/5/4/3 | True | v,q,k | 0.009693 | 0.011401 |
| 1/2/1/5/3/4/3 | False | q | 0.011207 | 0.012706 |
| 1/2/1/5/3/4/3 | False | k,q | 0.010991 | 0.012575 |
| 1/2/1/5/3/4/3 | False | v,q,k | 0.010441 | 0.012226 |
| 1/2/1/5/3/4/3 | True | q | 0.009708 | 0.013011 |
| 1/2/1/5/3/4/3 | True | k,q | 0.015946 | 0.018355 |
| 1/2/1/5/3/4/3 | True | v,q,k | 0.011483 | 0.017655 |
| 2/4/2/16/19/8/6 | False | q | 0.014024 | 0.028752 |
| 2/4/2/16/19/8/6 | False | k,q | 0.013383 | 0.028708 |
| 2/4/2/16/19/8/6 | False | v,q,k | 0.013092 | 0.028719 |
| 2/4/2/16/19/8/6 | True | q | 0.012877 | 0.025628 |
| 2/4/2/16/19/8/6 | True | k,q | 0.013450 | 0.026081 |
| 2/4/2/16/19/8/6 | True | v,q,k | 0.014197 | 0.025466 |

Compilation, forward device windows, backward device windows, capture wall
and backward wall are recorded separately. CUDA windows include C++/driver
dispatch gaps. Capture includes private copies/module load/forward; backward
wall includes allocation and synchronization. No speedup or isolated kernel
instruction-time claim.

## Remaining

Automatic bias derivatives/JVP, general composed graphs, broader dtype/layout
policies and other architecture native AD consumers remain open. This does
not close those programs or transfer RTX 5070 evidence to sibling backends.
Shape/dtype inference is shared; sibling physical consumers need exact-device
proof for independent value width and grouped-query envelopes.

## Reproduce

    source .build-sm120-w1-1/validation-env.sh
    .venv/bin/python benchmarks/nvidia/benchmark_jit_attention_vjp.py --samples 5 --reps 200 --output benchmarks/baselines/nvidia_jit_attention_vjp_20261002/rtx5070.json
