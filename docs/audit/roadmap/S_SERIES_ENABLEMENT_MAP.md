---
last_updated: 2026-06-27
audit_role: plan
plan_state: open
authority: Companion enablement map for `docs/audit/roadmap/ROADMAP_AUDIT.md`
---

# S-Series Primitive Enablement Map

> Generated from `primitive_coverage.py` (lowering + effect + manifest)
+ test-presence scan of `tests/`. 474 primitives. Each op is mapped to
its enablement path; "tested" = real device fixture OR the op name
appears in a test file.


## Enablement paths (taxonomy)

| Path | What enables the op | How it is tested |
|------|---------------------|------------------|
| **Kernel** | compiler-generated device kernel (codegen) and/or a **device library** call (Accelerate/BNNS/MPS/MPSGraph on Apple, WMMA/MFMA on ROCm, AMX/AVX-512 on x86, WGMMA/TMA on NVIDIA) | on-device numerical fixture vs numpy (`execute_compare_fixture`) |
| **IR-Transform** | a Graph/Schedule/Tile IR pass (sharding, collective insertion, layout, scheduling, structural transform) | `tessera-opt` lit/FileCheck fixtures |
| **AST-Frontend** | Python trace/transform (control flow, autodiff/grad transforms, optimizer steps, state-tree ops) | Python unit tests on the traced/transformed program |
| **Host-Library** | host-side runtime library (data/tokenizers, serialization, AOT, RNG/MCMC samplers, EBM partition) | Python unit tests (no device kernel) |
| **TSOL** (cross-cut) | the 47 canonical Tessera Standard Operations — composed from the above | spec catalog + per-axis dashboard |

## Summary by path

| Path | ops | tested | untested | x86 exec | rocm exec | uses device-lib |
|------|----:|-------:|---------:|---------:|----------:|----------------:|
| Kernel | 282 | 277 | 5 | 82 | 75 | 189 |
| IR-Transform | 82 | 68 | 14 | 0 | 0 | 50 |
| AST-Frontend | 49 | 42 | 7 | 2 | 3 | 21 |
| Host-Library | 61 | 61 | 0 | 1 | 0 | 3 |
| **TSOL (subset)** | 47 | 45 | 2 | 2 | 12 | 47 |

## Category -> path map

| Category | path | ops | tested | rocm | x86 |
|----------|------|----:|-------:|-----:|----:|
| elementwise | Kernel | 51 | 50 | 24 | 16 |
| loss | Kernel | 29 | 29 | 0 | 0 |
| attention | Kernel | 28 | 28 | 13 | 0 |
| tensor_algebra | IR-Transform | 19 | 14 | 0 | 0 |
| rng | Host-Library | 18 | 18 | 0 | 0 |
| layout_transform | IR-Transform | 17 | 16 | 0 | 0 |
| geometric_algebra | Kernel | 17 | 17 | 0 | 17 |
| reduction | Kernel | 16 | 16 | 5 | 6 |
| numeric_helper | Kernel | 15 | 15 | 9 | 9 |
| indexing | IR-Transform | 14 | 13 | 0 | 0 |
| ebm | Kernel | 13 | 13 | 0 | 13 |
| transform | IR-Transform | 11 | 7 | 0 | 0 |
| data | Host-Library | 11 | 11 | 0 | 0 |
| collective | IR-Transform | 10 | 7 | 0 | 0 |
| loop_nest | AST-Frontend | 10 | 10 | 3 | 2 |
| state_tree | AST-Frontend | 10 | 10 | 0 | 0 |
| stable_reduction | Kernel | 9 | 9 | 2 | 2 |
| spectral | Kernel | 9 | 9 | 0 | 0 |
| logical | Kernel | 8 | 8 | 8 | 8 |
| schedule | IR-Transform | 8 | 8 | 0 | 0 |
| stencil | Kernel | 8 | 6 | 0 | 5 |
| quantize | Kernel | 8 | 8 | 0 | 0 |
| grad_transform | AST-Frontend | 7 | 6 | 0 | 0 |
| control_flow | AST-Frontend | 7 | 4 | 0 | 0 |
| normalization | Kernel | 7 | 7 | 2 | 0 |
| functional_optimizer_step | AST-Frontend | 6 | 5 | 0 | 0 |
| aot | Host-Library | 6 | 6 | 0 | 0 |
| quantization | Kernel | 6 | 6 | 0 | 0 |
| extension | Host-Library | 6 | 6 | 0 | 0 |
| comparison | Kernel | 6 | 6 | 6 | 6 |
| serialization | Host-Library | 6 | 6 | 0 | 0 |
| tokenizer | Host-Library | 5 | 5 | 0 | 0 |
| pooling | Kernel | 4 | 4 | 0 | 0 |
| sparse | Kernel | 4 | 3 | 0 | 0 |
| vision | Kernel | 4 | 4 | 0 | 0 |
| linalg_decomposition | Kernel | 4 | 4 | 0 | 0 |
| rl_loss | Kernel | 4 | 4 | 0 | 0 |
| model_layer | Kernel | 4 | 4 | 1 | 0 |
| state_update | AST-Frontend | 4 | 4 | 0 | 0 |
| conformance | Host-Library | 4 | 4 | 0 | 0 |
| position_encoding | Kernel | 3 | 3 | 1 | 0 |
| sort | Kernel | 3 | 2 | 0 | 0 |
| recurrent | Kernel | 3 | 3 | 0 | 0 |
| optimizer | AST-Frontend | 3 | 1 | 0 | 0 |
| memory | Kernel | 3 | 3 | 0 | 0 |
| sharding | IR-Transform | 3 | 3 | 0 | 0 |
| linalg_solver | Kernel | 2 | 2 | 0 | 0 |
| quantization | AST-Frontend | 2 | 2 | 0 | 0 |
| diffusion | Kernel | 2 | 2 | 0 | 0 |
| diffusion_schedule | Kernel | 2 | 2 | 0 | 0 |
| moe_transport | Kernel | 2 | 2 | 0 | 0 |
| rotary_embedding | Kernel | 2 | 2 | 1 | 0 |
| random_source | Host-Library | 2 | 2 | 0 | 0 |
| random_mask | Host-Library | 1 | 1 | 0 | 0 |
| ebm | Host-Library | 1 | 1 | 0 | 1 |
| contraction | Kernel | 1 | 1 | 1 | 0 |
| fused_epilogue | Kernel | 1 | 1 | 1 | 0 |
| numerics | Host-Library | 1 | 1 | 0 | 0 |
| moe | Kernel | 1 | 1 | 0 | 0 |
| projection | Kernel | 1 | 1 | 1 | 0 |
| segment_reduce | Kernel | 1 | 1 | 0 | 0 |
| state_space | Kernel | 1 | 1 | 0 | 0 |

## Untested ops (no fixture, name not found in tests/)


### Kernel (5)

`bsmm`, `dbar`, `dz`, `sin`, `sort`

### IR-Transform (14)

`flip`, `jvp`, `pack`, `pad`, `pmap`, `pmax`, `pmin`, `psum`, `roll`, `take`, `tile`, `view`, `vjp`, `vmap`

### AST-Frontend (7)

`cond`, `lamb`, `lion`, `map`, `muon`, `optax_style_chain`, `scan`

## Test-coverage reality

- **Name-in-test heuristic:** 448/474 ops appear in a test or carry a device
  fixture. The 26 flagged "untested" are mostly **false negatives** on short op
  names (`sin`, `pad`, `vjp`, `jvp`, `vmap`, `scan`, `cond`, `map`, `sort`) — these
  are core ops that *are* exercised, just under wrapper/param names the scan
  missed. Genuinely-thin ones to confirm: optimizer variants (`lamb`, `lion`,
  `muon`, `optax_style_chain`), GA helpers (`bsmm`, `dbar`, `dz`).
- **The meaningful gap is device-kernel coverage on the Kernel path:**
  **108 / 283 kernel-path ops have ≥1 real device kernel** (fused / compiled /
  hardware-verified); **175 are numpy-reference-only** (correct, but unoptimized
  — no codegen/device-lib kernel yet). This is the real enablement queue.
- **`tests` contract axis is uniformly `complete`** in the registry (semantic
  test exists for all 474) — so it is NOT the signal for "has an optimized
  device test". Use `execute_compare_fixture` (device numerical proof) for that.

## Rebuilt enablement plan

### 1. Kernel path — 283 ops (108 real / **175 reference-only**) — main thrust
Two sub-mechanisms, both proven this session:
- **Compiler codegen** — `generate-rocm-<fam>-kernel` MLIR pass → ROCDL → hsaco
  (ROCm), and hand-written AVX-512 C-ABI kernels loaded via the
  `libtessera_x86_elementwise.so` ctypes lane (x86); validated on-device vs numpy.
- **Device library** — Accelerate/BNNS + MPS/MPSGraph (Apple), WMMA/MFMA (ROCm),
  WGMMA/TMA (NVIDIA, HW-gated), AMX (x86). 189 kernel-path ops already map to a
  device-lib on ≥1 backend.

Priority queue for the 175 reference-only (model value first):
  1. **Normalization + activations + stable-reduction** (softmax_safe, group/
     instance_norm, activation tail) — same dual-device lane as elementwise.
  2. **Attention variants** (28 ops; 13 on ROCm) — extend the flash/linear-attn
     lanes to the rest; NVIDIA HW-gated.
  3. **pooling / conv (vision) / contraction / projection / fused_epilogue.**
  4. **loss (29) + rl_loss (4)** — compose from reduction + elementwise lanes.
  5. **spectral (9) / linalg (6) / quantize (8) / moe / diffusion** — specialised.
  6. **geometric_algebra (17) + ebm (13)** — exotic tail (Apple MSL today);
     ROCm/x86 last.

### 2. IR-Transform path — 82 ops — IR passes, tested by lit
sharding / collective-insertion / layout / scheduling / structural transform.
Enablement = the Graph/Schedule/Tile passes (mostly lit-verified). Action: confirm
a `tessera-opt` FileCheck fixture per pass; the 14 "untested" (`vmap`/`vjp`/`pad`/
`roll`/`take`/`tile`/`view`/`psum`/`pmax`…) are core — verify, don't assume.

### 3. AST-Frontend path — 49 ops — trace/transform, tested in Python
control flow (`cond`/`scan`/`while`/`map`), autodiff (`vjp`/`jvp`/grad transforms),
optimizer steps, state-tree ops. Backend execution rides the compiled lanes
(`run_graph_scan_f32` …). Action: add Python unit tests for the thin optimizer
variants (`lamb`/`lion`/`muon`/`optax_style_chain`).

### 4. Host-Library path — 61 ops — host-side, complete
data/tokenizers, serialization, AOT cache, RNG + MCMC samplers, EBM partition. No
device kernel needed; all carry Python unit tests. **Done — no action.**

### 5. TSOL (47 canonical) — 45 tested, 12 on ROCm
The standard-op surface; mostly Kernel-path. Drives backend_kernel partial→more-
targets as the Kernel-path queue lands. Track via `tsol_coverage.py`.

**Bottom line:** enablement is no longer "what path?" (this map answers that) — it
is **driving the 175 reference-only Kernel-path ops to real device kernels** on
ROCm + x86 (the proven dual-device lane), in the model-value order above. The
IR-Transform / AST-Frontend / Host-Library paths are already structurally enabled
and need only test-presence confirmation for a handful of thin ops.

## Kernel-technique reference: CUB (CCCL) / rocPRIM — study, reimplement, don't wrap

Tessera is a **standalone compiler** that generates its **own** kernels — it does
NOT take a CUB/rocPRIM runtime dependency. But those two libraries are the
gold-standard *reference implementations* of the data-parallel primitives, so we
**mine them for the algorithmic techniques + intrinsics** and reimplement those
patterns in the Tessera codegen lanes (`generate-rocm-<fam>-kernel`, the AVX-512
C-ABI kernels, the NVIDIA Target IR). They expose the same three tiers — **device
/ block / warp** — and a set of cross-lane **intrinsics**; the techniques port
1:1 between CUB and rocPRIM (rocPRIM mirrors CUB).

- **NVIDIA → CUB** (CCCL): <https://github.com/NVIDIA/cub>, <https://nvidia.github.io/cccl/>
- **AMD ROCm → rocPRIM**: <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim>,
  <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/reference/reference.html>

### Techniques to adopt (and the kernel each one upgrades)

| Technique (from CUB/rocPRIM) | Intrinsics to emit | Improves which Tessera kernel |
|------------------------------|--------------------|-------------------------------|
| **Warp-shuffle reduction** — each warp reduces in-register (butterfly `shfl_xor` / down-shift), only per-warp partials touch LDS → block reduce = warp-reduce + 1 LDS round-trip (vs O(blockDim) LDS tree) | NV `__shfl_down_sync`/`__shfl_xor_sync`; RDNA `ds_swizzle` / DPP (`v_*_dpp`) / `ds_permute` / `__shfl`; x86 `_mm512_reduce_*` | `generate-rocm-reduce-kernel` (replace the LDS tree-reduce), and the row-reduce inside `norm`/`softmax` + online-softmax in flash-attn |
| **Block scan** — warp-scan (Kogge-Stone via `shfl_up`) + scan of per-warp totals + uniform add-back | `__shfl_up_sync`; RDNA DPP `wave_shr`/`ds_permute` | new `scan` lane (cumsum/cumprod/`associative_scan`) |
| **Decoupled look-back single-pass scan** (CUB chained-scan) — device-wide scan/compaction in ONE kernel: each block publishes its aggregate, looks back at predecessors | atomics + `__threadfence` / `s_waitcnt` + ballot | device-wide `scan`, stream-compaction `select` |
| **Stream compaction via scan-of-flags** — predicate→flags→exclusive-scan→scatter survivors | x86 **`vpcompressd`/`_mm512_mask_compress`** (direct analog!); RDNA `__ballot` + `v_mbcnt_{lo,hi}` for per-lane offsets | `where`/`nonzero`/`masked_select`/`boolean_mask`, MoE token dispatch |
| **Radix sort** — per-digit: block count (ballot/`match_any`) → exclusive scan of digit offsets → scatter | `__ballot`, `__match_any_sync` / RDNA `v_cmp`+`s_ff1` | `sort`/`argsort`/`top_k` |
| **Vectorized blocked/striped load-store** (BlockLoad/BlockStore) — wide vector loads + coalesced/blocked arrangement instead of scalar per-element loads | RDNA `global_load_dwordx4`; NV `ld.global.v4`; x86 already 512-bit | all elementwise/reduce lanes (raise mem throughput) |
| **Welford online mean+variance** (single fused pass, numerically stable) | — (algorithmic) | `layer_norm`/`group_norm`/`instance_norm` (replace the two-pass) |
| **Transform-fused-into-reduce** (CUB `TransformInputIterator` pattern) — compute the pre-map (`exp`, `x²`) inline in the reduce loop, **no temp buffer** | — (codegen pattern) | `stable_reduction` (logsumexp, softmax-denom, var/L2-norm) collapse to one pass |

### How this folds into the plan

- These are **codegen upgrades**, not a new dependency — each technique is
  reimplemented inside the existing/new `generate-*-kernel` passes + AVX-512
  lanes, lowering through the hardware-free Target IR (Decision #19). Same
  algorithm shape emits the right intrinsics per backend (RDNA DPP/`ds_swizzle`,
  NV `__shfl`, AVX-512 `reduce`/`compress`).
- **Highest-leverage first move:** retrofit the landed
  `generate-rocm-reduce-kernel` from the LDS tree-reduce to a **warp-shuffle
  reduce** (DPP / `ds_swizzle`), validate bit-parity vs the current kernel on
  gfx1151, then reuse that warp-reduce primitive in the `norm`/`softmax` lanes.
  That single technique upgrades every reduction-bearing kernel.
- Then bring the new families online with the matching technique: `scan`
  (block-scan), `select`/`where` (compaction; x86 gets `vpcompress` "for free"),
  `sort` (radix), `histogram` — all reimplemented in Tessera codegen, inspired by
  CUB/rocPRIM but owned by us.

## Appendix — full per-op table

| op | category | path | lowering | effect | status | TSOL | tested | exec targets |
|----|----------|------|----------|--------|--------|------|--------|--------------|
| associative_scan | control_flow | AST-Frontend | control_flow | pure | partial |  | Y | — |
| cond | control_flow | AST-Frontend | control_flow | pure | partial |  | — | — |
| fori_loop | control_flow | AST-Frontend | control_flow | pure | partial |  | Y | — |
| map | control_flow | AST-Frontend | control_flow | pure | partial |  | — | — |
| scan | control_flow | AST-Frontend | control_flow | pure | partial |  | — | — |
| switch | control_flow | AST-Frontend | control_flow | pure | partial |  | Y | — |
| while_loop | control_flow | AST-Frontend | control_flow | pure | partial |  | Y | — |
| adafactor | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | Y | apple_cpu,cpu |
| adam | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | Y | apple_cpu,cpu |
| adamw | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | Y | apple_cpu,cpu |
| lion | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | — | apple_cpu,cpu |
| momentum | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | Y | apple_cpu,cpu |
| sgd | functional_optimizer_step | AST-Frontend | functional_optimizer_step | pure | partial |  | Y | apple_cpu,cpu |
| add_decoupled_weight_decay | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| centralize_grad | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| clip_grad_norm | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| clip_grad_value | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| ema_update | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| optax_style_chain | grad_transform | AST-Frontend | grad_transform | pure | partial |  | — | — |
| polyak_avg | grad_transform | AST-Frontend | grad_transform | pure | partial |  | Y | — |
| batched_gemm | loop_nest | AST-Frontend | loop_nest | pure | partial | Y | Y | apple_cpu,cpu,rocm |
| factorized_matmul | loop_nest | AST-Frontend | loop_nest | pure | partial | Y | Y | apple_cpu,cpu,rocm |
| gemm | loop_nest | AST-Frontend | loop_nest | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,x86 |
| grouped_gemm | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,apple_gpu,cpu |
| latent_kv_compress | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,cpu |
| latent_kv_expand_k | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,cpu |
| latent_kv_expand_v | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,cpu |
| matmul | loop_nest | AST-Frontend | loop_nest | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,nvidia_sm120,rocm,x86 |
| moe_swiglu_block | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,apple_gpu,cpu |
| quantized_matmul | loop_nest | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,cpu |
| lamb | optimizer | AST-Frontend | optimizer | pure | partial |  | — | — |
| muon | optimizer | AST-Frontend | optimizer | pure | partial |  | — | — |
| nesterov | optimizer | AST-Frontend | optimizer | pure | partial |  | Y | — |
| dequant_grouped_gemm | quantization | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,cpu |
| dequant_matmul | quantization | AST-Frontend | loop_nest | pure | partial |  | Y | apple_cpu,apple_gpu,cpu |
| empty_state_tree | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| module_state_tree | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| state_collection_spec | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| state_filter | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| state_partition | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| tree_flatten | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| tree_map | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| tree_reduce | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| tree_transpose | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| tree_unflatten | state_tree | AST-Frontend | state_tree | pure | partial |  | Y | — |
| kv_cache_append | state_update | AST-Frontend | state_update | state | partial |  | Y | apple_cpu,cpu |
| kv_cache_prune | state_update | AST-Frontend | state_update | state | partial |  | Y | apple_cpu,cpu |
| kv_cache_read | state_update | AST-Frontend | state_update | state | partial |  | Y | apple_cpu,apple_gpu,cpu |
| online_softmax_state | state_update | AST-Frontend | state_update | state | partial |  | Y | — |
| aot_export | aot | Host-Library | aot | pure | partial |  | Y | — |
| aot_load | aot | Host-Library | aot | pure | partial |  | Y | — |
| compilation_cache | aot | Host-Library | aot | pure | partial |  | Y | — |
| gguf_export | aot | Host-Library | aot | pure | partial |  | Y | — |
| safetensors_export | aot | Host-Library | aot | pure | partial |  | Y | — |
| stablehlo_export | aot | Host-Library | aot | pure | partial |  | Y | — |
| tiny_attention_conformance | conformance | Host-Library | conformance | pure | partial |  | Y | — |
| tiny_diffusion_conformance | conformance | Host-Library | conformance | pure | partial |  | Y | — |
| tiny_recurrent_conformance | conformance | Host-Library | conformance | pure | partial |  | Y | — |
| tiny_training_step_conformance | conformance | Host-Library | conformance | pure | partial |  | Y | — |
| dataset_batch | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_checkpoint | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_filter | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_interleave | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_map | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_prefetch | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_repeat | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_shuffle | data | Host-Library | data | pure | partial |  | Y | — |
| dataset_zip | data | Host-Library | data | pure | partial |  | Y | — |
| iterable_dataset | data | Host-Library | data | pure | partial |  | Y | — |
| sharded_dataset | data | Host-Library | data | pure | partial |  | Y | — |
| ebm_langevin_step | ebm | Host-Library | ebm | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| custom_batching | extension | Host-Library | extension | pure | partial |  | Y | — |
| custom_call | extension | Host-Library | extension | pure | partial |  | Y | — |
| custom_jvp | extension | Host-Library | extension | pure | partial |  | Y | — |
| custom_lowering | extension | Host-Library | extension | pure | partial |  | Y | — |
| custom_primitive | extension | Host-Library | extension | pure | partial |  | Y | — |
| custom_vjp | extension | Host-Library | extension | pure | partial |  | Y | — |
| grad_scaler_step | numerics | Host-Library | numerics | pure | partial |  | Y | — |
| dropout | random_mask | Host-Library | random_mask | random | partial | Y | Y | apple_cpu,cpu |
| rng_normal | random_source | Host-Library | random_source | random | partial | Y | Y | apple_cpu,cpu |
| rng_uniform | random_source | Host-Library | random_source | random | partial | Y | Y | apple_cpu,cpu |
| rng_bernoulli | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_beta | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_categorical | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_clone | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_dirichlet | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_fold_in | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_gamma | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_gibbs_sample | rng | Host-Library | (none) | pure | planned |  | Y | — |
| rng_hmc_sample | rng | Host-Library | (none) | pure | planned |  | Y | — |
| rng_key | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_langevin_sample | rng | Host-Library | (none) | pure | planned |  | Y | — |
| rng_mala_sample | rng | Host-Library | (none) | pure | planned |  | Y | — |
| rng_multinomial | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_permutation | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_poisson | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_randint | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_split | rng | Host-Library | rng | pure | partial |  | Y | — |
| rng_truncated_normal | rng | Host-Library | rng | pure | partial |  | Y | — |
| load_sharded | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| load_state | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| partial_state_load | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| save_sharded | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| save_state | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| state_migration | serialization | Host-Library | serialization | pure | partial |  | Y | — |
| tokenizer_bpe | tokenizer | Host-Library | tokenizer | pure | partial |  | Y | — |
| tokenizer_byte | tokenizer | Host-Library | tokenizer | pure | partial |  | Y | — |
| tokenizer_sentencepiece_compat | tokenizer | Host-Library | tokenizer | pure | partial |  | Y | — |
| tokenizer_unigram | tokenizer | Host-Library | tokenizer | pure | partial |  | Y | — |
| tokenizer_wordpiece | tokenizer | Host-Library | tokenizer | pure | partial |  | Y | — |
| all_gather | collective | IR-Transform | collective | collective | partial | Y | Y | apple_cpu,cpu |
| all_reduce | collective | IR-Transform | collective | collective | partial | Y | Y | apple_cpu,cpu |
| all_to_all | collective | IR-Transform | collective | collective | partial | Y | Y | apple_cpu,cpu |
| broadcast_to_axis | collective | IR-Transform | collective | pure | partial |  | Y | — |
| collective_permute | collective | IR-Transform | collective | pure | partial |  | Y | — |
| pmax | collective | IR-Transform | collective | pure | partial |  | — | — |
| pmean | collective | IR-Transform | collective | pure | partial |  | Y | — |
| pmin | collective | IR-Transform | collective | pure | partial |  | — | — |
| psum | collective | IR-Transform | collective | pure | partial |  | — | — |
| reduce_scatter | collective | IR-Transform | collective | collective | partial | Y | Y | apple_cpu,cpu |
| dynamic_slice | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| dynamic_update_slice | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| index_select | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| index_update | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| masked_categorical | indexing | IR-Transform | indexing | random | partial |  | Y | apple_cpu,cpu |
| masked_scatter | indexing | IR-Transform | indexing | pure | partial |  | Y | — |
| memory_index_select | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| memory_index_select_ste | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| msa_select_blocks | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| nonzero | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| scatter | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| scatter_add | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| scatter_reduce | indexing | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| take | indexing | IR-Transform | indexing | pure | partial |  | — | apple_cpu,cpu |
| arange | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| cast | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| gather | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| masked_fill | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| mor_partition | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| mor_router | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| mor_scatter | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| pack | layout_transform | IR-Transform | layout_transform | movement | partial | Y | — | apple_cpu,cpu |
| patchify | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | — |
| pixel_shuffle | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | — |
| pixel_unshuffle | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | — |
| rearrange | layout_transform | IR-Transform | layout_transform | pure | partial | Y | Y | apple_cpu,cpu |
| rope_merge | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| rope_split | layout_transform | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| tile_view | layout_transform | IR-Transform | layout_transform | pure | partial | Y | Y | apple_cpu,cpu |
| transpose | layout_transform | IR-Transform | layout_transform | pure | partial | Y | Y | apple_cpu,cpu |
| unpack | layout_transform | IR-Transform | layout_transform | movement | partial | Y | Y | apple_cpu,cpu |
| chained_schedule | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| constant_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| cosine_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| cosine_warmup_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| cyclical_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| inverse_sqrt_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| linear_warmup_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| polynomial_lr | schedule | IR-Transform | schedule | pure | partial |  | Y | — |
| named_sharding | sharding | IR-Transform | sharding | pure | partial |  | Y | — |
| partition_spec | sharding | IR-Transform | sharding | pure | partial |  | Y | — |
| shard_map | sharding | IR-Transform | sharding | pure | partial |  | Y | — |
| broadcast | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| cat | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| chunk | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| expand | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| flatten | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| flip | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | — | apple_cpu,cpu |
| pad | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | — | apple_cpu,cpu |
| permute | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| repeat | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| reshape | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| roll | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | — | apple_cpu,cpu |
| select | tensor_algebra | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| slice | tensor_algebra | IR-Transform | indexing | pure | partial |  | Y | apple_cpu,cpu |
| split | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| squeeze | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| stack | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| tile | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | — | apple_cpu,cpu |
| unsqueeze | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | Y | apple_cpu,cpu |
| view | tensor_algebra | IR-Transform | layout_transform | pure | partial |  | — | apple_cpu,cpu |
| autocast | transform | IR-Transform | transform | pure | partial |  | Y | — |
| axis_index | transform | IR-Transform | transform | pure | partial |  | Y | — |
| axis_name | transform | IR-Transform | transform | pure | partial |  | Y | — |
| axis_size | transform | IR-Transform | transform | pure | partial |  | Y | — |
| checkpoint | transform | IR-Transform | transform | pure | partial |  | Y | — |
| jvp | transform | IR-Transform | transform | pure | partial |  | — | — |
| pmap | transform | IR-Transform | transform | pure | partial |  | — | — |
| remat | transform | IR-Transform | transform | pure | partial |  | Y | — |
| value_and_grad | transform | IR-Transform | transform | pure | partial |  | Y | — |
| vjp | transform | IR-Transform | transform | pure | partial |  | — | — |
| vmap | transform | IR-Transform | transform | pure | partial |  | — | — |
| attn_compressed_blocks | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| attn_local_window_2d | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| attn_sliding_window | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| attn_top_k_blocks | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| cross_attention | attention | Kernel | attention | pure | partial |  | Y | — |
| deepseek_sparse_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| flash_attn | attention | Kernel | attention | state | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| gated_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| gated_deltanet | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| gqa_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| hybrid_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| kimi_delta_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| lightning_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| linear_attn | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| linear_attn_state | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| lookahead_sparse_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| memory_index_score | attention | Kernel | attention | pure | partial |  | Y | apple_cpu,cpu |
| mla_decode | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| mla_decode_fused | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| modified_delta_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| mqa_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| msa_index_scores | attention | Kernel | attention | pure | partial |  | Y | apple_cpu,cpu |
| msa_sparse_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| multi_head_attention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu,rocm |
| perceiver_resampler | attention | Kernel | attention | pure | partial |  | Y | — |
| power_attn | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| retention | attention | Kernel | attention | state | partial |  | Y | apple_cpu,cpu |
| varlen_sdpa | attention | Kernel | attention | pure | partial |  | Y | apple_cpu,cpu |
| eq | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| ge | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| gt | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| le | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| lt | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| ne | comparison | Kernel | comparison | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| einsum | contraction | Kernel | contraction | pure | partial | Y | Y | apple_cpu,cpu,rocm |
| edm_loss_weight | diffusion | Kernel | diffusion | pure | partial |  | Y | — |
| edm_precondition | diffusion | Kernel | diffusion | pure | partial |  | Y | — |
| equiprob_band_partition | diffusion_schedule | Kernel | diffusion_schedule | pure | partial |  | Y | — |
| karras_sigma_schedule | diffusion_schedule | Kernel | diffusion_schedule | pure | partial |  | Y | — |
| ebm_bivector_langevin_sample | ebm | Kernel | (none) | pure | planned |  | Y | apple_cpu,x86 |
| ebm_bivector_langevin_step | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_decode_init | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_energy | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_energy_quadratic | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_inner_step | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_partition_ais | ebm | Kernel | (none) | pure | planned |  | Y | apple_cpu,x86 |
| ebm_partition_exact | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_partition_monte_carlo | ebm | Kernel | (none) | pure | planned |  | Y | apple_cpu,x86 |
| ebm_refinement | ebm | Kernel | elementwise | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_self_verify | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| ebm_sphere_langevin_sample | ebm | Kernel | (none) | pure | planned |  | Y | apple_cpu,x86 |
| ebm_sphere_langevin_step | ebm | Kernel | (none) | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| acos | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| add | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| asin | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| atan | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| atan2 | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| clip | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| complex_abs | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_arg | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_conjugate | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_div | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_exp | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,apple_gpu,cpu,x86 |
| complex_log | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_mul | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,apple_gpu,cpu,x86 |
| complex_pow | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| complex_sqrt | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| cos | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| cosh | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| cross_ratio | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| digamma | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| div | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| erf | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| erfc | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| exp | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| expm1 | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| floor_div | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| gelu | elementwise | Kernel | elementwise | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| is_concyclic | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| lgamma | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| log | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| log1p | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| mobius | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| mobius_from_three_points | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,x86 |
| mod | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| mul | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| popcount | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| pow | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| relu | elementwise | Kernel | elementwise | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu |
| rsqrt | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| score_combine | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| sigmoid | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| silu | elementwise | Kernel | elementwise | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| silu_mul | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| sin | elementwise | Kernel | elementwise | pure | partial |  | — | apple_cpu,cpu |
| sinh | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| softcap | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| softplus | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| sqrt | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| stereographic | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu |
| sub | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| tan | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| tanh | elementwise | Kernel | elementwise | pure | partial |  | Y | apple_cpu,cpu,rocm |
| fused_epilogue | fused_epilogue | Kernel | fused_epilogue | pure | partial |  | Y | apple_cpu,cpu,rocm |
| clifford_codiff | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_conjugate | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_exp | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_ext_deriv | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_geometric_product | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_grade_involution | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_grade_projection | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_hodge_star | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_inner | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_integral | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_left_contraction | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_log | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_norm | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_reverse | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_rotor_sandwich | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_vec_deriv | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| clifford_wedge | geometric_algebra | Kernel | (none) | pure | planned |  | Y | apple_cpu,apple_gpu,x86 |
| cholesky | linalg_decomposition | Kernel | linalg_decomposition | pure | partial | Y | Y | apple_cpu,cpu |
| lu | linalg_decomposition | Kernel | linalg_decomposition | pure | partial |  | Y | apple_cpu,cpu |
| qr | linalg_decomposition | Kernel | linalg_decomposition | pure | partial | Y | Y | apple_cpu,cpu |
| svd | linalg_decomposition | Kernel | linalg_decomposition | pure | partial | Y | Y | apple_cpu,cpu |
| cholesky_solve | linalg_solver | Kernel | linalg_solver | pure | partial |  | Y | apple_cpu,cpu |
| tri_solve | linalg_solver | Kernel | linalg_solver | pure | partial | Y | Y | apple_cpu,cpu |
| bitwise_and | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| bitwise_not | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| bitwise_or | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| bitwise_xor | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| logical_and | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| logical_not | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| logical_or | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| logical_xor | logical | Kernel | logical | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| asymmetric_bce | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| binary_cross_entropy_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| contrastive_divergence_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| contrastive_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| cosine_embedding_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| cross_entropy_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| ctc_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| ddpm_noise_pred_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| denoising_score_matching_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| focal_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| huber_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| implicit_score_matching_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| info_nce_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| js_divergence | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| kl_divergence | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| label_smoothed_cross_entropy | loss | Kernel | loss | pure | partial |  | Y | — |
| load_balance_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| log_cosh_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| mae_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| mse_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| nt_xent_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| persistent_cd_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| score_matching_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| seq2seq_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| smooth_l1_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| triplet_loss | loss | Kernel | loss | pure | partial |  | Y | — |
| vlb_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| wasserstein_distance | loss | Kernel | loss | pure | partial |  | Y | — |
| z_loss | loss | Kernel | loss | pure | partial |  | Y | apple_cpu,cpu |
| memory_evict | memory | Kernel | memory | pure | partial |  | Y | — |
| memory_read | memory | Kernel | memory | pure | partial |  | Y | — |
| memory_write | memory | Kernel | memory | pure | partial |  | Y | — |
| conv1d | model_layer | Kernel | model_layer | pure | partial |  | Y | — |
| conv_transpose | model_layer | Kernel | model_layer | pure | partial |  | Y | — |
| linear_general | model_layer | Kernel | model_layer | pure | partial |  | Y | apple_cpu,cpu,rocm |
| lora_linear | model_layer | Kernel | model_layer | pure | partial |  | Y | — |
| moe | moe | Kernel | moe | collective | partial | Y | Y | apple_cpu,cpu |
| moe_combine | moe_transport | Kernel | moe_transport | collective | partial | Y | Y | apple_cpu,cpu |
| moe_dispatch | moe_transport | Kernel | moe_transport | collective | partial | Y | Y | apple_cpu,cpu |
| group_norm | normalization | Kernel | normalization | pure | partial |  | Y | apple_cpu,cpu |
| instance_norm | normalization | Kernel | normalization | pure | partial |  | Y | apple_cpu,cpu |
| layer_norm | normalization | Kernel | normalization | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| rmsnorm | normalization | Kernel | normalization | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| rmsnorm_safe | normalization | Kernel | normalization | pure | partial |  | Y | apple_cpu,cpu |
| spectral_norm | normalization | Kernel | normalization | pure | partial |  | Y | — |
| weight_norm | normalization | Kernel | normalization | pure | partial |  | Y | apple_cpu,cpu |
| abs | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| absolute | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| ceil | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| clamp | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| floor | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| isfinite | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| isinf | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| isnan | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| maximum | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| minimum | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| reciprocal | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| round | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| sign | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| trunc | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| where | numeric_helper | Kernel | numeric_helper | pure | partial |  | Y | apple_cpu,cpu |
| adaptive_pool | pooling | Kernel | pooling | pure | partial |  | Y | — |
| avg_pool | pooling | Kernel | pooling | pure | partial |  | Y | — |
| max_pool | pooling | Kernel | pooling | pure | partial |  | Y | — |
| min_pool | pooling | Kernel | pooling | pure | partial |  | Y | — |
| alibi | position_encoding | Kernel | position_encoding | pure | partial |  | Y | apple_cpu,cpu,rocm |
| factorized_pos_emb | position_encoding | Kernel | position_encoding | pure | partial |  | Y | — |
| ntk_rope | position_encoding | Kernel | position_encoding | pure | partial |  | Y | apple_cpu,cpu |
| qkv_projection | projection | Kernel | projection | pure | partial | Y | Y | apple_cpu,cpu,rocm |
| calibration_observer | quantization | Kernel | quantization | pure | partial |  | Y | — |
| dequantize_int4 | quantization | Kernel | quantization | pure | partial |  | Y | — |
| dequantize_int8 | quantization | Kernel | quantization | pure | partial |  | Y | — |
| fake_quantize | quantization | Kernel | quantization | pure | partial |  | Y | — |
| quantize_int4 | quantization | Kernel | quantization | pure | partial |  | Y | — |
| quantize_int8 | quantization | Kernel | quantization | pure | partial |  | Y | — |
| dequantize_fp4 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| dequantize_fp6 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| dequantize_fp8 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| dequantize_nvfp4 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| quantize_fp4 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| quantize_fp6 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| quantize_fp8 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| quantize_nvfp4 | quantize | Kernel | quantize | pure | partial |  | Y | apple_cpu,cpu |
| bidirectional_scan | recurrent | Kernel | recurrent | pure | partial |  | Y | — |
| gru_cell | recurrent | Kernel | recurrent | pure | partial |  | Y | — |
| simple_rnn_cell | recurrent | Kernel | recurrent | pure | partial |  | Y | — |
| amax | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| amin | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| argmax | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| argmin | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| clifford_norm_squared | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,apple_gpu,x86 |
| count_nonzero | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| cummax | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| cummin | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| cumprod | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| cumsum | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| max | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| mean | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| min | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| prod | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| std | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| var | reduction | Kernel | reduction | pure | partial |  | Y | apple_cpu,cpu |
| cispo_policy_loss | rl_loss | Kernel | rl_loss | pure | partial |  | Y | apple_cpu,cpu |
| grpo_policy_loss | rl_loss | Kernel | rl_loss | pure | partial |  | Y | apple_cpu,cpu |
| normalize_group_advantages | rl_loss | Kernel | rl_loss | pure | partial |  | Y | apple_cpu,cpu |
| ppo_policy_loss | rl_loss | Kernel | rl_loss | pure | partial |  | Y | apple_cpu,cpu |
| mrope_2d | rotary_embedding | Kernel | rotary_embedding | pure | partial |  | Y | — |
| rope | rotary_embedding | Kernel | rotary_embedding | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| segment_reduce | segment_reduce | Kernel | segment_reduce | pure | partial | Y | Y | apple_cpu,cpu |
| argsort | sort | Kernel | sort | pure | partial |  | Y | apple_cpu,cpu |
| sort | sort | Kernel | sort | pure | partial |  | — | apple_cpu,cpu |
| top_k | sort | Kernel | sort | pure | partial |  | Y | apple_cpu,cpu |
| bsmm | sparse | Kernel | sparse | pure | partial | Y | — | apple_cpu,cpu |
| sddmm | sparse | Kernel | sparse | pure | partial | Y | Y | apple_cpu,cpu |
| spmm_coo | sparse | Kernel | sparse | pure | partial | Y | Y | apple_cpu,cpu |
| spmm_csr | sparse | Kernel | sparse | pure | partial | Y | Y | apple_cpu,cpu |
| dct | spectral | Kernel | spectral | pure | partial |  | Y | apple_cpu,cpu |
| fft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| ifft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| irfft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| istft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| rfft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| spectral_conv | spectral | Kernel | spectral | pure | partial |  | Y | apple_cpu,cpu |
| spectral_filter | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| stft | spectral | Kernel | spectral | pure | partial | Y | Y | apple_cpu,cpu |
| conformal_energy_on_sphere | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu,x86 |
| log_softmax | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu |
| logsumexp | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu |
| online_softmax | stable_reduction | Kernel | stable_reduction | state | partial |  | Y | — |
| reduce | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu |
| sigmoid_safe | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu |
| softmax | stable_reduction | Kernel | stable_reduction | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu,rocm |
| softmax_safe | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,apple_gpu,cpu |
| sum | stable_reduction | Kernel | stable_reduction | pure | partial |  | Y | apple_cpu,cpu,rocm,x86 |
| selective_ssm | state_space | Kernel | state_space | state | partial |  | Y | apple_cpu,cpu |
| check_cauchy_riemann | stencil | Kernel | stencil | pure | partial |  | Y | apple_cpu,cpu,x86 |
| conformal_jacobian | stencil | Kernel | stencil | pure | partial |  | Y | apple_cpu,cpu,x86 |
| conv2d | stencil | Kernel | stencil | pure | partial | Y | Y | apple_cpu,apple_gpu,cpu |
| conv3d | stencil | Kernel | stencil | pure | partial | Y | Y | apple_cpu,cpu |
| dbar | stencil | Kernel | stencil | pure | partial |  | — | apple_cpu,cpu,x86 |
| depthwise_conv1d | stencil | Kernel | stencil | state | partial |  | Y | — |
| dz | stencil | Kernel | stencil | pure | partial |  | — | apple_cpu,cpu,x86 |
| laplacian_2d | stencil | Kernel | stencil | pure | partial |  | Y | apple_cpu,cpu,x86 |
| center_crop | vision | Kernel | vision | pure | partial |  | Y | — |
| image_normalize | vision | Kernel | vision | pure | partial |  | Y | — |
| image_resize | vision | Kernel | vision | pure | partial |  | Y | — |
| interpolate | vision | Kernel | vision | pure | partial |  | Y | — |
