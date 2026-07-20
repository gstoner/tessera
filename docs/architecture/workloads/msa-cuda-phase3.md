---
classification: Architecture / Workload Plan
authority: CUDA detail for the MSA workload
last_updated: 2026-07-13
---

# MSA Phase 3 — CUDA/NVIDIA KV-Outer Sparse Attention Plan

> **Widened 2026-07-09:** this NVIDIA-only plan is now the CUDA column of the
> consolidated tri-backend plan
> [`attention-family.md`](attention-family.md), which
> covers the whole DFlash / MSA / Mamba2 family across ROCm, CUDA, and x86. Start
> there for the cross-backend status matrix and sequencing; this doc remains the
> detailed CUDA KV-outer contract.

This is the compiler-payoff step for MiniMax Sparse Attention after the Phase
0-2 API/Graph IR contract and selected-block layout verifier. The Python
Graph→Schedule→Tile→Target artifact path now emits the KV-outer sparse
attention contract below; a native CUDA/Hopper/Blackwell kernel remains future
work.

## Target Contract

Lower `tessera.msa_sparse_attention` to a CUDA Tile IR target shaped as
`tessera_attn.msa_kv_outer_sparse`.

Inputs:

| Tensor | Layout |
|---|---|
| `Q` | `memref<BxHqxSqxDxbf16/f16/f32>` |
| `K`, `V` | `memref<BxHkvxSkxDxbf16/f16/f32>` |
| `block_ids` | `memref<BxHkvxSqxtop_kxi64>` |
| `O` | `memref<BxHqxSqxDxbf16/f16/f32>` |

Attributes:

| Attribute | Meaning |
|---|---|
| `block_size` | KV tokens per selected block |
| `top_k` | selected blocks per `(B,Hkv,Sq)` row |
| `gqa_group_size` | `Hq / Hkv`; maps query head `h` to selector group `h / gqa_group_size` |
| `tile_q` | query tile height |
| `tile_kv` | selected KV tile width |
| `head_dim` | per-head feature dimension |
| `mode` | `"prefill"` or `"decode"` |
| `acc_dtype` | accumulator dtype, initially `"fp32"` |
| `dense_equivalence_oracle` | when true, `top_k == num_blocks` must match dense GQA |

## Prefill Path

Prefill uses the full selected-block worklist:

1. Materialize/consume `block_ids` as `(B,Hkv,Sq,top_k)`.
2. For each query tile and GQA group, iterate selected KV blocks.
3. Run KV-outer block-sparse online softmax over selected tokens.
4. Apply token-level causal masking inside the selected blocks when `causal`.

The first correctness oracle remains dense equivalence: with
`top_k == Sk / block_size`, output must match dense GQA attention.

## Decode Path

Decode uses the same `block_ids` layout but `Sq` is usually one or a small
micro-batch. The lowering should avoid prefill-sized staging:

1. Use the current decode query row's selected block ids directly.
2. Keep K/V reads KV-outer and block-granular.
3. Reuse online-softmax state per query head.
4. Preserve the same dense-equivalence oracle for `top_k == num_blocks`.

## Landed Artifact Path

`lower_graph_to_schedule_ir` maps `tessera.msa_sparse_attention` to
`schedule.attn.kv_outer_sparse` with explicit `block_ids_layout =
"B,Hkv,Sq,top_k"`, `gqa_group_size`, `tile_q`, `tile_kv`, `head_dim`, `mode`,
and `kv_traversal = "kv_outer"` metadata. `lower_schedule_to_tile_ir` preserves
that contract as `tessera_attn.msa_kv_outer_sparse`, and NVIDIA Target IR emits
an `artifact_only` `tessera_nvidia.cuda_kernel` with `kernel =
"msa_kv_outer_sparse"`.

Guard: `tests/unit/test_msa_kv_outer_schedule.py`.

## Fixture

[`msa_kv_outer_sparse_attention.mlir`](../../../tests/tessera-ir/phase3/cuda13/msa_kv_outer_sparse_attention.mlir)
locks the Tile IR target name and selected-block worklist metadata. It is a
contract fixture, not proof of a native kernel implementation.
