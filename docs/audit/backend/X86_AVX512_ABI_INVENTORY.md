---
last_updated: 2026-07-20
audit_role: reference
owner: X86-E2E-2
sync_key: X86-E2E2-COHORT2-2026-07-20
---

# X86 AVX-512 stable-ABI inventory

This is the first X86-E2E-2 inventory snapshot. It was extracted with `nm -D
--defined-only` from the LLVM 23 clean-build
`libtessera_x86_elementwise.so` on an AMD Ryzen AI MAX+ 395 host with AVX-512F,
DQ, BW, VL, VNNI, BF16, VBMI/VBMI2, BITALG, VPOPCNTDQ, and VP2INTERSECT.

The image exports 76 `tessera_x86_*` symbols: 31 AVX-512 direct entries, 19
portable reference entries, and 26 other direct/specialized entries. A symbol
is evidence that a native ABI exists, not that canonical compiler E2E exists.
Only rows with a typed Tile producer, typed lowering, image/descriptor package,
exact-host proof, and measured selector decision reach Level C.

## Direct AVX-512 entries (31)

| Cohort | Symbols | X86-E2E-2 disposition |
|---|---|---|
| Flat elementwise | `avx512_unary_f32`, `avx512_binary_f32`, `avx512_predicate_f32` | First cohort landing: 20 operation kinds, three stable ABIs, typed `tile.elementwise_kernel`, exact-host proof, measured size-aware selection. |
| Typed logic | `avx512_compare_f32`, `avx512_logical_i8`, `avx512_bitwise_i32` | Three follow-on slices landing: 15 operation kinds, typed f32/bool/i32 contracts, exact-host proof, and measured family-specific selectors. |
| Remaining flat elementwise | `avx512_where_f32`, `avx512_transcendental_f32`, `avx512_pow_f32`, `avx512_silu_mul_f32` | Three follow-on slices landed with typed ABIs, exact-host correctness, and measured selectors. Transcendental promotes at every valid static size, pow/SiLU-multiply at 8,224 elements, and where at 1,048,576 elements. |
| Reduction/normalization/position | `avx512_reduce_f32`, `avx512_argreduce_f32`, `avx512_scan_f32`, `avx512_rmsnorm_f32`, `avx512_layernorm_f32`, `avx512_softmax_f32`, `avx512_rope_f32`, `avx512_alibi_f32` | Reduction and softmax are X86-E2E-1. Cohort 2 now has typed argreduce, scan, normalization, RoPE, and ALiBi carriers, descriptors, exact-host proof, and retained-route measurements. |
| Dense/sparse compute | `avx512_gemm_f32`, `avx512_spmm_csr_f32`, `avx512_sddmm_f32` | GEMM is X86-E2E-1; sparse rows are cohort 3. |
| Loss/quantization/state | `avx512_pointwise_loss_f32`, `avx512_binary_loss_f32`, `avx512_policy_loss_f32`, `avx512_fpquant_f32`, `avx512_selective_ssm_f32`, `avx512_selective_ssm_f16`, `avx512_selective_ssm_bf16` | Cohort 4; half/BF16 storage remains a separate dtype/ABI proof. |
| Datatype matmul | `avx512_gemm_bf16`, `avx512_vnni_gemm_u8s8_s32`, `avx512_gemm_f64` | BF16→FP32, U8×S8→S32, and FP64 typed descriptors execute on the Ryzen AI Max+ 395. Kernel/reference and descriptor timing is recorded; automatic selector promotion remains unchanged. |

Every symbol in this table has the full exported prefix
`tessera_x86_`; the shortened names keep the inventory readable.

## Portable reference entries (19)

These remain explicit oracle/comparison entries and are not selector candidates:

`reference_argreduce_f32`, `reference_binary_f32`, `reference_bitwise_i32`,
`reference_compare_f32`, `reference_logical_i8`, `reference_moe_f32`,
`reference_optimizer_f32`, `reference_predicate_f32`, `reference_reduce_f32`,
`reference_scan_f32`, `reference_sddmm_f32`, `reference_selective_ssm_f32`,
`reference_spmm_csr_f32`, `reference_transcendental_f32`,
`reference_unary_f32`, `reference_where_f32`, `reference_gemm_bf16`,
`reference_gemm_u8s8_s32`, and `reference_gemm_f64`.

## Other direct and specialized entries (26)

| Family | Symbols | Cohort |
|---|---|---|
| Attention/cache | `flash_attn_f32`, `flash_attn_ext_f32`, `kv_cache_append_f32`, `kv_cache_read_f32`, `kv_cache_prune_f32` | Attention is X86-E2E-1; stateful cache remains cohort 4. |
| Gather/scatter/sort/spectral | `gather_f32`, `scatter_f32`, `bitonic_sort_kv_f32`, `fft_c2c_f32` | Cohort 3. |
| Linalg/sparse | `cholesky_f32`, `tri_solve_f32`, `lu_f32`, `qr_f32`, `svd_f32` | Cohort 3. |
| MoE/optimizer/state | `moe_f32`, `optimizer_f32`, `deltanet_f32`, `selective_ssm_bwd_f32` | Cohort 4. |
| RNG/EBM | `philox_uniform_f32`, `ebm_affine_langevin_f32`, `ebm_decode_init_noise_apply_f32`, `ebm_ebt_tiny_f32`, `ebm_energy_quadratic_f32`, `ebm_langevin_philox_f32`, `ebm_partition_exact_f32` | Cohort 4; state ownership must be explicit. |
| Algebra | `clifford_bilinear_f32` | Cohort 3. |

The runtime also contains host-orchestrated compiler paths that call one or
more of these entries. They do not become stable-ABI compiler lanes merely by
composition. X86-E2E-2 will either introduce one entry owning the complete
semantics or keep the composition explicit.

## First-cohort measured decision

The retained comparison is recorded in
`benchmarks/baselines/x86_avx512_e2e_elementwise_comparison.json` using 41
serial alternating CPU-wall trials. Unary and predicate descriptors meet the
10% bound at all three measured sizes. Binary has a fixed 4–6 microsecond
descriptor cost on the two sub-16K rows, but wins at the large row; a focused
crossover sweep first passed the bound at 16,384 elements. Canonical selection
therefore promotes unary and predicate for every valid static shape and binary
only at 16,384 elements or larger. Explicit typed packaging remains legal for
smaller binary tensors, and the retained executor remains the default there.

The compare/logical/bitwise record is
`benchmarks/baselines/x86_avx512_e2e_typed_logic_comparison.json`. Logical
passes the retained-route bound from 130 elements upward and is promoted for
every valid static shape. Compare and bitwise have fixed small-shape validation
cost and use a conservative 32,768-element automatic-promotion floor. At and
above that committed threshold both pass the 10% bound; smaller shapes remain
on `x86_compare_compiled` or `x86_bitwise_compiled`. Explicit descriptor
packaging remains available.

## Flat follow-on and datatype decisions

`benchmarks/baselines/x86_avx512_e2e_flat_followon_comparison.json` records 21
serial alternating retained/descriptor CPU-wall trials for where,
transcendental, pow, and SiLU-multiply at 130, 8,224, and 1,048,576 elements.
Every row is correct. Transcendental meets the 10% bound at all measured sizes;
pow and SiLU-multiply cross it at 8,224; where is conservatively promoted only
at the directly measured 1,048,576-element winner.

`benchmarks/baselines/x86_avx512_e2e_dtype_matmul_comparison.json` records
BF16, VNNI U8/S8, and FP64 aligned/ragged/large rows. All nine descriptors
match their typed references. Native-kernel speedups span 1.283--12.212x;
descriptor medians span 0.0290--0.1128 ms. The shared image disassembles to
`vdpbf16ps`, `vpdpbusd`, and packed-double FMA respectively. This is execution
and performance evidence, but it does not promote the new dtype descriptors
over a production route without a matching retained-route policy.

## Cohort-2 decision

`benchmarks/baselines/x86_avx512_e2e_cohort2_comparison.json` records 21 serial
alternating retained/descriptor CPU-wall trials on the Ryzen AI Max+ 395 for
argreduce, scan, RMSNorm, LayerNorm, RoPE, and ALiBi. Every descriptor matches
the retained executor. Median speedups are 0.980x, 0.986x, 1.103x, 1.146x,
1.030x, and 3.276x respectively. The compiler path is therefore executable and
measured, but automatic selection remains unchanged: the record contains one
representative shape per family and is not a size-crossover corpus.

## Cohort-3/4 closure

Cross-backend sync `X86-E2E2-BREADTH-2026-07-20` introduces the target-owned
`tile.x86_abi_kernel` carrier and `x86_breadth.py` total ABI registry. All 12
cohort-3 and 21 cohort-4 direct/specialized symbols have a versioned ABI ID,
exact ordered pointer/scalar schema, buffer direction, dtype, effect class,
status-return policy, and honest public-route disposition. The generic runtime
launcher consumes that compiler-produced schema and supports multi-output,
in-place, and status-returning calls.

LLVM 23 FileCheck proves representative sparse and stateful carriers lower to
the exact declared signatures. Exact Ryzen AI Max+ 395 execution proves the
complete export inventory plus Graph-descriptor gather, unreduced pointwise
loss, Cholesky, triangular solve, and stateful KV append including mutation and
zero status.

`benchmarks/baselines/x86_avx512_e2e_cohort34_comparison.json` is the
operation-total selector record. Its 12 measured rows use 21 serial alternating
retained/descriptor trials and cover three domains for each semantically
isomorphic Graph family. Automatic promotion begins at 1,048,576 gather
outputs, 16,384 unreduced loss elements, 2,048 Cholesky output elements, and
512 triangular-solve output elements. The record also carries all 33 ABI
dispositions; the remaining 29 are retained because no equivalent direct Graph
candidate exists. Host-packed SDDMM/Clifford/sort, radix-2-only FFT,
per-element/reduced loss distinctions, multi-output/stateful calls, and the
Philox-uniform core remain explicitly narrower than their public composed
operations. X86-E2E-2 is closed without converting those retained decisions
into false direct-support claims.
