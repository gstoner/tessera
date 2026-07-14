# Apple GPU kernel + lane guide (reference)

> This is a **human guide to kernel families, ABI conventions, and constraints**.
> It is not a competing status ledger. The **machine-readable** truth for the
> full exported C-ABI symbol surface is
> the generated, drift-gated
> [`docs/audit/generated/runtime_abi.csv`](../../audit/generated/runtime_abi.csv)
> (human summary: [`runtime_abi.md`](../../audit/generated/runtime_abi.md)). The op ×
> target × dtype matrix is
> [`apple_target_map.md`](../../audit/generated/apple_target_map.md);
> what actually executes per lane is
> [`runtime_execution_matrix.md`](../../audit/generated/runtime_execution_matrix.md).
> [`apple_execution_inventory.md`](../../audit/generated/apple_execution_inventory.md)
> adds the execution-unit view (generic route, Value Target-IR, package
> subgraph, and reference fallback). The tables below intentionally explain
> the implementation rather than certify current placement or proof status.
>
> Architecture, execution gates, the Metal 4 lane, datatypes, and the
> how-to-add-a-kernel guides live in the single backend reference,
> [Apple backend reference](README.md).

Two families of Apple GPU execution coexist:

1. **Native fused / MPS / MPSGraph kernels** — hand-written MSL, `MPSMatrix*`,
   MPSGraph, and Metal 4 cooperative-tensor kernels. These are the perf lanes.
2. **`runtime.launch()` reference + compute lanes** — one executor per
   `compiler_path` that reaches op-family parity with x86/ROCm. These either
   genuinely dispatch to the MPS/MSL compute lanes (`native_gpu`, numpy fallback
   off-Metal) or, where Apple ships no device kernel, run the same standalone
   reference the x86/ROCm device kernels are matched against (`reference_cpu`).

---

## 1. Native fused / MPS / MPSGraph kernels

All symbols live in
`src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm`
(Darwin) and `apple_gpu_runtime_stub.cpp` (non-Darwin).

### Single-op kernels

| Symbol | Graph IR op | Backend | Phase | Constraints |
|--------|-------------|---------|-------|-------------|
| `tessera_apple_gpu_mps_matmul_f32` | `tessera.matmul` (f32) | MPSMatrixMultiplication | 8.3 | rank-2, static |
| `tessera_apple_gpu_mps_matmul_f16` | `tessera.matmul` (f16) | MPSMatrixMultiplication | 8.4.4 | rank-2, static |
| `tessera_apple_gpu_mps_matmul_bf16` | `tessera.matmul` (bf16) | fp32-conversion + MPS | 8.4.4 | rank-2, static (no native MPS bf16) |
| `tessera_apple_gpu_rope_{f32,f16,bf16}` | `tessera.rope` | MSL (`half` for f16; bf16 fp32-conversion) | 8.4.0 / 8.4.4.1 | rank-2, K%2==0 |
| `tessera_apple_gpu_softmax_{f32,f16,bf16}` | `tessera.softmax` | MSL | 8.4.2 / 8.4.4.1 | rank-2, axis=-1 |
| `tessera_apple_gpu_gelu_{f32,f16,bf16}` | `tessera.gelu` | MSL (tanh-approx) | 8.4.2 / 8.4.4.1 | rank-2 |
| `tessera_apple_gpu_flash_attn_{f32,f16,bf16}` | `tessera.flash_attn` | MSL online softmax | 8.4.1 / 8.4.4.2 | rank-3, head_dim ≤ 256, optional causal |

### Fused 2-op / 3-op kernels

| Symbol | Graph IR chain | Backend | Phase | Constraints |
|--------|----------------|---------|-------|-------------|
| `tessera_apple_gpu_matmul_softmax_{f32,f16,bf16}` | `matmul → softmax` | MSL fused | 8.4.3 / 8.4.4.2 | rank-2, axis=-1, N ≤ 256 |
| `tessera_apple_gpu_matmul_softmax_tiled_f32` | `matmul → softmax` (large N) | MSL + threadgroup memory | 8.4.6 | rank-2, N ≤ 8192 |
| `tessera_apple_gpu_matmul_gelu_f32` | `matmul → gelu` | MSL fused | 8.4.7 | rank-2, N ≤ 256 |
| `tessera_apple_gpu_matmul_rmsnorm_f32` | `matmul → rmsnorm[_safe]` | MSL fused | 8.4.7 | rank-2, N ≤ 256 |
| `tessera_apple_gpu_matmul_softmax_matmul_{f32,f16,bf16}` | `matmul → softmax → matmul` | MSL fused (attention block) | 8.4.5 | rank-2, N ≤ 256, P ≤ 256 |

`matmul_softmax_f32` is a **router** (8.4.6): per-thread for N ≤ 256,
threadgroup-tiled for N > 256, reference fallback for N > 8192.

### MPSGraph lane — Tier-1 + long tail

One parametrized MPSGraph runner per shape class; compute is fp32 internally
(inputs cast up, outputs cast down); **no `N ≤ 256` limit**. Graphs cached by
`(shape-class, opcode, dtype, shape[, eps, weighted])`.

| Symbol | Graph IR op(s) | Shape class | dtypes |
|--------|----------------|-------------|--------|
| `tessera_apple_gpu_mpsgraph_unary_{f32,f16}` | relu/sigmoid(_safe)/tanh/softplus/silu/exp/log/sqrt/rsqrt/neg/abs + sin/cos/atan2 (op-coded) | elementwise | f32, f16 (bf16 host-upcast) |
| `tessera_apple_gpu_mpsgraph_binary_{f32,f16}` | silu_mul + add/sub/mul/div/max/min/pow/atan2 | elementwise | f32, f16 |
| `tessera_apple_gpu_layer_norm_{f32,f16}` | `tessera.layer_norm` | row op | f32, f16 |
| `tessera_apple_gpu_rmsnorm_gpu_{f32,f16}` | `tessera.rmsnorm[_safe]` | row op | f32, f16 |
| `tessera_apple_gpu_log_softmax_{f32,f16}` | `tessera.log_softmax` | row op | f32, f16 |
| `tessera_apple_gpu_mpsgraph_softmax_{f32,f16}` | `tessera.softmax` (no N limit) | row op | f32, f16 |
| `tessera_apple_gpu_mpsgraph_reduce_f32` | sum/mean/max/min/prod/var/std/logsumexp (op-coded) | reduction / scan | f32 |
| `tessera_apple_gpu_bmm_{f32,f16}` | `tessera.matmul` (rank-3+) / `tessera.batched_gemm` | batched matmul, `b_broadcast` for shared B | f32, f16 |

### Linear-algebra kernels (MPSMatrix + custom MSL)

Dense f32 factorizations/solves — the one lane MPSGraph cannot supply. Each
returns `0` (ran) / `2` (singular/non-PD) / `-1` (no Metal); the Python wrapper
falls back to numpy otherwise.

| Symbol | Graph IR op | Backend |
|--------|-------------|---------|
| `tessera_apple_gpu_cholesky_f32` | `tessera.cholesky` | MPSMatrixDecompositionCholesky |
| `tessera_apple_gpu_solve_cholesky_f32` | `tessera.cholesky_solve` | Cholesky + MPSMatrixSolveCholesky |
| `tessera_apple_gpu_solve_lu_f32` | `tessera.solve` | LU + MPSMatrixSolveLU |
| `tessera_apple_gpu_tri_solve_f32` | `tessera.tri_solve` | MPSMatrixSolveTriangular |

Custom batched MSL grid kernels (`cholesky_batched`, `tri_solve_batched`,
`_svd_bl_batched_f32` Brent–Luk Jacobi) measure 30–388× a per-matrix MPS loop.
QR = Cholesky-QR with a `‖QᵀQ−I‖` verify → Householder fallback. See
[Apple backend reference](README.md#gpu-linear-algebra-implementation-state).

### Capability / diagnostic + GPU-native RNG symbols

`tessera_apple_gpu_runtime_has_metal`, `_msl_cache_size`, `_simd_caps`,
`_device_handle`, `_command_queue_handle`, `ts_dev_mtl_buffer`,
`_mpsgraph_cache_size`; plus the opt-in Philox fills
`tessera_apple_gpu_random_{uniform,normal}_f32` (a separate stream, **not**
bit-identical to `tessera.rng` — Decision #18).

### ABI summary

Tensor pointers are `i64` at the `func.call` boundary; dims are `i32`; scale/eps
`f32`; bool flags `i32`. For f16/bf16 the pointers are `uint16_t*` bit patterns
(`.view(np.uint16)`). The element type is encoded in the **symbol name only**.

### Core-9 coverage matrix

|  | f32 | f16 | bf16 |
|---|---|---|---|
| mps_matmul | ✅ 8.3 | ✅ 8.4.4 | ✅ 8.4.4 |
| rope / softmax / gelu | ✅ | ✅ 8.4.4.1 | ✅ 8.4.4.1 |
| flash_attn | ✅ 8.4.1 | ✅ 8.4.4.2 | ✅ 8.4.4.2 |
| matmul_softmax | ✅ 8.4.3 | ✅ 8.4.4.2 | ✅ 8.4.4.2 |
| matmul_softmax (tiled, large N) | ✅ 8.4.6 | — | — |
| matmul_gelu / matmul_rmsnorm | ✅ 8.4.7 | — | — |
| matmul_softmax_matmul | ✅ 8.4.5 | ✅ 8.4.5 | ✅ 8.4.5 |

**9 kernel concepts × dtypes = 26 core runtime symbols.** The MTL4 cooperative
`matmul2d` lane (fp16/bf16 + fused bias/act epilogue + resident-weight session)
and the GA/EBM fused kernels add further symbols — see
[Apple backend reference](README.md) and `runtime_abi.csv`.

---

## 2. `runtime.launch()` reference + compute lanes (op-family parity)

Reached from `@jit(target="apple_gpu")` / `runtime.launch()` with the given
`compiler_path`. Each is F4-gated against the standalone reference (matching the
x86/ROCm lanes). `execution_kind` is honest per lane: `native_gpu` genuinely
dispatches to the MPS/MSL compute lanes above (numpy fallback when Metal is
unavailable); `reference_cpu` runs the CPU reference where Apple ships no device
kernel. Manifest/status truth:
[`apple_target_map.md`](../../audit/generated/apple_target_map.md);
per-lane execution truth:
[`runtime_execution_matrix.md`](../../audit/generated/runtime_execution_matrix.md).

| `compiler_path` | Ops | `execution_kind` |
|-----------------|-----|------------------|
| `apple_gpu_loss_compiled` | mse / mae / huber / smooth_l1 / log_cosh | **native_gpu** — residual + reduction on the MPSGraph binary/reduce lanes |
| `apple_gpu_loss_family_compiled` | binary-CE, class-axis (cross_entropy/kl/js/z_loss), RL policy (ppo/cispo/grpo), EBM-diffusion (8) | **native_gpu** — per-sample via reference, none/mean/sum reduction on the MPSGraph reduce lane |
| `apple_gpu_complex_compiled` | 9 pointwise complex (mul/div/conjugate/abs/arg/exp/log/sqrt/pow) + geometric/certificate (cross_ratio/dz/dbar/laplacian_2d/conformal_*/is_concyclic/…) | **native_gpu** — interleaved-f32 on the unary/binary/atan2 lanes; geometric ops reuse `tessera.complex` |
| `apple_gpu_conformal_compiled` | mobius, stereographic | **native_gpu** — interleaved-f32 complex_mul/complex_div/binary-div |
| `apple_gpu_reduce_compiled` | sum (axis/keepdims) | **native_gpu** — MPSGraph reduce lane |
| `apple_gpu_structured_compute_compiled` | conv1d / conv_transpose / depthwise_conv1d, ctc_loss / edm_loss_weight, + the structured tail (center_crop/image_resize/interpolate/patchify/pixel_(un)shuffle, gru_cell/simple_rnn_cell/bidirectional_scan, cross_attention/perceiver_resampler/lora_linear, arange/masked_fill/rearrange/pack/unpack/tile_view/rope_split/rope_merge/mor_*, edm_precondition/factorized_pos_emb/masked_scatter/memory_read/mrope_2d/online_softmax_state/spectral_norm) | reference_cpu |
| `apple_gpu_rng_compiled` | rng_uniform / rng_normal / dropout (Philox-4x32-10 reference core) + RNGKey (key/split/fold_in/clone) + samplers (bernoulli/beta/categorical/dirichlet/gamma/multinomial/permutation/poisson/randint/truncated_normal) + MCMC (langevin/mala/hmc/gibbs) | reference_cpu — Apple ships no device Philox |
| `apple_gpu_linalg_compiled` | cholesky_solve / lu / qr / svd (+ cholesky/tri_solve) | reference_cpu — `np.linalg` + a standalone partial-pivot LU (no MPS lu/qr/svd) |
| `apple_gpu_matmul_family_compiled` | einsum (single-contraction), factorized_matmul | reference_cpu |
| `apple_gpu_optimizer_compiled` | sgd / momentum / adam / adamw / lion (state m/v in/out) | reference_cpu — numpy update rules, matches `tessera.optim` |
| `apple_gpu_shape_compiled` | pad / roll / flip / tile / repeat / stack (0-move gather) + sort / argsort | reference_cpu |
| `apple_gpu_scatter_compiled` | scatter / scatter_add / scatter_reduce (set/add/min/max) | reference_cpu |
| `apple_gpu_sparse_compiled` | legacy sparse/MoE artifacts | reference_cpu — retained only for stamped artifacts; the current dedicated CSR/COO SpMM, SDDMM, dense-block BSMM, and local top-1 MoE executors below are the native f32 routes when their contracts hold |
| `apple_gpu_tail_compiled` | MLA latent-KV (compress/expand_k/expand_v), alibi, lgamma/digamma, fused_epilogue, asymmetric_bce (`tessera.loss.asymmetric_bce`), normalize_group_advantages (`tessera.rl.normalize_group_advantages`), spec_accept / spec_accept_sample / spec_accept_tree_sample | reference_cpu — reuses the public `tessera.ops`/`losses`/`rl` reference |

**Not yet covered:** `quantize`/`dequantize` fp4/fp6/fp8/nvfp4 +
`dequant_grouped_gemm`. The macOS 27 / Metal 4.1 tensor toolchain is necessary,
but each format also needs per-device numerical and throughput evidence before it
can be described as a native-performance lane. Everything else the x86/ROCm
device lanes cover now has an executable Apple GPU artifact path.

---

*Tests: `tests/unit/test_apple_gpu_*.py` (per-lane F4 gates) and the native
`test_apple_backend_roadmap.py` / `test_apple_gpu_{mpsgraph_lane,linalg,metal4,rng}.py`.*
