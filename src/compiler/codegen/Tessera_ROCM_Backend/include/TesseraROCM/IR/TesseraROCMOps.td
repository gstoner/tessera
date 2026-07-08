include "TesseraROCMDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TesseraROCM_Op<string mnemonic, list<Trait> traits = []>
  : Op<TesseraROCMDialect, mnemonic, traits>;

def ROCM_MemcpyOp : TesseraROCM_Op<"memcpy", [MemoryEffects<[MemWrite, MemRead]>]> {
  let arguments = (ins AnyType:$dst, AnyType:$src, I64:$bytes,
    OptionalAttr<MemSpaceAttr>:$dst_space, OptionalAttr<MemSpaceAttr>:$src_space);
  let assemblyFormat = "$dst `,` $src `,` $bytes attr-dict `:` type($dst) `,` type($src)";
}

def ROCM_AsyncCopyOp : TesseraROCM_Op<"async_copy", [MemoryEffects<[MemWrite, MemRead]>]> {
  let arguments = (ins AnyType:$dst, AnyType:$src, I64:$bytes,
    OptionalAttr<MemSpaceAttr>:$dst_space, OptionalAttr<MemSpaceAttr>:$src_space);
  let results = (outs TokenType:$tkn);
  let assemblyFormat = "$dst `,` $src `,` $bytes attr-dict `:` type($dst) `,` type($src) `->` type($tkn)";
}

def ROCM_WaitTokenOp : TesseraROCM_Op<"wait", []> {
  let summary = "Wait on an async-copy token via a targeted hardware counter";
  let description = [{
    Waits for an outstanding async memory operation to retire.  The optional
    `counter` attribute names the AMDGCN wait-counter class to gate on rather
    than draining the whole wavefront with a full `s_barrier`:

      * `"vmcnt"`   — vector-memory counter (global / buffer loads, incl. the
                      direct-to-LDS `global_load_lds` / `buffer_load ... lds`
                      path an async global→LDS copy uses);
      * `"lgkmcnt"` — LDS / GDS / constant counter (e.g. a `ds_read`);
      * absent      — a true synchronization point, lowered to `s_barrier`.

    Targeted counter waits are what let the matrix core keep issuing while a
    copy is still in flight (moonmath CDNA3 attention writeup, §"Decoupled
    Waits"); a blanket barrier would stall it.
  }];
  let arguments = (ins TokenType:$tkn, OptionalAttr<StrAttr>:$counter);
  let assemblyFormat = "$tkn attr-dict `:` type($tkn)";
}

def ROCM_MFMAOp : TesseraROCM_Op<"mfma"> {
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$acc);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($res)";
}

def ROCM_WMMAGemmOp : TesseraROCM_Op<"wmma_gemm"> {
  let summary = "A matmul tile to realize as a fragment-materialized WMMA kernel";
  let description = [{
    Stage K — the matmul *directive* the generating pass (`generate-wmma-gemm-
    kernel`) expands into a real WMMA GEMM kernel: it emits a `gpu.module` +
    `gpu.func` whose body loads the A/B tiles into RDNA WMMA fragment vectors
    (`vector<16xf16>`), calls `tessera_rocm.wmma` (Stage J → real `rocdl.wmma`),
    and stores the `vector<8xf32>` accumulator with the wave32 lane/element
    layout. So the GEMM is compiler-*generated*, not authored MLIR.

    `m`/`n`/`k` are the WMMA *instruction* tile (16x16x16 — the only tile RDNA's
    `V_WMMA` exposes; other extents are a named error). The emitted kernel is
    *problem-size-generic*: it takes the runtime `(M,N,K)` as `index` arguments,
    a 2-D grid of one wave per 16x16 output tile, an `scf.for` K-loop, and
    ragged-edge masking, so a single compiled kernel handles any shape. The
    `M=N=K=16` launch reduces to the single tile that is bit-identical to the
    hardware_verified hand-written oracle.

    `mt`/`nt` (default 1) are the **register-blocked macro-tile**: each wave
    computes an `mt`x`nt` grid of 16x16 output tiles, reusing a loaded A
    fragment across the `nt` B-tiles and a loaded B fragment across the `mt`
    A-tiles (output-tile register blocking — the occupancy/VGPR-budgeted knob
    the hand-written oracle sweeps; measured-best is 3x4 for large problems).
    The grid shrinks accordingly (one wave per `mt`x`nt` macro-tile). This is
    the Target-IR matmul-tile node: the Graph `tessera.matmul` → Tile → Target-IR
    lowering (`_lower_rocm_op` on `tile.mma`) now EMITS this directive, so it is
    produced by the IR stack (not only synthesized by the runtime).

    **Fused epilogue.** `bias` (default false) appends a per-output-column bias
    operand (a `memref<?>` of length N, accumulator dtype) that is added to each
    accumulator element before the activation. `activation` (default `"none"`)
    selects a pointwise activation fused on the f32 accumulator *before the
    store* — `"relu"` (max(x,0)), `"gelu"` (tanh approximation, matching the
    common `gelu(approximate='tanh')`), or `"silu"` (x·sigmoid(x)). The activation
    runs as VALU ops on the in-register accumulator (no extra global traffic);
    transcendentals lower through the same `math` → ROCDL path the flash_attn
    softmax uses. This is the `fused_epilogue` Target-IR node — matmul + bias +
    activation in one kernel, no intermediate D round-trip.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       DefaultValuedAttr<I64Attr, "1">:$mt,
                       DefaultValuedAttr<I64Attr, "1">:$nt,
                       DefaultValuedAttr<StrAttr, "\"f16\"">:$dtype,
                       DefaultValuedAttr<BoolAttr, "false">:$bias,
                       DefaultValuedAttr<StrAttr, "\"none\"">:$activation);
  let assemblyFormat = "attr-dict";
}

def ROCM_FlashAttnOp : TesseraROCM_Op<"flash_attn"> {
  let summary = "An FA-2 forward attention op to realize as a WMMA kernel";
  let description = [{
    The attention analog of `tessera_rocm.wmma_gemm`: the directive the
    `generate-wmma-flash-attn-kernel` pass expands into a real, fragment-
    materialized RDNA WMMA **flash-attention forward** kernel. One wave per
    (16-query tile, b*h): stage Q in LDS, then for each 16-key tile compute
    `S = scale * Q@K^T` (WMMA over head-dim chunks), apply the causal/ragged
    mask, do an online softmax (running max/sum, rescale), and accumulate
    `O += P@V` (WMMA). Scores are staged in LDS so the QK^T accumulator layout
    is reread in the P@V A-fragment layout. Mirrors the hardware_verified
    hand-written kernel (`tessera_rocm_wmma_flash_attn_*`).

    `head_dim` (a multiple of 16) is the compile-time D; B/H/Sq/Sk/scale/causal
    are runtime kernel arguments. `dtype` is the f16/bf16 storage (f32 softmax +
    accumulate). Q/K/V are `[b,h,s,d]` row-major; O is f32.

    `gqa` (default false) selects grouped-query / multi-query attention: H query
    heads share G<H key/value heads — query head h reads KV head h/kv_ratio
    (kv_ratio = H/G; 1 = MHA, H = MQA). In gqa mode the kernel gains two trailing
    runtime args (`heads` = H, `kv_ratio`) and reads K/V as `[b,G,s,d]` from the
    grouped head index `b*G + h/kv_ratio`. Same FA-2 WMMA body otherwise.

    `sliding_window` (default false) selects Mistral-style sliding-window
    attention: query position p attends only to keys in `(p - W, p]` — a causal
    band of width W (the trailing runtime arg, after the gqa args when both are
    set). A windowed kernel is implicitly causal; it skips KV tiles entirely
    below the window's lower edge and the per-element mask trims the boundary.

    `logit_softcap` (default false) selects Gemma-2-style attention logit
    soft-capping: each scaled score is passed through `cap * tanh(S / cap)`
    before masking + softmax, bounding the logits to `(-cap, cap)`. `cap` is the
    trailing runtime arg (after the gqa and window args when those are set).
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$head_dim,
                       DefaultValuedAttr<StrAttr, "\"f16\"">:$dtype,
                       DefaultValuedAttr<BoolAttr, "false">:$gqa,
                       DefaultValuedAttr<BoolAttr, "false">:$sliding_window,
                       DefaultValuedAttr<BoolAttr, "false">:$logit_softcap);
  let assemblyFormat = "attr-dict";
}

def ROCM_FlashAttnBwdOp : TesseraROCM_Op<"flash_attn_bwd"> {
  let summary = "An FA-2 backward pass to realize as three WMMA kernels";
  let description = [{
    The backward analog of `tessera_rocm.flash_attn`: the directive the
    `generate-wmma-flash-attn-bwd-kernel` pass expands into THREE real,
    fragment-materialized RDNA WMMA kernels — the textbook FA-2 backward
    structure (no stored attention matrix; recompute S/P per tile):

      * `<name>_pre`  — one wave per (16-query tile, b*h). Scalar pass:
        `D[q] = sum_d O[q,d]*dO[q,d]` and `L[q] = logsumexp_k(scale*Q@K^T)`
        (online max/sum). Writes the L / D scratch arrays the matmul kernels
        read so the backward is self-contained (does not depend on the forward
        having saved its softmax stats).
      * `<name>_dkdv` — one wave per (16-key tile, b*h). Loops query tiles:
        recompute `S=scale*Q@K^T` and `P=exp(S-L)` (WMMA), `dP=dO@V^T` (WMMA),
        `dS=P*(dP-D)`; accumulates `dV += P^T@dO` and `dK += scale*dS^T@Q`
        (WMMA, contraction over queries — P / dS are staged in LDS and reread
        transposed, the layout bridge).
      * `<name>_dq`   — one wave per (16-query tile, b*h). Loops key tiles:
        same `S`/`P`/`dP`/`dS`, accumulates `dQ += scale*dS@K` (WMMA,
        contraction over keys; dS reread from LDS in natural layout).

    `head_dim` (a multiple of 16) is the compile-time D; B/H/Sq/Sk/scale/causal
    are runtime kernel arguments. `dtype` is the f16/bf16 storage (f32 softmax +
    accumulate). Q/K/V/dO are `[b,h,s,d]` row-major store-type; O is f32; the
    dQ/dK/dV outputs and the L/D scratch are f32. Validated against a numpy
    attention-backward reference (itself checked vs finite differences).

    `gqa` (default false) is grouped-query backward: all three kernels gain two
    trailing runtime args (`heads` = H, `kv_ratio` = H/G) and read K/V from the
    grouped KV head. All three keep the B*H query-head grid; `_dkdv` accumulates
    dK/dV **atomically** — each query-head block adds its contribution into the
    grouped KV head's rows, so the kv_ratio blocks sharing a KV head sum into the
    same rows (host pre-zeros dK/dV). MHA (non-gqa) is unchanged (plain stores,
    no atomics).
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$head_dim,
                       DefaultValuedAttr<StrAttr, "\"f16\"">:$dtype,
                       DefaultValuedAttr<BoolAttr, "false">:$gqa);
  let assemblyFormat = "attr-dict";
}

def ROCM_LinearAttnOp : TesseraROCM_Op<"linear_attn"> {
  let summary = "Linear-attention forward (quadratic-parallel form) as a WMMA kernel";
  let description = [{
    The directive the `generate-wmma-linear-attn-kernel` pass expands into a
    fragment-materialized RDNA WMMA **linear-attention forward** kernel — the
    quadratic-parallel form `O = (φ(Q) φ(K)ᵀ ⊙ causal) @ V`. Unlike flash
    attention there is **no softmax** (no running max / exp / normalization): the
    score matrix `A = φ(Q)φ(K)ᵀ` is masked **multiplicatively** (masked entries
    → 0, not −∞) and `O += A @ V` accumulates directly. One wave per (16-query
    tile, b*h): for each 16-key tile up to the causal/ragged limit, compute
    `A = φ(Q)@φ(K)ᵀ` (WMMA over head-dim chunks), apply the causal/ragged mask,
    stage `A` in LDS, then accumulate `O += A@V` (WMMA, the same LDS layout
    bridge flash attention uses). No final divide.

    `feature_map` (default `"identity"`) is the elementwise φ applied to Q and K
    before the QKᵀ, on the loaded fragment: `"identity"` (no-op), `"relu"`
    (max(x,0)), or `"polynomial_2"` (x²). This matches the canonical reference
    `O = (φ(Q)φ(K)ᵀ ⊙ tril) @ V` (`_apple_gpu_dispatch_linear_attn`).

    `decay` (default false) selects the **decay-masked** variants
    (lightning_attention / retention): the score `A[i,j]` is additionally scaled
    by `λ^(i-j)` over the causal band (RetNet/lightning per-head exponential
    decay). `λ` is supplied as its log — a trailing f32 runtime arg `log_decay` —
    and the kernel applies `A[i,j] *= exp((i-j)·log_decay)` for kept entries.
    With `feature_map="identity"` this is lightning_attention; with
    `"polynomial_2"` it is (degree-2) retention. Matches the reference's
    `ratio[i,j] = dc[i]/dc[j]` decay mask for a per-head-constant decay.

    `head_dim` (a multiple of 16) is the compile-time D — query/key and value
    head dim are assumed equal (square). B/H/S/causal are runtime kernel args;
    `dtype` is the f16/bf16 storage (f32 accumulate). Q/K/V are `[b,h,s,d]`
    row-major; O is f32.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$head_dim,
                       DefaultValuedAttr<StrAttr, "\"f16\"">:$dtype,
                       DefaultValuedAttr<StrAttr, "\"identity\"">:$feature_map,
                       DefaultValuedAttr<BoolAttr, "false">:$decay);
  let assemblyFormat = "attr-dict";
}

def ROCM_SoftmaxOp : TesseraROCM_Op<"softmax"> {
  let summary = "Row-wise numerically-stable softmax as a reduction kernel";
  let description = [{
    The directive `generate-rocm-softmax-kernel` expands into a real
    **row-reduction** gpu kernel — the first non-matmul/non-WMMA compiler-
    generated ROCm kernel. For a rank-2 input `[M, K]` it computes the
    numerically-stable softmax over the last axis:
    `O[m,:] = exp(X[m,:] - max_k X[m,k]) / Σ_k exp(...)` (matching the reference
    `_apple_gpu_dispatch_softmax`).

    One workgroup per row (`blockIdx.x = m`, `blockDim = 256`); the lanes stride
    over K and tree-reduce the row max then the row sum through LDS (two passes,
    exp values staged into O between them, then divided in place). Reductions run
    in **f32** for stability regardless of storage dtype; `exp` lowers through the
    same `math` → ROCDL path the flash_attn softmax uses. `M`/`K` are runtime
    `index` args (problem-generic). `dtype` is the f16/bf16/f32 storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_WMMAOp : TesseraROCM_Op<"wmma"> {
  let summary = "RDNA WMMA matrix multiply-accumulate (16x16x16)";
  let description = [{
    The RDNA analog of `tessera_rocm.mfma`. RDNA GPUs (gfx11xx RDNA 3 / 3.5,
    gfx12xx RDNA 4) have no MFMA matrix core; the equivalent instruction is
    `V_WMMA_*` at a single 16x16x16 tile, wave32. RDNA 3 / 3.5 (gfx1100,
    gfx1151) expose no FP8 WMMA — an FP8 matmul on those arches is a hard,
    named error in the Tile->ROCm lowering rather than a silent fallback.
  }];
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$acc);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($res)";
}

def ROCM_ActivationKernelOp : TesseraROCM_Op<"activation"> {
  let summary = "Elementwise pointwise activation (gelu / silu / relu) kernel";
  let description = [{
    The directive `generate-rocm-activation-kernel` expands into a flat
    elementwise gpu kernel applying a pointwise activation over N elements
    (one thread per element, strided grid):

      * `kind = "gelu"` — tanh approximation
        `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (matches
        `gelu(approximate='tanh')`).
      * `kind = "silu"` — `x·σ(x) = x / (1 + exp(−x))`.
      * `kind = "relu"` — `max(x, 0)`.

    These are the **standalone** ops (the same activations the GEMM fused
    epilogue applies in-register); they compute in f32 regardless of storage
    dtype, and the transcendentals lower through the `math` → ROCDL path. `N` is
    a runtime `index` arg; `dtype` is the f16/bf16/f32 storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"gelu\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_SiluMulKernelOp : TesseraROCM_Op<"silu_mul"> {
  let summary = "SwiGLU gate-multiply (2-operand elementwise) kernel";
  let description = [{
    The directive `generate-rocm-silu-mul-kernel` expands into a flat 2-operand
    elementwise gpu kernel computing the SwiGLU gate-multiply over N elements
    (one thread per element, strided grid):

      O[i] = silu(A[i]) * B[i] = (A[i] / (1 + exp(-A[i]))) * B[i]

    The **standalone** analog of the gate-multiply the fused SwiGLU MLP applies
    in-register (`silu(x@Wg) * (x@Wu)`); it computes in f32 regardless of storage
    dtype and the transcendental lowers through the `math` → ROCDL path. `N` is a
    runtime `index` arg; `dtype` is the f16/bf16/f32 storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_PointwiseLossKernelOp : TesseraROCM_Op<"pointwise_loss"> {
  let summary = "Pointwise regression loss (2-operand elementwise) kernel";
  let description = [{
    The directive `generate-rocm-pointwise-loss-kernel` expands into a flat
    2-operand elementwise gpu kernel computing the PER-ELEMENT S11 regression
    loss over N elements (one thread per element). The runtime reduces
    (none/mean/sum) via the rocm reduce lane.

      kind 0 mse  (p−t)²; 1 mae |p−t|; 2 huber(δ); 3 smooth_l1(β);
      4 log_cosh  e + log1p(exp(−2e)) − log2

    `kind` and `param` (δ/β) are compile-time attrs. Computes in f32; exp/log1p
    lower through `math` → ROCDL. The CPU analog is `avx512_loss_f32.cpp`.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype,
                       DefaultValuedAttr<I64Attr, "0">:$kind,
                       DefaultValuedAttr<F32Attr, "1.0">:$param);
  let assemblyFormat = "attr-dict";
}

def ROCM_BinaryLossKernelOp : TesseraROCM_Op<"binary_loss"> {
  let summary = "Binary-cross-entropy loss (2-operand elementwise) kernel";
  let description = [{
    `generate-rocm-binary-loss-kernel` expands into a flat 2-operand elementwise
    per-element BCE-with-logits kernel over (z=logits, t=target):
    kind 0 bce `max(z,0) − z·t + log1p(exp(−|z|))`; kind 1 asymmetric_bce
    `pw·t·softplus(−z) + nw·(1−t)·softplus(z)`. `kind`/`pos_weight`/`neg_weight`
    are compile-time attrs; the runtime reduces. CPU analog: avx512_loss_f32.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype,
                       DefaultValuedAttr<I64Attr, "0">:$kind,
                       DefaultValuedAttr<F32Attr, "1.0">:$pos_weight,
                       DefaultValuedAttr<F32Attr, "1.0">:$neg_weight);
  let assemblyFormat = "attr-dict";
}

def ROCM_PolicyLossKernelOp : TesseraROCM_Op<"policy_loss"> {
  let summary = "PPO / CISPO RL policy-loss surrogate (3-operand) kernel";
  let description = [{
    `generate-rocm-policy-loss-kernel` expands into a flat 3-operand elementwise
    per-element RL policy-loss surrogate over (logp_new, logp_old, advantages),
    `ratio = exp(ln−lo)`: kind 0 ppo `−min(ratio·adv, clip(ratio,1−ε,1+ε)·adv)`;
    kind 1 cispo `−(min(ratio,ε_high)·adv·ln)`. `kind`/`clip` are compile-time
    attrs; the runtime reduces and normalizes grpo advantages. CPU analog:
    avx512_policy_loss_f32.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype,
                       DefaultValuedAttr<I64Attr, "0">:$kind,
                       DefaultValuedAttr<F32Attr, "0.2">:$clip);
  let assemblyFormat = "attr-dict";
}

def ROCM_FpQuantKernelOp : TesseraROCM_Op<"fpquant"> {
  let summary = "Low-precision float-grid quantization (1-operand) kernel";
  let description = [{
    `generate-rocm-fpquant-kernel` expands into a flat 1-operand elementwise
    kernel snapping each value to a low-precision float grid (mantissa-snap):
    `ax=min(|x|,max_normal)`; `e=max(floor(log2 ax),min_exp)`;
    `ulp=2^(e−mantissa_bits)`; `out=sign·min(roundeven(ax/ulp)·ulp,max_normal)`,
    NaN-propagating. `max_normal`/`mantissa_bits`/`min_exp` are compile-time
    attrs (fp8 e4m3/e5m2, fp6 e2m3/e3m2, fp4 e2m1). The runtime applies the
    per-tensor / per-block scale. CPU analog: avx512_fpquant_f32.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype,
                       DefaultValuedAttr<F32Attr, "6.0">:$max_normal,
                       DefaultValuedAttr<I64Attr, "1">:$mantissa_bits,
                       DefaultValuedAttr<I64Attr, "-126">:$min_exp);
  let assemblyFormat = "attr-dict";
}

def ROCM_DequantGemmKernelOp : TesseraROCM_Op<"dequant_gemm"> {
  let summary = "Fused packed-code dequantize-into-GEMM kernel";
  let description = [{
    The directive `generate-rocm-dequant-gemm-kernel` expands into the DK4
    packed-weight dequantize-into-GEMM kernel. It consumes f32 activations,
    packed int4 or int8 weight codes, per-group f32 scales, and emits f32
    accumulators without materializing a full dequantized weight matrix.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_DftKernelOp : TesseraROCM_Op<"dft"> {
  let summary = "Direct DFT (one-thread-per-bin) gpu kernel";
  let description = [{
    `generate-rocm-dft-kernel` expands into a one-thread-per-output-bin DFT over
    a batch of interleaved-complex rows: OUT[b,k]=Σ_j IN[b,j]·exp(sign·2πi·kj/n).
    `inverse` selects the sign (unnormalized; the runtime applies the plan
    scale). cos/sin via math→ROCDL. The correctness foundation for the ROCm
    spectral lane (radix-2/Bluestein perf is a follow-up). CPU analog:
    avx512_fft_f32 + the Bluestein/DFT executor.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<BoolAttr, "false">:$inverse);
  let assemblyFormat = "attr-dict";
}

def ROCM_SpmmKernelOp : TesseraROCM_Op<"spmm"> {
  let summary = "Row-wise CSR sparse-dense matmul gpu kernel";
  let description = [{
    `generate-rocm-spmm-kernel` expands into a genuinely sparse row-wise SpMM —
    C[M,N] = A_csr[M,K] @ B[K,N], one thread per output element (i,n):
    C[i,n] = Σ_{p=indptr[i]}^{indptr[i+1]} values[p]·B[indices[p]·N+n]. It
    iterates the CSR nonzero structure (indptr/indices i32, values/B/OUT f32),
    NOT densify-then-GEMM. CPU analog: avx512_sparse_f32 spmm_csr.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_SddmmKernelOp : TesseraROCM_Op<"sddmm"> {
  let summary = "Sampled dense-dense matmul gpu kernel";
  let description = [{
    `generate-rocm-sddmm-kernel` expands into a sampled dense-dense matmul —
    OUT[M,N] = (A[M,K]·Bt[N,K]^row) ⊙ mask[M,N], one thread per (i,j) that skips
    the length-K dot when mask=0 (genuinely sampled). B is passed row-major-
    transposed as Bt[N,K] so both dot operands are contiguous. All f32. CPU
    analog: avx512_sparse_f32 sddmm.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_SelectiveSsmKernelOp : TesseraROCM_Op<"selective_ssm"> {
  let summary = "Mamba2 selective state-space scan gpu kernel";
  let description = [{
    `generate-rocm-selective-ssm-kernel` expands into a Mamba2 selective scan,
    one thread per (b,d) channel (sequential over time, parallel over channels):
    A_bar=exp(delta[b,t,d]·A[d,n]); h[b,d,n]=A_bar·h+delta·B[b,t,n]·x[b,t,d];
    y[b,t,d]=Σ_n C[b,t,n]·h[b,d,n]. The N-length state slice h[b,d,:] is a global
    scratch buffer (init to the caller's state or zeros). exp via math→ROCDL. All
    f32. CPU analog: avx512_ssm_f32.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_CholeskyKernelOp : TesseraROCM_Op<"cholesky"> {
  let summary = "Batched Cholesky factorization gpu kernel";
  let description = [{
    `generate-rocm-cholesky-kernel` expands into a batched Cholesky (one thread
    per matrix) — A[b,n,n] SPD → L[b,n,n] lower, A=L·Lᵀ (Cholesky–Banachiewicz):
    L[j,j]=sqrt(A[j,j]-Σ_{k<j}L[j,k]²); L[i,j]=(A[i,j]-Σ_{k<j}L[i,k]·L[j,k])/L[j,j].
    sqrt via math→ROCDL. All f32. CPU analog: avx512_linalg_f32 cholesky.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_TriSolveKernelOp : TesseraROCM_Op<"tri_solve"> {
  let summary = "Batched triangular solve gpu kernel";
  let description = [{
    `generate-rocm-tri-solve-kernel` expands into a batched triangular solve
    A·X=B (one thread per matrix/RHS-column) — `lower` (BoolAttr) selects forward
    (lower) vs back (upper) substitution at codegen. A[b,n,n] (triangular part
    used), B/X[b,n,m]. All f32. CPU analog: avx512_linalg_f32 tri_solve.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<BoolAttr, "true">:$lower);
  let assemblyFormat = "attr-dict";
}

def ROCM_LuKernelOp : TesseraROCM_Op<"lu"> {
  let summary = "Batched LU (partial pivoting) gpu kernel";
  let description = [{
    `generate-rocm-lu-kernel` expands into a batched getrf (one thread per
    matrix) — A[b,n,n] → packed LU[b,n,n] (unit-lower L below diag, U on/above)
    + pivots[b,n] (0-based, piv[k]=row swapped with k). All f32 (pivots i32).
    CPU analog: avx512_lu_qr lu.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_QrKernelOp : TesseraROCM_Op<"qr"> {
  let summary = "Batched Householder QR gpu kernel";
  let description = [{
    `generate-rocm-qr-kernel` expands into a batched Householder QR (one thread
    per matrix) — A[b,m,n] → full Q[b,m,m] orthonormal + R[b,m,n] upper-trapezoid,
    A=Q·R (the runtime slices the reduced Q[:, :k] / R[:k, :], k=min(m,n)). The
    per-thread reflector v[m] lives in a global scratch buffer. norm via
    math→ROCDL. All f32. CPU analog: avx512_lu_qr qr.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_SvdKernelOp : TesseraROCM_Op<"svd"> {
  let summary = "Batched one-sided Jacobi SVD gpu kernel";
  let description = [{
    `generate-rocm-svd-kernel` expands into a batched one-sided Jacobi SVD (one
    thread per matrix, m≥n) — A[b,m,n]=U[b,m,n]·diag(S[b,n])·Vᵀ[b,n,n]. The
    column-major working copy + accumulated V live in global scratch buffers;
    fixed sweeps of 2×2 column rotations, then column norms → S, normalized
    columns → U, sorted descending. sqrt via math→ROCDL. All f32. CPU analog:
    avx512_svd_f32.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_MoeKernelOp : TesseraROCM_Op<"moe"> {
  let summary = "Mixture-of-experts compute (routed expert GEMV) gpu kernel";
  let description = [{
    `generate-rocm-moe-kernel` expands into the MoE compute part — the routed
    per-token expert matmuls (top-1), one thread per (token, out-column):
    out[t,o] = Σ_i x[t,i]·experts[route[t],i,o]. experts is [E,in,out]; route[t]
    (i32) is the resolved top-1 expert (routing done on host). All f32. CPU
    analog: avx512_moe_f32.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_PredicateKernelOp : TesseraROCM_Op<"predicate"> {
  let summary = "Unary predicate (isnan/isinf/isfinite) gpu kernel";
  let description = [{
    `generate-rocm-predicate-kernel` expands into a flat per-element unary
    predicate over f32 input with i8 (0/1) bool output; the `kind` StrAttr selects
    isnan (x!=x) / isinf (|x|==inf) / isfinite (ordered & |x|<inf). One thread per
    element. CPU analog: avx512_predicate_f32.
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$kind);
  let assemblyFormat = "attr-dict";
}

def ROCM_OptimizerKernelOp : TesseraROCM_Op<"optimizer"> {
  let summary = "Fused per-parameter optimizer-step gpu kernel";
  let description = [{
    `generate-rocm-optimizer-kernel` expands into a fused per-parameter update
    (one thread per element); the `kind` StrAttr selects sgd/momentum/adam/adamw/
    lion at codegen. p/g/m/v in, p_out/m_out/v_out out; lr/β1/β2/eps/wd + the
    host-computed bias-correction scalars (b1c,b2c) are f32 args. sqrt via
    math→ROCDL. All f32. CPU analog: avx512_optimizer_f32.
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$kind);
  let assemblyFormat = "attr-dict";
}

def ROCM_AlibiKernelOp : TesseraROCM_Op<"alibi"> {
  let summary = "ALiBi positional-bias generator kernel";
  let description = [{
    The directive `generate-rocm-alibi-kernel` expands into a flat elementwise gpu
    kernel generating the ALiBi positional-bias tensor of shape `[H, S, S]` (one
    thread per element, strided grid):

      bias[h, i, j] = slope[h] * (j - i)

    `slope` is a per-head scalar supplied as a length-H f32 buffer (the runtime
    fills the default `2^(-8(k+1)/H)` ramp when the caller passes none). Matches
    the `nn.functional.alibi` reference (distance = positions[j] - positions[i]).
    Computes in f32 and stores f16/bf16/f32. `H`/`S` are runtime `index` args;
    `dtype` is the output storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_DeltaNetKernelOp : TesseraROCM_Op<"deltanet"> {
  let summary = "Causal gated/delta linear-attention sequential-scan kernel";
  let description = [{
    The directive `generate-rocm-deltanet-kernel` expands into a causal
    sequential-scan gpu kernel for the gated/delta linear-attention recurrence —
    the compiled lane for `gated_deltanet` / `kimi_delta_attention` /
    `modified_delta_attention`. One workgroup per `(b, h)`; one thread per
    value-column `e` (blockDim = `d_v`). Thread `e` owns the state column
    `Ŝ[:, e]` (length `d_qk`) in LDS, so the per-step matvecs are independent
    per-thread `d`-loops with no cross-thread state hazard. Per timestep `t`
    (matching the numpy reference `_delta_attention_impl`, causal path):

      target = V[t]
      if erase:   target -= α_t · (Ŝᵀ k_t)      // read OLD state
      if decay:   Ŝ *= α_t
      delta = outer(k_t, target)  [/ (1 + ‖k_t‖·‖target‖) if modified]
      Ŝ += β_t · delta
      O[t] = Q_t @ Ŝ                             // read NEW state
      if gate: O[t] *= sigmoid(gate[t])

    `d_qk` / `d_v` are compile-time (static loop bounds + LDS sizing). The flags
    `erase` / `modified` / `has_gate` / `has_beta` / `has_decay` are emitted as
    straight-line code. Q/K are `[B,H,S,d_qk]`, V/O/gate `[B,H,S,d_v]`,
    beta/decay `[B,H,S]` (f32); B·H is the grid, S a runtime `index` arg. Compute
    is f32; `dtype` is the f16/bf16/f32 storage. State starts at zero
    (state=None); no state output.
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$d_qk, I64Attr:$d_v,
                       DefaultValuedAttr<BoolAttr, "false">:$erase,
                       DefaultValuedAttr<BoolAttr, "false">:$modified,
                       DefaultValuedAttr<BoolAttr, "false">:$has_gate,
                       DefaultValuedAttr<BoolAttr, "false">:$has_beta,
                       DefaultValuedAttr<BoolAttr, "false">:$has_decay,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_RopeKernelOp : TesseraROCM_Op<"rope"> {
  let summary = "Rotary position embedding (interleaved-pair) kernel";
  let description = [{
    The directive `generate-rocm-rope-kernel` expands into an elementwise-per-pair
    gpu kernel applying rotary position embedding over a rank-2 `[M, D]` input
    (D even). One workgroup per row; the lanes stride the `D/2` pairs. For pair p
    (with `e = X[m,2p]`, `o = X[m,2p+1]`, angle `a = Theta[m,2p]`):

      O[m,2p]   = e·cos(a) − o·sin(a)
      O[m,2p+1] = e·sin(a) + o·cos(a)

    Matches the reference `_runtime_rope` (interleaved pairs, the even-indexed
    `theta` entry is the pair angle). `X` and `Theta` are both `[M, D]`; cos/sin
    run in f32 and lower through the `math` → ROCDL path. `M`/`D` are runtime
    `index` args; `dtype` is the f16/bf16/f32 storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_DSparkDraftBlockKernelOp : TesseraROCM_Op<"dspark_draft_block"> {
  let summary = "DSpark fused draft-block decode kernel";
  let description = [{
    The directive `generate-rocm-dspark-draft-block-kernel` expands into the
    DS2 fused draft-block kernel for the vanilla DSpark contract. The generated
    kernel consumes target hidden states, previous tokens, anchor positions, and
    DSpark draft-head weights; it emits logits, confidence logits, greedy draft
    tokens, and per-step hidden states with the same ABI as
    `tessera.stdlib.dspark.draft_block_forward`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_MLAAbsorbDecodeKernelOp : TesseraROCM_Op<"mla_absorb_decode"> {
  let summary = "Absorbed-latent MLA decode kernel";
  let description = [{
    The directive `generate-rocm-mla-absorb-decode-kernel` expands into the DK1
    native ROCm proof for `stdlib.attention.mla_decode_step`'s absorbed path.
    The generated kernel consumes preprojected/scaled query nope+rope slices,
    compressed latent cache `c`, rope-key cache, and MLA up-projection weights;
    it emits decode output without materializing per-head K/V.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_BlockSparseAttentionKernelOp : TesseraROCM_Op<"block_sparse_attention"> {
  let summary = "Selected-block sparse attention decode kernel";
  let description = [{
    The directive `generate-rocm-block-sparse-attn-kernel` expands into the DK2
    selected-block sparse attention proof. It consumes Q/K/V, explicit selected
    KV block ids in `B,Hkv,Sq,top_k` layout, and query positions, then computes
    exact causal GQA attention over only those selected blocks. This is the
    shared ROCm lowering target for MSA and DSA/NSA selected-block layouts.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_BlockSparseAttentionTiledKernelOp : TesseraROCM_Op<"block_sparse_attention_tiled"> {
  let summary = "Row-tiled selected-block sparse attention decode kernel";
  let description = [{
    The directive `generate-rocm-block-sparse-attn-kernel` also expands this
    row-tiled DK2 promotion form: one workgroup owns each `(B,Hq,Sq)` query row
    and threads cover value lanes. It preserves the same selected-block ABI as
    `block_sparse_attention`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_BlockSparseTopKSelectKernelOp : TesseraROCM_Op<"block_sparse_topk_select"> {
  let summary = "GPU-resident sparse-attention block top-k selection";
  let description = [{
    The directive `generate-rocm-block-sparse-topk-kernel` expands into a DK2
    MSA/NSA selector. It consumes Index Branch scores, query positions, block
    size, top-k, causal, and force-local flags, then writes deterministic
    selected block ids in `B,Hkv,Sq,top_k` layout for the tiled sparse-attention
    kernel.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_NormKernelOp : TesseraROCM_Op<"norm"> {
  let summary = "Row-wise normalization (rmsnorm / layer_norm) as a reduction kernel";
  let description = [{
    The directive `generate-rocm-norm-kernel` expands into a row-reduction gpu
    kernel realizing the **unweighted** row normalizations over the last axis
    (one workgroup per row, the lanes stride K and tree-reduce through LDS):

      * `kind = "rmsnorm"`    — `O[m,:] = X[m,:] / sqrt(mean_k X[m,k]² + eps)`.
      * `kind = "layer_norm"` — `O[m,:] = (X[m,:] − μ) / sqrt(var + eps)`, with
        `μ = mean_k X`, `var = mean_k (X − μ)²` (the stable squared-deviation
        reduction — not `E[x²] − E[x]²`, which cancels for large-offset rows).

    Matches the unweighted reference (`_apple_gpu_rowop_numpy`; the per-channel
    affine, when present, is a separate mul/add). rmsnorm is one row reduction;
    layer_norm is two (mean, then variance). Reductions run in **f32** regardless
    of storage dtype. `eps` is a trailing f32 runtime arg; `M`/`K` are runtime
    `index` args. `dtype` is the f16/bf16/f32 storage.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"rmsnorm\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_ReduceKernelOp : TesseraROCM_Op<"reduce"> {
  let summary = "Row reduction (sum / mean / max / min) over the last axis";
  let description = [{
    The directive `generate-rocm-reduce-kernel` expands into a row-reduction gpu
    kernel — the ROCm analog of the x86 AVX-512 reduction lane. Reduces each row
    of a `[M, K]` input over the last axis to a single value `O[M]` (the runtime
    folds an arbitrary reduced axis to `[outer=M, inner=K]` by transposing the
    reduced axes to the end, matching `_apple_gpu_dispatch_reduce`):

      * `kind = "sum"`  — `O[m] = Σ_k X[m,k]`
      * `kind = "mean"` — `O[m] = (Σ_k X[m,k]) / K`
      * `kind = "max"`  — `O[m] = max_k X[m,k]`
      * `kind = "min"`  — `O[m] = min_k X[m,k]`

    One workgroup per row (one wave per row tree-reduces through LDS); reductions
    run in **f32** regardless of storage dtype. `M`/`K` are runtime `index` args;
    `dtype` is the f16/bf16/f32 storage. Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"sum\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_ArgReduceKernelOp : TesseraROCM_Op<"argreduce"> {
  let summary = "Row arg-reduction (argmax / argmin) over the last axis";
  let description = [{
    The directive `generate-rocm-argreduce-kernel` expands into a row
    arg-reduction gpu kernel — the CUB/rocPRIM `DeviceReduce::ArgMax`/`ArgMin`
    pattern. Each thread carries the best `(value, index)` pair over its strided
    columns, then a `gpu.shuffle xor` butterfly reduces the PAIR within a 32-lane
    subgroup; the 8 per-subgroup partials combine to the row winner:

      * `kind = "argmax"` — `O[m] = argmax_k X[m,k]`
      * `kind = "argmin"` — `O[m] = argmin_k X[m,k]`

    numpy tie-break: the FIRST occurrence wins (on equal value keep the lower
    index). Values compare in **f32** regardless of storage dtype; the output is
    an `i32` index. `M`/`K` are runtime `index` args. Validated vs numpy on
    gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"argmax\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_ScanKernelOp : TesseraROCM_Op<"scan"> {
  let summary = "Row inclusive prefix scan (cumsum / cumprod / cummax / cummin)";
  let description = [{
    The directive `generate-rocm-scan-kernel` expands into a row inclusive-scan
    gpu kernel — the CUB/rocPRIM `BlockScan` technique. One workgroup per row
    processes the row in BD-element tiles with a running carry; each tile does a
    warp inclusive scan via `gpu.shuffle up` (Kogge-Stone) within a 32-lane
    subgroup, exclusive-scans the per-subgroup totals for the subgroup offset,
    then combines with the cross-tile carry:

      * `kind = "cumsum"`  — `O[m,k] = Σ_{j<=k} X[m,j]`
      * `kind = "cumprod"` — `O[m,k] = Π_{j<=k} X[m,j]`
      * `kind = "cummax"`  — running max; `cummin` — running min

    Scan runs in **f32** regardless of storage dtype; output is `[M,K]` (same
    shape). `M`/`K` are runtime `index` args. Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"cumsum\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_UnaryKernelOp : TesseraROCM_Op<"unary"> {
  let summary = "Elementwise unary math kernel (exp / log / sqrt / erf / …)";
  let description = [{
    The directive `generate-rocm-unary-kernel` expands into a flat elementwise
    gpu kernel applying a pointwise unary math function over N elements (one
    thread per element, strided grid) — the standalone S2 scalar-math /
    stability family, the unary sibling of the activation lane. All kinds
    compute in **f32** regardless of storage dtype; the transcendentals lower
    through the `math` → ROCDL path:

      * algebraic:   `sqrt`, `rsqrt`, `reciprocal`, `abs`, `neg`, `sign`
      * transcend.:  `exp`, `log`, `erf`, `tanh`, `sigmoid`
      * stability:   `log1p`, `expm1`, `softplus`

    `N` is a runtime `index` arg; `dtype` is the f16/bf16/f32 storage.
    Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"exp\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_BinaryKernelOp : TesseraROCM_Op<"binary"> {
  let summary = "Elementwise binary arithmetic kernel (sub / div / pow / max / min)";
  let description = [{
    The directive `generate-rocm-binary-kernel` expands into a flat 2-operand
    elementwise gpu kernel applying a pointwise binary arithmetic function over
    N elements (one thread per element, strided grid) — the standalone S2
    binary-arithmetic family, the binary sibling of the unary-math lane. Both
    operands and the output share `dtype`; the body computes in **f32**
    regardless (`pow` lowers through the `math` → ROCDL path):

      * `sub` = a − b,  `div` = a / b,  `pow` = a ** b
      * `maximum` / `minimum` — IEEE NaN-propagating

    `N` is a runtime `index` arg; `dtype` is the f16/bf16/f32 storage.
    Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"sub\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_CompareKernelOp : TesseraROCM_Op<"compare"> {
  let summary = "Elementwise comparison kernel (eq / ne / lt / le / gt / ge)";
  let description = [{
    The directive `generate-rocm-compare-kernel` expands into a flat 2-operand
    elementwise gpu kernel applying a pointwise comparison over N elements (one
    thread per element, strided grid) — the standalone S2 comparison family,
    producing a boolean (`i8` 0/1) result. The two operands share a float
    storage `dtype` (f16/bf16/f32); the comparison runs in **f32**:

      * `eq` = a == b, `ne` = a != b, `lt` = a < b
      * `le` = a <= b, `gt` = a > b, `ge` = a >= b

    NaN semantics follow numpy: every predicate is ORDERED (NaN → false) EXCEPT
    `ne`, which is unordered-or-not-equal (NaN → true). `N` is a runtime `index`
    arg. Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"eq\"">:$kind,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_LogicalKernelOp : TesseraROCM_Op<"logical"> {
  let summary = "Elementwise logical kernel (and / or / xor / not) over i8 bool";
  let description = [{
    The directive `generate-rocm-logical-kernel` expands into a flat elementwise
    gpu kernel applying a pointwise logical operation over N elements (one thread
    per element, strided grid) — the standalone S2 logical family, operating on
    `i8` booleans:

      * `and` = a && b, `or` = a || b, `xor` = a ^ b  (binary)
      * `not` = !a                                     (unary)

    Inputs are normalized to a boolean via `a != 0` (matching numpy, where any
    nonzero value is true). The result is an `i8` mask (0/1). `not` takes one
    operand; the rest take two. `N` is a runtime `index` arg. Validated vs numpy
    on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"and\"">:$kind);
  let assemblyFormat = "attr-dict";
}

def ROCM_BitwiseKernelOp : TesseraROCM_Op<"bitwise"> {
  let summary = "Elementwise bitwise kernel (and / or / xor / not) over i32";
  let description = [{
    The directive `generate-rocm-bitwise-kernel` expands into a flat elementwise
    gpu kernel applying a pointwise bitwise operation over N elements (one thread
    per element, strided grid) — the standalone S2 bitwise family, operating on
    `i32` integers:

      * `and` = a & b, `or` = a | b, `xor` = a ^ b  (binary)
      * `not` = ~a (= a ^ -1)                        (unary)

    Unlike the logical lane, operands are NOT normalized — the op acts on the
    full integer bit pattern. `not` takes one operand; the rest take two. `N` is
    a runtime `index` arg. Validated vs numpy on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"and\"">:$kind);
  let assemblyFormat = "attr-dict";
}

def ROCM_WhereKernelOp : TesseraROCM_Op<"where"> {
  let summary = "Elementwise ternary select — where(cond, a, b)";
  let description = [{
    The directive `generate-rocm-where-kernel` expands into a flat 3-operand
    elementwise gpu kernel computing numpy `where`/select over N elements:

      `where(cond, a, b)[i] = cond[i] ? a[i] : b[i]`

    `cond` is an `i8` boolean (normalized via `!= 0`, matching numpy where any
    nonzero is true); `a`/`b`/output share a float storage `dtype` (f16/bf16/f32).
    `N` is a runtime `index` arg. Validated vs `np.where` on gfx1151.
  }];
  let arguments = (ins StrAttr:$name,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def ROCM_PhiloxKernelOp : TesseraROCM_Op<"philox"> {
  let summary = "Counter-based Philox-4x32-10 uniform RNG kernel";
  let description = [{
    The directive `generate-rocm-philox-kernel` expands into a flat counter-based
    RNG gpu kernel: one thread per counter block `b` writes 4 uniform-f32 in
    [0,1) to `out[4b..4b+3]` from Philox-4x32-10 (Salmon et al. 2011 — the
    JAX / cuRAND algorithm) keyed by `(seed_lo, seed_hi)` with counter
    `(cbase+b, 0, 0, 0)`. Bit-identical to the x86 kernel + the
    `tessera.rng_device` numpy reference. Args:
    `(out : memref<?xf32>, N : index, seed_lo i32, seed_hi i32,
      cbase_lo i32, cbase_hi i32)`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_GatherKernelOp : TesseraROCM_Op<"gather"> {
  let summary = "Masked strided-copy / gather kernel (0-move lane)";
  let description = [{
    The directive `generate-rocm-gather-kernel` expands into a flat masked-gather
    gpu kernel — the 0-move lane (P4) backing pad / cat / roll / flip / tile /
    repeat / stack: one thread per output element,
    `out[i] = (0 <= idx[i] < src_n) ? src[idx[i]] : out[i]`. A negative index
    leaves `out[i]` unchanged (pad fill / cat per-input accumulate). The host
    computes the integer index map from the op's numpy semantics on an arange
    grid; this kernel does the f32 data movement. Args:
    `(src : memref<?xf32>, src_n : index, idx : memref<?xi64>,
      out : memref<?xf32>, N : index)`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_EbmLangevinKernelOp : TesseraROCM_Op<"ebm_langevin"> {
  let summary = "EBM Langevin step with on-device Philox-4x32-10 noise";
  let description = [{
    The directive `generate-rocm-ebm-langevin-kernel` expands into a Langevin
    step gpu kernel that DRAWS its own Gaussian noise from counter-based
    Philox-4x32-10 (the P6 RNG generator), so the noise is never supplied from
    the host: one thread per element `i` computes
    `z = sqrt(-2 ln u0)·cos(2π u1)` (Box-Muller) from the first two Philox
    outputs for counter `(c0+i, c1, c2, c3)` under key `(k0, k1)` mapped to
    (0,1) by `(x+0.5)·2^-32`, then `out[i] = y[i] - eta·grad[i] +
    noise_scale·z`. Mirrors `tessera.ebm.langevin_step_philox`. CPU analog:
    avx512_ebm_langevin_f32. Args: `(y : memref<?xf32>, grad : memref<?xf32>,
    n : index, eta f32, noise_scale f32, k0 i32, k1 i32, c0 i32, c1 i32,
    c2 i32, c3 i32, out : memref<?xf32>)`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_EbmAffineLangevinKernelOp : TesseraROCM_Op<"ebm_affine_langevin"> {
  let summary = "EBM Langevin step taking host-drawn Gaussian noise as an input";
  let description = [{
    The directive `generate-rocm-ebm-affine-langevin-kernel` expands into a
    Langevin-step gpu kernel that takes its noise AS AN INPUT rather than drawing
    its own Philox noise — the manifold half of the P7 EBM follow-up. One thread
    per element `i` computes `out[i] = y[i] - eta·grad[i] + noise_scale·noise[i]`.
    `tessera.ebm.{bivector,sphere}_langevin_step` project the gradient and a
    host-drawn, grade-projected Gaussian onto a manifold subspace, then take this
    affine combination. Mirrors the Apple `_try_apple_gpu_langevin_step_f32`
    bridge; CPU analog: tessera_x86_ebm_affine_langevin_f32. Args:
    `(y : memref<?xf32>, grad : memref<?xf32>, noise : memref<?xf32>, n : index,
    eta f32, noise_scale f32, out : memref<?xf32>)`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_ScatterKernelOp : TesseraROCM_Op<"scatter"> {
  let summary = "Indexed-store / scatter kernel with atomic reduce (0-reduce)";
  let description = [{
    The directive `generate-rocm-scatter-kernel` expands into an indexed-store
    gpu kernel — the 0-reduce lane (P8), companion to `tessera_rocm.gather`:
    `out[idx[i]] (op)= src[i]` row-wise. One thread per source element (i, c):
    target row `idx[i]`, column `c`; `mode = "set"` plain-stores, while
    `"add"` / `"min"` / `"max"` use `memref.atomic_rmw` so concurrent updates to
    the same output row reduce correctly (scatter_add / scatter_reduce). The host
    moves the scatter axis to 0, flattens the trailing dims into a `row_len`-wide
    row, and preloads `out` with the base tensor. CPU analog: avx512_scatter_f32.
    Args: `(out : memref<?xf32>, out_rows : index, src : memref<?xf32>,
      idx : memref<?xi64>, n_idx : index, row_len : index)`.
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$mode);
  let assemblyFormat = "attr-dict";
}

def ROCM_SortKernelOp : TesseraROCM_Op<"sort"> {
  let summary = "Cooperative bitonic sort kernel (key + index, last-axis rows)";
  let description = [{
    The directive `generate-rocm-sort-kernel` expands into a cooperative bitonic
    sort gpu kernel — the sort lane (P9) backing sort / argsort / top_k. One
    block sorts one row of `pn` (power-of-two) f32 keys ascending, carrying the
    parallel i64 index array so the same compare-exchange schedule realizes
    argsort / top_k; the host pads each row to `pn` with +INF sentinels and
    flips the result for descending (matching the numpy reference). The bitonic
    network is data-independent — the same schedule as the x86 AVX-512 kernel —
    with a `gpu.barrier` between stages. The host passes `nstages = log2(pn)`
    so the kernel runs the schedule as two nested `scf.for` loops (no in-kernel
    log2). Grid = one block per row. Args:
    `(keys : memref<?xf32>, idx : memref<?xi64>, pn : index, nstages : index)`.
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "attr-dict";
}

def ROCM_CliffordKernelOp : TesseraROCM_Op<"clifford"> {
  let summary = "Table-driven geometric-algebra bilinear product kernel (Cl(3,0))";
  let description = [{
    The directive `generate-rocm-clifford-kernel` expands into a table-driven
    geometric-algebra bilinear-product gpu kernel — the GA lane (P12) backing
    geometric_product / wedge / left_contraction on Cl(3,0) (8 blades). A
    multivector product is a structured bilinear form driven by a compile-time
    Cayley table: `out[k] = Σ sign·a[i]·b[j]`. One thread per batch element
    loads the 8 blade coefficients of `a` and `b` and accumulates the 8 output
    blades by the (sign, i, j -> k) triples for `kind` (0 = geometric_product,
    1 = wedge, 2 = left_contraction), unrolled at generation time so the table
    is a true compile-time constant. Layout is batch-major ([n, 8]). CPU analog:
    avx512_clifford_f32. Args:
    `(A : memref<?xf32>, B : memref<?xf32>, n : index, out : memref<?xf32>)`.
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$kind);
  let assemblyFormat = "attr-dict";
}

def ROCM_BufferLoadOp : TesseraROCM_Op<"buffer_load", [MemoryEffects<[MemRead]>]> {
  let summary = "AMD buffer load with native out-of-bounds handling";
  let description = [{
    Global load via AMD buffer addressing: one 64-bit base pointer per wave plus
    a per-thread 32-bit offset, with hardware out-of-bounds handling (out-of-range
    lanes read zero).  This collapses the predicated masked-load control flow an
    explicit bounds check would emit (AMD Gluon GEMM tutorial: ~140 branches -> 4).
    The `oob` unit attribute records that hardware OOB is relied upon rather than
    an explicit mask.
  }];
  let arguments = (ins AnyType:$base, AnyType:$offset, UnitAttr:$oob);
  let results = (outs AnyType:$res);
  let assemblyFormat =
    "$base `,` $offset attr-dict `:` type($base) `,` type($offset) `->` type($res)";
}

def ROCM_DsReadTrOp : TesseraROCM_Op<"ds_read_tr", [MemoryEffects<[MemRead]>]> {
  let summary = "Transposing LDS read (ds_read_tr family)";
  let description = [{
    Reads a tile from LDS (shared memory) with an in-hardware transpose (the
    `ds_read_tr*` instructions), producing an MFMA/WMMA operand in the layout the
    matrix core expects without a separate software transpose.
  }];
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}

def ROCM_EmitOp : TesseraROCM_Op<"emit", [Terminator]> {
  let assemblyFormat = "attr-dict";
}
