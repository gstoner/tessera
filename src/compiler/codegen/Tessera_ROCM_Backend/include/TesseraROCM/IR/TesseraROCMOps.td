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
