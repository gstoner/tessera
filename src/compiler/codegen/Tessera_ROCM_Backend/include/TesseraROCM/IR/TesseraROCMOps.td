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
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$head_dim,
                       DefaultValuedAttr<StrAttr, "\"f16\"">:$dtype,
                       DefaultValuedAttr<BoolAttr, "false">:$gqa,
                       DefaultValuedAttr<BoolAttr, "false">:$sliding_window);
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
