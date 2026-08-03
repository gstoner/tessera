//===- TesseraX86Ops.td ---------------------------------------*- tablegen -*-===//
//
// Hardware-free x86 Target IR ops (Decision #19, built by W0.10).
//
// Two kinds live here, and the split is deliberate:
//
//   * VALUE-CARRYING ops (`amx_tile_load`, `amx_dpbf16ps`, ...) declare real
//     operands and results, so the verifier can reject a tile dot-product whose
//     operands never came from a tile load. This is the property that makes a
//     Target IR dialect worth having: a substring test cannot check it.
//
//   * DIRECTIVE ops (`avx512_gemm_microkernel`, `elementwise`, `abi_call`)
//     are attr-dict markers that fix placement and configuration without
//     carrying values -- the same shape ROCm uses for `wmma_gemm`.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_X86_OPS_TD
#define TESSERA_X86_OPS_TD

include "TesseraX86Dialect.td"

//===----------------------------------------------------------------------===//
// AMX -- tile registers and the BF16/INT8 dot-product accumulate
//===----------------------------------------------------------------------===//

def X86_AmxTileLoadOp : TesseraX86_Op<"amx_tile_load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a 2-D AMX tile register from strided memory";
  let description = [{
    The `tileloadd` family. `stride` is the row pitch in bytes. The result is an
    opaque `!tessera_x86.tile` so downstream dot-products can only consume a
    value that genuinely came from a tile load.
  }];
  let arguments = (ins AnyMemRef:$base, I64:$stride);
  let results = (outs X86_TileType:$tile);
  let assemblyFormat =
    "$base `,` $stride attr-dict `:` type($base) `,` type($stride) `->` type($tile)";
}

def X86_AmxTileStoreOp : TesseraX86_Op<"amx_tile_store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store an AMX tile register to strided memory";
  let arguments = (ins X86_TileType:$tile, AnyMemRef:$base, I64:$stride);
  let assemblyFormat =
    "$tile `,` $base `,` $stride attr-dict `:` type($tile) `,` type($base) `,` type($stride)";
}

def X86_AmxTileZeroOp : TesseraX86_Op<"amx_tile_zero"> {
  let summary = "Produce a zeroed AMX tile register (`tilezero`)";
  let description = [{
    The accumulator initializer. Having it as a real op means an accumulator
    chain starts from a defined value rather than an undef the verifier cannot
    distinguish from a bug.
  }];
  let results = (outs X86_TileType:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}

def X86_AmxDpbf16psOp : TesseraX86_Op<"amx_dpbf16ps"> {
  let summary = "AMX BF16 tile dot-product accumulate (`tdpbf16ps`)";
  let description = [{
    `acc += a * b` over BF16 tile operands with f32 accumulation -- the AMX
    matmul primitive the x86 backend's BF16 GEMM is built from. Accumulator
    precision is f32 by construction, matching Decision #15a's rule that the
    accumulator lives in `numeric_policy` and not in the storage dtype.
  }];
  let arguments = (ins X86_TileType:$a, X86_TileType:$b, X86_TileType:$acc);
  let results = (outs X86_TileType:$res);
  let assemblyFormat =
    "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($res)";
}

def X86_AmxDpbusdOp : TesseraX86_Op<"amx_dpbusd"> {
  let summary = "AMX INT8 tile dot-product accumulate (`tdpbusd`)";
  let description = [{
    The INT8 analog of `amx_dpbf16ps`, with i32 accumulation. The AMX INT8 GEMM
    device lane that landed in #489 is gated on AMX capability, which no machine
    in the current fleet reports -- so this op is IR-checkable today and
    execution-checkable only on Intel AMX silicon.

    **AMX lowering is OPTIONAL and deprioritized (project direction,
    2026-08-02).** Per the project owner, AMX is expected to be superseded by
    the ACE matrix instructions jointly agreed by Intel and AMD for future
    CPUs, so building an AMX -> `amx.*` lowering is not worth the cost now.
    (Recorded as owner direction; the ACE specifics are not independently
    verified here.) The AMX ops stay in the dialect as the IR-level contract --
    they cost nothing to keep, they document the tile/accumulator shape, and
    they give the eventual ACE ops a structure to follow. Do not read their
    presence as a commitment to emit AMX.
  }];
  let arguments = (ins X86_TileType:$a, X86_TileType:$b, X86_TileType:$acc);
  let results = (outs X86_TileType:$res);
  let assemblyFormat =
    "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// AVX-512 -- the directive layer
//===----------------------------------------------------------------------===//

def X86_Avx512GemmMicrokernelOp : TesseraX86_Op<"avx512_gemm_microkernel"> {
  let summary = "An AVX-512 register-blocked GEMM microkernel to realize";
  let description = [{
    The x86 analog of `tessera_rocm.wmma_gemm`: a matmul *directive* naming the
    register-blocked microkernel shape (`m` x `n` accumulator tile over a `k`
    loop) rather than authored vector IR. AVX-512 is the lane that actually
    executes on the current fleet (Zen 5 has AVX-512, no AMX).
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$m, I64Attr:$n, I64Attr:$k,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def X86_PackBPanelOp : TesseraX86_Op<"pack_b_panel"> {
  let summary = "Pack a B panel into the microkernel's expected layout";
  let description = [{
    The classic GEMM B-panel pack. Explicit in the IR because packing is a real
    cost the scheduler must be able to see and hoist, not an implementation
    detail hidden inside a shim call.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$kc, I64Attr:$nc,
                       DefaultValuedAttr<StrAttr, "\"f32\"">:$dtype);
  let assemblyFormat = "attr-dict";
}

def X86_ElementwiseOp : TesseraX86_Op<"elementwise"> {
  let summary = "Hardware-free marker for a pointwise/elementwise op on x86";
  let description = [{
    The generic pointwise node, mirroring `tessera_rocm.elementwise`. Attributes
    record the originating Graph IR op and the ISA level.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// The shim boundary -- modelled, not hidden
//===----------------------------------------------------------------------===//

def X86_AbiCallOp : TesseraX86_Op<"abi_call"> {
  let summary = "A call into the hand-written x86 C-ABI kernel shim";
  let description = [{
    Today `TileToX86Pass` lowers to bare `func::CallOp`s into a hand-written C
    shim. That is a legitimate lane -- but as a raw `func.call` it is invisible
    to any x86-specific analysis, and it was the reason x86 appeared to need no
    Target IR at all. Naming the boundary makes it explicit which work is
    compiler-generated and which is delegated to the shim, so the arbiter of
    Decision #28 can tell the two apart when scoring candidates.
  }];
  let arguments = (ins StrAttr:$symbol,
                       DefaultValuedAttr<StrAttr, "\"tessera_x86_c\"">:$abi);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// The Python Target-IR lowering's vocabulary
//
// `_lower_x86_op` emits these three today. They are declared here so the
// emitted artifact IS the dialect rather than merely naming it -- the same
// correction W0.9 applied to `tessera_rocm.elementwise` and friends.
//===----------------------------------------------------------------------===//

def X86_KernelOp : TesseraX86_Op<"kernel"> {
  let summary = "A lowered x86 kernel node (compiled lane or shim-resolved)";
  let description = [{
    The generic node the Tile->Target lowering emits per op. Attributes record
    the `abi` symbol, the resolved `runtime_lane`, and the proof `status`
    (`device_verified_jit` when a native lane realizes it, `lowered` otherwise).
  }];
  let assemblyFormat = "attr-dict";
}

def X86_KVCacheReadOp : TesseraX86_Op<"kv_cache_read"> {
  let summary = "Hardware-free marker for a KV-cache read lowered to x86";
  let assemblyFormat = "attr-dict";
}

def X86_UnsupportedOp : TesseraX86_Op<"unsupported"> {
  let summary = "An op with no native x86 target contract";
  let description = [{
    Decision #21 requires a backend that cannot lower an op to emit a stable
    diagnostic naming the op and the target, never a silent no-op. This is the
    IR-visible form of that refusal; `reason` carries the explanation.
  }];
  let assemblyFormat = "attr-dict";
}

#endif  // TESSERA_X86_OPS_TD
