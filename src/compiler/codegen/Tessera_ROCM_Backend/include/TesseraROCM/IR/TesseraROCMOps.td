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
