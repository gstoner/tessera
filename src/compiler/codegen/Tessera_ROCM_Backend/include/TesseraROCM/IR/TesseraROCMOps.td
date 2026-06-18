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
  let arguments = (ins TokenType:$tkn);
  let assemblyFormat = "$tkn attr-dict `:` type($tkn)";
}

def ROCM_MFMAOp : TesseraROCM_Op<"mfma"> {
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
