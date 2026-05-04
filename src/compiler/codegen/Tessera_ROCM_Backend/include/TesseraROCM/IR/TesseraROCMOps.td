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

def ROCM_EmitOp : TesseraROCM_Op<"emit", [Terminator]> {
  let assemblyFormat = "attr-dict";
}
