//===- TesseraTargetMetalium.td - TT-Metalium Target IR -*- tablegen -*-===//
//
// Draft dialect for targeting Tenstorrent Wormhole via TT-Metalium.
// Captures core primitives: DMA (DRAM<->SRAM), Load/Store (SRAM<->regs),
// and Matmul (tile-local FMA over vectors).
//
// This mirrors the style of the Tessera Target IR (CUDA/PTX) dialect,
// but specializes operations/attributes for the Metalium stack.
//
// NOTE: This is a *draft* intended for bring-up. Expect evolution.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TARGET_METALIUM
#define TESSERA_TARGET_METALIUM

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//------------------------------------------------------------------------------
// Dialect
//------------------------------------------------------------------------------

def TesseraMetalium_Dialect : Dialect {
  let name = "tessera_metalium";
  let summary = "Tessera Target IR for TT-Metalium (Tenstorrent Wormhole)";
  let description = [{
    This dialect expresses target-specific operations for Tenstorrent
    Wormhole devices through the TT-Metalium runtime/ABI. It focuses on
    explicit data movement between DRAM and on-core SRAM via DMA, on-SRAM
    loads/stores to vector registers, and tile-local matrix multiply-accumulate.

    Typical lowering flow:
      Tile IR (device-agnostic tiles) → tessera_metalium ops →
      vector/linalg (optional) → TT-Metalium codegen
  }];

  let cppNamespace = "::mlir::tessera::metalium";
  let dependentDialects = ["vector::VectorDialect"];
}

//------------------------------------------------------------------------------
// Attributes & Types
//------------------------------------------------------------------------------

def MemorySpaceAttr : StrEnumAttr<"MemorySpace",
  "Metalium memory space (dram|sram)",
  [ StrEnumAttrCase<"dram">, StrEnumAttrCase<"sram"> ]> {
  let cppNamespace = "::mlir::tessera::metalium";
}

def DmaDirectionAttr : StrEnumAttr<"DmaDirection",
  "DMA direction (dram_to_sram | sram_to_dram | sram_to_sram | dram_to_dram)",
  [
    StrEnumAttrCase<"dram_to_sram">,
    StrEnumAttrCase<"sram_to_dram">,
    StrEnumAttrCase<"sram_to_sram">,
    StrEnumAttrCase<"dram_to_dram">
  ]> {
  let cppNamespace = "::mlir::tessera::metalium";
}

// Optional token for async completion modeling.
def TokenType : TypeDef<TesseraMetalium_Dialect, "Token"> {
  let mnemonic = "token";
  let summary = "Async completion token for DMA/async ops";
  let description = [{
    Represents completion of an asynchronous operation (e.g., DMA).
    Can be passed to fence/wait ops if added in the future.
  }];
}

// A simple CoreRange attribute: [x0, y0]..[x1, y1]
def CoreRangeAttr : AttrDef<TesseraMetalium_Dialect, "CoreRange"> {
  let summary = "Inclusive 2D core coordinate range";
  let description = [{
    Describes a rectangular range on the Wormhole core mesh.

    Example:
      #tessera_metalium.core_range<[0,0],[7,11]>
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$startXY,
    ArrayRefParameter<"int64_t">:$endXY
  );
  let assemblyFormat = "`<` `[` $startXY `]` `,` `[` $endXY `]` `>`";
}

//------------------------------------------------------------------------------
// Base op class
//------------------------------------------------------------------------------

class Metalium_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraMetalium_Dialect, mnemonic, traits> {
}

//------------------------------------------------------------------------------
// DMA: bulk movement between DRAM and SRAM
//------------------------------------------------------------------------------

def Metalium_DmaOp : Metalium_Op<"dma",
  [MemoryEffects<[MemRead<1>, MemWrite<0>]>, AttrSizedOperandSegments]> {
  let summary = "DMA copy between DRAM/SRAM with optional 2D striding";
  let description = [{
    Performs an optimized DMA transfer. Commonly used for DRAM↔SRAM moves.

    Required operands:
      - %dst : memref<...>  Destination buffer (memref with memorySpace attr)
      - %src : memref<...>  Source buffer (memref with memorySpace attr)
      - %shape : index[]     Transfer shape in elements (1D or 2D)

    Optional operands:
      - %dst_strides : index[]   Per-dimension strides in elements
      - %src_strides : index[]   Per-dimension strides in elements

    Attributes:
      - direction : #tessera_metalium<DmaDirection>
      - element_size_bytes : i64 (required if types are opaque)
      - burst : i64 (optional)
      - async : unit (optional)  If present, op returns a !tessera_metalium.token

    Example:
      %t = tessera_metalium.dma %sram, %dram, [%c128, %c64],
               [%c128, %c1], [%c128, %c1]
          direction = #tessera_metalium<dma_direction dram_to_sram>,
          burst = 256, async
          : (memref<?x?xf16, #tessera_metalium.memspace<"sram">>,
             memref<?x?xf16, #tessera_metalium.memspace<"dram">>) -> !tessera_metalium.token
  }];

  let arguments = (ins
    AnyMemRef:$dst,
    AnyMemRef:$src,
    Variadic<Index>:$shape,
    Variadic<Index>:$dst_strides,
    Variadic<Index>:$src_strides,
    DmaDirectionAttr:$direction,
    OptionalAttr<I64Attr>:$element_size_bytes,
    OptionalAttr<I64Attr>:$burst,
    UnitAttr:$async
  );

  let results = (outs Optional<TokenType>:$completion);

  let assemblyFormat = [{
    $dst `,` $src `,` `[` $shape `]`
    (` ,` `[` $dst_strides `]` `,` `[` $src_strides `]`)? 
    attr-dict-with-keyword
    `:` type($dst) `,` type($src)
  }];
}

//------------------------------------------------------------------------------
// LOAD: SRAM → vector registers
//------------------------------------------------------------------------------

def Metalium_Load2DOp : Metalium_Op<"load2d",
  [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
   MemoryEffects<[MemRead<0>]>]> {
  let summary = "Load a 2D tile from SRAM into a vector register";
  let description = [{
    Loads a rectangular tile from SRAM into a vector register value.

    Operands:
      - %src : memref<?x?xT, #memspace<\"sram\">>
      - %offsets : index[], 2D offset in elements (row, col)
      - %shape : index[], 2D tile shape (rows, cols)
      - %strides : index[], 2D strides in elements (rowStride, colStride)

    Result:
      - vector<rows x cols x T>

    Example:
      %tile = tessera_metalium.load2d %sram[%rOff, %cOff],
                 [%rows, %cols], [%rStride, %cStride]
             : memref<?x?xf16, #tessera_metalium.memspace<"sram">>
               -> vector<?x?xf16>
  }];

  let arguments = (ins
    AnyMemRef:$src,
    Variadic<Index>:$offsets,
    Variadic<Index>:$shape,
    Variadic<Index>:$strides
  );

  let results = (outs AnyVector:$result);

  let assemblyFormat = [{
    $src `[` $offsets `]` `,` `[` $shape `]` `,` `[` $strides `]`
    attr-dict `:` type($src) `->` type($result)
  }];
}

//------------------------------------------------------------------------------
// STORE: vector registers → SRAM
//------------------------------------------------------------------------------

def Metalium_Store2DOp : Metalium_Op<"store2d",
  [MemoryEffects<[MemWrite<0>]>]> {
  let summary = "Store a 2D vector tile into SRAM";
  let description = [{
    Stores a register-resident 2D vector tile back into SRAM.

    Operands:
      - %val : vector<rows x cols x T>
      - %dst : memref<?x?xT, #memspace<\"sram\">>
      - %offsets : index[], 2D offset in elements (row, col)
      - %strides : index[], 2D strides in elements

    Example:
      tessera_metalium.store2d %tile, %sram[%rOff, %cOff], [%rStride, %cStride]
        : vector<?x?xf16>, memref<?x?xf16, #tessera_metalium.memspace<"sram">>
  }];

  let arguments = (ins
    AnyVector:$value,
    AnyMemRef:$dst,
    Variadic<Index>:$offsets,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $value `,` $dst `[` $offsets `]` `,` `[` $strides `]`
    attr-dict `:` type($value) `,` type($dst)
  }];
}

//------------------------------------------------------------------------------
// MATMUL: tile-local vector FMA (C = A x B + C)
//------------------------------------------------------------------------------

def Metalium_MatmulOp : Metalium_Op<"matmul",
  [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tile-local vector matmul over SRAM-resident tiles";
  let description = [{
    Performs a small GEMM over tiles resident in registers or SRAM.

    Common forms:
      1) Register tiles:
         %C = tessera_metalium.matmul %A, %B, %Cinit
              {tile = [M, N, K], layout = "row_col"}
              : (vector<MxKxTa>, vector<KxNxTb>, vector<MxNxf32>) -> vector<MxNxf32>

      2) SRAM-backed:
         %Areg = tessera_metalium.load2d ...
         %Breg = tessera_metalium.load2d ...
         %Creg = tessera_metalium.matmul %Areg, %Breg, %Cinit ...
         tessera_metalium.store2d %Creg, %Csram[...]

    Attributes:
      - tile : i64[3]  // [M, N, K]
      - layout : str   // "row_col", "col_row", etc.
      - accumulate : str (optional) // "f32" (default), "bf16", "i32" for int
  }];

  let arguments = (ins
    AnyVector:$a,
    AnyVector:$b,
    AnyVector:$cInit,
    ArrayAttr:$tile,
    OptionalAttr<StrAttr>:$layout,
    OptionalAttr<StrAttr>:$accumulate
  );

  let results = (outs AnyVector:$cOut);

  let assemblyFormat = [{
    $a `,` $b `,` $cInit
    attr-dict-with-keyword
    `:` type($a) `,` type($b) `,` type($cInit) `->` type($cOut)
  }];
}

//------------------------------------------------------------------------------
// Dialect-level helpers (pseudo memspace attr for memref pretty printing)
//------------------------------------------------------------------------------

def MemorySpacePrettyName : AttrDef<TesseraMetalium_Dialect, "memspace"> {
  let summary = "Wrapper attr to use in memref<..., #tessera_metalium.memspace<\"sram\">>";
  let parameters = (ins MemorySpaceAttr:$space);
  let assemblyFormat = "`<` $space `>`";
}

#endif // TESSERA_TARGET_METALIUM
