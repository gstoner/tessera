//===- TesseraTileOps.td - Tessera Tile IR Dialect -*- tablegen -*-===//
//
// This file defines the operations for Tessera's Tile IR dialect in MLIR.
// The Tile IR represents low-level compute and memory operations that execute
// within tiles, with explicit control over memory hierarchies, thread mapping,
// and hardware-specific optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_TILE_OPS
#define TESSERA_TILE_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Tessera Tile IR Dialect Definition
//===----------------------------------------------------------------------===//

def TesseraTile_Dialect : Dialect {
  let name = "tessera_tile";
  let summary = "Tessera Tile IR Dialect";
  let description = [{
    The Tessera Tile IR dialect represents low-level tile-based computation
    with explicit memory hierarchy management, thread mapping, and hardware
    optimization. This dialect serves as the final stage before target-specific
    code generation (CUDA, HIP, Triton, etc.).

    Key features:
    - Explicit memory hierarchy (global, shared, register)
    - Thread and warp-level operations
    - Hardware-specific optimizations (Tensor Cores, TMEM)
    - Cooperative memory operations
    - Fine-grained synchronization
    - Pipeline specialization for different algorithms
    - CuTe-style layout and access patterns
  }];

  let cppNamespace = "::mlir::tessera::tile";
  let dependentDialects = ["memref::MemRefDialect", "arith::ArithDialect", 
                          "scf::SCFDialect", "gpu::GPUDialect",
                          "tessera::schedule::TesseraScheduleDialect"];
}

//===----------------------------------------------------------------------===//
// Tessera Tile Types
//===----------------------------------------------------------------------===//

class TesseraTile_Type<string name, string typeMnemonic> : 
    TypeDef<TesseraTile_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Memory Space Type
def TesseraTile_MemorySpaceType : TesseraTile_Type<"MemorySpace", "memory_space"> {
  let summary = "Memory hierarchy level specification";
  
  let parameters = (ins 
    "StringAttr":$space,
    OptionalParameter<"IntegerAttr">:$bank,
    OptionalParameter<"IntegerAttr">:$alignment
  );
  
  let assemblyFormat = "`<` $space (`,` `bank` `:` $bank^)? "
                       "(`,` `align` `:` $alignment^)? `>`";
}

// Thread Mapping Type  
def TesseraTile_ThreadMapType : TesseraTile_Type<"ThreadMap", "thread_map"> {
  let summary = "Thread mapping configuration";
  
  let parameters = (ins 
    ArrayRefParameter<"int64_t">:$threadShape,
    ArrayRefParameter<"int64_t">:$warpShape,
    OptionalParameter<"StringAttr">:$strategy
  );
  
  let assemblyFormat = "`<` $threadShape `,` $warpShape "
                       "(`,` $strategy^)? `>`";
}

// Tile Fragment Type
def TesseraTile_FragmentType : TesseraTile_Type<"Fragment", "fragment"> {
  let summary = "Tile fragment for register-level operations";
  
  let parameters = (ins 
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape,
    "StringAttr":$layout,
    OptionalParameter<"StringAttr">:$accumulate
  );
  
  let assemblyFormat = "`<` $elementType `,` $shape `,` $layout "
                       "(`,` `acc` `:` $accumulate^)? `>`";
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class TesseraTile_Op<string mnemonic, list<Trait> traits = []> :
    Op<TesseraTile_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def TesseraTile_AllocOp : TesseraTile_Op<"alloc", [
    MemoryEffects<[MemAlloc]>
]> {
  let summary = "Allocate memory in specified memory space";
  let description = [{
    Allocates memory in the specified memory hierarchy level (global, shared,
    register, or TMEM). Supports alignment requirements and banking hints.
    
    Example:
    ```mlir
    %shared = tessera_tile.alloc() : memref<128x64xf16, #tessera_tile.memory_space<"shared", align: 128>>
    %tmem = tessera_tile.alloc() : memref<64x32xf32, #tessera_tile.memory_space<"tmem", bank: 0>>
    ```
  }];
  
  let arguments = (ins 
    Variadic<Index>:$dynamicSizes,
    OptionalAttr<StrAttr>:$alignment,
    OptionalAttr<I64Attr>:$bank
  );
  
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref);
  
  let assemblyFormat = "`(` $dynamicSizes `)` attr-dict `:` type($memref)";
  
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def TesseraTile_LoadOp : TesseraTile_Op<"load", [
    TypesMatchWith<"result type matches element type of 'memref'",
                   "memref", "result", "$_self.cast<MemRefType>().getElementType()">
]> {
  let summary = "Load from memory with thread mapping";
  let description = [{
    Loads data from memory with explicit thread mapping and vectorization.
    Supports cooperative loading patterns and cache control hints.
    
    Example:
    ```mlir
    %value = tessera_tile.load %memref[%i, %j] 
             thread_map = #tessera_tile.thread_map<[32, 4], [32, 1]>
             : memref<1024x512xf16> -> f16
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$memref,
    Variadic<Index>:$indices,
    OptionalAttr<TesseraTile_ThreadMapType>:$thread_map,
    OptionalAttr<StrAttr>:$cache_hint
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$memref `[` $indices `]` "
                       "(`thread_map` `=` $thread_map^)? "
                       "(`cache` `=` $cache_hint^)? "
                       "attr-dict `:` type($memref) `->` type($result)";
  
  let hasFolder = 1;
}

def TesseraTile_StoreOp : TesseraTile_Op<"store"> {
  let summary = "Store to memory with thread mapping";
  let description = [{
    Stores data to memory with explicit thread mapping and write policies.
    Supports cooperative storing patterns and cache control hints.
  }];
  
  let arguments = (ins 
    AnyType:$value,
    AnyMemRef:$memref,
    Variadic<Index>:$indices,
    OptionalAttr<TesseraTile_ThreadMapType>:$thread_map,
    OptionalAttr<StrAttr>:$write_policy
  );
  
  let assemblyFormat = "$value `,` $memref `[` $indices `]` "
                       "(`thread_map` `=` $thread_map^)? "
                       "(`policy` `=` $write_policy^)? "
                       "attr-dict `:` type($value) `,` type($memref)";
}

def TesseraTile_CopyOp : TesseraTile_Op<"copy", [
    MemoryEffects<[MemRead<0>, MemWrite<1>]>
]> {
  let summary = "Cooperative memory copy between hierarchy levels";
  let description = [{
    Performs cooperative memory copy between different memory hierarchy levels
    with optimal access patterns and vectorization.
    
    Example:
    ```mlir
    tessera_tile.copy %global_mem to %shared_mem
                     thread_map = #tessera_tile.thread_map<[32, 4], [32, 1]>
                     vectorize = 4
                     : memref<128x64xf16> to memref<128x64xf16, #shared>
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$source,
    AnyMemRef:$dest,
    OptionalAttr<TesseraTile_ThreadMapType>:$thread_map,
    OptionalAttr<I64Attr>:$vectorize,
    OptionalAttr<BoolAttr>:$async
  );
  
  let assemblyFormat = "$source `to` $dest "
                       "(`thread_map` `=` $thread_map^)? "
                       "(`vectorize` `=` $vectorize^)? "
                       "(`async` $async^)? "
                       "attr-dict `:` type($source) `to` type($dest)";
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Compute Operations
//===----------------------------------------------------------------------===//

def TesseraTile_GemmOp : TesseraTile_Op<"gemm", [Pure]> {
  let summary = "Tile-level GEMM with hardware optimization";
  let description = [{
    Performs General Matrix Multiply at the tile level with support for
    various hardware acceleration (Tensor Cores, WMMA, MMA).
    
    Example:
    ```mlir
    %C = tessera_tile.gemm %A, %B, %C_init 
         layout = "tn" 
         precision = "tf32"
         threads = #tessera_tile.thread_map<[16, 8], [32, 1]>
         : (memref<128x64xf16>, memref<64x128xf16>, memref<128x128xf32>) -> memref<128x128xf32>
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$lhs,
    AnyMemRef:$rhs, 
    AnyMemRef:$acc,
    OptionalAttr<StrAttr>:$layout,
    OptionalAttr<StrAttr>:$precision,
    OptionalAttr<TesseraTile_ThreadMapType>:$threads,
    OptionalAttr<BoolAttr>:$use_tensor_cores
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$lhs `,` $rhs `,` $acc "
                       "(`layout` `=` $layout^)? "
                       "(`precision` `=` $precision^)? "
                       "(`threads` `=` $threads^)? "
                       "(`tensor_cores` $use_tensor_cores^)? "
                       "attr-dict `:` functional-type(operands, results)";
  
  let hasVerifier = 1;
}

def TesseraTile_ReduceOp : TesseraTile_Op<"reduce", [Pure]> {
  let summary = "Tile-level reduction with cooperative threads";
  let description = [{
    Performs reduction across threads/warps with support for various
    reduction operations (sum, max, min) and data types.
    
    Example:
    ```mlir
    %sum = tessera_tile.reduce %input kind = "sum" 
           axis = [1] 
           threads = #tessera_tile.thread_map<[32], [32]>
           : memref<128x64xf32> -> memref<128xf32>
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$input,
    StrAttr:$kind,
    I64ArrayAttr:$axis,
    OptionalAttr<TesseraTile_ThreadMapType>:$threads,
    OptionalAttr<BoolAttr>:$keep_dims
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$input `kind` `=` $kind "
                       "`axis` `=` $axis "
                       "(`threads` `=` $threads^)? "
                       "(`keep_dims` $keep_dims^)? "
                       "attr-dict `:` type($input) `->` type($result)";
}

def TesseraTile_ElementwiseOp : TesseraTile_Op<"elementwise", [Pure]> {
  let summary = "Tile-level elementwise operations";
  let description = [{
    Performs elementwise operations with thread mapping and vectorization.
    Supports fusion with other elementwise operations.
    
    Example:
    ```mlir
    %result = tessera_tile.elementwise "add" %lhs, %rhs 
              threads = #tessera_tile.thread_map<[32, 4], [32, 1]>
              vectorize = 2
              : (memref<128x64xf32>, memref<128x64xf32>) -> memref<128x64xf32>
    ```
  }];
  
  let arguments = (ins 
    StrAttr:$op,
    Variadic<AnyMemRef>:$inputs,
    OptionalAttr<TesseraTile_ThreadMapType>:$threads,
    OptionalAttr<I64Attr>:$vectorize
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$op $inputs "
                       "(`threads` `=` $threads^)? "
                       "(`vectorize` `=` $vectorize^)? "
                       "attr-dict `:` functional-type($inputs, $result)";
}

//===----------------------------------------------------------------------===//
// Fragment Operations (Register-level)
//===----------------------------------------------------------------------===//

def TesseraTile_LoadFragmentOp : TesseraTile_Op<"load_fragment", [Pure]> {
  let summary = "Load data into register fragment";
  let description = [{
    Loads data from memory into a register fragment for fine-grained operations.
    Used for register-level tiling and Tensor Core operations.
    
    Example:
    ```mlir
    %frag = tessera_tile.load_fragment %memref[%offset] 
            layout = "row_major"
            : memref<64x64xf16> -> !tessera_tile.fragment<f16, [16, 16], "wmma_matrix_a">
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$memref,
    Variadic<Index>:$offset,
    OptionalAttr<StrAttr>:$layout
  );
  
  let results = (outs TesseraTile_FragmentType:$fragment);
  
  let assemblyFormat = "$memref `[` $offset `]` "
                       "(`layout` `=` $layout^)? "
                       "attr-dict `:` type($memref) `->` type($fragment)";
}

def TesseraTile_StoreFragmentOp : TesseraTile_Op<"store_fragment"> {
  let summary = "Store register fragment to memory";
  let description = [{
    Stores a register fragment back to memory with specified layout.
  }];
  
  let arguments = (ins 
    TesseraTile_FragmentType:$fragment,
    AnyMemRef:$memref,
    Variadic<Index>:$offset,
    OptionalAttr<StrAttr>:$layout
  );
  
  let assemblyFormat = "$fragment `,` $memref `[` $offset `]` "
                       "(`layout` `=` $layout^)? "
                       "attr-dict `:` type($fragment) `,` type($memref)";
}

def TesseraTile_MmaOp : TesseraTile_Op<"mma", [Pure]> {
  let summary = "Matrix-multiply-accumulate on fragments";
  let description = [{
    Performs matrix-multiply-accumulate on register fragments using
    hardware acceleration (Tensor Cores, WMMA).
    
    Example:
    ```mlir
    %c_frag = tessera_tile.mma %a_frag, %b_frag, %c_frag_init
              : (!tessera_tile.fragment<f16, [16, 16], "wmma_matrix_a">,
                 !tessera_tile.fragment<f16, [16, 16], "wmma_matrix_b">,
                 !tessera_tile.fragment<f32, [16, 16], "wmma_accumulator">)
              -> !tessera_tile.fragment<f32, [16, 16], "wmma_accumulator">
    ```
  }];
  
  let arguments = (ins 
    TesseraTile_FragmentType:$lhs,
    TesseraTile_FragmentType:$rhs,
    TesseraTile_FragmentType:$acc
  );
  
  let results = (outs TesseraTile_FragmentType:$result);
  
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` functional-type(operands, results)";
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def TesseraTile_BarrierOp : TesseraTile_Op<"barrier"> {
  let summary = "Thread synchronization barrier";
  let description = [{
    Synchronizes threads at the specified level (thread, warp, block).
    
    Example:
    ```mlir
    tessera_tile.barrier "block"
    tessera_tile.barrier "warp" mask = %active_mask
    ```
  }];
  
  let arguments = (ins 
    StrAttr:$level,
    Optional<AnyType>:$mask
  );
  
  let assemblyFormat = "$level (`mask` `=` $mask^)? attr-dict (`:` type($mask)^)?";
}

def TesseraTile_MemFenceOp : TesseraTile_Op<"mem_fence"> {
  let summary = "Memory fence operation";
  let description = [{
    Issues memory fence for the specified memory level and scope.
  }];
  
  let arguments = (ins 
    StrAttr:$memory_level,
    OptionalAttr<StrAttr>:$scope
  );
  
  let assemblyFormat = "$memory_level (`scope` `=` $scope^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def TesseraTile_IfOp : TesseraTile_Op<"if", [
    SingleBlockImplicitTerminator<"tessera_tile::YieldOp">,
    RecursiveMemoryEffects,
    NoRegionArguments
]> {
  let summary = "Thread-divergent conditional execution";
  let description = [{
    Conditional execution with explicit handling of thread divergence.
    
    Example:
    ```mlir
    tessera_tile.if %condition divergent = true {
      // Executed by threads where condition is true
      tessera_tile.store %value, %memref[%idx]
    }
    ```
  }];
  
  let arguments = (ins 
    I1:$condition,
    OptionalAttr<BoolAttr>:$divergent
  );
  
  let regions = (region SizedRegion<1>:$thenRegion, 
                        OptionalRegion:$elseRegion);
  
  let assemblyFormat = "$condition "
                       "(`divergent` `=` $divergent^)? "
                       "$thenRegion "
                       "(`else` $elseRegion^)? "
                       "attr-dict";
}

def TesseraTile_ForOp : TesseraTile_Op<"for", [
    DeclareOpInterfaceMethods<LoopLikeOpInterface>,
    SingleBlockImplicitTerminator<"tessera_tile::YieldOp">,
    RecursiveMemoryEffects
]> {
  let summary = "Thread-mapped loop operation";
  let description = [{
    Loop with explicit thread mapping and iteration distribution.
    
    Example:
    ```mlir
    tessera_tile.for %i = %lb to %ub step %step 
                     thread_map = #tessera_tile.thread_map<[32], [32]> {
      %val = tessera_tile.load %memref[%i]
      %result = arith.addf %val, %const
      tessera_tile.store %result, %out_memref[%i]
    }
    ```
  }];
  
  let arguments = (ins 
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    OptionalAttr<TesseraTile_ThreadMapType>:$thread_map,
    OptionalAttr<BoolAttr>:$parallel
  );
  
  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = "$lowerBound `to` $upperBound `step` $step "
                       "(`thread_map` `=` $thread_map^)? "
                       "(`parallel` $parallel^)? "
                       "$region attr-dict";
  
  let hasVerifier = 1;
}

def TesseraTile_YieldOp : TesseraTile_Op<"yield", [
    Pure, ReturnLike, Terminator
]> {
  let summary = "Yield operation for tile regions";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Pipeline Operations
//===----------------------------------------------------------------------===//

def TesseraTile_PipelineOp : TesseraTile_Op<"pipeline", [
    SingleBlockImplicitTerminator<"tessera_tile::YieldOp">
]> {
  let summary = "Software pipeline with stages";
  let description = [{
    Creates a software pipeline with multiple stages for overlapping
    computation and memory operations.
    
    Example:
    ```mlir
    tessera_tile.pipeline num_stages = 3 {
    ^stage0(%stage: index):
      %data = tessera_tile.load %input[%stage]
      tessera_tile.pipeline_stage %data : tensor<64xf16>
    ^stage1(%data: tensor<64xf16>):
      %result = tessera_tile.compute %data
      tessera_tile.pipeline_stage %result : tensor<64xf16>  
    ^stage2(%result: tensor<64xf16>):
      tessera_tile.store %result, %output[%stage]
    }
    ```
  }];
  
  let arguments = (ins 
    I64Attr:$num_stages,
    OptionalAttr<BoolAttr>:$async
  );
  
  let regions = (region VariadicRegion<AnyRegion>:$stages);
  
  let assemblyFormat = "`num_stages` `=` $num_stages "
                       "(`async` $async^)? "
                       "$stages attr-dict";
  
  let hasVerifier = 1;
}

def TesseraTile_PipelineStageOp : TesseraTile_Op<"pipeline_stage", [Terminator]> {
  let summary = "Pass values to next pipeline stage";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// Hardware-Specific Operations
//===----------------------------------------------------------------------===//

def TesseraTile_TMEMLoadOp : TesseraTile_Op<"tmem_load", [Pure]> {
  let summary = "Load from Tensor Memory (TMEM) on Blackwell";
  let description = [{
    Loads data from TMEM with specified access patterns and banking.
    Only available on NVIDIA Blackwell architecture.
    
    Example:
    ```mlir
    %data = tessera_tile.tmem_load %tmem_ptr bank = 0 
            pattern = "sequential"
            : !tessera_tile.tmem_ptr<f16> -> memref<128x64xf16>
    ```
  }];
  
  let arguments = (ins 
    AnyType:$tmem_ptr,
    OptionalAttr<I64Attr>:$bank,
    OptionalAttr<StrAttr>:$pattern
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$tmem_ptr "
                       "(`bank` `=` $bank^)? "
                       "(`pattern` `=` $pattern^)? "
                       "attr-dict `:` type($tmem_ptr) `->` type($result)";
}

def TesseraTile_TMEMStoreOp : TesseraTile_Op<"tmem_store"> {
  let summary = "Store to Tensor Memory (TMEM) on Blackwell";
  let description = [{
    Stores data to TMEM with specified banking and write patterns.
  }];
  
  let arguments = (ins 
    AnyMemRef:$data,
    AnyType:$tmem_ptr,
    OptionalAttr<I64Attr>:$bank,
    OptionalAttr<StrAttr>:$pattern
  );
  
  let assemblyFormat = "$data `,` $tmem_ptr "
                       "(`bank` `=` $bank^)? "
                       "(`pattern` `=` $pattern^)? "
                       "attr-dict `:` type($data) `,` type($tmem_ptr)";
}

def TesseraTile_CTAPairOp : TesseraTile_Op<"cta_pair", [
    SingleBlockImplicitTerminator<"tessera_tile::YieldOp">
]> {
  let summary = "CTA pair cooperative operation for Blackwell";
  let description = [{
    Coordinates computation between paired CTAs (Cooperative Thread Arrays)
    for large-scale operations on Blackwell architecture.
    
    Example:
    ```mlir
    tessera_tile.cta_pair role = "primary" {
      %shared_data = tessera_tile.alloc() : memref<256x128xf16, #shared>
      tessera_tile.cooperative_load %global_data to %shared_data
      // Primary CTA operations
    }
    ```
  }];
  
  let arguments = (ins 
    StrAttr:$role,
    OptionalAttr<I64Attr>:$pair_id
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = "`role` `=` $role "
                       "(`pair_id` `=` $pair_id^)? "
                       "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// Layout and Access Pattern Operations  
//===----------------------------------------------------------------------===//

def TesseraTile_LayoutCastOp : TesseraTile_Op<"layout_cast", [
    Pure,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
]> {
  let summary = "Change memory layout without data movement";
  let description = [{
    Changes the logical layout of data without physical data movement.
    Used for optimizing access patterns and enabling vectorization.
    
    Example:
    ```mlir
    %col_major = tessera_tile.layout_cast %row_major 
                 layout = "column_major"
                 : memref<128x64xf32, #row_major> to memref<128x64xf32, #col_major>
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$source,
    StrAttr:$layout
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$source `layout` `=` $layout "
                       "attr-dict `:` type($source) `to` type($result)";
}

def TesseraTile_SwizzleOp : TesseraTile_Op<"swizzle", [Pure]> {
  let summary = "Apply swizzle pattern to avoid bank conflicts";
  let description = [{
    Applies swizzle patterns to memory accesses to avoid bank conflicts
    in shared memory and maximize memory bandwidth.
    
    Example:
    ```mlir
    %swizzled = tessera_tile.swizzle %data 
                pattern = "xor_8"
                banks = 32
                : memref<128x64xf16, #shared> -> memref<128x64xf16, #shared_swizzled>
    ```
  }];
  
  let arguments = (ins 
    AnyMemRef:$input,
    StrAttr:$pattern,
    OptionalAttr<I64Attr>:$banks
  );
  
  let results = (outs AnyMemRef:$result);
  
  let assemblyFormat = "$input `pattern` `=` $pattern "
                       "(`banks` `=` $banks^)? "
                       "attr-dict `:` type($input) `->` type($result)";
}

#endif // TESSERA_TILE_OPS