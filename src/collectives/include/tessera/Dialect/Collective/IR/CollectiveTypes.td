include "mlir/IR/OpBase.td"

// MLIR operation names split the dialect namespace at the first dot.  The old
// `tessera.collective` spelling therefore registered a dialect that could not
// own operations spelled `tessera.collective.*`.  Keep the C++ namespace while
// giving the production Target IR a real, independently registered namespace.
def Tessera_Collective_Dialect : Dialect {
  let name = "tessera_collective";
  let cppNamespace = "tessera::collective";
  let summary = "Target-neutral asynchronous collective transport";
  let useDefaultTypePrinterParser = 1;
}
def Tessera_FutureType : TypeDef<Tessera_Collective_Dialect, "Future"> { let mnemonic = "future"; let parameters = (ins "::mlir::Type":$valueType); let assemblyFormat = "`<` $valueType `>`"; }
def Tessera_ShardType : TypeDef<Tessera_Collective_Dialect, "Shard"> { let mnemonic = "shard"; let parameters = (ins "::mlir::Type":$valueType); let assemblyFormat = "`<` $valueType `>`"; }
def Tessera_WindowType : TypeDef<Tessera_Collective_Dialect, "Window"> {
  let mnemonic = "window";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}
