include "mlir/IR/OpBase.td"
def Tessera_Collective_Dialect : Dialect { let name = "tessera.collective"; let cppNamespace = "tessera::collective"; }
def Tessera_FutureType : TypeDef<Tessera_Collective_Dialect, "Future"> { let mnemonic = "future"; let parameters = (ins "::mlir::Type":$valueType); let assemblyFormat = "`<` $valueType `>`"; }
def Tessera_ShardType : TypeDef<Tessera_Collective_Dialect, "Shard"> { let mnemonic = "shard"; let parameters = (ins "::mlir::Type":$valueType); let assemblyFormat = "`<` $valueType `>`"; }
