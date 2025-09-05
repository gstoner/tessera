include "mlir/IR/OpBase.td"
def Tessera_Collective_Dialect : Dialect { let name = "tessera.collective"; let cppNamespace = "tessera::collective"; }
def Tessera_FutureType : TypeDef<Tessera_Collective_Dialect, "Future"> { let parameters = (ins Type:$valueType); let assemblyFormat = "`<` type($valueType) `>`"; }
