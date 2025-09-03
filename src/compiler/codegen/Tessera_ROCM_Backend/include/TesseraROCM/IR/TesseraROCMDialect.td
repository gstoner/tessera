include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def TesseraROCMDialect : Dialect {
  let name = "tessera_rocm";
  let cppNamespace = "mlir::tessera_rocm";
}

def MemSpaceAttr : StrAttrBase<"MemSpace", "tessera_rocm.memspace"> {}
def TokenType : TypeDef<TesseraROCMDialect, "Token"> {}
def ROCM_KernelAttr : StrAttrBase<"Kernel", "tessera_rocm.kernel"> {}
