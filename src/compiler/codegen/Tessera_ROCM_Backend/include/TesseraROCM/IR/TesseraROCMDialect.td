include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

def TesseraROCMDialect : Dialect {
  let name = "tessera_rocm";
  let cppNamespace = "mlir::tessera_rocm";
  let useDefaultTypePrinterParser = 1;
}

class ROCMStringAttr<string summaryText>
    : StringBasedAttr<StrAttrPred, summaryText>;

def MemSpaceAttr : ROCMStringAttr<"tessera_rocm.memspace">;
def TokenType : TypeDef<TesseraROCMDialect, "Token"> {
  let mnemonic = "token";
}
def ROCM_KernelAttr : ROCMStringAttr<"tessera_rocm.kernel">;
