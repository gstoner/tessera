//===- TPPAttrs.td ---------------------------------------*- tablegen -*-===//
include "mlir/IR/AttrTypeBase.td"
include "TPP.td"

class TPP_StringAttr<string name, string mnemonicStr, string summaryText>
    : AttrDef<TPP_Dialect, name> {
  let mnemonic = mnemonicStr;
  let summary = summaryText;
  let parameters = (ins StringRefParameter<summaryText>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def TPP_UnitsAttr : TPP_StringAttr<"Units", "units", "unit expression">;
def TPP_BCAttr    : TPP_StringAttr<"BC", "bc", "boundary condition">;
