//===- TPPTypes.td ---------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "TPP.td"

class OptionalStringRefParameter<string desc = ""> :
    OptionalParameter<"::llvm::StringRef", desc> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let printer = [{$_printer.printString($_self);}];
  let cppStorageType = "std::string";
}

def TPP_FieldType : TypeDef<TPP_Dialect, "Field"> {
  let mnemonic = "field";
  let summary = "Space–time field over a mesh/grid with layout/time metadata";
  let parameters = (ins
    TypeParameter<"::mlir::Type", "Element type">:$elemTy,
    StringRefParameter<"Layout string">:$layout,
    StringRefParameter<"Space string (e.g., grid(latlon,...))">:$space,
    OptionalStringRefParameter<"Time string (e.g., time(dt=...))">:$time
  );
  let assemblyFormat = "`<` $elemTy `,` $layout `,` $space (`,` $time^)? `>`";
}

def TPP_MeshType : TypeDef<TPP_Dialect, "Mesh"> {
  let mnemonic = "mesh";
  let summary = "Mesh/topology description";
  let parameters = (ins
    StringRefParameter<"Topology">:$topology,
    OptionalStringRefParameter<"Coords">:$coords,
    OptionalParameter<"::mlir::IntegerAttr", "Levels">:$levels
  );
  let assemblyFormat = "`<` $topology (`,` $coords^)? (`,` $levels^)? `>`";
}
