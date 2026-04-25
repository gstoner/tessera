//===- tessera_neighbors.td - Tessera Neighbors Dialect -------*- tablegen -*-===//
// This is a minimal ODS surface for directions, Δ, stencil, and pipeline ops.
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Neighbors_Dialect : Dialect {
  let name = "tessera.neighbors";
  let cppNamespace = "tessera::neighbors";
}

def TopologyType : TypeDef<Tessera_Neighbors_Dialect, "Topology"> {
  let mnemonic = "topology";
  let summary = "Handle for a neighbor topology (mesh/torus/hex/graph)";
}

def HaloType : TypeDef<Tessera_Neighbors_Dialect, "Halo"> {
  let mnemonic = "halo";
  let summary = "Halo view derived from a tile with width/axes/time";
}

def DeltaArrayAttr : AttrDef<Tessera_Neighbors_Dialect, "DeltaArray"> {
  let mnemonic = "delta_array";
  let summary = "Array of integer deltas per axis (Δ)";
  let parameters = (ins ArrayRefParameter<"int64_t">:$values);
  let assemblyFormat = "`<` $values `>`";
}

def StrAttr : AttrDef<Tessera_Neighbors_Dialect, "Str"> {
  let mnemonic = "str";
  let summary = "String attribute (utility)";
  let parameters = (ins StringRefParameter<"">:$value);
  let assemblyFormat = "`<` $value `>`";
}

// --- Ops ---

def CreateTopologyOp : Op<Tessera_Neighbors_Dialect, "topology.create", []> {
  let summary = "Create a topology object";
  let arguments = (ins
    StrAttr:$kind,               // e.g., "2d_mesh","3d_mesh","hex_2d","custom_graph"
    OptionalAttr<StrAttr>:$axes, // e.g., "y,x" or "z,y,x"
    OptionalAttr<StrAttr>:$wrap, // "none","all","[y]"
    OptionalAttr<StrAttr>:$defaults // default neighbor set
  );
  let results = (outs TopologyType:$topo);
  let assemblyFormat = "`kind` `(` $kind `)` (`,` `axes` `(` $axes `)`)? (`,` `wrap` `(` $wrap `)`)? (`,` `defaults` `(` $defaults `)`)? attr-dict";
}

def HaloRegionOp : Op<Tessera_Neighbors_Dialect, "halo.region", []> {
  let summary = "Create a halo region view (width/axes/time)";
  let arguments = (ins AnyType:$tile,
                      OptionalAttr<DeltaArrayAttr>:$width,
                      OptionalAttr<StrAttr>:$axes,
                      OptionalAttr<I64Attr>:$time,
                      OptionalAttr<StrAttr>:$bc,
                      OptionalAttr<StrAttr>:$pack,
                      OptionalAttr<StrAttr>:$space);
  let results = (outs HaloType:$halo);
  let assemblyFormat = "$tile (`width` `=` $width)? (`axes` `=` $axes)? (`time` `=` $time)? (`bc` `=` $bc)? (`pack` `=` $pack)? (`space` `=` $space)? attr-dict";
}

def HaloExchangeOp : Op<Tessera_Neighbors_Dialect, "halo.exchange", [MemoryEffects<[MemWrite]>]> {
  let summary = "Initiate halo exchange for a region (async by default)";
  let arguments = (ins HaloType:$halo,
                      OptionalAttr<StrAttr>:$neighbors,
                      OptionalAttr<I64Attr>:$priority,
                      OptionalAttr<StrAttr>:$policy);
  let results = (outs AnyType:$token); // model async.token without dependency
  let assemblyFormat = "$halo (`neighbors` `=` $neighbors)? (`priority` `=` $priority)? (`policy` `=` $policy)? attr-dict";
}

def NeighborReadOp : Op<Tessera_Neighbors_Dialect, "neighbor.read", []> {
  let summary = "Read from a logical neighbor by relative Δ";
  let arguments = (ins AnyType:$tile, TopologyType:$topo, DeltaArrayAttr:$delta);
  let results = (outs AnyType:$slice);
  let assemblyFormat = "$tile `,` $topo `,` $delta attr-dict";
}

def StencilDefineOp : Op<Tessera_Neighbors_Dialect, "stencil.define", []> {
  let summary = "Define a stencil taps/coeffs object with BC";
  let arguments = (ins ArrayAttr:$taps, OptionalAttr<StrAttr>:$bc);
  let results = (outs AnyType:$stencil);
  let assemblyFormat = "`taps` `=` $taps (`,` `bc` `=` $bc)? attr-dict";
}

def StencilApplyOp : Op<Tessera_Neighbors_Dialect, "stencil.apply", []> {
  let summary = "Apply stencil to field(s), auto-infer halos, emit exchanges";
  let arguments = (ins AnyType:$stencil, AnyType:$inField, TopologyType:$topo);
  let results = (outs AnyType:$outField);
  let assemblyFormat = "$stencil `,` $inField `,` $topo attr-dict";
}

def PipelineConfigOp : Op<Tessera_Neighbors_Dialect, "pipeline.config", []> {
  let summary = "Configure pipeline overlap/double-buffer/reuse/order";
  let arguments = (ins OptionalAttr<I64Attr>:$stages,
                      OptionalAttr<BoolAttr>:$double_buffer,
                      OptionalAttr<StrAttr>:$order,
                      OptionalAttr<StrAttr>:$overlap,
                      OptionalAttr<StrAttr>:$reuse);
  let assemblyFormat = "attr-dict";
}
