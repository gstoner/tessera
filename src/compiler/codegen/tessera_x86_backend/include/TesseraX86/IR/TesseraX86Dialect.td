//===- TesseraX86Dialect.td -----------------------------------*- tablegen -*-===//
//
// The hardware-free Target IR dialect for the x86 backend (Decision #19).
//
// Until W0.10 (2026-08-02) x86 was the ONE backend with no Target IR dialect at
// all, while Decision #19 states that backends MUST expose one. `TileToX86Pass`
// lowered Tile IR straight to 21 `func::CallOp`s into a hand-written C shim,
// using neither a Tessera target dialect nor MLIR's upstream `amx`/`x86vector`
// dialects -- structurally the same `func.call`-to-a-C-symbol shape CLAUDE.md
// flags for Apple GPU. The Python emitter even named a `tessera_x86.func` op
// that did not exist.
//
// The decision recorded there was to BUILD this dialect rather than grant a
// carve-out, so the rule stays universal and x86 becomes lit-testable at the
// target level like ROCm / NVIDIA / Apple.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_X86_DIALECT_TD
#define TESSERA_X86_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TesseraX86Dialect : Dialect {
  let name = "tessera_x86";
  let cppNamespace = "::mlir::tessera_x86";
  let summary =
    "Hardware-free Target IR for the x86 backend (AMX tiles, AVX-512 vectors, "
    "GEMM packing, and the C-ABI shim boundary).";
  let description = [{
    The abstract layer between Tile IR and final x86 emission. Ops here name
    *what* the target must do -- load an AMX tile, issue a BF16 tile dot-product,
    run an AVX-512 GEMM microkernel -- without committing to an instruction
    encoding, so they can be checked by lit fixtures on any host, including
    hosts with no AMX (no machine in the current fleet has AMX; Zen 5 has
    AVX-512 only, and AMX is Intel-only).
  }];
  let useDefaultTypePrinterParser = 1;
}

class TesseraX86_Op<string mnemonic, list<Trait> traits = []>
  : Op<TesseraX86Dialect, mnemonic, traits>;

// An AMX tile register value. AMX operates on a small file of 2-D tile
// registers loaded from strided memory; modelling them as an SSA type is what
// lets the verifier reject a dot-product whose operands never came from a tile
// load. Mirrors the role `!tessera_rocm.token` plays for async copies.
def X86_TileType : TypeDef<TesseraX86Dialect, "Tile"> {
  let mnemonic = "tile";
  let summary = "An AMX tile register value";
}

#endif  // TESSERA_X86_DIALECT_TD
