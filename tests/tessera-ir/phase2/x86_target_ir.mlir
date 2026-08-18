// RUN: tessera-opt %s | FileCheck %s
//
// W0.10 — the hardware-free x86 Target IR dialect (Decision #19).
//
// x86 was the ONE backend with no Target IR dialect, though Decision #19 says
// backends MUST expose one: `TileToX86Pass` lowered Tile IR straight to 21
// `func::CallOp`s into a hand-written C shim, and the Python emitter named a
// `tessera_x86.func` op that did not exist in any dialect. The decision was to
// build the dialect rather than grant a carve-out.
//
// This fixture is what that buys: the x86 target layer is now checkable on any
// host, with no AMX silicon required — which matters, because **no machine in
// the current fleet has AMX** (AMX is Intel-only; Zen 5 has AVX-512 only).

module attributes {tessera.ir.level = "target", tessera.target = "x86"} {

  // CHECK-LABEL: func.func @amx_bf16_gemm_tile
  func.func @amx_bf16_gemm_tile(%a: memref<16x32xbf16>,
                                %b: memref<16x32xbf16>,
                                %c: memref<16x16xf32>) {
    %stride = arith.constant 64 : i64

    // An accumulator chain starts from a DEFINED value, not an undef the
    // verifier cannot tell from a bug.
    // CHECK: tessera_x86.amx_tile_zero
    %acc = tessera_x86.amx_tile_zero : !tessera_x86.tile

    // CHECK: tessera_x86.amx_tile_load
    %ta = tessera_x86.amx_tile_load %a, %stride
        : memref<16x32xbf16>, i64 -> !tessera_x86.tile
    // CHECK: tessera_x86.amx_tile_load
    %tb = tessera_x86.amx_tile_load %b, %stride
        : memref<16x32xbf16>, i64 -> !tessera_x86.tile

    // The point of the typed layer: `amx_dpbf16ps` can only consume values that
    // genuinely came from a tile load or zero. A substring test cannot check
    // this; the verifier can.
    // CHECK: tessera_x86.amx_dpbf16ps
    %res = tessera_x86.amx_dpbf16ps %ta, %tb, %acc
        : !tessera_x86.tile, !tessera_x86.tile, !tessera_x86.tile
        -> !tessera_x86.tile

    // CHECK: tessera_x86.amx_tile_store
    tessera_x86.amx_tile_store %res, %c, %stride
        : !tessera_x86.tile, memref<16x16xf32>, i64
    return
  }

  // The AVX-512 lane — the one that actually executes on this fleet.
  // CHECK-LABEL: func.func @avx512_gemm
  func.func @avx512_gemm() {
    // CHECK: tessera_x86.pack_b_panel
    // CHECK-SAME: kc = 256
    "tessera_x86.pack_b_panel"() {name = "pack_b", kc = 256 : i64, nc = 64 : i64,
                                 dtype = "f32"} : () -> ()
    // CHECK: tessera_x86.avx512_gemm_microkernel
    // CHECK-SAME: m = 6
    "tessera_x86.avx512_gemm_microkernel"() {name = "ukernel", m = 6 : i64,
                                             n = 16 : i64, k = 256 : i64,
                                             dtype = "f32"} : () -> ()
    return
  }

  // The shim boundary is MODELLED, not hidden. As a bare `func.call` it was
  // invisible to any x86-specific analysis — which is why x86 looked like it
  // needed no Target IR at all.
  // CHECK-LABEL: func.func @shim_boundary
  func.func @shim_boundary() {
    // CHECK: tessera_x86.abi_call
    // CHECK-SAME: symbol = "tessera_x86_sgemm"
    "tessera_x86.abi_call"() {symbol = "tessera_x86_sgemm",
                              abi = "tessera_x86_c"} : () -> ()
    // CHECK: tessera_x86.elementwise
    "tessera_x86.elementwise"() {source = "tessera.gelu",
                                 isa = "avx512"} : () -> ()
    return
  }
}
// REQUIRES: tessera-x86-target-ir
