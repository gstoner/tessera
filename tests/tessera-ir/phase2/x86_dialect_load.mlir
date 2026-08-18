// RUN: %tessera_strict_opt %s | FileCheck %s
//
// Direct load-time regression for X86-DIALECT-LOAD-CRASH-2026-08-12.  Keep
// this fixture smaller than the positive operation fixture: merely parsing a
// function signature containing !tessera_x86.tile must initialize the dialect
// and register TileType without crashing.

module attributes {tessera.ir.level = "target", tessera.target = "x86"} {
  // CHECK-LABEL: func.func @load_x86_tile_type
  // CHECK-SAME: !tessera_x86.tile
  func.func @load_x86_tile_type(%tile: !tessera_x86.tile) {
    return
  }
}
// REQUIRES: tessera-x86-target-ir
