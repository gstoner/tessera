// RUN: tessera-opt --tessera-tile-buffer-arena --allow-unregistered-dialect -verify-diagnostics %s
func.func @forged_async_reuse(%a: memref<16xf32>, %b: memref<16xf32>) {
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32>) -> ()
  "tile.async_copy"(%a) : (memref<16xf32>) -> ()
  // expected-error @+1 {{TILE_BARRIER_REUSE_MISSING_BARRIER: arena reuse group lacks a disjoint lifetime proof}}
  "tile.alloc_shared"(%b) {tile.buffer_group = 0 : i64} : (memref<16xf32>) -> ()
  return
}

// A pre-existing view cannot be silently left pointing at the old storage.
func.func @preexisting_alias(%a: memref<16xf32>) {
  %v = memref.cast %a : memref<16xf32> to memref<?xf32>
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32>) -> ()
  return
}

// Opaque operations may retain or return an alias of their memref operand.
func.func @opaque_alias(%a: memref<16xf32>) {
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32>) -> ()
  %alias = "foreign.alias"(%a) : (memref<16xf32>) -> memref<16xf32>
  return
}
