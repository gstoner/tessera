// RUN: tessera-opt --tessera-tile-buffer-arena --allow-unregistered-dialect -split-input-file -verify-diagnostics %s
func.func private @escape(%a: memref<16xf32, 3>) -> memref<16xf32, 3> {
  return %a : memref<16xf32, 3>
}
func.func @returned_alias(%a: memref<16xf32, 3>) {
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32, 3>) -> ()
  %v = call @escape(%a) : (memref<16xf32, 3>) -> memref<16xf32, 3>
  return
}
// -----
func.func private @recursive(%a: memref<16xf32, 3>) {
  call @recursive(%a) : (memref<16xf32, 3>) -> ()
  return
}
func.func @recursion_unproven(%a: memref<16xf32, 3>) {
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32, 3>) -> ()
  call @recursive(%a) : (memref<16xf32, 3>) -> ()
  return
}
// -----
func.func private @host_space(%a: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %v = memref.load %a[%c0] : memref<16xf32>
  return
}
func.func @address_space_change_unproven(%a: memref<16xf32>) {
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32>) -> ()
  call @host_space(%a) : (memref<16xf32>) -> ()
  return
}
// -----
func.func private @external(%a: memref<16xf32, 3>)
func.func @external_ownership_unproven(%a: memref<16xf32, 3>) {
  // expected-error @+1 {{arena cannot rebase a pre-existing, escaping or nonidentity alias}}
  "tile.alloc_shared"(%a) {tile.buffer_group = 0 : i64} : (memref<16xf32, 3>) -> ()
  call @external(%a) : (memref<16xf32, 3>) -> ()
  return
}
