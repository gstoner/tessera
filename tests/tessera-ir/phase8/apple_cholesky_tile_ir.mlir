// L-series linalg pilot — L3: tessera.cholesky through Schedule→Tile.
//
// TilingPass tiles tessera.matmul into scf.for nests, but cholesky is a
// sequential blocked factorization, so it lowers 1:1 to the opaque Tile-IR op
// `tile.cholesky` (carrying the `lower` attribute).  This fixture proves the
// full Graph → Schedule → Tile traversal lands a distinct Tile-layer op.
//
// Full spine (distribution-lowering → tiling):
// RUN: tessera-opt --tessera-distribution-lowering='mesh-axes=dp mesh-sizes=1' \
// RUN:   --tessera-tiling --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=SPINE
//
// Tiling alone on a bare (single-device) cholesky:
// RUN: tessera-opt --tessera-tiling --allow-unregistered-dialect %s \
// RUN:   | FileCheck %s --check-prefix=BARE

// SPINE-LABEL: func.func @chol_step
// SPINE:       schedule.mesh.define
// SPINE:       %[[R:.*]] = "schedule.mesh.region"
// SPINE:         "tile.cholesky"
// SPINE-SAME:    lower = true
// SPINE-SAME:    source = "tessera.cholesky"
// SPINE:         schedule.yield
// SPINE:       return %[[R]]

// BARE-LABEL:  func.func @chol_step
// BARE:        "tile.cholesky"
// BARE-SAME:   lower = true
// BARE-SAME:   source = "tessera.cholesky"
// The Graph-IR op spelling (`= tessera.cholesky %arg`) must be gone — only the
// `source` *attribute* string remains, which the BARE-SAME above matches.
// BARE-NOT:    = tessera.cholesky %
func.func @chol_step(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tessera.cholesky %a : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
