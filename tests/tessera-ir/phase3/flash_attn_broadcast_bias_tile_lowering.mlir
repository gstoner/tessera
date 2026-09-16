// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=17 tile-kv=16 sm=90' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s
//
// FRONTEND-IR-MEDIUM-1 (2026-09-15): an additive attention mask may broadcast
// on the query or key axis, not only batch/head. The FA-4 Tile lowering keeps
// the PHYSICAL bias block -- a key-padding row is [1, tkv], a per-query column
// [tq, 1] -- and `tessera_attn.score_bias` verifies a bias axis as 1 or the
// scores extent. Nothing is expanded: the padded copy and every per-block
// slice stay at the storage that exists.

// A key-padding mask: one row shared by every batch, head and query. It is
// padded on the key axis to the tile multiple (19 -> 32) as a 1-row tensor.
// CHECK-LABEL: func.func @key_padding
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [1, 19] [1, 1] : tensor<1x19xf32> into tensor<1x32xf32>
// CHECK: tessera_attn.score_bias %{{.*}}, %{{.*}} : tensor<17x16xf32>, tensor<1x16xf32> -> tensor<17x16xf32>
func.func @key_padding(%q: tensor<1x4x17x64xf32>, %key: tensor<1x2x19x64xf32>, %v: tensor<1x2x19x64xf32>, %bias: tensor<1x1x1x19xf32>) -> tensor<1x4x17x64xf32> {
  %o = "tessera.flash_attn"(%q, %key, %v, %bias) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> {causal = true, dropout_p = 0.0 : f64, dropout_seed = 0 : i64, head_dim = 64 : i64, scale = 0.125 : f32, softcap = 0.0 : f32, tessera.tile_q = 17 : i32, tessera.tile_kv = 16 : i32, window_left = -1 : i64, window_right = -1 : i64} : (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>, tensor<1x2x19x64xf32>, tensor<1x1x1x19xf32>) -> tensor<1x4x17x64xf32>
  return %o : tensor<1x4x17x64xf32>
}

// A per-(batch, head, query) scalar broadcast over every key: the block is a
// [17, 1] column; no key-axis padding is materialized for it.
// CHECK-LABEL: func.func @per_query
// CHECK-NOT: into tensor<17x32xf32>
// CHECK: tessera_attn.score_bias %{{.*}}, %{{.*}} : tensor<17x16xf32>, tensor<17x1xf32> -> tensor<17x16xf32>
func.func @per_query(%q: tensor<1x4x17x64xf32>, %key: tensor<1x2x19x64xf32>, %v: tensor<1x2x19x64xf32>, %bias: tensor<1x4x17x1xf32>) -> tensor<1x4x17x64xf32> {
  %o = "tessera.flash_attn"(%q, %key, %v, %bias) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> {causal = true, dropout_p = 0.0 : f64, dropout_seed = 0 : i64, head_dim = 64 : i64, scale = 0.125 : f32, softcap = 0.0 : f32, tessera.tile_q = 17 : i32, tessera.tile_kv = 16 : i32, window_left = -1 : i64, window_right = -1 : i64} : (tensor<1x4x17x64xf32>, tensor<1x2x19x64xf32>, tensor<1x2x19x64xf32>, tensor<1x4x17x1xf32>) -> tensor<1x4x17x64xf32>
  return %o : tensor<1x4x17x64xf32>
}
