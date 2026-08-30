// Per-instance dropout RNG streams for distributed rank-4 FlashAttention.
//
// `DistributeRank4FlashAttn` splits a rank-4 attention into B*H rank-2
// instances inside two annotated `scf.for` loops, and each instance restarts
// its own KV `boundary` iter-arg at 0.  The dropout counter therefore has to
// carry a second, per-instance axis or every one of the B*H instances replays
// one identical mask (Decision #18 stream separation).  The batch/head
// coordinates are loop induction variables, so `tessera_attn.block_dropout`
// takes the stream base as an SSA operand rather than folding it into `seed`.
//
// RUN: tessera-opt --tessera-tile-ir-lowering='tile-q=8 tile-kv=8 sm=90' \
// RUN:   --allow-unregistered-dialect %s | FileCheck %s

// B=2, H=4, Sq=8, Sk=8 -> instance index b*4 + h, stride Sq*Sk_padded = 64.
// CHECK-LABEL: func.func @rank4_dropout_gets_per_instance_streams
// CHECK-DAG: %[[STRIDE:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[HEADS:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[B:.*]] = %{{.*}} to %{{.*}} step
// CHECK: scf.for %[[H:.*]] = %{{.*}} to %{{.*}} step
// CHECK: %[[SCALED:.*]] = arith.muli %[[B]], %[[HEADS]]
// CHECK: %[[INSTANCE:.*]] = arith.addi %[[SCALED]], %[[H]]
// CHECK: %[[STREAM:.*]] = arith.muli %[[INSTANCE]], %[[STRIDE]]
// The stream base is loop-invariant, so it is hoisted above the KV loop.
// CHECK: scf.for %{{.*}} iter_args
// CHECK: tessera_attn.block_dropout %{{.*}} kv_off = %{{.*}} stream = %[[STREAM]] p = 2.500000e-01 seed = 37
// CHECK: tessera.attention_distribution = "query_head"
// CHECK: tessera.attention_distribution = "batch"
func.func @rank4_dropout_gets_per_instance_streams(
    %q: tensor<2x4x8x16xf16>,
    %k: tensor<2x4x8x16xf16>,
    %v: tensor<2x4x8x16xf16>) -> tensor<2x4x8x16xf32> {
  %o = "tessera.flash_attn"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    causal = true,
    dropout_p = 0.25 : f64,
    dropout_seed = 37 : i64,
    head_dim = 16 : i64,
    scale = 0.25 : f32,
    tessera.tile_q = 8 : i32,
    tessera.tile_kv = 8 : i32
  } : (tensor<2x4x8x16xf16>, tensor<2x4x8x16xf16>, tensor<2x4x8x16xf16>)
      -> tensor<2x4x8x16xf32>
  return %o : tensor<2x4x8x16xf32>
}

// An attention that was never distributed is its own single instance: it must
// keep stream base 0 and therefore the exact mask it drew before, so the fix
// cannot perturb a rank-2 lane.  The negative half matters — a producer that
// invented a stream for every attention would silently change every existing
// rank-2 dropout mask.
// CHECK-LABEL: func.func @rank2_dropout_keeps_stream_zero
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: tessera_attn.block_dropout %{{.*}} kv_off = %{{.*}} stream = %[[ZERO]] p = 2.500000e-01 seed = 37
// CHECK-NOT: tessera.attention_distribution
// CHECK-NOT: tessera.flash_attn
func.func @rank2_dropout_keeps_stream_zero(
    %q: tensor<8x16xf16>,
    %k: tensor<8x16xf16>,
    %v: tensor<8x16xf16>) -> tensor<8x16xf32> {
  %o = "tessera.flash_attn"(%q, %k, %v)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {
    causal = true,
    dropout_p = 0.25 : f64,
    dropout_seed = 37 : i64,
    head_dim = 16 : i64,
    scale = 0.25 : f32,
    tessera.tile_q = 8 : i32,
    tessera.tile_kv = 8 : i32
  } : (tensor<8x16xf16>, tensor<8x16xf16>, tensor<8x16xf16>)
      -> tensor<8x16xf32>
  return %o : tensor<8x16xf32>
}
