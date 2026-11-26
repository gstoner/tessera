// APPLE-NTBI-2 slice 9 (2026-11-25): an isolated static rank-two f16/bf16
// unweighted rmsnorm/layer_norm on apple7 is a native scheduled norm lowered
// to one Tile launch: one serial row thread (workgroup 1), f32 accumulation,
// the epsilon the Schedule hashed. Its consumer is compiler-synthesized MSL.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_f16(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.rmsnorm %x {eps = 1.0e-06 : f64} : (tensor<5x11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_f16
// SCHEDULE: %[[GRAPH:.*]] = tessera.rmsnorm{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.norm %[[GRAPH]]
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: epsilon = 9.99999997E-7 : f32
// SCHEDULE-SAME: kind = "rmsnorm"
// SCHEDULE-SAME: storage = "f16"
// SCHEDULE-SAME: workgroup_size = 1 : i64
// SCHEDULE: schedule.artifact

// TILE-LABEL: func.func @apple_rmsnorm_f16
// TILE-NOT: tessera.rmsnorm
// TILE-NOT: schedule.
// TILE-DAG: %[[ROWS:.*]] = arith.constant 5 : i64
// TILE-DAG: %[[COLS:.*]] = arith.constant 11 : i64
// TILE-DAG: %[[EPS:.*]] = arith.constant 9.99999997E-7 : f32
// TILE: tile.norm_kernel %{{.*}}, %{{.*}}, %[[ROWS]], %[[COLS]], %[[EPS]]
// TILE-SAME: affine = false
// TILE-SAME: kind = "rmsnorm"
// TILE-SAME: storage = "f16"
// TILE-SAME: tessera.workgroup_size = 1 : i64

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_layer_norm_bf16(%x: tensor<5x11xbf16>) -> tensor<5x11xbf16> {
    %0 = tessera.layer_norm %x : (tensor<5x11xbf16>) -> tensor<5x11xbf16>
    return %0 : tensor<5x11xbf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_layer_norm_bf16
// SCHEDULE: schedule.norm
// SCHEDULE-SAME: kind = "layernorm"
// SCHEDULE-SAME: storage = "bf16"

// TILE-LABEL: func.func @apple_layer_norm_bf16
// TILE: tile.norm_kernel
// TILE-SAME: kind = "layernorm"
// TILE-SAME: storage = "bf16"

// -----

// f32 storage has no apple7 norm consumer: the op is not claimed.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_f32(%x: tensor<5x11xf32>) -> tensor<5x11xf32> {
    %0 = tessera.rmsnorm %x : (tensor<5x11xf32>) -> tensor<5x11xf32>
    return %0 : tensor<5x11xf32>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_f32
// SCHEDULE-NOT: schedule.norm
// SCHEDULE: return

// -----

// A norm that is not the whole function body is not claimed by name alone.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_norm_chain(%x: tensor<5x11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.rmsnorm %x : (tensor<5x11xf16>) -> tensor<5x11xf16>
    %1 = tessera.relu %0 : (tensor<5x11xf16>) -> tensor<5x11xf16>
    return %1 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_norm_chain
// SCHEDULE-NOT: schedule.norm
// SCHEDULE: return

// -----

// APPLE-NTBI-2 slice 10: a weighted f16 rmsnorm passes gamma between the
// source and the destination; the Schedule hashes the affine form.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_gamma_f16(%x: tensor<5x11xf16>, %g: tensor<11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.rmsnorm %x, %g {eps = 1.0e-06 : f64} : (tensor<5x11xf16>, tensor<11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_gamma_f16
// SCHEDULE: schedule.norm
// SCHEDULE-SAME: affine = true
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: bias = false
// SCHEDULE-SAME: kind = "rmsnorm"

// TILE-LABEL: func.func @apple_rmsnorm_gamma_f16
// TILE-NOT: schedule.
// TILE: %[[X:.*]] = llvm.inttoptr
// TILE: %[[O:.*]] = llvm.inttoptr
// TILE: %[[G:.*]] = llvm.inttoptr
// TILE: tile.norm_kernel %[[X]], %[[G]], %[[O]], %{{.*}}, %{{.*}}, %{{.*}} {
// TILE-SAME: affine = true
// TILE-SAME: bias = false
// TILE-SAME: kind = "rmsnorm"
// TILE-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32

// -----

// layer_norm with gamma and beta: both vectors, in operand order.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_layer_norm_gamma_beta_bf16(%x: tensor<5x11xbf16>, %g: tensor<11xbf16>, %b: tensor<11xbf16>) -> tensor<5x11xbf16> {
    %0 = tessera.layer_norm %x, %g, %b : (tensor<5x11xbf16>, tensor<11xbf16>, tensor<11xbf16>) -> tensor<5x11xbf16>
    return %0 : tensor<5x11xbf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_layer_norm_gamma_beta_bf16
// SCHEDULE: schedule.norm
// SCHEDULE-SAME: affine = true
// SCHEDULE-SAME: bias = true
// SCHEDULE-SAME: kind = "layernorm"

// TILE-LABEL: func.func @apple_layer_norm_gamma_beta_bf16
// TILE: %[[X:.*]] = llvm.inttoptr
// TILE: %[[O:.*]] = llvm.inttoptr
// TILE: %[[G:.*]] = llvm.inttoptr
// TILE: %[[B:.*]] = llvm.inttoptr
// TILE: tile.norm_kernel %[[X]], %[[G]], %[[B]], %[[O]], %{{.*}}, %{{.*}}, %{{.*}} {
// TILE-SAME: affine = true
// TILE-SAME: bias = true
// TILE-SAME: !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32

// -----

// A weighted norm chain is not the isolated request: nothing is claimed.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_weighted_norm_chain(%x: tensor<5x11xf16>, %g: tensor<11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.rmsnorm %x, %g : (tensor<5x11xf16>, tensor<11xf16>) -> tensor<5x11xf16>
    %1 = tessera.rmsnorm %0, %g : (tensor<5x11xf16>, tensor<11xf16>) -> tensor<5x11xf16>
    return %1 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_weighted_norm_chain
// SCHEDULE-NOT: schedule.norm

// -----

// A two-operand layer_norm may be layer_norm(x, beta=b): it is not claimed.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_two_operand_layer_norm(%x: tensor<5x11xf16>, %v: tensor<11xf16>) -> tensor<5x11xf16> {
    %0 = tessera.layer_norm %x, %v : (tensor<5x11xf16>, tensor<11xf16>) -> tensor<5x11xf16>
    return %0 : tensor<5x11xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_two_operand_layer_norm
// SCHEDULE-NOT: schedule.norm
