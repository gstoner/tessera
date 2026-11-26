// APPLE-NTBI-2 slice 11 (2026-11-26): an apple7 row norm of at least 16
// columns takes one 32-lane simdgroup per row (schedule simdgroup_32,
// workgroup 32); a shorter row keeps one serial thread (workgroup 1, no Tile
// schedule attribute). The 16-column cut is a measured performance key.
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_16_columns(%x: tensor<5x16xf16>) -> tensor<5x16xf16> {
    %0 = tessera.rmsnorm %x {eps = 1.0e-06 : f64} : (tensor<5x16xf16>) -> tensor<5x16xf16>
    return %0 : tensor<5x16xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_16_columns
// SCHEDULE: schedule.norm
// SCHEDULE-SAME: arch = "apple7"
// SCHEDULE-SAME: kind = "rmsnorm"
// SCHEDULE-SAME: schedule = "simdgroup_32"
// SCHEDULE-SAME: storage = "f16"
// SCHEDULE-SAME: workgroup_size = 32 : i64

// TILE-LABEL: func.func @apple_rmsnorm_16_columns
// TILE: tile.norm_kernel
// TILE-SAME: affine = false
// TILE-SAME: kind = "rmsnorm"
// TILE-SAME: schedule = "simdgroup_32"
// TILE-SAME: tessera.workgroup_size = 32 : i64

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_15_columns(%x: tensor<5x15xf16>) -> tensor<5x15xf16> {
    %0 = tessera.rmsnorm %x {eps = 1.0e-06 : f64} : (tensor<5x15xf16>) -> tensor<5x15xf16>
    return %0 : tensor<5x15xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_15_columns
// SCHEDULE: schedule.norm
// SCHEDULE-NOT: simdgroup_32
// SCHEDULE-SAME: workgroup_size = 1 : i64

// TILE-LABEL: func.func @apple_rmsnorm_15_columns
// TILE: tile.norm_kernel
// TILE-NOT: schedule =
// TILE-SAME: tessera.workgroup_size = 1 : i64

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_layer_norm_gamma_beta_simdgroup(%x: tensor<4x64xbf16>, %g: tensor<64xbf16>, %b: tensor<64xbf16>) -> tensor<4x64xbf16> {
    %0 = tessera.layer_norm %x, %g, %b : (tensor<4x64xbf16>, tensor<64xbf16>, tensor<64xbf16>) -> tensor<4x64xbf16>
    return %0 : tensor<4x64xbf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_layer_norm_gamma_beta_simdgroup
// SCHEDULE: schedule.norm
// SCHEDULE-SAME: affine = true
// SCHEDULE-SAME: bias = true
// SCHEDULE-SAME: kind = "layernorm"
// SCHEDULE-SAME: schedule = "simdgroup_32"
// SCHEDULE-SAME: workgroup_size = 32 : i64

// TILE-LABEL: func.func @apple_layer_norm_gamma_beta_simdgroup
// TILE: tile.norm_kernel %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// TILE-SAME: affine = true
// TILE-SAME: bias = true
// TILE-SAME: schedule = "simdgroup_32"
// TILE-SAME: tessera.workgroup_size = 32 : i64


// -----

// A tall tensor whose rows * 32 overflows int32 keeps the serial launch it
// fits (rows * columns <= INT32_MAX), rather than a simdgroup launch the
// int32 runtime ABIs cannot dispatch.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_rmsnorm_tall_rows(%x: tensor<67108864x16xf16>) -> tensor<67108864x16xf16> {
    %0 = tessera.rmsnorm %x : (tensor<67108864x16xf16>) -> tensor<67108864x16xf16>
    return %0 : tensor<67108864x16xf16>
  }
}

// SCHEDULE-LABEL: func.func @apple_rmsnorm_tall_rows
// SCHEDULE: schedule.norm
// SCHEDULE-NOT: simdgroup_32
// SCHEDULE-SAME: workgroup_size = 1 : i64

// TILE-LABEL: func.func @apple_rmsnorm_tall_rows
// TILE: tile.norm_kernel
// TILE-NOT: schedule =
// TILE-SAME: tessera.workgroup_size = 1 : i64
