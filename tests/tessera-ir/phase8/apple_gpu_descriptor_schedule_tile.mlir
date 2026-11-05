// E2E-REAL-6 Apple GPU runtime-descriptor families (2026-11-04): an opt-in
// Apple GPU module (launch bindings) records its descriptor op as a
// content-addressed Schedule contract, lowers it to a Tile value op carrying
// that contract, and the Apple GPU value lowering turns it into one verified
// tessera_apple.gpu.kernel_call whose runtime entry it derives itself
// (NativeAppleGPUDescriptor.h).
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --tile-to-apple_gpu=value=true --split-input-file %s | FileCheck %s --check-prefix=TARGET

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "values", "indices"]} {
  func.func @topk(%x: tensor<?x?xf32>) -> (tensor<?x3xf32>, tensor<?x3xi32>) {
    %v, %i = tessera.top_k %x {k = 3, axis = -1} : (tensor<?x?xf32>) -> (tensor<?x3xf32>, tensor<?x3xi32>)
    return %v, %i : tensor<?x3xf32>, tensor<?x3xi32>
  }
}

// SCHEDULE-LABEL: func.func @topk
// SCHEDULE: tessera.top_k{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: abi = "tessera.apple.top_k.x_values_indices_rows_columns_k.dynamic.f32_i32.v1"
// SCHEDULE-SAME: scalars = {{\[}}["Rows", "int32"], ["Columns", "int32"], ["K", "int32"]]
// SCHEDULE-SAME: semantics = {axis = -1 : i64, k = 3 : i64, nan_policy = "last", tie_policy = "lower_index", value_order = "descending"}
// SCHEDULE-SAME: shapes = [array<i64: -1, -1>, array<i64: -1, 3>, array<i64: -1, 3>]
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_topk_f32"
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE-SAME: shape_key = "family=apple_gpu_descriptor;kind=dynamic_topk"

// TILE-LABEL: func.func @topk
// TILE-NOT: tessera.top_k
// TILE-NOT: schedule.
// TILE: "tile.top_k"
// TILE-SAME: source = "tessera.top_k"
// TILE-SAME: tessera.apple_gpu_descriptor_contract =
// TILE-SAME: tessera.schedule_hash =

// TARGET-LABEL: func.func @topk
// TARGET-NOT: tile.top_k
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "tessera.apple.top_k.x_values_indices_rows_columns_k.dynamic.f32_i32.v1"
// TARGET-SAME: framework = "Metal", op_kind = "top_k", status = "executable", symbol = "tessera_apple_gpu_topk_f32"
// TARGET-SAME: tessera.schedule_hash =

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y"]} {
  func.func @transpose(%x: tensor<2x3x5xbf16>) -> tensor<5x2x3xbf16> {
    %0 = tessera.transpose %x {permutation = array<i64: 2, 0, 1>} : (tensor<2x3x5xbf16>) -> tensor<5x2x3xbf16>
    return %0 : tensor<5x2x3xbf16>
  }
}

// SCHEDULE-LABEL: func.func @transpose
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: framework = "MPSGraph"
// SCHEDULE-SAME: semantics = {axes = array<i64: 2, 0, 1>}
// SCHEDULE-SAME: storage = ["bf16", "bf16"]
// SCHEDULE-SAME: shape_key = "family=apple_gpu_descriptor;kind=transpose"

// TARGET-LABEL: func.func @transpose
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.bf16.v1"
// TARGET-SAME: symbol = "tessera_apple_gpu_mpsgraph_transpose_f16"

// -----

// GELU is claimed only in its dynamic form; the static form belongs to the
// shared semantic-kernel contract.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["x", "y"]} {
  func.func @dynamic_gelu(%x: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %0 = tessera.gelu %x : (tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}

// SCHEDULE-LABEL: func.func @dynamic_gelu
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: scalars = {{\[}}["Elements", "int64"]]
// SCHEDULE-SAME: semantics = {approximation = "tanh"}
// SCHEDULE-SAME: shape_key = "family=apple_gpu_descriptor;kind=dynamic_gelu"

// TARGET-LABEL: func.func @dynamic_gelu
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: symbol = "tessera_apple_gpu_gelu_f16_status"
