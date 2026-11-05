// E2E-REAL-6 Apple GPU linalg (2026-11-04): an opt-in Apple GPU module
// (launch bindings) records its cholesky / tri_solve / cholesky_solve / reduced
// svd op as a content-addressed Schedule contract, lowers it to the registered
// Tile value op with every semantic flag written explicitly, and the Apple GPU
// value lowering turns that into one verified tessera_apple.gpu.kernel_call
// carrying the schedule hash (NativeAppleGPUKernel.h).
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --tile-to-apple_gpu=value=true --split-input-file %s | FileCheck %s --check-prefix=TARGET

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "b", "x"]} {
  func.func @tri_solve(%a: tensor<4x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
    %0 = tessera.tri_solve %a, %b {lower = false, trans = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// SCHEDULE-LABEL: func.func @tri_solve
// SCHEDULE: tessera.tri_solve{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: abi = "tessera.apple.value.tri_solve.f32.v1"
// SCHEDULE-SAME: framework = "Metal"
// SCHEDULE-SAME: scalars = {Batch = 1 : i64, K = 3 : i64, N = 4 : i64}
// SCHEDULE-SAME: semantics = {lower = false, trans = true, unit_diag = false}
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_tri_solve_f32"
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE-SAME: shape_key = "family=apple_gpu_kernel;kind=tri_solve"

// TILE-LABEL: func.func @tri_solve
// TILE-NOT: tessera.tri_solve
// TILE-NOT: schedule.
// TILE: tile.tri_solve
// TILE-SAME: lower = false
// TILE-SAME: source = "tessera.tri_solve"
// TILE-SAME: tessera.apple_gpu_kernel_contract =
// TILE-SAME: trans = true, unit_diag = false

// TARGET-LABEL: func.func @tri_solve
// TARGET-NOT: tile.
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "tessera.apple.value.tri_solve.f32.v1"
// TARGET-SAME: framework = "Metal"
// TARGET-SAME: lower = false
// TARGET-SAME: op_kind = "tri_solve"
// TARGET-SAME: status = "executable"
// TARGET-SAME: symbol = "tessera_apple_gpu_tri_solve_f32"
// TARGET-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// TARGET-SAME: trans = true, unit_diag = false

// -----

// A rank-3 batch of SPD solves (the Graph verifier admits it since this slice).
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "b", "x"]} {
  func.func @cholesky_solve(%a: tensor<2x3x3xf32>, %b: tensor<2x3x2xf32>) -> tensor<2x3x2xf32> {
    %0 = tessera.cholesky_solve %a, %b : (tensor<2x3x3xf32>, tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
    return %0 : tensor<2x3x2xf32>
  }
}

// SCHEDULE-LABEL: func.func @cholesky_solve
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: scalars = {Batch = 2 : i64, K = 2 : i64, N = 3 : i64}
// SCHEDULE-SAME: semantics = {lower = true}
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_solve_cholesky_f32"
// SCHEDULE-SAME: shape_key = "family=apple_gpu_kernel;kind=cholesky_solve"

// TARGET-LABEL: func.func @cholesky_solve
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: lower = true
// TARGET-SAME: op_kind = "cholesky_solve"
// TARGET-SAME: symbol = "tessera_apple_gpu_solve_cholesky_f32"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "u", "s", "vh"]} {
  func.func @svd(%a: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>) {
    %u, %s, %v = tessera.svd %a : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>)
    return %u, %s, %v : tensor<6x4xf32>, tensor<4xf32>, tensor<4x4xf32>
  }
}

// SCHEDULE-LABEL: func.func @svd
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: abi = "tessera.apple.svd.a_u_s_vh_m_n.f32.v1"
// SCHEDULE-SAME: scalars = {M = 6 : i64, N = 4 : i64}
// SCHEDULE-SAME: semantics = {full_matrices = false}
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_svd_reduced_f32"
// SCHEDULE-SAME: shape_key = "family=apple_gpu_kernel;kind=svd"

// TILE-LABEL: func.func @svd
// TILE: %{{.*}}:3 = tile.svd
// TILE-SAME: full_matrices = false

// TARGET-LABEL: func.func @svd
// TARGET: %{{.*}}:3 = tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "tessera.apple.svd.a_u_s_vh_m_n.f32.v1"
// TARGET-SAME: full_matrices = false
// TARGET-SAME: op_kind = "svd"
// TARGET-SAME: result_order = "u,s,vh"
// TARGET-SAME: symbol = "tessera_apple_gpu_svd_reduced_f32"

// -----

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu", tessera.launch_bindings = ["a", "u", "s", "vh"]} {
  func.func @svd_batched(%a: tensor<2x6x4xf32>) -> (tensor<2x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>) {
    %u, %s, %v = tessera.svd %a : (tensor<2x6x4xf32>) -> (tensor<2x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>)
    return %u, %s, %v : tensor<2x6x4xf32>, tensor<2x4xf32>, tensor<2x4x4xf32>
  }
}

// SCHEDULE-LABEL: func.func @svd_batched
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: scalars = {Batch = 2 : i64, M = 6 : i64, N = 4 : i64}
// SCHEDULE-SAME: symbol = "tessera_apple_gpu_svd_reduced_batched_f32"

// TARGET-LABEL: func.func @svd_batched
// TARGET: tessera_apple.gpu.kernel_call
// TARGET-SAME: abi = "tessera.apple.svd.a_u_s_vh_batch_m_n.f32.v1"
// TARGET-SAME: symbol = "tessera_apple_gpu_svd_reduced_batched_f32"

// -----

// No launch bindings: the module is not opted in, and the op passes
// Graph -> Schedule untouched.
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple_gpu"} {
  func.func @not_owned(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// SCHEDULE-LABEL: func.func @not_owned
// SCHEDULE: tessera.cholesky
// SCHEDULE-NOT: family=apple_gpu_kernel
