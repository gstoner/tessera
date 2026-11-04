// E2E-REAL-6 Apple CPU factorizations (2026-11-04): an opt-in Apple CPU
// module (launch bindings) records its batched-GEMM/LAPACK op as a
// content-addressed Schedule contract, lowers it to the registered Tile value
// op with every semantic flag written explicitly, and the Apple CPU value
// lowering turns that into one verified tessera_apple.cpu.call carrying the
// schedule hash (NativeAppleCPUKernel.h).
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefix=TILE
// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --tile-to-apple_cpu=value=true --split-input-file %s | FileCheck %s --check-prefix=TARGET

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "b", "x"]} {
  func.func @tri_solve(%a: tensor<4x4xf32>, %b: tensor<4x3xf32>) -> tensor<4x3xf32> {
    %0 = tessera.tri_solve %a, %b {lower = false, trans = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>
  }
}

// SCHEDULE-LABEL: func.func @tri_solve
// SCHEDULE: tessera.tri_solve{{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: abi = "lapack_strtrs"
// SCHEDULE-SAME: scalars = {K = 3 : i64, N = 4 : i64}
// SCHEDULE-SAME: semantics = {lower = false, trans = true, unit_diag = false}
// SCHEDULE-SAME: symbol = "tessera_apple_cpu_tri_solve_f32"
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE-SAME: shape_key = "family=apple_cpu_kernel;kind=tri_solve"

// TILE-LABEL: func.func @tri_solve
// TILE-NOT: tessera.tri_solve
// TILE-NOT: schedule.
// TILE: tile.tri_solve
// TILE-SAME: lower = false
// TILE-SAME: source = "tessera.tri_solve"
// TILE-SAME: tessera.apple_cpu_kernel_contract =
// TILE-SAME: trans = true, unit_diag = false

// TARGET-LABEL: func.func @tri_solve
// TARGET-NOT: tile.
// TARGET: tessera_apple.cpu.call
// TARGET-SAME: abi = "lapack_strtrs"
// TARGET-SAME: lower = false
// TARGET-SAME: op_kind = "tri_solve"
// TARGET-SAME: symbol = "tessera_apple_cpu_tri_solve_f32"
// TARGET-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// TARGET-SAME: trans = true, unit_diag = false

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "lu", "pivots"]} {
  func.func @lu(%a: tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5xi32>) {
    %lu, %pivots = tessera.lu %a : (tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5xi32>)
    return %lu, %pivots : tensor<5x5xf32>, tensor<5xi32>
  }
}

// SCHEDULE-LABEL: func.func @lu
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: storage = ["f32", "f32", "i32"]
// SCHEDULE-SAME: shape_key = "family=apple_cpu_kernel;kind=lu"

// TILE-LABEL: func.func @lu
// TILE: %[[R:.*]]:2 = tile.lu
// TILE: return %[[R]]#0, %[[R]]#1

// TARGET-LABEL: func.func @lu
// TARGET: tessera_apple.cpu.call
// TARGET-SAME: op_kind = "lu"
// TARGET-SAME: (tensor<5x5xf32>) -> (tensor<5x5xf32>, tensor<5xi32>)

// -----

module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu", tessera.launch_bindings = ["a", "b", "c"]} {
  func.func @batched(%a: tensor<2x7x9xf32>, %b: tensor<2x9x5xf32>) -> tensor<2x7x5xf32> {
    %0 = tessera.batched_gemm %a, %b : (tensor<2x7x9xf32>, tensor<2x9x5xf32>) -> tensor<2x7x5xf32>
    return %0 : tensor<2x7x5xf32>
  }
}

// SCHEDULE-LABEL: func.func @batched
// SCHEDULE: schedule.artifact {
// SCHEDULE-SAME: scalars = {Batch = 2 : i64, K = 9 : i64, M = 7 : i64, N = 5 : i64, StrideA = 63 : i64, StrideB = 45 : i64, StrideC = 35 : i64}
// SCHEDULE-SAME: semantics = {}

// TARGET-LABEL: func.func @batched
// TARGET: tessera_apple.cpu.call
// TARGET-SAME: abi = "cblas_sgemm_batched_loop"
// TARGET-SAME: symbol = "tessera_apple_cpu_gemm_f32_batched"

// -----

// Without launch bindings the module is not opted in, and the op passes
// Graph -> Schedule untouched, exactly as before this contract.
module attributes {tessera.target = "apple_cpu", tessera.arch = "apple_silicon_cpu"} {
  func.func @unclaimed(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// SCHEDULE-LABEL: func.func @unclaimed
// SCHEDULE-NOT: schedule.artifact
// SCHEDULE: tessera.cholesky
// SCHEDULE-NOT: schedule.artifact
