// RUN: tessera-opt --tessera-graph-to-schedule --tessera-schedule-to-tile --split-input-file %s | FileCheck %s --check-prefixes=X86,ROCM
// RUN: tessera-opt --tessera-graph-to-schedule --split-input-file %s | FileCheck %s --check-prefix=SCHEDULE

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_matmul(%a: tensor<64x32xf32>, %b: tensor<32x48xf32>)
      -> tensor<64x48xf32> {
    %0 = tessera.matmul %a, %b
        : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    return %0 : tensor<64x48xf32>
  }
}

// X86-LABEL: func.func @x86_matmul
// X86-NOT: tessera.matmul
// X86-NOT: schedule.
// X86-COUNT-1: tile.matmul_kernel
// X86-SAME: %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}}
// X86-SAME: family = "auto"
// X86-SAME: a = "f32"
// X86-SAME: b = "f32"
// X86-SAME: acc = "f32"
// X86-SAME: a_layout = "row_major"
// X86-SAME: b_layout = "col_major"
// X86-SAME: tessera.canonical_k_loop = true
// X86-SAME: tessera.macro_tile_m = 16
// X86-SAME: tessera.macro_tile_n = 16
// X86-SAME: tessera.pipeline_depth = 1
// X86-SAME: tessera.raster_group = 1
// X86-SAME: tessera.raster_order = "row_major"
// X86-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// X86-SAME: tessera.tile_k = 16
// X86-SAME: tessera.tile_m = 16
// X86-SAME: tessera.tile_n = 16
// X86: return %{{.*}} : tensor<64x48xf32>

// SCHEDULE-LABEL: func.func @x86_matmul
// SCHEDULE: %[[GRAPH:.*]] = tessera.matmul {{.*}}schedule.artifact_hash = "[[HASH:[0-9a-f]+]]"
// SCHEDULE: %[[SCHEDULED:.*]] = schedule.matmul %[[GRAPH]]
// SCHEDULE-SAME: artifact_hash = "[[HASH]]"
// SCHEDULE-SAME: macro_tile_m = 16
// SCHEDULE-SAME: macro_tile_n = 16
// SCHEDULE-SAME: storage = "f32"
// SCHEDULE-SAME: tile_m = 16
// SCHEDULE: schedule.artifact
// SCHEDULE-SAME: hash = "[[HASH]]"
// SCHEDULE: return %[[SCHEDULED]] : tensor<64x48xf32>

// -----

module attributes {tessera.target = "rocm", tessera.arch = "gfx1151"} {
  func.func @rocm_matmul(%a: tensor<48x32xf16>, %b: tensor<32x80xf16>)
      -> tensor<48x80xf32> {
    %0 = tessera.matmul %a, %b
        : (tensor<48x32xf16>, tensor<32x80xf16>) -> tensor<48x80xf32>
    return %0 : tensor<48x80xf32>
  }
}

// ROCM-LABEL: func.func @rocm_matmul
// ROCM-NOT: tessera.matmul
// ROCM-NOT: schedule.
// ROCM-COUNT-1: tile.matmul_kernel
// ROCM-SAME: family = "wmma"
// ROCM-SAME: a = "f16"
// ROCM-SAME: b = "f16"
// ROCM-SAME: acc = "f32"
// ROCM-SAME: tessera.macro_tile_m = 32
// ROCM-SAME: tessera.macro_tile_n = 64
// ROCM-SAME: tessera.schedule_hash = "{{[0-9a-f]+}}"
// ROCM-SAME: tessera.tile_k = 16
// ROCM-SAME: tessera.tile_m = 16
// ROCM-SAME: tessera.tile_n = 16
// ROCM: return %{{.*}} : tensor<48x80xf32>
