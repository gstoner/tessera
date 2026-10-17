module attributes {tessera.ir.version = "1.0", tessera.frontend.authority = "tracer", tessera.target = "rocm", tessera.arch = "gfx1201", tessera.launch_bindings = ["parent", "v2"]} {
  func.func @gate_up(%parent: tensor<3x34xf32>) -> (tensor<3x17xf32>) attributes {tessera.frontend.authority = "tracer", tessera.structured_cfg.schema = "tessera.structured_cfg.v1", tessera.structured_cfg.digest = "7b8562d68b1ad2ebb1317b1f8b8360688b6562f0be0c6d324bbc39937b462465", tessera.structured_cfg.blocks = 2} {
    %v0 = tessera.slice %parent {start_indices = [0, 0], slice_sizes = [3, 17], tessera.effect_kind = "pure"} : (tensor<3x34xf32>) -> tensor<3x17xf32> loc("/home/angstorms/scratch/rocm-native-gate-up-views-20261016/record-post-slice-boundary.py":8:10)
    %v1 = tessera.slice %parent {start_indices = [0, 17], slice_sizes = [3, 17], tessera.effect_kind = "pure"} : (tensor<3x34xf32>) -> tensor<3x17xf32> loc("/home/angstorms/scratch/rocm-native-gate-up-views-20261016/record-post-slice-boundary.py":9:8)
    %v2 = tessera.silu_mul %v0, %v1 {tessera.effect_kind = "pure"} : (tensor<3x17xf32>, tensor<3x17xf32>) -> tensor<3x17xf32> loc("/home/angstorms/scratch/rocm-native-gate-up-views-20261016/record-post-slice-boundary.py":10:12)
    return %v2 : tensor<3x17xf32>
  }
}