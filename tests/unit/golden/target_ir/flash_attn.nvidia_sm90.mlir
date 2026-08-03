module attributes {tessera.arch = "sm_90a", tessera.ir.level = "target", tessera.target = "nvidia_sm90", tessera.target_features = "{\"device_timers\": false, \"family\": \"nvidia\", \"tensor_cores\": true}"} {
  func.func @flash() {
    "tessera_nvidia.cuda_kernel"() {arch = "sm_90a", kernel = "flash_attn_contract", launch = "{\"grid\": \"elements\", \"kernel_id\": \"flash_attn\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, result = "O", source = "tessera.flash_attn", status = "artifact_only"} : () -> ()
    "tessera_nvidia.cuda_kernel"() {arch = "sm_90a", kernel = "flash_attn_contract", launch = "{\"grid\": \"elements\", \"kernel_id\": \"flash_attn\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, result = "O", source = "tessera.flash_attn", status = "artifact_only"} : () -> ()
    "tessera_nvidia.cuda_kernel"() {arch = "sm_90a", kernel = "flash_attn_contract", launch = "{\"grid\": \"elements\", \"kernel_id\": \"flash_attn\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, result = "O", source = "tessera.flash_attn", status = "artifact_only"} : () -> ()
    "tessera_nvidia.cuda_kernel"() {arch = "sm_90a", kernel = "flash_attn_contract", launch = "{\"grid\": \"elements\", \"kernel_id\": \"flash_attn\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, result = "O", source = "tessera.flash_attn", status = "artifact_only"} : () -> ()
    return
  }
}