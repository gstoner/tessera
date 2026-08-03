module attributes {tessera.arch = "arm64-apple-silicon", tessera.execution_mode = "cpu_accelerate", tessera.ir.level = "target", tessera.target = "apple_cpu", tessera.target_features = "{\"accelerate\": true, \"device_timers\": false, \"family\": \"apple\"}"} {
  func.func @main() {
    "tessera_apple.cpu.accelerate_gemm"() {abi = "cblas_sgemm", dtype = "f32", framework = "Accelerate", launch = "{\"block\": \"warpgroup\", \"grid\": \"mn_tiles\", \"kernel_id\": \"matmul\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, resource = "{\"async_copy_bytes\": 64, \"barrier_count\": 2, \"queue_depth\": 2, \"register_estimate\": 64, \"shared_memory_bytes\": 65536}", result = "C", source = "tessera.matmul"} : () -> ()
    "tessera_apple.cpu.vector_reduce"() {abi = "vDSP", dtype = "f32", framework = "Accelerate", launch = "{\"block\": \"256\", \"grid\": \"rows\", \"kernel_id\": \"softmax\", \"measurement\": \"wall_clock_pending\"}", ordinal = 1 : i64, resource = "{\"async_copy_bytes\": 0, \"barrier_count\": 1, \"queue_depth\": 0, \"register_estimate\": 24, \"shared_memory_bytes\": 1024}", result = "P", source = "tessera.softmax"} : () -> ()
    return
  }
}