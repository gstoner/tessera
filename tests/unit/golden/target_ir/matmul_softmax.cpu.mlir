module attributes {tessera.arch = "x86_64", tessera.execution_mode = "numpy", tessera.ir.level = "target", tessera.target = "cpu", tessera.target_features = "{\"device_timers\": false, \"family\": \"cpu\", \"wall_clock\": true}"} {
  func.func @main() {
    "tessera.cpu.reference"() {abi = "numpy", launch = "{\"block\": \"warpgroup\", \"grid\": \"mn_tiles\", \"kernel_id\": \"matmul\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, resource = "{\"async_copy_bytes\": 64, \"barrier_count\": 2, \"queue_depth\": 2, \"register_estimate\": 64, \"shared_memory_bytes\": 65536}", result = "C", source = "tessera.matmul"} : () -> ()
    "tessera.cpu.reference"() {abi = "numpy", launch = "{\"block\": \"256\", \"grid\": \"rows\", \"kernel_id\": \"softmax\", \"measurement\": \"wall_clock_pending\"}", ordinal = 1 : i64, resource = "{\"async_copy_bytes\": 0, \"barrier_count\": 1, \"queue_depth\": 0, \"register_estimate\": 24, \"shared_memory_bytes\": 1024}", result = "P", source = "tessera.softmax"} : () -> ()
    return
  }
}