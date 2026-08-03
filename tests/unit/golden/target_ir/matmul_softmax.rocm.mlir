module attributes {tessera.arch = "gfx90a", tessera.ir.level = "target", tessera.target = "rocm", tessera.target_features = "{\"device_timers\": false, \"family\": \"rocm\", \"mfma\": true}"} {
  func.func @main() {
    %mfma_a_0 = ub.poison : vector<16xf16>
    %mfma_b_0 = ub.poison : vector<16xf16>
    %mfma_acc_0 = ub.poison : vector<8xf32>
    %mfma_res_0 = "tessera_rocm.mfma"(%mfma_a_0, %mfma_b_0, %mfma_acc_0) {accum = "f32", arch = "gfx90a", launch = "{\"block\": \"warpgroup\", \"grid\": \"mn_tiles\", \"kernel_id\": \"matmul\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, resource = "{\"async_copy_bytes\": 64, \"barrier_count\": 2, \"queue_depth\": 2, \"register_estimate\": 64, \"shared_memory_bytes\": 65536}", result = "C", shape = "m16n16k16", source = "tessera.matmul", target = "rocm"} : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
    "tessera_rocm.wmma_gemm"() {dtype = "f16", k = 16 : i64, launch = "{\"block\": \"warpgroup\", \"grid\": \"mn_tiles\", \"kernel_id\": \"matmul\", \"measurement\": \"wall_clock_pending\"}", m = 16 : i64, n = 16 : i64, name = "gemm", ordinal = 0 : i64, resource = "{\"async_copy_bytes\": 64, \"barrier_count\": 2, \"queue_depth\": 2, \"register_estimate\": 64, \"shared_memory_bytes\": 65536}", result = "C", source = "tessera.matmul", target = "rocm"} : () -> ()
    %copy_dst_0 = ub.poison : memref<16xf16>
    %copy_src_0 = ub.poison : memref<16xf16>
    %copy_bytes_0 = arith.constant 16 : i64
    %copy_tok_0 = "tessera_rocm.async_copy"(%copy_dst_0, %copy_src_0, %copy_bytes_0) {bytes = 16 : i64, dst_space = "lds", launch = "{\"block\": \"warpgroup\", \"grid\": \"mn_tiles\", \"kernel_id\": \"matmul\", \"measurement\": \"wall_clock_pending\"}", ordinal = 0 : i64, resource = "{\"async_copy_bytes\": 64, \"barrier_count\": 2, \"queue_depth\": 2, \"register_estimate\": 64, \"shared_memory_bytes\": 65536}", result = "C", source = "tessera.matmul", src_space = "global", target = "rocm"} : (memref<16xf16>, memref<16xf16>, i64) -> !tessera_rocm.token
    "tessera_rocm.wait"(%copy_tok_0) {ordinal = 0 : i64} : (!tessera_rocm.token) -> ()
    "tessera_rocm.elementwise"() {arch = "gfx90a", launch = "{\"block\": \"256\", \"grid\": \"rows\", \"kernel_id\": \"softmax\", \"measurement\": \"wall_clock_pending\"}", ordinal = 1 : i64, resource = "{\"async_copy_bytes\": 0, \"barrier_count\": 1, \"queue_depth\": 0, \"register_estimate\": 24, \"shared_memory_bytes\": 1024}", result = "P", source = "tessera.softmax", target = "rocm"} : () -> ()
    %copy_dst_1 = ub.poison : memref<16xf16>
    %copy_src_1 = ub.poison : memref<16xf16>
    %copy_bytes_1 = arith.constant 16 : i64
    %copy_tok_1 = "tessera_rocm.async_copy"(%copy_dst_1, %copy_src_1, %copy_bytes_1) {bytes = 16 : i64, dst_space = "lds", launch = "{\"block\": \"256\", \"grid\": \"rows\", \"kernel_id\": \"softmax\", \"measurement\": \"wall_clock_pending\"}", ordinal = 1 : i64, resource = "{\"async_copy_bytes\": 0, \"barrier_count\": 1, \"queue_depth\": 0, \"register_estimate\": 24, \"shared_memory_bytes\": 1024}", result = "P", source = "tessera.softmax", src_space = "global", target = "rocm"} : (memref<16xf16>, memref<16xf16>, i64) -> !tessera_rocm.token
    "tessera_rocm.wait"(%copy_tok_1) {ordinal = 1 : i64} : (!tessera_rocm.token) -> ()
    return
  }
}