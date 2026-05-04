// Phase 8 Target IR contract fixtures.
// These checks document the hardware-free Target IR spellings for backend
// bring-up. Backend-specific lit suites run these through real tools when the
// corresponding target is enabled.

// ROCM:       tessera_rocm.mfma
// ROCM:       tessera_rocm.async_copy
// ROCM:       tessera_rocm.wait

// METALIUM:   tessera_metalium.dma
// METALIUM:   tessera_metalium.matmul
// METALIUM-NOT: tile.mma

// APPLE-CPU:  tessera_apple.cpu.accelerate_gemm
// APPLE-CPU:  framework = "Accelerate"

// APPLE-GPU:  tessera_apple.gpu.metal_kernel
// APPLE-GPU:  tessera_apple.gpu.dispatch

module attributes {tessera.ir.level = "target"} {
  "tessera_rocm.mfma"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, arch = "gfx90a"} : () -> ()
  "tessera_rocm.async_copy"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, bytes = 16 : i64} : () -> ()
  "tessera_rocm.wait"() {ordinal = 0 : i64} : () -> ()

  "tessera_metalium.dma"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, direction = "dram_to_sram"} : () -> ()
  "tessera_metalium.matmul"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, tile = [64, 64, 32]} : () -> ()

  "tessera_apple.cpu.accelerate_gemm"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, framework = "Accelerate"} : () -> ()
  "tessera_apple.gpu.metal_kernel"() {source = "tessera.matmul", result = "C", ordinal = 0 : i64, framework = "Metal"} : () -> ()
  "tessera_apple.gpu.dispatch"() {ordinal = 0 : i64, artifact = "metallib"} : () -> ()
}
