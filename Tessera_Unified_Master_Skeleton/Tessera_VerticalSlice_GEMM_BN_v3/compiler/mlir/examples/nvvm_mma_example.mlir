// compiler/mlir/examples/nvvm_mma_example.mlir
// Illustrative NVVM dialect snippet (not guaranteed to assemble standalone)

module {
  llvm.func @gemm_kernel() {
    // Pseudo: load A/B fragments from shared memory to registers
    // %a = nvvm.ldmatrix ...
    // %b = nvvm.ldmatrix ...

    // Accumulator fragment (f32)
    %c0 = llvm.mlir.undef : vector<4xf32>

    // nvvm.mma.sync with bf16 inputs -> f32 acc
    %c1 = nvvm.mma.sync {m=16, n=8, k=16} bf16 bf16 f32 %a, %b, %c0
          : (vector<8xbf16>, vector<8xbf16>, vector<4xf32>) -> vector<4xf32>

    // store %c1 ...
    llvm.return
  }
}
