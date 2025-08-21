// compiler/mlir/examples/nvvm_wgmma_example.mlir
// Illustrative: NVVM wgmma.mma_async for larger tile shapes (Hopper/Blackwell).

module {
  // Placeholder function showing shape annotations
  llvm.func @wgmma_kernel() {
    // %a, %b loaded as matrix fragments (not shown)
    // %acc0 : vector<8xf32> = llvm.mlir.undef
    // %acc1 = nvvm.wgmma.mma_async {m=64, n=128, k=32} bf16 bf16 f32 %a, %b, %acc0
    // ... cp.async fences, commit_group, wait_group ...
    llvm.return
  }
}
