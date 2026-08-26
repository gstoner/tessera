// RUN: not %trop --pass-pipeline='builtin.module(lower-tile-to-rocm{arch=gfx1151})' %s 2>&1 | FileCheck %s

module {
  llvm.func @spectral_backward(
      %dy: !llvm.ptr, %x: !llvm.ptr, %filter: !llvm.ptr,
      %dx: !llvm.ptr, %df: !llvm.ptr) {
    tile.spectral_backward_kernel %dy, %x, %filter, %dx, %df {
      target = "rocm", arch = "gfx1151",
      kind = "tessera.stft", axis = -1 : i64,
      logical_length = 8 : i64, normalization = "backward",
      spectrum_layout = "full_complex", center = false, onesided = true,
      pad_mode = "constant",
      input_count = 3 : i64, output_count = 2 : i64,
      input_signature = "tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>",
      output_signature = "tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>",
      mutation_lineage = "inputs_immutable_outputs_fresh_v1",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
    llvm.return
  }
}

// CHECK: error: native ROCm STFT/ISTFT adjoint requires the bounded uncentered onesided contiguous policy and explicit fp32 accumulation
