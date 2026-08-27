// RUN: not tessera-opt %s --tessera-tile-to-x86 --split-input-file 2>&1 \
// RUN:   | FileCheck %s

// AD-TSOL-STFT-BWD-1 (integrated-plan queue order 7) — the x86 counterpart of
// `gfx1151_spectral_backward_fail_closed.mlir`.
//
// `TileToX86Pass` accepts `tessera.spectral_filter` and `tessera.spectral_conv`
// for the compound spectral adjoint and refuses every other kind. That refusal
// was correct and **untested**, while the identical gfx1151 refusal has had a
// fixture since it was written. An untested refusal is the one that quietly
// becomes a fallthrough: the next `else if` added above it changes which
// kinds reach the error, and nothing notices.
//
// Order 7 now has a bounded AVX-512 STFT/ISTFT package through the native VJP
// plugin, but this generic TileToX86 kind is not wired to that content-addressed
// ABI. It must therefore continue to fail here — never silently pick a
// neighbouring kernel. The two obvious candidates are both measurably wrong:
// ISTFT is not the STFT adjoint up to any global scale (best-fit residual
// 0.968), nor after undoing the COLA window-sum division (0.887). Both
// refutations and the implemented adjoint oracle are pinned in
// tests/unit/test_stft_adjoint_contract.py.

module {
  llvm.func @spectral_backward(
      %dy: !llvm.ptr, %x: !llvm.ptr, %filter: !llvm.ptr,
      %dx: !llvm.ptr, %df: !llvm.ptr) {
    tile.spectral_backward_kernel %dy, %x, %filter, %dx, %df {
      target = "x86", arch = "zen5-avx512",
      kind = "tessera.stft", axis = -1 : i64,
      logical_length = 8 : i64, normalization = "backward",
      spectrum_layout = "full_complex", center = false, onesided = true,
      pad_mode = "constant",
      window_broadcast = "trailing_batch_broadcast_v1",
      input_count = 3 : i64, output_count = 2 : i64,
      input_signature = "tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>",
      output_signature = "tensor<8xcomplex<f32>>,tensor<8xcomplex<f32>>",
      mutation_lineage = "inputs_immutable_outputs_fresh_v1",
      tessera.schedule_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    } : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
    llvm.return
  }
}

// CHECK: error: x86 compound spectral adjoint kind has no native package

// -----

// ── the ABI precondition, pinned separately ──
// The kind check is reached only after `arch == "zen5-avx512"` and a 64-hex
// content-addressed schedule hash. Writing the first module above with
// `arch = "x86-avx512"` — the spelling the rest of the tree uses — made it
// fail at THIS check instead, and the fixture would have passed while testing
// something else entirely. Both refusals are therefore pinned, so neither can
// stand in for the other.
module {
  llvm.func @wrong_arch(
      %dy: !llvm.ptr, %x: !llvm.ptr, %filter: !llvm.ptr,
      %dx: !llvm.ptr, %df: !llvm.ptr) {
    tile.spectral_backward_kernel %dy, %x, %filter, %dx, %df {
      target = "x86", arch = "x86-avx512",
      kind = "tessera.spectral_filter", axis = -1 : i64,
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
// CHECK: error: native x86 spectral adjoint requires the content-addressed Zen 5 AVX-512 ABI
