// RUN: not tessera-opt --split-input-file %s 2>&1 | FileCheck %s
// REQUIRES: tessera-apple-backend
//
// W1.1 step 6 — the Apple runtime-call ops state their value domain.
//
// `tessera_apple.{cpu.call, gpu.kernel_call, gpu.package_call}` took
// `Variadic<AnyType>`. Those ops lower to an Accelerate/LAPACK entry point or a
// compiled MSL kernel, and those take BUFFERS OF DATA — so `AnyType` admitted
// operands that are meaningless at that boundary (a `!tile.fragment`, an async
// token, a symbol ref) with the ODS saying nothing about it.
//
// Scalar parameters are not operands on these ops; they ride as attributes
// (`lower`, `trans`, `unit_diag`, `dtype`, …), which is why the domain can be
// buffers-only without losing expressiveness. All 303 existing fixtures pass
// unchanged, so no producer was relying on the wider set.
//
// The control-flow ops (`gpu.control_{if,loop,while}`) deliberately KEEP
// `AnyType`: they carry `iter_args`, which are legitimately polymorphic just as
// `scf.for`'s are. That is documented on those ops so a later audit does not
// read them as leftovers.

func.func @scalar_is_not_a_buffer(%x: index) -> index {
  // CHECK: 'tessera_apple.cpu.call' op operand #0 must be variadic of a ranked
  // CHECK-SAME: tensor or memref data buffer
  %0 = tessera_apple.cpu.call %x
      {op_kind = "gemm", symbol = "s", abi = "cblas_sgemm", status = "executable"}
      : (index) -> index
  return %0 : index
}

// -----

func.func @kernel_call_result_must_be_a_buffer(%a: tensor<4x4xf32>) -> i32 {
  // A scalar RESULT is equally meaningless — the kernel writes a buffer.
  // CHECK: 'tessera_apple.gpu.kernel_call' op result #0 must be variadic of a
  // CHECK-SAME: ranked tensor or memref data buffer
  %0 = tessera_apple.gpu.kernel_call %a
      {op_kind = "gemm", symbol = "k", status = "executable"}
      : (tensor<4x4xf32>) -> i32
  return %0 : i32
}
