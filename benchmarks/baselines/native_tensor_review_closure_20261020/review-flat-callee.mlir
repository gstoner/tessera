module attributes {tessera.arch = "sm_120", tessera.target = "nvidia_sm120",
tessera.native.tensor_program = 1 : i64, tessera.native.tensor_program_target = "nvidia_sm120"} {
func.func @host(%a: memref<1x1xf16>) -> memref<1x1xf32>
attributes {tessera.native.tensor_program = 1 : i64} {
 %o = memref.alloc() : memref<1x1xf32>
 tessera_nvidia.program_invoke %a, %o {arch = "sm_120", callee = @kernel,
 input_count = 1 : i64, geometry = array<i64: 1, 1, 1, 32, 1, 1>, scalars = array<i64>} : memref<1x1xf16>, memref<1x1xf32>
 return %o : memref<1x1xf32>
}
llvm.func @kernel(%a: !llvm.ptr, %o: !llvm.ptr)
attributes {nvvm.kernel, tessera.native.program_buffer_types = [memref<1x1xf16>, memref<1x1xf32>],
tessera.native.program_geometry = array<i64: 1, 1, 1, 32, 1, 1>, tessera.native.program_scalars = array<i64>} {
 llvm.return
}
}
