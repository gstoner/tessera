// RUN: %tnv %s | FileCheck %s
//
// The FULL Python-emitted tessera_nvidia surface parses as REGISTERED, WITHOUT
// --allow-unregistered-dialect (Codex review on PR #371). The C++ lowering only
// produces the inner contract ops, but target_ir.py / value_target_contract.py
// also emit the `func` wrapper, `profiler_probe`, and `kernel_call`; typing them
// keeps the whole surface parseable now that the dialect is not `isExtensible`.

module {
  "tessera_nvidia.func"() ({
    tessera_nvidia.cuda_kernel
        {arch = "sm_90a", kernel = "flash_attn_contract",
         source = "tessera.flash_attn", status = "artifact_only"} : () -> ()
    tessera_nvidia.profiler_probe
        {kernel_id = "flash_attn", measurement = "wall_clock_pending"} : () -> ()
  }) {sym_name = "flash"} : () -> ()

  func.func @call(%a: f32, %b: f32) -> f32 {
    %r = tessera_nvidia.kernel_call %a, %b
        {callee = "tessera_nvidia_flash", arch = "sm_120"} : (f32, f32) -> f32
    return %r : f32
  }
}

// CHECK: tessera_nvidia.func
// CHECK: tessera_nvidia.cuda_kernel
// CHECK: tessera_nvidia.profiler_probe
// CHECK: tessera_nvidia.kernel_call
