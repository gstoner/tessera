// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-x86)' %s | FileCheck %s --check-prefix=X86
// RUN: tessera-opt --allow-unregistered-dialect --pass-pipeline='builtin.module(tessera-lower-to-gpu)' %s | FileCheck %s --check-prefix=NVIDIA
// RUN: tessera-opt --allow-unregistered-dialect --tessera-compute-legalize --tessera-storage-legalize --tessera-storage-pack-consume %s | FileCheck %s --check-prefix=EXPLICIT-PACK
//
// CORE-COMPILER-2: target defaults are capability-driven. x86 and NVIDIA run
// compute legalization. SM90 leaves terminal packing disabled; SM120 enables
// it only for operation families with a registered physical consumer. The
// standalone passes remain inspectable explicitly. ROCm's architecture-owned
// default is covered in its backend lit suite.

module {
  func.func @target_dtype_defaults() {
    "test.dtype_request"() {numeric_policy = {storage = "int4"}}
        : () -> ()
    return
  }
}

// X86: "test.dtype_request"
// X86-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// X86-NOT: tessera.storage_pack

// NVIDIA: "test.dtype_request"
// NVIDIA-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// NVIDIA-NOT: tessera.storage_pack

// EXPLICIT-PACK: "test.dtype_request"
// EXPLICIT-PACK-SAME: numeric_policy = {accum = "int32", storage = "int4"}
// EXPLICIT-PACK-SAME: tessera.storage_pack = #tile.packed_format<logical = "int4", container = "int8", logical_bits = 4, elements_per_container = 2, signedness = "signed_twos_complement", encoding = "twos_complement", lane_order = "low_to_high">
