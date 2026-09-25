// RUN: %trop --pass-pipeline='builtin.module(gpu.module(tessera-rocm-lower-vector-to-vector))' %s 2>&1 | FileCheck %s --implicit-check-not=error:
//
// tessera-rocm-executable runs this pass on each gpu.module ahead of
// convert-gpu-to-rocdl: the vector-to-vector stage of upstream
// convert-vector-to-llvm, which is what materializes a fixed-length
// `vector.create_mask` (convert-gpu-to-rocdl does not). It deliberately stops
// before that pass's partial LLVM conversion, which has no GPU address-space
// mapping and printed a spurious error on every workgroup memref -- so a masked
// load from LDS must come out with its mask materialized, still a vector op on
// the workgroup memref, and with no diagnostic.

gpu.module @kernels {
  gpu.func @masked_lds_load(%lds: memref<64xf16, #gpu.address_space<workgroup>>,
                            %n: index, %out: memref<8xf16>) kernel {
    %c0 = arith.constant 0 : index
    %pad = arith.constant dense<0.0> : vector<8xf16>
    %mask = vector.create_mask %n : vector<8xi1>
    %v = vector.maskedload %lds[%c0], %mask, %pad
        : memref<64xf16, #gpu.address_space<workgroup>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
    vector.store %v, %out[%c0] : memref<8xf16>, vector<8xf16>
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @masked_lds_load
// CHECK-NOT: vector.create_mask
// The mask is the lane-index vector compared against the broadcast bound.
// CHECK: %[[BOUND:.*]] = vector.broadcast %{{.*}} : i32 to vector<8xi32>
// CHECK: %[[MASK:.*]] = arith.cmpi sgt, %[[BOUND]], %{{.*}} : vector<8xi32>
// CHECK: vector.maskedload %{{.*}}, %[[MASK]], %{{.*}} : memref<64xf16, #gpu.address_space<workgroup>>
// CHECK-NOT: llvm.
// CHECK: gpu.return
