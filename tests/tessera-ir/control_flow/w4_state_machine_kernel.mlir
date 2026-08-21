// W4-PRODUCT-1 — GenerateROCMStateMachineKernel: the bounded irreducible-CFG
// program-counter state machine that --tessera-autodiff-paired structurizes
// becomes per-thread gpu.func device kernels — the FORWARD and the generated
// BACKWARD alike. Each thread runs the whole machine on its own element
// (per-element PC; SIMT divergence carries data-dependent control flow), the
// structured-CFG digest is stamped onto the kernel so the execution row binds
// the exact CFG identity, and the cf.assert bound check becomes the STATUS
// output the host enforces. (On-device execution on gfx1151 is proven by
// tests/unit/test_rocm_state_machine_exec.py.)
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s --tessera-autodiff-paired | \
// RUN:   tessera-opt --generate-rocm-state-machine-kernel | FileCheck %s

module {
  func.func @irreducible(%enter_left: i1, %x: tensor<8xf32>)
      -> tensor<8xf32> attributes {tessera.autodiff = "reverse"} {
    %out = scf.execute_region -> tensor<8xf32> {
      %c0 = arith.constant 0 : index
      cf.cond_br %enter_left, ^bb1(%c0, %x : index, tensor<8xf32>),
                              ^bb2(%c0, %x : index, tensor<8xf32>)
    ^bb1(%i: index, %state: tensor<8xf32>):
      %next = "tessera.tanh"(%state) :
          (tensor<8xf32>) -> tensor<8xf32>
      cf.br ^bb2(%i, %next : index, tensor<8xf32>)
    ^bb2(%j: index, %right_state: tensor<8xf32>):
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %next_i = arith.addi %j, %c1 : index
      %continue = arith.cmpi slt, %next_i, %c2 : index
      cf.cond_br %continue,
          ^bb1(%next_i, %right_state : index, tensor<8xf32>),
          ^bb3(%right_state : tensor<8xf32>)
    ^bb3(%result: tensor<8xf32>):
      scf.yield %result : tensor<8xf32>
    } {tessera.structured_cfg.digest = "5555555555555555555555555555555555555555555555555555555555555555",
       tessera.structured_cfg.max_steps = 8 : i64}
    return %out : tensor<8xf32>
  }
}

// The host functions are tagged with the kernels that realize them.
// CHECK: func.func @irreducible(
// CHECK-SAME: tessera.rocm_kernel = "tessera_state_machine_irreducible"
// CHECK: func.func @irreducible__bwd(
// CHECK-SAME: tessera.rocm_kernel = "tessera_state_machine_irreducible__bwd"

// Forward kernel: (FLAGS, X, O, STATUS, N) ABI; digest bound; per-thread
// machine (scf.for over the PC steps) with the tensor slots scalarized to
// f32 and tessera.tanh translated to math.tanh.
// CHECK: gpu.module @tessera_state_machine_irreducible_mod
// CHECK: gpu.func @tessera_state_machine_irreducible(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK-SAME: tessera.structured_cfg.digest = "5555555555555555555555555555555555555555555555555555555555555555"
// CHECK: gpu.block_id x
// CHECK: gpu.thread_id x
// CHECK: scf.if
// CHECK: memref.load
// CHECK: scf.for {{.*}} -> (index, i1, index, f32, index, f32, f32, f32)
// CHECK: math.tanh %{{.*}} : f32
// CHECK: memref.store
// CHECK: gpu.return

// Backward kernel: (FLAGS, X, DOUT, DX, STATUS, N) — one more memref than the
// forward — same digest, recompute_all replay machine inside.
// CHECK: gpu.module @tessera_state_machine_irreducible__bwd_mod
// CHECK: gpu.func @tessera_state_machine_irreducible__bwd(%{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>, %{{.*}}: index) kernel
// CHECK-SAME: tessera.structured_cfg.digest = "5555555555555555555555555555555555555555555555555555555555555555"
// CHECK: math.tanh
// CHECK: memref.store
// CHECK: gpu.return
