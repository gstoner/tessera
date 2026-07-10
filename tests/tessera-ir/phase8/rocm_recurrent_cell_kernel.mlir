// generate-rocm-recurrent-cell-kernel expands a tessera_rocm.recurrent_cell
// directive into a fused single-step simple_rnn / gru gpu.func (one thread per
// output element; the two gate GEMMs + elementwise gate math fused). The exact
// ABI (arg count/order) + numerics are proven on gfx1151 by
// tests/unit/test_rocm_recurrent_cell_compiled.py; this fixture pins the emitted
// kernel STRUCTURE GPU-free. Distinct kernel names keep the split blocks'
// CHECK-LABELs from colliding.
//
// REQUIRES: tessera-rocm-backend
// RUN: tessera-opt %s -split-input-file --generate-rocm-recurrent-cell-kernel \
// RUN:   | FileCheck %s

// ─── simple_rnn (tanh): two dot-product loops, ending in tanh ────────────────
// CHECK-LABEL: gpu.func @rnn_tanh
// the Σ_i x·Wih loop then the Σ_k h·Whh loop (each an f32 accumulator)
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (f32)
// CHECK: arith.mulf
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (f32)
// CHECK: math.tanh
// CHECK: gpu.return
"tessera_rocm.recurrent_cell"() {name = "rnn_tanh", cell = "simple_rnn", dtype = "f32", act = "tanh"} : () -> ()

// -----
// ─── simple_rnn (relu): activation is max(x, 0), no tanh ─────────────────────
// CHECK-LABEL: gpu.func @rnn_relu
// CHECK: arith.maximumf
// CHECK-NOT: math.tanh
"tessera_rocm.recurrent_cell"() {name = "rnn_relu", cell = "simple_rnn", dtype = "f32", act = "relu"} : () -> ()

// -----
// ─── gru: two 3-gate loops, two sigmoids (exp) + tanh + (1−z) ────────────────
// CHECK-LABEL: gpu.func @gru
// x-side and h-side loops each carry 3 gate accumulators
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (f32, f32, f32)
// CHECK: scf.for {{.*}}iter_args({{.*}}) -> (f32, f32, f32)
// z = σ(..), r = σ(..) → two exps; n = tanh(..); h' uses (1 − z)
// CHECK: math.exp
// CHECK: math.exp
// CHECK: math.tanh
// CHECK: arith.subf
// CHECK: gpu.return
"tessera_rocm.recurrent_cell"() {name = "gru", cell = "gru", dtype = "f32", act = "tanh"} : () -> ()

// -----
// ─── f16 storage: kernel args are memref<?xf16>, compute still f32 (ext/trunc) ─
// CHECK-LABEL: gpu.func @rnn_f16
// CHECK-SAME: memref<?xf16>
// CHECK: arith.extf
// CHECK: arith.truncf
"tessera_rocm.recurrent_cell"() {name = "rnn_f16", cell = "simple_rnn", dtype = "f16", act = "tanh"} : () -> ()
