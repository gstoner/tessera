// E2E-REAL-6 Apple GPU domain value families (2026-11-04): a Schedule record
// whose contract no longer matches its Graph parent is refused at Schedule ->
// Tile, before any Target IR is produced (NativeAppleGPUDomain.h).
// RUN: tessera-opt --tessera-graph-to-schedule %s -o %t.sched
// RUN: sed -e 's/noise_scale = 3.000000e-01 : f64}/noise_scale = 9.000000e-01 : f64}/' %t.sched > %t.tampered
// RUN: not grep -c 'noise_scale = 3.000000e-01 : f64}' %t.tampered
// RUN: not tessera-opt --tessera-schedule-to-tile %t.tampered 2>&1 | FileCheck %s

module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7", tessera.launch_bindings = ["y", "g", "z", "out"]} {
  func.func @langevin(%y: tensor<3x4xf32>, %g: tensor<3x4xf32>, %z: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tessera.ebm.langevin_step %y, %g, %z {eta = 1.250000e-01, noise_scale = 3.000000e-01} : (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK: error: Apple GPU native domain kernel Schedule contract was altered
