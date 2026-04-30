// RUN: cat %s | FileCheck %s
// This sample isn't executed by the wrapper here; it's a placeholder for your in-tree pipeline.
module {
  // Expect downstream pipelines to materialize:
  // CHECK: scf.for
  // CHECK: scf.for
  // And to choose the JVP path when requested:
  // CHECK: energy_bilinear_jvp
}
