
// RUN: tessera-opt %s -tessera-atlas-window-plan -tessera-atlas-featuremap-lower -tessera-atlas-memory-lower | FileCheck %s
module {
  // Create memory
  %mem = "atlas.memory.create"(%c1024_i64, %c256_i64, %c4_i64) : (i64, i64, i64) -> !atlas.memory
  // Feature map keys/queries (poly degree=3)
  %Kf = "atlas.feature.map"(%K) {kind = "poly", degree = 3} : (tensor<?x256xf16>) -> tensor<?xMxf16>
  %Qf = "atlas.feature.map"(%Q) {kind = "poly", degree = 3} : (tensor<?x256xf16>) -> tensor<?xMxf16>
  // Sliding-window update with W=4096
  %mem2 = "atlas.memory.update"(%mem, %Kf, %V, 4096) : (!atlas.memory, tensor<?xMxf16>, tensor<?xNxf16>) -> !atlas.memory
  // Read
  %out = "atlas.memory.read"(%mem2, %Qf) : (!atlas.memory, tensor<?xMxf16>) -> tensor<?xNxf16>
  // CHECK: atlas.memory.read
}
