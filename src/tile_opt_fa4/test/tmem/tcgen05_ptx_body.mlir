// RUN: tessera-opt %s -tessera-lower-tile-to-ptx | FileCheck %s
// CHECK: tcgen05.mma
%acc = "tessera.tile.mma.tcgen05"(%a, %b -> %c, 2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> memref<64x64xf32>
