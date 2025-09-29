// RUN: tessera-opt %s -tessera-lower-tile-to-ptx | FileCheck %s
// TMEM allocate and tcgen05 path
%buf = tessera.tile.alloc_tmem memref<128x64xf32, #tmem<262144, 64>>
%acc = "tessera.tile.mma.tcgen05"(%a, %b -> %buf, 2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, memref<128x64xf32>) -> memref<128x64xf32>
// CHECK: asm "tcgen05.mma"
