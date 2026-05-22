// RUN: tessera-opt %s -tessera-autotune-p3d -tessera-lower-p3d | FileCheck %s
module {
  %0 = memref.alloc() : memref<1x8x64x64x64xf16> // N,C,D,H,W
  %w = memref.alloc() : memref<8x8x3x3x3xf16>     // K,C,KT,KH,KW
  %b = memref.alloc() : memref<8xf32>

  %y = p3d.conv3d %0, %w, %b
        { stride = [1,1,1], dilation=[1,1,1], padding=[1,1,1] }
        : memref<1x8x64x64x64xf16>, memref<8x8x3x3x3xf16> -> memref<1x8x64x64x64xf16>

  %pyramid:2 = p3d.pyramid.build %y, [[2,2,2],[4,4,4]]
        : memref<1x8x64x64x64xf16> -> (memref<1x8x32x32x32xf16>, memref<1x8x16x16x16xf16>)

  %ctx = p3d.global_context %y : memref<1x8x64x64x64xf16> -> memref<1x8x64x64x64xf16>

  %up = p3d.upsample3d %ctx, [64,64,64], "trilinear"
        : memref<1x8x64x64x64xf16> -> memref<1x8x64x64x64xf16>

  return
}
// CHECK: module
