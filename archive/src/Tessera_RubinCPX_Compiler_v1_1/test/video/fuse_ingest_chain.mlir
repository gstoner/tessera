
# RUN: %cpx_opt %s -tessera-fuse-video-ingest | FileCheck %s

module {
  %bs = tensor<1x1024xi8>
  %frames = "tessera.target.cpx.video.decode"(%bs, "h264" : !llvm.ptr<i8>)
            : (tensor<1x1024xi8>, !llvm.ptr<i8>) -> tensor<1x16x224x224xi8>
  %p = "tessera.video.patchify"(%frames) : (tensor<1x16x224x224xi8>) -> tensor<1x196x256xi8>
  %t = "tessera.video.tokenize"(%p) : (tensor<1x196x256xi8>) -> tensor<1x4096xi8>
  %kv = memref.alloc() : memref<1x4096x64xf16>
  %o = "tessera.target.cpx.attn.prefill_fused"(%t, %t, %t, %kv, arith.constant 8192 : i64)
        : (tensor<1x4096xi8>, tensor<1x4096xi8>, tensor<1x4096xi8>, memref<1x4096x64xf16>, i64)
          -> tensor<1x4096xi8>
  // CHECK: "tessera.target.cpx.video.ingest_fused"
}
