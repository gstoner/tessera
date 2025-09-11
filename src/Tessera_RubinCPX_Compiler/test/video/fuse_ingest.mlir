
// RUN: %cpx_opt %s -tessera-fuse-video-ingest | FileCheck %s
// CHECK: module
module {
  %bs = tensor<1x1024xi8>
  %frames = "tessera.target.cpx.video.decode"(%bs, "h264" : !llvm.ptr<i8>) :
             (tensor<1x1024xi8>, !llvm.ptr<i8>) -> tensor<1x16x224x224xi8>
  // ... patchify -> tokenizer -> attn.prefill_fused (omitted)
}
