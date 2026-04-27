
// RUN: %cpx_opt %s -tessera-fuse-video-ingest | FileCheck %s
// CHECK: module
module {
  func.func @video(%bs: tensor<1x1024xi8>) -> tensor<1x16x224x224xi8> {
    %frames = "tessera.target.cpx.video.decode"(%bs) {codec = "h264"} :
              (tensor<1x1024xi8>) -> tensor<1x16x224x224xi8>
    return %frames : tensor<1x16x224x224xi8>
  }
}
