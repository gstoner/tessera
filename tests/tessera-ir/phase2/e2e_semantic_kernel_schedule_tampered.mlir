// RUN: not tessera-opt --tessera-schedule-to-tile %s 2>&1 | FileCheck %s

module attributes {tessera.target = "x86", tessera.arch = "zen5-avx512"} {
  func.func @x86_softmax(%x: tensor<3x17xf32>) -> tensor<3x17xf32> {
    %graph = tessera.softmax %x {
      axis = -1 : i64,
      schedule.artifact_hash = "c98ce43a6015a2b9606d0a73e839eabd57dca6fed3a952a20937b4706d955f8e"
    } : (tensor<3x17xf32>) -> tensor<3x17xf32>
    %scheduled = schedule.softmax %graph {
      accum = "f32", arch = "zen5-avx512",
      artifact_hash = "c98ce43a6015a2b9606d0a73e839eabd57dca6fed3a952a20937b4706d955f8e",
      axis = -1 : i64, exp_mode = "accurate", ftz = false,
      storage = "f32", workgroup_size = 2 : i64
    } : tensor<3x17xf32> -> tensor<3x17xf32>
    schedule.artifact {
      arch = "zen5-avx512",
      hash = "c98ce43a6015a2b9606d0a73e839eabd57dca6fed3a952a20937b4706d955f8e",
      numeric_policy = "f32->f32", shape_key = "family=softmax;storage=f32;axis=-1",
      tile = {workgroup_size = 1 : i64}
    }
    return %scheduled : tensor<3x17xf32>
  }
}

// CHECK: error: scheduled semantic-kernel policy was altered after hashing
