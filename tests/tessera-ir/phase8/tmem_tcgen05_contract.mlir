// RUN: FileCheck %s < %s
//
// Hardware-free Blackwell TMEM contract fixture. Native execution is not
// claimed by this file; it records the target gate and emitted PTX contract
// spelling expected from the artifact path.

// CHECK: tcgen05.mma.cta_group::2
// CHECK: tessera_acc_tmem
// CHECK: tessera_a_desc
// CHECK: tessera_b_desc
// CHECK: requires target/arch containing sm100

module attributes {target = "nvidia_sm100", arch = "sm_100"} {
  func.func @tcgen05_contract() {
    "tessera.tile.mma.tcgen05"() {
      shape = "m128n128k32",
      accum = "tmem_f32",
      cta_group = 2 : i64
    } : () -> ()
    return
  }
}
