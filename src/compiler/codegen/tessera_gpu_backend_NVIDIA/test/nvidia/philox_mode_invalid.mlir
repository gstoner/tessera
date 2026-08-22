// RUN: not %tnv %s 2>&1 | FileCheck %s

module {
  "tessera_nvidia.philox"() {name = "bad", mode = "stateful"} : () -> ()
}

// CHECK: error: 'tessera_nvidia.philox' op attribute 'mode' failed to satisfy constraint: supported NVIDIA Philox generation mode
