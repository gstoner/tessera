// RUN: %tessera_cerebras_opt < %s | FileCheck %s
// RUN: test -f %t/out.csl
// RUN: test -f %t/layout.json

ttarget.region {0, 0, 63, 63} {
  %a = "ttarget.alloc"() : () -> tensor<64x64xf16>
  %b = "ttarget.alloc"() : () -> tensor<64x64xf16>
  ttarget.copy %a -> %b { src_space = "global", dst_space = "sram" }
  ttarget.matmul %a, %b -> %b (64, 64, 64)
}

cerebras.emit { csl_out = "%t/out.csl", layout_out = "%t/layout.json" }

// CHECK: cerebras.region { 0, 0, 63, 63, 0 }
// CHECK: cerebras.load_sram
// CHECK: cerebras.matmul
// CHECK: CSL-EMIT: wrote
