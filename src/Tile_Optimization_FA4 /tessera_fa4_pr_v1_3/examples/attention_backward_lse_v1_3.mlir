// Backward demo using LSE (v1.3)
module {
  %lse = "tessera.attn.lse.load"() : () -> memref<128xf32>
  // toy gradient propagation using %lse (pseudo):
  // %grad = "tessera.attn.backward"(%lse, %dOut, %Q, %K, %V) : (...) -> (...)
}
