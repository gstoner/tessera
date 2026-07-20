// Backward demo using LSE (v1.3)
module {
  %lse = "tessera_attn.lse.load"() : () -> memref<128xf32>
  // toy gradient propagation using %lse (pseudo):
  // %grad = "tessera_attn.backward"(%lse, %dOut, %Q, %K, %V) : (...) -> (...)
}
