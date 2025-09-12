
// Sketch: Spectral norm layer
%plan = "tessera_spectral.plan"() {axes=[-1], elem_precision="fp8_e4m3",
                                   acc_precision="f32", scaling="blockfp_per_stage",
                                   inplace=false, is_real_input=false, norm_policy="backward"} : () -> !any
// %y = spectral_norm(%x) ≈ ifft( normalize( fft(x) ) )
"tessera_spectral.fft"(%plan, %x, %fx) : (!any, memref<?xcomplex<f32>>, memref<?xcomplex<f32>>) -> ()
