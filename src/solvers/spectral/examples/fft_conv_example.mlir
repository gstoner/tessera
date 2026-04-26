
// Sketch of FFT-based convolution using tessera_spectral.conv_fft
%plan = "tessera_spectral.plan"() {axes=[0,1], elem_precision="fp16",
                                   acc_precision="f32", scaling="blockfp_per_stage",
                                   inplace=false, is_real_input=false, norm_policy="backward"} : () -> !any
"tessera_spectral.conv_fft"(%plan, %image, %kernel, %out)
    : (!any, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
