
# TilingInterface Notes

This drop wires the ODS to declare `TilingInterface` for `matmul` and `conv2d_nhwc`,
and provides C++ **scaffolding** with TODOs to compute tiles via `tensor.extract_slice`.
The default implementation returns `failure()` so upstream passes won't use it until
you flip `-DTESSERA_ENABLE_TILING_INTERFACE=ON` and fill in the TODOs.
