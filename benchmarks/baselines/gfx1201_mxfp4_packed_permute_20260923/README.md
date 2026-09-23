# gfx1201 packed-word E2M1 decode experiment

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-PERMUTE-DECODE-2026-09-23`. The packet is an exact-device
RX 9070 XT/gfx1201, matched HIP-event kernel comparison. It binds the timed
HSACO hash to the selected-symbol ISA census and passes bitwise BF16 output
agreement with the exact per-K32 route for both lossless-fold prefill shapes.
Separate device tests cover ragged N48/N80, lossy folds, every E2M1 code,
subnormal exponent differences, and reserved E8M0 zero blocks.

| Shape M×N×K | B-batched integer | B-batched permute | expanded folded | Radiance | permute / Radiance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256×5120×8704 | 224.0 µs | 178.1 µs | 168.8 µs | 137.8 µs | 1.29× |
| 1024×17408×5120 | 1398.6 µs | 1158.5 µs | 1116.9 µs | 888.4 µs | 1.30× |

The opt-in candidate performs the same per-word E2M1→E4M3 fold but uses
`v_perm_b32` to look up four nibbles at a time. Its packed magnitude table is
derived from Tessera's existing scalar E4M3 oracle, not copied verbatim from
Radiance: the tables differ at some subnormal rounding points for exponent
differences 9–11. The emitted timed symbol has eight static `v_perm_b32`,
75 `s_wait_loadcnt`, 110 VGPR, 25,600 LDS bytes, and no spills. Static ISA
counts are not dynamic instruction or memory-traffic measurements.

The benchmark gives both kernels the same pre-quantized FP8 E4M3 activation
bytes, packed E2M1 weights, E8M0 K32 codes, FP32 per-token scales, and BF16
output. It does not measure Radiance's vLLM custom-op dispatch, activation
quantization, optional AITER/libr4d fallback, or Tessera's host-array launch
path. Tessera's public `_submit_rocm_mxfp4_w4a8` currently loads the HSACO,
allocates five device buffers, copies four inputs, synchronizes, copies the
output back, frees the buffers, and unloads the module on every call. The
matched benchmark instead preloads modules and preallocates device buffers,
so the kernel result must not be described as serving/runtime parity.

In pinned Radiance revision `dfdfa383`, the serving call tree is
`process_weights_after_loading` (scale-plane transpose, row reference,
one-time fragment permutation) → `apply_weights` → registered
`mxfp4_linear` custom op → FP8 activation quantization if needed →
`_ext.launch` for the folded HIP kernel, with shape-dependent AITER/libr4d
fallbacks outside this forced benchmark route. The benchmark directly calls
`_ext.launch`, sets `RADIANCE_MXFP4_WPERM=1`, and feeds pre-quantized data;
for these two M values its launch dispatch selects the TN2 folded kernel.
The synthetic inputs exercise a restricted set of finite E4M3/E2M1 values
and lossless K32 exponent differences, not the full model distribution.

This remains a manual candidate. Before selector admission, separate
model-load residency from launch, test graph capture with stable device
buffers, compare end-to-end dispatch and activation quantization, then study
remaining A/B vector load and LDS staging differences on the timed ISA.
The exact K32 route stays the correctness default.
