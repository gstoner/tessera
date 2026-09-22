# gfx1201 MXFP4 K-step and fragment-prefill evidence

This packet was recorded on Tajasarus (`AMD Radeon RX 9070 XT`, `gfx1201`) at
source revision `39605e560415cd629bd5d1f1769cc9980e81f4ef`. It uses HIP-event
device timing, rotates three weight copies, checks every engine against an
independent FP32-dequantized oracle before timing, and alternates engine order
within each shape to limit clock and thermal bias.

The selected Tessera route records the exact HSACO's FP8-WMMA count, waits,
barriers, VGPRs, SGPRs, LDS, scratch, and spills. The archive-derived RDNA4
counter observes two split barrier instructions in decode and four in prefill.
Every case emits two `v_wmma_f32_16x16x16_fp8_fp8` instructions and uses zero
scratch/spills. The two independently built comparison binaries are
content-bound in `evidence.json` to Radiance revision
`dfdfa3832922c9a4253133f09c1f5c0d39748fc7` and libr4d revision
`5dc6302b87d598d1d3bf2ad3b50aab365461a63c`.

| case | Tessera ms | Radiance ms | Tessera / Radiance | libr4d ms |
|---|---:|---:|---:|---:|
| decode 8x5120x8704 | 0.048607 | 0.036830 | 1.32x | 0.038864 |
| decode 8x17408x5120 | 0.087689 | 0.086170 | 1.02x | 0.107830 |
| prefill 256x5120x8704 | 0.514184 | 0.151350 | 3.40x | not admitted |
| prefill 1024x17408x5120 | 4.380297 | 0.910922 | 4.81x | not admitted |

The exact-route tuning loop also rejected the two-stage producer/consumer
pipeline (13–15% slower) and streaming cache policy (1.4–3.5% slower). Group-M,
waves-per-EU, and 0/1/2/4-dword LDS padding differences stayed below a stable
promotion threshold. These measurements do not authorize an approximate
row-reference fold; that requires its own numerical policy and ABI.

See [evidence.json](evidence.json) for raw samples and provenance.
