# Bounded independent Quark W4A4 projection reference

The pinned AMD Qwen3.8-27B checkpoint's first 32 K elements in two rows
of each gate/down projection are recorded in `reference.json`, with absolute
HTTP byte offsets. This extends the byte-bound [preflight](../gfx1201_quark_byte_oracle_20260923/README.md).
The packed activation is a synthetic all-code E2M1 vector with E8M0 code
127; it is **not** an activation captured from the Quark model.

An isolated Torch 2.14.0 process ran the independently maintained SGLang
`MXFP4QuantizeUtil.dequantize` on those exact packed planes. The source is
[pinned at commit 28262c2](https://github.com/sgl-project/sglang/blob/28262c20df6f2945c1b1cd7b27b0faef59cb0438/python/sglang/srt/layers/quantization/mxfp4_tensor.py)
and SHA-256-bound in the JSON. Multiplying its dequantized activation and
weights produced the recorded FP32 values and BF16 bits. Tests compare
Tessera's separately implemented byte oracle to those constants without
depending on SGLang, Torch, network access, or a downloaded checkpoint.

This is a bounded independent *projection-slice* arithmetic reference, not
matching Quark 0.13 exporter source, a whole-checkpoint layout certificate,
or proof of dynamic activation quantization. E8M0 codes 0 and 255 remain
unresolved and the opt-in W4A4 probe rejects them. The W4A8 and automatic
selection routes stay closed to Quark. The scalar gfx1201 probe is a
correctness carrier, not a production schedule or performance claim.

The [exact-device evidence](evidence.json) binds the current generator,
fixtures, reference packet, and each *launched* HSACO payload. HIP builds
embed build-specific data, so fresh recompilations need not have the same
payload digest. On
Tajasarus's AMD Radeon RX 9070 XT (`gfx1201`), the three opt-in BF16
device cases passed; 61 focused host unit/audit tests also passed. The
recorded LLVM version is ROCm's development 23.0.0git build, not a claim
that this host ran an official 23.1.1 release.
