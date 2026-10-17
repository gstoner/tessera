# gfx1201 native typed E4M3 conversion

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1 / FRONTEND-IR-MEDIUM-1.
Sync: ROCM-NATIVE-E4M3-CAST-20261017.

The standard GPU converter leaves scalar FP32-to-E4M3 arith.truncf behind;
HSACO emission then fails. The existing Target-to-ROCDL pass now lowers this
cast to the archived RDNA4 RNE packed-conversion instruction, extracts the
byte and retains typed FP8 storage. The scope is exact gfx1201 GPU modules.
Host functions, gfx1151/gfx1200 and explicitly directed rounding are unchanged.

## Numerical and code-generation evidence

- check_native_cast.py records the diagnostic exact-device comparison through
  the checked C++ native owner. It is not a Python numerical backend.
- 266,560 owning-gfx1201 values have zero mismatches: all 65,536 FP16 bit
  patterns widened to FP32, 756 rounding-boundary values, 100,000 random FP32
  bit patterns, 100,000 uniform values, representable FP8 values and explicit
  overflow, infinity, NaN and signed-zero cases.
- Finite outputs, including signed zero, compare bitwise. NaNs compare
  semantically; no NaN sign/payload canonicalization claim.
- Changed inputs and retained outputs pass. Three correctness-gated captured
  windows exceed 10ms; per-iteration device graph dispatch is 3.650–3.680us
  for the entire 266,560-value array, excluding uploads, readback, graph capture,
  instantiation and CPU enqueue. No speedup or end-to-end quantizer claim.
- fp8-cast.s contains v_cvt_pk_fp8_f32. The authoritative opcode is in the
  archived RDNA4 instructions.json, not inferred from gfx1151 hardware.
- Final GPU-only scoping preserves the exact device-validated HSACO bytes.
  final-source-and-image-parity.json binds final source/compiler hashes.
- 19 final native/WMMA/parent-view cases pass. Dtype/target/cache/pass/
  diagnostic/audit drift gates pass 546 cases. Counts overlap.

## Existing-route controls

check_existing_images.py recompiles the original Qwen M19/M256 expansion,
product and activation members using frozen and candidate tools under the
same ROCm host SDK. All six payloads, Target digests, entries, executable
sections, AMDGPU metadata and device-library records match. Compiler-bound
artifact identities intentionally differ with the compiler fingerprint.

An earlier cross-host payload comparison was not accepted as no-change proof:
Target/entry matched but payload bytes differed. The same-host control is the
accepted isolation experiment. Reconciled checkout mtimes may trigger the
existing warning; source/tool hashes are bound independently and timestamps
were not manipulated.

## Remaining integration

This packet proves a typed GPU storage primitive. It does not prove a public
frontend Graph quantizer, K16 max/scale/normalize semantics, two-result package
ABI, resident activation quantization, down projection or full-model quality.
The existing quantize_fp8 fake-quant operation keeps its dequantized payload
contract. Those integration steps must preserve explicit semantic Graph IR,
native Schedule/Tile, native image and checked runtime ownership.
