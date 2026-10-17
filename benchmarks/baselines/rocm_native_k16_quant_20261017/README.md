# Native K16 quantizer Target and device proof

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1 / FRONTEND-IR-MEDIUM-1.
Sync: ROCM-K16-QUANT-SCHEDULE-20261017.

## Proven scope

Explicit BF16 semantic Graph SSA traverses native Schedule, typed Tile,
ROCm Target, GPU/ROCDL and LLVM serialization to a real gfx1201 HSACO.
Schema 4 retains its three-pointer/gid/group-count body. Target memory uses
BF16 input, byte-backed E4M3 codes and FP32 scales, with a bitcast at the code
storage boundary. The checked native frame exports two output lifetimes.
The launch adapter clones the numerical/addressing SSA and validates count,
flat memref offsets, strides, spans and block size before executing it.
It contains no quantization arithmetic recipe.

[build_images.py](build_images.py) is the frozen native image recorder;
[check_quantizer.py](check_quantizer.py) is the diagnostic exact-device oracle
and timing recorder. Both run from the Tessera source root. Their scratch
paths and the checked runtime provider are explicit. The checker supports
TESSERA_K16_QUANT_EVIDENCE_DIR for its image/output folder.

[target-image-records.json](target-image-records.json) pins tools, source
frames and image bytes. All three source envelopes share HSACO SHA256
`6be7bab64720e6d921fa6d2c8287eba2f3fd7f43a19739bcf7afd2a98afdaa97`;
shape-specific source and artifact seals remain distinct. This is byte reuse
proof, not a claim that the package cache already avoids every compilation.
[isa-excerpt.txt](isa-excerpt.txt) records native FP8 conversion and byte/FP32
stores; static ISA inspection is not hardware-counter attribution.

## Exact-device numerical result

[device-quantizer.json](device-quantizer.json) records RX 9070 XT, live gfx1201,
GPU UUID, checker hash, provider hash and compiler/image identities.
All three shapes have zero code and zero scale mismatches. The large case
includes every BF16 bit pattern repeated as a uniform K16 group, heterogeneous
bit-pattern groups, signed zeros, subnormals, maximum finite values, NaNs
and infinities. The medium and large cases also include 252 explicit FP8
midpoint groups, each anchored at scale=1 with adjacent BF16 values. Finite codes and signed zeros compare bitwise; NaNs compare
by classification. FP32 scales compare bitwise except NaN payload/sign.
Changed roots and retained independent outputs also pass.

The first timing attempt rejected a short captured window; adaptive repeat
calibration establishes at least 10 ms for every retained sample. No speedup
or default-route promotion is claimed. These synthetic diagnostic packets
are not a checkpoint MLP or model-quality result.

| Input shape | Captured member median (µs) | Prepared invoke + two reads median (ms) |
|---|---:|---:|
| 3 × 32 | 4.447 | 0.357 |
| 19 × 12288 | 5.026 | 0.316 |
| 256 × 12288 | 26.822 | 0.621 |

Device windows include graph dispatch and exclude capture/instantiation,
CPU enqueue and copies. Completed host calls exclude compilation/preparation
and include both checked output readbacks. These columns measure different
scopes and are not interchangeable.

## Tests and remaining gates

[target-final-gates.txt](target-final-gates.txt): **532 passed** across the
native, transport, canonical dtype, tensor-attribute, diagnostic, pass-metadata
and audit gates. This includes the narrower counts below; they are not additive.


[target-tests.txt](target-tests.txt): 44 native quantizer and adjacent
parent-view cases pass, including real HSACO serialization, a changed frame
and sibling architecture refusal. [target-read-tests.txt](target-read-tests.txt):
26 checked C++ transport cases pass, including mixed FP8/FP32 outputs through
both single and batched completion, with pinned and ordinary readback.

[target-nvidia-tests.txt](target-nvidia-tests.txt): twelve fresh exact RTX5070
producer/ragged/ReLU controls pass with matching rebuilt tools. This is sibling
regression evidence; it does not transfer HIP proof to NVIDIA.
[target-recorder-naming-final.txt](target-recorder-naming-final.txt) confirms
the frozen recorders are named by their tracked README.
[target-doc-gates.txt](target-doc-gates.txt) retains the original 87 passing
documentation/registry checks and the earlier naming failure while the new
README was untracked; the separately linked naming rerun passes after staging.
The full generated-document drift check passes without derived-file changes.

Still required: canonical quantizer family/ABI packaging, authored public
JIT capture and portable replay, resident quantized activation/down-projection
chaining, full native MLP and checkpoint/model-quality acceptance. The device
recorder uses a diagnostic subclass of the existing generic C++ multi-output
owner; it is not evidence of those public integrations. No sibling backend
physical execution is claimed. Existing attention schemas are unchanged.
