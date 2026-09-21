# GFX1201 exact MXFP4 W4A8 proof

Owner: `ROCM-MXFP4-W4A8-1`. Sync:
`ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21`.

Recorded on Tajasarus, Radeon RX 9070 XT (`gfx1201`), with the ROCm LLVM 23
toolchain. `evidence.json` binds the source revision, compiler/toolchain
fingerprints, launch ABI, HSACO SHA-256 and size, code-object resources, and
the disassembled matrix instruction.

## Result

The scalar executable specification and the optimized WMMA route both pass the
independent exact-per-K32 oracle at `17x19x64` and `32x32x128`: **4 passed**.
The ragged row proves masked tile edges; the K=128 row crosses four independent
MX scale groups. Comparison is bit-exact after BF16 round-to-nearest-even, not
tolerance based.

The optimized loop emits exactly
`v_wmma_f32_16x16x16_fp8_fp8`, twice in the static K32 loop body. Packed E2M1
is converted exactly to E4M3 bytes, the two K16 results form one local FP32
partial, and the E8M0/per-token scale outer product is applied before that
partial joins the running accumulator. Independent A/B loads are emitted as a
batch before decode/packing and the dependent WMMAs.

| route | HSACO | SGPR | VGPR | LDS | scratch | wave | FP8 WMMA/body |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar exact oracle | 6,264 B | 22 | 26 | 0 | 0 | 32 | 0 |
| exact FP8 WMMA | 8,264 B | 32 | 94 | 0 | 0 | 32 | 2 |

LLVM 23's `hipcc --genco` returns a clang offload bundle on this installation,
not a raw ELF. The packager now identifies that container and extracts the
exact `hipv4-amdgcn-amd-amdhsa--gfx1201` image with the matched
`clang-offload-bundler`; it still refuses any unknown/non-ELF result.

## Promotion boundary

This packet promotes the two exact `gfx1201` launch ABIs into the owning-device
proof registry. It does not promote `gfx1200`, the approximate folded-row
policy, a public Graph dtype, or a generic compiler selector. The dedicated
MXFP4 package selector defaults to the proved WMMA mechanism. Tajasarus WSL exposes no
usable hardware counters, so the packet proves correctness, mechanism, and
resource viability—not throughput or counter attribution. The scalar route
remains the executable oracle; the WMMA route is the production mechanism.

Reproduce:

```sh
ROCM_PATH=/opt/rocm/core PYTHONPATH=python \
TESSERA_GFX1201_DEVICE_PROOF=1 python -m pytest \
  tests/device/rocm/test_mxfp4_w4a8_exact.py -q -s

ROCM_PATH=/opt/rocm/core PYTHONPATH=python python -m \
  benchmarks.rocm.record_gfx1201_mxfp4 \
  --output /tmp/gfx1201_mxfp4_evidence.json
```
