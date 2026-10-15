# Native gfx1201 NVFP4 expansion primitive

Owner: ROCM-NVFP4-INGEST-1. Synchronization: NVFP4-NATIVE-EXPAND-20261015.

The registered semantic operation produces exact E4M3 representations of all
E2M1 codes, including both zeros, and a transposed FP32 K16 scale plane.
The policy declares a single FP64-product-to-FP32 rounding step.
Graph → Schedule → Tile → ROCm Target IR → LLVM → HSACO owns all arithmetic.
The checked package validates source scales/globals and private output ownership.

## Evidence

- Matching core and ROCm tools rebuilt in host WSL.
- 17 reference/import/result-shape cases passed.
- 326 operator/dtype-attribute/diagnostic/pass registry cases passed.
- 53 combined reference/native-route/device/legacy-ingest/package cases passed,
  including 10 owning-device
  cases on gfx1201. Signed zero, odd K16 groups, scale-plane order, FP32
  subnormals and ties-to-even are checked bitwise.
- The initial 3 failures were erroneous fixture assertions looking for textual
  arithmetic inside the final serialized binary, after both numerical output
  comparisons had passed. The corrected fixture retains both bitwise checks;
  arithmetic IR assertions remain in the separate compiler-route tests.
- 353 backend manifest, shape, dialect, canonical dtype, audit and generated-doc cases passed.
- Source and matching tool hashes are in source-manifest.json.
- Raw windows, node counts, clock/event agreement, source/input/image digests,
  and compressed native ISA are preserved beside this file.

## Characterization

All inputs here are seeded synthetic arrays, including the merged gate/up
physical size. This packet does not contain real checkpoint execution proof.

| N × K | device graph median | completed host-buffer median |
| --- | ---: | ---: |
| 7 × 48 | 0.003047 ms | 3.824 ms |
| 128 × 4096 | 0.003746 ms | 4.554 ms |
| 24576 × 4096 | 0.695189 ms | 55.735 ms |

Each device window is longer than 20 ms, uses warm repeated resident inputs,
and agrees with HIP events within 5%. Device measurements include GPU graph
dispatch and exclude allocation, validation, transfers, compilation, and graph
construction. Host measurements include validation, staging, allocation,
submission, transfer back, synchronization, and cleanup; compilation is separate.
Both outputs are checked bitwise before and after timing.

## Open integration gates

Public @jit packaging/serialized replay, resident producer-to-matmul composition,
native ownership of the complete launch sequence, real checkpoint parity, and
end-to-end model quality remain open. No default-route promotion, sibling GPU
proof, packed consumer execution, or overall five-slice closure is claimed.
The capability/manifest remains artifact_only pending public-route integration.

The final recorder constructs Graph IR through catalog multi-result inference;
its source hashes and raw log name end in the final measurement epoch.
