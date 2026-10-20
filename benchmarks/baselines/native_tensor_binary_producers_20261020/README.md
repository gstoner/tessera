# Native SM120 binary tensor producers — 2026-10-20

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-BINARY-PRODUCERS-20261020. Base: merged PR997,
502f2322db3dc02573e63e605d5ecef82b2d0653.

Current review follow-through: [scalar-width/oracle closure](review-closure/README.md).
The initial receipts below remain historical; current runtime and oracle proof is
recorded there.

## Compiler and exact-device proof

Original Graph `add` on the left, right or both matmul operands, add/ReLU
chains and repeated-intermediate SSA now pass through the native C++
Graph -> Schedule -> Tile -> NVIDIA LLVM/NVVM/PTX route. Python supplies
frontend inputs, an independent oracle and measurements; it does not implement
the numerical backend. The shared native-frame v1 ABI is unchanged.
Static compact rank-two FP16/BF16/FP32 storage is admitted. This is a named
producer-family migration, not arbitrary/dynamic/pitched/fused/AD closure.

`frozen-source-identity.json` and `candidate-native.patch.gz` bind the six native
source changes and compiler 68358e312c1cd8623ebf6ba3b07aec0d0bd4e4dfb16298888474b8d491788312.
The gzip payload expands to the frozen patch SHA-256 in the identity receipt.
The frozen patch includes the 60-case compiler fixture; later recorder, device
fixture and documentation additions do not change that compiler. The final
packet records their current hashes independently.

Host WSL compiler proof: 60 focused contracts/ownership checks; 701 neighboring
scalar/attention/scaled and registry checks; 2,754 full production compiler-route
cases, zero skips, with the executed-case gate passing. Fifteen recorder tests
cover CLI hardware independence, capture limits and invalid timings.

Super-Bear RTX5070, compute 12.0, driver/toolchain identity in the packet:
99 compiler/device cases pass (60 compiler + 39 device, zero skips), including
ordinary JIT, serialized replay requiring the matching compiler, independently
pending resident producer streams, immutable inputs, independent output lifetime
and stable allocation reuse. Three further diagnostic projected-member tests
expose ties-to-even, signed zero, subnormals, cancellation, overflow, NaN/Inf
classification and output canaries before a matmul can erase producer bits.
That shortened frame is diagnostic transport, not standalone public add admission.

Final source-bound cohort: `binary-delivery-device.xml` executes all 117 cases
(60 compiler, 42 device and 15 recorder), zero skips. The adjacent identity
receipt binds all four current fixture/recorder files and the compiler. Earlier
99-case and three-case receipts above remain historical, not additional distinct
cases. All five surface audits and claim lint pass; 33 generated docs are in
sync, typing has zero errors, and the final Graphify refresh completes.

## Separate timing domains

Recorder: `benchmarks/nvidia/record_native_tensor_binary.py`.
Final packet: `binary-recorded-long-k.json`. Every row is M256/K4096/N256;
FP16/BF16 producer storage rounds at each store, then the matmul accumulates to
FP32. Independent FP64 arithmetic and the FP32 gamma-K forward bound validate
before and after timing without an absolute tolerance floor.

Six windows per mode alternate enqueued/captured order. Native captured repeats
respect the 4,096 limit. All 36 captured windows meet the minimum 10 ms regional
duration; all isolated regions are measured independently. Do not sum producer
and consumer medians to reconstruct whole-program time. Public calls include
binding, transfer, native execution and readback; compile/setup and numerical
checks are excluded. Eager execution, tracing and compiler subprocesses are
forbidden during measured warm public calls. No baseline speedup is claimed.

| Storage | Producer placement | Producer event medians (us) | Consumer event median (us) | Public median (ms) |
|---|---|---|---|---|
| f16 | lhs | 5.056 | 81.454 | 1.3220 |
| f16 | rhs | 5.126 | 81.830 | 1.3602 |
| f16 | both | 5.116, 5.118 | 81.771 | 1.9781 |
| bf16 | lhs | 5.118 | 81.916 | 1.3730 |
| bf16 | rhs | 5.117 | 81.801 | 1.2895 |
| bf16 | both | 5.115, 5.116 | 81.867 | 1.9332 |

## ODS inventory follow-through

The first full CPU run passed 30,841 cases with 3,721 declared skips and one
failure: its exact ODS count still expected 697 instead of 698. The added
`schedule.binary_kernel` now has an inventory entry, a parse pin and a shipped
compiler-consumer pin. All 783 affected ODS tests pass. The native compiler,
images, runtime and recorder are unchanged; the required full CPU retry passes 30,842 cases with 3,721
declared skips and no failures. `cpu-unit-initial-summary.json` preserves the initial failure scope.

## Unchanged-route and sibling assessment

`unary-image-comparison.json` proves eight FP16/BF16 ReLU/RMSNorm/two-sided/
reversed-root programs have identical source, frame, projected members and all
18 PTX images versus merged compiler 01922c4a. `image-attribution.json` compares
on gfx1201: the exact MXFP4 composed numerical case passes; its native images
and FP8 K128/K32, MXFP8 and approximate folded MXFP4 neighbor images are identical.
This is architecture-specific control evidence, not NVIDIA performance proof.

All four backend queues assess the shared internal Schedule/Tile changes. Apple
and x86 retain their existing owners and ABIs; no hardware proof transfers.
ROCm binary program admission and broader scalar exact-device policy parity
remain follow-up required. General layouts, dynamic/encoded producers, AD,
NVFP4 model-quality acceptance, attention and aggregate five-slice scope remain
open. Full delivery gates and final CI/review are recorded separately.

Reproduce on Super-Bear after sourcing the matching compiler and native provider:

```sh
python -m pytest tests/unit/test_native_tensor_binary_producers.py tests/device/nvidia/test_native_tensor_binary_producers.py --timeout=180
python -m benchmarks.nvidia.record_native_tensor_binary --output /tmp/native-binary.json
```

The compiler and runtime provider must match the source identities; selecting an
unrelated compiler is not an equivalent numerical or performance reproduction.

Final CPU receipt: `cpu-unit-final-summary.json` and `binary-full-unit.txt`.
