# gfx1201 native folded MXFP4 package

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-NATIVE-PACKAGE-2026-10-02.

Authored typed Graph -> Schedule -> Tile -> ROCm Target -> compiler-generated
LDS views / typed fragments / full-K WMMA / folded scale / bf16 store ->
ROCDL/LLVM -> native image -> checked runtime ABI -> RX 9070 XT execution.
The approximate folded policy is explicitly requested; this does not replace
the exact per-K32 contract. The frozen Python HIP emitter is the independent
matched control, not the native route's materializer.

The logical five-buffer ABI explicitly identifies the native image's expanded
memref presentation (28 physical arguments). Image pipeline and metadata
must agree before HIP probing or allocation. Legacy HIP uses eight physical
arguments. Both existing folded benchmark launchers use the checked adapter.

## Correctness and timing

Five randomized package shapes, including ragged M/N, match a decoded fp64
oracle and the matched HIP control bit-for-bit. Six authored Tile cases
separately cover overflow, underflow, zero and ragged folded-scale semantics.
Malformed argument-layout rejection tests pass before HIP probing.
Combined native Target/Tile/ABI adapter and folded device gate: 72 passed.
Host WSL frontend/schedule/diagnostic/pass gates: 320 passed.
The retained receipt is device-tests.txt.

Production timing uses M=256, K=5120, seven alternating trials and three
rotating resident copies. Every output matches HIP bit-for-bit and independent
sampled dequantized reference values before timing. Compiler-built device-clock
windows exceed 5 ms and agree with their HIP-event witness within 5%.
Windows include host dispatch gaps, not isolated instruction-phase measurements.
Public runtime wall time includes transfers, allocation and module loading.

| M | N | K | Native window us/launch | HIP control us/launch | Native/HIP | Native public ms |
|---|---|---|---|---|---|---|
| 256 | 4096 | 5120 | 74.76 | 69.31 | 1.0787 | 16.91 |
| 256 | 8192 | 5120 | 147.25 | 143.19 | 1.0284 | 43.25 |
| 256 | 16384 | 5120 | 286.37 | 264.73 | 1.0817 | 82.49 |

Native is 2.8–8.2% slower in this packet: 177 VGPRs versus 123 in HIP,
25,600 bytes LDS in both, no spills. Selected-symbol ISA contains three barrier
signal/wait pairs in native versus two in HIP. These are measured resource
differences and candidate causes, not profiler attribution. Identical selected
schedule keys do not guarantee identical instruction schedules.
No default performance promotion or Radiance result is claimed.

Image keys remain shape-dependent. Reduce register pressure and unnecessary
staging/barrier work, then rerun numerical and paired timing gates.
Model/checkpoint quality and wider scale/layout coverage remain open.

## Reproduce

From the owning WSL checkout, source scripts/_rocm_env.sh, expose the matching
compiler/LLVM dependencies, then run:

    python benchmarks/rocm/record_gfx1201_folded_native_package.py --tessera-opt "$TESSERA_OPT" --llvm-bin "$LLVM_BIN" --output benchmarks/baselines/rocm_folded_native_package_20261002/gfx1201.json

Packet records actual live architecture/device, dirty source fingerprints,
compiler hash, image identity, selected-symbol ISA/resources, timing windows
and separate frontend/package/public-launch costs. It binds to its recorded
revision; a subsequent stricter Target option-admission change does not relabel
the original image measurements as rerun.

Sibling assessment: gfx1151 cannot consume this FP8 WMMA producer under
RDNA3.5. Apple, NVIDIA and x86 need a physical folded-scale consumer only if
they admit the internal operation; no sibling execution parity is claimed.
