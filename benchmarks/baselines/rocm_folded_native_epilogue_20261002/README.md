# gfx1201 native full-K folded-scale epilogue

Owner ROCM-MXFP4-W4A8-1. Sync ROCM-FOLDED-NATIVE-EPILOGUE-2026-10-02.

## Implemented and proved

Internal Tile operation tile.fragment_folded_scale takes a stated f32
accumulator, rank-1 token f32 scales, raw E8M0 column references and logical
bounds. The verifier rejects incorrect accumulator/scale types. Its native
ROCm dialect-conversion pattern uses the architecture accumulator map,
keeps all scale loads in bounds and scales the full-K partial once.
It computes partial*(reference*token) in fp32. For a zero/nonfinite scale
product it uses ordered fp64 partial*reference*token, rounded to fp32;
zero partial with finite token scale yields positive zero. The bf16 store
then performs one final rounding. E8M0 code zero maps to zero and reserved
code 255 must be excluded by the checked payload loader.

A hand-authored native Tile/GPU fixture carries typed e4m3 views/fragments
through a four-step K64 MMA loop, this epilogue, and a bounded bf16 store.
It compiles through ROCm/ROCDL/LLVM to HSACO with no Python HIP shader.
RX 9070 XT/gfx1201 proof covers normal scales, overflowing/underflowing scale
products, zero partials, zero references and ragged 9x11 output bounds.
Bitwise bf16 comparison against an independent fp64 oracle passes all six.
The combined verifier/new-device/existing-W8A8 suite passes 110 tests.
Host WSL operation/dtype-attribute/diagnostic/pass gates pass 318 tests.

## Boundary and remaining integration

This is executable Tile/native epilogue proof, not closure of the folded
Graph-to-Target package. Current folded Target packaging still delegates
to Python-emitted HIP; it must migrate to the native LDS producer, carrying
its raster/prefetch/row guard/epilogue schedule and checked ABI explicitly.
Full package correctness, shape-independent image keys and production
kernel/end-to-end benchmarks remain required. No throughput, Radiance,
AITER or selector promotion claim is made from this small numerical fixture.
The full-K folded approximate policy stays distinct from exact per-K32 math.

Apple, NVIDIA and x86 share the Tile declaration/verifier but have no added
physical consumer or advertised frontend route. Sibling consumer parity is
follow-up work; no gfx1201 numerical result establishes it. gfx1151 cannot
run the FP8 WMMA producer under RDNA3.5.
