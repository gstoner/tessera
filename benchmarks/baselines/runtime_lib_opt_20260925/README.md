# Runtime libraries without `-O` in empty-build-type trees — evidence, 2026-09-25

Owner `RUNTIME-LIB-OPT-1`; sync `RUNTIME-LIB-OPT-1-2026-09-25`. This directory
holds the raw output behind the queue entries in
`docs/audit/backend/{x86,rocm,nvidia,apple}/todo.md`. No build tree was
reconfigured.

## Method

- `scripts/inventory_build_types.sh` reports each build tree's
  `CMAKE_BUILD_TYPE` and the optimization flags each runtime library's compile
  commands actually carry (`ninja -t commands`). Its output per box is
  `inventory_{mac,luna,taj,bear}.txt`.
- `scripts/rebuild_library_with_flags.py <tree> <lib> <outdir> [flags]` rebuilds
  one library from its tree's own commands into a separate directory. With no
  flags the copy was **byte-identical** to the tree's library (checked with
  `cmp` for the x86 library on Princess-Luna). So an optimized copy differs
  only in the flag, and it is loaded through the runtime's path override.

## x86 AVX-512 (Princess-Luna, Zen 5, head `017a54d8`)

`libtessera_x86_elementwise.so` as built (no `-O`) vs `-O3`, loaded through
`TESSERA_X86_ELEMENTWISE_LIB`, 9 trials each:

- `benchmarks/x86/benchmark_x86_e2e_elementwise.py` →
  `x86_elementwise_{o0,o3}.json`. The script exits 1 in both builds; that is
  its own retained-vs-compiler non-regression gate, not an error.
- `benchmarks/x86/benchmark_x86_e2e_real_matmul.py` →
  `x86_real_matmul_{o0,o3}.json`.
- Medians: `x86_medians.txt`. At 1024×1024 the retained kernels run
  2.1–3.3× faster and the compiler route and matmul about 1.3× faster. The
  130 and 32×257 shapes are within noise at this trial count.

## ROCm gfx1151 (Princess-Luna)

`rocm_device_isa_gfx1151.txt` compiles `SpectralComposite.hip` device-only
from `build/`'s own command, with and without `-O2`. The results are 48,703 vs
7,584 device instructions and 48 vs 0 kernels with non-zero `ScratchSize`
(max 1,248 bytes). The same file shows `clang++ -###` passing no `-O` to the
amdgcn `cc1` when none is given.

The latency effect (STFT VJP 15.0 → 3.9 ms, ISTFT VJP 23.7 → 4.5 ms) is the
11-repeat experiment recorded as such in
`benchmarks/baselines/rocm_spectral_20260925/README.md` ("Build type"). It is
not a packet.

## NVIDIA (The-Super-Bear, RTX 5070)

`tessera_nvidia_spectral.cu` built as-is vs `-O3` (nvcc `-O` is host-only)
gave the same SASS (9,048 instructions in both, `cuobjdump -sass`). Only the
host object changed (16,918 → 19,844 x86 instructions, `objdump -d`). The
latency effect is in `benchmarks/baselines/nvidia_spectral_20260925/README.md`.

## Apple (Mac M1 Max)

`libTesseraAppleRuntime.dylib` was built as-is (1.80 MB) and at `-O2`
(1.15 MB), loaded through `TESSERA_APPLE_GPU_RUNTIME_LIB` (confirmed with
`vmmap`). Apple GPU matmul, add and softmax at 64² and 512², 200 calls each,
two alternating runs per build, gave medians within run-to-run noise. For
example, matmul 512 was 0.378/0.419 ms as-is vs 0.400/0.378 ms at `-O2`. These
were console-only runs (not a packet). The conclusion is "no measurable effect
on small dispatch", not a speed figure.

## Correction

A first count of "76 kernels using scratch" double-matched two metadata fields
per kernel (`.private_segment_fixed_size` and `ScratchSize`). The count is 48.
