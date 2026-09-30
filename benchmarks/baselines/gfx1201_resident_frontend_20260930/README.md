# gfx1201 public frontend resident RMSNorm -> matmul

This packet measures the explicit public `output_dtype="fp32"` frontend route:

`from_text tracer -> Graph IR -> gfx1201 Schedule/Tile -> two HSACO images -> resident HIP session`

The test checks an fp16 RMSNorm result materialized in one resident allocation, then consumed by fp16 matmul with fp32 accumulation/output. Every measured output is compared with an independent NumPy fp32 RMSNorm / f16 edge / fp32 matmul oracle. Producer and consumer HIP-event intervals are reported separately. The session reuses the same packages, stream, and allocations. The static packet is in [gfx1201.json](gfx1201.json), and [dynamic_m.json](dynamic_m.json) adds a bounded M=128 package reused at active M=64 and M=128. All packets record image digests, compiler/toolchain fingerprints, buffer addresses, the branch revision, and a dirty-diff digest.

Owning device: Tajasaurus RX 9070 XT, gfx1201. Compiler: branch-local `tessera-opt` rebuilt from this checkout using the existing `.build-current` configuration and LLVM/MLIR 23.1.1. The packet is diagnostic and does not promote a schedule: it is one host/run and timings varied across repetitions and reruns.

Reproduce from the repository scratch checkout:

```sh
source scripts/_rocm_env.sh
export PYTHONPATH=python TESSERA_BUILD_DIR=.build-current TESSERA_ROCM_CHIP=gfx1201
python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py
python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py --dtype fp16 --dynamic-m
```

Raw values: [`gfx1201.json`](gfx1201.json).
