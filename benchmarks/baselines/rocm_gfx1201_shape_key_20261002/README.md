# gfx1201 scheduled matmul image-key evidence

Owner: E2E-REAL-6-ROCM-MATMUL-CACHE.

Retained packet: matmul_cache_reuse_warmed.json. Recorder:
benchmarks/rocm/record_gfx1201_matmul_shape_key.py. The packet records
Tajasarus, configured/live gfx1201, compiler SHA256, source hashes and a dirty
source revision. It is historical diagnostic evidence, not a fresh run of
the current recorder or promotion evidence.

Six static shapes retain one image digest and entry symbol, while each
shape retains its own Schedule/Tile digests and ABI guards. Correctness is
checked before timing, during five warmups, and after each end-to-end sample.

| M×N×K | Image cache | Package ms | Kernel event ms | End-to-end ms |
| --- | --- | ---: | ---: | ---: |
| 16×16×16 | cold | 281.202 | 0.008639 | 2.137 |
| 32×32×32 | warm_cache | 52.680 | 0.014799 | 2.370 |
| 48×32×16 | warm_cache | 51.711 | 0.016119 | 2.622 |
| 128×128×128 | warm_cache | 50.566 | 0.019199 | 2.990 |
| 256×256×256 | warm_cache | 51.523 | 0.016039 | 5.816 |
| 512×512×512 | warm_cache | 52.306 | 0.024960 | 5.853 |

Kernel HIP events bracket the kernel launch. End-to-end wall measurements
include module load, host/device allocation, copies, synchronization, descriptor
validation and Python dispatch. Package and Schedule costs are separate.
The aggregate event median pools different shapes and must not be used as a
single-shape throughput claim. The packet declares promotion_eligible=false.
Current recorder options and native module caching have evolved since this
receipt; this packet does not prove those newer paths.
