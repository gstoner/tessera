# GFX1201 packed-folded prefill proof and matched timing

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-FOLDED-DECODE-2026-09-23`.

The recorder is
`benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py`. It compares
the exact K32, expanded-folded, packed table-decode, packed integer-decode,
and pinned Radiance routes on the same logical quantized inputs. Each shape
must pass the independent sampled exact oracle and bitwise BF16 agreement
before alternating HIP-event measurements. Package/model-load conversion and
HSACO compilation are outside the timed launch. `RADIANCE_MXFP4_WPERM=1`
selects the same fragment-order weight layout. The evidence packet records
timed payload SHA-256, selected-symbol ISA, toolchain, source, and benchmark
identities. It does not claim an AMD profiler phase fraction.

The packed integer route is manually executable and receipt-bound, but it
is not automatically selected. Exact K32 remains the correctness default;
the expanded-folded route remains faster for the measured prefill shapes.
