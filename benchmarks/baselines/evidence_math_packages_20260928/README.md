# Serialized x86 physical-math evidence

Owner EVIDENCE-PACKET-1; sync `EVIDENCE-MATH-PACKAGES-2026-09-28`.
Based on PR #878 (`160ebd593`) plus this benchmark-consumer change.
Princess-Luna WSL, Zen 5 AVX-512; no other architecture claim.

All seven f32 rows passed independent NumPy checks after compiling native
Graph → Schedule → Tile → Target and serializing/reloading the image and
checked descriptor. Each launch receipt must match all three package identities.
The focused suite passed 16 tests. `x86.json` retains 31 warm samples per row,
IR hashes, complete descriptors and native-image identity (payload bytes omitted
from the report, retained during the executed serialization roundtrip).

Run with `PYTHONPATH=python:. TESSERA_BUILD_DIR=$PWD/build`:
`python benchmarks/math/benchmark_physical_math.py --target x86 --iterations 31`.
Host-wrapper timing excludes packaging, serialization and argument allocation.
No clean paired performance admission, kernel-time claim or route promotion.
ROCm metadata consumers and the broader evidence-envelope families remain open.
