# Physical math diagnostics

`benchmark_physical_math.py --target x86` compiles seven f32 Graph operations
through native Schedule → Tile → Target. It serializes and reloads each native
image and checked launch descriptor before execution. Every launch must report
the exact artifact, image and descriptor identities. Rows retain raw host-wall
samples, compiler/image identity, descriptor and Graph/Schedule/Tile/Target
hashes. Compilation, JSON roundtrip and argument allocation are outside the
launch timing; these are completed host-wrapper times, not isolated kernel time.
The direct x86 scan C ABI comparison remains a separate diagnostic.

ROCm probes still construct explicit runtime metadata and require observed native
execution and finite, correctly shaped outputs. Their `metadata_runtime_probe`
boundary does not prove serialized compiler ancestry. ROCm package migration
remains open; no x86 evidence transfers to gfx1151 or gfx1201.

Run from the repository root with `PYTHONPATH=.:python`, choose `--target x86`
or `--target rocm`, and use positive `--iterations`. ROCm requires `--dtype all`.
Owning-device/toolchain prerequisites apply. New packets remain diagnostic and
cannot promote a route. Historical packet schemas and eligibility are unchanged.
