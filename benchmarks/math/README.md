# Physical math diagnostics

`benchmark_physical_math.py --target x86` compiles seven f32 Graph operations
through native Schedule → Tile → Target. It serializes and reloads each native
image and checked launch descriptor before execution. Every launch must report
the exact artifact, image and descriptor identities. Rows retain raw host-wall
samples, compiler/image identity, descriptor and Graph/Schedule/Tile/Target
hashes. Compilation, JSON roundtrip and argument allocation are outside the
launch timing; these are completed host-wrapper times, not isolated kernel time.
The direct x86 scan C ABI comparison remains a separate diagnostic.

The gfx1151 f32/f16/bf16 `sum` rows now reload native Graph → Schedule →
Tile → Target packages and bind each launch receipt to the serialized image.
The other 18 ROCm rows still construct explicit runtime metadata and retain
their `metadata_runtime_probe` label. Their execution does not prove compiler
ancestry. The legacy cache comparison retains only those six metadata rows;
packaged sum has a separate module lifetime. All ROCm timing remains
synchronized host-wrapper time; broader package migration and gfx1201 proof
remain open.

Run from the repository root with `PYTHONPATH=.:python`, choose `--target x86`
or `--target rocm`, and use positive `--iterations`. ROCm requires `--dtype all`.
Owning-device/toolchain prerequisites apply. New packets remain diagnostic and
cannot promote a route. Historical packet schemas and eligibility are unchanged.
