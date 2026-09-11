# Physical math diagnostics

`benchmark_physical_math.py` probes x86/ROCm runtime dispatch using explicit
metadata and measures completed host-wrapper calls. It requires observed native
execution and finite, correctly shaped outputs. Its metadata construction is
not a proof of serialized Graph → Schedule → Tile → target ancestry.

Run from the repository root with `PYTHONPATH=.:python`, choose `--target x86`
or `--target rocm`, and use positive `--iterations`. ROCm requires `--dtype all`.
Owning-device/toolchain prerequisites still apply. Packets from new runs are
regression-only; they cannot promote a route. See the
[alignment review](../COMPILER_ALIGNMENT.md) for the package-migration work.
