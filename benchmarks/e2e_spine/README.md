# E2E spine evidence and validation overhead

The Apple, SM120, gfx1151 and base-x86 recorders measure bounded native families
and seal packets through `tessera.compiler.e2e_fleet`. Each target owns its
hardware/runtime/compiler evidence; no certificate transfers between targets.
The base-x86 recorder does not establish AVX-512 or GPU performance.

`benchmark_packet_validation.py` intentionally builds a synthetic validation-only
packet in a temporary directory to measure report/seal/validation overhead. It
is infrastructure evidence, not native execution or selector performance evidence.

Keep historical packets immutable. Changed runtime/compiler fingerprints require
fresh owning-device measurements; updating a hash alone is not a re-seal.
Use the [compiler alignment review](../COMPILER_ALIGNMENT.md#additional-suite-review--2026-09-10)
for the relationship to other benchmark suites and outstanding promotion gates.
