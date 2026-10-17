# SM120 AD witness digest reuse

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-AD-WITNESS-DIGEST-20261017.

Recorder: benchmarks/nvidia/benchmark_attention_witness_reuse.py.
Measured implementation: e6804ac81aa5c840d4f7d8181ef105ffc6298835.
The packet binds recorder, implementation, native images, tools and GPU identity.

Six RTX 5070 FP16/BF16 profiles compare the same strict parser and native
program with zero versus default digest-cache byte budget. Alternating order,
30 warm reverse samples and 12 capture-plus-first-reverse samples per arm
retain every numerical/metadata/ABI check. Independent FP64 comparisons run
outside each timed region; control/candidate outputs are bit-identical.
Packages are constructed before measurement and the candidate digest cache is
warm. Compilation, package construction, external uploads, oracle/readback and
close are excluded; each measured capture creates a fresh resident generation.
Capture plus first reverse improves 1.33-1.37x (13.76-15.13 ms versus
10.03-11.39 ms). Warm reverse is largely unchanged. These are completed host
intervals, not isolated device-kernel speedups. Allocations remain 6/3 per
reverse. No native arithmetic or schedule/default selection changes.

The cache holds only immutable encoded-witness to SHA256 results: at most
32 entries and 256 KiB UTF8 per entry; oversized inputs use the full decoder.
Every call still validates exact qualified name and multiplicity. PID change
replaces the CPU cache. No package certificate, runtime shape, tensor, native
function, stream or GPU context is cached. Four red reproductions established
the prior lookalike-name admission, including a consistently repinned member.
Final focused package/refusal/device validation passes 135 cases.

The eager row-workspace prototype was rejected: allocation traffic fell but
matched total time did not improve consistently. Its decision and failure
cleanup evidence are retained; the candidate was not integrated. Profiler
counts attribute repeated witness decoding only and are not timing denominators.

Canonical source Ruff and the changed recorder/tests pass. The full mypy
ratchet passes with zero errors against baseline zero. The generated-document
drift check passes; publication-focused tests pass 38 cases and the recorder
naming gate passes. Broad unit validation is still running.

All five aggregate engineering slices remain open. Broad unit and public
CI/review are pending for this increment; previous PR evidence is not reused
as validation of this source.
