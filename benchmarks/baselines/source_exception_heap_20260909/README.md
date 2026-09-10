# Indexed source exception completion (2026-09-09)

Independent RTX 5070 SM120 and gfx1151 runs each pass 19 correctness cases via
`benchmarks/record_source_exception_bindings_gpu.py`. Packets fingerprint the
compiler and source, including the indexed exception carrier. These are fresh
measurements, not updates of prior packets.

Coverage: source VJP success/failure, explicit custom host constructors, retained
cause payloads, scoped completion and signed/nested slicing. Host regressions
separately cover long/cyclic heaps and uncertain driver failure retention.

No device-side arbitrary object allocation, native CPython frame reconstruction,
driver reset/recovery, kernel tuning or performance promotion is established.
