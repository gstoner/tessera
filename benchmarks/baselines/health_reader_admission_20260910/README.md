# Workload health admission and external-reader evidence

On SM120 and gfx1151 independently, fresh ANN workers executed bounded probes
against the analytic oracle before readiness, recovered from a stopped process,
and admitted a replacement with fresh probes. Each reader packet also records
four tape generations read through two native asynchronous device-to-device copy
streams before asynchronous pool frees on a third stream. No context barrier was
used by the tracked retirement path.

Reproduce using `benchmarks/record_isolated_ann.py` and
`benchmarks/record_reader_retirement.py`, passing the owning compiler and backend.
Packets carry source and artifact/compiler identities. No performance promotion
or overlap claim follows. SIGSTOP is a process fault, not an injected GPU hang.
Health probes establish readiness for this bounded workload at this time, not
permanent or device-wide health. Arbitrary raw pointer escape is unsupported.

Host validation: 95 tests passed and one skipped on Super-Bear in the broad
ownership run; three ROCm artifact tests failed there with `lld invocation
failed`. The complete ROCm admission file passed on Princess-Luna (13 passed,
one skipped). Tests cover rejected numerical health, bounded hung probes,
replacement preconditions, multistream exception unwinding and no hidden wait
for an eventless dependency. The full unit suite was not run.

The 11 audit tests, compiler-plan ownership check, Ruff and the zero-error mypy
ratchet also passed on Super-Bear WSL.
