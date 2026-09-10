# Asynchronous process isolation evidence

CUDA SM120 and ROCm gfx1151 independently executed the native ANN pair, timed out
an idle worker stopped with SIGSTOP, polled bounded off-thread process teardown,
and executed a fresh replacement worker. Packets bind the worker source hashes
and native artifacts. Reproduce with `benchmarks/record_isolated_ann.py` on each
owning device.

This does not inject a GPU driver hang, prove device recovery after a real hang,
or justify performance promotion. No raw device pointer crosses the worker
boundary. Shared lifecycle tests separately exercise unconfirmed termination,
owner retention, admission saturation and module cleanup ordering.

Validation on Super-Bear WSL: 32 focused isolation, ANN transport, module
retirement and audit tests passed. Compiler-plan ownership checks, Ruff and the
mypy zero-error ratchet passed. This increment did not run the full unit suite.
