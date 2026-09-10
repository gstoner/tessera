# Completion retirement and assertions corpus (2026-09-09)

The NVIDIA SM120 and ROCm gfx1151 packets independently pass 19 source-completion
cases on the final source. They fingerprint the source and owning compiler.
They do not inject destructive GPU failures or establish performance promotion.
Host tests separately verify cached exception-root release and uncertain free
quarantine; arbitrary native object allocation/collection and real CPython
frame reconstruction remain open.

The assertions packet records all 474 discovered MLIR fixtures: 474 passed,
0 unsupported and 0 failed in a hardware-free all-target compiler lane built
against assertions-enabled LLVM/MLIR 23.1.1 on Super-Bear. CUDA and HIP runtime
integration were both disabled for this structural compiler proof.
The fixed forward-storage fixture previously aborted on loading Tile during a
pass; AutodiffForwardPass now declares that dependency. Fifteen x86/Apple
fixtures previously ran without their omitted backend and failed during
registration or parsing. They now declare the backend feature they consume.

The earlier core-only report's 115 unsupported results were configuration-local:
53 required Apple, 36 ROCm, 21 x86 Target IR, four NVIDIA, and one was data for
a separate Python exact-execution test. The data file now lives under
`tests/fixtures/`; the all-target compiler lane exercises every active lit
fixture. This closes structural assertions coverage. It does not establish
owning-device execution or performance for Apple, NVIDIA or ROCm.
