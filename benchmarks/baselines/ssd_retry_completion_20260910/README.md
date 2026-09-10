# Shared SSD and retryable completion

Host validation on Super-Bear WSL, 2026-09-10. Assertions-enabled tessera-opt:
`e6db4057cdcebadd82904c823d770d54395da63b26076f512cdb7e6dd96d7d83`.

The internal Schedule SSD recurrence lowers to structured tensor loops and
executes through the native CPU JIT. Numerical tests use T=5, H=2, N=3, P=2,
chunk sizes 1/2/5, seeded nonzero initial carry and signed inputs. Y, final carry
and partial-chunk checkpoints agree with the independent recurrence oracle;
initial carry is unchanged. Replay rejects changed arithmetic, and native
verification rejects negative/oversized chunks, incompatible shapes, fp64,
zero extents and oversized tensors. The lit fixture passes LLVM FileCheck.

The combined focused run passed 351 tests (including audit/registry gates).
After final recovery/exception refinements, 37 affected tests passed again;
Ruff and the zero-error mypy ratchet pass. Full unit suite not run.

Recovery tests inject uncertain termination and delayed process exit, then
concurrently poll to prove exactly-once owner/slot release without retrying
termination. They do not induce a GPU driver hang. Paired AD tests cover both
child failure orders and resume only children without submitted frees. Heap
tests create an actual rooted cycle and reclaim it after root release while
ABI readers continue to exclude mutations. Exception tests preserve recorded
cause/context identities and bypass custom setters/descriptors during decode.

No GPU SSD package, tuned performance, native exception allocator producer,
CPython frame reconstruction, general checkpoint AD or attention raising is
claimed. Source hashes capture this host-contract increment; earlier CUDA/ROCm
ANN measurements remain bound to their own recorded compiler and source.
