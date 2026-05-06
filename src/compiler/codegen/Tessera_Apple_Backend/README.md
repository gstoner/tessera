# Tessera Apple Backend

This backend defines the hardware-free Target IR contracts for Apple Silicon.
It is currently an artifact path with verifier-backed Python lowering from the
object compiler spine:

```text
textual DSL / @jit -> Graph IR -> Schedule IR -> Tile IR -> Apple Target IR
```

- `tessera_apple.cpu.*` models Accelerate/vecLib/BNNS CPU calls.
- `tessera_apple.gpu.*` models Metal/MPS-style GPU kernels and dispatch.

The active Python object model lives in
`python/tessera/compiler/target_ir.py` and is covered by
`tests/unit/test_target_ir.py` plus target-contract tests. It verifies required
Apple CPU attrs such as `framework`, `abi`, and `dtype`, and Apple GPU attrs
such as `kernel`, `framework`, `status`, dispatch queue, and artifact kind.

Native `metallib` compilation and GPU runtime launch are follow-up milestones.
Apple CPU execution for supported matmul-style JIT paths may use the existing
CPU fallback/Accelerate-facing artifact path where configured; GPU artifacts
remain hardware-free unless a future backend doc says otherwise.
