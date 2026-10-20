# Native bounded original-source attention reverse projection

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-SOURCE-ATTENTION-20261019.

The native paired AD pass now retains its original static source/paired witnesses
and projects cloned members plus typed call edges into an independently verified,
capacity-scoped program. Source schemas v3/v4 are additive; integer attention
schema5 still means saved-LSE plus bounded. No Python derivative or backend Graph
constructor is added. Logical shapes retain the MLIR23 dynamic sentinel, with
separate trace and capacity shapes and checked byte/ownership/lifetime metadata.

## Proof and boundaries

- 359 shared tests pass:44 projection/transport/auto-policy/target-boundary cases
  and315 diagnostic/pass/audit gates. An earlier36-case cohort is retained but
  is a subset, not additional coverage.
- 171 adjacent static source AD, transport, portable-validation and bounded-norm
  tests pass.
- 18 static storage/role/row-policy source programs have byte-identical complete
  native outlines and manifests against the frozen normalization compiler.
- 24 owning RTX5070 cases execute the actual projected images for FP16/BF16,
  inline/row programs, reordered roots/results and two capacities. Each compiles
  once from trace Sq=Sk=1, then replays five shrink/grow pairs on fixed allocations.
  Independent primal/VJP oracles, unchanged root bytes and every inactive tail
  pass. XML records program and member image hashes; identity binds actual
  device/compiler/provider/source. Five replays are not five distinct cases.
- A native function-signature mismatch was reproduced and fixed by rebuilding
  signatures from projected block arguments. A test expectation was corrected
  to the existing MLIR23 dynamic sentinel. An auto-policy fixture originally
  had Dv=3, outside its unchanged Dv>=4 policy; corrected Dv=7 cases pass.
  Original failing native and shared logs are preserved.

This is projected-member diagnostic execution, not retained composed-owner,
ordinary public JIT or compiler-free portable proof. Native CUDA capacity
ownership and atomic runtime extent updates remain required, followed by public
and portable integration and full delivery/benchmark gates. No source-program
performance promotion follows; the earlier normalization timing packet has a
separate measurement scope. All five aggregate goal slices remain open.

## Reproduce in the owning host environment

Use the matching native compiler and compatible LLVM/CUDA provider. Run all
tests in host WSL. The device suite requires the owning SM120 GPU.

```sh
python -m pytest -q tests/unit/test_native_bounded_source_attention_reverse.py
python -m pytest -q tests/device/nvidia/test_bounded_source_reverse_members.py --junitxml=bounded-source-members.xml -o junit_family=legacy
```

The initial xunit2 run passed with property-format warnings; its XML retains the
per-case program/image properties, which were checked before packet acceptance.
Using legacy JUnit formatting removes that format mismatch on reproduction.
