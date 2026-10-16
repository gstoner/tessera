# Native AD correction product projection

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

The native paired AD pass exports the actual correction tensor product from the
original attention Graph. It removes rounded saved O from correction inputs,
retains the original full paired-AD witness, frontend input permutation and
requested physical gradient activity, and preserves source/residual argument
properties with read-only input ownership. No Python Graph/GPU IR reconstruction
is used. Preserving a layout attribute is not proof that a physical layout
executes; backend admission remains separate.

Twenty-one new projection cases cover FP16/BF16, matching-half/F32 dO, bias/LSE
seed variants, reordered frontend/activity and layout attributes. Together with
the original contract, legacy AD and diagnostic/pass registries, 410 host-free
cases pass with zero skips/failures. The old compiler rejects the new mode. A
layout-loss counterexample was reproduced on the first implementation and fixed.

record.json binds frozen sources and receipts. This proves Graph/AD projection,
not Schedule/Tile lowering, target codegen, runtime ABI, numerical execution,
portable replay or performance. Those next integration gates remain open.
