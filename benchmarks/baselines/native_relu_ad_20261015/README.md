# Native ReLU tangent and named producer AD census

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-RELU-AD-20261015.

The native MLIR ReLU TangentInterface reuses the strict-positive reverse
predicate. Its zero tangent at the kink and real f16/bf16/f32/f64 domains
are checked against emitted paired SSA, including non-finite primal values.
Direct unencoded symbolic ReLU is admitted by verified dataflow; compound
dynamic programs retain their existing contracts.

The 23 focused cases include three RMSNorm -> ReLU -> matmul FP32 paired
Graph derivatives (A, B, A+B), independently compared with analytic and
central-difference oracles. Four provenance cases keep Graph transformation
separate from executable target AD. The shared lane passes 469 checks.

`before/route-census.json` preserves the original missing active-ReLU
TangentInterface failures. `after/route-census.json` records actual native
primal execution on RTX 5070 and emitted f16/bf16 paired Graph for all six
storage/activity combinations. Both retain source/compiler identities and
full recorder source. These are different compiler epochs, not interchangeable
receipts. The 150 owning-device controls pass with the rebuilt compiler,
covering producer partition/DAG, captured profiles and public saved-LSE
attention forward/prepared reverse.

## Still open

Public composed JVP and VJP packages remain unavailable. The paired Graph
has multiple outputs and consumers; the current SM120 tensor exporter and
checked owner describe one final matmul. The next step is native paired-SSA
outlining with explicit input/output buffer IDs, shared-edge last-use and
returned-output ownership, followed by actual native execution and numerical
proof. General active-normalization tangent members remain additional work.
No Python derivative program or eager fallback is introduced here.

No new performance claim: this change is differentiation IR support and
leaves primal Schedule/Tile arithmetic unchanged. Existing flat/serial timing
packets keep their original source/compiler identities. No sibling-device
proof or aggregate five-slice closure follows from these results.
