# Historical reverse integration gate

The requirements below preceded the role-aware exporter and native sum integration. The rebuilt compiler now passes 419 host checks and 86 gfx1201 owning checks, including composed independent/shared-scale reverse gradients. See ../gfx1201_composed_scaled_vjp_20261008/ for the follow-up evidence. Generic batching/transpose and wider dynamic/storage AD remain open.

# Next native reverse integration gate

The matching tessera-opt probe of the real six-input composed frontend Graph
fails during export with: scaled transpose export needs explicit scale roles
and one output seed. The terminal diagnostic is preserved in
composed-scaled-vjp-export-probe-20261008.log. No reverse execution is proved.

Native paired AD already emits four tensor.generate scale-adjoint regions
and two encoded-matrix zero results. Export, checked program binding and
public backward routing must all be extended together:

1. NativeScaledMatmulProgram.h currently requires four returned argument
   adjoints and five input-frame arguments. Derive the frame from the forward
   signature and validate requested roles against actual scale operands.
   Preserve the paired backward Graph as witness and actual SSA captures.
2. NativeScaledProgram.validate currently requires five arguments and one/two
   gradient roles; it treats each reduction capture as the complement of the
   differentiated argument. Validate each member's actual ordered captures
   and cotangent slot for the expanded frame. Keep ownership, storage,
   image/witness equality and selected gradient order checks.
3. native_vjp_plugins.py requires four forward inputs and reads cotangent
   metadata from buffer 4. Bind the explicit frame rather than a fixed slot.
4. JIT.native_backward currently dispatches plugins only for one operation.
   Certify the composed frontend, preserve the entire Graph in package
   identity, and admit only the proven scale-role/product/sum contract.
5. Add owning gfx1201 tests for selected/reordered independent and shared
   scale gradients, changed cotangents, private scratch/returned lifetime,
   malformed roles/captures and warm compiler/reference refusal.
6. Record separate complete native event and public host timings, with
   independent finite-difference or float64 scale-gradient oracles.

The existing physical isolated reduction contract takes four captures:
two E4M3 matrices, one opposite FP32 scale and one FP32 cotangent. Keep that
contract for individual reductions; multiple products sharing a scale require
native sum members and dependency/lifetime validation rather than dropping
any adjoint contribution. Generic batching/transpose labels remain open.
