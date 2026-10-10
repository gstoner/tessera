# Native scaled-matmul JVP foundation — 2026-10-07

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
This packet establishes Graph/Schedule artifact evidence on Super-Bear WSL.
It does not establish AMD execution, a numerical benchmark, generic AD closure,
or a green full unit suite.

The native TangentInterface builds one unchanged-policy scaled product per
active floating operand and sums its results. Original block layout, physical
contract and transpose flags remain attached. Derived Schedule hash is not
copied from the primal. Encoded scale/code tensors have no implicit
straight-through derivative. Only exact_per_block f32 output is admitted;
rounded output and other numerical policies need explicit derivative policies.

Both native tools rebuild successfully after a corrected C++ op API error.
Four focused lit fixtures pass: two floating scale seeds, a transposed FP8
matrix seed with inactive encoded scales, encoded-seed rejection, ordinary
matmul regression, and native Schedule derivation of the scale-seed products.
The negative encoded seed is rejected by the forward pass argument gate.

The full Tile experiment fails: repeated primal/JVP products emit identical
content hashes and multiple schedule.artifact instances, while the Tile
consumer requires exactly one matching artifact across the module. Fix
artifact instance ownership and lifetime before executable multi-product AD.
The transposed FP8 fixture also remains outside current Graph-to-Schedule
admission. The residual tessera.add needs native program integration,
buffer lifetime and exact-device numerical/timing proof. No coverage states
or closure assertions have been weakened.

Build and initial/final focused test logs, including failing experiments,
are retained. Reproduction:
source .build-sm120-w1-1/validation-env.sh
export PATH="$PWD/.venv/bin:$PATH"
Run LLVM lit on phase_f4/autodiff_forward_scaled_matmul*.mlir and
phase_f4/autodiff_forward_core.mlir.
