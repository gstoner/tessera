# SM120 native tensor program pipeline — diagnostic candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

This packet proves emitted native member arithmetic and typed compiler dataflow.
It does **not** prove retained runtime/public packaging or performance closure.

## Evidence

- counterexample-control.json: six recorded FP16/BF16 general-pipeline
  miscompilations on RTX5070; the old route emitted bare matmul and produced
  -16 instead of source-correct 0 or -32.
- candidate-device.json and candidate-device.txt: twelve final diagnostic
  replays on NVIDIA GeForce RTX5070 / SM12.0, driver610.88. All six source
  counterexamples pass exactly, plus six changed-input cases. Every unary
  intermediate equals its independently rounded reference in these samples;
  maximum final absolute error is below 1.4e-6. Inputs and all sixteen-element
  allocation tails remain unchanged.
- images/index.json, images/*.frame.json and images/*.ptx: compiler identity,
  actual typed buffer/launch frames and hashes of the executed native images.
  Fourteen stage references share eight content-addressed images; duplicate
  matmul/producer image copies are pruned.
  The driver uses frame buffer IDs, scalar arguments and selected geometry.
- operator-gates.txt: 23 operator-registry foundation checks pass.
- image-equivalence.json and check-image-equivalence.txt: the final standalone
  dialect-dependency fix recompiles every member; all six native frames and all
  eight executed PTX images are byte-identical to the device-tested candidate.
  This is image-equivalence binding, not an additional GPU execution count.
- source-identity.json and *.graph.mlir: source/compiler fingerprints and the
  six byte-identical original counterexample Graph inputs, retaining explicit
  RMSNorm FP32 policy. Final device source/program hashes match every frame;
  all six source hashes also match counterexample-control.json.
- final-compiler-gates.txt: 342 passes, including 27 positive/projection/adversarial
  program cases (including standalone Graph-to-Schedule) plus audit, diagnostic and pass-metadata checks.
- adjacent-owning-gates.txt: 224 passes / 80 declared hardware/architecture skips.
  Three owning gfx1201 matmul cases execute after a fresh ROCm probe. Skips
  include gfx1151, SM120 and Apple lanes; they are not numerical parity.
- source-ad-adjacent.txt: 55 source-AD/attribute checks pass; 86 declared
  SM120 device cases skip on the ROCm host and provide no NVIDIA device proof.
- sibling-assessment.json: host-free direct Schedule/Tile retains the source
  ReLU and Graph function on gfx1151/gfx1201, x86 and Apple. This is structural
  assessment and does not substitute for sibling execution.
- CONTROL.json: frozen pre-candidate compiler and layout-library identities.

## Numerical scope and replay

M17/K32/N11, FP16/BF16, ReLU-to-matmul, explicit-policy RMSNorm-to-matmul
and two-sided ReLU/softmax-to-matmul. Constant cases distinguish the dropped
producer from plain matmul. Changed-input checks compare every rounded producer
and final product against NumPy float64 references, with explicit storage
rounding and an accumulation bound.

compile_candidate.py and candidate_device.py preserve the original capture
commands, including scratch paths. The first runs the actual standard compiler
pipeline and native tensor-member projection before existing PTX packaging.
The second is a **diagnostic generic CUDA-driver sequencer**, not the production
runtime. Its native frame comes from the compiler; Python constructs no kernel
or derivative body. Reproduction requires the source/tool paths recorded by
these scripts, or relocating their input/output roots to a scratch directory.
candidate-device.exit and final-compiler-gates.exit are zero. Earlier
candidate-gates.txt records the preceding 340-test cohort.

No timing, general producer/layout/AD closure, or physical schedule promotion
is claimed. Retained native owner integration, compiler-free public replay and
separate kernel/completed-call benchmarks remain required before delivery.
