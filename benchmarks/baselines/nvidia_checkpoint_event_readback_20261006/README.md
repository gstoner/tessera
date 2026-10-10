# Native attention event output readback

Owner NVIDIA-LSE-1 / E2E-REAL-6; sync NVIDIA-EVENT-READBACK-2026-10-06.

The native CUDA-event forward and standard backward profilers now copy the
final timed outputs to the supplied host buffers after the stop event has
completed. O/LSE and dQ/dK/dV can therefore be checked against the independent
FP64 oracle for the actual event arm. D2H occurs outside the reported event
interval. Compact backward already owned post-window readback.

The recorder poisons only output buffers before every event window, then
checks all outputs after that window and after every end-to-end wall window.
Argument validation precedes the hardware gate; selected GPU identity must
report one SM120 device. Version 4 records per-window numerical errors and
both matching compiler binary hashes plus the native runtime hash.

- native-build.txt: host WSL native CUDA profiler build passed.
- device-and-contract-tests.txt: 63 passed, including eight poisoned-output
  device cases across forward/backward, saved/recompute and plain/full bias.
- registry-tests.txt: 292 passed. mypy.txt: zero errors. lint.txt: Ruff clean.
- plain-rtx5070.json / bias-rtx5070.json: six profiles and 24 measured arms,
  three shapes, grouped heads, ragged sequences and batch two. Each arm has
  five 100-launch CUDA-event windows and five end-to-end wall windows;
  independent oracle records cover all 120 event and 120 wall windows.
  Sources and core/NVIDIA compiler/runtime hashes match the final checkout.
- Recorded by benchmarks/nvidia/record_lse_checkpoint.py. Final timing was
  collected after the owning device test process terminated. Earlier runs
  overlapping tests are retained under the ignored build diagnostics folder.

Measured backward event medians span 11.11–38.82 us for saved O/LSE and
24.92–792.19 us for recompute. Public wall medians are recorded separately.
These characterize the existing routes; selector promotion remains false.
Recompute backward already imports the original Graph through native
Schedule/Tile and carries full ancestry. Ordinary forward comparator ancestry is now complete under the later
forward-lineage integration below.

The first poison-test run allocated FP64 launch buffers from its FP64 oracle;
the checked f32 ABI rejected them. Final tests preserve FP64 references and
use f32 physical buffers. Failure receipts remain in build diagnostics.

Native explicit-LSE cotangent differentiation, wider composed/dynamic AD,
full scaled-matmul batching/transpose closure and publication remain open.
No new Apple, ROCm or x86 device execution is claimed.

## Ordinary forward lineage certification

Sync NVIDIA-FORWARD-LINEAGE-2026-10-06; owner NVIDIA-LSE-1 / E2E-REAL-6.
The ordinary forward package exports retained Graph and Schedule digests,
plus the Target IR digest from its native image. Packaging replays original
Graph -> Schedule and Schedule -> Tile before target compilation. Divergent
Graph or Schedule policy is rejected before target code generation.

forward-lineage-tests.txt: 69 passed, 6 sibling-environment skips. Includes
valid-but-changed policy checks and 46 RTX 5070 forward cases with serialized
native execution; Target digest agrees with the image after serialization.
forward-lineage-mypy.txt reports zero errors; forward-lineage-lint.txt is clean.

The two live JSON packets were regenerated after that device test process
completed. All 24 saved/recompute forward/backward arms now have complete
Graph/Schedule/Tile ancestry, five event and five wall-window oracle records.
Source hashes match the final checkout. Prior packets are preserved in the
ignored pre-forward-lineage build diagnostics folder. No new selector choice.

Explicit LSE-cotangent native AD, wider composed/dynamic routes, general
scaled-matmul closure gates and ROCm performance obligations remain open.
