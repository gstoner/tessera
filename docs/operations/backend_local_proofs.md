---
last_updated: 2026-07-15
---

# Local Exact-Device Proofs

Hardware evidence is collected on the physical backend machine before a PR is
submitted. GitHub-hosted CI remains responsible for portable CPU, lint, build,
and documentation checks; it does not claim device execution.

| Machine | Exact target | Local proof entry point |
|---|---|---|
| RTX 5070 Ti | NVIDIA `sm_120` | `bash scripts/run_nvidia_release_gate.sh` |
| Ryzen AI Max+ 395 / Radeon 8060S | ROCm `gfx1151` | Canonical commands in `docs/audit/backend/rocm/todo.md` |
| Apple M1 Max | Metal `Apple7` | Canonical commands in `docs/audit/backend/apple/todo.md` |

Each proof run records the checked-out commit, hardware identity, driver and
toolchain versions, exact-device JUnit reports, selected routes/provenance,
and serial performance evidence. Keep the generated bundle with the PR and
post its commit SHA plus a concise result summary in the PR description or a
comment. Do not represent the result as hosted CI evidence or reuse it for a
different physical target.

The NVIDIA command writes its bundle to `artifacts/nvidia-release/` by default.
Set `TESSERA_NVIDIA_REPORT_DIR` to place the bundle elsewhere.
