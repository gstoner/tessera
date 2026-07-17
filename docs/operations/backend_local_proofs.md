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

## NVIDIA local release ownership

NVIDIA-TEST-7 intentionally uses no GitHub or self-hosted runner. Ordinary PR
CI owns the complete cross-backend CPU suite. The WSL release command owns the
physical RTX 5070 Ti directly, rejects concurrent runs through a host lock,
records success or failure in every bundle, and keeps NVIDIA host-free/shared
registry gates, compiler artifact, two-run device correctness, and serial
performance selectable as separate layers:

```bash
bash scripts/run_nvidia_release_gate.sh --layer compiler
bash scripts/run_nvidia_release_gate.sh --layer device
bash scripts/run_nvidia_release_gate.sh --layer performance
```

Running without `--layer` executes all four layers fail-closed. By default,
bundles are retained under a timestamped commit directory in
`artifacts/nvidia-release/`; set `TESSERA_NVIDIA_REPORT_DIR` for an explicit
archive location. The performance bundle includes the committed sm_120
baseline corpus and SHA-256 inventory. Summarize or attach the local bundle to
the coordinating PR; do not describe it as GitHub-hosted evidence.
