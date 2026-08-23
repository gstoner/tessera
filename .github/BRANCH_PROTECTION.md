# Branch protection — required CI checks

Tessera's `Validate` workflow (`.github/workflows/validate.yml`) is
split into 6 lanes plus one aggregator job. The aggregator
(`validate-required`) is the single status check we wire into branch
protection — it succeeds iff every required lane succeeds.

## Required status checks (configure once, in GitHub UI)

Settings → Branches → branch protection rule for `main` → "Require
status checks to pass before merging":

| Check                | Source            | Why required                    |
|----------------------|-------------------|---------------------------------|
| `validate-required`  | `validate.yml`    | Fans in lint / unit / audit — one check, three lanes. |

Selecting just `validate-required` is sufficient. Each underlying
lane (`lint (ruff + mypy ratchet)`, `unit (pytest -m "not slow")`, and
`audit (drift + claim_lint + examples)`) is still reported
individually in the PR Checks tab so contributors can see which lane failed
without expanding the aggregator log.

## Opt-in lanes (NOT required for merge)

| Check                                       | Trigger                                          |
|---------------------------------------------|--------------------------------------------------|
| `lit (MLIR FileCheck — opt-in)`             | PR label `lit-smoke` · manual dispatch · push to main |
| `sanitizer (asan / tsan / ubsan — opt-in)`  | PR label `sanitizer-smoke` · manual dispatch     |
| `rocm hsaco serialization (host-free — opt-in)` | PR label `lit-smoke` · manual dispatch · push to main |

Apply the labels from the PR's right-side sidebar.

## Apple Metal 4 promotion

Apple exact-device promotion is a local backend-host proof, never a registered
GitHub self-hosted runner. Run `scripts/run_apple_metal4_release_gate.sh` on the
named Metal 4 Mac and publish its sealed packet under
`docs/audit/evidence/apple/metal4/` in the coordinating PR. The ordinary
required `validate-required` fan-in remains portable: its unit and audit lanes
verify the pushed packet's schema, hashes, commit provenance, two clean
correctness reports, paired device/end-to-end evidence, fresh LLVM/MLIR 23
cache, and explicit power/thermal/GPU-contention availability. Metal 3 is a
non-blocking compatibility surface.

## Configuration via GitHub CLI

```sh
gh api -X PUT \
  "repos/tessera-ai/tessera/branches/main/protection" \
  -f required_status_checks.strict=true \
  -F 'required_status_checks.checks[]={"context":"validate-required"}' \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -f enforce_admins=false
```

(Adjust the repo slug and reviewer count to match your governance
policy.)

## Lane-by-lane wall-clock budget

| Lane         | Wall-clock target | Notes |
|--------------|------------------:|-------|
| lint         | ~30s              | ruff + mypy ratchet (defends 0). |
| unit         | ~2min             | `pytest -m "not slow"`, ~4300 tests. |
| audit        | ~10s              | support_table drift + claim_lint + examples audit. |
| lit          | ~10min if installed | LLVM/MLIR 23 install + tessera-opt build + lit. |
| sanitizer    | ~15min per matrix | asan + tsan + ubsan run in parallel. |
| rocm-serialize | ~15min if installed | LLVM/MLIR 23 + lld-23 install + HIP-less `tessera-rocm-opt` build + hsaco proof. |

The standalone C++ runtime and collectives compile-check are intentionally
local-only. Run `scripts/validate.sh` on the owning host; it builds and tests
the standalone CPU runtime and compiles the collectives execution unit without
making pull-request approval depend on GitHub-hosted apt mirrors.

The ROCm compiler suite is likewise local-only (removed from CI 2026-08-19:
an apt LLVM/MLIR 23 install plus a from-scratch `tessera-rocm-opt` build is
~25min — too heavy for hosted runners). `scripts/validate.sh` runs
`check-tessera-rocm` when the build tree has the ROCm backend configured
(`-DTESSERA_BUILD_ROCM_BACKEND=ON`); run it on the primary box before merging
ROCm backend changes. This is the ONLY automated coverage for
`src/compiler/codegen/Tessera_ROCM_Backend/test/rocm/` — `check-tessera` does
not include that suite and `lit tests/tessera-ir/` runs a different one
through a different driver, so skipping it lets a ROCm backend fixture
regression reach main unnoticed.

`rocm hsaco serialization` proves the compiled ROCm lane still emits an
AMDGPU code object. It needs **no GPU and no ROCm install** — serialization
is compile-time work that shells out to `ld.lld` — so it runs on a stock
hosted runner and closes the blind spot that let PR #619's total serializer
outage go unnoticed. It does NOT prove the object runs or is numerically
correct; that evidence needs the real gfx1151 device.

The lit + sanitizer lanes are intentionally off the critical path so a
contributor doesn't have to wait 15+ minutes on every PR.

## How "required" interacts with `if:` filters

`validate-required` uses `if: always()` and pulls `needs.<lane>.result`
explicitly so a *skipped* required lane (which would normally pass
GitHub's default status check logic) is treated as a failure. The three named
lanes must all report `success`.

## Adding a new required lane

1. Add the job to `validate.yml`.
2. Append it to the `needs:` list on the `validate-required` job.
3. Append a `"${{ needs.<job>.result }}"` line to the `required`
   array in the verification step.
4. Open a PR; once it lands, no branch-protection change is needed —
   the aggregator already covers the new lane.
