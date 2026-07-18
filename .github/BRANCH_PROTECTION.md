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
| `validate-required`  | `validate.yml`    | Fans in lint / unit / audit / build — one check, four lanes. |

Selecting just `validate-required` is sufficient. Each underlying
lane (`lint (ruff + mypy ratchet)`, `unit (pytest -m "not slow")`,
`audit (drift + claim_lint + examples)`, `build (runtime +
collectives)`) is still reported individually in the PR Checks tab so
contributors can see which lane failed without expanding the
aggregator log.

## Opt-in lanes (NOT required for merge)

| Check                                       | Trigger                                          |
|---------------------------------------------|--------------------------------------------------|
| `lit (MLIR FileCheck — opt-in)`             | PR label `lit-smoke` · manual dispatch · push to main |
| `sanitizer (asan / tsan / ubsan — opt-in)`  | PR label `sanitizer-smoke` · manual dispatch     |

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
| build        | ~5min             | CMake runtime + collectives compile-check. |
| lit          | ~10min if installed | LLVM/MLIR 23 install + tessera-opt build + lit. |
| sanitizer    | ~15min per matrix | asan + tsan + ubsan run in parallel. |

The lit + sanitizer lanes are intentionally off the critical path so a
contributor doesn't have to wait 15+ minutes on every PR.

## How "required" interacts with `if:` filters

`validate-required` uses `if: always()` and pulls `needs.<lane>.result`
explicitly so a *skipped* required lane (which would normally pass
GitHub's default status check logic) is treated as a failure. There's
no escape hatch — the four named lanes must all `success`.

## Adding a new required lane

1. Add the job to `validate.yml`.
2. Append it to the `needs:` list on the `validate-required` job.
3. Append a `"${{ needs.<job>.result }}"` line to the `required`
   array in the verification step.
4. Open a PR; once it lands, no branch-protection change is needed —
   the aggregator already covers the new lane.
