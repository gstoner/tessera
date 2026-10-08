# Frontend parity numeric policy — 2026-10-08

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync FRONTEND-PARITY-NUMERIC-POLICY-2026-10-08.

Frontend differential certificate bodies already bind rtol/atol, but the
reuse key previously retained only tensor signature and permitted effects.
A strict request could therefore receive a prior loose certificate.
The reuse key now includes both tolerances. No Graph lowering or derivative
rule is reconstructed in Python; native compilation and runtime ABI are
unchanged.

Host WSL validation:
- 26 focused frontend/coverage/audit tests pass.
- 243 mapped differential/certificate/frontend tests pass, 56 deselected.
- Source Ruff passes.

The new tests prove changed rtol and changed atol produce matching distinct
certificate policies, identical requests reuse the exact certificate, and
a 1e-4 injected numerical discrepancy passes atol=1e-3 but fails atol=1e-7
through TesseraJitError with the underlying numerical parity ValueError.
The looser valid certificate remains reusable after strict rejection.
Raw passing logs are retained losslessly; source hashes are recorded here.

This is host frontend numerical-proof evidence. No fresh Apple, x86, gfx1151,
gfx1201 or SM120 device execution/performance result is claimed. All four
backend plans assess the shared guard. Generic scaled-matmul batching and
transpose, wider derivatives and physical backend envelopes remain open.
