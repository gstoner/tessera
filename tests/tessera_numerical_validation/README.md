# Tessera Numerical Validation Suite (v1)

> Drop-in test package to *verify numerical correctness* across Tessera backends, dtypes, and schedules.

## Goals
- Catch silent accuracy regressions early.
- Make tolerance choices explicit per op/dtype/backend.
- Validate determinism, NaN/Inf propagation, and gradient consistency.
- Exercise mixed precision and re-association sensitive reductions.

## Quick Start
```bash
# From repo root (suggested location: tests/numerics/)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Configure Backends
Set `TESSERA_NUMERICS_BACKEND` to select the op provider:
- `numpy` (default): pure NumPy reference/impl
- `torch`: PyTorch reference/impl (if CUDA available, uses GPU for DUT or REF depending on adapter config)
- `tessera`: call into Tessera python bindings / CLI (see `tessera_adapter.py` for stub)

Example:
```bash
export TESSERA_NUMERICS_BACKEND=tessera
pytest -q
```

Per-op tolerances and shapes live in `config.yaml`. Override at runtime:
```bash
pytest -q --config overrides.yaml
```

## Test Coverage (initial set)
- Matmul (fp64 ref, check fp32/bf16/fp16/ifx mixed)
- Conv2d NHWC (im2col reference / torch.nn.functional)
- Softmax + LogSoftmax (stability vs. log-sum-exp; masking cases)
- Reductions (sum/mean/max with reorder/tiling permutations)
- Random ops (uniform/normal): seeding + distribution tests (KS test, moments)
- Gradients (finite-difference & analytic when available)
- Edge cases (NaN/Inf/denormals; huge/small magnitudes; subnormal flush modes)

## Determinism
The suite enforces a known RNG seed. Stochastic ops also verify reproducibility given the same seed.

## CI Hint
Add a weekly GitHub Action on GPU labels to run:
- `pytest -q --maxfail=1 --disable-warnings`
- Publish `reports/numerics_summary.json` as artifact.

## Extending
- Add an op: implement in `tessera_adapter.py` and add a test_*.py.
- Update tolerances in `config.yaml` to reflect backend-specific realities.
