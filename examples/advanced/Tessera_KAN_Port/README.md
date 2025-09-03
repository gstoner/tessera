# Tessera KAN Port (Starter v1)

This is a starter-quality port of Kolmogorov‑Arnold Networks to the Tessera Programming Model.

- `KANLinear` constructs a Tessera graph using a shared **B‑spline** basis + linear mixing.
- CPU fallback (`NumPy`) is included for quick validation.
- MLIR stubs (`kan_ops.td`, `LowerKANToTessera.cpp`) show how to introduce `kan.*` ops and lower them to matmuls.

## Try it (CPU fallback)
```bash
python -m pip install numpy
python examples/mnist_kan_tessera.py
```

## Integrate into Tessera
- Add this folder to your repo under `src/` and `tools/` as desired.
- Register `createLowerKANToTesseraPass()` and the `kan` dialect with your existing driver.
- Replace the CPU reference with your GPU kernels using the Tessera tile runtime.

## Notes
- Shapes and recurrence follow the standard Efficient‑KAN idea (basis precomputation + linear combination).
- L1/L2 regularization is left to your training loop; attach attributes or loss terms as appropriate.

MIT-like starter; attribute upstream Efficient‑KAN.