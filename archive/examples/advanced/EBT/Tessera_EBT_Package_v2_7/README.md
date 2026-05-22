# Tessera EBT — v2.7

New in this cut:
- **tessera.ebt dialect (.td)** with core ops (energy, grad_y, inner_step, jvp, etc.).
- **Autodiff MLP** example showing `tessera.autodiff.grad_y` on the MLP head.
- **CPU unit test** proving `JVP == directional derivative` for a quadratic energy.

## Build & run the CPU unit test
cmake -S . -B build && cmake --build build -j
./build/jvp_equals_directional
# -> prints finite-diff and JVP values then "PASS ..."

## Use the ODS
Drop `models/ebt/dialect/EBTDialect.td` into your in-tree dialect build and hook to your C++ op defs.
