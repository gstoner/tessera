# Tessera EBT — v2.4

**New**
- Concrete pass scaffolds:
  - `materialize_loops.{h,cc}`
  - `select_grad_path.{h,cc}`
- Sample MLIR with expected `scf.for` and JVP swap markers.
- CPU numeric check proving monotone energy descent for a quadratic energy.

## Build
cmake -S . -B build && cmake --build build -j

## Run wrapper on sample
./build/tessera-ebt-opt models/ebt/ir/samples/loops_and_gradswap.mlir --ebt-K=3 --ebt-T=2 --ebt-use-jvp=true | FileCheck models/ebt/ir/samples/loops_and_gradswap.mlir

## Run numerical check
./build/ebt_check_cpu | head -n 1 > out.txt && diff -u models/ebt/tests/numcheck_expect.txt out.txt
