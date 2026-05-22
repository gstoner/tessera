# Tessera EBT — v2.3
- Registered pipelines (scaffold C++)
- Grad-path selector pass (autodiff VJP ↔ custom JVP)
- End-to-end samples for NV/ROCm/CPU
- CSV→roofline HTML tool

## Build wrapper (scaffold)
cmake -S . -B build && cmake --build build -j

## Run end-to-end (echo + checks)
./build/tessera-ebt-opt models/ebt/ir/samples/end2end_nv.mlir -tessera-ebt-canonicalize -tessera-ebt-lower --backend=nv --ebt-K=2 --ebt-T=3 --report=reports

## Generate roofline HTML from CSV
python3 tools/report/roofline.py reports/roofline.csv reports/roofline.html
