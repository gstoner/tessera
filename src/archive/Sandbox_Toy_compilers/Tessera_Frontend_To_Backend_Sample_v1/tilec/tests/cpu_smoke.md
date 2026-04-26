# CPU backend smoke test

```
python -m tilec.driver ../examples/matmul.tss --backend cpu --impl naive --out ../build/mm_naive
make -C ../build/mm_naive
M=64 N=64 K=64 ../build/mm_naive/mm
```
