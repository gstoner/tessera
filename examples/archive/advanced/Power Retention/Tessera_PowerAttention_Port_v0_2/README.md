# Tessera Power Attention Port (v0.1)


## v0.2 updates
- Adopted **Vidrial-style** kernel structure (static `Cfg`, staged SMEM pipelines).
- Added **Retention** op (training/inference semantics, state/sum_of_keys).
- CUDA scaffolds for retention inference (`src/kernels/cuda/retention_infer.cu`).

