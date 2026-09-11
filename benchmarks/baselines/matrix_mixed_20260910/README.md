# Scheduled matrix adapters and mixed capture — 2026-09-10

CUDA: RTX 5070 SM120 on Super-Bear WSL, Nsight Systems 2026.1.3.
`cuda-profiled.json` records three GEMM and three attention calls after one
warmup each. GEMM uses fp16 inputs, fp32 output and descriptor-projected
column-major B, shape M=32,K=64,N=48. Attention is fp32, B=1,H=2,Sq=8,Sk=12,
D=Dv=4. Independent NumPy oracle errors are <1.5e-7 and <2.3e-8 respectively.

`cuda-attribution.json` matches all six measured NVTX ranges to six kernels,
using owning-process/thread range containment plus successful API correlation.
Labels include run, Schedule artifact and native image digests. Two warmup
launches are explicitly excluded. Raw capture remains on Bear at
`/tmp/mixed-matrix.{nsys-rep,sqlite}`; the JSON includes the SQLite hash and all
attributed kernel/API timestamps. Sequential synchronous calls only; no claim
about arbitrary streams, nested/multi-thread ranges or concurrent completion.

ROCm: gfx1151 on Princess-Luna WSL. Both scheduled fp16 GEMM and fp16 attention
fail during native lowering: LLVM GPUFuncOpLowering creates duplicate
DictionaryAttr names in the installed assertions-enabled compiler. The two
retained stderr files include the exact pipelines and backtraces. No runtime
or numerical result is claimed for these routes. This is distinct from the
previously validated ANN/SSD lanes and the ROCm profiler kernel-record gap.

Reproduce CUDA: source the NVIDIA environment, set `TESSERA_OPT`,
`TESSERA_NVIDIA_OPT` and the PTX launch library, then wrap
`matrix_native.py --backend nvidia --nvtx` with
`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`. Extract its JSON
stdout, export SQLite, and run `attribute_mixed_cuda.py --packet <json>
--sqlite <sqlite> --output <json>`. ROCm repro uses the same adapter with
`--backend rocm --workload gemm` or `attention` and the ROCm environment.

All packet timing includes the checked host package call and numerical oracle.
No clean/bare-metal comparison or performance promotion. Source identities
are in `source-hashes.json`; compiler identity is in the CUDA packet.
