# CUDA 13.4.1 arithmetic refresh

Owner: NUMPOL-CARRIER-1. Parent revision: `145b32777413e93d256a36860529b8060f167d26`,
with the uncommitted toolkit-selection changes identified by `sources.json`.

Super-Bear / RTX 5070 (SM120), WSL, driver 610.88, CUDA SDK 13.4.1,
nvcc V13.4.59. All 34 scalar/vector arithmetic rows pass. The four FP8 rows
check all 65,536 input byte pairs each. Bool and bounded complex probes retain
the earlier recorder's restricted envelope; this is not matrix instruction,
public frontend, general complex division or performance promotion evidence.

Unlike the historical 13.3 packets, this run explicitly supplies
`gpu-module-to-binary{toolkit=/usr/local/cuda-13.4}` to LLVM's serializer.
`arithmetic.json` records the SDK metadata, ptxas/disassembler hashes, source,
image and assembly hashes and observed instruction mnemonics. `sources.json`
identifies the modified builder and recorder; the packet pins tessera-opt.
No previous packet has been relabeled. The raw generated images and assembly
remain on Super-Bear at `/tmp/tessera-cuda1341/artifacts`; regenerate them with
this command rather than treating temporary files as durable evidence:

```sh
source scripts/_nvidia_env.sh
PYTHONPATH=python .venv/bin/python benchmarks/record_dtype_arithmetic.py \
  --backend nvidia --compiler build/tools/tessera-opt/tessera-opt \
  --llvm-bin /usr/lib/llvm-23/bin --toolkit /usr/local/cuda-13.4 \
  --artifacts /tmp/tessera-cuda1341/artifacts \
  --output /tmp/tessera-cuda1341/arithmetic.json
```

The command requires the stated assertions-enabled LLVM build and native driver.
This refresh covers arithmetic compilation/execution only. Attention, ownership,
packed formats and performance need their own workload-specific evidence.
