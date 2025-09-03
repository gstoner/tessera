# GEMM example (scaffold)

- `gemm.csl` — CSL-like scaffold emitted by the codegen.
- `host_run_gemm.py` — host launcher stub that shows how to pass inputs/outputs.

**Build & Run (conceptual):**
1. Compile `gemm.csl` with your SDK compiler (replace `cs_compiler` with the real command).  
   ```bash
   cs_compiler gemm.csl -o gemm.elf
   ```
2. Run with the host stub:
   ```bash
   python host_run_gemm.py --elf ./gemm.elf
   ```
