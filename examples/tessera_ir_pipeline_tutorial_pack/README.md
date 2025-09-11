
# Tessera IR Pipeline Tutorial Pack

This pack contains:
- **Tessera_IR_Pipeline_Tutorial.ipynb** — a Jupyter notebook that demonstrates a Tessera-style pipeline.  
  It attempts *real* compilation/IR dumps if `tessera` is installed; otherwise it shows illustrative IR snippets from your docs.
- **tessera_ir_pipeline_demo.py** — a standalone Python script that mirrors the notebook flow with a small shim so it runs even without `tessera`.
- **requirements.txt** — minimal Python deps for running the script and notebook cells.

## Quick Start

### Option A: Run the Notebook
1. Open `Tessera_IR_Pipeline_Tutorial.ipynb` in JupyterLab or VS Code.
2. Run cells top to bottom.  
   - If `tessera` is installed, you’ll get real compilation and (if supported) IR dumps.  
   - If not, you’ll still see realistic, illustrative IR snippets pulled from your uploaded docs.

### Option B: Run the Script
```bash
python tessera_ir_pipeline_demo.py
```

## Notes
- The shim returns plausible outputs and profiling fields so the flow is demonstrable in any environment.
- To see true IR and execution, install your Tessera Python package and ensure the underlying MLIR runtime and GPU toolchain are available.
