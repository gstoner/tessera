
# Tessera IR Pipeline Tutorial

This compiler example demonstrates Tessera's current Python-to-IR artifact flow:

- Python API with `@tessera.jit`
- Graph IR emission through `JitFn.ir_text()`
- Schedule IR, Tile IR, and Target IR artifacts from the CPU compiler path
- A numerical check that the compiled `matmul -> relu` result matches the
  expected value

## Files

- `Tessera_IR_Pipeline_Tutorial.ipynb` - notebook version of the walkthrough.
- `tessera_ir_pipeline_demo.py` - standalone script version.
- `requirements.txt` - notebook/runtime helper dependencies.

## Quick Start

### Option A: Run the Notebook

1. Open `Tessera_IR_Pipeline_Tutorial.ipynb` in JupyterLab or VS Code.
2. Run cells top to bottom.

### Option B: Run the Script
```bash
PYTHONPATH=python python3 examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py
```

## Notes

- Run from the repo root with `PYTHONPATH=python`, or install Tessera with
  `pip install -e .`.
- The script intentionally uses the CPU-supported `matmul -> relu` path so it
  can emit all four compiler artifacts in a lightweight environment.
- The notebook and script fail if Tessera cannot import or an expected compiler
  artifact is empty. They do not substitute shim execution, illustrative IR,
  or fabricated profiling numbers.
