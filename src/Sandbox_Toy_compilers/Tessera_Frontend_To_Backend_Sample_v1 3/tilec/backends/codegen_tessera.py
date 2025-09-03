import pathlib, textwrap, subprocess, shutil, os
from typing import Optional
from ..ir import IRModule

TESSERA_DIALECT_HEADER = r'''
// --- Tessera Target-IR (text) ---
// (This is a textual sketch that aligns with your Tessera dialect docs.)
// We'll build a tiny module with one function and matmul/add ops lowered into
// Tessera's target ops namespace. You can adjust op names/attrs to match your repo.
'''

def _emit_func_text(f):
    lines = []
    # Symbols
    dims = f.dims or ["M","N","K"]
    sym = {k: f"%{k}" for k in dims}
    # Func header
    params = ", ".join([f"%{p}: !tessera.buffer<f32>" for p in f.params] + [f"%{d}: i32" for d in dims])
    lines.append(f"t.module {{")
    lines.append(f"  t.func @{f.name}({params}) -> !tessera.buffer<f32> {{")
    # Body: create temporaries as needed (alias output to last param if common pattern)
    for op in f.body:
        if op['op'] == 'matmul':
            # c = tessera.matmul a, b : (MxK,KxN)->(MxN)
            lines.append(f"    %{op['out']} = t.matmul %{op['lhs']}, %{op['rhs']} "
                         f"{{M = %{dims[0]}, N = %{dims[1]}, K = %{dims[2]}}} "
                         f": (!tessera.buffer<f32>, !tessera.buffer<f32>) -> !tessera.buffer<f32>")
        elif op['op'] == 'add':
            lines.append(f"    %{op['out']} = t.add %{op['lhs']}, %{op['rhs']} "
                         f": (!tessera.buffer<f32>, !tessera.buffer<f32>) -> !tessera.buffer<f32>")
    lines.append(f"    t.return %{f.ret} : !tessera.buffer<f32>")
    lines.append(f"  }}")
    lines.append(f"}}")
    return "\n".join(lines) + "\n"

def emit(ir: IRModule, out_dir: str, run_pipeline: bool = False, pipeline: Optional[str] = None, tool: Optional[str] = None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = ir.funcs[0]

    ir_text = TESSERA_DIALECT_HEADER + _emit_func_text(f)
    ir_path = out / f"{f.name}.tessera.mlir"
    ir_path.write_text(ir_text)

    # Optionally run an external pipeline if provided (your repo's tools)
    # Example: tool="mlir-opt", pipeline="--pass-pipeline='builtin.module(tessera-lower,llvm-emit)'"
    if run_pipeline and tool and pipeline:
        cmd = [tool] + pipeline.split()
        # Allow file inputs/outputs via stdin/stdout redirection
        with open(ir_path, "rb") as fin, open(out / f"{f.name}.after.mlir", "wb") as fout:
            proc = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed: {' '.join(cmd)}\n{proc.stderr.decode()}")
    # Drop a Makefile stub for downstream compilation if your tool emits C/LLVM/OBJ
    (out / "README_TESSERA_BACKEND.md").write_text(textwrap.dedent(f'''
        # Tessera backend output

        - Input textual IR: {f.name}.tessera.mlir
        - Optional: run your repo's pass pipeline to lower to LLVM/ROCDL/NVVM, e.g.:
          mlir-opt {f.name}.tessera.mlir -pass-pipeline="builtin.module(tessera-lower,convert-to-llvm)" > {f.name}.llvm.mlir

        Edit `tilec/backends/codegen_tessera.py` to match exact op/attr names in your repo.
    ''').strip()+"\n")
    return str(ir_path)
