import argparse, pathlib
import os
from .parser import Parser
from .ir import ast_to_ir, dump_text
from .backends import codegen_c

def main():
    ap = argparse.ArgumentParser(description="TileScript → Backend sample compiler")
    ap.add_argument("input", help=".tss source file")
    ap.add_argument("--backend", choices=["c","cpu","tessera","llvm","cuda"], default="c")
    ap.add_argument("--out", default="build/out", help="output directory (created if missing)")
    ap.add_argument("--dump-ir", action="store_true", help="print textual IR")
    ap.add_argument("--impl", choices=["naive","openmp","blas","avx2"], default="naive"); args = ap.parse_args()

    src = pathlib.Path(args.input).read_text()
    mod = Parser(src).parse()
    irmod = ast_to_ir(mod)

    if args.dump_ir:
        print(dump_text(irmod))

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "c":
        cfile = codegen_c.emit(irmod, str(out_dir))
        print(f"[tilec] generated C backend: {cfile}")
        print(f"[tilec] next: make -C {out_dir}")
    elif args.backend == "cpu":
            from .backends import codegen_cpu
            cfile = codegen_cpu.emit(irmod, str(out_dir), impl=args.impl)
            print(f"[tilec] generated CPU backend ({args.impl}): {cfile}")
            print(f"[tilec] next: make -C {out_dir}")
        
        elif args.backend == "tessera":
            from .backends import codegen_tessera as tz
            # Allow optional pipeline via env to stay simple
            run_pipe = bool(os.environ.get("TESSERA_RUN_PIPE"))
            tool = os.environ.get("TESSERA_TOOL")  # e.g., mlir-opt or your tessera-opt
            pipe = os.environ.get("TESSERA_PIPELINE")  # e.g., --pass-pipeline='...'
            mlir_file = tz.emit(irmod, str(out_dir), run_pipeline=run_pipe, pipeline=pipe, tool=tool)
            print(f"[tilec] tessera IR -> {mlir_file}")
        elif args.backend == "rocm":
            from .backends import codegen_rocm
            arch = os.environ.get("ARCH")
            codegen_rocm.emit(irmod, str(out_dir), impl=args.impl, arch=arch)
        elif args.backend == "nvidia":
            from .backends import codegen_nvidia
            arch = os.environ.get("ARCH")
            codegen_nvidia.emit(irmod, str(out_dir), impl=args.impl, arch=arch)
        elif args.backend == "llvm":
        from .backends import codegen_llvm_stub as llvm
        llvm.emit(irmod, str(out_dir))
    else:
        from .backends import codegen_cuda_stub as cuda
        cuda.emit(irmod, str(out_dir))

if __name__ == "__main__":
    main()
