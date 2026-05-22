import argparse, pathlib
from .parser import Parser
from .ir import ast_to_ir, dump_text
from .backends import codegen_c

def main():
    ap = argparse.ArgumentParser(description="TileScript → Backend sample compiler")
    ap.add_argument("input", help=".tss source file")
    ap.add_argument("--backend", choices=["c","llvm","cuda"], default="c")
    ap.add_argument("--out", default="build/out", help="output directory (created if missing)")
    ap.add_argument("--dump-ir", action="store_true", help="print textual IR")
    args = ap.parse_args()

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
    elif args.backend == "llvm":
        from .backends import codegen_llvm_stub as llvm
        llvm.emit(irmod, str(out_dir))
    else:
        from .backends import codegen_cuda_stub as cuda
        cuda.emit(irmod, str(out_dir))

if __name__ == "__main__":
    main()
