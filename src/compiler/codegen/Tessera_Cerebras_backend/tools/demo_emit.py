import os, sys, pathlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tessera.targets.cerebras.lowering import compile_to_cerebras

def main():
    gemm = {
        "kind": "gemm",
        "name": "gemm_kernel",
        "params": {"M": 256, "N": 256, "K": 256, "tile_m": 64, "tile_n": 64, "tile_k": 32},
        "grid": (64,64),
        "regions": (4,4),
    }
    artifacts = compile_to_cerebras(gemm, execution_mode="pipeline")
    outdir = pathlib.Path(__file__).parent.parent / "examples" / "gemm"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gemm.csl").write_text(artifacts["csl"])
    (outdir / "layout.json").write_text(artifacts["layout.json"])
    print("Wrote:", outdir / "gemm.csl")
    print("Wrote:", outdir / "layout.json")

    fa = {
        "kind": "flashattn_tiny",
        "name": "flashattn_tiny",
        "params": {"B": 1, "H": 2, "S": 128, "D": 64, "BK": 64},
        "grid": (64,64),
        "regions": (4,4),
    }
    artifacts2 = compile_to_cerebras(fa, execution_mode="pipeline")
    outdir2 = pathlib.Path(__file__).parent.parent / "examples" / "flashattn_tiny"
    outdir2.mkdir(parents=True, exist_ok=True)
    (outdir2 / "flashattn_tiny.csl").write_text(artifacts2["csl"])
    (outdir2 / "layout.json").write_text(artifacts2["layout.json"])
    print("Wrote:", outdir2 / "flashattn_tiny.csl")
    print("Wrote:", outdir2 / "layout.json")

if __name__ == "__main__":
    main()
