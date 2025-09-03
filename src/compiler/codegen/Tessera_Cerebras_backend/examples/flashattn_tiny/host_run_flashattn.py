import argparse, numpy as np, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from tessera.targets.cerebras.runtime import CerebrasRuntime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True, help="Path to wafer image (ELF)")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--scale", type=float, default=0.125)
    args = ap.parse_args()

    B, H, S, D = args.B, args.H, args.S, args.D
    Q = (np.random.rand(B,H,S,D).astype(np.float16) - 0.5)
    K = (np.random.rand(B,H,S,D).astype(np.float16) - 0.5)
    V = (np.random.rand(B,H,S,D).astype(np.float16) - 0.5)
    O = np.zeros((B,H,S,D), dtype=np.float16)

    rt = CerebrasRuntime(execution_mode="pipeline")
    result = rt.launch(args.elf, inputs={"Q": Q, "K": K, "V": V}, outputs={"O": O}, scalars={"scale": args.scale})
    print("Launch result:", result)
    print("O (preview, first item):\n", O.reshape(-1, D)[:4,:4])

if __name__ == "__main__":
    main()
