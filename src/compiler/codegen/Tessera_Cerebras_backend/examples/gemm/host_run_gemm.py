import argparse, numpy as np, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from tessera.targets.cerebras.runtime import CerebrasRuntime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", required=True, help="Path to wafer image (ELF)")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    args = ap.parse_args()

    M, N, K = args.M, args.N, args.K
    A = (np.random.rand(M, K).astype(np.float16) - 0.5)
    B = (np.random.rand(K, N).astype(np.float16) - 0.5)
    C = np.zeros((M, N), dtype=np.float16)

    rt = CerebrasRuntime(execution_mode="pipeline")
    result = rt.launch(args.elf, inputs={"A": A, "B": B}, outputs={"C": C}, scalars={})
    print("Launch result:", result)
    # In a real run, C would be filled by the wafer program.
    print("C (preview, first 4x4):\n", C[:4,:4])

if __name__ == "__main__":
    main()
