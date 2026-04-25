import time, csv, argparse, torch, json
from pathlib import Path
from tessera_kernels.tiled import flashattn_fwd_tiled

def flops_flashattn_fwd(B,H,S,D):
    # QK^T: 2*S*S*D, softmax ~ S*S, PV: 2*S*S*D → ~4*S*S*D per head
    return B*H*4.0*S*S*D

def run_case(B,H,S,D,iters=20,warmup=5,causal=True,dropout_p=0.0,seed=1234):
    dtype = torch.float16 if torch.cuda.get_device_capability(0)[0]>=8 else torch.float32
    Q = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    K = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    V = torch.randn(B,H,S,D, device="cuda", dtype=dtype)
    for _ in range(warmup): flashattn_fwd_tiled(Q,K,V,mask=None,dropout_mask=None,is_causal=causal,dropout_p=dropout_p, seed=seed)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        flashattn_fwd_tiled(Q,K,V,mask=None,dropout_mask=None,is_causal=causal,dropout_p=dropout_p, seed=seed)
        torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort()
    p50 = ts[len(ts)//2]; p95 = ts[int(len(ts)*0.95)]
    flops = flops_flashattn_fwd(B,H,S,D)
    tflops_p50 = flops/p50/1e12
    tflops_p95 = flops/p95/1e12
    # naive bytes estimate: read Q,K,V once; write O once
    bytes_io = B*H*S*D*(4 if dtype==torch.float32 else 2)* (3 + 1)
    bw_est = bytes_io / p50 / 1e9  # GB/s
    ai = flops / bytes_io  # flop/byte
    return {"p50_s":p50,"p95_s":p95,"tflops_p50":tflops_p50,"tflops_p95":tflops_p95,"gbps_est":bw_est,"ai_est":ai,"dropout_p":dropout_p,"causal":int(causal)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--seqs", type=str, default="128,256,512,1024,2048")
    ap.add_argument("--dropouts", type=str, default="0.0,0.1")
    ap.add_argument("--causals", type=str, default="0,1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dims", type=str, default="64,128,256")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out_csv", type=str, default="runs/flashattn_sweep.csv")
    ap.add_argument("--out_jsonl", type=str, default="runs/flashattn_sweep.jsonl")
    args = ap.parse_args()

    Path("runs").mkdir(parents=True, exist_ok=True)
    seqs = [int(x) for x in args.seqs.split(",") if x]
    dims = [int(x) for x in args.dims.split(",") if x]

    rows = []
    with open(args.out_csv, "w", newline="") as fcsv, open(args.out_jsonl,"a") as fj:
        writer = csv.writer(fcsv)
        writer.writerow(["B","H","S","D","dropout_p","causal","p50_s","p95_s","tflops_p50","tflops_p95","gbps_est","ai_est"])
        dropouts = [float(x) for x in args.dropouts.split(",") if x]
        causals = [bool(int(x)) for x in args.causals.split(",") if x]
        for S in seqs:
            for D in dims:
                for dp in dropouts:
                    for cz in causals:
                        res = run_case(args.B,args.H,S,D,args.iters,args.warmup,causal=cz,dropout_p=dp,seed=args.seed)
                        row = [args.B,args.H,S,D,dp,int(cz),res["p50_s"],res["p95_s"],res["tflops_p50"],res["tflops_p95"],res["gbps_est"],res["ai_est"]]
                writer.writerow(row)
                fj.write(json.dumps({"B":args.B,"H":args.H,"S":S,"D":D, **res})+"\n")
                print("S=",S,"D=",D,res)
