
#!/usr/bin/env python3
import argparse, json, time, math
ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--heads", type=int, default=12)
args = ap.parse_args()

# Placeholder: pretend we run a Tessera attention kernel here
# We simulate work proportional to O(seq^2 * d * heads)
ops = args.seq * args.seq * args.d * args.heads
start = time.time()
time.sleep(min(0.1, ops / 1e11))  # quick dummy
dur = time.time() - start

row = {
  "throughput_tokens_per_s": args.seq / max(dur, 1e-6),
  "latency_ms": dur * 1000.0
}
print(json.dumps(row))
