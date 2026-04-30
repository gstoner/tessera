from __future__ import annotations

import json
from pathlib import Path
import random
import time


def load_config() -> dict[str, int]:
    cfg_path = Path("candidate_config.json")
    if not cfg_path.exists():
        return {"block_m": 16, "block_n": 16, "unroll": 1}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def reference(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k, n = len(a), len(a[0]), len(b[0])
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for p in range(k):
            av = a[i][p]
            for j in range(n):
                out[i][j] += av * b[p][j]
    return out


def tiled(a: list[list[float]], b: list[list[float]], cfg: dict[str, int]) -> list[list[float]]:
    m, k, n = len(a), len(a[0]), len(b[0])
    bm, bn = cfg["block_m"], cfg["block_n"]
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for ii in range(0, m, bm):
        for jj in range(0, n, bn):
            for p in range(k):
                for i in range(ii, min(ii + bm, m)):
                    av = a[i][p]
                    row = out[i]
                    brow = b[p]
                    for j in range(jj, min(jj + bn, n)):
                        row[j] += av * brow[j]
    return out


def main() -> None:
    rng = random.Random(3)
    cfg = load_config()
    m = n = k = 48
    a = [[rng.random() for _ in range(k)] for _ in range(m)]
    b = [[rng.random() for _ in range(n)] for _ in range(k)]

    expected = reference(a, b)
    start = time.perf_counter()
    actual = tiled(a, b, cfg)
    runtime_s = time.perf_counter() - start
    max_err = max(abs(expected[i][j] - actual[i][j]) for i in range(m) for j in range(n))
    correct = max_err < 1e-9
    tile_area = cfg["block_m"] * cfg["block_n"] * cfg["unroll"]
    occupancy_proxy = min(tile_area / 4096.0, 1.0)
    score = (1.0 / max(runtime_s, 1e-9)) * occupancy_proxy if correct else 0.0
    print(json.dumps({"correct": correct, "max_err": max_err, "runtime_s": runtime_s, "score": score, "config": cfg}))


if __name__ == "__main__":
    main()
