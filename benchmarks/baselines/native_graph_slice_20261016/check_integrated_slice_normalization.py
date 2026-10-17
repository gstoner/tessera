from pathlib import Path
import hashlib
import json
import random
import subprocess

w = Path("/home/angstorms/scratch/rocm-native-gate-up-views-20261016")
limit = (1 << 63) - 1
edges = [-(1 << 63), -(1 << 63) + 1, -limit // 2, -17, -1, 0, 1, 17, limit // 2, limit - 1, limit]
extents = [0, 1, 2, 17, 34, 24576, limit // 4, limit // 2, limit - 1, limit]
cases = [(n, s, z) for n in extents for s in edges for z in edges]
rng = random.Random(20261017)
cases += [
    (rng.randrange(limit + 1), rng.randrange(-limit - 1, limit + 1), rng.randrange(-limit - 1, limit + 1))
    for _ in range(100000)
]
binary = w / "slice-contract-normalize"
if not binary.exists():
    raise RuntimeError(
        "Existing compiled normalization prototype is missing; do not silently substitute a Python implementation"
    )
result = subprocess.run(
    [str(binary)], input="".join(f"{n} {s} {z}\n" for n, s, z in cases), capture_output=True, text=True, check=True
)
lines = result.stdout.splitlines()
assert len(lines) == len(cases)
for case, line in zip(cases, lines):
    n, s, z = case
    first, end, step = slice(s, s + z).indices(n)
    expected = (first, max(0, end - first))
    actual = tuple(map(int, line.split()))
    assert actual == expected, (case, actual, expected)
record = {
    "cases": len(cases),
    "passed": True,
    "oracle": "CPython slice.indices with unbounded Python start+size arithmetic; no tensor allocation",
    "cpp": "LLVM APInt signed 65-bit end-coordinate normalization",
    "extent_max": limit,
    "seed": 20261017,
    "source_sha256": hashlib.sha256((w / "slice-contract-normalize.cpp").read_bytes()).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "scope": "actual Graph SliceContract.h algorithm differential proof; compiler parser cases tested separately; no Schedule/GPU or performance proof",
}
record["contract_header_sha256"] = hashlib.sha256(
    Path(
        "/home/angstorms/scratch/sm120-relu-producer-20261014/src/compiler/ir/include/Tessera/IR/SliceContract.h"
    ).read_bytes()
).hexdigest()
(w / "integrated-slice-normalization-oracle.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
