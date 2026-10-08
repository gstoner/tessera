"""Compare independently recorded, correctness-gated RHS layout packets."""
from pathlib import Path
import hashlib
import json
import os
import statistics

root = Path(__file__).parent
pairs = [("row_rhs.json", "col_rhs.json"), ("row_rhs_reverse.json", "col_rhs_reverse.json")]
def key(row):
    return row["dtype"], row["kind"], row["fused"], tuple(row["shape_mkn"])
comparisons = []
for row_name, col_name in pairs:
    row, col = (json.loads((root / name).read_text()) for name in (row_name, col_name))
    if row["gpu"] != col["gpu"] or row["compiler_sha256"] != col["compiler_sha256"]:
        raise ValueError("device/compiler identity differs")
    if row["source_sha256"] != col["source_sha256"]:
        raise ValueError("matched recorder source differs")
    for path, digest in row["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("current recorder source differs: " + path)
    rr, cc = ({key(value): value for value in packet["rows"]} for packet in (row, col))
    if set(rr) != set(cc) or len(rr) != 24:
        raise ValueError("incomplete matched layout envelope")
    ratios = []
    for identity in sorted(rr):
        r, c = rr[identity], cc[identity]
        if r["rhs_layout"] != "row_major" or c["rhs_layout"] != "col_major":
            raise ValueError("wrong recorded layout")
        if r["image_digests"][0] != c["image_digests"][0]:
            raise ValueError("producer image changed across layouts")
        if r["image_digests"][1] == c["image_digests"][1]:
            raise ValueError("consumer layout images unexpectedly identical")
        ratios.append(dict(dtype=identity[0], producer=identity[1], fused=identity[2],
            shape_mkn=list(identity[3]),
            consumer_row_over_col=r["resident_event_dispatch_median_ms"]["consumer"] /
                                  c["resident_event_dispatch_median_ms"]["consumer"],
            public_row_over_col=r["warm_wall_median_ms"] / c["warm_wall_median_ms"]))
    comparisons.append(dict(row_packet=row_name, col_packet=col_name, cases=ratios,
        median_consumer_row_over_col=statistics.median(x["consumer_row_over_col"] for x in ratios),
        median_public_row_over_col=statistics.median(x["public_row_over_col"] for x in ratios)))
runtime = Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"])
result = dict(comparisons=comparisons, runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
    scope="Named static FP16/BF16 producer-to-matmul layout parity; no universal layout or format promotion")
(root / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
for value in comparisons:
    print(value["row_packet"], "consumer ratio", value["median_consumer_row_over_col"],
          "public ratio", value["median_public_row_over_col"])
