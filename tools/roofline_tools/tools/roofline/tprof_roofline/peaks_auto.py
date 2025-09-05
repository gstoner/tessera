import json, subprocess, yaml, sys
from typing import Optional

TEMPLATE = {
    "name": "Auto Device",
    "hbm_bw_GBps": 900.0,
    "compute_peaks_GFLOPs": {"fp32": 60000.0, "bf16_tensor": 100000.0, "fp16_tensor": 120000.0},
    "links": [
        {"name": "NVLink (per-link)", "bw_GBps": 50.0},
        {"name": "PCIe Gen5 x16", "bw_GBps": 64.0},
        {"name": "400G NIC (1 dir)", "bw_GBps": 50.0}
    ]
}

def try_tprof_json() -> Optional[dict]:
    for cmd in (["tprof", "peaks", "--json"], ["tprof", "peaks", "-j"]):
        try:
            out = subprocess.check_output(cmd, timeout=2.0)
            return json.loads(out)
        except Exception:
            continue
    return None

def json_to_yaml_peaks(obj: dict) -> dict:
    d = dict(TEMPLATE)
    if "device" in obj:
        d["name"] = obj["device"].get("name", d["name"])
    if "memory" in obj:
        d["hbm_bw_GBps"] = float(obj["memory"].get("hbm_bw_GBps", d["hbm_bw_GBps"]))
    if "compute" in obj and isinstance(obj["compute"], dict):
        d["compute_peaks_GFLOPs"] = {k: float(v) for k,v in obj["compute"].items()}
    if "links" in obj and isinstance(obj["links"], list):
        d["links"] = [{"name": L.get("name","link"), "bw_GBps": float(L.get("bw_GBps", 0.0))} for L in obj["links"]]
    return d

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "auto_peaks.yaml"
    obj = try_tprof_json()
    y = json_to_yaml_peaks(obj) if obj else TEMPLATE
    with open(out_path, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
