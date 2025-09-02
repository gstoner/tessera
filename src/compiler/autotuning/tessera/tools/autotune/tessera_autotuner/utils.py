
import os, json, hashlib, platform, random, math, time
from typing import Dict, Any, Optional

try:
    import torch
except Exception:
    torch = None

def set_deterministic(seed: int = 17):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def now_iso():
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat() + "Z"

def gpu_fingerprint() -> dict:
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    # Try torch
    if torch is not None and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "gpu_name": torch.cuda.get_device_name(idx),
                "sm_count": props.multi_processor_count,
                "total_mem_bytes": getattr(props, "total_memory", None),
                "cuda": torch.version.cuda or "unknown",
            })
        except Exception:
            pass
    return info

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class Objective:
    def __init__(self, mode: str, key: str):
        self.mode = mode
        self.key = key
    @staticmethod
    def parse(s: str) -> "Objective":
        s = s.strip()
        if ":" not in s:
            raise ValueError("Objective must look like 'max:tflops' or 'min:latency_ms'")
        mode, key = s.split(":",1)
        mode = mode.strip().lower()
        key = key.strip()
        if mode not in ("max","min"):
            raise ValueError("Objective mode must be max or min")
        return Objective(mode, key)
    def better(self, a: float, b: float) -> bool:
        return (a > b) if self.mode == "max" else (a < b)
