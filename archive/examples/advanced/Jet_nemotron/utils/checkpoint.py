# utils/checkpoint.py
"""Checkpoint save/load utilities with IR hash + tuned schedule metadata (scaffold).

We save:
- 'model_state' and 'head_state' if modules expose .state_dict()/.load_state_dict()
- 'optimizer_state' if optimizer exposes .state_dict()/.load_state_dict()
- 'scaler_state' for GradScaler (scale value)
- 'scheduler_state' for LR scheduler (step, state)
- 'ir_hash' computed from model IR or config
- 'schedule_cache' metadata (export path) via tools.schedule_cache

Note: This is a scaffold; wire to real Tessera APIs in your environment.
"""
import os, json, hashlib, time
from typing import Any, Dict

def compute_ir_hash(model) -> str:
    # Try to use inspect_ir if available; fallback to hashing the model config
    try:
        blob = []
        for stage in ("graph", "sched", "tile", "target"):
            try:
                blob.append(str(model.inspect_ir(stage)))
            except Exception:
                pass
        if not blob:
            blob = [repr(getattr(model, "cfg", ""))]
        s = "\n".join(blob).encode("utf-8")
    except Exception:
        s = repr(model).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def _maybe_state_dict(obj):
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:
            return None
    return None

def _maybe_load_state_dict(obj, state):
    if state is None: return
    if hasattr(obj, "load_state_dict"):
        try:
            obj.load_state_dict(state)
        except Exception:
            pass

def save_checkpoint(path: str, *, model, head, optimizer, scaler=None, scheduler=None, schedule_cache_dir=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "created_at": time.time(),
        "ir_hash": compute_ir_hash(model),
        "model_state": _maybe_state_dict(model),
        "head_state": _maybe_state_dict(head),
        "optimizer_state": _maybe_state_dict(optimizer),
        "scaler_state": {"scale": getattr(scaler, "scale", None)} if scaler else None,
        "scheduler_state": getattr(scheduler, "state_dict", lambda: None)(),
        "schedule_cache": None,
    }
    # Schedule cache export (scaffold)
    if schedule_cache_dir:
        try:
            from tessera_jetnemotron.tools.schedule_cache import export_schedule_cache
            meta = export_schedule_cache(schedule_cache_dir)
            payload["schedule_cache"] = meta
        except Exception:
            payload["schedule_cache"] = {"path": schedule_cache_dir, "exported": False}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload["ir_hash"]

def load_checkpoint(path: str, *, model, head, optimizer=None, scaler=None, scheduler=None, schedule_cache_dir=None):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    _maybe_load_state_dict(model, payload.get("model_state"))
    _maybe_load_state_dict(head, payload.get("head_state"))
    if optimizer is not None:
        _maybe_load_state_dict(optimizer, payload.get("optimizer_state"))
    if scaler is not None and payload.get("scaler_state"):
        scaler.scale = payload["scaler_state"].get("scale", scaler.scale)
    if scheduler is not None and payload.get("scheduler_state"):
        try:
            scheduler.load_state_dict(payload["scheduler_state"])
        except Exception:
            pass
    # Import schedule cache (scaffold)
    if schedule_cache_dir and payload.get("schedule_cache"):
        try:
            from tessera_jetnemotron.tools.schedule_cache import import_schedule_cache
            import_schedule_cache(payload["schedule_cache"], schedule_cache_dir)
        except Exception:
            pass
    return payload
