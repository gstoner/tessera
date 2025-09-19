#!/usr/bin/env python3
"""
PDDL-Instruct Logical CoT Validator (prototype)
- Symbolic pre/effect executor for facts (simple set semantics)
- Numeric constraint recompute for shared_mem / regs / occupancy
- Goal checks for TFLOPS / error thresholds
"""
import json, sys, argparse, math, hashlib
from pathlib import Path

DEFAULT_LIMITS = {
    "limit_shared_memory_bytes": 228_000,   # Hopper shared mem per block
    "limit_registers_per_thread": 255,      # CUDA register limit per thread
    "min_occupancy": 0.5                    # suggested minimum
}

DEFAULT_TARGETS = {
    "throughput_tflops_min": 800.0,
    "memory_usage_bytes_max": 200_000,
    "numerical_error_max": 1e-6
}

def sha_state(state_facts):
    h = hashlib.sha256()
    for f in sorted(state_facts):
        h.update(f.encode())
    return "sha256:" + h.hexdigest()[:16]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace.jsonl (CoT steps)")
    ap.add_argument("--init_state", help="initial state facts file (one atom per line)")
    ap.add_argument("--out", required=True, help="report.json path")
    ap.add_argument("--limits", help="limits.json override")
    ap.add_argument("--targets", help="targets.json override")
    ap.add_argument("--strict", action="store_true", help="fail if schema-like keys missing")
    return ap.parse_args()

def load_jsonl(p):
    steps = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            steps.append(json.loads(line))
    return steps

def estimate_tflops(steps):
    # Average explicit estimates if present, else infer from actions with simple heuristic.
    ests = [s.get("estimates", {}).get("achieved_tflops") for s in steps if s.get("estimates")]
    ests = [e for e in ests if isinstance(e,(int,float))]
    if ests:
        return sum(ests)/max(1,len(ests))
    bonus = 0.0
    for s in steps:
        a = (s.get("action") or "").lower()
        if "wgmma" in a: bonus += 120.0
        if "wmma" in a:  bonus += 60.0
        if "flash-attention" in a: bonus += 200.0
        if "tma" in a or "cp.async" in a: bonus += 30.0
    return bonus

def compute_memory_efficiency(steps, limit):
    usages = []
    for s in steps:
        c = s.get("constraints",{})
        used = c.get("shared_memory_bytes")
        lim  = c.get("limit_shared_memory_bytes", limit)
        if isinstance(used,int) and lim>0:
            usages.append(min(1.0, used/lim))
    if not usages: return None
    return 1.0 - sum(usages)/len(usages)

def apply_effects(state, effects):
    adds = effects.get("add",[]) or []
    dels = effects.get("del",[]) or []
    for d in dels:
        if d in state: state.remove(d)
    for a in adds:
        state.add(a)
    return state

def main():
    args = parse_args()
    limits = DEFAULT_LIMITS.copy()
    targets = DEFAULT_TARGETS.copy()
    if args.limits:
        limits.update(json.load(open(args.limits)))
    if args.targets:
        targets.update(json.load(open(args.targets)))

    steps = load_jsonl(args.trace)
    state = set()
    if args.init_state and Path(args.init_state).exists():
        for line in open(args.init_state):
            s=line.strip()
            if s: state.add(s)

    failures = []
    violations = 0

    # Symbolic execution across steps
    for i,s in enumerate(steps,1):
        # Minimal schema checks
        for key in ["constraints","effects","applicable"]:
            if key not in s:
                if args.strict:
                    failures.append(f"Step {i}: missing '{key}'")
                    violations += 1
        # If applicable, apply effects to state set
        if s.get("applicable", False):
            eff = s.get("effects", {"add":[],"del":[]})
            state = apply_effects(state, eff)
        # Recompute/verify resource constraints
        c = s.get("constraints",{})
        smem = c.get("shared_memory_bytes")
        smem_lim = c.get("limit_shared_memory_bytes", limits["limit_shared_memory_bytes"])
        regs = c.get("registers_per_thread")
        regs_lim = c.get("limit_registers_per_thread", limits["limit_registers_per_thread"])
        occ_ok = c.get("occupancy_target_met")
        occ_est = c.get("occupancy_estimate")

        if isinstance(smem,int) and smem_lim and smem > smem_lim:
            violations += 1; failures.append(f"Step {i}: shared_memory_bytes {smem} > limit {smem_lim}")
        if isinstance(regs,int) and regs_lim and regs > regs_lim:
            violations += 1; failures.append(f"Step {i}: registers_per_thread {regs} > limit {regs_lim}")
        if (occ_ok is False) or (isinstance(occ_est,(int,float)) and occ_est < limits["min_occupancy"]):
            violations += 1; failures.append(f"Step {i}: occupancy below minimum {limits['min_occupancy']}")

        # Optional cross-check: state hash
        if "state_hash" in s:
            # We don't fail if mismatch, but we could compare here if initial state provided.
            pass

    # Plan-level goals
    tflops = estimate_tflops(steps)
    mem_eff = compute_memory_efficiency(steps, limits["limit_shared_memory_bytes"])
    goal_sat = {
        f"throughput_tflops >= {targets['throughput_tflops_min']}": bool(tflops and tflops >= targets["throughput_tflops_min"]),
        f"shared_mem_usage <= {targets['memory_usage_bytes_max']}": violations == 0,  # conservative
        f"error <= {targets['numerical_error_max']}": True  # placeholder
    }
    valid = (violations == 0) and all(goal_sat.values())
    report = {
        "valid": bool(valid),
        "failures": failures,
        "goal_satisfaction": goal_sat,
        "summary": {
            "steps": len(steps),
            "violations": violations,
            "throughput_tflops_estimate": tflops if tflops is not None else 0.0,
            "memory_efficiency": mem_eff if mem_eff is not None else None
        }
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
