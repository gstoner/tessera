print('missing baseline gate; please rerun setup')
def _add_tiled_gate(base, failures, load_last_record):
    p, rec = load_last_record("runs/micro_flashattn_tiled.jsonl")
    if rec:
        key = "micro_flashattn_tiled.p50_s_max"
        max_allowed = base.get(key, None)
        if max_allowed is not None and rec.get("p50_s", 1e9) > max_allowed * 1.01:
            failures.append(f"micro_flashattn_tiled p50 {rec.get('p50_s')}s > {max_allowed}s ({p})")
