"""apple_gpu Gumiho speculative-decoding benchmark.

Two measurements, both emitted in the standard ``benchmark_gemm.py`` JSON schema
(roofline-ingestible):

1. **Algorithmic speculative speedup** (``op="gumiho_decode"``) — tokens
   committed per target forward pass for vanilla autoregressive decode (1.0) vs.
   speculative decode before and after distilling the draft. This is the real
   speculative-decoding win and is **model-size independent**: at toy scale the
   draft is no cheaper than the target, so wall-clock won't show it, but the
   ``tokens_per_step`` ratio is exactly what scales to a real draft≪target gap.

2. **Resident vs per-op serial draft** (``op="gumiho_serial_draft"``) — wall-clock
   of the serial draft run GPU-resident (one command buffer per token) vs. the
   per-op path (one dispatch + sync per op). This sync-reduction **is** a real
   wall-clock win even at toy scale, because per-op CPU↔GPU sync dominates
   small-batch decode.

Best-effort: runs only on Darwin with Metal; elsewhere it writes an empty run
list and exits 0.

Usage:
    python benchmarks/apple_gpu/benchmark_gumiho.py --reps 5 --max-new-tokens 24
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "gumiho"
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from gumiho.backend import NumpyBackend, make_backend  # noqa: E402
from gumiho.config import tiny_config  # noqa: E402
from gumiho.decode import run_multistep_decode  # noqa: E402
from gumiho.model import SerialHead, TargetModel, make_weights  # noqa: E402
from gumiho.resident import ResidentSerialDraft  # noqa: E402
from gumiho.training import distill, trajectory_contexts  # noqa: E402


def _median_ms(fn, reps: int) -> tuple[float, float]:
    fn()  # warm up
    s = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        s.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(s), (statistics.stdev(s) if reps > 1 else 0.0)


def _vanilla_decode(cfg, weights, prompts, n, backend):
    """Plain autoregressive decode — one target forward per token."""
    tgt = TargetModel(weights, cfg)
    total = 0
    for prompt in prompts:
        seq = list(int(t) for t in prompt)
        for _ in range(n):
            ctx = np.asarray(seq[-cfg.context_len:], np.int64)
            _h, logits = tgt.forward(backend, ctx)
            seq.append(int(np.argmax(logits[-1])))
            total += 1
    return total


def _device_name() -> str:
    return "apple_silicon_metal" if sys.platform == "darwin" else "non-darwin-fallback"


def _tessera_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("tessera")
    except Exception:
        return "dev"


def _row(op, mode, shape, ms, stdev, reps, device, version, **extra):
    row = {
        "backend": "apple_gpu", "op": op, "shape": shape, "dtype": "f32",
        "mode": mode, "reps": reps, "latency_ms": ms, "stdev_ms": stdev,
        "tflops": 0.0, "memory_bw_gb_s": 0.0,
        "device": device, "tessera_version": version,
    }
    row.update(extra)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--target", default="apple_gpu",
                        choices=["apple_gpu", "apple_cpu", "numpy"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin" and args.target != "numpy":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "non-Darwin host — no Metal"},
                indent=2, sort_keys=True))
        print("gumiho benchmark: skipping (non-Darwin host)", file=sys.stderr)
        return 0

    cfg = tiny_config()
    device, version = _device_name(), _tessera_version()
    shape = (f"v{cfg.vocab}_d{cfg.d_model}_s{cfg.serial_tokens}"
             f"_p{cfg.parallel_heads}_t{args.max_new_tokens}")
    rows: list[dict[str, Any]] = []

    weights = make_weights(cfg, seed=0)
    rng = np.random.default_rng(0)
    prompts = rng.integers(0, cfg.vocab, size=(args.prompts, cfg.context_len),
                           dtype=np.int64)
    ctxs = trajectory_contexts(cfg, weights, prompts, args.max_new_tokens)
    trained = distill(cfg, weights, contexts=ctxs, serial_steps=args.train_steps,
                      parallel_steps=args.train_steps, lr=0.05, seed=0)

    # ── 1. Algorithmic speculative speedup (tokens per target pass) ──────────
    n = args.max_new_tokens
    ref = NumpyBackend(eps=cfg.rmsnorm_eps)
    vbackend = make_backend(args.target, eps=cfg.rmsnorm_eps)

    vms, vsd = _median_ms(
        lambda: _vanilla_decode(cfg, weights, prompts, n, vbackend), args.reps)
    vtoks = args.prompts * n
    rows.append(_row("gumiho_decode", "vanilla", shape, vms, vsd, args.reps,
                     device, version, tokens_per_step=1.0, speedup_vs_vanilla=1.0,
                     tokens_per_sec=vtoks / (vms / 1000.0) if vms else 0.0))

    for mode, w in (("speculative_untrained", weights),
                    ("speculative_trained", trained)):
        holder: dict[str, Any] = {}

        def _run(_w=w, _h=holder):
            _h["m"] = run_multistep_decode(cfg, _w, prompts=prompts,
                                           max_new_tokens=n, target=args.target,
                                           trained=(_w is trained))
        ms, sd = _median_ms(_run, args.reps)
        m = holder["m"]
        tps = m.tokens_per_step
        tok_s = (m.tokens_generated / (ms / 1000.0)) if ms else 0.0
        rows.append(_row("gumiho_decode", mode, shape, ms, sd, args.reps,
                         device, version, tokens_per_step=tps,
                         speedup_vs_vanilla=m.speedup_vs_vanilla,
                         mean_accepted_length=m.mean_accepted_length,
                         tokens_per_sec=tok_s))

    # ── 2. Resident vs per-op serial draft (wall-clock sync reduction) ───────
    tgt = TargetModel(trained, cfg)
    last_hidden, _ = tgt.forward(ref, prompts[0].astype(np.int64))
    h_init, root = last_hidden[-1], int(prompts[0][-1])

    rd = ResidentSerialDraft(trained, cfg)
    sh = SerialHead(trained, cfg)
    op_backend = make_backend(args.target, eps=cfg.rmsnorm_eps)

    rms, rsd = _median_ms(lambda: rd.generate(h_init, root), args.reps)
    pms, psd = _median_ms(
        lambda: sh.generate(op_backend, tgt, h_init, root), args.reps)
    sshape = f"d{cfg.d_model}_layers{cfg.serial_layers}_steps{cfg.serial_tokens}"
    speedup = pms / rms if rms else 0.0
    rows.append(_row("gumiho_serial_draft", "resident", sshape, rms, rsd,
                     args.reps, device, version, command_buffers=cfg.serial_tokens,
                     speedup_vs_per_op=speedup, backend_path=rd._metal and "metal" or "numpy"))
    rows.append(_row("gumiho_serial_draft", "per_op", sshape, pms, psd,
                     args.reps, device, version,
                     ops_per_step=rd.ops_per_step, speedup_vs_per_op=1.0))
    rd.free()

    payload = {"runs": rows}
    out = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
