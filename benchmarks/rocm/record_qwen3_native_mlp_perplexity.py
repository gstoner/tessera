"""Stage 2 of the quantized-MLP model-quality gate: Qwen3-8B perplexity.

Reference/diagnostic driver (Decision #23): Torch runs the pinned BF16 model
on CPU in the separate reference environment and never imports Tessera. Every
decoder layer's MLP is served by one arm, each with its own residual stream
from the embeddings to the logits. ``bf16`` and ``shipped`` are always run;
``--arms`` adds candidates:

* ``bf16``: the upstream Torch MLP, unmodified;
* ``shipped``: the decoded shipped NVFP4 weights, FP64 products, no activation
  quantization (the format's own loss);
* ``native``: the public seven-member native quantized MLP on the live owning
  ROCm architecture (gfx1201 or gfx1151);
* ``mxfp4_direct`` / ``mxfp4_ingested``: weight-only MXFP4 host references,
  BF16 quantized once versus the shipped NVFP4 requantized (see the worker).

Every arm but ``bf16`` runs in ``qwen3_native_mlp_worker`` (the Tessera
environment) over a pipe. Layers execute layer-major (every window through layer i before
layer i+1), so the worker holds one layer's weights at a time. The split
decoder loop is checked against the stock model: the ``bf16`` arm's logits
must equal ``model(input_ids)`` bitwise for every window.

Acceptance (fixed before measurement): with ``delta(arm) = ppl(arm) -
ppl(bf16)``, a candidate arm passes only when ``delta(shipped) > 0`` and
``delta(arm) <= THRESHOLD_RATIO * delta(shipped)``. The same ratio applies to
every candidate on every architecture (fixed 2026-11-02 for the gfx1151 native
arm and both MXFP4 arms, before either was measured).

The ``native_mxfp4`` arm (ROCM-MXFP4-NATIVE-MLP-20261102) is measured against
its own weights rather than the shipped NVFP4 ones: it passes only when
``delta(mxfp4_direct) > 0`` and ``delta(native_mxfp4) <= THRESHOLD_RATIO *
delta(mxfp4_direct)``, so the gate isolates what native execution and the
E4M3 activation quantizer cost on top of the MXFP4 weights. It requires the
``mxfp4_direct`` arm in the same run (fixed 2026-11-02, before measuring).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_qwen3_all_layer_mlp_inputs as capture  # noqa: E402
import qwen3_local_checkpoint as local  # noqa: E402
import qwen3_native_mlp_worker as worker  # noqa: E402

SCHEMA = "tessera.quality.qwen3_native_mlp_perplexity.v3"
THRESHOLD_RATIO = 1.5
BASELINE_ARMS = ("bf16", "shipped")
CANDIDATE_ARMS = ("native", "native_mxfp4", "mxfp4_direct", "mxfp4_ingested")
NATIVE_ARMS = ("native", "native_mxfp4")
#: The arm each candidate's delta is compared with; ``shipped`` unless named.
REFERENCE_ARM = {"native_mxfp4": "mxfp4_direct"}
ARMS = ("bf16", "shipped", "native")


def parse_arms(text: str) -> tuple[str, ...]:
    """``bf16`` and ``shipped`` first, then at least one distinct candidate."""
    candidates = tuple(arm for arm in text.split(",") if arm)
    if not candidates or len(set(candidates)) != len(candidates) \
            or any(arm not in CANDIDATE_ARMS for arm in candidates):
        raise ValueError(f"candidate arms must be distinct members of {CANDIDATE_ARMS}")
    for arm in candidates:
        if REFERENCE_ARM.get(arm, "shipped") not in BASELINE_ARMS + candidates:
            raise ValueError(f"{arm} is gated against {REFERENCE_ARM[arm]}, which must run too")
    return BASELINE_ARMS + candidates


def verdict(ppl: dict[str, float], ratio: float = THRESHOLD_RATIO, arm: str = "native",
            reference: str | None = None) -> dict:
    """Apply the pre-registered acceptance rule to one candidate arm."""
    reference = reference or REFERENCE_ARM.get(arm, "shipped")
    if not all(math.isfinite(ppl[name]) and ppl[name] > 0 for name in (*BASELINE_ARMS, reference, arm)):
        raise ValueError("perplexity must be finite and positive for every arm")
    base, candidate = ppl[reference] - ppl["bf16"], ppl[arm] - ppl["bf16"]
    common = {"arm": arm, "reference": reference, f"delta_{reference}": base, f"delta_{arm}": candidate,
              "threshold_ratio": ratio}
    if base <= 0:
        return {**common, "ratio": None, "passed": False,
                "reason": f"{reference} does not raise perplexity; the ratio is undefined"}
    return {**common, "ratio": candidate / base, "passed": candidate <= ratio * base,
            "reason": f"delta({arm}) <= threshold_ratio * delta({reference})"}


class Worker:
    def __init__(self, python: str, repo: Path, cache_root: Path, certify: str, arms: tuple[str, ...]) -> None:
        self.proc = subprocess.Popen(
            [python, "-m", "benchmarks.rocm.qwen3_native_mlp_worker", "--cache-root", str(cache_root),
             "--certify", certify, "--arms", ",".join(arms)],
            cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def call(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        try:
            worker.write_frame(self.proc.stdin, header, payload)
            reply, data = worker.read_frame(self.proc.stdout)
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError(f"native worker died during {header.get('op')!r} "
                               f"(exit {self.proc.wait()}); see its stderr") from exc
        if reply.get("ok") is not True:
            raise RuntimeError(f"native worker refused {header.get('op')!r}: {reply}")
        return reply, data

    def close(self) -> dict:
        reply, _ = self.call({"op": "close"})
        if self.proc.wait() != 0:
            raise RuntimeError("native worker exited non-zero after close")
        return reply


def _nll(logits, targets) -> np.ndarray:
    import torch

    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    return (-logp.gather(1, targets[1:, None])[:, 0]).double().numpy()


def _relative_rms(reference, candidate) -> float:
    ref = reference.double()
    return float(((ref - candidate.double()).square().sum() / ref.square().sum()).sqrt())


def record(cache_root: Path, corpus_path: Path, corpus_revision: str, output: Path, worker_python: str,
           sequences: int, length: int, threads: int, certify: str, arms: tuple[str, ...] = ARMS) -> dict:
    if arms[:2] != BASELINE_ARMS:
        raise ValueError("arms must start with bf16, shipped")
    served = arms[1:]
    if length % worker.CHUNK or length <= 1:
        raise ValueError(f"window length must be a multiple of {worker.CHUNK}")
    import tokenizers
    import torch
    import transformers
    from transformers.masking_utils import create_causal_mask

    torch.set_num_threads(threads)
    started = time.perf_counter()
    root = local.snapshot_dir(cache_root, local.BF16_REPO, local.BF16_REVISION)
    snapshot = local.LocalSnapshot(local.BF16_REPO, local.BF16_REVISION, root)
    shard_sha = {shard: snapshot.verify_shard(shard) for shard in sorted(set(snapshot.weight_map.values()))}
    tokenizer = tokenizers.Tokenizer.from_str((root / "tokenizer.json").read_text())
    token_ids, corpus = capture.corpus_windows(tokenizer, corpus_path, sequences, length)
    corpus = {"repo": capture.CORPUS_REPO, "revision": corpus_revision, **corpus}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        root, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    model.eval().requires_grad_(False)
    body, layers = model.model, model.model.layers
    if body.has_sliding_layers or any(layer.attention_type != "full_attention" for layer in layers):
        raise RuntimeError("the split decoder loop handles full-attention layers only")
    repo = Path(__file__).resolve().parents[2]
    pool = Worker(worker_python, repo, cache_root, certify, served)
    identity, _ = pool.call({"op": "hello"})

    with torch.inference_mode():
        ids = [torch.from_numpy(row) for row in token_ids]
        stock = [hashlib.sha256(model(row[None, :], use_cache=False).logits.view(torch.int16).numpy().tobytes())
                 .hexdigest() for row in ids]
        hidden = {arm: [body.embed_tokens(row[None, :]) for row in ids] for arm in arms}
        position = torch.arange(length)[None, :]
        cache_position = torch.arange(length)
        rotary = body.rotary_emb(hidden["bf16"][0], position)
        mask = create_causal_mask(config=body.config, input_embeds=hidden["bf16"][0], attention_mask=None,
                                  cache_position=cache_position, past_key_values=None, position_ids=position)
        drift: list[dict] = []
        sources, weight_error = [], []
        for index, layer in enumerate(layers):
            reply, _ = pool.call({"op": "layer", "layer": index})
            sources.append(reply["source"])
            weight_error.append({"layer": index, **reply["weight_relative_rms_vs_bf16"]})
            mlp_inputs = {}
            for arm in arms:
                mlp_inputs[arm] = []
                for w, state in enumerate(hidden[arm]):
                    attn, _ = layer.self_attn(hidden_states=layer.input_layernorm(state), attention_mask=mask,
                                              position_ids=position, past_key_values=None, use_cache=False,
                                              cache_position=cache_position, position_embeddings=rotary)
                    state = state + attn
                    x = layer.post_attention_layernorm(state)
                    mlp_inputs[arm].append(x)
                    if arm == "bf16":
                        out = layer.mlp(x)
                    else:
                        bits = x.contiguous().view(torch.int16).numpy().tobytes()
                        _, data = pool.call({"op": "mlp", "arm": arm, "layer": index, "rows": length}, bits)
                        out = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16)
                        out = out.reshape(x.shape)
                    hidden[arm][w] = state + out
            drift.append({"layer": index, **{
                f"mlp_input_{arm}_vs_bf16": _relative_rms(torch.cat(mlp_inputs["bf16"]), torch.cat(mlp_inputs[arm]))
                for arm in served}})
            print(json.dumps({"layer": index, "elapsed_s": round(time.perf_counter() - started, 1),
                              **{k: round(v, 5) for k, v in drift[-1].items() if k != "layer"}}), flush=True)
        nll, split = {}, []
        for arm in arms:
            nll[arm] = []
            for w, state in enumerate(hidden[arm]):
                logits = model.lm_head(body.norm(state))
                if arm == "bf16":
                    split.append(hashlib.sha256(logits.view(torch.int16).numpy().tobytes()).hexdigest())
                nll[arm].append(_nll(logits, ids[w]))
    if split != stock:
        raise RuntimeError("the split decoder loop does not reproduce the stock model's logits")
    worker_summary = pool.close()
    if "tessera" in sys.modules:
        raise RuntimeError("the Torch driver imported Tessera")

    per_window = {arm: [float(np.exp(v.mean())) for v in nll[arm]] for arm in arms}
    ppl = {arm: float(np.exp(np.concatenate(nll[arm]).mean())) for arm in arms}
    diff = {arm: np.concatenate(nll[arm]) - np.concatenate(nll["bf16"]) for arm in served}
    verdicts = {arm: verdict(ppl, arm=arm) for arm in arms[2:]}
    diagnostics = {}
    if {"mxfp4_direct", "mxfp4_ingested"} <= set(arms) and ppl["mxfp4_direct"] > ppl["bf16"]:
        # Not gated: how much of the ingested arm's cost is the double rounding.
        diagnostics["delta_ingested_over_delta_direct"] = \
            (ppl["mxfp4_ingested"] - ppl["bf16"]) / (ppl["mxfp4_direct"] - ppl["bf16"])
    if "native_mxfp4" in arms and ppl["shipped"] > ppl["bf16"]:
        # Not gated: the native MXFP4 MLP against the shipped NVFP4 weights.
        diagnostics["delta_native_mxfp4_over_delta_shipped"] = \
            (ppl["native_mxfp4"] - ppl["bf16"]) / (ppl["shipped"] - ppl["bf16"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "token_nll.npz", **{arm: np.stack(nll[arm]) for arm in arms}, token_ids=token_ids)
    result = {
        "schema": SCHEMA, **{k: identity[k] for k in ("architecture", "device_uuid", "device_name")},
        "host": socket.gethostname(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **{k: identity[k] for k in ("worker_sha256", "compiler_sha256", "provider_sha256")},
        "model": {"repo": local.BF16_REPO, "revision": local.BF16_REVISION, "bf16_shard_sha256": shard_sha},
        "nvfp4": {"repo": local.NVFP4_REPO, "revision": local.NVFP4_REVISION,
                  "index_sha256": sources[0]["nvfp4_index_sha256"]},
        "corpus": corpus, "token_ids_sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "scored_tokens": int(sum(v.size for v in nll["bf16"])),
        "arms": list(arms),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "numpy": np.__version__},
        "split_loop_matches_stock_logits": True, "stock_logits_sha256": stock,
        "perplexity": ppl, "per_window_perplexity": per_window,
        "mean_token_nll_delta_vs_bf16": {arm: float(v.mean()) for arm, v in diff.items()},
        "verdicts": verdicts, "diagnostics": diagnostics, "mlp_input_drift": drift,
        "weight_relative_rms_vs_bf16": weight_error, "worker": worker_summary,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "scope": "MLP substitution only; attention, norms and lm_head stay BF16 Torch. The shipped and "
                 "mxfp4_* arms are weight-only host references (no activation quantization), not the "
                 "checkpoint's W4A4 deployment recipe. The native arms execute W4A8 on the device.",
    }
    result["model_quality_acceptance"] = all(v["passed"] for v in verdicts.values()) \
        and (not set(NATIVE_ARMS) & set(arms) or certify == "every-layer")
    (output / "perplexity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"perplexity": ppl, "verdicts": verdicts, "diagnostics": diagnostics}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-python", required=True, help="the Tessera project environment's python")
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--certify", choices=("every-layer", "first-layer", "none"), default="every-layer")
    parser.add_argument("--arms", default="native", help="comma-separated candidate arms; bf16, shipped always run")
    args = parser.parse_args()
    record(args.cache_root, args.corpus, args.corpus_revision, args.output, args.worker_python,
           args.sequences, args.length, args.threads, args.certify, parse_arms(args.arms))
