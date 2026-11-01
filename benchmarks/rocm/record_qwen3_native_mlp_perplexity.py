"""Stage 2 of the quantized-MLP model-quality gate: Qwen3-8B perplexity, gfx1201.

Reference/diagnostic driver (Decision #23): Torch runs the pinned BF16 model
on CPU in the separate reference environment and never imports Tessera. Every
decoder layer's MLP is served by one of three arms, each with its own residual
stream from the embeddings to the logits:

* ``bf16``: the upstream Torch MLP, unmodified;
* ``shipped``: the decoded shipped NVFP4 weights, FP64 products, no activation
  quantization (the format's own loss);
* ``native``: the public seven-member native quantized MLP on gfx1201.

The last two run in ``qwen3_native_mlp_worker`` (the Tessera environment)
over a pipe. Layers execute layer-major (every window through layer i before
layer i+1), so the worker holds one layer's weights at a time. The split
decoder loop is checked against the stock model: the ``bf16`` arm's logits
must equal ``model(input_ids)`` bitwise for every window.

Acceptance (fixed before measurement): with ``delta(arm) = ppl(arm) -
ppl(bf16)``, the gate passes only when ``delta(shipped) > 0`` and
``delta(native) <= THRESHOLD_RATIO * delta(shipped)``.
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

SCHEMA = "tessera.quality.qwen3_native_mlp_perplexity.v1"
THRESHOLD_RATIO = 1.5
ARMS = ("bf16", "shipped", "native")


def verdict(ppl: dict[str, float], ratio: float = THRESHOLD_RATIO) -> dict:
    """Apply the pre-registered acceptance rule to per-arm perplexities."""
    if not all(math.isfinite(ppl[arm]) and ppl[arm] > 0 for arm in ARMS):
        raise ValueError("perplexity must be finite and positive for every arm")
    shipped, native = ppl["shipped"] - ppl["bf16"], ppl["native"] - ppl["bf16"]
    if shipped <= 0:
        return {"delta_shipped": shipped, "delta_native": native, "ratio": None, "threshold_ratio": ratio,
                "passed": False, "reason": "shipped NVFP4 does not raise perplexity; the ratio is undefined"}
    return {"delta_shipped": shipped, "delta_native": native, "ratio": native / shipped,
            "threshold_ratio": ratio, "passed": native <= ratio * shipped,
            "reason": "delta(native) <= threshold_ratio * delta(shipped)"}


class Worker:
    def __init__(self, python: str, repo: Path, cache_root: Path, certify: str) -> None:
        self.proc = subprocess.Popen(
            [python, "-m", "benchmarks.rocm.qwen3_native_mlp_worker", "--cache-root", str(cache_root),
             "--certify", certify], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

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
           sequences: int, length: int, threads: int, certify: str) -> dict:
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
    pool = Worker(worker_python, repo, cache_root, certify)
    identity, _ = pool.call({"op": "hello"})

    with torch.inference_mode():
        ids = [torch.from_numpy(row) for row in token_ids]
        stock = [hashlib.sha256(model(row[None, :], use_cache=False).logits.view(torch.int16).numpy().tobytes())
                 .hexdigest() for row in ids]
        hidden = {arm: [body.embed_tokens(row[None, :]) for row in ids] for arm in ARMS}
        position = torch.arange(length)[None, :]
        cache_position = torch.arange(length)
        rotary = body.rotary_emb(hidden["bf16"][0], position)
        mask = create_causal_mask(config=body.config, input_embeds=hidden["bf16"][0], attention_mask=None,
                                  cache_position=cache_position, past_key_values=None, position_ids=position)
        drift: list[dict] = []
        sources = []
        for index, layer in enumerate(layers):
            reply, _ = pool.call({"op": "layer", "layer": index})
            sources.append(reply["source"])
            mlp_inputs = {}
            for arm in ARMS:
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
                for arm in ("shipped", "native")}})
            print(json.dumps({"layer": index, "elapsed_s": round(time.perf_counter() - started, 1),
                              **{k: round(v, 5) for k, v in drift[-1].items() if k != "layer"}}), flush=True)
        nll, split = {}, []
        for arm in ARMS:
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

    per_window = {arm: [float(np.exp(v.mean())) for v in nll[arm]] for arm in ARMS}
    ppl = {arm: float(np.exp(np.concatenate(nll[arm]).mean())) for arm in ARMS}
    diff = {arm: np.concatenate(nll[arm]) - np.concatenate(nll["bf16"]) for arm in ("shipped", "native")}
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "token_nll.npz", **{arm: np.stack(nll[arm]) for arm in ARMS}, token_ids=token_ids)
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
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "numpy": np.__version__},
        "split_loop_matches_stock_logits": True, "stock_logits_sha256": stock,
        "perplexity": ppl, "per_window_perplexity": per_window,
        "mean_token_nll_delta_vs_bf16": {arm: float(v.mean()) for arm, v in diff.items()},
        "verdict": verdict(ppl), "mlp_input_drift": drift, "worker": worker_summary,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "scope": "MLP substitution only; attention, norms and lm_head stay BF16 Torch. The shipped arm is "
                 "weight-only NVFP4, not the checkpoint's W4A4 deployment recipe.",
    }
    result["model_quality_acceptance"] = result["verdict"]["passed"] and certify == "every-layer"
    (output / "perplexity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"perplexity": ppl, "verdict": result["verdict"]}), flush=True)
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
    args = parser.parse_args()
    record(args.cache_root, args.corpus, args.corpus_revision, args.output, args.worker_python,
           args.sequences, args.length, args.threads, args.certify)
