"""Capture every Qwen3-8B decoder layer's MLP input on a pinned text corpus.

Reference/diagnostic only (Decision #23): Torch runs the upstream BF16 model on
CPU in a separate environment and never executes a Tessera package. The
output is the teacher-forced BF16 input of each layer's MLP, which the native
recorder (``record_qwen3_mlp_layer_quality``) consumes. This is stage 1 of the
model-quality gate: per-layer error in isolation, not end-to-end accuracy.

Sequences are evenly spaced, non-overlapping windows of the tokenized
wikitext-2-raw-v1 test split, so they cover the whole split rather than its
opening article. The model runs twice and every layer's capture must be
bitwise identical across the two runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen3_local_checkpoint as local  # noqa: E402

SCHEMA = "tessera.checkpoint.qwen3_all_layer_mlp_inputs.v1"
CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def corpus_windows(tokenizer, corpus_path: Path, sequences: int, length: int) -> tuple[np.ndarray, dict]:
    import pyarrow.parquet as pq

    raw = corpus_path.read_bytes()
    text = "".join(pq.read_table(corpus_path).column("text").to_pylist())
    ids = np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, np.int64)
    if ids.size < sequences * length:
        raise ValueError(f"corpus has {ids.size} tokens, fewer than {sequences} x {length}")
    stride = ids.size // sequences
    starts = [i * stride for i in range(sequences)]
    windows = np.stack([ids[s:s + length] for s in starts])
    return windows, {"file": CORPUS_FILE, "parquet_sha256": digest(raw), "text_sha256": digest(text.encode()),
                     "total_tokens": int(ids.size), "window_starts": starts, "window_length": length}


def capture(cache_root: Path, corpus_path: Path, corpus_revision: str, output: Path,
            sequences: int, length: int, threads: int) -> dict:
    if sequences <= 0 or not 0 < length <= 4096 or threads <= 0:
        raise ValueError("capture requires positive sequences/threads and length 1..4096")
    if not corpus_revision or not all(c in "0123456789abcdef" for c in corpus_revision):
        raise ValueError("corpus revision must be the pinned commit hash")
    import tokenizers
    import torch
    import transformers

    torch.set_num_threads(threads)
    root = local.snapshot_dir(cache_root, local.BF16_REPO, local.BF16_REVISION)
    snapshot = local.LocalSnapshot(local.BF16_REPO, local.BF16_REVISION, root)
    shards = sorted(set(snapshot.weight_map.values()))
    shard_sha = {shard: snapshot.verify_shard(shard) for shard in shards}
    tokenizer_raw = (root / "tokenizer.json").read_bytes()
    config_raw = (root / "config.json").read_bytes()
    tokenizer = tokenizers.Tokenizer.from_str(tokenizer_raw.decode())
    token_ids, corpus = corpus_windows(tokenizer, corpus_path, sequences, length)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        root, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    model.eval().requires_grad_(False)
    layers = model.model.layers
    captured: list[list[np.ndarray]] = [[] for _ in layers]

    def hook_for(index):
        def hook(module, args):
            value = args[0].detach()
            if value.dtype != torch.bfloat16:
                raise RuntimeError(f"layer {index} MLP input is {value.dtype}, not BF16")
            captured[index].append(value.contiguous().view(torch.int16).numpy().view(np.uint16).copy())
        return hook

    handles = [layer.mlp.register_forward_pre_hook(hook_for(i)) for i, layer in enumerate(layers)]
    runs: list[list[str]] = []
    try:
        with torch.inference_mode():
            for _ in range(2):
                for item in captured:
                    item.clear()
                for row in token_ids:
                    model(torch.from_numpy(row[None, :]), use_cache=False)
                runs.append([digest(np.concatenate(item).tobytes()) for item in captured])
    finally:
        for handle in handles:
            handle.remove()
    if runs[0] != runs[1]:
        raise RuntimeError("the two reference runs captured different MLP inputs")

    output.mkdir(parents=True, exist_ok=True)
    hidden = model.config.hidden_size
    records = []
    for index, item in enumerate(captured):
        bits = np.ascontiguousarray(np.concatenate(item).reshape(-1, hidden))
        np.save(output / f"layer_{index:02d}.npy", bits)
        as_f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(as_f32).all():
            raise RuntimeError(f"layer {index} captured non-finite MLP inputs")
        records.append({"layer": index, "file": f"layer_{index:02d}.npy", "shape": list(bits.shape),
                        "bf16_bits_sha256": digest(bits.tobytes()),
                        "rms": float(np.sqrt(np.mean(as_f32.astype(np.float64) ** 2))),
                        "max_abs": float(np.abs(as_f32).max())})
    np.save(output / "token_ids.npy", token_ids)
    metadata = {
        "schema": SCHEMA,
        "capture_kind": "pinned_bf16_model_all_layer_mlp_inputs",
        "capture_boundary": "upstream_decoder_mlp_forward_pre_hook",
        "reference_execution": "torch_cpu_bfloat16_eager_teacher_forced",
        "storage": "bf16 bit patterns as little-endian uint16 .npy, rows = sequence-major tokens",
        "repo": local.BF16_REPO, "revision": local.BF16_REVISION,
        "bf16_index_sha256": snapshot.index_sha256, "bf16_shard_sha256": shard_sha,
        "config_sha256": digest(config_raw), "tokenizer_sha256": digest(tokenizer_raw),
        "corpus": {"repo": CORPUS_REPO, "revision": corpus_revision, **corpus},
        "token_ids_sha256": digest(token_ids.tobytes()), "batch_sequence_shape": list(token_ids.shape),
        "layers": records, "repeat_bitwise_equal": True, "threads": threads, "host": socket.gethostname(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "numpy": np.__version__},
        "capture_source_sha256": digest(Path(__file__).read_bytes()),
        "reader_source_sha256": digest(Path(local.__file__).read_bytes()),
        "model_quality_acceptance": False,
        "scope": "Teacher-forced per-layer MLP inputs; stage 1 input only, not end-to-end model accuracy",
    }
    (output / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"layers": len(records), "rows": records[0]["shape"][0]}), flush=True)
    return metadata


def load_layer(capture_dir: Path, layer: int, metadata: dict | None = None) -> np.ndarray:
    """Return one layer's verified MLP input as BF16 values."""
    import ml_dtypes

    metadata = metadata or json.loads((Path(capture_dir) / "capture.json").read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("repeat_bitwise_equal") is not True \
            or metadata.get("revision") != local.BF16_REVISION or metadata.get("model_quality_acceptance") is not False:
        raise ValueError("all-layer capture contract differs")
    records = {item["layer"]: item for item in metadata["layers"]}
    if layer not in records:
        raise ValueError(f"capture has no layer {layer}")
    record = records[layer]
    bits = np.load(Path(capture_dir) / record["file"], allow_pickle=False)
    if bits.dtype != np.uint16 or list(bits.shape) != record["shape"] \
            or digest(np.ascontiguousarray(bits).tobytes()) != record["bf16_bits_sha256"]:
        raise ValueError(f"layer {layer} capture identity differs")
    return np.ascontiguousarray(bits).view(ml_dtypes.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    capture(args.cache_root, args.corpus, args.corpus_revision, args.output, args.sequences, args.length, args.threads)
