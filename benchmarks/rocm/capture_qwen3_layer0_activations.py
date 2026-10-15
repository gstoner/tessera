"""Capture pinned BF16 Qwen3 layer-0 MLP inputs with the upstream CPU model.

Reference/diagnostic only: Torch never executes a Tessera production package.
Only executed parameter ranges and selected embedding windows are fetched.
"""

from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import socket
from typing import cast
import numpy as np
import ml_dtypes
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint

PROMPTS = (
    "A compiler translates a typed mathematical program into operations that a processor can execute. "
    "The graph describes values and effects. A schedule chooses parallel work and storage. "
    "Verification checks shapes, numerical policies and buffer lifetimes before execution. "
    "Measurements distinguish compilation, transfers, kernel time and completed application calls. "
    "Explain the tradeoffs using a matrix product and a reduction. ",
    "Consider a numerical simulation of heat flowing through a metal plate. "
    "At each time step, neighboring cells exchange energy while the boundary conditions remain fixed. "
    "Describe how conservation, stability and rounding error affect the result. "
    "Write a short Python function for a finite difference update, then explain how to test it. ",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class PinnedBF16Reader:
    def __init__(self):
        self.repo = checkpoint.BF16_REPO
        self.revision = checkpoint.BF16_REVISION
        index, raw = checkpoint._get_json(checkpoint._url(self.repo, self.revision, "model.safetensors.index.json"))
        self.weight_map = cast(dict[str, str], index["weight_map"])
        self.headers = {}
        self.provenance = {"repo": self.repo, "revision": self.revision, "index_sha256": digest(raw), "tensors": {}}

    def header(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            self.headers[shard] = checkpoint._safetensors_header(self.repo, self.revision, shard)
        return shard, self.headers[shard]

    def tensor(self, name):
        self.header(name)
        values, raw, spec, shard = checkpoint._read_tensor(
            self.repo, self.revision, name, self.weight_map, self.headers
        )
        if spec["dtype"] != "BF16":
            raise ValueError("reference requires original BF16 parameters")
        self.provenance["tensors"][name] = {"shard": shard, "storage": spec, "sha256": digest(raw), "bytes": len(raw)}
        return values

    def embedding_rows(self, ids, *, block_rows=128):
        name = "model.embed_tokens.weight"
        shard, (url, header, data_start) = self.header(name)
        spec = header[name]
        if spec["dtype"] != "BF16" or len(spec["shape"]) != 2 or block_rows <= 0:
            raise ValueError("embedding requires rank-two BF16 storage and positive windows")
        rows, columns = spec["shape"]
        flat = np.asarray(ids).reshape(-1)
        if not np.issubdtype(flat.dtype, np.integer) or not flat.size or np.any(flat < 0) or np.any(flat >= rows):
            raise ValueError("embedding token IDs lie outside the pinned vocabulary")
        start, stop = map(int, spec["data_offsets"])
        if stop - start != rows * columns * 2:
            raise ValueError("embedding byte capacity differs from shape")
        values = np.empty((flat.size, columns), dtype=ml_dtypes.bfloat16)
        windows = []
        for block in np.unique(flat // block_rows):
            lo = int(block) * block_rows
            hi = min(lo + block_rows, rows)
            first = data_start + start + lo * columns * 2
            last = data_start + start + hi * columns * 2 - 1
            raw = checkpoint._fetch_range(url, first, last)
            decoded = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(hi - lo, columns)
            selected = (flat >= lo) & (flat < hi)
            values[selected] = decoded[flat[selected] - lo]
            windows.append(
                {"row_start": lo, "row_stop": hi, "byte_start": first, "byte_end": last, "sha256": digest(raw)}
            )
        self.provenance["tensors"][name] = {
            "shard": shard,
            "storage": spec,
            "selected_row_sha256": digest(values.tobytes()),
            "windows": windows,
            "note": "Only selected row windows were fetched; no full embedding-tensor checksum is claimed.",
        }
        return values.reshape(*np.asarray(ids).shape, columns)


def capture(output: Path, sequence_length=128, threads=4):
    if output.suffix != ".npz" or not 0 < sequence_length <= 4096 or threads <= 0:
        raise ValueError("capture requires .npz output, sequence length 1..4096 and positive threads")
    import torch
    import transformers
    import tokenizers
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as model
    from transformers.masking_utils import create_causal_mask

    torch.set_num_threads(threads)
    config_dict, config_raw = checkpoint._get_json(
        checkpoint._url(checkpoint.BF16_REPO, checkpoint.BF16_REVISION, "config.json")
    )
    _, tokenizer_raw = checkpoint._get_json(
        checkpoint._url(checkpoint.BF16_REPO, checkpoint.BF16_REVISION, "tokenizer.json")
    )
    tokenizer = tokenizers.Tokenizer.from_str(tokenizer_raw.decode())
    token_ids = []
    for prompt in PROMPTS:
        text = prompt
        while len(encoded := tokenizer.encode(text, add_special_tokens=False).ids) < sequence_length:
            text += prompt
        token_ids.append(encoded[:sequence_length])
    token_ids = np.asarray(token_ids, np.int64)
    reader = PinnedBF16Reader()
    embedded = reader.embedding_rows(token_ids)

    def tensor(values):
        # Bit-preserving bridge: Torch's NumPy bridge does not recognize ml_dtypes.
        return torch.from_numpy(values.view(np.uint16).copy()).view(torch.bfloat16)

    config = Qwen3Config(**config_dict)
    config._attn_implementation = "eager"
    names = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    with torch.device("meta"):
        layer = model.Qwen3DecoderLayer(config, layer_idx=0)
    state = {}
    for name in names:
        print("loading pinned BF16", name, flush=True)
        state[name] = tensor(reader.tensor("model.layers.0." + name))
    status = layer.load_state_dict(state, strict=False, assign=True)
    missing = {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
    if set(status.missing_keys) != missing or status.unexpected_keys:
        raise RuntimeError("executed decoder parameters differ from the pinned reference")
    if {name for name, param in layer.named_parameters() if param.is_meta} != missing:
        raise RuntimeError("reference has an uninitialized executed parameter")
    layer.eval().requires_grad_(False)
    hidden = tensor(embedded)
    positions = torch.arange(sequence_length).unsqueeze(0)
    rope = model.Qwen3RotaryEmbedding(config, device=torch.device("cpu"))
    cos_sin = rope(hidden, positions)
    mask = create_causal_mask(
        config=config,
        input_embeds=hidden,
        attention_mask=None,
        cache_position=torch.arange(sequence_length),
        past_key_values=None,
        position_ids=positions,
    )
    captured = []

    class CaptureComplete(Exception):
        pass

    def hook(module, args):
        captured.append(args[0].detach().float().numpy().copy())
        raise CaptureComplete

    handle = layer.mlp.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for _ in range(2):
                try:
                    layer(hidden, attention_mask=mask, position_ids=positions, position_embeddings=cos_sin)
                except CaptureComplete:
                    pass
                else:
                    raise RuntimeError("upstream decoder did not reach the declared capture boundary")
    finally:
        handle.remove()
    if len(captured) != 2:
        raise RuntimeError("upstream capture count differs")
    np.testing.assert_array_equal(*captured)
    activations = np.ascontiguousarray(captured[0].reshape(-1, config.hidden_size))
    if not np.isfinite(activations).all():
        raise RuntimeError("reference activations are non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, activations=activations, token_ids=token_ids)
    sources = {
        "capture": Path(__file__),
        "checkpoint_loader": Path(checkpoint.__file__),
        "upstream_qwen3": Path(model.__file__),
        "upstream_masking": Path(inspect.getfile(create_causal_mask)),
    }
    metadata = {
        "schema": "tessera.checkpoint.model_activations.v1",
        "capture_kind": "pinned_bf16_model_layer_input",
        "repo": reader.repo,
        "revision": reader.revision,
        "tensor": "model.layers.0.mlp.input",
        "capture_boundary": "upstream_decoder_mlp_forward_pre_hook",
        "reference_execution": "torch_cpu_bfloat16_eager",
        "host": socket.gethostname(),
        "shape": list(activations.shape),
        "batch_sequence_shape": list(token_ids.shape),
        "config_sha256": digest(config_raw),
        "tokenizer_sha256": digest(tokenizer_raw),
        "token_ids_sha256": digest(token_ids.tobytes()),
        "activation_sha256": digest(activations.tobytes()),
        "npz_sha256": digest(output.read_bytes()),
        "prompt_texts": list(PROMPTS),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
        },
        "sources_sha256": {key: digest(path.read_bytes()) for key, path in sources.items()},
        "weight_source": reader.provenance,
        "unmaterialized_unexecuted_parameters": sorted(missing),
        "repeat_bitwise_equal": True,
        "threads": threads,
        "model_quality_acceptance": False,
        "scope": "BF16 original layer-0 inputs on two authored text sequences; no full-model PPL or Tessera execution proof",
        "activation_statistics": {
            "rms": float(np.sqrt(np.mean(activations.astype(np.float64) ** 2))),
            "max_abs": float(np.abs(activations).max()),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shape": metadata["shape"],
                "activation_sha256": metadata["activation_sha256"],
                "reference": metadata["reference_execution"],
            }
        ),
        flush=True,
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    capture(args.output, args.sequence_length, args.threads)
