"""MiniMax-M3 importer helpers: tokenizer, safetensors, multimodal gate.

This module is the production-shaped importer boundary for MiniMax-M3. It reads
HF-style metadata from local files, validates it against Tessera's MiniMax-M3
text-tower contract, and prepares multimodal prompt spans without claiming a
vision/video tower implementation. Text-only prompts can execute through the
reference MoE-transformer runtime; image/video segments fail loudly until a
real vision encoder/projector path exists.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..data import ByteTokenizer, Tokenizer, VocabTokenizer
from . import vision_transformer as vt
from .multimodal import MediaProcessorConfig, MediaSpan, ProjectedMediaEmbeddings, processor_config_from_hf
from . import moe_transformer_runtime as rt
from .moe_transformer import MoETransformerConfig, verify_config
from .minimax_m3 import MiniMaxM3VisionMetadata, VISION_METADATA, config as minimax_m3_config
from .minimax_m3 import processor_config as minimax_processor_config
from .minimax_m3 import vision_transformer_config as minimax_vision_transformer_config


class MiniMaxM3ImportError(ValueError):
    """Raised when MiniMax-M3 importer metadata is missing or inconsistent."""


class MiniMaxM3VisionExecutionError(RuntimeError):
    """Raised when a prompt asks for vision/video execution before it exists."""


@dataclass(frozen=True)
class MiniMaxM3TokenizerSpec:
    kind: str
    vocab_size: int
    special_tokens: dict[str, int]
    tokenizer_file: str | None = None
    tokenizer_config_file: str | None = None
    chat_template_present: bool = False


@dataclass(frozen=True)
class MiniMaxM3TokenizerImport:
    tokenizer: Tokenizer
    spec: MiniMaxM3TokenizerSpec


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    filename: str

    @property
    def nbytes(self) -> int:
        return int(self.data_offsets[1] - self.data_offsets[0])


@dataclass(frozen=True)
class SafetensorsManifest:
    tensors: dict[str, TensorSpec]
    shards: tuple[str, ...]
    total_bytes: int


@dataclass(frozen=True)
class MiniMaxM3ImportManifest:
    root: str
    config: MoETransformerConfig
    hf_config: dict[str, Any]
    tokenizer: MiniMaxM3TokenizerImport | None
    safetensors: SafetensorsManifest | None
    vision: MiniMaxM3VisionMetadata
    processor: MediaProcessorConfig | None = None
    text_execution_supported: bool = True
    vision_execution_supported: bool = False


@dataclass(frozen=True)
class PromptSegment:
    kind: str
    text: str = ""
    frames: int = 1


@dataclass(frozen=True)
class PromptSpan:
    kind: str
    start: int
    end: int
    token_index: int
    frames: int = 1


@dataclass(frozen=True)
class PreparedMultimodalPrompt:
    token_ids: tuple[int, ...]
    spans: tuple[PromptSpan, ...]
    vision_execution_supported: bool

    @property
    def has_vision(self) -> bool:
        return bool(self.spans)


MediaEmbeddingInput = Sequence[np.ndarray] | Mapping[int | str, np.ndarray]
MediaInput = Sequence[Any] | Mapping[int | str, Any]


@dataclass(frozen=True)
class MiniMaxM3VisionWeights:
    tower: vt.VisionTowerWeights


@dataclass(frozen=True)
class MiniMaxM3ProjectorWeights:
    projector: vt.VisionProjectorWeights


@dataclass(frozen=True)
class MiniMaxM3VisionRuntime:
    config: vt.VisionTransformerConfig
    processor: MediaProcessorConfig
    weights: vt.VisionRuntimeWeights


_ST_DTYPE_NBYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
_ST_DTYPE = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
try:  # bf16 is optional in numpy environments.
    import ml_dtypes

    _ST_DTYPE["BF16"] = ml_dtypes.bfloat16
except Exception:  # pragma: no cover
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_dir() else p.parent


def _field(hf: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in hf:
            return hf[name]
    return default


def validate_hf_config(
    hf_config: Mapping[str, Any],
    *,
    expected: MoETransformerConfig | None = None,
) -> MoETransformerConfig:
    """Validate an HF-style MiniMax-M3 config against the Tessera contract.

    The importer accepts common HF aliases but only checks fields present in the
    file. This lets small local fixtures validate the same surface without
    needing a 427B checkpoint.
    """
    cfg = expected if expected is not None else minimax_m3_config()
    verify_config(cfg)
    checks: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
        ("hidden_size", cfg.hidden_size, ("hidden_size",)),
        ("num_layers", cfg.num_layers, ("num_hidden_layers", "num_layers", "n_layers")),
        ("vocab_size", cfg.vocab_size, ("vocab_size",)),
        ("context_length", cfg.context_length, ("max_position_embeddings", "context_length", "max_seq_len")),
        ("num_attention_heads", cfg.num_attention_heads, ("num_attention_heads",)),
        ("num_kv_heads", cfg.num_kv_heads, ("num_key_value_heads", "num_kv_heads")),
        ("head_dim", cfg.head_dim, ("head_dim",)),
        ("rope_head_dim", cfg.rope_head_dim, ("rope_head_dim", "qk_rope_head_dim")),
        ("rope_theta", cfg.rope_theta, ("rope_theta",)),
        ("num_experts", cfg.num_experts, ("num_experts", "n_routed_experts", "num_routed_experts")),
        ("num_experts_per_tok", cfg.num_experts_per_tok, ("num_experts_per_tok", "num_experts_per_token", "top_k")),
        ("num_shared_experts", cfg.num_shared_experts, ("num_shared_experts",)),
        ("first_k_dense", cfg.first_k_dense, ("first_k_dense", "num_dense_layers", "first_k_dense_layers")),
        ("msa_top_k_blocks", cfg.msa_top_k_blocks, ("msa_top_k_blocks", "sparse_topk", "sparse_top_k")),
        ("msa_block_size", cfg.msa_block_size, ("msa_block_size", "sparse_block_size")),
        ("msa_index_dim", cfg.msa_index_dim, ("msa_index_dim", "index_dim")),
        ("msa_num_index_heads", cfg.msa_num_index_heads, ("msa_num_index_heads", "num_index_heads")),
        ("msa_score_type", cfg.msa_score_type, ("msa_score_type", "sparse_score_type")),
    )
    mismatches: list[str] = []
    for label, expected_value, aliases in checks:
        actual = _field(hf_config, *aliases)
        if actual is None:
            continue
        if isinstance(expected_value, float):
            ok = abs(float(actual) - expected_value) <= 1e-9
        else:
            ok = actual == expected_value
        if not ok:
            mismatches.append(f"{label}: HF {actual!r} != Tessera {expected_value!r}")
    freq = _field(hf_config, "msa_sparse_layer_freq", "sparse_attention_freq")
    if freq is not None and tuple(int(v) for v in freq) != tuple(cfg.msa_sparse_layer_freq):
        mismatches.append("msa_sparse_layer_freq does not match Tessera dense/MSA layer contract")
    if mismatches:
        raise MiniMaxM3ImportError("; ".join(mismatches))
    return cfg


def import_tokenizer(root: str | Path) -> MiniMaxM3TokenizerImport:
    """Import a local HF tokenizer manifest as a Tessera tokenizer contract.

    If ``tokenizer.json`` has a vocab map, Tessera exposes a deterministic
    whitespace-compatible ``VocabTokenizer`` for tests and text-only smoke
    execution. Full BPE/processor parity remains an HF frontend concern.
    """
    root_path = _root(root)
    tok_path = root_path / "tokenizer.json"
    cfg_path = root_path / "tokenizer_config.json"
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    specials = _special_tokens_from_config(cfg)
    chat_template_present = bool(cfg.get("chat_template"))
    if tok_path.exists():
        tok = _read_json(tok_path)
        model = tok.get("model", {})
        vocab = model.get("vocab")
        if not isinstance(vocab, Mapping):
            raise MiniMaxM3ImportError("tokenizer.json exists but model.vocab is missing")
        str_vocab = {str(k): int(v) for k, v in vocab.items()}
        for token, idx in specials.items():
            str_vocab.setdefault(token, idx)
        tokenizer = VocabTokenizer(str_vocab, unk_token=str(cfg.get("unk_token", "<unk>")), specials=specials)
        spec = MiniMaxM3TokenizerSpec(
            kind=str(model.get("type", "vocab")).lower(),
            vocab_size=tokenizer.vocab_size,
            special_tokens=specials,
            tokenizer_file=str(tok_path),
            tokenizer_config_file=str(cfg_path) if cfg_path.exists() else None,
            chat_template_present=chat_template_present,
        )
        return MiniMaxM3TokenizerImport(tokenizer=tokenizer, spec=spec)
    byte_tokenizer = ByteTokenizer(specials=specials)
    return MiniMaxM3TokenizerImport(
        tokenizer=byte_tokenizer,
        spec=MiniMaxM3TokenizerSpec(
            kind="byte_fallback",
            vocab_size=byte_tokenizer.vocab_size,
            special_tokens=specials,
            tokenizer_config_file=str(cfg_path) if cfg_path.exists() else None,
            chat_template_present=chat_template_present,
        ),
    )


def _special_tokens_from_config(cfg: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in cfg.items():
        if not key.endswith("_token"):
            continue
        token = value.get("content") if isinstance(value, Mapping) else value
        token_id = cfg.get(key + "_id")
        if isinstance(token, str) and isinstance(token_id, int):
            out[token] = int(token_id)
    added = cfg.get("added_tokens_decoder", {})
    if isinstance(added, Mapping):
        for raw_id, entry in added.items():
            if isinstance(entry, Mapping) and isinstance(entry.get("content"), str):
                out[entry["content"]] = int(raw_id)
    return out


def read_safetensors_manifest(path: str | Path) -> SafetensorsManifest:
    """Read safetensors headers from a file, shard index, or checkpoint dir."""
    p = Path(path)
    if p.is_dir():
        indexes = sorted(p.glob("*.safetensors.index.json"))
        if indexes:
            return _manifest_from_index(indexes[0])
        files = sorted(p.glob("*.safetensors"))
        if not files:
            raise MiniMaxM3ImportError(f"no safetensors files under {p}")
    elif p.name.endswith(".index.json"):
        return _manifest_from_index(p)
    else:
        files = [p]
    tensors: dict[str, TensorSpec] = {}
    for file in files:
        tensors.update(_read_safetensors_header(file))
    return SafetensorsManifest(
        tensors=dict(sorted(tensors.items())),
        shards=tuple(str(f) for f in files),
        total_bytes=sum(spec.nbytes for spec in tensors.values()),
    )


def load_safetensors_tensors(
    path: str | Path,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load selected tensors from a safetensors file/index/directory."""
    manifest = read_safetensors_manifest(path)
    wanted = set(names) if names is not None else set(manifest.tensors)
    missing = sorted(wanted - set(manifest.tensors))
    if missing:
        raise MiniMaxM3ImportError(f"requested tensors not found: {missing[:5]}")
    by_file: dict[str, list[str]] = {}
    for name in wanted:
        by_file.setdefault(manifest.tensors[name].filename, []).append(name)
    out: dict[str, np.ndarray] = {}
    for filename, file_names in by_file.items():
        out.update(_load_safetensors_from_file(Path(filename), file_names))
    return out


def _manifest_from_index(index_path: Path) -> SafetensorsManifest:
    index = _read_json(index_path)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, Mapping):
        raise MiniMaxM3ImportError("safetensors index missing weight_map")
    tensors: dict[str, TensorSpec] = {}
    for filename in sorted(set(str(v) for v in weight_map.values())):
        tensors.update(_read_safetensors_header(index_path.parent / filename))
    missing = [name for name in weight_map if name not in tensors]
    if missing:
        raise MiniMaxM3ImportError(f"safetensors index references missing tensors: {missing[:3]}")
    return SafetensorsManifest(
        tensors=dict(sorted(tensors.items())),
        shards=tuple(str(index_path.parent / f) for f in sorted(set(str(v) for v in weight_map.values()))),
        total_bytes=sum(spec.nbytes for spec in tensors.values()),
    )


def _read_safetensors_header(path: Path) -> dict[str, TensorSpec]:
    data = path.read_bytes()
    if len(data) < 8:
        raise MiniMaxM3ImportError(f"{path} is too small to be safetensors")
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_len])
    out: dict[str, TensorSpec] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = str(meta.get("dtype", ""))
        shape = tuple(int(v) for v in meta.get("shape", ()))
        offsets = tuple(int(v) for v in meta.get("data_offsets", ()))
        if dtype not in _ST_DTYPE_NBYTES:
            raise MiniMaxM3ImportError(f"unsupported safetensors dtype {dtype!r} for {name!r}")
        if len(offsets) != 2 or offsets[1] < offsets[0]:
            raise MiniMaxM3ImportError(f"bad data_offsets for tensor {name!r}")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * _ST_DTYPE_NBYTES[dtype]
        if expected_bytes != offsets[1] - offsets[0]:
            raise MiniMaxM3ImportError(
                f"tensor {name!r} byte count mismatch: header={offsets[1] - offsets[0]} expected={expected_bytes}")
        out[name] = TensorSpec(name=name, dtype=dtype, shape=shape, data_offsets=(offsets[0], offsets[1]),
                               filename=str(path))
    return out


def _load_safetensors_from_file(path: Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_len])
    base = 8 + header_len
    out: dict[str, np.ndarray] = {}
    for name in names:
        meta = header.get(name)
        if meta is None:
            raise MiniMaxM3ImportError(f"{name!r} not found in {path}")
        dtype_name = str(meta["dtype"])
        dtype = _ST_DTYPE.get(dtype_name)
        if dtype is None:
            raise MiniMaxM3ImportError(f"unsupported safetensors dtype {dtype_name!r} for {name!r}")
        start, end = (int(v) for v in meta["data_offsets"])
        shape = tuple(int(v) for v in meta["shape"])
        out[name] = np.frombuffer(data[base + start:base + end], dtype=dtype).reshape(shape).copy()
    return out


def expected_hf_text_tensor_shapes(
    cfg: MoETransformerConfig,
    *,
    layers: Sequence[int] = (0,),
) -> dict[str, tuple[int, ...]]:
    """Expected HF-layout tensor shapes for core MiniMax-M3 text weights."""
    H, Hq, Hkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads
    D, Dv = cfg.qk_per_head_dim, cfg.value_per_head_dim
    out = {
        "model.embed_tokens.weight": (cfg.vocab_size, H),
        "model.norm.weight": (H,),
        "lm_head.weight": (cfg.vocab_size, H),
    }
    for li in layers:
        prefix = f"model.layers.{li}."
        out[prefix + "input_layernorm.weight"] = (H,)
        out[prefix + "post_attention_layernorm.weight"] = (H,)
        out[prefix + "self_attn.q_proj.weight"] = (Hq * D, H)
        out[prefix + "self_attn.k_proj.weight"] = (Hkv * D, H)
        out[prefix + "self_attn.v_proj.weight"] = (Hkv * Dv, H)
        out[prefix + "self_attn.o_proj.weight"] = (H, Hq * Dv)
        if cfg.is_moe_layer(li):
            out[prefix + "mlp.router.weight"] = (cfg.num_experts, H)
        else:
            F = cfg.dense_intermediate_size or cfg.shared_expert_intermediate_size
            out[prefix + "mlp.gate_proj.weight"] = (F, H)
            out[prefix + "mlp.up_proj.weight"] = (F, H)
            out[prefix + "mlp.down_proj.weight"] = (H, F)
    return out


def expected_hf_vision_tensor_shapes(
    text_cfg: MoETransformerConfig,
    vision_cfg: vt.VisionTransformerConfig,
    *,
    layers: Sequence[int] = (0,),
) -> dict[str, tuple[int, ...]]:
    """Expected HF-style vision/projector tensor shapes for MiniMax-M3.

    Linear weights use the PyTorch convention ``(out_features, in_features)``;
    the loader transposes them into the reference runtime's ``x @ W`` layout.
    """
    H = vision_cfg.hidden_size
    F = vision_cfg.mlp_hidden_size or 4 * H
    out = {
        "vision_tower.patch_embed.weight": (H, vision_cfg.patch_dim),
        "vision_tower.patch_embed.bias": (H,),
        "vision_tower.pos_embed": (max(vision_cfg.image_seq_length, vision_cfg.merged_tokens), H),
        "vision_tower.norm.weight": (H,),
        "multi_modal_projector.weight": (text_cfg.hidden_size, H),
        "multi_modal_projector.bias": (text_cfg.hidden_size,),
    }
    for li in layers:
        prefix = f"vision_tower.layers.{li}."
        out[prefix + "norm1.weight"] = (H,)
        out[prefix + "qkv.weight"] = (3 * H, H)
        out[prefix + "o_proj.weight"] = (H, H)
        out[prefix + "norm2.weight"] = (H,)
        out[prefix + "mlp.gate_proj.weight"] = (F, H)
        out[prefix + "mlp.up_proj.weight"] = (F, H)
        out[prefix + "mlp.down_proj.weight"] = (H, F)
    return out


def validate_safetensors_shapes(
    manifest: SafetensorsManifest,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    require_all: bool = True,
) -> None:
    missing: list[str] = []
    mismatches: list[str] = []
    for name, shape in expected_shapes.items():
        spec = manifest.tensors.get(name)
        if spec is None:
            if require_all:
                missing.append(name)
            continue
        if tuple(spec.shape) != tuple(shape):
            mismatches.append(f"{name}: {spec.shape} != {tuple(shape)}")
    if missing or mismatches:
        parts = []
        if missing:
            parts.append(f"missing tensors: {missing[:5]}")
        if mismatches:
            parts.append(f"shape mismatches: {mismatches[:5]}")
        raise MiniMaxM3ImportError("; ".join(parts))


def load_vision_runtime_weights(
    tensors: Mapping[str, np.ndarray],
    vision_cfg: vt.VisionTransformerConfig,
) -> vt.VisionRuntimeWeights:
    """Map HF-style MiniMax vision/projector tensors into runtime weights."""
    H = vision_cfg.hidden_size
    F = vision_cfg.mlp_hidden_size or 4 * H

    def need(name: str, shape: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(tensors.get(name))
        if arr.shape != shape:
            raise MiniMaxM3ImportError(f"{name} shape {arr.shape} != expected {shape}")
        return arr.astype(np.float64)

    layers: list[vt.VisionLayerWeights] = []
    for li in range(vision_cfg.num_layers):
        prefix = f"vision_tower.layers.{li}."
        layers.append(vt.VisionLayerWeights(
            norm1=need(prefix + "norm1.weight", (H,)),
            qkv=need(prefix + "qkv.weight", (3 * H, H)).T,
            o=need(prefix + "o_proj.weight", (H, H)).T,
            norm2=need(prefix + "norm2.weight", (H,)),
            gate=need(prefix + "mlp.gate_proj.weight", (F, H)).T,
            up=need(prefix + "mlp.up_proj.weight", (F, H)).T,
            down=need(prefix + "mlp.down_proj.weight", (H, F)).T,
        ))
    tower = vt.VisionTowerWeights(
        patch_weight=need("vision_tower.patch_embed.weight", (H, vision_cfg.patch_dim)).T,
        patch_bias=need("vision_tower.patch_embed.bias", (H,)),
        pos_embed=need("vision_tower.pos_embed", (max(vision_cfg.image_seq_length, vision_cfg.merged_tokens), H)),
        layers=tuple(layers),
        norm=need("vision_tower.norm.weight", (H,)),
    )
    projector = vt.VisionProjectorWeights(
        weight=need("multi_modal_projector.weight", (vision_cfg.output_hidden_size, H)).T,
        bias=need("multi_modal_projector.bias", (vision_cfg.output_hidden_size,)),
    )
    return vt.VisionRuntimeWeights(tower=tower, projector=projector)


def import_processor_config(
    root: str | Path,
    *,
    vision: MiniMaxM3VisionMetadata = VISION_METADATA,
) -> MediaProcessorConfig:
    """Import local HF processor metadata, falling back to MiniMax config facts."""
    root_path = _root(root)
    default = minimax_processor_config(vision)
    for name in ("processor_config.json", "preprocessor_config.json", "image_processor_config.json"):
        path = root_path / name
        if path.exists():
            return processor_config_from_hf(_read_json(path), default=default)
    return default


def synthetic_vision_runtime(
    text_cfg: MoETransformerConfig,
    *,
    vision: MiniMaxM3VisionMetadata = VISION_METADATA,
    seed: int = 0,
    vision_hidden_size: int | None = None,
    num_layers: int = 1,
    num_heads: int = 4,
) -> MiniMaxM3VisionRuntime:
    """Construct a deterministic reference MiniMax vision runtime."""
    processor = minimax_processor_config(vision)
    vision_cfg = minimax_vision_transformer_config(
        text_hidden_size=text_cfg.hidden_size,
        vision=vision,
        vision_hidden_size=vision_hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    return MiniMaxM3VisionRuntime(
        config=vision_cfg,
        processor=processor,
        weights=vt.synthetic_weights(vision_cfg, seed=seed),
    )


def import_minimax_m3_hf(
    root: str | Path,
    *,
    expected_config: MoETransformerConfig | None = None,
    require_tokenizer: bool = True,
    require_weights: bool = False,
) -> MiniMaxM3ImportManifest:
    root_path = Path(root)
    cfg_path = root_path / "config.json"
    if not cfg_path.exists():
        raise MiniMaxM3ImportError(f"missing config.json under {root_path}")
    hf_config = _read_json(cfg_path)
    cfg = validate_hf_config(hf_config, expected=expected_config)
    if require_tokenizer and not (
        (root_path / "tokenizer.json").exists() or (root_path / "tokenizer_config.json").exists()
    ):
        raise MiniMaxM3ImportError(f"missing tokenizer.json/tokenizer_config.json under {root_path}")
    tok = import_tokenizer(root_path) if require_tokenizer else None
    processor = import_processor_config(root_path, vision=VISION_METADATA)
    st: SafetensorsManifest | None = None
    try:
        st = read_safetensors_manifest(root_path)
    except MiniMaxM3ImportError:
        if require_weights:
            raise
    return MiniMaxM3ImportManifest(
        root=str(root_path),
        config=cfg,
        hf_config=dict(hf_config),
        tokenizer=tok,
        safetensors=st,
        vision=VISION_METADATA,
        processor=processor,
        text_execution_supported=True,
        vision_execution_supported=VISION_METADATA.vision_execution_supported,
    )


def prepare_multimodal_prompt(
    segments: Sequence[str | PromptSegment | Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    vision: MiniMaxM3VisionMetadata = VISION_METADATA,
) -> PreparedMultimodalPrompt:
    token_ids: list[int] = []
    spans: list[PromptSpan] = []
    for raw in segments:
        seg = _coerce_segment(raw)
        if seg.kind == "text":
            token_ids.extend(int(i) for i in tokenizer.encode(seg.text))
        elif seg.kind == "image":
            start = len(token_ids)
            token_ids.extend([vision.image_token_index] * vision.image_seq_length)
            spans.append(PromptSpan("image", start, len(token_ids), vision.image_token_index, frames=1))
        elif seg.kind == "video":
            if seg.frames < 1 or seg.frames > vision.vision_segment_max_frames:
                raise MiniMaxM3ImportError(
                    f"video frames={seg.frames} outside [1, {vision.vision_segment_max_frames}]")
            start = len(token_ids)
            count = vision.image_seq_length * seg.frames
            token_ids.extend([vision.video_token_index] * count)
            spans.append(PromptSpan("video", start, len(token_ids), vision.video_token_index, frames=seg.frames))
        else:
            raise MiniMaxM3ImportError(f"unknown prompt segment kind {seg.kind!r}")
    return PreparedMultimodalPrompt(
        token_ids=tuple(token_ids),
        spans=tuple(spans),
        vision_execution_supported=vision.vision_execution_supported,
    )


def _coerce_segment(raw: str | PromptSegment | Mapping[str, Any]) -> PromptSegment:
    if isinstance(raw, PromptSegment):
        return raw
    if isinstance(raw, str):
        return PromptSegment("text", text=raw)
    return PromptSegment(str(raw.get("kind", "text")), text=str(raw.get("text", "")),
                         frames=int(raw.get("frames", 1)))


def execute_multimodal_prompt(
    config: MoETransformerConfig,
    weights: rt.ModelWeights,
    prepared: PreparedMultimodalPrompt,
    *,
    media_embeddings: MediaEmbeddingInput | None = None,
    media_inputs: MediaInput | None = None,
    vision_runtime: MiniMaxM3VisionRuntime | None = None,
):
    """Execute a prepared prompt through the shared decoder stack.

    Image/video spans require projected media embeddings with one row per
    placeholder token. Without those embeddings, media prompts still fail
    loudly instead of silently running as fake text tokens.
    """
    if prepared.has_vision:
        if media_embeddings is None and media_inputs is not None and vision_runtime is not None:
            media_embeddings = encode_media_inputs(
                config, prepared, media_inputs=media_inputs, vision_runtime=vision_runtime).embeddings
        if media_embeddings is None:
            raise MiniMaxM3VisionExecutionError(
                "MiniMax-M3 vision/video execution requires projected media embeddings "
                "or raw media_inputs plus a vision_runtime; "
                f"prompt contains {len(prepared.spans)} media span(s)")
        input_embeds = splice_media_embeddings(
            config, weights, prepared, media_embeddings=media_embeddings)
        return rt.forward_embeds(config, weights, input_embeds)
    bad_ids = [tok for tok in prepared.token_ids if tok < 0 or tok >= config.vocab_size]
    if bad_ids:
        raise MiniMaxM3ImportError(
            f"text token ids outside vocab_size={config.vocab_size}: {bad_ids[:5]}")
    return rt.forward(config, weights, list(prepared.token_ids))


def prefill_multimodal_prompt(
    config: MoETransformerConfig,
    weights: rt.ModelWeights,
    prepared: PreparedMultimodalPrompt,
    *,
    media_embeddings: MediaEmbeddingInput | None = None,
    media_inputs: MediaInput | None = None,
    vision_runtime: MiniMaxM3VisionRuntime | None = None,
    max_seq: int | None = None,
):
    """Prefill decoder KV caches for a text or projected-media prompt."""
    if prepared.has_vision:
        if media_embeddings is None and media_inputs is not None and vision_runtime is not None:
            media_embeddings = encode_media_inputs(
                config, prepared, media_inputs=media_inputs, vision_runtime=vision_runtime).embeddings
        if media_embeddings is None:
            raise MiniMaxM3VisionExecutionError(
                "MiniMax-M3 vision/video prefill requires projected media embeddings "
                "or raw media_inputs plus a vision_runtime; "
                f"prompt contains {len(prepared.spans)} media span(s)")
        input_embeds = splice_media_embeddings(
            config, weights, prepared, media_embeddings=media_embeddings)
        return rt.prefill_embeds(config, weights, input_embeds, max_seq=max_seq)
    return rt.prefill(config, weights, list(prepared.token_ids), max_seq=max_seq)


def encode_media_inputs(
    config: MoETransformerConfig,
    prepared: PreparedMultimodalPrompt,
    *,
    media_inputs: MediaInput,
    vision_runtime: MiniMaxM3VisionRuntime,
) -> ProjectedMediaEmbeddings:
    """Run raw image/video tensors through the reference tower + projector."""
    if vision_runtime.config.output_hidden_size != config.hidden_size:
        raise MiniMaxM3ImportError(
            "vision runtime projector hidden size "
            f"{vision_runtime.config.output_hidden_size} != text hidden size {config.hidden_size}")
    embeddings: list[np.ndarray] = []
    spans = tuple(MediaSpan(span.kind, span.start, span.end, frames=span.frames) for span in prepared.spans)
    for ordinal, span in enumerate(prepared.spans):
        media = _media_input_for_span(media_inputs, span, ordinal)
        if span.kind == "image":
            emb = vt.encode_image(
                media,
                vision_runtime.config,
                vision_runtime.weights,
                processor=vision_runtime.processor,
            )
        elif span.kind == "video":
            emb = vt.encode_video(
                media,
                vision_runtime.config,
                vision_runtime.weights,
                processor=vision_runtime.processor,
                frames=span.frames,
            )
        else:
            raise MiniMaxM3ImportError(f"unsupported media span kind {span.kind!r}")
        expected = (span.end - span.start, config.hidden_size)
        if emb.shape != expected:
            raise MiniMaxM3ImportError(
                f"encoded media span {ordinal} shape {emb.shape}, expected {expected}")
        embeddings.append(emb)
    projected = ProjectedMediaEmbeddings(tuple(embeddings), spans=spans, hidden_size=config.hidden_size)
    projected.validate()
    return projected


def _media_input_for_span(media_inputs: MediaInput, span: PromptSpan, ordinal: int):
    if isinstance(media_inputs, Mapping):
        candidates: tuple[int | str, ...] = (
            ordinal,
            str(ordinal),
            span.kind,
            f"{span.kind}:{ordinal}",
        )
        for key in candidates:
            if key in media_inputs:
                return media_inputs[key]
        raise MiniMaxM3ImportError(f"missing media input for span {ordinal} ({span.kind})")
    if ordinal >= len(media_inputs):
        raise MiniMaxM3ImportError(f"missing media input for span {ordinal} ({span.kind})")
    return media_inputs[ordinal]


def splice_media_embeddings(
    config: MoETransformerConfig,
    weights: rt.ModelWeights,
    prepared: PreparedMultimodalPrompt,
    *,
    media_embeddings: MediaEmbeddingInput,
) -> np.ndarray:
    """Replace image/video placeholder-token positions with projector outputs.

    ``media_embeddings`` may be a sequence aligned to ``prepared.spans`` or a
    mapping keyed by span ordinal (``0``), span kind (``"image"``/``"video"``),
    or ``"{kind}:{ordinal}"``. Each value must be shaped
    ``(span.end - span.start, hidden_size)``.
    """
    token_ids = np.asarray(prepared.token_ids, dtype=np.int64)
    x = np.zeros((token_ids.size, config.hidden_size), dtype=np.float64)
    media_mask = np.zeros(token_ids.size, dtype=bool)
    for span in prepared.spans:
        media_mask[span.start:span.end] = True

    text_positions = np.nonzero(~media_mask)[0]
    if text_positions.size:
        text_ids = token_ids[text_positions]
        bad_ids = [int(tok) for tok in text_ids if tok < 0 or tok >= config.vocab_size]
        if bad_ids:
            raise MiniMaxM3ImportError(
                f"text token ids outside vocab_size={config.vocab_size}: {bad_ids[:5]}")
        x[text_positions] = weights.embed[text_ids].astype(np.float64)

    if len(prepared.spans) == 0:
        return x

    for ordinal, span in enumerate(prepared.spans):
        emb = _media_embedding_for_span(media_embeddings, span, ordinal)
        arr = np.asarray(emb, dtype=np.float64)
        expected = (span.end - span.start, config.hidden_size)
        if arr.shape != expected:
            raise MiniMaxM3ImportError(
                f"media embedding for span {ordinal} has shape {arr.shape}, expected {expected}")
        x[span.start:span.end] = arr
    return x


def _media_embedding_for_span(
    media_embeddings: MediaEmbeddingInput,
    span: PromptSpan,
    ordinal: int,
) -> np.ndarray:
    if isinstance(media_embeddings, Mapping):
        candidates: tuple[int | str, ...] = (
            ordinal,
            str(ordinal),
            span.kind,
            f"{span.kind}:{ordinal}",
        )
        for key in candidates:
            if key in media_embeddings:
                return media_embeddings[key]
        raise MiniMaxM3ImportError(f"missing media embedding for span {ordinal} ({span.kind})")
    if ordinal >= len(media_embeddings):
        raise MiniMaxM3ImportError(f"missing media embedding for span {ordinal} ({span.kind})")
    return media_embeddings[ordinal]


__all__ = [
    "MiniMaxM3ImportError",
    "MiniMaxM3VisionExecutionError",
    "MiniMaxM3TokenizerSpec",
    "MiniMaxM3TokenizerImport",
    "TensorSpec",
    "SafetensorsManifest",
    "MiniMaxM3ImportManifest",
    "PromptSegment",
    "PromptSpan",
    "PreparedMultimodalPrompt",
    "validate_hf_config",
    "import_tokenizer",
    "read_safetensors_manifest",
    "load_safetensors_tensors",
    "expected_hf_text_tensor_shapes",
    "expected_hf_vision_tensor_shapes",
    "import_processor_config",
    "load_vision_runtime_weights",
    "validate_safetensors_shapes",
    "import_minimax_m3_hf",
    "prepare_multimodal_prompt",
    "encode_media_inputs",
    "synthetic_vision_runtime",
    "splice_media_embeddings",
    "execute_multimodal_prompt",
    "prefill_multimodal_prompt",
    "MiniMaxM3VisionRuntime",
    "MiniMaxM3VisionWeights",
    "MiniMaxM3ProjectorWeights",
]
