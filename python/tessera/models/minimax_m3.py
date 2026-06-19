"""MiniMax-M3 config factory — GQA + MSA + MoE + multimodal metadata.

MiniMax-M3's released Hugging Face config identifies a native multimodal
``minimax_m3_vl`` model with a 60-layer text tower: GQA attention, MiniMax
Sparse Attention (MSA) after the first three dense layers, 128 routed experts
with top-4 routing, one shared expert, and BF16 weights.  This module keeps the
text tower as a compiler-visible contract and records the vision/video metadata
as importer-side staging only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diffusion_gemma import GraphNode
from .moe_transformer import MoETransformerConfig
from .multimodal import MediaProcessorConfig, processor_config_from_metadata
from .vision_transformer import VisionTransformerConfig, config_from_processor


MINIMAX_M3_SPARSE_LAYER_FREQ: tuple[int, ...] = (0, 0, 0) + (1,) * 57


@dataclass(frozen=True)
class MiniMaxM3VisionMetadata:
    """Importer-visible multimodal metadata; execution is intentionally staged."""

    image_token_index: int = 200025
    video_token_index: int = 200026
    image_seq_length: int = 576
    patch_size: int = 14
    image_size: int = 2016
    projector_hidden_size: int = 6144
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    vision_segment_max_frames: int = 4
    vision_execution_supported: bool = False


VISION_METADATA = MiniMaxM3VisionMetadata()


@dataclass(frozen=True)
class MiniMaxM3MultimodalGraph:
    """Shape-only MiniMax-M3 image/video tower + splice contract graph."""

    nodes: tuple[GraphNode, ...]
    text_config: MoETransformerConfig
    vision: MiniMaxM3VisionMetadata
    frames: int

    def op_sequence(self) -> tuple[str, ...]:
        return tuple(n.op for n in self.nodes)

    def find(self, op: str) -> GraphNode:
        for n in self.nodes:
            if n.op == op:
                return n
        raise KeyError(op)

    def find_all(self, op: str) -> tuple[GraphNode, ...]:
        return tuple(n for n in self.nodes if n.op == op)


def processor_config(vision: MiniMaxM3VisionMetadata = VISION_METADATA) -> MediaProcessorConfig:
    return processor_config_from_metadata(
        image_size=vision.image_size,
        patch_size=vision.patch_size,
        image_seq_length=vision.image_seq_length,
        spatial_merge_size=vision.spatial_merge_size,
        temporal_patch_size=vision.temporal_patch_size,
        max_frames=vision.vision_segment_max_frames,
    )


def vision_transformer_config(
    *,
    text_hidden_size: int | None = None,
    vision: MiniMaxM3VisionMetadata = VISION_METADATA,
    vision_hidden_size: int | None = None,
    num_layers: int = 1,
    num_heads: int = 4,
) -> VisionTransformerConfig:
    text_hidden = text_hidden_size or vision.projector_hidden_size
    return config_from_processor(
        processor_config(vision),
        output_hidden_size=text_hidden,
        hidden_size=vision_hidden_size or min(text_hidden, 1024),
        num_layers=num_layers,
        num_heads=num_heads,
    )


def scaled_vision_metadata() -> MiniMaxM3VisionMetadata:
    """Small executable metadata for reference image/video tower tests."""
    return MiniMaxM3VisionMetadata(
        image_token_index=2000,
        video_token_index=2001,
        image_seq_length=4,
        patch_size=2,
        image_size=8,
        projector_hidden_size=256,
        spatial_merge_size=2,
        temporal_patch_size=1,
        vision_segment_max_frames=2,
        vision_execution_supported=True,
    )


def build_multimodal_graph(
    text_config: MoETransformerConfig | None = None,
    *,
    vision: MiniMaxM3VisionMetadata = VISION_METADATA,
    frames: int = 1,
    include_image: bool = True,
    include_video: bool = True,
) -> MiniMaxM3MultimodalGraph:
    """Build a compiler-visible MiniMax-M3 media tower/splice graph.

    Full MiniMax geometry is a shape/import contract.  Scaled configs can run
    through the numpy reference tower, while this graph records the backend
    surfaces expected to lower natively later: preprocess/frame sampling, patch
    embed, patch merge, projector, and text/media splice.
    """
    cfg = text_config or config()
    if frames < 1 or frames > vision.vision_segment_max_frames:
        raise ValueError(f"frames={frames} outside [1, {vision.vision_segment_max_frames}]")
    if vision.projector_hidden_size != cfg.hidden_size:
        raise ValueError(
            f"projector_hidden_size={vision.projector_hidden_size} != hidden_size={cfg.hidden_size}")

    nodes: list[GraphNode] = []
    raw_patches = (vision.image_size // vision.patch_size) ** 2

    def add(op: str, inputs, output, **attrs) -> None:
        nodes.append(GraphNode(op=op, inputs=tuple(inputs), output=tuple(output), attrs=attrs))

    media_inputs: list[tuple] = []
    if include_image:
        add(
            "image_preprocess",
            [("image_pixels", vision.image_size, vision.image_size, 3)],
            (vision.image_size, vision.image_size, 3),
            image_size=vision.image_size,
            patch_size=vision.patch_size,
            image_seq_length=vision.image_seq_length,
        )
        add(
            "patch_embed",
            [(vision.image_size, vision.image_size, 3)],
            (raw_patches, vision.projector_hidden_size),
            patch_size=vision.patch_size,
            media_kind="image",
        )
        add(
            "patch_merge",
            [(raw_patches, vision.projector_hidden_size)],
            (vision.image_seq_length, vision.projector_hidden_size),
            spatial_merge_size=vision.spatial_merge_size,
            temporal_patch_size=1,
            media_kind="image",
        )
        add(
            "media_project",
            [(vision.image_seq_length, vision.projector_hidden_size),
             (vision.projector_hidden_size, cfg.hidden_size)],
            (vision.image_seq_length, cfg.hidden_size),
            media_kind="image",
        )
        media_inputs.append(("image_projected", vision.image_seq_length, cfg.hidden_size))

    if include_video:
        add(
            "video_frame_sample",
            [("video_pixels", frames, vision.image_size, vision.image_size, 3)],
            (frames, vision.image_size, vision.image_size, 3),
            frames=frames,
            max_frames=vision.vision_segment_max_frames,
            temporal_patch_size=vision.temporal_patch_size,
        )
        add(
            "patch_embed",
            [(frames, vision.image_size, vision.image_size, 3)],
            (frames * raw_patches, vision.projector_hidden_size),
            patch_size=vision.patch_size,
            media_kind="video",
        )
        add(
            "patch_merge",
            [(frames * raw_patches, vision.projector_hidden_size)],
            (frames * vision.image_seq_length, vision.projector_hidden_size),
            spatial_merge_size=vision.spatial_merge_size,
            temporal_patch_size=vision.temporal_patch_size,
            media_kind="video",
        )
        add(
            "media_project",
            [(frames * vision.image_seq_length, vision.projector_hidden_size),
             (vision.projector_hidden_size, cfg.hidden_size)],
            (frames * vision.image_seq_length, cfg.hidden_size),
            media_kind="video",
        )
        media_inputs.append(("video_projected", frames * vision.image_seq_length, cfg.hidden_size))

    add(
        "splice_embeddings",
        [("text_embeddings", "T", cfg.hidden_size), *media_inputs],
        ("T+media", cfg.hidden_size),
        image_token_index=vision.image_token_index,
        video_token_index=vision.video_token_index,
        vision_execution_supported=vision.vision_execution_supported,
    )
    graph = MiniMaxM3MultimodalGraph(
        nodes=tuple(nodes),
        text_config=cfg,
        vision=vision,
        frames=frames,
    )
    verify_multimodal_graph(graph)
    return graph


def verify_multimodal_graph(graph: MiniMaxM3MultimodalGraph) -> None:
    cfg = graph.text_config
    vision = graph.vision
    projects = graph.find_all("media_project")
    if not projects:
        raise ValueError("multimodal graph must project at least one media stream")
    for node in projects:
        if node.output[-1] != cfg.hidden_size:
            raise ValueError("media_project output width must match text hidden_size")
    splice = graph.find("splice_embeddings")
    if splice.output[-1] != cfg.hidden_size:
        raise ValueError("splice_embeddings output width must match text hidden_size")
    image_projects = tuple(n for n in projects if n.attrs.get("media_kind") == "image")
    if image_projects and image_projects[0].output[0] != vision.image_seq_length:
        raise ValueError("image media_project length must equal image_seq_length")
    video_projects = tuple(n for n in projects if n.attrs.get("media_kind") == "video")
    if video_projects and video_projects[0].output[0] != graph.frames * vision.image_seq_length:
        raise ValueError("video media_project length must equal frames * image_seq_length")


def config() -> MoETransformerConfig:
    """Full-scale MiniMax-M3 text-tower contract from the released HF config."""
    return MoETransformerConfig(
        name="minimax_m3",
        hidden_size=6144,
        num_layers=60,
        vocab_size=200064,
        context_length=1_048_576,
        attn_kind="gqa",
        num_attention_heads=64,
        num_kv_heads=4,
        head_dim=128,
        rope_head_dim=64,
        rope_theta=5_000_000.0,
        sparse="msa",
        msa_top_k_blocks=16,
        msa_block_size=128,
        msa_index_dim=128,
        msa_num_index_heads=4,
        msa_score_type="max",
        msa_sparse_layer_freq=MINIMAX_M3_SPARSE_LAYER_FREQ,
        num_experts=128,
        num_experts_per_tok=4,
        num_shared_experts=1,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=3072,
        first_k_dense=3,
        dense_intermediate_size=12288,
        dtype="bf16",
        total_params_b=427.04,
        active_params_b=23.0,
        hf_model_size_b=427.04,
    )


def scaled_config() -> MoETransformerConfig:
    """Mac-executable shrink preserving dense warmup, GQA, MSA, and MoE shape.

    The small ``msa_block_size`` intentionally creates multiple KV blocks in
    unit tests, so runtime decode proves real MSA selection instead of a
    one-block dense fallback.
    """
    return MoETransformerConfig(
        name="minimax_m3_scaled",
        hidden_size=256,
        num_layers=4,
        vocab_size=2048,
        context_length=512,
        attn_kind="gqa",
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=32,
        rope_head_dim=16,
        rope_theta=5_000_000.0,
        sparse="msa",
        msa_top_k_blocks=2,
        msa_block_size=4,
        msa_index_dim=32,
        msa_num_index_heads=2,
        msa_score_type="max",
        msa_sparse_layer_freq=(0, 1, 1, 1),
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        first_k_dense=1,
        dense_intermediate_size=512,
        dtype="bf16",
    )


__all__ = [
    "MiniMaxM3MultimodalGraph",
    "MiniMaxM3VisionMetadata",
    "MINIMAX_M3_SPARSE_LAYER_FREQ",
    "VISION_METADATA",
    "build_multimodal_graph",
    "config",
    "processor_config",
    "scaled_config",
    "scaled_vision_metadata",
    "verify_multimodal_graph",
    "vision_transformer_config",
]
