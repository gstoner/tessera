"""Current-compiler FlashMLA sample package."""

from .compiler_smoke import build_toy_graph_ir, compile_toy_graph
from .config import MLAConfig, tiny_config
from .numpy_reference import MLAResult, MultiLatentAttentionNumpy

__all__ = [
    "MLAConfig",
    "MLAResult",
    "MultiLatentAttentionNumpy",
    "build_toy_graph_ir",
    "compile_toy_graph",
    "tiny_config",
]
