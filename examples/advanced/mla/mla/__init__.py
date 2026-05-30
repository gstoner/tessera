"""Current-compiler FlashMLA sample package."""

from .compiler_smoke import build_toy_graph_ir, compile_toy_graph
from .config import MLAConfig, tiny_config
from .gpu_decode import GPUDecodeSummary, run_gpu_decode_demo
from .numpy_reference import MLAResult, MultiLatentAttentionNumpy

__all__ = [
    "MLAConfig",
    "MLAResult",
    "MultiLatentAttentionNumpy",
    "GPUDecodeSummary",
    "build_toy_graph_ir",
    "compile_toy_graph",
    "run_gpu_decode_demo",
    "tiny_config",
]
