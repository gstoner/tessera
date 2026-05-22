"""
tessera_gemma — Tessera port of the Gemma decoder-only language model.

Quick start::

    from tessera_gemma import GemmaConfig, TesseraGemmaForCausalLM

    cfg   = GemmaConfig.gemma4_4b()
    model = TesseraGemmaForCausalLM(cfg).eval()
    out   = model.generate(input_ids, max_new_tokens=32)
"""

from .configs import GemmaConfig
from .model_tessera import TesseraGemmaForCausalLM, GemmaDecoderBlock

__version__ = "0.4.0"
__all__ = ["GemmaConfig", "TesseraGemmaForCausalLM", "GemmaDecoderBlock"]
