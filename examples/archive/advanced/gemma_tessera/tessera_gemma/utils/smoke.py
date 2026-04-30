"""
Quick smoke test — verifies the model runs end-to-end with debug_tiny config.
Run directly:  python -m tessera_gemma.utils.smoke
"""

import torch
from ..configs import GemmaConfig
from ..model_tessera import TesseraGemmaForCausalLM


def main() -> None:
    print("=== Tessera-Gemma smoke test ===")

    # --- basic forward ---
    cfg = GemmaConfig.debug_tiny()
    m = TesseraGemmaForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 16, cfg.vocab_size), f"unexpected shape {y.shape}"
    print(f"forward OK: {tuple(y.shape)}")

    # --- generate (greedy, no cache) ---
    with torch.no_grad():
        out = m.generate(x, max_new_tokens=4, top_k=1, use_cache=False)
    assert out.shape == (2, 20), f"generate shape {out.shape}"
    print(f"generate OK: {tuple(out.shape)}")

    # --- config variants ---
    for name, factory in [
        ("gemma4_4b",  GemmaConfig.gemma4_4b),
        ("gemma4_12b", GemmaConfig.gemma4_12b),
        ("gemma4_27b", GemmaConfig.gemma4_27b),
    ]:
        cfg2 = factory()
        assert cfg2.head_dim == 256
        assert cfg2.layer_attention_kind(0) == "full"
        assert cfg2.layer_attention_kind(1) == "sliding_window"
        print(f"config {name} OK: hidden={cfg2.hidden_size}, "
              f"layers={cfg2.num_hidden_layers}, groups={cfg2.groups}")

    # --- LoRA smoke ---
    from ..peft import apply_lora, lora_state_dict
    m2 = TesseraGemmaForCausalLM(GemmaConfig.debug_tiny()).eval()
    n = apply_lora(m2, patterns=["q_proj"], rank=4)
    sd = lora_state_dict(m2)
    assert len(sd) == n * 2   # A + B per adapter
    print(f"LoRA OK: {n} adapters, {len(sd)} tensors in lora_state_dict")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
