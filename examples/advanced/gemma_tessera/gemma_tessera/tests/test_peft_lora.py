import torch
from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM
from tessera_gemma.peft.lora import apply_lora, lora_state_dict, load_lora_state_dict, merge_lora, unmerge_lora

def test_lora_roundtrip():
    cfg = GemmaConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_kv_heads=2)
    m = TesseraGemmaForCausalLM(cfg)
    _ = apply_lora(m, rank=4, alpha=8.0)

    # Save state
    s1 = lora_state_dict(m)
    assert len(s1) > 0

    # Modify and load back
    for k in list(s1.keys()):
        s1[k] = s1[k] + 1.0
        break
    load_lora_state_dict(m, s1)

    # Merge/Unmerge no-crash
    merge_lora(m)
    unmerge_lora(m)
