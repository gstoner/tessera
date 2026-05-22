import torch, re
from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM
from tessera_gemma.peft.lora import apply_lora_regex, lora_state_dict, merge_lora, unmerge_lora, LoRALinear, apply_qlora_sim

def test_regex_and_composition():
    cfg = GemmaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_kv_heads=2)
    m = TesseraGemmaForCausalLM(cfg)

    rules = [
        {'pattern': r'.*attn.*qkv.*', 'name': 'qkv', 'rank': 4, 'alpha': 8.0, 'dropout': 0.0},
        {'pattern': r'.*attn.*proj.*', 'name': 'proj', 'rank': 2, 'alpha': 4.0, 'dropout': 0.0},
    ]
    created = apply_lora_regex(m, rules)
    assert created > 0

    # check at least one linear became LoRALinear with 2 adapters
    has_two = False
    for _, mod in m.named_modules():
        if isinstance(mod, LoRALinear) and len(mod.adapters) >= 2:
            has_two = True; break
    assert has_two

    # qlora sim wraps some linears
    wrapped = apply_qlora_sim(m, patterns=("qkv","proj"))
    assert wrapped >= 1

    # merge/unmerge runs
    merge_lora(m); unmerge_lora(m)

def test_paged_decode_runs():
    cfg = GemmaConfig(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_kv_heads=2, max_position_embeddings=512)
    m = TesseraGemmaForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    from tessera_gemma.utils.decoding import greedy_generate_paged
    with torch.no_grad():
        y = greedy_generate_paged(m, x, max_new_tokens=8)
    assert y.shape == (1, 8)
