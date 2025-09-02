
from typing import List
import torch

@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    return texts, inputs
