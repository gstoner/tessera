
from dataclasses import dataclass
from typing import Dict, Any
import torch, torch.nn as nn

@dataclass
class PolicyOutput:
    tokens: torch.Tensor   # [T]
    logits: torch.Tensor   # [T, V]
    text: str
    info: Dict[str, Any]

class TinyLSTMPolicy(nn.Module):
    def __init__(self, vocab_size=256, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, hc=None):
        emb = self.embedding(x)
        y, hc = self.lstm(emb, hc)
        logits = self.head(y)
        return logits, hc

    @torch.no_grad()
    def sample(self, prompt: str, max_len=256, temperature=1.0) -> PolicyOutput:
        self.eval()
        device = next(self.parameters()).device
        eos = 3
        x = torch.tensor(list(prompt.encode('utf-8'))[: max_len-1] + [eos], dtype=torch.long, device=device).unsqueeze(0)
        logits, hc = self.forward(x)
        tokens = [int(t) for t in x[0]]
        text = prompt
        info = {"turn_tool_calls": 0, "format": {"answer_tags": 0}}
        suffix = "\nansweranswer\n \\boxed{0} \n /answeranswer"
        suffix_ids = list(suffix.encode('utf-8'))
        tokens.extend(suffix_ids[:max_len-len(tokens)])
        T = len(tokens)
        out_logits = torch.zeros(T, logits.shape[-1], device=device)
        out_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        return PolicyOutput(out_tokens, out_logits, bytes(tokens).decode('utf-8', errors='ignore'), info)

    def logprob(self, tokens: torch.Tensor, prompt_len: int) -> torch.Tensor:
        return torch.tensor(0.0, device=tokens.device)

class HFPolicy:
    def __init__(self, model_name: str, device='cuda'):
        raise NotImplementedError("Wire your HF model here to compute per-trajectory logprobs.")
