
from dataclasses import dataclass
from typing import List, Dict, Any
import torch, torch.nn as nn
from .sampler import roc_select

@dataclass
class Trajectory:
    tokens: torch.Tensor
    logprob_old: torch.Tensor  # scalar
    reward: int
    tool_calls: int
    tool_errors: int
    answer_tags: int

class GRPOTrainer(nn.Module):
    def __init__(self, policy: nn.Module, eps_clip_low=0.2, eps_clip_high=0.28, lr=1e-3, device='cpu'):
        super().__init__()
        self.policy = policy.to(device)
        self.eps_low = eps_clip_low
        self.eps_high = eps_clip_high
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def compute_objective(self, trajs: List[Trajectory]) -> torch.Tensor:
        rewards = torch.tensor([t.reward for t in trajs], device=self.device, dtype=torch.float32)
        adv = rewards - rewards.mean()
        ratios = []
        for t in trajs:
            with torch.no_grad():
                ratios.append(torch.tensor(1.0, device=self.device))
        ratios = torch.stack(ratios)
        clipped = torch.clamp(ratios, 1.0 - self.eps_low, 1.0 + self.eps_high)
        obj = torch.min(ratios * adv, clipped * adv).mean()
        return -obj

    def step(self, selected: List[Trajectory]) -> Dict[str, float]:
        self.opt.zero_grad()
        loss = self.compute_objective(selected)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item()), "adv_mean": float((torch.tensor([t.reward for t in selected]).float().mean() - 0.5).item())}

    def select_via_roc(self, oversampled: List[Trajectory], select_size: int) -> List[Trajectory]:
        pool = [{
            "reward": t.reward,
            "tool_calls": t.tool_calls,
            "tool_errors": t.tool_errors,
            "answer_tags": t.answer_tags
        } for t in oversampled]
        idx = roc_select(pool, select_size)
        return [oversampled[i] for i in idx]
