from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSLConfig:
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    ema_momentum: float = 0.996
    gram_weight: float = 0.0
    gram_layers: List[int] = field(default_factory=lambda: [])  # e.g., [6, 12]


def _gram_matrix(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens: (B, N, C). Return averaged Gram over batch: (C, C).
    G = (X^T X) / N, averaged over batch.
    """
    B, N, C = tokens.shape
    X = tokens.reshape(B * N, C)
    G = (X.t() @ X) / (B * N)
    return G / C  # mild scale normalization


class DINOSSL(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, head_student: nn.Module, head_teacher: nn.Module, cfg: SSLConfig):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.head_student = head_student
        self.head_teacher = head_teacher
        self.cfg = cfg
        self.register_buffer("center", torch.zeros(1, head_teacher.prototypes.out_features))

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _ema_update_teacher(self, m: float):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.lerp_(ps.data, 1.0 - m)

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor, m: float):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(m).add_(batch_center * (1 - m))

    def forward(self, views: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        views: list of augmented images; first two are 'global' crops
        """
        # Student
        s_logits_list = []
        s_layers_list: List[Dict[int, torch.Tensor]] = []
        for v in views:
            s_out, s_tokens, s_layers = self.student(v, return_tokens=True, return_layers=True)
            s_logits = self.head_student(s_out)
            s_logits_list.append(s_logits)
            s_layers_list.append(s_layers or {})

        # Teacher on global views only
        with torch.no_grad():
            t_logits_list = []
            t_layers_list: List[Dict[int, torch.Tensor]] = []
            for v in views[:2]:
                t_out, t_tokens, t_layers = self.teacher(v, return_tokens=True, return_layers=True)
                t_logits = self.head_teacher(t_out)
                t_logits_list.append(t_logits)
                t_layers_list.append(t_layers or {})

        # Temperatures and centering
        t_targets = [(t - self.center) / self.cfg.teacher_temp for t in t_logits_list]
        t_targets = [F.softmax(t, dim=-1) for t in t_targets]
        s_logits_t = [s / self.cfg.student_temp for s in s_logits_list]

        # Cross-view CE
        total_loss = 0.0
        n_terms = 0
        for i, s in enumerate(s_logits_t):
            for j, t in enumerate(t_targets):
                if i == j and i < 2:
                    continue  # skip same global-global pairing
                total_loss = total_loss + torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1).mean()
                n_terms += 1
        total_loss = total_loss / max(n_terms, 1)

        # Token-level Gram anchoring across selected layers
        gram_loss_val = torch.tensor(0.0, device=views[0].device)
        if self.cfg.gram_weight > 0.0 and len(self.cfg.gram_layers) > 0:
            for layer_idx in self.cfg.gram_layers:
                # Teacher reference Gram from global crops (average the two globals)
                grams_t = []
                for t_layers in t_layers_list:
                    if layer_idx in t_layers:
                        grams_t.append(_gram_matrix(t_layers[layer_idx].detach()))
                if len(grams_t) == 0:
                    continue
                Gt = torch.stack(grams_t, dim=0).mean(dim=0)

                # Match each student view to teacher Gram
                grams_s = []
                for s_layers in s_layers_list:
                    if layer_idx in s_layers:
                        grams_s.append(_gram_matrix(s_layers[layer_idx]))
                if len(grams_s) == 0:
                    continue
                Gs = torch.stack(grams_s, dim=0).mean(dim=0)
                gram_loss_val = gram_loss_val + F.mse_loss(Gs, Gt)

            total_loss = total_loss + self.cfg.gram_weight * gram_loss_val

        # EMA & center update
        with torch.no_grad():
            self._ema_update_teacher(self.cfg.ema_momentum)
            self._update_center(torch.cat(t_logits_list, dim=0), self.cfg.center_momentum)

        metrics = {
            "loss_ce": (total_loss - self.cfg.gram_weight * gram_loss_val).detach(),
            "loss_gram": gram_loss_val.detach(),
            "center_norm": self.center.norm().detach(),
        }
        return total_loss, metrics
