from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def temporal_attention_regularizers(
    attention: torch.Tensor, token_time_index: torch.Tensor, *, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coverage and consecutive-tick redundancy from attention [B,T,H,L]."""
    attention = attention.float().mean(dim=2)  # [B,T,L], averaged over heads
    times, inverse = torch.unique(token_time_index, sorted=True, return_inverse=True)
    temporal = attention.new_zeros((*attention.shape[:2], len(times)))
    temporal.scatter_add_(2, inverse[None, None, :].expand_as(attention), attention)
    temporal = temporal / temporal.sum(dim=-1, keepdim=True).clamp_min(eps)
    mean_time = temporal.mean(dim=1)
    uniform = 1.0 / len(times)
    coverage = (uniform * (torch.log(mean_time.new_tensor(uniform)) - mean_time.clamp_min(eps).log())).sum(dim=-1).mean()
    if temporal.shape[1] > 1:
        redundancy = F.cosine_similarity(temporal[:, 1:], temporal[:, :-1], dim=-1, eps=eps).mean()
    else:
        redundancy = temporal.new_zeros(())
    return coverage, redundancy


def ctm_classification_loss(
    predictions: torch.Tensor,
    certainties: torch.Tensor,
    targets: torch.Tensor,
    *,
    final_ce_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Official-style CTM adaptive classification loss + optional final-tick CE.

    predictions: [B,C,T]
    certainties: [B,T]

    The adaptive term averages CE at (a) each sample's minimum-loss tick and
    (b) each sample's most-certain tick, mirroring the public CTM classification
    objective. final_ce_weight is an explicit task-specific stabilizer; set it
    to 0.0 for the pure adaptive objective.
    """
    B, C, T = predictions.shape
    target_ticks = targets[:, None].expand(B, T)
    per_tick_ce = F.cross_entropy(predictions, target_ticks, reduction="none")
    best_loss_tick = per_tick_ce.argmin(1)
    most_certain_tick = certainties.argmax(1)
    b = torch.arange(B, device=predictions.device)
    adaptive = 0.5 * (per_tick_ce[b, best_loss_tick].mean() + per_tick_ce[b, most_certain_tick].mean())
    final_ce = F.cross_entropy(predictions[..., -1], targets)
    total = adaptive + final_ce_weight * final_ce
    return total, {
        "adaptive_loss": adaptive.detach(),
        "final_ce": final_ce.detach(),
        "best_loss_tick": best_loss_tick.detach(),
        "most_certain_tick": most_certain_tick.detach(),
    }
