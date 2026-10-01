from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, d: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d, d * 2, kernel_size, padding=pad),
            nn.GLU(dim=1),
            nn.GroupNorm(1, d),
            nn.Dropout(dropout),
            nn.Conv1d(d, d, kernel_size, padding=pad),
        )
        self.norm = nn.GroupNorm(1, d)

    def forward(self, x):
        return self.norm(x + self.net(x))


class DynamicGraphBlock(nn.Module):
    """Permutation-equivariant fully-connected message passing at each physical time."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        edge_in = d + 2 + 2 + 1 + 1  # h_j, relative pos/vel, distance, source mass
        self.message = nn.Sequential(
            nn.Linear(edge_in, d * 2), nn.GLU(), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.update = nn.Sequential(
            nn.Linear(d * 2, d * 2), nn.GLU(), nn.LayerNorm(d), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, state: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        # h [B,T,N,D], state [B,T,N,4], mass [B,N]
        B, T, N, D = h.shape
        pos, vel = state[..., :2], state[..., 2:]
        rel_pos = pos[:, :, None, :, :] - pos[:, :, :, None, :]
        rel_vel = vel[:, :, None, :, :] - vel[:, :, :, None, :]
        dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True).clamp_min(1e-5)
        source_mass = mass[:, None, None, :, None].expand(B, T, N, N, 1)
        hj = h[:, :, None, :, :].expand(B, T, N, N, D)
        edge = torch.cat([hj, rel_pos, rel_vel, dist, source_mass], dim=-1)
        msg = self.message(edge)
        mask = (~torch.eye(N, dtype=torch.bool, device=h.device))[None, None, :, :, None]
        msg = msg * mask
        agg = msg.sum(3) / max(N - 1, 1)
        upd = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + upd)


class GraphTemporalEncoder(nn.Module):
    """Local temporal + pairwise gravitational feature extractor.

    It intentionally does not perform global self-attention. The encoder extracts local
    motion/interaction motifs; the CTM remains responsible for iterative global reading.
    """
    def __init__(
        self,
        d_hidden: int = 96,
        d_out: int = 128,
        temporal_blocks: int = 2,
        graph_blocks: int = 2,
        temporal_kernel: int = 5,
        temporal_stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = d_out
        self.temporal_stride = temporal_stride
        self.input_proj = nn.Sequential(
            nn.Linear(5, d_hidden * 2), nn.GLU(), nn.LayerNorm(d_hidden), nn.SiLU()
        )
        self.temporal = nn.ModuleList([TemporalResidualBlock(d_hidden, temporal_kernel, dropout) for _ in range(temporal_blocks)])
        self.graph = nn.ModuleList([DynamicGraphBlock(d_hidden, dropout) for _ in range(graph_blocks)])
        self.out_proj = nn.Sequential(nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out))

    def _temporal_apply(self, h: torch.Tensor, block: nn.Module) -> torch.Tensor:
        B, T, N, D = h.shape
        x = h.permute(0, 2, 3, 1).reshape(B * N, D, T)
        x = block(x)
        return x.reshape(B, N, D, T).permute(0, 3, 1, 2)

    def forward(self, state: torch.Tensor, mass: torch.Tensor, return_token_map: bool = False):
        B, T, N, _ = state.shape
        m = mass[:, None, :, None].expand(B, T, N, 1)
        h = self.input_proj(torch.cat([state, m], dim=-1))
        depth = max(len(self.temporal), len(self.graph))
        for i in range(depth):
            if i < len(self.temporal):
                h = self._temporal_apply(h, self.temporal[i])
            if i < len(self.graph):
                h = self.graph[i](h, state, mass)
        h = self.out_proj(h)
        if self.temporal_stride > 1:
            h = h[:, :: self.temporal_stride]
        Ts = h.shape[1]
        # Flatten [time, body] while preserving a map for analysis.
        tokens = h.reshape(B, Ts * N, self.d_out)
        if not return_token_map:
            return tokens
        time_idx = torch.arange(0, T, self.temporal_stride, device=state.device)[:Ts]
        token_time = time_idx[:, None].expand(Ts, N).reshape(-1)
        token_body = torch.arange(N, device=state.device)[None, :].expand(Ts, N).reshape(-1)
        return tokens, {"time_index": token_time, "body_index": token_body}
