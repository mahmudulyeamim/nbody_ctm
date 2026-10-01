from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SuperLinear(nn.Module):
    """N independent linear transforms, one private transform per CTM neuron.

    x: [B, D, in_dims]
    weights: [in_dims, out_dims, D]
    out: [B, D, out_dims]
    """
    def __init__(self, in_dims: int, out_dims: int, n_neurons: int, dropout: float = 0.0, do_norm: bool = False):
        super().__init__()
        lim = 1.0 / math.sqrt(in_dims + out_dims)
        self.weight = nn.Parameter(torch.empty(in_dims, out_dims, n_neurons).uniform_(-lim, lim))
        self.bias = nn.Parameter(torch.zeros(1, n_neurons, out_dims))
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(in_dims) if do_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.dropout(x))
        out = torch.einsum("BDM,MHD->BDH", x, self.weight) + self.bias
        return out / self.temperature.clamp_min(1e-3)


class NeuronLevelModels(nn.Module):
    """Private per-neuron temporal MLPs over pre-activation history."""
    def __init__(self, d_model: int, memory_length: int, hidden: int, dropout: float = 0.0, deep: bool = True):
        super().__init__()
        if deep:
            self.net = nn.Sequential(
                SuperLinear(memory_length, 2 * hidden, d_model, dropout=dropout),
                nn.GLU(dim=-1),
                SuperLinear(hidden, 2, d_model, dropout=dropout),
                nn.GLU(dim=-1),
                Squeeze(-1),
            )
        else:
            self.net = nn.Sequential(
                SuperLinear(memory_length, 2, d_model, dropout=dropout),
                nn.GLU(dim=-1),
                Squeeze(-1),
            )

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        return self.net(trace)


class SynapseUNet(nn.Module):
    """Small U-Net-shaped MLP used to mix information across CTM neurons."""
    def __init__(self, d_model: int, depth: int = 3, minimum_width: int = 24, dropout: float = 0.1):
        super().__init__()
        if depth < 2:
            raise ValueError("SynapseUNet depth must be >= 2")
        widths = [int(x) for x in torch.linspace(d_model, minimum_width, depth).tolist()]
        self.first = nn.Sequential(nn.LazyLinear(widths[0]), nn.LayerNorm(widths[0]), nn.SiLU())
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.skip_norm = nn.ModuleList()
        for i in range(depth - 1):
            self.down.append(nn.Sequential(
                nn.Dropout(dropout), nn.Linear(widths[i], widths[i + 1]), nn.LayerNorm(widths[i + 1]), nn.SiLU()
            ))
            self.up.append(nn.Sequential(
                nn.Dropout(dropout), nn.Linear(widths[i + 1], widths[i]), nn.LayerNorm(widths[i]), nn.SiLU()
            ))
            self.skip_norm.append(nn.LayerNorm(widths[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_states = [self.first(x)]
        for layer in self.down:
            down_states.append(layer(down_states[-1]))
        y = down_states[-1]
        for reverse_idx in reversed(range(len(self.up))):
            y = self.up[reverse_idx](y)
            y = self.skip_norm[reverse_idx](y + down_states[reverse_idx])
        return y


class ContinuousThoughtMachine(nn.Module):
    """CTM core adapted for tokenized non-image inputs.

    This implementation follows the public CTM architecture's ordering:
      action synchronization -> attention -> synapse model -> pre-activation trace
      -> private neuron-level temporal models -> output synchronization -> prediction.

    Synchronization uses random neuron pairings with a learnable exponential decay.
    """
    def __init__(
        self,
        *,
        iterations: int,
        d_model: int,
        d_input: int,
        heads: int,
        n_synch_out: int,
        n_synch_action: int,
        synapse_depth: int,
        memory_length: int,
        memory_hidden_dims: int,
        out_dims: int,
        dropout: float = 0.1,
        dropout_nlm: float = 0.0,
        deep_nlms: bool = True,
        n_random_pairing_self: int = 16,
        pairing_seed: int = 17,
        coverage_rho: float = 1.0,
        coverage_beta: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= coverage_rho <= 1.0 or coverage_beta < 0.0:
            raise ValueError("coverage_rho must be in [0, 1] and coverage_beta must be nonnegative")
        self.iterations = iterations
        self.coverage_rho = coverage_rho
        self.coverage_beta = coverage_beta
        self.heads = heads
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.out_dims = out_dims

        self.kv_proj = nn.Sequential(nn.LazyLinear(d_input), nn.LayerNorm(d_input))
        self.q_proj = nn.Linear(n_synch_action, d_input)
        self.attention = nn.MultiheadAttention(d_input, heads, dropout=dropout, batch_first=True)

        if synapse_depth <= 1:
            self.synapses = nn.Sequential(
                nn.Dropout(dropout), nn.LazyLinear(d_model * 2), nn.GLU(), nn.LayerNorm(d_model)
            )
        else:
            self.synapses = SynapseUNet(d_model, synapse_depth, minimum_width=max(16, d_model // 8), dropout=dropout)
        self.trace_processor = NeuronLevelModels(
            d_model, memory_length, memory_hidden_dims, dropout=dropout_nlm, deep=deep_nlms
        )

        bound = math.sqrt(1 / d_model)
        self.start_activated_state = nn.Parameter(torch.empty(d_model).uniform_(-bound, bound))
        tb = math.sqrt(1 / (d_model + memory_length))
        self.start_trace = nn.Parameter(torch.empty(d_model, memory_length).uniform_(-tb, tb))

        self._register_pairings("action", n_synch_action, n_random_pairing_self, pairing_seed)
        self._register_pairings("out", n_synch_out, n_random_pairing_self, pairing_seed + 1)
        self.decay_params_action = nn.Parameter(torch.zeros(n_synch_action))
        self.decay_params_out = nn.Parameter(torch.zeros(n_synch_out))
        self.output_projector = nn.Linear(n_synch_out, out_dims)

    def _register_pairings(self, name: str, n_pairs: int, n_self: int, seed: int):
        if n_self > n_pairs:
            raise ValueError("n_random_pairing_self cannot exceed n_pairs")
        g = torch.Generator(device="cpu").manual_seed(seed)
        left = torch.randint(0, self.d_model, (n_pairs,), generator=g)
        right = torch.randint(0, self.d_model, (n_pairs,), generator=g)
        if n_self:
            right[:n_self] = left[:n_self]
        self.register_buffer(f"{name}_left", left)
        self.register_buffer(f"{name}_right", right)

    @staticmethod
    def normalized_certainty(logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1)
        normalized_entropy = entropy / math.log(logits.size(-1))
        return 1.0 - normalized_entropy

    def _sync(self, activated: torch.Tensor, alpha, beta, *, kind: str):
        left_idx = getattr(self, f"{kind}_left")
        right_idx = getattr(self, f"{kind}_right")
        decay_param = getattr(self, f"decay_params_{kind}")
        r = torch.exp(-decay_param.clamp(0, 15))[None, :]
        pair_product = activated[:, left_idx] * activated[:, right_idx]
        if alpha is None:
            alpha = pair_product
            beta = torch.ones_like(pair_product)
        else:
            alpha = r * alpha + pair_product
            beta = r * beta + 1.0
        sync = alpha / torch.sqrt(beta.clamp_min(1e-8))
        return sync, alpha, beta

    def forward(
        self, tokens: torch.Tensor, *, track: bool = False,
        attention_mask: torch.Tensor | None = None,
        token_time_index: torch.Tensor | None = None,
    ):
        B = tokens.size(0)
        kv = self.kv_proj(tokens)
        coverage = None
        if self.coverage_beta > 0:
            if token_time_index is None or token_time_index.numel() != tokens.size(1):
                raise ValueError("Coverage memory requires one physical-time index per encoded token")
            _, time_inverse = torch.unique(token_time_index, sorted=True, return_inverse=True)
            coverage = kv.new_zeros((B, int(time_inverse.max().item()) + 1), dtype=torch.float32)
        state_trace = self.start_trace[None].expand(B, -1, -1)
        activated = self.start_activated_state[None].expand(B, -1)

        predictions = []
        certainties = []
        alpha_action = beta_action = None
        alpha_out = beta_out = None
        # Prime output recurrence from the learned starting state, like the public implementation.
        _, alpha_out, beta_out = self._sync(activated, alpha_out, beta_out, kind="out")

        tracking = {
            "pre_activations": [], "post_activations": [],
            "sync_out": [], "sync_action": [], "attention": [],
        }

        for _ in range(self.iterations):
            sync_action, alpha_action, beta_action = self._sync(activated, alpha_action, beta_action, kind="action")
            q = self.q_proj(sync_action).unsqueeze(1)
            tick_mask = attention_mask
            if coverage is not None:
                # [B,L] penalty, shared by every body token at the same physical time.
                penalty = -self.coverage_beta * coverage[:, time_inverse]
                tick_mask = penalty[:, None, :].expand(B, self.heads, -1).reshape(B * self.heads, 1, -1)
                if attention_mask is not None:
                    tick_mask = tick_mask + attention_mask
                tick_mask = tick_mask.to(dtype=q.dtype)
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True, attn_mask=tick_mask
            )
            if coverage is not None:
                # Sum body-token mass after averaging heads; retain the graph so
                # later CTM ticks can backpropagate through earlier attention.
                # MHA returns post-dropout weights during training; renormalize
                # them to a temporal probability distribution before recurrence.
                token_mass = attn_weights.squeeze(2).float().mean(1)
                token_mass = token_mass / token_mass.sum(1, keepdim=True).clamp_min(1e-8)
                temporal_mass = coverage.new_zeros(coverage.shape)
                temporal_mass.scatter_add_(1, time_inverse[None, :].expand(B, -1), token_mass)
                coverage = self.coverage_rho * coverage + temporal_mass
            pre_synapse = torch.cat([attn_out.squeeze(1), activated], dim=-1)
            pre_activation = self.synapses(pre_synapse)
            state_trace = torch.cat([state_trace[:, :, 1:], pre_activation.unsqueeze(-1)], dim=-1)
            activated = self.trace_processor(state_trace)
            sync_out, alpha_out, beta_out = self._sync(activated, alpha_out, beta_out, kind="out")
            logits = self.output_projector(sync_out)
            predictions.append(logits)
            certainties.append(self.normalized_certainty(logits))
            if track:
                tracking["pre_activations"].append(pre_activation)
                tracking["post_activations"].append(activated)
                tracking["sync_out"].append(sync_out)
                tracking["sync_action"].append(sync_action)
                # [B,H,1,L] -> [B,H,L]
                tracking["attention"].append(attn_weights.squeeze(2))

        # Public CTM convention is [B,C,T] for predictions.
        pred = torch.stack(predictions, dim=-1)
        cert = torch.stack(certainties, dim=-1)  # [B,T]
        if track:
            tracking = {k: torch.stack(v, dim=1) for k, v in tracking.items()}
            return pred, cert, tracking
        return pred, cert, sync_out
