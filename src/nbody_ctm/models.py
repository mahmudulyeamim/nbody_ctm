from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .ctm import ContinuousThoughtMachine
from .graph_encoder import GraphTemporalEncoder


class SimpleTokenEncoder(nn.Module):
    """Minimal non-graph encoder for an ablation: per body-time token MLP only."""
    def __init__(self, d_out: int = 128):
        super().__init__()
        self.d_out = d_out
        self.net = nn.Sequential(nn.Linear(5, d_out * 2), nn.GLU(), nn.LayerNorm(d_out))

    def forward(self, state, mass, return_token_map=False):
        B, T, N, _ = state.shape
        m = mass[:, None, :, None].expand(B, T, N, 1)
        h = self.net(torch.cat([state, m], -1))
        tokens = h.reshape(B, T * N, self.d_out)
        if not return_token_map:
            return tokens
        time_idx = torch.arange(T, device=state.device)[:, None].expand(T, N).reshape(-1)
        body_idx = torch.arange(N, device=state.device)[None].expand(T, N).reshape(-1)
        return tokens, {"time_index": time_idx, "body_index": body_idx}


def make_encoder(cfg: Dict) -> nn.Module:
    kind = cfg.get("type", "graph_temporal")
    if kind == "simple_tokens":
        return SimpleTokenEncoder(d_out=int(cfg.get("d_out", 128)))
    if kind != "graph_temporal":
        raise ValueError(f"Unknown encoder type: {kind}")
    return GraphTemporalEncoder(
        d_hidden=int(cfg.get("d_hidden", 96)),
        d_out=int(cfg.get("d_out", 128)),
        temporal_blocks=int(cfg.get("temporal_blocks", 2)),
        graph_blocks=int(cfg.get("graph_blocks", 2)),
        temporal_kernel=int(cfg.get("temporal_kernel", 5)),
        temporal_stride=int(cfg.get("temporal_stride", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


class NBodyCTMClassifier(nn.Module):
    def __init__(self, encoder_cfg: Dict, ctm_cfg: Dict, n_classes: int = 3, available_time_regions=None):
        super().__init__()
        allowed = tuple(available_time_regions or ("early", "middle", "late"))
        if not allowed or len(set(allowed)) != len(allowed) or any(x not in {"early", "middle", "late"} for x in allowed):
            raise ValueError(f"Invalid available_time_regions: {allowed}")
        self.available_time_regions = allowed
        self.encoder = make_encoder(encoder_cfg)
        self.ctm = ContinuousThoughtMachine(
            iterations=int(ctm_cfg["iterations"]),
            d_model=int(ctm_cfg["d_model"]),
            d_input=int(ctm_cfg["d_input"]),
            heads=int(ctm_cfg["heads"]),
            n_synch_out=int(ctm_cfg["n_synch_out"]),
            n_synch_action=int(ctm_cfg["n_synch_action"]),
            synapse_depth=int(ctm_cfg["synapse_depth"]),
            memory_length=int(ctm_cfg["memory_length"]),
            memory_hidden_dims=int(ctm_cfg["memory_hidden_dims"]),
            out_dims=n_classes,
            dropout=float(ctm_cfg.get("dropout", 0.1)),
            dropout_nlm=float(ctm_cfg.get("dropout_nlm", 0.0)),
            deep_nlms=bool(ctm_cfg.get("deep_nlms", True)),
            n_random_pairing_self=int(ctm_cfg.get("n_random_pairing_self", 16)),
            pairing_seed=int(ctm_cfg.get("pairing_seed", 17)),
            coverage_rho=float(ctm_cfg.get("coverage_memory", {}).get("rho", 1.0)),
            coverage_beta=float(ctm_cfg.get("coverage_memory", {}).get("beta", 0.0)),
        )

    def forward(self, state, mass, *, track=False):
        attention_mask = None
        if len(self.available_time_regions) < 3:
            # Also hide raw unavailable times: temporal convolutions in the encoder
            # would otherwise leak their physical state into nearby allowed tokens.
            raw_thirds = torch.tensor_split(torch.arange(state.size(1), device=state.device), 3)
            names = ("early", "middle", "late")
            hidden_raw = torch.cat([raw_thirds[i] for i, name in enumerate(names) if name not in self.available_time_regions])
            state = state.clone()
            state[:, hidden_raw] = 0.0  # training-set mean after normalization; masses unchanged
            tokens, token_map = self.encoder(state, mass, return_token_map=True)
            times = torch.unique(token_map["time_index"], sorted=True)
            encoded_thirds = torch.tensor_split(times, 3)
            hidden_times = torch.cat([encoded_thirds[i] for i, name in enumerate(names) if name not in self.available_time_regions])
            blocked = torch.isin(token_map["time_index"], hidden_times)
            if not blocked.any() or blocked.all():
                raise AssertionError("Invalid encoded-time token block.")
            attention_mask = torch.zeros((1, tokens.size(1)), device=tokens.device, dtype=tokens.dtype)
            attention_mask[:, blocked] = float("-inf")
            if track:
                pred, cert, tracking = self.ctm(
                    tokens, track=True, attention_mask=attention_mask,
                    token_time_index=token_map["time_index"],
                )
                tracking["token_time_index"] = token_map["time_index"]
                tracking["token_body_index"] = token_map["body_index"]
                return pred, cert, tracking
            return self.ctm(
                tokens, track=False, attention_mask=attention_mask,
                token_time_index=token_map["time_index"],
            )
        if track:
            tokens, token_map = self.encoder(state, mass, return_token_map=True)
            pred, cert, tracking = self.ctm(tokens, track=True, token_time_index=token_map["time_index"])
            tracking["token_time_index"] = token_map["time_index"]
            tracking["token_body_index"] = token_map["body_index"]
            return pred, cert, tracking
        if self.ctm.coverage_beta > 0:
            tokens, token_map = self.encoder(state, mass, return_token_map=True)
            return self.ctm(tokens, track=False, token_time_index=token_map["time_index"])
        tokens = self.encoder(state, mass)
        return self.ctm(tokens, track=False)


class GraphPoolClassifier(nn.Module):
    """Same graph-temporal extractor as CTM, followed by simple invariant pooling."""
    def __init__(self, encoder_cfg: Dict, n_classes: int = 3):
        super().__init__()
        self.encoder = make_encoder(encoder_cfg)
        d = int(encoder_cfg.get("d_out", 128))
        self.head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, n_classes)
        )

    def forward(self, state, mass):
        x = self.encoder(state, mass)
        pooled = torch.cat([x.mean(1), x.max(1).values], -1)
        return self.head(pooled)


class GraphGRUClassifier(nn.Module):
    """Graph-temporal encoder plus a GRU across physical time after body pooling."""
    def __init__(self, encoder_cfg: Dict, n_classes: int = 3, gru_hidden: int = 160):
        super().__init__()
        self.encoder = make_encoder(encoder_cfg)
        d = int(encoder_cfg.get("d_out", 128))
        self.gru = nn.GRU(d, gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(gru_hidden), nn.Linear(gru_hidden, n_classes))

    def forward(self, state, mass):
        B, _, N, _ = state.shape
        tokens, token_map = self.encoder(state, mass, return_token_map=True)
        # Tokens are time-major then body-major, so reshape safely.
        Ts = tokens.size(1) // N
        x = tokens.reshape(B, Ts, N, -1).mean(2)
        y, _ = self.gru(x)
        return self.head(y[:, -1])


def build_model(cfg: Dict, model_name: str):
    enc = cfg["model"]["encoder"]
    if model_name == "ctm":
        return NBodyCTMClassifier(
            enc, cfg["model"]["ctm"], n_classes=3,
            available_time_regions=cfg.get("training", {}).get("available_time_regions"),
        )
    if model_name == "graph_pool":
        return GraphPoolClassifier(enc, n_classes=3)
    if model_name == "graph_gru":
        return GraphGRUClassifier(enc, n_classes=3, gru_hidden=int(cfg["model"].get("gru_hidden", 160)))
    if model_name == "linear_ctm":
        enc2 = dict(enc)
        enc2["type"] = "simple_tokens"
        return NBodyCTMClassifier(enc2, cfg["model"]["ctm"], n_classes=3)
    raise ValueError(f"Unknown model: {model_name}")
