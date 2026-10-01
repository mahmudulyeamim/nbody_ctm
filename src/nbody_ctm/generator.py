from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .physics import potential_energy, simulate_and_label


LABEL_NAMES = {0: "bound", 1: "collision", 2: "escape"}
PROPOSAL_NAMES = {0: "virial", 1: "cold_close", 2: "hot_kick", 3: "binary_intruder"}


@dataclass
class GenerationStats:
    simulated: int = 0
    rejected_observed_event: int = 0
    rejected_ambiguous: int = 0
    skipped_quota_full: int = 0
    retained: int = 0
    retained_counterfactual: int = 0
    counterfactual_bases_tested: int = 0
    counterfactual_bases_successful: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


def _rand_uniform(shape, lo: float, hi: float, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen)


def _center_system(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tm = mass.sum(-1, keepdim=True)
    com_pos = (pos * mass[..., None]).sum(1) / tm
    com_vel = (vel * mass[..., None]).sum(1) / tm
    return pos - com_pos[:, None, :], vel - com_vel[:, None, :]


def sample_initial_conditions(batch: int, cfg: Dict, device: torch.device, gen: torch.Generator):
    """Sample a mixture of interacting gravitational systems.

    Proposal type is *not* supplied to the classifier. It exists only to make dataset
    generation cover collapse, slingshot/ejection, and long-lived bound behavior.
    Labels always come from simulation, never from proposal type.
    """
    n = int(cfg["n_bodies"])
    dtype = torch.float32
    # Mild mass diversity; binary-intruder proposal will add one heavy body.
    mass = torch.exp(_rand_uniform((batch, n), np.log(0.55), np.log(1.65), device, gen)).to(dtype)

    pos = torch.randn((batch, n, 2), device=device, generator=gen, dtype=dtype)
    pos = pos / pos.square().sum(-1).mean(-1).sqrt().clamp_min(1e-6)[:, None, None]
    vel = torch.randn((batch, n, 2), device=device, generator=gen, dtype=dtype)
    pos, vel = _center_system(pos, vel, mass)

    probs = torch.tensor(cfg.get("proposal_probs", [0.35, 0.25, 0.25, 0.15]), device=device, dtype=dtype)
    probs = probs / probs.sum()
    proposal = torch.multinomial(probs, batch, replacement=True, generator=gen)

    # Proposal-dependent virial ratios Q=K/|U|.
    q = torch.empty(batch, device=device, dtype=dtype)
    q[proposal == 0] = _rand_uniform(((proposal == 0).sum().item(),), 0.34, 0.72, device, gen)
    q[proposal == 1] = _rand_uniform(((proposal == 1).sum().item(),), 0.16, 0.46, device, gen)
    q[proposal == 2] = _rand_uniform(((proposal == 2).sum().item(),), 0.62, 1.02, device, gen)
    q[proposal == 3] = _rand_uniform(((proposal == 3).sum().item(),), 0.30, 0.72, device, gen)

    # Create a close but non-colliding pair in collapse/binary proposals.
    selected = (proposal == 1) | (proposal == 3)
    if selected.any():
        idx = selected.nonzero(as_tuple=False).squeeze(1)
        theta = _rand_uniform((idx.numel(),), 0.0, 2 * np.pi, device, gen)
        direction = torch.stack([theta.cos(), theta.sin()], -1)
        sep = _rand_uniform((idx.numel(),), cfg["collision_radius"] * 1.7, 0.30, device, gen)
        midpoint = 0.25 * torch.randn((idx.numel(), 2), device=device, generator=gen)
        pos[idx, 0] = midpoint - 0.5 * sep[:, None] * direction
        pos[idx, 1] = midpoint + 0.5 * sep[:, None] * direction
        if (proposal == 3).any():
            bidx = (proposal == 3).nonzero(as_tuple=False).squeeze(1)
            mass[bidx, 0] *= 1.5
            mass[bidx, 1] *= 1.35

    # Scale velocities to desired virial ratio.
    pos, vel = _center_system(pos, vel, mass)
    U = potential_energy(pos, mass, G=cfg["G"], softening=cfg["softening"]).abs().clamp_min(1e-6)
    K = (0.5 * mass * vel.square().sum(-1)).sum(-1).clamp_min(1e-6)
    scale = torch.sqrt(q * U / K)
    vel = vel * scale[:, None, None]

    # Hot-kick proposal: a random body receives an additional velocity impulse.
    hot = proposal == 2
    if hot.any():
        idx = hot.nonzero(as_tuple=False).squeeze(1)
        body = torch.randint(0, n, (idx.numel(),), device=device, generator=gen)
        direction = torch.randn((idx.numel(), 2), device=device, generator=gen)
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        rms = vel[idx].square().mean(dim=(1, 2)).sqrt()
        kick = _rand_uniform((idx.numel(),), 0.25, 0.95, device, gen) * rms
        vel[idx, body] += kick[:, None] * direction

    # Binary intruder: send body 2 roughly toward the close pair.
    bi = proposal == 3
    if bi.any() and n >= 3:
        idx = bi.nonzero(as_tuple=False).squeeze(1)
        pair_mid = 0.5 * (pos[idx, 0] + pos[idx, 1])
        pos[idx, 2] = 1.4 * pos[idx, 2] / torch.linalg.norm(pos[idx, 2], dim=-1, keepdim=True).clamp_min(1e-6)
        toward = pair_mid - pos[idx, 2]
        toward = toward / torch.linalg.norm(toward, dim=-1, keepdim=True).clamp_min(1e-6)
        speed = vel[idx].square().mean(dim=(1, 2)).sqrt() * _rand_uniform((idx.numel(),), 0.7, 1.4, device, gen)
        vel[idx, 2] = toward * speed[:, None]

    pos, vel = _center_system(pos, vel, mass)

    # Randomly permute body IDs per system to discourage fixed-index shortcuts.
    for b in range(batch):
        p = torch.randperm(n, generator=gen, device=device)
        pos[b] = pos[b, p]
        vel[b] = vel[b, p]
        mass[b] = mass[b, p]

    return pos, vel, mass, proposal, q


def _sim_kwargs(cfg: Dict) -> Dict:
    keys = [
        "obs_steps", "hidden_steps", "dt", "G", "softening", "collision_radius",
        "escape_radius", "escape_energy_margin", "escape_radial_velocity",
        "escape_persistence", "guard_steps", "bound_radius_fraction",
        "bound_collision_clearance", "bound_energy_ceiling",
    ]
    return {k: cfg[k] for k in keys}


def _empty_buffers():
    return {c: defaultdict(list) for c in range(3)}


def _append_selected(buffers, cls: int, traj, mass, summary, diagnostics, proposal, q, indices, pair_ids, is_cf):
    if len(indices) == 0:
        return
    idx = torch.as_tensor(indices, device=traj.device, dtype=torch.long)
    b = buffers[cls]
    b["state"].append(traj[idx].cpu())
    b["mass"].append(mass[idx].cpu())
    b["label"].append(summary.label[idx].cpu())
    b["event_step"].append(summary.event_step[idx].cpu())
    b["event_body_a"].append(summary.event_body_a[idx].cpu())
    b["event_body_b"].append(summary.event_body_b[idx].cpu())
    b["outcome_margin"].append(summary.outcome_margin[idx].cpu())
    b["min_hidden_separation"].append(summary.min_hidden_separation[idx].cpu())
    b["max_hidden_radius"].append(summary.max_hidden_radius[idx].cpu())
    b["max_hidden_specific_energy"].append(summary.max_hidden_specific_energy[idx].cpu())
    b["initial_energy"].append(diagnostics["initial_total_energy"][idx].cpu())
    b["final_energy"].append(diagnostics["final_total_energy"][idx].cpu())
    b["proposal"].append(proposal[idx].cpu())
    b["virial_ratio_target"].append(q[idx].cpu())
    b["pair_id"].append(torch.as_tensor(pair_ids, dtype=torch.long))
    b["is_counterfactual"].append(torch.full((len(indices),), bool(is_cf), dtype=torch.bool))


def _count(buffers, cls: int) -> int:
    return sum(x.numel() for x in buffers[cls].get("label", []))


def _quotas(n_samples: int) -> List[int]:
    base = n_samples // 3
    q = [base, base, base]
    for i in range(n_samples - 3 * base):
        q[i] += 1
    return q


def _finalize(buffers, n_samples: int, seed: int) -> Dict[str, np.ndarray]:
    parts = defaultdict(list)
    for cls in range(3):
        for key, chunks in buffers[cls].items():
            parts[key].append(torch.cat(chunks, dim=0))
    out = {k: torch.cat(v, dim=0) for k, v in parts.items()}
    if out["label"].numel() != n_samples:
        raise RuntimeError(f"Expected {n_samples} retained samples, got {out['label'].numel()}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n_samples, generator=g)
    out = {k: v[perm].numpy() for k, v in out.items()}
    return out


def _update_rejection_stats(stats: GenerationStats, summary) -> None:
    stats.rejected_observed_event += int(summary.observed_event.sum().item())
    # Ambiguous means not valid and not already counted as observed-window event.
    stats.rejected_ambiguous += int((~summary.valid & ~summary.observed_event).sum().item())


def generate_split(n_samples: int, cfg: Dict, *, seed: int, device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Generate one split with exact class quotas and optional counterfactual twins."""
    quotas = _quotas(n_samples)
    buffers = _empty_buffers()
    stats = GenerationStats()
    gen = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(cfg.get("simulation_batch_size", 128))
    max_batches = int(cfg.get("max_generation_batches", 5000))
    target_cf_samples = int(n_samples * float(cfg.get("counterfactual_fraction", 0.0)))
    target_cf_samples -= target_cf_samples % 2
    cf_kept = 0
    next_pair_id = 0

    # Phase 1: seek counterfactual twins with tiny opposite perturbations.
    cf_batches = 0
    while cf_kept < target_cf_samples and cf_batches < max_batches:
        base_b = max(8, batch_size // 2)
        pos, vel, mass, proposal, q = sample_initial_conditions(base_b, cfg, device, gen)
        stats.counterfactual_bases_tested += base_b
        body = torch.randint(0, cfg["n_bodies"], (base_b,), device=device, generator=gen)
        direction = torch.randn((base_b, 2), device=device, generator=gen)
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        vrms = vel.square().mean(dim=(1, 2)).sqrt()
        eps = float(cfg.get("counterfactual_velocity_epsilon", 0.04))
        delta = eps * vrms[:, None] * direction
        vplus, vminus = vel.clone(), vel.clone()
        vplus[torch.arange(base_b, device=device), body] += delta
        vminus[torch.arange(base_b, device=device), body] -= delta
        pos2 = torch.cat([pos, pos], 0)
        vel2 = torch.cat([vplus, vminus], 0)
        mass2 = torch.cat([mass, mass], 0)
        proposal2 = torch.cat([proposal, proposal], 0)
        q2 = torch.cat([q, q], 0)
        traj, summary, diag = simulate_and_label(pos2, vel2, mass2, **_sim_kwargs(cfg))
        stats.simulated += 2 * base_b
        _update_rejection_stats(stats, summary)
        for b in range(base_b):
            i, j = b, b + base_b
            if not (summary.valid[i] and summary.valid[j]):
                continue
            yi, yj = int(summary.label[i]), int(summary.label[j])
            if yi == yj:
                continue
            if _count(buffers, yi) >= quotas[yi] or _count(buffers, yj) >= quotas[yj]:
                stats.skipped_quota_full += 2
                continue
            _append_selected(buffers, yi, traj, mass2, summary, diag, proposal2, q2, [i], [next_pair_id], True)
            _append_selected(buffers, yj, traj, mass2, summary, diag, proposal2, q2, [j], [next_pair_id], True)
            next_pair_id += 1
            cf_kept += 2
            stats.retained += 2
            stats.retained_counterfactual += 2
            stats.counterfactual_bases_successful += 1
            if cf_kept >= target_cf_samples:
                break
        cf_batches += 1

    # Phase 2: ordinary candidates fill exact class quotas.
    batches = 0
    while any(_count(buffers, c) < quotas[c] for c in range(3)):
        if batches >= max_batches:
            counts = [_count(buffers, c) for c in range(3)]
            raise RuntimeError(f"Generation exhausted max batches. counts={counts}, quotas={quotas}. Tune simulator proposal ranges or max_generation_batches.")
        pos, vel, mass, proposal, q = sample_initial_conditions(batch_size, cfg, device, gen)
        traj, summary, diag = simulate_and_label(pos, vel, mass, **_sim_kwargs(cfg))
        stats.simulated += batch_size
        _update_rejection_stats(stats, summary)
        for cls in range(3):
            remaining = quotas[cls] - _count(buffers, cls)
            idx_all = ((summary.label == cls) & summary.valid).nonzero(as_tuple=False).squeeze(1)
            if remaining <= 0:
                stats.skipped_quota_full += int(idx_all.numel())
                continue
            take = idx_all[:remaining]
            stats.skipped_quota_full += max(0, int(idx_all.numel()) - int(take.numel()))
            if take.numel():
                _append_selected(
                    buffers, cls, traj, mass, summary, diag, proposal, q,
                    take.tolist(), [-1] * int(take.numel()), False,
                )
                stats.retained += int(take.numel())
        batches += 1

    data = _finalize(buffers, n_samples, seed)
    report = stats.as_dict()
    report["class_counts"] = {LABEL_NAMES[c]: int((data["label"] == c).sum()) for c in range(3)}
    report["proposal_counts_retained"] = {PROPOSAL_NAMES[p]: int((data["proposal"] == p).sum()) for p in range(4)}
    report["counterfactual_fraction_realized"] = float(data["is_counterfactual"].mean())
    return data, report


def generate_dataset(cfg: Dict, *, device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict]:
    base_seed = int(cfg.get("seed", 42))
    sizes = {"train": int(cfg["n_train"]), "val": int(cfg["n_val"]), "test": int(cfg["n_test"])}
    all_data = {}
    report = {"splits": {}}
    for offset, (split, n) in enumerate(sizes.items()):
        data, stats = generate_split(n, cfg, seed=base_seed + 1009 * offset, device=device)
        for key, value in data.items():
            all_data[f"{split}_{key}"] = value
        report["splits"][split] = stats
    report["label_names"] = LABEL_NAMES
    report["proposal_names"] = PROPOSAL_NAMES
    return all_data, report
