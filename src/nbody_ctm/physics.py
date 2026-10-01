from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class EventSummary:
    label: torch.Tensor                 # [B], -1 ambiguous, 0 bound, 1 collision, 2 escape
    valid: torch.Tensor                 # [B]
    observed_event: torch.Tensor        # [B]
    event_step: torch.Tensor            # [B], -1 if none
    event_body_a: torch.Tensor          # [B]
    event_body_b: torch.Tensor          # [B], -1 for escape/bound
    min_hidden_separation: torch.Tensor # [B]
    max_hidden_radius: torch.Tensor     # [B]
    max_hidden_specific_energy: torch.Tensor # [B]
    outcome_margin: torch.Tensor        # [B], larger = farther from nearest decision boundary


def pairwise_displacements(pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return r_j-r_i and softened-free pair distances.

    pos: [B, N, 2]
    returns disp [B,N,N,2], dist [B,N,N]
    """
    disp = pos[:, None, :, :] - pos[:, :, None, :]
    dist = torch.linalg.norm(disp, dim=-1)
    return disp, dist


def accelerations(pos: torch.Tensor, mass: torch.Tensor, G: float = 1.0, softening: float = 0.03) -> torch.Tensor:
    """Newtonian pairwise acceleration with Plummer softening.

    pos: [B,N,2], mass: [B,N]
    """
    disp, dist = pairwise_displacements(pos)
    dist2 = (dist * dist + softening * softening)
    eye_bool = torch.eye(pos.size(1), dtype=torch.bool, device=pos.device)[None]
    # Mask the diagonal *before* inversion so softening=0 never creates inf*0 -> NaN.
    dist2 = dist2.masked_fill(eye_bool, float("inf"))
    inv_r3 = dist2.pow(-1.5)
    # source body j contributes m_j (r_j-r_i)/r^3
    return G * (disp * inv_r3[..., None] * mass[:, None, :, None]).sum(dim=2)


def total_energy(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor, G: float = 1.0, softening: float = 0.03) -> torch.Tensor:
    kinetic = 0.5 * (mass * vel.square().sum(-1)).sum(-1)
    _, dist = pairwise_displacements(pos)
    n = pos.size(1)
    i, j = torch.triu_indices(n, n, offset=1, device=pos.device)
    rij = torch.sqrt(dist[:, i, j].square() + softening * softening)
    potential = -G * (mass[:, i] * mass[:, j] / rij).sum(-1)
    return kinetic + potential


def potential_energy(pos: torch.Tensor, mass: torch.Tensor, G: float = 1.0, softening: float = 0.03) -> torch.Tensor:
    _, dist = pairwise_displacements(pos)
    n = pos.size(1)
    i, j = torch.triu_indices(n, n, offset=1, device=pos.device)
    rij = torch.sqrt(dist[:, i, j].square() + softening * softening)
    return -G * (mass[:, i] * mass[:, j] / rij).sum(-1)


def body_specific_energy(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor, G: float = 1.0, softening: float = 0.03) -> torch.Tensor:
    """Approximate body-specific orbital energy relative to all other bodies."""
    total_mass = mass.sum(-1, keepdim=True)
    com_pos = (pos * mass[..., None]).sum(1) / total_mass
    com_vel = (vel * mass[..., None]).sum(1) / total_mass
    rel_vel = vel - com_vel[:, None, :]
    kinetic_specific = 0.5 * rel_vel.square().sum(-1)
    _, dist = pairwise_displacements(pos)
    softened = torch.sqrt(dist.square() + softening * softening)
    eye_bool = torch.eye(pos.size(1), dtype=torch.bool, device=pos.device)[None]
    softened = softened.masked_fill(eye_bool, float("inf"))
    inv = 1.0 / softened
    phi = -G * (inv * mass[:, None, :]).sum(-1)
    return kinetic_specific + phi


def com_radius_and_radial_velocity(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tm = mass.sum(-1, keepdim=True)
    com_pos = (pos * mass[..., None]).sum(1) / tm
    com_vel = (vel * mass[..., None]).sum(1) / tm
    r = pos - com_pos[:, None, :]
    v = vel - com_vel[:, None, :]
    radius = torch.linalg.norm(r, dim=-1)
    radial_v = (r * v).sum(-1) / radius.clamp_min(1e-8)
    return radius, radial_v


def velocity_verlet_step(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor, dt: float, G: float, softening: float) -> Tuple[torch.Tensor, torch.Tensor]:
    a0 = accelerations(pos, mass, G=G, softening=softening)
    v_half = vel + 0.5 * dt * a0
    pos_new = pos + dt * v_half
    a1 = accelerations(pos_new, mass, G=G, softening=softening)
    vel_new = v_half + 0.5 * dt * a1
    return pos_new, vel_new


def simulate_and_label(
    pos0: torch.Tensor,
    vel0: torch.Tensor,
    mass: torch.Tensor,
    *,
    obs_steps: int,
    hidden_steps: int,
    dt: float,
    G: float = 1.0,
    softening: float = 0.03,
    collision_radius: float = 0.08,
    escape_radius: float = 4.0,
    escape_energy_margin: float = 0.02,
    escape_radial_velocity: float = 0.02,
    escape_persistence: int = 3,
    guard_steps: int = 4,
    bound_radius_fraction: float = 0.72,
    bound_collision_clearance: float = 1.5,
    bound_energy_ceiling: float = 0.01,
    record_stride: int = 1,
) -> Tuple[torch.Tensor, EventSummary, Dict[str, torch.Tensor]]:
    """Simulate a batch and classify the *post-observation* outcome.

    Labels:
      0 = robustly bound through horizon
      1 = collision after observation+guard
      2 = escape/ejection after observation+guard
     -1 = ambiguous/rejected

    The input trajectory contains exactly obs_steps snapshots, including the initial state.
    Any collision/escape detected during this observed window causes rejection.
    """
    device = pos0.device
    B, N, _ = pos0.shape
    total_steps = obs_steps + hidden_steps
    pos = pos0.clone()
    vel = vel0.clone()

    obs = []
    observed_event = torch.zeros(B, dtype=torch.bool, device=device)
    guard_event = torch.zeros(B, dtype=torch.bool, device=device)
    first_event_step = torch.full((B,), -1, dtype=torch.long, device=device)
    first_event_type = torch.full((B,), -1, dtype=torch.long, device=device)
    event_a = torch.full((B,), -1, dtype=torch.long, device=device)
    event_b = torch.full((B,), -1, dtype=torch.long, device=device)
    escape_streak = torch.zeros((B, N), dtype=torch.long, device=device)

    min_hidden_sep = torch.full((B,), float("inf"), dtype=pos.dtype, device=device)
    max_hidden_radius = torch.zeros(B, dtype=pos.dtype, device=device)
    max_hidden_energy = torch.full((B,), -float("inf"), dtype=pos.dtype, device=device)

    for step in range(total_steps):
        if step < obs_steps and step % record_stride == 0:
            obs.append(torch.cat([pos, vel], dim=-1))

        _, dist = pairwise_displacements(pos)
        eye = torch.eye(N, dtype=torch.bool, device=device)[None]
        masked_dist = dist.masked_fill(eye, float("inf"))
        min_sep, min_flat = masked_dist.view(B, -1).min(-1)
        pair_i = min_flat // N
        pair_j = min_flat % N

        radius, radial_v = com_radius_and_radial_velocity(pos, vel, mass)
        spec_e = body_specific_energy(pos, vel, mass, G=G, softening=softening)
        escape_now_body = (radius > escape_radius) & (spec_e > escape_energy_margin) & (radial_v > escape_radial_velocity)
        escape_streak = torch.where(escape_now_body, escape_streak + 1, torch.zeros_like(escape_streak))
        escape_now, escape_body = (escape_streak >= escape_persistence).max(-1)
        collision_now = min_sep <= collision_radius

        if step < obs_steps:
            observed_event |= collision_now | escape_now
        else:
            if step < obs_steps + guard_steps:
                guard_event |= collision_now | escape_now
            min_hidden_sep = torch.minimum(min_hidden_sep, min_sep)
            max_hidden_radius = torch.maximum(max_hidden_radius, radius.max(-1).values)
            max_hidden_energy = torch.maximum(max_hidden_energy, spec_e.max(-1).values)
            allowed = step >= obs_steps + guard_steps
            no_event_yet = first_event_step < 0
            collision_take = no_event_yet & collision_now & allowed
            # if both happen on same step, mark ambiguous by event type 3
            simultaneous = collision_take & escape_now
            escape_take = no_event_yet & escape_now & allowed & ~collision_now
            first_event_step[collision_take | escape_take | simultaneous] = step
            first_event_type[collision_take] = 1
            first_event_type[escape_take] = 2
            first_event_type[simultaneous] = 3
            event_a[collision_take] = pair_i[collision_take]
            event_b[collision_take] = pair_j[collision_take]
            event_a[escape_take] = escape_body[escape_take]

        if step < total_steps - 1:
            pos, vel = velocity_verlet_step(pos, vel, mass, dt, G, softening)

    trajectory = torch.stack(obs, dim=1)
    if trajectory.size(1) != obs_steps:
        raise RuntimeError("record_stride must currently be 1 for dataset generation.")

    label = torch.full((B,), -1, dtype=torch.long, device=device)
    collision = first_event_type == 1
    escape = first_event_type == 2
    no_event = first_event_type < 0

    # Robust bound: no event, comfortably inside both event boundaries at horizon.
    bound = (
        no_event
        & (max_hidden_radius < escape_radius * bound_radius_fraction)
        & (min_hidden_sep > collision_radius * bound_collision_clearance)
        & (max_hidden_energy < bound_energy_ceiling)
    )
    label[bound] = 0
    label[collision] = 1
    label[escape] = 2

    # Robust margins, scaled so larger means farther from the decision boundary.
    bound_margin = torch.minimum(
        (min_hidden_sep - collision_radius) / max(collision_radius, 1e-8),
        (escape_radius - max_hidden_radius) / max(escape_radius, 1e-8),
    )
    collision_margin = (collision_radius - min_hidden_sep) / max(collision_radius, 1e-8)
    escape_margin = torch.minimum(
        (max_hidden_radius - escape_radius) / max(escape_radius, 1e-8),
        (max_hidden_energy - escape_energy_margin) / max(abs(escape_energy_margin) + 0.05, 1e-8),
    )
    margin = torch.zeros(B, dtype=pos.dtype, device=device)
    margin[bound] = bound_margin[bound]
    margin[collision] = collision_margin[collision]
    margin[escape] = escape_margin[escape]

    valid = (~observed_event) & (~guard_event) & (label >= 0) & (first_event_type != 3)
    label = torch.where(valid, label, torch.full_like(label, -1))

    summary = EventSummary(
        label=label,
        valid=valid,
        observed_event=observed_event,
        event_step=first_event_step,
        event_body_a=event_a,
        event_body_b=event_b,
        min_hidden_separation=min_hidden_sep,
        max_hidden_radius=max_hidden_radius,
        max_hidden_specific_energy=max_hidden_energy,
        outcome_margin=margin,
    )
    diagnostics = {
        "initial_total_energy": total_energy(pos0, vel0, mass, G=G, softening=softening),
        "final_total_energy": total_energy(pos, vel, mass, G=G, softening=softening),
    }
    return trajectory, summary, diagnostics
